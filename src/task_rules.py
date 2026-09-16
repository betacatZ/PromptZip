"""任务类型识别规则：解析 user prompt 判断 qa / summarize / tool_call / other。

设计定位（见 PLAN_task_rules.md）：
- 面向真实用户问答系统：一次请求 = system_prompt(系统模板) + user_prompt(用户输入) + long_text(长文档)。
- long_text 超阈值才需要压缩 -> 才需要任务判断（classify_pipeline 负责触发）。
- 任务信号在指令里不在长文档里 -> 只分析 user_prompt；为空或无信号命中时 fallback 到 system_prompt
  （真实流量的关键词查询 "iPhone 15 价格" 非空但无信号，指令其实在 system 的助手模板里）。
- 纯规则（正则 + 权重），零第三方依赖，微秒级延迟。

用法：
    from src.task_rules import classify_pipeline, classify_task
    classify_pipeline(user_prompt="帮我查明天北京天气", long_text=doc,
                      system_prompt="You have access to the following tools: ...")
    classify_task("What is the refund policy?")

命令行单条调试：
    python src/task_rules.py "你的prompt"
"""

import re

__all__ = [
    "classify_pipeline",
    "classify_task",
    "classify_task_detailed",
    "split_bfcl_user_prompt",
]

# ----------------------------------------------------------------------------
# 规则表：(compiled_pattern, 权重, 规则名)
# 面向真实措辞优先；基准模板短语（Question:/Type: 等）可用但不作为唯一依赖。
# 权重在 dev split 上调出，全表总分约 3 分即视为"有信号"（SIGNAL_THRESHOLD）。
# ----------------------------------------------------------------------------

# 工具/函数调用：结构锚点，最高优先级，命中即短路（不与 qa 比权重——BFCL 的 CURRENT
# turn 里 63/204 含 ?，礼貌问句不能盖过结构证据）。
# CODE_SENSITIVE 标记的规则是中文祈使短语——代码注释里的"调用该函数"是叙述不是指令，
# 高代码密度文本中会被压制；结构性锚点（schema/标签/系统模板）不受压制。
TOOL_CALL_RULES = [
    # -- system 侧：真实工具系统的模板措辞（fallback 通路的主判据） --
    (re.compile(r"you have access to (?:the following|these) tools?", re.I), 3.0, "sys_tools_available"),
    (re.compile(r"tools? (?:are|is) (?:now )?available", re.I), 2.0, "sys_tools_available2"),
    (re.compile(r"available (?:tool|function)s?\s*:", re.I), 2.0, "sys_available_tools_colon"),
    (re.compile(r"set of possible functions", re.I), 2.0, "sys_possible_functions"),
    # 工具 JSON schema 结构特征（OpenAI 风格工具列表渲染进文本后的形态）
    (re.compile(r'"type"\s*:\s*"function"', re.I), 3.0, "schema_type_function"),
    (re.compile(r'"parameters"\s*:\s*\{[^{}]*"properties"', re.I | re.S), 3.0, "schema_parameters"),
    (re.compile(r'<tool[_-]?call>', re.I), 3.0, "tool_call_tag"),
    # -- user 侧：BFCL 包装指令 / 用户显式要求调用 --
    (re.compile(r"return only the tool calls?", re.I), 3.0, "bfcl_return_only"),
    (re.compile(r"function[- ]calling task", re.I), 3.0, "bfcl_fc_task"),
    (re.compile(r"current user turn", re.I), 2.0, "bfcl_current_turn"),
    (re.compile(r"decide (?:which|what) tool", re.I), 2.0, "user_decide_tool"),
    (re.compile(r"(?:调用|使用|帮我调|执行)[^。\n]{0,12}(?:工具|函数|接口|api)", re.I), 3.0, "zh_call_tool", True),
    (re.compile(r"(?:调|用)一下[^。\n]{0,12}(?:工具|函数|接口)", re.I), 2.0, "zh_call_tool2", True),
    # 动词 + 工具名的祈使形态（"book a flight"/"send a message"）语义太宽，不在此判——
    # 纯 user 层判 tool 意图不可靠是已知边界，靠 system 工具模板 fallback 兜住。
]

# 代码敏感的工具规则（4 元组标记的中文祈使短语）：高代码密度文本中不计分
_TOOL_CODE_SENSITIVE = {rule[2] for rule in TOOL_CALL_RULES if len(rule) == 4}

# 总结：生成式动词/后缀。负向排除先行——passage_retrieval 的 input 是裸摘要文本，
# "The text summarizes..."/"根据摘要" 是对已有摘要的指代，不是总结指令。
SUMMARIZE_RULES = [
    # -- 英文祈使/生成式 --
    (re.compile(r"\bsummar(?:ize|ise|ized)\b", re.I), 3.0, "en_summarize"),
    (re.compile(r"\bwrite (?:a|an|the|one[- ])?.{0,20}summary\b", re.I), 3.0, "en_write_summary"),
    (re.compile(r"\bone[- ]page summary\b", re.I), 3.0, "en_onepage"),
    (re.compile(r"\brecap\b", re.I), 2.0, "en_recap"),
    (re.compile(r"\bkey points\b", re.I), 2.0, "en_key_points"),
    # TL;DR 降权 + 次数封顶 2：网络文本里成片出现的是引用/定义不是请求
    # （triviaqa 有 passage 刷 8 次 TL;DR 的样例；真实请求 <=2 次）
    (re.compile(r"\btldr\b|\btl;dr\b", re.I), 1.0, "en_tldr", 2),
    # 句尾生成锚点（samsum 的 "Dialogue: ...\nSummary:"）
    (re.compile(r"summary\s*[:：]\s*$", re.I | re.M), 3.0, "en_summary_suffix"),
    # -- 中文（祈使语境：行首/句首/请|帮我 之后。新闻正文"在总结…基础上"等叙述形态
    #    靠此语境要求天然不命中，见 SUMMARIZE_NEGATIVE_RULES 注释） --
    (re.compile(r"(?:^|[\n。！!？?]\s*|请|帮我|麻烦|给我)(?:帮我)?(?:给我)?(?:总结|概括|提炼)", re.I | re.M), 3.0, "zh_summary"),
    (re.compile(r"会议纪要|写.{0,8}(?:纪要|摘要|总结)|生成.{0,8}(?:摘要|总结)|(?:会议)?总结助手", re.I), 3.0, "zh_write_summary"),
    (re.compile(r"会议总结\s*[:：]\s*$", re.I | re.M), 3.0, "zh_meeting_summary_suffix"),
    (re.compile(r"要点[:：]", re.I), 1.5, "zh_key_points"),
]

# 指代型"摘要"负向排除：命中则抵消 summarize 信号（不计入任何类得分）
SUMMARIZE_NEGATIVE_RULES = [
    (re.compile(r"the (?:text|following|article|passage|abstract) (?:summarizes|is a summary of|provides a summary)", re.I), "neg_descriptive_summary"),
    (re.compile(r"(?:以下|下面|这个|上述|该)?(?:是)?(?:一个|一段)?摘要[:：]?", re.I), "neg_zh_referential_abstract"),
    (re.compile(r"根据摘要|摘要所属|摘要来判断", re.I), "neg_zh_based_on_abstract"),
    (re.compile(r"\bis a summary of\b", re.I), "neg_en_is_summary_of"),
    # 叙述形态：描述既有研究/文章/技术做了什么，不是总结指令
    # （lsht 新闻正文"这一研究总结了…意义"、"在总结试点经验的基础上"、
    #  passage_retrieval_zh"可概括为四种模式"）
    (re.compile(
        r"总结了|总结出|总结过|可概括为|概括为|概括这部|概括该|"
        r"在总结.{0,12}(?:经验|基础上)|经(?:验)?总结|是对.{0,20}的总结|"
        r"这篇文本|该研究|这一研究|本文总结",
        re.I), "neg_zh_narrated_summary"),
]

# 寒暄/确认型问句负向排除：单发的"在吗/好吗/行吗/可以吗"是社交确认不是知识提问，
# 不应判 qa。限短文本（<=20 字符）整体即该形态，避免误伤"文档里有说怎么做吗？"这类真问句。
QA_NEGATIVE_RULES = [
    (re.compile(r"^.{0,16}(?:在|好|行|可以|对了|明白|清楚|懂)(?:吗|么)[?？。!]?\s*$", re.I), "neg_zh_greeting_question"),
    (re.compile(r"^(?:are|is)\s+you\s+(?:there|still\s+there|ok)[?!.]?\s*$", re.I), "neg_en_greeting_question"),
]

# 问答：宽松口径（问答/分类/检索/计数都算 qa）。注意 查/搜索/find 不作 qa 锚点
# （它们是工具意图的典型形态，加了会误伤 tool_call 的 fallback 通路）。
QA_RULES = [
    # -- 英文疑问词（句首或独立出现） --
    (re.compile(r"(?:^|[\n.!?]\s*)?(?:what|where|when|why|who|whom|which)\b", re.I), 2.0, "en_wh"),
    (re.compile(r"\bhow (?:many|much|do|does|did|can|would|should|to)\b", re.I), 2.0, "en_how"),
    # -- 中文疑问词 --
    (re.compile(r"什么|哪里|哪儿|为何|为什么|怎么|怎样|如何|多少|哪一|哪所|是谁|几个|吗[?？。]"), 2.0, "zh_wh"),
    (re.compile(r"多[大久重远高长]|几岁"), 2.0, "zh_wh2"),
    # 标点：小权重辅助（对话/代码里的 ? 是噪声源，靠权重差压制）
    (re.compile(r"[?？]"), 1.0, "question_mark"),
    # -- 模板锚点（基准 + 真实系统常见格式） --
    (re.compile(r"^question\s*[:：]", re.I | re.M), 2.0, "tpl_question"),
    (re.compile(r"问题\s*[:：]", re.I), 2.0, "tpl_question_zh"),
    (re.compile(r"^(?:query|answer)\s*[:：]", re.I | re.M), 1.5, "tpl_query_answer"),
    (re.compile(r"^type\s*[:：]\s*$", re.I | re.M), 2.0, "tpl_type"),
    (re.compile(r"类别\s*[:：]\s*$", re.I | re.M), 2.0, "tpl_category_zh"),
    # 分类/检索/计数指令（宽松口径收纳）
    (re.compile(r"determine the type|classify", re.I), 2.5, "cls_determine_type"),
    (re.compile(r"判断.{0,10}类别|新闻的类别", re.I), 2.5, "cls_zh_category"),
    (re.compile(r"which paragraph|paragraph.{0,20}(?:belongs? to|from)|段落.{0,6}编号|属于哪个段落", re.I), 2.5, "retr_paragraph"),
    (re.compile(r"count how many|how many unique", re.I), 2.5, "cnt_how_many"),
    (re.compile(r"answer the (?:question|query)|only give me the answer|answer the question based", re.I), 2.5, "ans_answer_question"),
    (re.compile(r"回答.{0,6}问题|请根据.{0,10}(?:回答|文章)", re.I), 2.5, "ans_zh_answer"),
]

# 代码密度检测：压制代码文本里的 ?/疑问词（repobench 的三目 ?: 等）。
# 用符号密度而非语句头模式：Python/Java 代码的 continuation 行、注释、装饰器
# 没有统一句头，但结构性符号的密度显著高于自然语言。
# 实测（LongBench 全量）：
#   - 符号集排除 ()（散文括注太常见）：repobench 代码 p10≈0.005 中位≈0.017；
#     自然语言（triviaqa 长文本）p90≈0.003。阈值 0.005 + 长度门槛 200（短中文
#     新闻一处 (Europa Press) 括注即可越阈，而真实代码输入 min=474 字符）。
#   - 稀疏代码（纯 import 块）由 _looks_like_code 的语句头检测兜住。
_CODE_SYMBOL_RE = re.compile(r"[{};=<>@&|]")
_CODE_DENSITY_THRESHOLD = 0.005  # 每字符代码符号占比超过此值视为代码文本
# 含分号的网络缩写：密度计算前剥离（分号是缩写一部分，非代码符号）
_NET_ABBREV_RE = re.compile(r"\btl;dr\b|\btldr\b", re.I)

# 对话 transcript 结构检测：`Speaker: text` 行成片出现（>=3 行）。
# samsum 的 input 是完整对话，里面的 ?/wh-词是聊天内容不是用户提问 -> 压制 qa。
_DIALOGUE_LINE_RE = re.compile(r"(?:^|\n)[A-Za-z一-龥][\w 一-龥]{0,15}:\s*\S")
_DIALOGUE_LINES_THRESHOLD = 3

# 信号阈值：加权总分 >= 该值才视为"该文本携带了任务信号"（用于 fallback 判断）
SIGNAL_THRESHOLD = 2.0


def _match_rules(rules, text):
    """返回 [(规则名, 权重, 次数), ...]。

    规则表支持两种条目：(pattern, weight, name) 三元组与 (pattern, name) 二元组
    （负向规则无权重，计 0.0）。
    """
    hits = []
    for rule in rules:
        if len(rule) >= 3:
            pattern, weight, name = rule[0], rule[1], rule[2]
            cap = rule[3] if len(rule) >= 4 else 3
        else:
            pattern, name = rule
            weight, cap = 0.0, 3
        found = pattern.findall(text)
        if found:
            hits.append((name, weight, len(found), min(len(found), cap)))
    return hits


def _score(rules, text):
    """加权总分。

    单 pattern 多次命中默认计 min(n,3) 次（规则可自带更小 cap），防长文本刷分；
    question_mark 单独递减（sqrt 语义）：真实问句通常 1-2 个问号，对话 transcript
    里成片的问号是聊天噪声，不应线性放大 qa 分。
    """
    import math

    total = 0.0
    for name, weight, n, capped in _match_rules(rules, text):
        if name == "question_mark":
            total += weight * min(math.sqrt(n), 3)
        else:
            total += weight * capped
    return total


def _code_density(text):
    """代码符号密度（每字符的 [{};=<>@&|] 占比）。

    - 短文本（<200 字符）返回 0：中文新闻里一处 (Europa Press) 括注就能把
      密度推过阈值（实测 lsht 71 字符 0.028 误报），而真实代码输入均在
      数百字符以上（repobench min=474）。
    - 先剥离 TL;DR 类网络缩写：其分号不是代码分号（triviaqa 有 passage
      含 8 个 TL;DR，密度 0.0063 误触代码判定；剥离后 0.0014）。
    """
    if not text or len(text) < 200:
        return 0.0
    stripped = _NET_ABBREV_RE.sub("", text)
    if not stripped:
        return 0.0
    return len(_CODE_SYMBOL_RE.findall(stripped)) / len(stripped)


def _looks_like_code(text):
    """代码文本判定：符号密度高，或开头连续多行 import/语句头（稀疏代码形态，
    如纯 import 块符号密度低但结构唯一）。"""
    if not text or len(text) < 60:
        return False
    if _code_density(text) >= _CODE_DENSITY_THRESHOLD:
        return True
    lines = [ln.strip() for ln in re.split(r"[\r\n]+", text) if ln.strip()]
    if len(lines) >= 3:
        head = lines[: min(6, len(lines))]
        import_like = sum(
            1 for ln in head
            if re.match(r"(?:import\s+[\w.*]|from\s+[\w.]+\s+import|package\s+[\w.]+;|#include|using\s+\w|@\w+)", ln)
        )
        if import_like >= min(3, len(head)):
            return True
    return False


def _is_dialogue_transcript(text):
    """`Speaker: text` 行 >= 3 视为对话转写文本。"""
    if not text:
        return False
    return len(_DIALOGUE_LINE_RE.findall(text)) >= _DIALOGUE_LINES_THRESHOLD


def classify_task_detailed(text: str) -> dict:
    """单文本核心规则。返回 label / 各类得分 / 命中规则明细。

    label: 'qa' | 'summarize' | 'tool_call' | 'other'（other = 无任何任务信号，
    由上层 decide 是否 fallback 到 system）
    """
    if not text or not text.strip():
        return {"label": "other", "scores": {}, "hits": {}, "signal": False, "code_density": 0.0}

    # 0) 代码文本判定：代码敏感的工具祈使规则（注释里的"调用该函数"）不计分
    density = _code_density(text)
    is_code = _looks_like_code(text)

    # 1) tool_call 结构锚点短路（结构性锚点不受代码密度压制）
    tool_hits = _match_rules(TOOL_CALL_RULES, text)
    if is_code:
        tool_hits = [h for h in tool_hits if h[0] not in _TOOL_CODE_SENSITIVE]
    tool_score = sum(weight * capped for _, weight, _, capped in tool_hits)
    if tool_score >= SIGNAL_THRESHOLD:
        return {
            "label": "tool_call",
            "scores": {"tool_call": tool_score, "summarize": 0.0, "qa": 0.0},
            "hits": {"tool_call": tool_hits},
            "signal": True,
            "code_density": density,
        }

    # 2) 负向排除先行：指代型"摘要"存在时，summarize 直接清零；
    #    寒暄确认型问句（"在吗？"）存在时，qa 信号清零
    neg_hits = _match_rules(SUMMARIZE_NEGATIVE_RULES, text)
    sum_negated = bool(neg_hits)
    qa_neg_hits = _match_rules(QA_NEGATIVE_RULES, text)
    qa_negated = bool(qa_neg_hits)

    sum_hits = _match_rules(SUMMARIZE_RULES, text)
    sum_score = 0.0 if sum_negated else _score(SUMMARIZE_RULES, text)

    qa_hits = _match_rules(QA_RULES, text)
    qa_score = 0.0 if qa_negated else _score(QA_RULES, text)

    # 3) 噪声形态压制：代码文本 / 对话转写文本里 ?/疑问词是内容不是指令
    if is_code:
        qa_score *= 0.2
    elif _is_dialogue_transcript(text):
        qa_score *= 0.2

    scores = {"tool_call": tool_score, "summarize": sum_score, "qa": qa_score}

    # 4) 判定
    if sum_score < SIGNAL_THRESHOLD and qa_score < SIGNAL_THRESHOLD:
        # 负向命中（指代型"摘要"）恰恰说明该文本是内容而非指令 -> 更需要 fallback 到 system
        label, signal = "other", False
    elif sum_score >= qa_score:
        label, signal = "summarize", True
    else:
        label, signal = "qa", True

    hits = {"qa": qa_hits, "summarize": sum_hits, "tool_call": tool_hits}
    if neg_hits:
        hits["summarize_negative"] = neg_hits
    if qa_neg_hits:
        hits["qa_negative"] = qa_neg_hits
    return {"label": label, "scores": scores, "hits": hits, "signal": signal, "code_density": density}


def classify_task(text: str) -> str:
    """返回 'qa' / 'summarize' / 'tool_call' / 'other'。"""
    return classify_task_detailed(text)["label"]


def classify_pipeline(user_prompt: str, long_text: str, system_prompt: str = "",
                      min_long_chars: int = 2000) -> dict | None:
    """管线入口：模拟真实问答系统的三段输入。

    - long_text 低于阈值：返回 None（不压缩，无需判断）。
    - 超阈值：先分析 user_prompt；为空或无信号命中 -> fallback 分析 system_prompt；
      两者都无信号 -> other。

    返回 dict(label, source, scores, hits)。source 标记判定来源：
    'user' | 'system' | 'none'。
    """
    if long_text is None or len(long_text) <= min_long_chars:
        return None

    result = classify_task_detailed(user_prompt)
    if result["signal"]:
        return {**result, "source": "user"}

    sys_result = classify_task_detailed(system_prompt)
    if sys_result["signal"]:
        return {**sys_result, "source": "system"}

    # 两级都无信号：保留 user 层的明细（更接近用户意图）便于排查
    return {**result, "source": "none"}


# ----------------------------------------------------------------------------
# BFCL 专用切分
# ----------------------------------------------------------------------------

_BFCL_ENV_RE = re.compile(
    r"Initial environment state:\s*\n(.*?)(?=\n\nPrevious turns|\n\nCURRENT user turn|$)", re.S
)
_BFCL_PREV_RE = re.compile(r"Previous turns and already executed gold calls:.*?(?=\nCURRENT user turn|$)", re.S)


def split_bfcl_user_prompt(user_prompt: str) -> tuple[str, str]:
    """把 BFCL 的 user_prompt 切成 (user_prompt_part, long_text_part)。

    long_text_part = 环境状态 JSON + 历史轮次块（压缩对象）；
    user_prompt_part = 包装指令 + CURRENT user turn（分析对象）。
    """
    long_parts = []
    rest = user_prompt

    m = _BFCL_PREV_RE.search(rest)
    if m:
        long_parts.append(m.group(0))
        rest = rest.replace(m.group(0), "\n")

    m = _BFCL_ENV_RE.search(rest)
    if m:
        long_parts.append(m.group(0))
        rest = rest.replace(m.group(0), "\n")

    return re.sub(r"\n{3,}", "\n\n", rest).strip(), "\n\n".join(long_parts).strip()


# ----------------------------------------------------------------------------
# 命令行调试
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) < 2:
        print(__doc__)
        _sys.exit(1)
    import json as _json

    text = " ".join(_sys.argv[1:])
    print(_json.dumps(classify_task_detailed(text), ensure_ascii=False, indent=2, default=str))
