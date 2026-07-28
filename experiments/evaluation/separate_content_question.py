"""长文本「文本内容 vs 用户提问」分离示例（纯中文 + jieba）

给定一段纯中文长文本，把其中两类东西分开：
1. 文本内容：陈述性正文，含正文里形式上像问句但其实是内容的句子
   （反问句、设问、引述/嵌入疑问）。
2. 用户提问：用户真正想问、期待被回答的疑问句。

判别三步（窗口内逐句执行）：
1. 疑问形式判定：问号 / 疑问代词词典 / 正反问（V不V / V没V 重叠）/ 选择问
   （「还是」）/ 句末语气词（吗/么/呢/吧）。
2. 反问排除（伪提问→归内容）：反问副词（难道/岂/莫非/不成）+ 固定反问构式
   （「难道…吗」「不是…吗」等）+ 嵌入疑问引导词（研究/探讨/分析…+ 疑问小句）。
3. 窗口回溯定位用户提问：
   - 末尾窗口从后向前回溯，开头窗口从前向后回溯；
   - 连续的「疑问形式且非反问且非嵌入」句子 → 用户提问；
   - 遇到「请回答下面的问题。」「问题如下：」等提问引导句 → 跳过，不中断回溯；
   - 遇到其他陈述句 → 停止回溯。
   反问/设问/嵌入疑问天然留在 content：反问被标记排除；设问（自问自答）和
   嵌入疑问前后都有陈述句阻断回溯，或在回溯起点之外。

无问号也能识别：问号只是诸多信号之一。「是否/是不是/有没有/能否/会不会/
能不能」等是非问、正反问书面合成词不是 V不V 重叠结构（正则抓不到），显式收录
进疑问词词典后，即便去掉问号也能靠这些标志词判定为疑问形式。

窗口策略（不全量分析）：
- 用 jieba 对整文分词一次，取「开头 N 个 token」+「末尾 N 个 token」两个窗口
  做疑问判别（用户提问要么在末尾 LongBench 式，要么在开头「先问再贴文档」式）。
- 窗口外正文直接归 content，不做 jieba 词性/句法分析。
- **取舍**：窗口越小越省时，但离边缘稍远的提问可能漏判（被挤出窗口）。
  默认 window_tokens=300 兼顾覆盖与成本；若提问可能距边缘更远，调大此值。

实测依据（影响词典设计）：
- jieba 默认模式把疑问代词（谁/什么/怎么）都标成 `r`，不细分 `ry` → 必须用词典。
- 语气词吗/呢/吧标 `y` → 可作辅助信号；「是否」标 `v`、「几/多少」标 `m` →
  不能依赖词性，需词典显式收录。

依赖：jieba（项目已有依赖，metrics.py 在用）。
运行：
    python experiments/evaluation/separate_content_question.py
"""

import re
import sys
from typing import Dict, List, Tuple

import jieba
import jieba.posseg as pseg


# --------------------------------------------------------------------------- #
# 词典与句法模式
# --------------------------------------------------------------------------- #

# 中文疑问代词（特指问标志）。注：jieba 默认模式把它们都标成 `r`，不细分 `ry`，
# 所以必须用词典显式收录，不能依赖词性。
ZH_WH_WORDS = {
    "什么", "啥", "谁", "哪", "哪里", "哪儿", "哪一", "哪几",
    "怎么", "怎样", "怎么样", "如何", "为何", "为什么", "干嘛",
    "几", "多少", "何", "何故", "何以", "何时", "何地", "何人",
    # 是非问 / 正反问标志词。书面合成词「是否」「有无」「能否」「可否」
    # 不是 V不V 重叠结构，正则抓不到，必须显式收录，否则去问号后无法识别。
    "是否", "是不是", "有无", "有没有", "能否", "可否", "会不会", "能不能",
    "可不可以", "能不能够", "要不要", "该不该",
}

# 句末疑问语气词。jieba 对吗/呢/吧标 `y`，其余可能不准，用词典兜底。
ZH_FINAL_PARTICLES = {"吗", "么", "嘛", "呢", "吧", "啊", "呀", "哇"}

# 反问副词 / 反问构式触发词。命中即判反问句，归入内容而非用户提问。
ZH_RHETORICAL_MARKERS = {
    "难道", "岂", "岂非", "莫非", "不成", "何尝", "何必", "哪里", "哪儿",
    "怎么", "怎能", "怎会", "怎么会", "怎能不", "岂能", "难道说",
}
# 固定反问构式（正则），形如「不是...吗」「还能...吗」「难道不是...吗」
ZH_RHETORICAL_PATTERNS = [
    re.compile(r"不是.*吗"),
    re.compile(r"难道.*吗"),
    re.compile(r"岂.*吗"),
    re.compile(r"怎么(能|会|可以).*吗"),
    re.compile(r"还能.*吗"),
    re.compile(r"何(尝|必|须).*吗"),
]

# 正反问结构：V不V / V没V（去不去/有没有/能不能）。V 限 1~2 字避免误匹配。
RE_V_NOT_V = re.compile(r"([一-龥]{1,2})不\1")
RE_V_MEI_V = re.compile(r"([一-龥]{1,2})没\1")

# 嵌入疑问引导词（「研究…哪种」「探讨…哪」「他问道…」）—— 含这类引导词时，
# 疑问小句是内容的一部分，不是用户提问。
# 注意：只用多字词，避免单字「问」误伤「问答/问句」等无关词。
ZH_EMBEDDING_LEADS = {
    "请问", "追问", "反问", "问道", "询问", "质问", "诘问",
    "讨论", "研究", "探讨", "调查", "分析", "思考",
    "不知道", "未知", "有待", "值得",
}

# 末尾回溯时允许「跳过」的引导句。这类句子形式上不是疑问句，但明显是
# 提问前的铺垫（「请回答下面的问题。」「问题如下：」），不应中断回溯。
# 仅当句子短且命中下列短语时才跳过，避免把正文陈述句误当引导句。
ZH_QUESTION_INTRO_PATTERNS = [
    re.compile(r"请.*回答"),
    re.compile(r"请.*回答.*问题"),
    re.compile(r"回答.*下面"),
    re.compile(r"回答.*以下"),
    re.compile(r"问题.*如下"),
    re.compile(r"如下[：:]?$"),
    re.compile(r"下列.*问题"),
    re.compile(r"以下.*问题"),
    re.compile(r"请.*解答"),
    re.compile(r"请.*分析.*问题"),
]


def is_question_intro(s: str) -> bool:
    """是否为提问前的引导句（短且命中引导短语）。回溯时可跳过，不中断。"""
    if len(s) > 20:  # 限长，避免误吞正文陈述句
        return False
    return any(p.search(s) for p in ZH_QUESTION_INTRO_PATTERNS)

# 句子切分标点
_SENT_END_PUNCT = set("。！？!?；;\n")
# 尾部需去掉的标点，用于取「最后一个实义汉字」
_TAIL_STRIP = "。！？!?；;，,、：: \n\t\"'\"'()[]（）"


# --------------------------------------------------------------------------- #
# 句子切分（仅在窗口内）
# --------------------------------------------------------------------------- #

def split_sentences(text: str) -> List[str]:
    """按中文句末标点切分，保留标点归属。"""
    sents: List[str] = []
    buf: List[str] = []
    for ch in text:
        buf.append(ch)
        if ch in _SENT_END_PUNCT:
            s = "".join(buf).strip()
            if s:
                sents.append(s)
            buf.clear()
    tail = "".join(buf).strip()
    if tail:
        sents.append(tail)
    return sents


def _real_last_char(s: str) -> str:
    """去掉尾部标点/空白后的最后一个汉字。"""
    s = s.rstrip(_TAIL_STRIP)
    return s[-1] if s else ""


# --------------------------------------------------------------------------- #
# 单句特征提取
# --------------------------------------------------------------------------- #

def _has_wh_word(s: str) -> Tuple[bool, List[str]]:
    """是否含疑问代词（词典匹配）。返回 (是否命中, 信号列表)。"""
    hits = [w for w in ZH_WH_WORDS if w in s]
    if hits:
        return True, [f"wh:{hits[0]}"]
    return False, []


def _has_v_not_v(s: str) -> Tuple[bool, List[str]]:
    """是否含正反问结构 V不V / V没V。"""
    if RE_V_NOT_V.search(s):
        return True, ["v不v"]
    if RE_V_MEI_V.search(s):
        return True, ["v没v"]
    return False, []


def _has_choice(s: str) -> Tuple[bool, List[str]]:
    """是否含选择问结构（「…还是…」）。简单启发式。"""
    if "还是" in s:
        return True, ["还是"]
    return False, []


def _has_final_particle(s: str) -> Tuple[bool, List[str]]:
    """句末是否为疑问语气词。同时用 jieba 词性 `y` 辅助验证。"""
    last = _real_last_char(s)
    signals: List[str] = []
    if last in ZH_FINAL_PARTICLES:
        signals.append(f"final_particle:{last}")
    # jieba 词性辅助
    for w, flag in pseg.cut(s[-4:] if len(s) > 4 else s):
        if flag == "y" and w in ZH_FINAL_PARTICLES:
            signals.append(f"jieba_y:{w}")
            break
    return bool(signals), signals


def is_interrogative_form(s: str) -> Tuple[bool, List[str]]:
    """判定是否为疑问形式（含问号 / 疑问词 / 正反 / 选择 / 语气词）。

    返回 (是否疑问形式, 命中信号列表)。
    反问句也是疑问形式的一种，由 is_rhetorical 单独标注。
    """
    signals: List[str] = []
    score = 0
    if s.rstrip().endswith(("?", "？")):
        score += 3
        signals.append("final '?'")

    for ok, sig in (_has_wh_word(s), _has_v_not_v(s),
                    _has_choice(s), _has_final_particle(s)):
        if ok:
            score += 2
            signals.extend(sig)

    return score >= 2, signals


def is_rhetorical(s: str) -> Tuple[bool, List[str]]:
    """判定是否为反问句（伪提问，归入内容）。

    强信号：反问副词命中 / 固定反问构式命中。
    """
    signals: List[str] = []
    hits = [w for w in ZH_RHETORICAL_MARKERS if w in s]
    if hits:
        signals.append(f"rhet_marker:{hits[0]}")
    for pat in ZH_RHETORICAL_PATTERNS:
        m = pat.search(s)
        if m:
            signals.append(f"rhet_pattern:{m.group(0)}")
    return bool(signals), signals


def has_embedding_lead(s: str) -> bool:
    """是否含嵌入疑问引导词（「他问…」「研究…哪种」），表示疑问小句属内容。"""
    return any(w in s for w in ZH_EMBEDDING_LEADS)


def extract_features(sent: str) -> Dict:
    """提取单句特征。返回 {is_q_form, is_rhetorical, is_embedded, signals}。"""
    q_form, q_signals = is_interrogative_form(sent)
    rhet, rhet_signals = is_rhetorical(sent)
    embedded = has_embedding_lead(sent)
    return {
        "is_q_form": q_form,
        "is_rhetorical": rhet,
        "is_embedded": embedded,
        "signals": q_signals + rhet_signals,
    }


# --------------------------------------------------------------------------- #
# 窗口定位：末尾回溯 / 开头回溯
# --------------------------------------------------------------------------- #

def _is_user_question_candidate(feat: Dict) -> bool:
    """是否为「用户提问」候选：疑问形式 且 非反问 且 非嵌入疑问。"""
    return feat["is_q_form"] and not feat["is_rhetorical"] and not feat["is_embedded"]


def locate_tail_user_questions(sents: List[str], feats: List[Dict]) -> List[int]:
    """从末尾句子向前回溯，连续的「用户提问候选」归为用户提问。

    - 用户提问候选（疑问形式且非反问且非嵌入）→ 加入。
    - 提问前引导句（「请回答下面的问题。」）→ 跳过，不中断回溯。
    - 其他句子 → 停止回溯。
    末尾若非候选且非引导句，返回空。
    """
    idxs: List[int] = []
    started = False  # 是否已遇到第一个候选句（避免把开头孤立引导句当提问）
    for i in range(len(sents) - 1, -1, -1):
        if _is_user_question_candidate(feats[i]):
            idxs.append(i)
            started = True
        elif started and is_question_intro(sents[i]):
            # 已进入提问块后才跳过引导句
            continue
        else:
            break
    idxs.reverse()
    return idxs


def locate_head_user_questions(sents: List[str], feats: List[Dict]) -> List[int]:
    """从开头句子向后扫描，连续的「用户提问候选」归为用户提问。

    - 用户提问候选 → 加入。
    - 提问前引导句（「问题如下：」）→ 跳过，不中断扫描。
    - 其他句子 → 停止。
    覆盖「先引导再提问再贴文档」与「先提问再贴文档」两种写法。
    """
    idxs: List[int] = []
    started = False
    for i in range(len(sents)):
        if _is_user_question_candidate(feats[i]):
            idxs.append(i)
            started = True
        elif not started and is_question_intro(sents[i]):
            continue  # 开头的引导句，跳过
        else:
            break
    return idxs


# --------------------------------------------------------------------------- #
# 整文分词 + 取头/尾窗口
# --------------------------------------------------------------------------- #

def take_window_text(text: str, n_tokens: int, side: str) -> str:
    """用 jieba 对整文分词一次，取头/尾 n_tokens 个 token 拼回字符串。

    side: "head" 或 "tail"。
    """
    tokens = list(jieba.cut(text))
    if side == "head":
        window = tokens[:n_tokens]
    elif side == "tail":
        window = tokens[-n_tokens:] if n_tokens < len(tokens) else tokens
    else:
        raise ValueError(f"side must be 'head' or 'tail', got {side!r}")
    return "".join(window)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def separate_content_question(
    text: str,
    window_tokens: int = 300,
) -> Dict[str, List[dict]]:
    """分离纯中文长文本的「文本内容」与「用户提问」。

    只分析开头/末尾各 window_tokens 个 token 的窗口；窗口外正文直接归 content。

    返回 {content_list, user_question_list}，每项 {text, role, signals, where}。
    where 标注该句来自哪个窗口（head/tail）或 body（窗口外正文）。
    """
    # 窗口外正文（不分析，直接归 content）
    tokens = list(jieba.cut(text))
    n = len(tokens)
    head_cut = min(window_tokens, n)
    tail_cut = max(0, n - window_tokens)
    body_tokens = tokens[head_cut:tail_cut] if tail_cut > head_cut else []
    body_text = "".join(body_tokens)
    body_sents = split_sentences(body_text)

    content_list: List[dict] = [
        {"text": s, "role": "content", "signals": [], "where": "body"}
        for s in body_sents
    ]

    # 头/尾窗口分析。按句子文本去重：任一窗口判为 user_question 即归入用户提问。
    q_text_to_item: Dict[str, dict] = {}
    content_text_to_item: Dict[str, dict] = {}

    for side, window_text in (
        ("head", take_window_text(text, window_tokens, "head")),
        ("tail", take_window_text(text, window_tokens, "tail")),
    ):
        win_sents = split_sentences(window_text)
        win_feats = [extract_features(s) for s in win_sents]
        locator = locate_head_user_questions if side == "head" else locate_tail_user_questions
        q_idxs = set(locator(win_sents, win_feats))
        for i, s in enumerate(win_sents):
            if i in q_idxs:
                # 升格为用户提问（即便之前在 content 里出现过，也移走）
                if s in content_text_to_item:
                    del content_text_to_item[s]
                if s not in q_text_to_item:
                    q_text_to_item[s] = {
                        "text": s,
                        "role": "user_question",
                        "signals": win_feats[i]["signals"],
                        "where": side,
                    }
            else:
                if s not in q_text_to_item and s not in content_text_to_item:
                    content_text_to_item[s] = {
                        "text": s,
                        "role": "content",
                        "signals": win_feats[i]["signals"],
                        "where": side,
                    }

    # 拼接最终结果：正文(body, 不分析) + 窗口内句子，保持出现顺序
    # body 句子已先入 content_list；再追加窗口去重后的 content。
    content_list.extend(content_text_to_item.values())

    return {"content_list": content_list, "user_question_list": list(q_text_to_item.values())}


# --------------------------------------------------------------------------- #
# 打印与自检 demo
# --------------------------------------------------------------------------- #

def _print_bucket(title: str, items: List[dict]) -> None:
    print(f"\n【{title}】共 {len(items)} 条")
    for i, it in enumerate(items, 1):
        sig = ", ".join(it["signals"]) if it["signals"] else "—"
        print(f"  {i:>2} [{it['where']}|{it['role']}] {it['text']}")
        if it["signals"]:
            print(f"      signals: {sig}")


# demo：长正文（>1000 jieba token），混入反问、设问、嵌入疑问；末尾是真正的用户提问块。
DEMO_TEXT = """
PromptZip 是一个用于提示压缩的研究项目。它旨在把过长的提示词在送入大语言模型之前先做压缩，从而在不显著损失回答质量的前提下降低推理成本与延迟。项目实现了多种压缩策略，覆盖了从片段级选择到 token 级裁剪的不同粒度。
压缩的研究背景源于这样一个事实：随着上下文长度增加，主流大语言模型的延迟与显存占用都会显著上升。长上下文不仅拖慢首 token 响应，还会推高单次调用的费用，使大规模评测变得昂贵。因此如何在保留关键信息的前提下减少 token 数量，成为一个有工程价值的问题。
研究者探讨了哪种策略最适合长文本问答任务，但结论尚不统一。不同数据集、不同语言、不同压缩率下，各方法的相对表现并不稳定，需要更系统的对比。
项目内置的第一类策略是基于重排序器的压缩。RerankCompressor 使用 Qwen3-Reranker 对切分后的文本片段打分，按相关性保留高分区段。它支持多种选择模式，包括 topk、topp、cluster、cluster-zscore 以及 mmr。其中 mmr 在保证相关性的同时兼顾片段之间的多样性，避免保留过多语义重复的片段。
项目内置的第二类策略是基于困惑度的压缩。PPLCompressor 利用语言模型对每个片段计算困惑度，丢弃信息量较低、可预测性较强的片段。这类方法背后的假设是：模型本来就能猜到的内容，即便不写进提示，模型也能补全，因此可以安全删除。
项目内置的第三类策略是基于注意力分数的压缩。AttnScoreCompressor 借助模型内部的注意力分布来判断哪些 token 对当前问题更重要，从而保留高注意力区段。难道基于注意力的方法在摘要任务上表现不好吗？显然并非如此，它在部分摘要数据集上确实带来了可观的保留率与质量平衡。
第四类是基于嵌入相似度的压缩。EmbeddingCompressor 把片段与问题分别编码为向量，按相似度排序后保留与问题最贴近的片段。这类方法依赖一个稳健的句向量模型，对中文场景需要选用合适的中文嵌入模型。
第五类是 token 级压缩，对应 LongLLMLinguaTokenCompressor 与 PromptCompressor。前者做迭代式 token 裁剪，后者则是 LLMLingua 与 LLMLingua-2 的综合实现，支持 context 级、sentence 级与 token 级的多层过滤，并能处理结构化 JSON 输入。
此外还有基于 Qwen 模型的摘要式压缩。QwenCompressor 与 QwenVLLMCompressor 把长文本先做摘要，再用摘要替代原文送入下游模型。摘要式压缩天然会损失细节，但在超长文本下能换取可观的压缩率。
为什么需要提示压缩？因为长上下文会显著增加延迟与费用，而且在很多任务里，真正与回答相关的信息只占提示的一小部分。把无关内容裁掉，既省钱又不伤质量。
实验评测主要在 LongBench 基准上进行。LongBench 涵盖了摘要、单文档问答、多文档问答、抽取、检索、代码与数学等多个任务族，支持中英双语，是衡量长上下文模型能力的常用基准。项目用 vLLM 做推理后端，以便在评测时获得更稳定的吞吐。
配置由 YAML 文件定义，路径为 experiments/config。每份配置包含三个组件块：reranker_config 定义重排序模型、分块大小、压缩率与引擎选择；compressor_config 指定压缩器类型与参数；llm_config 指定目标大语言模型与 vLLM 采样参数。修改配置即可快速切换实验组合。
评测脚本位于 experiments/evaluation 目录下。eval_longbench 是主入口，负责加载数据集、构建压缩器与目标模型、异步生成预测并计算每个数据集的指标，最后写出 JSON 与 CSV 结果。benchmark_e2e 则用于在多个目标 token 长度下做端到端延迟基准。
项目没有配置正式的 lint 与 CI。所有测试与评测都是手工本地进行的。仓库里只有一个基于 assert 的独立测试脚本作为最小自检，遵循同样的约定。
依赖管理使用 uv，Python 版本固定为 3.12.3。flash-attn 有特殊的构建依赖，需要在安装时与 torch 匹配。vLLM、onnxruntime-gpu 等组件都依赖 CUDA，因此实际运行需要一张可用的 GPU。
请回答下面的问题。
那么在 LongBench 上哪种压缩策略的 F1 最高？是否有实验对比 rerank 与 PPL 方法在中文问答上的推理延迟？基于注意力的方法是否对超长文本更鲁棒？
"""


def main() -> None:
    result = separate_content_question(DEMO_TEXT, window_tokens=300)
    _print_bucket("文本内容 content", result["content_list"])
    _print_bucket("用户提问 user_question", result["user_question_list"])

    # assert 自检（项目约定：assert-based standalone）
    uq = result["user_question_list"]
    assert uq, "应识别出末尾用户提问"
    uq_texts = [it["text"] for it in uq]
    print("\n[自检] 末尾用户提问识别结果：", uq_texts)
    assert any("F1 最高" in t for t in uq_texts), "末尾真提问（F1 最高）应被识别"
    assert any("推理延迟" in t for t in uq_texts), "末尾连续提问（推理延迟）应被识别"
    assert any("更鲁棒" in t for t in uq_texts), "末尾连续提问（更鲁棒）应被识别"
    # 引导句「请回答下面的问题。」被跨过，不进 user_question 也不阻断
    assert all("请回答" not in t for t in uq_texts), "引导句不应进入 user_question"

    # 反问 / 设问 / 嵌入疑问必须留在 content，不能进 user_question
    c_texts = [it["text"] for it in result["content_list"]]
    assert any("难道" in t for t in c_texts), "反问句应留在 content"
    assert any("为什么需要提示压缩" in t for t in c_texts), "设问句应留在 content"
    assert any("探讨了哪种策略" in t for t in c_texts), "嵌入疑问应留在 content"
    print("\n[OK] 自检通过。")


if __name__ == "__main__":
    main()
