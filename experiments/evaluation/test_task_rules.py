"""任务类型识别规则（src/task_rules.py）的测试脚本。

三组测试（见 PLAN_task_rules.md）：
1. 冒烟集：手写真实问答系统流量 ~25 条，逐一 assert，必须 100%（最接近最终目标）。
2. LongBench + BFCL 全量：按真实系统三段结构 (system, user, long_text) 构造输入的 proxy 指标。
3. 参考模式：system+user 合并分析的上界对照，量化两级 fallback 的信息损失（主要是 qmsum）。

数据映射：
- LongBench: system=dataset2prompt[ds][0], user=sample['input'], long_text=sample['context']
- BFCL: system=SYSTEM_WITH_TOOLS, user/long_text=split_bfcl_user_prompt(user_prompt)

dev/test 按行号奇偶 50/50 划分（dev=偶数行），规则在 dev 上调，test 出报告。

用法:
    python test_task_rules.py                # 全量
    python test_task_rules.py --limit 20     # 每子集取前 20 条（调试）
    python test_task_rules.py --min-long-chars 1000  # 触发阈值敏感性
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.append(os.path.dirname(__file__))

from task_rules import (  # noqa: E402
    SIGNAL_THRESHOLD,
    classify_pipeline,
    classify_task,
    classify_task_detailed,
    split_bfcl_user_prompt,
)
from eval_longbench import dataset2prompt  # noqa: E402  只 import 模板，不触发其重依赖（vllm 等在函数内才用）
from eval_bfcl_parallel_multi_turn import SYSTEM_WITH_TOOLS  # noqa: E402

# ----------------------------------------------------------------------------
# 数据位置与期望标签
# ----------------------------------------------------------------------------

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
LONGBENCH_DIR = os.path.join(REPO_ROOT, "datasets", "LongBench", "data")
BFCL_PATH = os.path.join(REPO_ROOT, "experiments", "data", "eval.jsonl")

DATASET2TASK = {
    # qa（宽松口径：问答/分类/检索/计数）
    "narrativeqa": "qa",
    "qasper": "qa",
    "multifieldqa_en": "qa",
    "multifieldqa_zh": "qa",
    "hotpotqa": "qa",
    "2wikimqa": "qa",
    "musique": "qa",
    "triviaqa": "qa",
    "dureader": "qa",
    "trec": "qa",
    "lsht": "qa",
    "passage_retrieval_en": "qa",
    "passage_retrieval_zh": "qa",
    "passage_count": "qa",
    # summarize
    "gov_report": "summarize",
    "qmsum": "summarize",
    "multi_news": "summarize",
    "vcsum": "summarize",
    "samsum": "summarize",
    # other（代码补全，无 QA/总结信号）
    "lcc": "other",
    "repobench-p": "other",
    # BFCL 全部 tool_call
    "BFCL": "tool_call",
}

# qmsum 特殊：140/200 input 为问句形态，严格两级判定下判 qa 是"线上方向正确"的设计代价
# （真实文档问答就该路由 qa），不设门槛，单列报告。
QMSUM_EXEMPT = {"qmsum"}

# ----------------------------------------------------------------------------
# 冒烟集：手写真实问答系统流量（一等公民，必须 100%）
# ----------------------------------------------------------------------------

# (user_prompt, system_prompt, expected) —— system 为真实系统的助手/工具模板形态
SMOKE_CASES = [
    # QA：完整问句
    ("What is the refund policy for December orders?", "You are a helpful assistant.", "qa"),
    ("Qwen2.5 的上下文窗口是多大？", "You are a helpful assistant.", "qa"),
    ("这部小说的主角最后怎么样了？", "你是一个阅读助手。", "qa"),
    (
        "how many unique paragraphs are there in the given set of paragraphs after removing duplicates?",
        "You are a helpful assistant.",
        "qa",
    ),
    # QA：关键词查询（无信号 -> fallback system 的 QA 助手模板）
    ("iPhone 15 价格", "你是一个中文问答助手。请根据给定文章回答问题。", "qa"),
    ("qwen2.5 context window", "You are a QA assistant. Answer the question based on the given documents.", "qa"),
    ("热诚传说结局", "你是一个中文问答助手。请根据给定文章回答问题。", "qa"),
    # QA：检索/分类形态
    ("Determine the type of the following question.", "You are a classification assistant.", "qa"),
    # summarize：显式请求
    ("帮我总结一下这份会议纪要", "You are a helpful assistant.", "summarize"),
    ("Summarize the key points of this report", "You are a helpful assistant.", "summarize"),
    ("Write a one-page summary of the report.", "You are a helpful assistant.", "summarize"),
    ("总结", "You are a helpful assistant.", "summarize"),
    ("概括这段话的主要观点", "你是一个阅读助手。", "summarize"),
    # tool_call：system 含工具模板（真实系统的主路径）
    ("帮我查明天北京天气", "You have access to the following tools: get_weather, book_flight.", "tool_call"),
    ("Book a flight to NYC tomorrow morning", "You have access to the following tools: book_flight.", "tool_call"),
    # tool_call：用户显式要求
    ("调用搜索工具查一下量子计算的最新进展", "You are a helpful assistant.", "tool_call"),
    ("Return only the tool calls for the CURRENT user turn.", "You are an expert in composing functions.", "tool_call"),
    # other：闲聊/寒暄/无任务信号
    ("你好", "You are a helpful assistant.", "other"),
    ("hello", "You are a helpful assistant.", "other"),
    ("讲个笑话", "You are a helpful assistant.", "other"),
    ("在吗？", "You are a helpful assistant.", "other"),
    ("谢谢，明白了", "You are a helpful assistant.", "other"),
    ("今天心情不太好", "You are a helpful assistant.", "other"),
    ("再见", "You are a helpful assistant.", "other"),
    # other：写作/翻译/润色等非 QA 非总结任务（真实流量常见形态）
    ("帮我翻译一下这段话成英文", "You are a helpful assistant.", "other"),
    ("translate this paragraph to French", "You are a helpful assistant.", "other"),
    ("帮我写一封请假邮件", "You are a helpful assistant.", "other"),
    ("写一首关于秋天的诗", "You are a helpful assistant.", "other"),
    ("帮我润色这段话", "You are a helpful assistant.", "other"),
    # 边界：粘贴长内容 + 简短问句 -> qa；真问句不被寒暄负向误伤
    ("这是什么意思？", "You are a helpful assistant.", "qa"),
    ("文档里有说怎么做吗？", "You are a helpful assistant.", "qa"),
    # ---- 疑问句形态的非 QA（report/QUESTION_FORM_NON_QA.md 六类）----
    # A 疑问形式祈使：问号是礼貌包装，意图由动词决定
    ("可以帮我把这段话翻译成英文吗？", "You are a helpful assistant.", "other"),
    ("你能写一首诗吗？", "You are a helpful assistant.", "other"),
    ("可以帮我查一下明天的天气吗？", "You are a helpful assistant.", "other"),
    ("你能帮我总结一下这篇文章吗？", "You are a helpful assistant.", "summarize"),
    ("Can you summarize this report?", "You are a helpful assistant.", "summarize"),
    # B 对话中间态：质疑/追问上一轮
    ("你确定吗？", "You are a helpful assistant.", "other"),
    ("你刚才说的是什么？", "You are a helpful assistant.", "other"),
    # C 助手元问题：能力/身份询问
    ("你能做什么？", "You are a helpful assistant.", "other"),
    ("what can you do?", "You are a helpful assistant.", "other"),
    ("你是什么模型？", "You are a helpful assistant.", "other"),
    ("who are you?", "You are a helpful assistant.", "other"),
    # D 意见征询：问主观偏好
    ("What do you think about this?", "You are a helpful assistant.", "other"),
    ("你觉得哪个方案好？", "You are a helpful assistant.", "other"),
    # E 反问句：不期待回答
    ("这难道不是明摆着的吗？", "You are a helpful assistant.", "other"),
    ("难道就我一个人觉得贵吗？", "You are a helpful assistant.", "other"),
    # F 对照：请求框架内的真 QA 不被降权误伤
    ("你能告诉我退款政策是什么吗？", "You are a helpful assistant.", "qa"),
    ("Could you tell me what the refund policy is?", "You are a helpful assistant.", "qa"),
    # ---- 非 QA 任务的问句形态（LongBench-Pro/v2 鲁棒性，PLAN_task_rules_robustness.md）----
    # MC 选项模板（引用对齐/选择题形态）-> other（问句是选项模板噪声）
    ("以下哪项说法正确？\nA、选项一\nB、选项二", "You are a helpful assistant.", "other"),
    ("Which one is correct? Output the answer option letter ('A'或'B'或'C').", "You are a helpful assistant.", "other"),
    # 排序祈使 -> other
    ("请根据时间顺序恢复下列事件的排序", "You are a helpful assistant.", "other"),
    ("Please rearrange the fragments in chronological order.", "You are a helpful assistant.", "other"),
    # 矛盾检查 -> other
    ("请找出文档中的矛盾之处", "You are a helpful assistant.", "other"),
    ("Identify any inconsistencies between the two sections.", "You are a helpful assistant.", "other"),
    # 版本对比 -> other
    ("对比两个版本的差异并说明", "You are a helpful assistant.", "other"),
    # 计算祈使 -> other（"相差多少年"是数值推理不是文档问答）
    ("请计算 2024 与 1998 相差多少年", "You are a helpful assistant.", "other"),
    # 翻译祈使 -> other
    ("请把这段话翻译成英文", "You are a helpful assistant.", "other"),
    # 引用对齐："Generated Summary:" 在文本中段（非末尾）-> 非 summarize
    ("Here is the text. Generated Summary: first sentence.\nPlease align Sentence 1 to its source.", "You are a helpful assistant.", "other"),
    # T4 zh 祈使形态 -> summarize
    ("请将全文整理成一段不超过200字的摘要", "You are a helpful assistant.", "summarize"),
    ("以此形成一篇摘要，不超过150字", "You are a helpful assistant.", "summarize"),
    # en 生成式祈使 -> summarize
    ("Based on the text, generate a summary of no more than 200 words.", "You are a helpful assistant.", "summarize"),
    # 对照：真问句不被 MC/祈使负向误伤
    ("What is the average magnetic moment per column in these films?", "You are a helpful assistant.", "qa"),
    ("文档里提到的毛利率是多少？", "You are a helpful assistant.", "qa"),
]

# long_text 触发用：冒烟集统一给一段超阈值的占位长文本
SMOKE_LONG_TEXT = "背景文档。" * 2000  # 10000 字符 > 默认阈值


# ----------------------------------------------------------------------------
# 数据加载与三段构造
# ----------------------------------------------------------------------------


def load_longbench(limit=None):
    """yield (tag, system, user, long_text, expected)；tag=子集名。"""
    for ds, expected in DATASET2TASK.items():
        if ds == "BFCL":
            continue
        path = os.path.join(LONGBENCH_DIR, f"{ds}.jsonl")
        if not os.path.exists(path):
            print(f"[warn] 缺少 {path}，跳过 {ds}")
            continue
        system = dataset2prompt[ds][0]["content"]
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                d = json.loads(line)
                if limit and i >= limit:
                    break
                yield ds, system, d["input"], d["context"], expected


def load_bfcl(limit=None):
    """yield (tag, system, user, long_text, expected)。"""
    with open(BFCL_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            d = json.loads(line)
            user_part, long_part = split_bfcl_user_prompt(d["user_prompt"])
            yield "BFCL", SYSTEM_WITH_TOOLS, user_part, long_part, DATASET2TASK["BFCL"]


def load_all(limit=None):
    return list(load_longbench(limit)) + list(load_bfcl(limit))


# ----------------------------------------------------------------------------
# 评测
# ----------------------------------------------------------------------------


def run_smoke(min_long_chars):
    print("=" * 70)
    print("① 冒烟集（手写真实流量，必须 100%）")
    print("=" * 70)
    passed, failed = 0, []
    for user, system, expected in SMOKE_CASES:
        result = classify_pipeline(user, SMOKE_LONG_TEXT, system, min_long_chars)
        assert result is not None, "冒烟集 long_text 必超阈值"
        got = result["label"]
        if got == expected:
            passed += 1
        else:
            failed.append((user, system, expected, got, result.get("source"), result["hits"]))
    total = len(SMOKE_CASES)
    print(f"通过 {passed}/{total}")
    for user, system, expected, got, source, hits in failed:
        print(f"  ✗ {user[:50]!r} 期望 {expected} 实得 {got} (source={source})")
        print(f"    system={system[:60]!r}")
        print(f"    hits={json.dumps(hits, ensure_ascii=False, default=str)[:300]}")
    return passed == total


def run_full(samples, min_long_chars, mode="pipeline"):
    """mode='pipeline': 严格两级 (user -> system)；mode='merged': system+user 合并（参考上界）。"""
    dev_rows = [s for i, s in enumerate(samples) if i % 2 == 0]
    test_rows = [s for i, s in enumerate(samples) if i % 2 == 1]
    reports = {}
    for split, rows in [("dev", dev_rows), ("test", test_rows)]:
        reports[split] = _evaluate(rows, min_long_chars, mode)
    return reports


def _evaluate(rows, min_long_chars, mode):
    n_total = 0
    n_correct = 0
    per_ds = defaultdict(
        lambda: {
            "n": 0,
            "correct": 0,
            "exempt_n": 0,
            "exempt_correct": 0,
            "source": Counter(),
            "errors": [],
            "total": 0,
        }
    )
    confusion = Counter()  # (expected, got) -> n
    triggered = Counter()  # 触发率统计（long_text 超阈值）
    triggered_total = Counter()  # 每子集总行数（触发率分母）

    for tag, system, user, long_text, expected in rows:
        if mode == "merged":
            combined = (system + "\n" + user).strip()
            r = classify_task_detailed(combined)
            result = {**r, "source": "merged"}
        else:
            result = classify_pipeline(user, long_text, system, min_long_chars)
            if result is None:
                # 未触发（long_text 短于阈值）——生产里不会进分类器，不计入标签评测
                per_ds[tag]["total"] += 1
                if len(long_text or "") > min_long_chars:
                    triggered[tag] += 1
                continue
        got = result["label"]

        n_total += 1
        is_exempt = tag in QMSUM_EXEMPT
        entry = per_ds[tag]
        entry["n"] += 1
        entry["total"] += 1
        entry["source"][result["source"]] += 1
        if len(long_text or "") > min_long_chars:
            triggered[tag] += 1

        correct = got == expected
        if correct:
            n_correct += 1
            entry["correct"] += 1
            if is_exempt:
                entry["exempt_correct"] += 1
        confusion[(expected, got)] += 1
        if is_exempt:
            entry["exempt_n"] += 1
        elif not correct and len(entry["errors"]) < 3:
            entry["errors"].append(
                {
                    "user": user[:200],
                    "expected": expected,
                    "got": got,
                    "source": result["source"],
                    "scores": result["scores"],
                    "hits": {k: v for k, v in result["hits"].items() if v},
                }
            )

    labels = ["qa", "summarize", "tool_call", "other"]
    prf = {}
    for lab in labels:
        tp = confusion[(lab, lab)]
        fp = sum(v for (e, g), v in confusion.items() if g == lab and e != lab)
        fn = sum(v for (e, g), v in confusion.items() if e == lab and g != lab)
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
        prf[lab] = (prec, rec, f1)

    return {
        "n_total": n_total,
        "n_correct": n_correct,
        "per_ds": dict(per_ds),
        "confusion": dict(confusion),
        "prf": prf,
        "triggered": dict(triggered),
    }


def print_report(name, rep):
    print("=" * 70)
    print(name)
    print("=" * 70)
    acc = rep["n_correct"] / rep["n_total"] if rep["n_total"] else 0
    print(f"总体: {rep['n_correct']}/{rep['n_total']} = {acc:.4f}")
    print("\n各类 P/R/F1:")
    for lab, (p, r, f) in rep["prf"].items():
        ps = f"{p:.3f}" if p is not None else "-"
        rs = f"{r:.3f}" if r is not None else "-"
        fs = f"{f:.3f}" if f is not None else "-"
        print(f"  {lab:10s} P={ps} R={rs} F1={fs}")
    print("\n混淆矩阵 (expected -> got):")
    for (e, g), n in sorted(rep["confusion"].items()):
        mark = "" if e == g else "  ← 误判"
        print(f"  {e:10s} -> {g:10s} : {n}{mark}")
    print("\n每子集准确率 (source 分布: user/system/none|untriggered):")
    for ds, entry in sorted(rep["per_ds"].items()):
        base_n = entry["n"] - entry["exempt_n"]
        base_c = entry["correct"] - entry["exempt_correct"]
        line = f"  {ds:24s} {entry['correct']}/{entry['n']}"
        if entry["exempt_n"]:
            line += f"（qmsum 豁免口径 {base_c}/{base_n}）"
        src = dict(entry["source"])
        line += f"  src={src}"
        print(line)
        for err in entry["errors"]:
            print(f"      ✗ 期望 {err['expected']} 实得 {err['got']} (source={err['source']})")
            print(f"        user: {err['user']!r}")
            print(f"        scores: {err['scores']}")
            hits_str = json.dumps(err["hits"], ensure_ascii=False, default=str)
            print(f"        hits: {hits_str[:400]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="每子集取前 N 条（调试）")
    ap.add_argument("--min-long-chars", type=int, default=2000, help="long_text 触发阈值（字符）")
    ap.add_argument("--skip-smoke", action="store_true")
    args = ap.parse_args()

    ok = True

    # ① 冒烟集
    if not args.skip_smoke:
        ok &= run_smoke(args.min_long_chars)
        print()

    # ② 全量 pipeline 模式（严格两级）
    samples = load_all(args.limit)
    print(f"加载 {len(samples)} 条（LongBench + BFCL，limit={args.limit}）")
    reports = run_full(samples, args.min_long_chars, mode="pipeline")
    print_report("② pipeline 模式（user -> system 两级，主指标）", reports["test"])

    # ③ 参考模式（merged，上界对照）
    merged_reports = run_full(samples, args.min_long_chars, mode="merged")
    print_report("③ merged 参考模式（system+user 合并，上界对照）", merged_reports["test"])

    # ---- 断言（test split，pipeline 模式）----
    rep = reports["test"]
    acc = rep["n_correct"] / rep["n_total"]
    _, tool_recall, _ = rep["prf"]["tool_call"]
    _, qa_recall, _ = rep["prf"]["qa"]
    # 非 qmsum 的 summarize recall
    sum_tp = rep["confusion"].get(("summarize", "summarize"), 0)
    sum_fn = sum(v for (e, g), v in rep["confusion"].items() if e == "summarize" and g != "summarize")
    qmsum_n = rep["per_ds"].get("qmsum", {}).get("n", 0)
    sum_recall_no_qmsum = (sum_tp - qmsum_n * 0) / (sum_tp + sum_fn - 0) if (sum_tp + sum_fn) else None
    # 上面的 qmsum 豁免需要从分子分母里扣掉 qmsum 的行——直接用 per_ds 算更稳
    qmsum_correct = rep["per_ds"].get("qmsum", {}).get("correct", 0)

    print("=" * 70)
    print("验收断言")
    print("=" * 70)
    failures = []
    # 1. 总体 >= 95%
    if acc < 0.95:
        failures.append(f"总体准确率 {acc:.4f} < 0.95")
    # 2. tool_call recall 100%
    if tool_recall is not None and tool_recall < 1.0:
        failures.append(f"tool_call recall {tool_recall:.3f} < 1.0")
    # 3. 非 qmsum 的 summarize recall 100%
    non_qmsum_sum_tp = sum_tp - qmsum_correct
    non_qmsum_sum_total = (sum_tp + sum_fn) - qmsum_n
    if non_qmsum_sum_total > 0 and non_qmsum_sum_tp < non_qmsum_sum_total:
        failures.append(f"非 qmsum summarize recall {non_qmsum_sum_tp}/{non_qmsum_sum_total} < 1.0")
    # 4. qa recall >= 95%
    if qa_recall is not None and qa_recall < 0.95:
        failures.append(f"qa recall {qa_recall:.3f} < 0.95")

    if failures:
        for f in failures:
            print(f"  ✗ {f}")
        ok = False
    else:
        print("  全部通过 ✓")
        print(
            f"  总体 {acc:.4f} | tool_call R={tool_recall} | qa R={qa_recall:.3f} | "
            f"非qmsum summarize R={non_qmsum_sum_tp}/{non_qmsum_sum_total} | "
            f"qmsum {qmsum_correct}/{qmsum_n}（豁免，仅报告）"
        )

    # 触发率报告
    print("\n触发率（long_text > 阈值 比例，未触发行不进标签评测）:")
    for ds, entry in sorted(rep["per_ds"].items()):
        n_trig = rep["triggered"].get(ds, 0)
        total_n = entry["total"]
        if total_n:
            print(f"  {ds:24s} {n_trig}/{total_n} = {n_trig / total_n:.2f}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
