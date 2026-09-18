#!/usr/bin/env python
"""任务规则鲁棒性测试：LongBench-Pro / LongBench-v2 非 QA/Summ 任务混入。

口径（PLAN_task_rules_robustness.md，用户明确指示）：**低召回、高精度**——
qa/summarize 判定宁可漏判（真 QA 掉到 other 可接受），但判成 qa/summarize 的
必须是真 QA/Summ 任务。非 QA/Summ 任务被误判为 qa/summarize/tool_call 即为违规。

标签口径（严格 benchmark 分类）：
- LB-Pro: T3=qa, T4=summarize, 其余 9 个 primary_task=other（1260 条）
- v2: Single/Multi-Doc QA=qa（300），其余 4 域=other（203 条）

输入构造（对齐真实评测管线）：
- LB-Pro: system="", user=question_nonthinking, long_text=context
  （官方管线：context + 4 换行 + question 单条 user message，无 system）
- v2 主模式（with-choices）: user = 官方 0shot 模板的 question+choices 渲染
  （THUDM/LongBench prompts/0shot.txt："What is the correct answer to this
  question: $Q$\nChoices:\n(A) …\n(B) …\n(C) …\n(D) …"）
- v2 对照模式（bare）: user = question 原文（不渲染选项），量化模板信号承载

已知边界（设计代价，不设门槛，见报告）：
- MC 选项负向规则使 LB-Pro T3 / v2 QA 域的 MC 形态真 QA 大量掉 other
  （低召回口径接受；线上靠 system QA 模板 fallback 兜底）
- 裸问句形态的非 QA 任务（T8 计算问句/T11 对话追踪/v2-dialogue 等）与真
  doc-QA 表层不可分，属设计边界（同 qmsum 先例）

用法：
    python test_task_rules_robustness.py            # 全量（LB-Pro 1500 + v2 503）
    python test_task_rules_robustness.py --limit 20 # 调试模式
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from src.task_rules import classify_pipeline  # noqa: E402

LBPRO_PATH = os.path.join(REPO_ROOT, "datasets", "LongBench-Pro", "longbench_pro.json")
V2_PATH = os.path.join(REPO_ROOT, "datasets", "LongBench-v2", "data.json")

# ----------------------------------------------------------------------------
# 数据加载与标签映射
# ----------------------------------------------------------------------------

# LB-Pro primary_task 前缀 -> 期望标签（严格 benchmark 分类）。
# primary_task 字段形如 "T3. Evidence-Grounded QA"，前缀取 3 字符（含点）。
LBPRO_TASK2LABEL = {
    "T3.": "qa",   # Evidence-Grounded QA
    "T4.": "summarize",  # Summarization & Synthesis
    # 其余（T1/T2/T5/T6/T7/T8/T9/T10/T11）均为 other
}

# v2 domain -> 期望标签
V2_DOMAIN2LABEL = {
    "Single-Document QA": "qa",
    "Multi-Document QA": "qa",
    # ICL / Code Repo / Dialogue / Structured Data 均为 other
}


def load_lbpro(limit=None):
    """yield (tag, system, user, long_text, expected)；tag=primary_task 前缀。"""
    with open(LBPRO_PATH, encoding="utf-8") as f:
        data = json.load(f)
    for i, d in enumerate(data):
        if limit and i >= limit:
            break
        task_prefix = d["primary_task"][:3]  # 形如 "T3."
        expected = LBPRO_TASK2LABEL.get(task_prefix, "other")
        # 对齐官方评测管线：无 system，单条 user message（context 与 question 的
        # 拼接由上游完成），分类器只接 question 作为 user_prompt
        yield f"LBPro-{task_prefix.rstrip('.')}", "", d["question_nonthinking"], d["context"], expected


def render_v2_user(d):
    """v2 官方 0shot 模板的 question+choices 渲染（prompts/0shot.txt）。"""
    return (
        f"What is the correct answer to this question: {d['question'].strip()}\n"
        f"Choices:\n(A) {d['choice_A']}\n(B) {d['choice_B']}\n"
        f"(C) {d['choice_C']}\n(D) {d['choice_D']}"
    )


def load_v2(limit=None, with_choices=True):
    """yield (tag, system, user, long_text, expected)；tag=domain 简写。"""
    with open(V2_PATH, encoding="utf-8") as f:
        data = json.load(f)
    for i, d in enumerate(data):
        if limit and i >= limit:
            break
        expected = V2_DOMAIN2LABEL.get(d["domain"], "other")
        user = render_v2_user(d) if with_choices else d["question"]
        tag = "v2-" + {
            "Single-Document QA": "DocQA1",
            "Multi-Document QA": "DocQA2",
            "Long In-context Learning": "ICL",
            "Code Repository Understanding": "Code",
            "Long-dialogue History Understanding": "Dialog",
            "Long Structured Data Understanding": "Struct",
        }[d["domain"]]
        yield tag, "", user, d["context"], expected


# ----------------------------------------------------------------------------
# 评测
# ----------------------------------------------------------------------------

# 机械类违规（规则形态应归零，实现缺陷才能产生）：修复后归零才达标
MECHANICAL_ZERO_TAGS = [
    "LBPro-T5",   # 引用对齐（en_summary_suffix \Z 化 + neg_citation_align）
    "LBPro-T9",   # 代码 diff（zh_call_tool 反引号排除）
    "v2-ICL",     # with-choices 模式下 MC 模板清零
    "v2-Code",
    "v2-Dialog",
    "v2-Struct",
]

# 排序/矛盾/版本类：修复后残留极少，门槛 <=1
NEAR_ZERO_TAGS = ["LBPro-T2", "LBPro-T7"]

# 边界残余（裸问句形态与真 doc-QA 不可分）：单列报告不设门槛
BOUNDARY_TAGS = ["LBPro-T8", "LBPro-T11", "LBPro-T1", "LBPro-T6", "LBPro-T10"]


def evaluate(samples, header):
    """跑一遍评测，返回统计 dict 并打印报告。"""
    print("=" * 70)
    print(header)
    print("=" * 70)
    per_tag = defaultdict(lambda: {"n": 0, "viol": 0, "by_got": Counter(),
                                   "recall_n": 0, "recall_ok": 0, "errors": []})
    for tag, system, user, long_text, expected in samples:
        res = classify_pipeline(user, long_text, system_prompt=system)
        if res is None:
            # v2/LB-Pro context 均超阈值；不触发只可能是 limit 调试下偶发
            continue
        got = res["label"]
        e = per_tag[tag]
        if expected == "other":
            e["n"] += 1
            if got in ("qa", "summarize", "tool_call"):
                e["viol"] += 1
                e["by_got"][got] += 1
                if len(e["errors"]) < 3:
                    e["errors"].append({
                        "user": user[:180], "expected": expected, "got": got,
                        "source": res["source"],
                        "hits": {k: v for k, v in res["hits"].items() if v},
                    })
        else:
            e["recall_n"] += 1
            if got == expected:
                e["recall_ok"] += 1

    # 违规总览（分母 = 非 qa/summ 任务数）
    total_viol = sum(e["viol"] for e in per_tag.values())
    total_other = sum(e["n"] for e in per_tag.values())
    print(f"\n主指标：非 QA/Summ 任务违规 {total_viol}/{total_other}"
          f" = {total_viol / total_other:.3f}")
    print(f"{'任务':<14}{'违规/总数':>12}  {'误判形态':<24}{'QA/Summ recall（低召回口径，仅报告）'}")
    for tag in sorted(per_tag):
        e = per_tag[tag]
        viol_str = f"{e['viol']}/{e['n']}" if e["n"] else "-"
        got_str = ",".join(f"{k}x{v}" for k, v in e["by_got"].most_common()) or "-"
        recall_str = (f"{e['recall_ok']}/{e['recall_n']}" if e["recall_n"] else "-")
        print(f"  {tag:<12}{viol_str:>10}  {got_str:<24}{recall_str}")
        for err in e["errors"]:
            print(f"      ✗ [{err['got']}|{err['source']}] {err['user']!r}")
            print(f"        hits={json.dumps(err['hits'], ensure_ascii=False, default=str)[:240]}")
    return per_tag


def main():
    parser = argparse.ArgumentParser(description="任务规则鲁棒性测试（LB-Pro + v2 混入）")
    parser.add_argument("--limit", type=int, default=None, help="每数据集取样上限（调试用）")
    parser.add_argument("--min-long-chars", type=int, default=2000, help="long_text 触发阈值")
    args = parser.parse_args()

    samples = list(load_lbpro(args.limit)) + list(load_v2(args.limit, with_choices=True))
    per_tag = evaluate(samples, "主模式：LB-Pro（user=question_nonthinking）+ v2（官方模板带选项）")

    # 对照模式（仅报告）：v2 裸 question，量化模板信号承载
    samples_bare = list(load_lbpro(args.limit)) + list(load_v2(args.limit, with_choices=False))
    per_tag_bare = evaluate(samples_bare, "对照模式：v2 裸 question（不渲染选项，仅报告）")

    # ---- 断言（主模式）----
    print("=" * 70)
    print("验收断言（主模式）")
    print("=" * 70)
    failures = []

    def viol_of(tag, table):
        return table[tag]["viol"] if tag in table else 0

    # 机械类违规归零
    for tag in MECHANICAL_ZERO_TAGS:
        v = viol_of(tag, per_tag)
        status = "✓" if v == 0 else f"✗ {v}"
        print(f"  {tag:<12} 机械类违规 = 0        {status}")
        if v > 0:
            failures.append(f"{tag} 机械类违规 {v} 条（应为 0）")
    # 排序/矛盾/版本类 <=1
    for tag in NEAR_ZERO_TAGS:
        v = viol_of(tag, per_tag)
        status = "✓" if v <= 1 else f"✗ {v}"
        print(f"  {tag:<12} 违规 <= 1            {status}")
        if v > 1:
            failures.append(f"{tag} 违规 {v} 条（应 <=1）")
    # 总体违规率：非边界任务（机械类+排序/矛盾/版本）合计为 0；边界任务单独报告
    mechanical_total = sum(viol_of(t, per_tag) for t in MECHANICAL_ZERO_TAGS + NEAR_ZERO_TAGS)
    print(f"  非边界任务合计违规 = 0      {'✓' if mechanical_total == 0 else f'✗ {mechanical_total}'}")
    if mechanical_total > 0:
        failures.append(f"非边界任务合计违规 {mechanical_total} 条（应为 0）")
    # 边界任务（T8 计算问句/T11 裸问句等）：不设门槛，仅报告
    for tag in BOUNDARY_TAGS:
        v = viol_of(tag, per_tag)
        n = per_tag[tag]["n"] if tag in per_tag else 0
        print(f"  {tag:<12} 边界残余（不设门槛）  {v}/{n} = {v / n:.2f}" if n else f"  {tag:<12} 边界残余（不设门槛）  -")

    if failures:
        print("\n未达标：")
        for msg in failures:
            print(f"  ✗ {msg}")
        sys.exit(1)
    print("\n  全部通过 ✓（边界残余单列，见上方报告）")


if __name__ == "__main__":
    main()

