"""统计 BFCL V4 Parallel Multi-Turn 各样本的压缩效果、加速比与函数调用精度，并按 official_category / task_type 分类统计。

baseline 与 compress 分开执行（--mode 选择）：
  - baseline：不压缩，跑完整生成 + 判分，输出 baseline 统计（精度 + 压缩率=1 + 加速比=1）
  - compress_func：reranker 压缩，跑完整生成 + 判分，输出 compress 统计（精度 + 压缩率 + 加速比）
    compress 的输出不含 baseline 行；加速比所需 baseline prefill 由 compress 模式内部补测一次
    （max_tokens=1，仅计时，不写进输出表）。

每条样本输出指标：
  - tools 个数（原始）/ 真值 tools 个数（ground_truth）
  - 原始 prompt 总 token / 压缩后 prompt 总 token（chat template 渲染整条 prompt）
  - 实际 token 压缩率 = 压缩后 prompt token / 原始 prompt token
  - tools 个数压缩率 = 保留 tools 数 / 原始 tools 数
  - prefill 加速比 = baseline_prefill_s / (reranker_s + 压缩后 prefill_s)
    （口径同 benchmark_bfcl.py：只统计 prefill，压缩端到端 = reranker + 压缩后 prefill）
  - 函数调用精度 5 口径：exact / superset / subset / top1 / top3（0/1，调 bfcl_metrics 判分）

分类统计（均值 / 最大值 / 最小值）：
  1. 先按 official_category 分类
  2. 再在每个 official_category 下按 task_type 分类（category × task_type 子类）

用法:
  # baseline（单独跑，输出 baseline 统计）
  python stat_bfcl_compress.py -c ../config/bfcl_parallel_multi_turn.yaml --mode baseline

  # compress（单独跑，输出 compress 统计，不含 baseline 行）
  python stat_bfcl_compress.py -c ../config/bfcl_parallel_multi_turn.yaml --mode compress_func

  # 调试（前 20 条）
  python stat_bfcl_compress.py -c ../config/bfcl_parallel_multi_turn.yaml --mode compress_func --debug
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import yaml
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import SamplingParams

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.append(os.path.dirname(__file__))

from eval_bfcl_parallel_multi_turn import (  # noqa: E402
    DEFAULT_INSTRUCTION,
    _bfcl_to_openai_tools,
    _build_prompt,
    _compress_tools,
    _extract_current_turn,
    build_components,
    load_bfcl_eval,
)
import bfcl_metrics  # noqa: E402


def _tools_token_count(tokenizer, func_list):
    """工具文档 token 数：OpenAI 格式 JSON 序列化后 tokenize（与 benchmark_bfcl.py 同口径）。

    只统计函数文档本身，不含 system/user_prompt 部分，直接反映压缩目标。
    """
    tools_json = "\n".join(
        json.dumps(t, ensure_ascii=False) for t in _bfcl_to_openai_tools(func_list)
    )
    return len(tokenizer.encode(tools_json, add_special_tokens=False))

# baseline 在 rate 列里的标记
BASELINE_TAG = "baseline"

# 5 种 AST 精度口径
AST_FIELDS = ["exact_match", "superset_match", "subset_match", "top1_match", "top3_match"]


def _gt_func_names(sample):
    """从 ground_truth 取真值工具名集合。ground_truth: [{func_name: {params...}}, ...]。"""
    gt = sample.get("ground_truth") or []
    return {list(it.keys())[0] for it in gt if it}


def _judge(func_list, pred_text, ground_truth):
    """调 bfcl_metrics 判分，返回 5 口径 dict（值已转 0/1 int，便于聚合）。"""
    ast = bfcl_metrics.compute_ast_metrics(func_list, pred_text, ground_truth)
    return {f: int(bool(ast[f])) for f in AST_FIELDS}


def _run_batch(yaml_args, mode, rate, rows, tokenizer, ranker, llm,
               max_total_token, max_gen, instruction, warmup=0):
    """跑一个 mode（baseline / compress_func）。

    对每个 batch 做两次 generate：
      1) prefill_sp(max_tokens=1)：测纯 prefill 时间（与 benchmark_bfcl.py 同口径）
      2) gen_sp(max_tokens=max_gen)：拿模型输出文本 → 调 bfcl_metrics 判分拿 5 口径精度

    返回 per-sample 记录 dict，key = sample id。
      baseline: n_tools_kept=n_tools_total, 压缩率=1, reranker_s=0, speedup=1
      compress_func: 记录压缩后 token + reranker 耗时 + 压缩后 prefill + 精度
    """
    prefill_sp = SamplingParams(temperature=0.0, max_tokens=1, top_p=1.0)
    gen_sp = SamplingParams(
        temperature=yaml_args["llm_config"]["sampling"].get("temperature", 0.0),
        max_tokens=max_gen,
        top_p=yaml_args["llm_config"]["sampling"].get("top_p", 1.0),
    )

    records = {}
    batch_size = yaml_args["exp_config"].get("batch_size", 8)
    global_idx = 0

    for i in tqdm(range(0, len(rows), batch_size), desc=f"[{mode} rate={rate}]"):
        batch = rows[i: i + batch_size]
        metas = []
        prompts = []

        # ---- producer：压缩 + 拼 prompt，per-sample 计 reranker 时间 ----
        for sample in batch:
            query = _extract_current_turn(sample["user_prompt"])
            func_list = sample["function"]
            gt_funcs = _gt_func_names(sample)

            if mode == "compress_func" and ranker is not None:
                t0 = time.perf_counter()
                kept_funcs, _ = _compress_tools(
                    ranker, func_list, query, rate, instruction, sample["id"], ""
                )
                reranker_s = time.perf_counter() - t0
            else:
                kept_funcs = func_list
                reranker_s = 0.0

            t0 = time.perf_counter()
            prompt, _ = _build_prompt(
                tokenizer, kept_funcs, sample["user_prompt"], max_total_token, max_gen
            )
            build_s = time.perf_counter() - t0
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            # 工具文档 token（原始全部 + 压缩后保留），与 benchmark_bfcl.py 同口径
            tools_tokens_total = _tools_token_count(tokenizer, func_list)
            tools_tokens_compressed = _tools_token_count(tokenizer, kept_funcs)

            prompts.append(prompt)
            metas.append({
                "global_idx": global_idx,
                "sample": sample,
                "kept_funcs": kept_funcs,
                "n_tools_total": len(func_list),
                "n_tools_kept": len(kept_funcs),
                "n_gt_funcs": len(gt_funcs),
                "reranker_s": reranker_s,
                "build_s": build_s,
                "prompt_tokens": prompt_tokens,
                "tools_tokens_total": tools_tokens_total,
                "tools_tokens_compressed": tools_tokens_compressed,
            })
            global_idx += 1

        # ---- consumer 1：prefill 计时（max_tokens=1，纯 prefill）----
        t0 = time.perf_counter()
        with torch.cuda.device("cuda:0"):
            llm.generate(prompts, prefill_sp)
        prefill_s_per = (time.perf_counter() - t0) / len(batch)

        # ---- consumer 2：完整生成（max_tokens=max_gen），拿 pred 文本判分 ----
        with torch.cuda.device("cuda:0"):
            outs = llm.generate(prompts, gen_sp)
        preds = [o.outputs[0].text for o in outs]

        for j, m in enumerate(metas):
            if m["global_idx"] < warmup:
                continue  # warmup 不计入统计
            sid = m["sample"]["id"]
            # 精度判分（用模型实际可见的工具列表 + ground_truth）
            ast = _judge(m["kept_funcs"], preds[j], m["sample"]["ground_truth"])
            records[sid] = {
                "id": sid,
                "official_category": m["sample"]["official_category"],
                "task_type": m["sample"]["task_type"],
                "n_tools_total": m["n_tools_total"],
                "n_tools_kept": m["n_tools_kept"],
                "n_gt_funcs": m["n_gt_funcs"],
                "prompt_tokens": m["prompt_tokens"],
                "tools_tokens_total": m["tools_tokens_total"],
                "tools_tokens_compressed": m["tools_tokens_compressed"],
                "reranker_s": m["reranker_s"],
                "build_s": m["build_s"],
                "prefill_s": prefill_s_per,
                "pred": preds[j],
                **ast,
            }
    return records


def _per_sample_table_compress(rows, comp_rec, base_prefill_by_id):
    """compress 模式的 per-sample 明细：每条样本 × 每个 rate 一行（不含 baseline 行）。

    压缩率 / 加速比基于同一样本的原始数据与压缩结果配对计算。
    base_prefill_by_id: {id: baseline_prefill_s}，用于算加速比。
    """
    out = []
    for sample in rows:
        sid = sample["id"]
        c = comp_rec.get(sid)
        if c is None:
            continue
        b_prefill = base_prefill_by_id.get(sid)
        # 原始 prompt token：compress 模式下未直接记录，用样本重算
        # 注：compress 模式 records 里 prompt_tokens 是压缩后的；原始需另算
        # 这里原始 prompt token 取自 c 里未存，改由调用方补——见 _per_sample_table_compress_v2
        out.append((sid, c, b_prefill))
    return out


def _per_sample_rows(mode, rows, rec, rate, base_prefill_by_id=None, orig_tokens_by_id=None):
    """构造 per-sample 明细行列表。

    mode=baseline: 每样本一行，只保留 baseline 本身指标（tools 个数/token/精度/prefill 耗时），
                  压缩相关列（压缩后 token、压缩率、加速比、reranker 耗时、压缩后 prefill）留空。
    mode=compress_func: 每样本一行（单 rate），压缩率<1，加速比基于 baseline prefill。
      orig_tokens_by_id: {id: 原始 prompt token}（compress 模式由 baseline 结果或重算提供）
      base_prefill_by_id: {id: baseline prefill_s}（compress 模式算加速比用）
    """
    out = []
    for sample in rows:
        sid = sample["id"]
        r = rec.get(sid)
        if r is None:
            continue

        # baseline 与 compress 共有的指标
        n_total = r["n_tools_total"]
        tools_tok_total = r["tools_tokens_total"]
        orig_tok = r["prompt_tokens"]  # baseline 下即原始 prompt token；compress 下为压缩后，需覆盖
        row = {
            "id": sid,
            "official_category": r["official_category"],
            "task_type": r["task_type"],
            "rate": BASELINE_TAG if mode == "baseline" else rate,
            "n_tools_total": n_total,
            "n_tools_kept": r["n_tools_kept"],
            "n_gt_funcs": r["n_gt_funcs"],
            "tools_tokens_total": tools_tok_total,
            "tools_tokens_compressed": "",
            "tools_token_compress_ratio": "",
            "tools_ratio_in_prompt_total": "",
            "tools_ratio_in_prompt_compressed": "",
            "prompt_tokens_total": "",
            "prompt_tokens_compressed": "",
            "token_compress_ratio": "",
            "tools_compress_ratio": "",
            "exact_match": r["exact_match"],
            "superset_match": r["superset_match"],
            "subset_match": r["subset_match"],
            "top1_match": r["top1_match"],
            "top3_match": r["top3_match"],
            "reranker_s": "",
            "prefill_s_compressed": "",
            "prefill_s_baseline": f"{r['prefill_s']:.4f}",
            "speedup_prefill": "",
        }

        if mode == "baseline":
            # baseline：补原始 prompt token + tools 占比（baseline 视角下的占比，不涉及压缩）
            row["prompt_tokens_total"] = orig_tok
            row["tools_ratio_in_prompt_total"] = (
                f"{tools_tok_total / orig_tok:.4f}" if orig_tok else ""
            )
            # baseline 的 prefill 即 baseline prefill，复填进 compressed 列做完整记录
            row["prefill_s_compressed"] = f"{r['prefill_s']:.4f}"
        else:  # compress_func
            # 原始 prompt token 取自补测的 baseline（未压缩）；压缩后取自 records
            if orig_tokens_by_id:
                orig_tok = orig_tokens_by_id.get(sid, r["prompt_tokens"])
            comp_tok = r["prompt_tokens"]
            tools_tok_comp = r["tools_tokens_compressed"]
            token_cr = (comp_tok / orig_tok) if orig_tok else 0.0
            tools_cr = (r["n_tools_kept"] / n_total) if n_total else 0.0
            tools_tok_cr = (tools_tok_comp / tools_tok_total) if tools_tok_total else 0.0
            reranker_s = r["reranker_s"]
            prefill_comp = r["prefill_s"]
            prefill_base = base_prefill_by_id.get(sid) if base_prefill_by_id else None
            e2e = reranker_s + prefill_comp
            speedup = (prefill_base / e2e) if (prefill_base and e2e) else 0.0

            row["tools_tokens_compressed"] = tools_tok_comp
            row["tools_token_compress_ratio"] = f"{tools_tok_cr:.4f}"
            row["tools_ratio_in_prompt_total"] = (
                f"{tools_tok_total / orig_tok:.4f}" if orig_tok else ""
            )
            row["tools_ratio_in_prompt_compressed"] = (
                f"{tools_tok_comp / comp_tok:.4f}" if comp_tok else ""
            )
            row["prompt_tokens_total"] = orig_tok
            row["prompt_tokens_compressed"] = comp_tok
            row["token_compress_ratio"] = f"{token_cr:.4f}"
            row["tools_compress_ratio"] = f"{tools_cr:.4f}"
            row["reranker_s"] = f"{reranker_s:.4f}"
            row["prefill_s_compressed"] = f"{prefill_comp:.4f}"
            row["prefill_s_baseline"] = f"{prefill_base:.4f}" if prefill_base is not None else ""
            row["speedup_prefill"] = f"{speedup:.4f}" if prefill_base is not None else ""
        out.append(row)
    return out


def _agg(records):
    """对一组 per-sample 记录算均值/最大/最小。

    baseline 模式下压缩相关列为空字符串，stats() 跳过这些字段（返回 None 值），
    _flatten_agg 会把 None 值写为空，CSV 即显示空白。
    """
    if not records:
        return None
    n = len(records)

    def stats(field, fmt="{:.4f}"):
        # 跳过空值字段（baseline 下压缩列留空，不参与统计）
        vals = [float(r[field]) for r in records if r.get(field) not in ("", None)]
        if not vals:
            return {f"{field}_mean": "", f"{field}_max": "", f"{field}_min": ""}
        return {
            f"{field}_mean": fmt.format(sum(vals) / len(vals)),
            f"{field}_max": fmt.format(max(vals)),
            f"{field}_min": fmt.format(min(vals)),
        }

    out = {"num": n}
    out["n_tools_total"] = stats("n_tools_total", "{:.2f}")
    out["n_tools_kept"] = stats("n_tools_kept", "{:.2f}")
    out["n_gt_funcs"] = stats("n_gt_funcs", "{:.2f}")
    out["tools_tokens_total"] = stats("tools_tokens_total", "{:.1f}")
    out["tools_tokens_compressed"] = stats("tools_tokens_compressed", "{:.1f}")
    out["tools_token_compress_ratio"] = stats("tools_token_compress_ratio")
    out["tools_ratio_in_prompt_total"] = stats("tools_ratio_in_prompt_total")
    out["tools_ratio_in_prompt_compressed"] = stats("tools_ratio_in_prompt_compressed")
    out["prompt_tokens_total"] = stats("prompt_tokens_total", "{:.1f}")
    out["prompt_tokens_compressed"] = stats("prompt_tokens_compressed", "{:.1f}")
    out["token_compress_ratio"] = stats("token_compress_ratio")
    out["tools_compress_ratio"] = stats("tools_compress_ratio")
    for f in AST_FIELDS:
        out[f] = stats(f)  # 精度均值 = 准确率
    out["speedup_prefill"] = stats("speedup_prefill")
    out["reranker_s"] = stats("reranker_s")
    out["prefill_s_compressed"] = stats("prefill_s_compressed")
    out["prefill_s_baseline"] = stats("prefill_s_baseline")
    return out


def _flatten_agg(agg):
    """把 _agg 返回的嵌套 dict 摊平为 {列名: 值}。

    stats() 返回的 key 已带字段名前缀，这里直接用 sk，不再拼 key（避免双重前缀）。
    """
    if agg is None:
        return {}
    flat = {"num": agg["num"]}
    for key, sub in agg.items():
        if key == "num" or not isinstance(sub, dict):
            continue
        for sk, sv in sub.items():
            flat[sk] = sv
    return flat


# 分类统计的列顺序（统一表头，含精度 5 口径）
CATEGORY_COLS = [
    "num",
    "n_tools_total_mean", "n_tools_total_max", "n_tools_total_min",
    "n_tools_kept_mean", "n_tools_kept_max", "n_tools_kept_min",
    "n_gt_funcs_mean", "n_gt_funcs_max", "n_gt_funcs_min",
    "tools_tokens_total_mean", "tools_tokens_total_max", "tools_tokens_total_min",
    "tools_tokens_compressed_mean", "tools_tokens_compressed_max", "tools_tokens_compressed_min",
    "tools_token_compress_ratio_mean", "tools_token_compress_ratio_max", "tools_token_compress_ratio_min",
    "tools_ratio_in_prompt_total_mean", "tools_ratio_in_prompt_total_max", "tools_ratio_in_prompt_total_min",
    "tools_ratio_in_prompt_compressed_mean", "tools_ratio_in_prompt_compressed_max", "tools_ratio_in_prompt_compressed_min",
    "prompt_tokens_total_mean", "prompt_tokens_total_max", "prompt_tokens_total_min",
    "prompt_tokens_compressed_mean", "prompt_tokens_compressed_max", "prompt_tokens_compressed_min",
    "token_compress_ratio_mean", "token_compress_ratio_max", "token_compress_ratio_min",
    "tools_compress_ratio_mean", "tools_compress_ratio_max", "tools_compress_ratio_min",
    "exact_match_mean", "exact_match_max", "exact_match_min",
    "superset_match_mean", "superset_match_max", "superset_match_min",
    "subset_match_mean", "subset_match_max", "subset_match_min",
    "top1_match_mean", "top1_match_max", "top1_match_min",
    "top3_match_mean", "top3_match_max", "top3_match_min",
    "speedup_prefill_mean", "speedup_prefill_max", "speedup_prefill_min",
    "reranker_s_mean", "reranker_s_max", "reranker_s_min",
    "prefill_s_compressed_mean", "prefill_s_compressed_max", "prefill_s_compressed_min",
    "prefill_s_baseline_mean", "prefill_s_baseline_max", "prefill_s_baseline_min",
]

PER_SAMPLE_COLS = [
    "id", "official_category", "task_type", "rate",
    "n_tools_total", "n_tools_kept", "n_gt_funcs",
    "tools_tokens_total", "tools_tokens_compressed", "tools_token_compress_ratio",
    "tools_ratio_in_prompt_total", "tools_ratio_in_prompt_compressed",
    "prompt_tokens_total", "prompt_tokens_compressed",
    "token_compress_ratio", "tools_compress_ratio",
    "exact_match", "superset_match", "subset_match", "top1_match", "top3_match",
    "reranker_s", "prefill_s_compressed", "prefill_s_baseline", "speedup_prefill",
]


def _write_csv(path, rows, cols):
    """写 CSV（UTF-8 with BOM，Excel 可直接打开中文）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _all_rates_sorted(per_sample):
    """从 per_sample 发现所有 rate（baseline 排最前，其余数值升序）。"""
    rates = {r["rate"] for r in per_sample}
    def _key(r):
        return (-1, 0) if r == BASELINE_TAG else (0, float(r))
    return sorted(rates, key=_key)


def _category_rows(per_sample):
    """按 official_category 分类统计（每行 = category × rate）。"""
    groups = defaultdict(list)
    for r in per_sample:
        groups[(r["official_category"], r["rate"])].append(r)
    all_rates = _all_rates_sorted(per_sample)
    out = []
    for cat in sorted({r["official_category"] for r in per_sample}):
        for rate in all_rates:
            items = groups.get((cat, rate), [])
            row = {"official_category": cat, "rate": rate}
            row.update(_flatten_agg(_agg(items)))
            out.append(row)
    return out


def _category_tasktype_rows(per_sample):
    """按 official_category × task_type 二级分类统计（每行 = category × task_type × rate）。"""
    groups = defaultdict(list)
    for r in per_sample:
        groups[(r["official_category"], r["task_type"], r["rate"])].append(r)
    all_rates = _all_rates_sorted(per_sample)
    out = []
    for cat in sorted({r["official_category"] for r in per_sample}):
        for tt in sorted({r["task_type"] for r in per_sample if r["official_category"] == cat}):
            for rate in all_rates:
                items = groups.get((cat, tt, rate), [])
                row = {"official_category": cat, "task_type": tt, "rate": rate}
                row.update(_flatten_agg(_agg(items)))
                out.append(row)
    return out


def _overall_rows(per_sample):
    """总体统计（每行 = 一个 rate）。"""
    all_rates = _all_rates_sorted(per_sample)
    out = []
    for rate in all_rates:
        items = [r for r in per_sample if r["rate"] == rate]
        row = {"official_category": "OVERALL", "rate": rate}
        row.update(_flatten_agg(_agg(items)))
        out.append(row)
    return out


def _measure_baseline_prefill(rows, tokenizer, llm, max_total_token, max_gen, warmup=0):
    """compress 模式专用：只测原始 prompt 的 prefill 时间（max_tokens=1），不判分、不输出统计。

    返回 {id: prefill_s}，供 compress 算加速比。同时返回 {id: 原始 prompt token}。
    """
    prefill_sp = SamplingParams(temperature=0.0, max_tokens=1, top_p=1.0)
    prefill_by_id = {}
    tokens_by_id = {}
    batch_size = 8
    global_idx = 0
    for i in tqdm(range(0, len(rows), batch_size), desc="[baseline prefill 测速]"):
        batch = rows[i: i + batch_size]
        prompts = []
        metas = []
        for sample in batch:
            prompt, _ = _build_prompt(
                tokenizer, sample["function"], sample["user_prompt"], max_total_token, max_gen
            )
            prompts.append(prompt)
            metas.append({"id": sample["id"], "idx": global_idx,
                          "tok": len(tokenizer.encode(prompt, add_special_tokens=False))})
            global_idx += 1
        t0 = time.perf_counter()
        with torch.cuda.device("cuda:0"):
            llm.generate(prompts, prefill_sp)
        pf = (time.perf_counter() - t0) / len(batch)
        for m in metas:
            if m["idx"] < warmup:
                continue
            prefill_by_id[m["id"]] = pf
            tokens_by_id[m["id"]] = m["tok"]
    return prefill_by_id, tokens_by_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--mode", default="compress_func",
                    choices=["baseline", "compress_func"],
                    help="baseline=不压缩；compress_func=reranker 压缩（输出不含 baseline 行）")
    ap.add_argument("--debug", action="store_true", help="只用前 20 条")
    ap.add_argument("-n", "--n", type=int, default=0, help="总样本数（含 warmup，0=全部）")
    ap.add_argument("--warmup", type=int, default=2, help="前 N 条 warmup 不计入统计")
    ap.add_argument("--rates", type=str, default="", help="覆盖 rate 列表，逗号分隔，如 10,8,6（仅 compress_func）")
    ap.add_argument("--output", default="../output", help="输出根目录")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg.get("dataset_dir"):
        cfg["dataset_dir"] = os.path.join(os.path.dirname(__file__), "..", "data")

    rows = load_bfcl_eval(cfg.get("dataset_dir"))
    if args.n > 0:
        rows = rows[: args.n]
    elif args.debug:
        rows = rows[:20]
    if args.warmup >= len(rows):
        print(f"[warn] warmup({args.warmup}) >= 样本数({len(rows)})，统计样本为 0")
    stat_n = max(0, len(rows) - args.warmup)
    print(f"模式: {args.mode} | 样本数: {len(rows)}（warmup {args.warmup}，统计 {stat_n} 条）")

    rates = [float(x) for x in args.rates.split(",")] if args.rates else cfg["reranker_config"]["rate"]
    if args.mode == "compress_func":
        print(f"压缩 rate 档位: {rates}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["llm_config"]["llm"]["model_name"], trust_remote_code=True
    )
    max_total_token = cfg["exp_config"].get("max_total_token", 32768)
    max_gen = cfg["llm_config"]["sampling"].get("max_tokens", 512)
    instruction = cfg["reranker_config"].get("instruction") or DEFAULT_INSTRUCTION

    ranker, llm = build_components(cfg)

    # ---- 输出目录 ----
    exp_name = cfg["exp_config"]["name"]
    mode_tag = args.mode
    if args.mode == "compress_func":
        rate_tag = "_rate" + "-".join(str(r) for r in rates)
    else:
        rate_tag = ""
    sub = f"stat_{mode_tag}{rate_tag}_n{args.n if args.n else 'all'}_warmup{args.warmup}"
    if args.debug:
        sub += "_debug"
    report_dir = os.path.join(args.output, exp_name, sub)
    os.makedirs(report_dir, exist_ok=True)

    per_sample = []

    if args.mode == "baseline":
        print("\n===== baseline（不压缩，生成 + 判分 + prefill 计时）=====")
        rec = _run_batch(
            cfg, "baseline", None, rows, tokenizer, ranker, llm,
            max_total_token, max_gen, instruction, warmup=args.warmup,
        )
        per_sample = _per_sample_rows("baseline", rows, rec, None)

        # 存 baseline 结果，供 compress 模式读取算加速比
        base_dump = {
            "prefill_by_id": {sid: r["prefill_s"] for sid, r in rec.items()},
            "tokens_by_id": {sid: r["prompt_tokens"] for sid, r in rec.items()},
        }
        with open(os.path.join(report_dir, "baseline_prefill.json"), "w", encoding="utf-8") as f:
            json.dump(base_dump, f, ensure_ascii=False, indent=2)

    else:  # compress_func
        # compress 算加速比需要 baseline prefill：内部补测一次（仅计时，不输出统计）
        print("\n===== 补测 baseline prefill（仅用于算加速比，不计入输出）=====")
        base_prefill_by_id, orig_tokens_by_id = _measure_baseline_prefill(
            rows, tokenizer, llm, max_total_token, max_gen, warmup=args.warmup
        )

        for rate in rates:
            print(f"\n===== compress_func rate={rate}（压缩 + 生成 + 判分 + prefill 计时）=====")
            rec = _run_batch(
                cfg, "compress_func", rate, rows, tokenizer, ranker, llm,
                max_total_token, max_gen, instruction, warmup=args.warmup,
            )
            per_sample.extend(_per_sample_rows(
                "compress_func", rows, rec, rate,
                base_prefill_by_id=base_prefill_by_id,
                orig_tokens_by_id=orig_tokens_by_id,
            ))

    # ---- 写 CSV ----
    _write_csv(os.path.join(report_dir, "per_sample.csv"), per_sample, PER_SAMPLE_COLS)
    _write_csv(os.path.join(report_dir, "overall.csv"), _overall_rows(per_sample),
               ["official_category", "rate"] + CATEGORY_COLS)
    _write_csv(os.path.join(report_dir, "by_official_category.csv"), _category_rows(per_sample),
               ["official_category", "rate"] + CATEGORY_COLS)
    _write_csv(os.path.join(report_dir, "by_official_category_task_type.csv"),
               _category_tasktype_rows(per_sample),
               ["official_category", "task_type", "rate"] + CATEGORY_COLS)

    # per-sample JSON（含完整记录，便于复用）
    with open(os.path.join(report_dir, "stat.json"), "w", encoding="utf-8") as f:
        json.dump({
            "mode": args.mode, "config": args.config,
            "num_stat_samples": stat_n, "warmup": args.warmup,
            "rates": rates if args.mode == "compress_func" else None,
            "model": cfg["llm_config"]["llm"]["model_name"],
            "per_sample": per_sample,
        }, f, ensure_ascii=False, indent=2)

    # ---- 终端摘要 ----
    all_rates = _all_rates_sorted(per_sample)
    print("\n" + "=" * 60)
    print(f"统计完成 [{args.mode}]，{len(per_sample)} 条 per-sample 记录")
    print(f"输出目录: {report_dir}")
    print(f"  - per_sample.csv / overall.csv / by_official_category.csv / by_official_category_task_type.csv")
    print(f"  - stat.json")
    if args.mode == "baseline":
        print(f"  - baseline_prefill.json（供 compress 模式算加速比）")
    print("=" * 60)

    print(f"\n【总体统计·均值】 mode={args.mode}")
    print(f"{'rate':>10} {'num':>4} {'tools_kpt':>9} {'gt':>5} {'tools_tok':>9} "
          f"{'tok_comp':>8} {'exact':>6} {'super':>6} {'subset':>6} {'top1':>6} {'top3':>6} {'speedup':>8}")
    for rate in all_rates:
        items = [r for r in per_sample if r["rate"] == rate]
        if not items:
            continue
        n = len(items)
        avg = lambda k: sum(float(r[k]) for r in items) / n
        sp = f"{avg('speedup_prefill'):.4f}" if items[0].get("speedup_prefill") not in ("", None) else "-"
        print(f"{str(rate):>10} {n:>4} {avg('n_tools_kept'):>9.2f} {avg('n_gt_funcs'):>5.2f} "
              f"{avg('tools_tokens_compressed'):>9.0f} {avg('token_compress_ratio'):>8.4f} "
              f"{avg('exact_match'):>6.3f} {avg('superset_match'):>6.3f} {avg('subset_match'):>6.3f} "
              f"{avg('top1_match'):>6.3f} {avg('top3_match'):>6.3f} {sp:>8}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
