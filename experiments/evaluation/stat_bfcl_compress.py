"""统计 BFCL V4 Parallel Multi-Turn 各样本的压缩效果与加速比，并按 official_category / task_type 分类统计。

输出指标（每条样本）：
  - tools 个数（原始）
  - 真值 tools 个数（ground_truth）
  - 原始 prompt 总 token 数（chat template 渲染整条 prompt：system+tools+user）
  - 压缩后 prompt 总 token 数
  - 实际 token 压缩率 = 压缩后 prompt token / 原始 prompt token
  - tools 个数压缩率 = 保留 tools 数 / 原始 tools 数
  - prefill 加速比 = baseline_prefill_s / (reranker_s + 压缩后 prefill_s)
    （口径与 benchmark_bfcl.py 一致：只统计 prefill，压缩端到端 = reranker 耗时 + 压缩后 prefill 耗时）

分类统计（均值 / 最大值 / 最小值）：
  1. 先按 official_category 分类
  2. 再在每个 official_category 下按 task_type 分类（category × task_type 子类）

用法:
  python stat_bfcl_compress.py -c ../config/bfcl_parallel_multi_turn.yaml [--debug] [--n N] [--warmup W]
  python stat_bfcl_compress.py -c ../config/bfcl_parallel_multi_turn.yaml --rates 10,8,6 --output ../output
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
    _build_prompt,
    _compress_tools,
    _extract_current_turn,
    build_components,
    load_bfcl_eval,
)

# baseline 在 rate 列里的标记（字符串，便于和数值 rate 区分；baseline 行始终排在所有压缩档位之前）
BASELINE_TAG = "baseline"


def _gt_func_names(sample):
    """从 ground_truth 取出真值工具名集合。ground_truth 结构: [{func_name: {params...}}, ...]。"""
    gt = sample.get("ground_truth") or []
    return {list(it.keys())[0] for it in gt if it}


def _run_batch(yaml_args, mode, rate, rows, tokenizer, ranker, llm,
               max_total_token, max_gen, instruction, warmup=0):
    """跑一个 mode（baseline / compress_func）。

    返回 per-sample 记录 dict，key = sample id。
    baseline: 不压缩，记录原始 prompt token + baseline prefill 耗时。
    compress_func: reranker 压缩，记录压缩后 prompt token + reranker 耗时 + 压缩后 prefill 耗时。
    warmup: 前 warmup 条样本照常跑（预热），但不计入返回结果。
    """
    # 测纯 prefill：max_tokens=1，decode 几乎为 0，wall-clock ≈ prefill（与 benchmark_bfcl.py 同口径）
    prefill_sp = SamplingParams(temperature=0.0, max_tokens=1, top_p=1.0)

    records = {}
    batch_size = yaml_args["exp_config"].get("batch_size", 8)
    global_idx = 0

    for i in tqdm(range(0, len(rows), batch_size), desc=f"[{mode} rate={rate}]"):
        batch = rows[i: i + batch_size]
        prompts = []
        metas = []

        # ---- producer：压缩 + 拼 prompt，per-sample 计 reranker 时间 ----
        for sample in batch:
            query = _extract_current_turn(sample["user_prompt"])
            func_list = sample["function"]
            gt_funcs = _gt_func_names(sample)

            if mode == "compress_func" and ranker is not None:
                t0 = time.perf_counter()
                kept_funcs, kept_names = _compress_tools(
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

            prompts.append(prompt)
            metas.append({
                "global_idx": global_idx,
                "id": sample["id"],
                "official_category": sample["official_category"],
                "task_type": sample["task_type"],
                "n_tools_total": len(func_list),
                "n_tools_kept": len(kept_funcs),
                "n_gt_funcs": len(gt_funcs),
                "reranker_s": reranker_s,
                "build_s": build_s,
                "prompt_tokens": prompt_tokens,
            })
            global_idx += 1

        # ---- consumer：llm.generate（max_tokens=1，时间 ≈ prefill）----
        t0 = time.perf_counter()
        with torch.cuda.device("cuda:0"):
            llm.generate(prompts, prefill_sp)
        prefill_s_per = (time.perf_counter() - t0) / len(batch)

        for j, m in enumerate(metas):
            if m["global_idx"] < warmup:
                continue  # warmup 不计入统计
            records[m["id"]] = {
                "id": m["id"],
                "official_category": m["official_category"],
                "task_type": m["task_type"],
                "n_tools_total": m["n_tools_total"],
                "n_tools_kept": m["n_tools_kept"],
                "n_gt_funcs": m["n_gt_funcs"],
                "prompt_tokens": m["prompt_tokens"],
                "reranker_s": m["reranker_s"],
                "build_s": m["build_s"],
                "prefill_s": prefill_s_per,
            }
    return records


def _build_per_sample_table(rows, base_rec, comp_rec_by_rate):
    """构造 per-sample 明细表：每条样本 × 每个 rate 一行，外加每条样本一行 baseline。

    压缩率 / 加速比都基于同一样本的 baseline 与压缩结果配对计算。
    baseline 行作为对照基准：压缩率=1.0、加速比=1.0、reranker 耗时=0、
    压缩后 token=原始 token、压缩后 prefill=baseline prefill。
    """
    out = []
    for sample in rows:
        sid = sample["id"]
        b = base_rec.get(sid)
        if b is None:
            continue  # warmup 跳过的样本无 baseline

        # baseline 行（对照基准）
        out.append({
            "id": sid,
            "official_category": b["official_category"],
            "task_type": b["task_type"],
            "rate": BASELINE_TAG,
            "n_tools_total": b["n_tools_total"],
            "n_tools_kept": b["n_tools_total"],          # baseline 不压缩，保留全部
            "n_gt_funcs": b["n_gt_funcs"],
            "prompt_tokens_total": b["prompt_tokens"],      # 原始 prompt 总 token
            "prompt_tokens_compressed": b["prompt_tokens"],  # baseline = 原始
            "token_compress_ratio": "1.0000",                # baseline 压缩率 = 1
            "tools_compress_ratio": "1.0000",                # baseline tools 压缩率 = 1
            "reranker_s": "0.0000",                          # baseline 无 reranker
            "prefill_s_compressed": f"{b['prefill_s']:.4f}",  # baseline prefill
            "prefill_s_baseline": f"{b['prefill_s']:.4f}",
            "speedup_prefill": "1.0000",                    # baseline 加速比 = 1
        })

        # 各压缩 rate 行
        for rate, comp_map in comp_rec_by_rate.items():
            c = comp_map.get(sid)
            if c is None:
                continue
            # token 压缩率 = 压缩后 prompt token / 原始 prompt token
            token_cr = (c["prompt_tokens"] / b["prompt_tokens"]) if b["prompt_tokens"] else 0.0
            # tools 个数压缩率 = 保留 tools 数 / 原始 tools 数
            tools_cr = (c["n_tools_kept"] / b["n_tools_total"]) if b["n_tools_total"] else 0.0
            # prefill 加速比 = baseline_prefill / (reranker + 压缩后 prefill)
            e2e = c["reranker_s"] + c["prefill_s"]
            speedup = (b["prefill_s"] / e2e) if e2e else 0.0
            out.append({
                "id": sid,
                "official_category": b["official_category"],
                "task_type": b["task_type"],
                "rate": rate,
                "n_tools_total": b["n_tools_total"],
                "n_tools_kept": c["n_tools_kept"],
                "n_gt_funcs": b["n_gt_funcs"],
                "prompt_tokens_total": b["prompt_tokens"],      # 原始 prompt 总 token
                "prompt_tokens_compressed": c["prompt_tokens"],  # 压缩后 prompt token
                "token_compress_ratio": f"{token_cr:.4f}",       # token 压缩率
                "tools_compress_ratio": f"{tools_cr:.4f}",        # tools 个数压缩率
                "reranker_s": f"{c['reranker_s']:.4f}",
                "prefill_s_compressed": f"{c['prefill_s']:.4f}",
                "prefill_s_baseline": f"{b['prefill_s']:.4f}",
                "speedup_prefill": f"{speedup:.4f}",             # prefill 加速比
            })
    return out


def _agg(records):
    """对一组 per-sample 记录算均值/最大/最小。返回 dict（值已四舍五入为字符串，便于直写 CSV）。"""
    if not records:
        return None
    n = len(records)

    def stats(field, fmt="{:.4f}"):
        vals = [float(r[field]) for r in records]
        return {
            f"{field}_mean": fmt.format(sum(vals) / n),
            f"{field}_max": fmt.format(max(vals)),
            f"{field}_min": fmt.format(min(vals)),
        }

    out = {"num": n}
    out["n_tools_total"] = stats("n_tools_total", "{:.2f}")
    out["n_tools_kept"] = stats("n_tools_kept", "{:.2f}")
    out["n_gt_funcs"] = stats("n_gt_funcs", "{:.2f}")
    out["prompt_tokens_total"] = stats("prompt_tokens_total", "{:.1f}")
    out["prompt_tokens_compressed"] = stats("prompt_tokens_compressed", "{:.1f}")
    out["token_compress_ratio"] = stats("token_compress_ratio")
    out["tools_compress_ratio"] = stats("tools_compress_ratio")
    out["speedup_prefill"] = stats("speedup_prefill")
    out["reranker_s"] = stats("reranker_s")
    out["prefill_s_compressed"] = stats("prefill_s_compressed")
    out["prefill_s_baseline"] = stats("prefill_s_baseline")
    return out


def _flatten_agg(agg, prefix=""):
    """把 _agg 返回的嵌套 dict 摊平为 {列名: 值}，便于写 CSV 一行。

    注意：stats() 返回的 key 已带字段名前缀（如 n_tools_total_mean），
    这里直接用 sk 即可，不要再拼 key（否则变 n_tools_total_n_tools_total_mean 双重前缀）。
    """
    if agg is None:
        return {}
    flat = {"num": agg["num"]}
    for key, sub in agg.items():
        if key == "num" or not isinstance(sub, dict):
            continue
        for sk, sv in sub.items():
            flat[f"{prefix}{sk}"] = sv
    return flat


# 分类统计的列顺序（统一表头）
CATEGORY_COLS = [
    "num",
    "n_tools_total_mean", "n_tools_total_max", "n_tools_total_min",
    "n_tools_kept_mean", "n_tools_kept_max", "n_tools_kept_min",
    "n_gt_funcs_mean", "n_gt_funcs_max", "n_gt_funcs_min",
    "prompt_tokens_total_mean", "prompt_tokens_total_max", "prompt_tokens_total_min",
    "prompt_tokens_compressed_mean", "prompt_tokens_compressed_max", "prompt_tokens_compressed_min",
    "token_compress_ratio_mean", "token_compress_ratio_max", "token_compress_ratio_min",
    "tools_compress_ratio_mean", "tools_compress_ratio_max", "tools_compress_ratio_min",
    "speedup_prefill_mean", "speedup_prefill_max", "speedup_prefill_min",
    "reranker_s_mean", "reranker_s_max", "reranker_s_min",
    "prefill_s_compressed_mean", "prefill_s_compressed_max", "prefill_s_compressed_min",
    "prefill_s_baseline_mean", "prefill_s_baseline_max", "prefill_s_baseline_min",
]

PER_SAMPLE_COLS = [
    "id", "official_category", "task_type", "rate",
    "n_tools_total", "n_tools_kept", "n_gt_funcs",
    "prompt_tokens_total", "prompt_tokens_compressed",
    "token_compress_ratio", "tools_compress_ratio",
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
    """从 per_sample 里动态发现所有 rate（含 baseline），排序返回。

    baseline 始终排最前，其余按数值升序。rate 可能是字符串(baseline)或数值/字符串数字。
    """
    rates = {r["rate"] for r in per_sample}
    def _key(r):
        return (-1, 0) if r == BASELINE_TAG else (0, float(r))
    return sorted(rates, key=_key)


def _category_rows(per_sample):
    """按 official_category 分类统计（所有 rate 合在一张表，每行 = category × rate）。

    rate 含 baseline 与各压缩档位，baseline 排最前。
    每个指标输出均值/最大/最小。
    """
    # 按 (official_category, rate) 分组
    groups = defaultdict(list)
    for r in per_sample:
        groups[(r["official_category"], r["rate"])].append(r)

    all_rates = _all_rates_sorted(per_sample)
    out = []
    for cat in sorted({r["official_category"] for r in per_sample}):
        for rate in all_rates:
            items = groups.get((cat, rate), [])
            agg = _agg(items)
            row = {"official_category": cat, "rate": rate}
            row.update(_flatten_agg(agg))
            out.append(row)
    return out


def _category_tasktype_rows(per_sample):
    """按 official_category × task_type 二级分类统计（每行 = category × task_type × rate）。

    先按 official_category 分，再在每个 category 下按 task_type 分。
    rate 含 baseline 与各压缩档位，baseline 排最前。
    每个指标输出均值/最大/最小。
    """
    groups = defaultdict(list)
    for r in per_sample:
        groups[(r["official_category"], r["task_type"], r["rate"])].append(r)

    all_rates = _all_rates_sorted(per_sample)
    out = []
    for cat in sorted({r["official_category"] for r in per_sample}):
        # 该 category 下的所有 task_type
        for tt in sorted({r["task_type"] for r in per_sample if r["official_category"] == cat}):
            for rate in all_rates:
                items = groups.get((cat, tt, rate), [])
                agg = _agg(items)
                row = {"official_category": cat, "task_type": tt, "rate": rate}
                row.update(_flatten_agg(agg))
                out.append(row)
    return out


def _overall_rows(per_sample):
    """总体统计（每行 = 一个 rate 含 baseline，跨所有 category/task_type）。"""
    all_rates = _all_rates_sorted(per_sample)
    out = []
    for rate in all_rates:
        items = [r for r in per_sample if r["rate"] == rate]
        agg = _agg(items)
        row = {"official_category": "OVERALL", "rate": rate}
        row.update(_flatten_agg(agg))
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--debug", action="store_true", help="只用前 20 条（快速验证）")
    ap.add_argument("-n", "--n", type=int, default=0, help="总样本数（含 warmup，0=全部）")
    ap.add_argument("--warmup", type=int, default=2, help="前 N 条 warmup 不计入统计（默认 2）")
    ap.add_argument("--rates", type=str, default="", help="覆盖配置的 rate 列表，逗号分隔，如 10,8,6")
    ap.add_argument("--output", default="../output", help="输出根目录")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # dataset_dir 默认指向本地 eval.jsonl
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
    print(f"样本数: {len(rows)}（warmup {args.warmup} 条不计入，统计 {stat_n} 条）")

    rates = [float(x) for x in args.rates.split(",")] if args.rates else cfg["reranker_config"]["rate"]
    print(f"压缩 rate 档位: {rates}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["llm_config"]["llm"]["model_name"], trust_remote_code=True
    )
    max_total_token = cfg["exp_config"].get("max_total_token", 32768)
    max_gen = cfg["llm_config"]["sampling"].get("max_tokens", 512)
    instruction = cfg["reranker_config"].get("instruction") or DEFAULT_INSTRUCTION

    ranker, llm = build_components(cfg)

    # ---- baseline：原始 prompt 的 token + prefill 耗时 ----
    print("\n===== baseline（原始，不压缩）=====")
    base_rec = _run_batch(
        cfg, "baseline", None, rows, tokenizer, ranker, llm,
        max_total_token, max_gen, instruction, warmup=args.warmup,
    )

    # ---- compress_func：每个 rate 跑压缩 ----
    comp_rec_by_rate = {}
    for rate in rates:
        print(f"\n===== compress_func rate={rate} =====")
        comp_rec_by_rate[rate] = _run_batch(
            cfg, "compress_func", rate, rows, tokenizer, ranker, llm,
            max_total_token, max_gen, instruction, warmup=args.warmup,
        )

    # ---- 构造 per-sample 明细 ----
    per_sample = _build_per_sample_table(rows, base_rec, comp_rec_by_rate)

    # ---- 输出目录 ----
    exp_name = cfg["exp_config"]["name"]
    rate_tag = "rate" + "-".join(str(r) for r in rates)
    sub = f"stat_n{args.n if args.n else 'all'}_warmup{args.warmup}_{rate_tag}"
    if args.debug:
        sub += "_debug"
    report_dir = os.path.join(args.output, exp_name, sub)
    os.makedirs(report_dir, exist_ok=True)

    # ---- 写 CSV ----
    # 1) per-sample 明细
    per_sample_path = os.path.join(report_dir, "per_sample.csv")
    _write_csv(per_sample_path, per_sample, PER_SAMPLE_COLS)

    # 2) 总体统计（每行一个 rate，含 baseline）
    overall_path = os.path.join(report_dir, "overall.csv")
    _write_csv(overall_path, _overall_rows(per_sample),
               ["official_category", "rate"] + CATEGORY_COLS)

    # 3) 按 official_category 分类统计（含 baseline）
    by_cat_path = os.path.join(report_dir, "by_official_category.csv")
    _write_csv(by_cat_path, _category_rows(per_sample),
               ["official_category", "rate"] + CATEGORY_COLS)

    # 4) 按 official_category × task_type 二级分类统计（含 baseline）
    by_cat_tt_path = os.path.join(report_dir, "by_official_category_task_type.csv")
    _write_csv(by_cat_tt_path, _category_tasktype_rows(per_sample),
               ["official_category", "task_type", "rate"] + CATEGORY_COLS)

    # ---- 同时输出 per-sample JSON（含完整 baseline/压缩记录，便于后续复用）----
    json_path = os.path.join(report_dir, "stat.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": args.config,
            "num_stat_samples": stat_n,
            "warmup": args.warmup,
            "rates": rates,
            "model": cfg["llm_config"]["llm"]["model_name"],
            "per_sample": per_sample,
            "baseline_records": base_rec,
            "compress_records": {str(k): v for k, v in comp_rec_by_rate.items()},
        }, f, ensure_ascii=False, indent=2)

    all_rates = _all_rates_sorted(per_sample)
    print("\n" + "=" * 60)
    print(f"统计完成，共 {len(per_sample)} 条 per-sample 记录"
          f"（{stat_n} 样本 × {len(all_rates)} 档[baseline+{len(rates)}压缩]）")
    print(f"输出目录: {report_dir}")
    print(f"  - per_sample.csv              (逐条样本明细，含 baseline 行)")
    print(f"  - overall.csv                  (总体统计，含 baseline)")
    print(f"  - by_official_category.csv     (按 official_category 分类统计，含 baseline)")
    print(f"  - by_official_category_task_type.csv (category × task_type 二级分类，含 baseline)")
    print(f"  - stat.json                    (完整数据，便于后续复用)")
    print("=" * 60)

    # 终端打印总体摘要（含 baseline 对照）
    print("\n【总体统计·均值】")
    print(f"{'rate':>10} {'num':>4} {'tools_orig':>10} {'tools_kept':>10} {'gt':>5} "
          f"{'prompt_tok':>10} {'comp_tok':>9} {'tok_cr':>7} {'tool_cr':>8} {'speedup':>8}")
    for rate in all_rates:
        items = [r for r in per_sample if r["rate"] == rate]
        if not items:
            continue
        n = len(items)
        avg = lambda k: sum(float(r[k]) for r in items) / n
        print(f"{str(rate):>10} {n:>4} {avg('n_tools_total'):>10.2f} {avg('n_tools_kept'):>10.2f} "
              f"{avg('n_gt_funcs'):>5.2f} {avg('prompt_tokens_total'):>10.0f} "
              f"{avg('prompt_tokens_compressed'):>9.0f} {avg('token_compress_ratio'):>7.4f} "
              f"{avg('tools_compress_ratio'):>8.4f} {avg('speedup_prefill'):>8.4f}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
