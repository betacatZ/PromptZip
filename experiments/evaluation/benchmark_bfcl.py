"""BFCL V4 Parallel Multi-Turn 速度测试：对比 baseline vs reranker 压缩工具文档的端到端耗时与加速比。

输出每条样本的：reranker 耗时、llm 耗时、总耗时，并汇总平均 + 加速比。

用法:
  python benchmark_bfcl.py -c ../config/bfcl_parallel_multi_turn.yaml [--debug] [--n N]
"""

import argparse
import copy
import json
import os
import sys
import time

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
    _bfcl_to_openai_tools,
    _compress_tools,
    _extract_current_turn,
    build_components,
    load_bfcl_eval,
)


def _run_mode(yaml_args, mode, rate, rows, tokenizer, ranker, llm, max_total_token, max_gen, instruction, warmup=0):
    """跑一个 mode（baseline/compress_func），返回 per-sample 耗时列表。

    warmup: 前 N 条样本照常跑（预热模型/CUDA/KV cache），但不计入 timings 统计。
    """
    sampling_params = SamplingParams(
        temperature=yaml_args["llm_config"]["sampling"].get("temperature", 0.0),
        max_tokens=max_gen,
        top_p=yaml_args["llm_config"]["sampling"].get("top_p", 1.0),
    )

    timings = []  # 每条: {id, n_tools_total, n_tools_kept, reranker_s, llm_s, total_s, pred_tokens}
    batch_size = yaml_args["exp_config"].get("batch_size", 8)
    global_idx = 0  # 全局样本序号，用于 warmup 判断

    for i in tqdm(range(0, len(rows), batch_size), desc=f"[{mode} rate={rate}]"):
        batch = rows[i : i + batch_size]
        prompts = []
        metas = []
        # ---- producer 阶段：压缩 + 拼 prompt（per-sample 计 reranker 时间）----
        for sample in batch:
            query = _extract_current_turn(sample["user_prompt"])
            func_list = sample["function"]
            if mode == "compress_func" and ranker is not None:
                t0 = time.perf_counter()
                kept_funcs, kept_names = _compress_tools(
                    ranker, func_list, query, rate, instruction, sample["id"], ""
                )
                reranker_s = time.perf_counter() - t0
            else:
                kept_funcs = func_list
                kept_names = {f["name"] for f in func_list}
                reranker_s = 0.0

            t0 = time.perf_counter()
            prompt, _ = _build_prompt(tokenizer, kept_funcs, sample["user_prompt"], max_total_token, max_gen)
            build_s = time.perf_counter() - t0
            prompts.append(prompt)
            metas.append({
                "global_idx": global_idx,
                "id": sample["id"],
                "n_tools_total": len(func_list),
                "n_tools_kept": len(kept_funcs),
                "reranker_s": reranker_s,
                "build_s": build_s,
            })
            global_idx += 1

        # ---- consumer 阶段：llm.generate（per-batch 计时，均摊到每条）----
        t0 = time.perf_counter()
        with torch.cuda.device("cuda:0"):
            outs = llm.generate(prompts, sampling_params)
        llm_s_batch = time.perf_counter() - t0
        llm_s_per = llm_s_batch / len(batch)

        for j, m in enumerate(metas):
            if m["global_idx"] < warmup:
                continue  # warmup 样本不计入统计
            timings.append({
                "id": m["id"],
                "n_tools_total": m["n_tools_total"],
                "n_tools_kept": m["n_tools_kept"],
                "reranker_s": m["reranker_s"],
                "build_s": m["build_s"],
                "llm_s": llm_s_per,
                "total_s": m["reranker_s"] + m["build_s"] + llm_s_per,
                "pred_tokens": len(outs[j].outputs[0].token_ids),
            })
    return timings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--debug", action="store_true", help="只用前 20 条")
    ap.add_argument("-n", "--n", type=int, default=0, help="总样本数（含 warmup，0=全部）")
    ap.add_argument("--warmup", type=int, default=0, help="前 N 条 warmup 不计入统计")
    ap.add_argument("--rates", type=str, default="", help="覆盖配置的 rate 列表，逗号分隔，如 6,8,10")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rows = load_bfcl_eval(cfg.get("dataset_dir"))
    if args.n > 0:
        rows = rows[: args.n]
    elif args.debug:
        rows = rows[:20]
    if args.warmup > len(rows):
        print(f"[warn] warmup({args.warmup}) > 样本数({len(rows)})，统计样本为 0")
    stat_n = max(0, len(rows) - args.warmup)
    print(f"样本数: {len(rows)}（warmup {args.warmup} 条不计入，统计 {stat_n} 条）")

    tokenizer = AutoTokenizer.from_pretrained(cfg["llm_config"]["llm"]["model_name"], trust_remote_code=True)
    max_total_token = cfg["exp_config"].get("max_total_token", 32768)
    max_gen = cfg["llm_config"]["sampling"].get("max_tokens", 1024)
    instruction = cfg["reranker_config"].get("instruction") or DEFAULT_INSTRUCTION

    rates = [float(x) for x in args.rates.split(",")] if args.rates else cfg["reranker_config"]["rate"]

    ranker, llm = build_components(cfg)

    all_results = {}
    # baseline
    print("\n===== baseline =====")
    t_base = _run_mode(cfg, "baseline", None, rows, tokenizer, ranker, llm, max_total_token, max_gen, instruction, warmup=args.warmup)
    all_results["baseline"] = t_base

    # compress_func 各 rate
    for rate in rates:
        print(f"\n===== compress_func rate={rate} =====")
        t_comp = _run_mode(cfg, "compress_func", rate, rows, tokenizer, ranker, llm, max_total_token, max_gen, instruction, warmup=args.warmup)
        all_results[f"compress_func_{rate}"] = t_comp

    # ---- 汇总 ----
    print("\n" + "=" * 80)
    print("速度汇总（每条平均耗时，秒）")
    print("=" * 80)
    print(f"{'mode':<22} {'reranker':>10} {'llm':>10} {'build':>10} {'total':>10} {'pred_tok':>10} {'n_tools_kept':>14}")
    base_total = None
    summary = {}
    for mode, ts in all_results.items():
        n = len(ts)
        reranker_avg = sum(t["reranker_s"] for t in ts) / n
        llm_avg = sum(t["llm_s"] for t in ts) / n
        build_avg = sum(t["build_s"] for t in ts) / n
        total_avg = sum(t["total_s"] for t in ts) / n
        pred_tok_avg = sum(t["pred_tokens"] for t in ts) / n
        n_tools_avg = sum(t["n_tools_kept"] for t in ts) / n
        if mode == "baseline":
            base_total = total_avg
        speedup = base_total / total_avg if base_total else 0
        print(f"{mode:<22} {reranker_avg:>10.4f} {llm_avg:>10.4f} {build_avg:>10.4f} {total_avg:>10.4f} {pred_tok_avg:>10.1f} {n_tools_avg:>14.1f}")
        summary[mode] = {
            "reranker_s": reranker_avg, "llm_s": llm_avg, "build_s": build_avg,
            "total_s": total_avg, "pred_tokens": pred_tok_avg, "n_tools_kept": n_tools_avg,
            "speedup_vs_baseline": speedup,
        }

    print("\n" + "=" * 80)
    print("加速比（vs baseline，>1 表示更快）")
    print("=" * 80)
    for mode, s in summary.items():
        if mode == "baseline":
            continue
        print(f"{mode}: total={s['total_s']:.4f}s  speedup={s['speedup_vs_baseline']:.2f}x  "
              f"(n_tools {s['n_tools_kept']:.1f} vs baseline全量)")

    # 写 JSON
    out_path = os.path.join(os.path.dirname(cfg.get("_result_path") or "."), "benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_sample": all_results}, f, ensure_ascii=False, indent=2)
    print(f"\n详细数据写入: {out_path}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
