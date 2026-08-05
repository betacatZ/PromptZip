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
            # ground_truth 需要的工具集
            gt = sample.get("ground_truth") or []
            gt_funcs = {list(it.keys())[0] for it in gt} if gt else set()
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
            # reranker 召回率：gt 工具里被保留的比例（baseline 全保留 = 1.0）
            recall = (len(kept_names & gt_funcs) / len(gt_funcs)) if gt_funcs else 1.0
            # 精确率：保留的工具里有多少是 gt 需要的（衡量是否留了无关工具）
            precision = (len(kept_names & gt_funcs) / len(kept_names)) if kept_names else 0.0

            t0 = time.perf_counter()
            prompt, _ = _build_prompt(tokenizer, kept_funcs, sample["user_prompt"], max_total_token, max_gen)
            build_s = time.perf_counter() - t0
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            # 拆分统计：工具文档 token（压缩前后对比）+ user_prompt token（不参与压缩，应稳定）
            # 工具文档：用 chat template 渲染时的同款 JSON（OpenAI 格式）序列化后 tokenize
            kept_tools_json = "\n".join(
                json.dumps(t, ensure_ascii=False) for t in _bfcl_to_openai_tools(kept_funcs)
            )
            all_tools_json = "\n".join(
                json.dumps(t, ensure_ascii=False) for t in _bfcl_to_openai_tools(func_list)
            )
            tools_tokens = len(tokenizer.encode(kept_tools_json, add_special_tokens=False))
            all_tools_tokens = len(tokenizer.encode(all_tools_json, add_special_tokens=False))
            user_prompt_tokens = len(tokenizer.encode(sample["user_prompt"], add_special_tokens=False))
            prompts.append(prompt)
            metas.append({
                "global_idx": global_idx,
                "id": sample["id"],
                "official_category": sample["official_category"],
                "n_tools_total": len(func_list),
                "n_tools_kept": len(kept_funcs),
                "n_gt_funcs": len(gt_funcs),
                "recall": recall,
                "precision": precision,
                "tools_tokens": tools_tokens,
                "all_tools_tokens": all_tools_tokens,
                "user_prompt_tokens": user_prompt_tokens,
                "reranker_s": reranker_s,
                "build_s": build_s,
                "prompt_tokens": prompt_tokens,
            })
            global_idx += 1

        # ---- consumer 阶段：llm.generate ----
        t0 = time.perf_counter()
        with torch.cuda.device("cuda:0"):
            outs = llm.generate(prompts, sampling_params)
        llm_s_batch = time.perf_counter() - t0
        llm_s_per = llm_s_batch / len(batch)

        for j, m in enumerate(metas):
            if m["global_idx"] < warmup:
                continue  # warmup 样本不计入统计
            # 从 vLLM metrics 取 prefill 时间（TTFT = first_token_time - arrival_time）
            metrics = getattr(outs[j], "metrics", None)
            prefill_s = None
            if metrics is not None:
                arrival = getattr(metrics, "arrival_time", None)
                first_tok = getattr(metrics, "first_token_time", None)
                if arrival is not None and first_tok is not None:
                    prefill_s = first_tok - arrival
            timings.append({
                "id": m["id"],
                "official_category": m["official_category"],
                "n_tools_total": m["n_tools_total"],
                "n_tools_kept": m["n_tools_kept"],
                "n_gt_funcs": m["n_gt_funcs"],
                "recall": m["recall"],
                "precision": m["precision"],
                "reranker_s": m["reranker_s"],
                "build_s": m["build_s"],
                "llm_s": llm_s_per,
                "prefill_s": prefill_s,
                "total_s": m["reranker_s"] + m["build_s"] + llm_s_per,
                "prompt_tokens": m["prompt_tokens"],
                "tools_tokens": m["tools_tokens"],
                "all_tools_tokens": m["all_tools_tokens"],
                "user_prompt_tokens": m["user_prompt_tokens"],
                "pred_tokens": len(outs[j].outputs[0].token_ids),
            })
    return timings


class Tee:
    """stdout 双写：同时输出到终端和文件，保留 tqdm 进度条只显示终端。"""

    def __init__(self, *files):
        self.files = files

    def write(self, s):
        # tqdm 写 \r 开头的行不进文件（进度条刷新），只写终端
        if s.startswith("\r"):
            for f in self.files:
                if hasattr(f, "isatty") and f.isatty():
                    f.write(s)
        else:
            for f in self.files:
                f.write(s)
                if hasattr(f, "flush"):
                    f.flush()

    def flush(self):
        for f in self.files:
            if hasattr(f, "flush"):
                f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--debug", action="store_true", help="只用前 20 条")
    ap.add_argument("-n", "--n", type=int, default=0, help="总样本数（含 warmup，0=全部）")
    ap.add_argument("--warmup", type=int, default=0, help="前 N 条 warmup 不计入统计")
    ap.add_argument("--rates", type=str, default="", help="覆盖配置的 rate 列表，逗号分隔，如 6,8,10")
    ap.add_argument("--output", default="../output", help="输出根目录")
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

    # 先算 rates（决定输出目录名）
    rates = [float(x) for x in args.rates.split(",")] if args.rates else cfg["reranker_config"]["rate"]

    # 报告双写：终端 + markdown 文件。输出到 output/<exp_name>/benchmark_n<N>_warmup<W>_rate<R>/
    exp_name = cfg["exp_config"]["name"]
    rate_tag = "rate" + "-".join(str(r) for r in rates)
    sub = f"benchmark_n{args.n if args.n else 'all'}_warmup{args.warmup}_{rate_tag}"
    if args.debug:
        sub += "_debug"
    report_dir = os.path.join(args.output, exp_name, sub)
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "benchmark_report.md")
    report_file = open(report_path, "w", encoding="utf-8")
    orig_stdout = sys.stdout
    sys.stdout = Tee(orig_stdout, report_file)

    print(f"# BFCL V4 Parallel Multi-Turn Benchmark 报告\n")
    print(f"- 配置: {args.config}")
    print(f"- 样本数: {len(rows)}（warmup {args.warmup} 条不计入，统计 {stat_n} 条）")
    print(f"- 压缩率: {rates}")
    print(f"- 模型: {cfg['llm_config']['llm']['model_name']}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["llm_config"]["llm"]["model_name"], trust_remote_code=True)
    max_total_token = cfg["exp_config"].get("max_total_token", 32768)
    max_gen = cfg["llm_config"]["sampling"].get("max_tokens", 1024)
    instruction = cfg["reranker_config"].get("instruction") or DEFAULT_INSTRUCTION

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
    def _safe_avg(vals):
        """对可能含 None 的列表求平均（跳过 None）。全 None 返回 None。"""
        xs = [v for v in vals if v is not None]
        return sum(xs) / len(xs) if xs else None

    def _agg(ts):
        """算一组 timings 的平均统计。"""
        n = len(ts)
        if n == 0:
            return None
        return {
            "n": n,
            "reranker_s": sum(t["reranker_s"] for t in ts) / n,
            "llm_s": sum(t["llm_s"] for t in ts) / n,
            # prefill 可能 None（metrics 没拿到时），只算有值的
            "prefill_s": _safe_avg([t.get("prefill_s") for t in ts]),
            "build_s": sum(t["build_s"] for t in ts) / n,
            "total_s": sum(t["total_s"] for t in ts) / n,
            "prompt_tokens": sum(t["prompt_tokens"] for t in ts) / n,
            "tools_tokens": sum(t["tools_tokens"] for t in ts) / n,
            "all_tools_tokens": sum(t["all_tools_tokens"] for t in ts) / n,
            "user_prompt_tokens": sum(t["user_prompt_tokens"] for t in ts) / n,
            "pred_tokens": sum(t["pred_tokens"] for t in ts) / n,
            "n_tools_kept": sum(t["n_tools_kept"] for t in ts) / n,
            "n_gt_funcs": sum(t["n_gt_funcs"] for t in ts) / n,
            "recall": sum(t["recall"] for t in ts) / n,
            "precision": sum(t["precision"] for t in ts) / n,
        }

    def _print_table(title, groups, base_by_cat):
        """打印两张表：主表（recall/precision/速度）+ token 拆分表。"""
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
        # 主表：召回/精度 + 速度。重点看 prefill（不含 decode，纯 prompt 处理时间）。
        # speedup 用 prefill 算（baseline 自身显示 -）
        print(f"{'group':<26} {'n':>4} {'recall':>7} {'precision':>9} {'rerank':>8} {'prefill':>9} {'speedup':>8}")
        # 取本表 baseline 的 prefill 作参照
        base_prefill = None
        if "baseline" in groups and groups["baseline"]:
            base_prefill = groups["baseline"]["prefill_s"]
        elif base_by_cat:
            for a in base_by_cat.values():
                if a and a["prefill_s"]:
                    base_prefill = a["prefill_s"]
                    break
        for label, a in groups.items():
            if a is None:
                print(f"{label:<26} {'-':>4} (无样本)")
                continue
            pf = a["prefill_s"]
            pf_str = f"{pf:>9.4f}" if pf is not None else f"{'N/A':>9}"
            if label == "baseline":
                sp_str = "    -"
            else:
                sp = (base_prefill / pf) if base_prefill and pf else 0
                sp_str = f"{sp:>7.2f}x"
            print(f"{label:<26} {a['n']:>4} {a['recall']:>7.3f} {a['precision']:>9.3f} {a['reranker_s']:>8.4f} {pf_str} {sp_str:>8}")
        # token 拆分表
        print("\n  token 拆分（平均）:")
        print(f"  {'group':<26} {'tools_kept':>11} {'tools_tok':>11} {'all_tools_tok':>15} {'user_prompt_tok':>16} {'prompt_tok':>11}")
        for label, a in groups.items():
            if a is None:
                continue
            print(f"  {label:<26} {a['n_tools_kept']:>11.1f} {a['tools_tokens']:>11.0f} {a['all_tools_tokens']:>15.0f} {a['user_prompt_tokens']:>16.0f} {a['prompt_tokens']:>11.0f}")

    # 总体汇总（所有类别合计）
    overall = {m: _agg(ts) for m, ts in all_results.items()}
    _print_table("Reranker 召回/精度 + 速度·总体（每条平均）", overall, None)
    # 总体压缩比
    base_overall = overall.get("baseline")
    if base_overall:
        print("\n压缩比·总体（vs baseline，speedup 基于 prefill）:")
        for m, a in overall.items():
            if m == "baseline" or a is None:
                continue
            tool_cr = base_overall["all_tools_tokens"] / a["tools_tokens"] if a["tools_tokens"] else 0
            prompt_cr = base_overall["prompt_tokens"] / a["prompt_tokens"] if a["prompt_tokens"] else 0
            pf_sp = (base_overall["prefill_s"] / a["prefill_s"]) if base_overall["prefill_s"] and a["prefill_s"] else 0
            print(f"  {m}: recall={a['recall']:.3f}  prefill {a['prefill_s']:.4f}s (speedup {pf_sp:.2f}x)  "
                  f"工具文档 {a['tools_tokens']:.0f}tok (压缩{tool_cr:.2f}x)  "
                  f"prompt {a['prompt_tokens']:.0f}tok (压缩{prompt_cr:.2f}x)")

    # 分类别汇总
    categories = sorted({t["official_category"] for ts in all_results.values() for t in ts})
    summary_by_cat = {}  # {category: {mode: agg}}
    base_by_cat = {}
    for cat in categories:
        groups = {}
        for m, ts in all_results.items():
            sub = [t for t in ts if t["official_category"] == cat]
            groups[f"{m}"] = _agg(sub)
            if m == "baseline" and groups[m]:
                base_by_cat[cat] = groups[m]
        summary_by_cat[cat] = groups
        _print_table(f"Reranker 召回/精度 + 速度·{cat}（每条平均）", groups, base_by_cat)
        # 该类别的压缩比 + prefill speedup
        b = base_by_cat.get(cat)
        if b:
            print(f"  压缩比（vs baseline，speedup 基于 prefill）:")
            for m, a in groups.items():
                if m == "baseline" or a is None:
                    continue
                tool_cr = b["all_tools_tokens"] / a["tools_tokens"] if a["tools_tokens"] else 0
                prompt_cr = b["prompt_tokens"] / a["prompt_tokens"] if a["prompt_tokens"] else 0
                pf_sp = (b["prefill_s"] / a["prefill_s"]) if b["prefill_s"] and a["prefill_s"] else 0
                print(f"    {m}: recall={a['recall']:.3f}  prefill {a['prefill_s']:.4f}s (speedup {pf_sp:.2f}x)  "
                      f"工具文档 {a['tools_tokens']:.0f}tok (压缩{tool_cr:.2f}x)  "
                      f"prompt {a['prompt_tokens']:.0f}tok (压缩{prompt_cr:.2f}x)")

    # 写 JSON：总体 + 分类别
    out = {"overall": overall, "by_category": summary_by_cat, "per_sample": all_results}
    out_path = os.path.join(report_dir, "benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n详细数据写入: {out_path}")
    print(f"报告写入: {report_path}")

    # 恢复 stdout，关闭报告文件
    sys.stdout = orig_stdout
    report_file.close()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
