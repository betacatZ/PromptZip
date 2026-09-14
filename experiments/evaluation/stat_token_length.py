"""
统计 LongBench-Pro 数据集的平均 token 长度(用 Qwen3 tokenizer 实际编码计算)。

数据集自带的 token_length 字段只是档位标签("8k"/"32k"/"128k"),
本脚本对每个样本的 context 实际分词,输出:
- 总体统计:均值/中位数/最小/最大/分位数
- 分维度统计:按官方 token_length 档位、primary_task、language、difficulty 分组

仅需 CPU,无需 GPU。用法:
    cd experiments/evaluation
    python stat_token_length.py
    python stat_token_length.py --tokenizer Qwen/Qwen3-8B --save token_stats.csv
"""

import argparse
import json
import os
import statistics
import csv
from collections import defaultdict

from tqdm import tqdm
from transformers import AutoTokenizer

DEFAULT_DATASET_PATH = "/home/zdm/code/PromptZip/datasets/LongBench-Pro/longbench_pro.json"
DEFAULT_TOKENIZER = "Qwen/Qwen3-8B"  # Qwen3 系列 tokenizer 通用,用哪个权重都一样


def load_lbpro_dataset(dataset_path):
    """加载 LongBench-Pro 数据集,本地 json 优先,不存在则回退 HuggingFace。"""
    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)
    # 懒导入,与 eval_longbench_pro.py 一致
    from datasets import load_dataset

    ds = load_dataset("caskcsg/LongBench-Pro", split="test")
    return list(ds)


def batch_encode_lengths(tokenizer, contexts, batch_size):
    """分批编码,返回每个 context 的 token 数(add_special_tokens=False)。"""
    lengths = []
    for i in tqdm(range(0, len(contexts), batch_size), desc="tokenize"):
        batch = contexts[i : i + batch_size]
        encodings = tokenizer(batch, add_special_tokens=False)["input_ids"]
        lengths.extend(len(ids) for ids in encodings)
    return lengths


def describe(values):
    """给定 token 数列表,返回统计描述 dict。"""
    if not values:
        return {}
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def percentile(p):
        idx = min(int(n * p / 100), n - 1)
        return sorted_vals[idx]

    return {
        "count": n,
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "p25": percentile(25),
        "p75": percentile(75),
        "p95": percentile(95),
    }


def print_stats(title, values):
    """打印一组 token 数的统计信息。"""
    s = describe(values)
    if not s:
        print(f"\n[{title}] 无样本")
        return
    print(f"\n[{title}] (n={s['count']})")
    print(f"  mean={s['mean']:.1f}  median={s['median']:.1f}")
    print(f"  min={s['min']}  p25={s['p25']}  p75={s['p75']}  p95={s['p95']}  max={s['max']}")


# token 长度分桶边界(与官方档位对齐):8k-16k, 16k-32k, 32k-64k, 64k-128k, 128k-256k
LENGTH_BUCKETS = [
    ("8k-16k", 8 * 1024, 16 * 1024),
    ("16k-32k", 16 * 1024, 32 * 1024),
    ("32k-64k", 32 * 1024, 64 * 1024),
    ("64k-128k", 64 * 1024, 128 * 1024),
    ("128k-256k", 128 * 1024, 256 * 1024),
]


def bucket_lengths(tok_lengths):
    """按长度范围分桶,返回 [(范围, 个数, 占比)],含 <8k 和 >256k 的溢出桶。"""
    n = len(tok_lengths)
    buckets = []
    overflow_low = sum(1 for l in tok_lengths if l < LENGTH_BUCKETS[0][1])
    if overflow_low:
        buckets.append((f"<{LENGTH_BUCKETS[0][0].split('-')[0]}", overflow_low, overflow_low / n))
    for label, low, high in LENGTH_BUCKETS:
        count = sum(1 for l in tok_lengths if low <= l < high)
        buckets.append((label, count, count / n))
    overflow_high = sum(1 for l in tok_lengths if l >= LENGTH_BUCKETS[-1][2])
    if overflow_high:
        buckets.append((f">{LENGTH_BUCKETS[-1][0].split('-')[1]}", overflow_high, overflow_high / n))
    return buckets


def print_bucket_stats(tok_lengths):
    """打印各长度范围的样本个数与占比。"""
    print(f"\n===== 按 token 长度范围分桶 =====")
    print(f"{'范围':<12} {'样本数':>6} {'占比':>8}")
    print("-" * 30)
    for label, count, ratio in bucket_lengths(tok_lengths):
        print(f"{label:<12} {count:>6} {ratio:>7.2%}")


def group_and_print(data, tok_lengths, group_key, label):
    """按样本的某个字段分组,打印各组 token 数统计。"""
    groups = defaultdict(list)
    for sample, length in zip(data, tok_lengths):
        groups[sample.get(group_key, "unknown")].append(length)

    print(f"\n===== 按 {label} 分组 =====")
    print(f"{'分组':<40} {'样本数':>6} {'平均token':>10} {'中位数':>10}")
    print("-" * 70)
    for key in sorted(groups):
        vals = groups[key]
        print(f"{key:<40} {len(vals):>6} {statistics.mean(vals):>10.1f} {statistics.median(vals):>10.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="LongBench-Pro json 路径")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="Qwen3 tokenizer 模型名")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save", default=None, help="可选:逐样本结果保存为 CSV")
    args = parser.parse_args()

    print(f"加载数据集: {args.dataset}")
    data = load_lbpro_dataset(args.dataset)
    print(f"共 {len(data)} 个样本")

    print(f"加载 tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    contexts = [sample["context"] for sample in data]
    tok_lengths = batch_encode_lengths(tokenizer, contexts, args.batch_size)

    # 总体统计
    print_stats("总体 context token 长度", tok_lengths)

    # 长度范围分桶
    print_bucket_stats(tok_lengths)

    # 分维度统计
    group_and_print(data, tok_lengths, "token_length", "官方 token_length 档位")
    group_and_print(data, tok_lengths, "primary_task", "primary_task")
    group_and_print(data, tok_lengths, "language", "language")
    group_and_print(data, tok_lengths, "difficulty", "difficulty")

    # 逐样本 CSV
    if args.save:
        with open(args.save, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "token_length_label", "language", "primary_task", "secondary_task", "difficulty", "context_tokens"])
            for sample, length in zip(data, tok_lengths):
                writer.writerow(
                    [
                        sample["id"],
                        sample["token_length"],
                        sample["language"],
                        sample["primary_task"],
                        sample["secondary_task"],
                        sample["difficulty"],
                        length,
                    ]
                )
        print(f"\n逐样本结果已写入: {args.save}")


if __name__ == "__main__":
    main()
