"""
端侧reranker评分 → 本地topk筛选 → 拼接 → 生成端侧LLM推理JSON

完整流程：
  Step 1: extract_dataset.py --mode chunk → 切分 + reranker推理JSON + chunks.json
  Step 2: prepare_and_run.sh → 端侧reranker推理 → 回收分数.txt
  Step 3: parse_and_collect.py → 端侧分数 → CSV
  Step 3.5: 本脚本 → 读取chunks + 端侧分数 → topk筛选 → 拼接 → 端侧LLM推理JSON
  Step 4: run_baseline_device.sh → 端侧LLM推理 → 回收回答
  Step 5: parse_baseline_output.py → 评分
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

import numpy as np
from transformers import AutoTokenizer

# 添加 src 目录到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# 从 extract_dataset.py 导入共享函数和常量
sys.path.insert(0, os.path.dirname(__file__))
from extract_dataset import (
    INSTRUCTION,
    dataset2prompt,
    dataset2maxlen,
    build_llm_text_messages,
    build_baseline_json_template,
    save_baseline_json,
    read_input_samples,
)


def load_chunks_info(chunks_dir: str) -> dict:
    """
    加载所有样本的 chunks 信息

    Args:
        chunks_dir: 包含 *_chunks.json 文件的目录路径（通常为 output_dir/inputs/）

    Returns:
        dict: 以 sample_id 为 key 的 chunks 信息字典
    """
    chunks_map = {}

    if not os.path.exists(chunks_dir):
        print(f"⚠️ 找不到目录 {chunks_dir}")
        return chunks_map

    for filename in os.listdir(chunks_dir):
        if filename.endswith("_chunks.json"):
            file_path = os.path.join(chunks_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    chunk_info = json.load(f)
                sample_id = chunk_info["sample_id"]
                chunks_map[sample_id] = chunk_info
            except Exception as e:
                print(f"⚠️ 读取文件 {filename} 失败: {e}")

    print(f"✅ 加载了 {len(chunks_map)} 个样本的 chunks 信息")
    return chunks_map


def load_scores_csv(scores_csv: str) -> dict:
    """
    从 parse_and_collect.py 生成的 CSV 加载端侧reranker分数

    CSV 格式: sample_id, chunk_index, correlation_coefficient

    Args:
        scores_csv: CSV 文件路径

    Returns:
        dict: 以 sample_id 为 key，值为 {chunk_index: score} 的字典
    """
    scores_map = {}

    if not os.path.exists(scores_csv):
        print(f"⚠️ 找不到文件 {scores_csv}")
        return scores_map

    with open(scores_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row["sample_id"]
            chunk_index = int(row["chunk_index"])
            score = float(row["correlation_coefficient"])

            if sample_id not in scores_map:
                scores_map[sample_id] = {}
            scores_map[sample_id][chunk_index] = score

    print(f"✅ 加载了 {len(scores_map)} 个样本的端侧分数")
    return scores_map


def select_chunks_topk(chunks: list, scores: list, rate: float):
    """
    topk 选择模式，与 RerankCompressor.compress() 的 topk 逻辑一致

    策略：
      - 1/3 uniform sampling（均匀采样）
      - 2/3 importance sampling（按分数排序选取）
      - 首尾chunk始终保留

    Args:
        chunks: 文本块列表
        scores: 对应的分数列表
        rate: 压缩率（保留比例）

    Returns:
        tuple: (selected_chunks, selected_indices, actual_rate)
    """
    n = len(chunks)
    k = max(1, int(n * rate))

    # 1/3 uniform sampling
    k_uni = k // 3
    # 2/3 importance sampling
    k_imp = k - k_uni

    # uniform: 均匀分布的索引
    uniform_indices = np.linspace(0, n - 1, k_uni, dtype=int).tolist()
    selected = set(uniform_indices)

    # importance: 从未被均匀选中的chunk中，按分数取top
    remaining_indices = [i for i in range(n) if i not in selected]
    topk_imp = sorted(sorted(remaining_indices, key=lambda i: scores[i], reverse=True)[:k_imp])

    # 首尾始终保留
    selected_indices = sorted(selected.union(topk_imp).union({0, n - 1}))
    selected_chunks = [chunks[i] for i in selected_indices]

    actual_rate = len(selected_chunks) / n
    return selected_chunks, selected_indices, actual_rate


def build_scores_list(chunks_map_entry: dict, scores_map_entry: dict) -> list:
    """
    根据 chunks 信息和端侧分数，构建与 chunks 顺序对应的分数列表

    Args:
        chunks_map_entry: 单个样本的 chunks 信息
        scores_map_entry: 单个样本的端侧分数 {chunk_index: score}

    Returns:
        list: 与 chunks 顺序对应的分数列表
    """
    num_chunks = len(chunks_map_entry["chunks"])
    scores = []
    for idx in range(num_chunks):
        if idx in scores_map_entry:
            scores.append(scores_map_entry[idx])
        else:
            # 如果端侧推理缺少某个chunk的分数，使用 0 作为默认值
            print(f"⚠️ 样本 {chunks_map_entry['sample_id']} 缺少 chunk {idx} 的分数，使用默认值 0")
            scores.append(0.0)
    return scores


def truncate_context(tokenizer, context: str, max_ctx: int, max_gen: int) -> str:
    """
    截断超长context：保留前半和后半（中间截断），与 eval_longbench.py 的做法一致

    Args:
        tokenizer: tokenizer实例
        context: 原始context文本
        max_ctx: 最大上下文token数
        max_gen: 最大生成token数

    Returns:
        str: 截断后的context（如果未超长则返回原文本）
    """
    token_ids = tokenizer.encode(context)
    max_input = max_ctx - max_gen
    if len(token_ids) > max_input:
        half = int(max_input / 2) - 1
        truncated = tokenizer.decode(token_ids[:half]) + tokenizer.decode(token_ids[-half:])
        print(f"  截断: 原文 {len(token_ids)} tokens → ~{2 * half} tokens")
        return truncated
    return context


def process_sample(
    sample_id,
    chunks_map_entry,
    scores_map_entry,
    rate,
    json_template,
    output_dir,
    tokenizer,
    max_ctx,
    max_gen,
    save_compressed_info=False,
    compressed_info_dir=None,
):
    """
    处理单个样本：topk筛选 → 拼接 → 截断 → 生成端侧LLM推理JSON

    Args:
        sample_id: 样本ID
        chunks_map_entry: 单个样本的 chunks 信息（instruction, query, chunks, dataset）
        scores_map_entry: 单个样本的端侧分数 {chunk_index: score}
        rate: 压缩率
        json_template: 端侧LLM推理JSON模板
        output_dir: 输出目录
        tokenizer: tokenizer实例（用于截断）
        max_ctx: 最大上下文token数
        max_gen: 最大生成token数
        save_compressed_info: 是否保存压缩信息
        compressed_info_dir: 压缩信息保存目录

    Returns:
        bool: 是否成功处理
    """
    instruction = chunks_map_entry["instruction"]
    query = chunks_map_entry["query"]
    chunks = chunks_map_entry["chunks"]
    dataset = chunks_map_entry["dataset"]

    # 构建分数列表
    scores = build_scores_list(chunks_map_entry, scores_map_entry)

    # topk 筛选
    selected_chunks, selected_indices, actual_rate = select_chunks_topk(chunks, scores, rate)
    print(f"  样本 {sample_id}: {len(chunks)} chunks → 选中 {len(selected_chunks)} chunks (rate={actual_rate:.4f})")

    # 拼接压缩文本
    compressed_context = "".join(selected_chunks)

    # 截断超长context
    compressed_context = truncate_context(tokenizer, compressed_context, max_ctx, max_gen)

    # 构建端侧LLM推理JSON的text字段
    text_messages = build_llm_text_messages(dataset, compressed_context, query)

    # 保存JSON
    save_baseline_json(
        output_dir=output_dir,
        sample_id=sample_id,
        text_messages=text_messages,
        json_template=json_template,
    )

    # 保存压缩信息（可选）
    if save_compressed_info and compressed_info_dir:
        os.makedirs(compressed_info_dir, exist_ok=True)
        info = {
            "sample_id": sample_id,
            "dataset": dataset,
            "instruction": instruction,
            "query": query,
            "original_num_chunks": len(chunks),
            "selected_num_chunks": len(selected_chunks),
            "selected_indices": selected_indices,
            "actual_rate": actual_rate,
            "rate": rate,
            "scores": scores,
            "compressed_context_length": len(compressed_context),
        }
        info_file = os.path.join(compressed_info_dir, f"{sample_id}_compressed.json")
        with open(info_file, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="端侧reranker评分 → 本地topk筛选 → 拼接 → 生成端侧LLM推理JSON"
    )
    parser.add_argument(
        "--chunks_dir",
        type=str,
        required=True,
        help="包含 *_chunks.json 文件的目录路径（通常为 extract_dataset.py 输出的 inputs/ 子目录）",
    )
    parser.add_argument(
        "--scores_csv",
        type=str,
        required=True,
        help="parse_and_collect.py 生成的端侧分数CSV文件",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="原始数据集 JSONL 文件路径（用于获取 dataset、input 字段）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录，存放端侧LLM推理JSON文件",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0.5,
        help="压缩率（保留比例），默认 0.5",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="端侧LLM tokenizer 路径（用于截断判断），如不指定则不做截断",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="端侧LLM模型名称（用于加载tokenizer，仅在未指定 --tokenizer_path 时使用）",
    )
    parser.add_argument(
        "--params_path",
        type=str,
        required=True,
        help="端侧LLM 参数路径（用于JSON模板）",
    )
    parser.add_argument(
        "--max_ctx",
        type=int,
        default=8192,
        help="端侧LLM 最大上下文token数，默认 8192",
    )
    parser.add_argument(
        "--save_compressed",
        action="store_true",
        help="保存压缩信息（选中的chunk索引、实际压缩率等），便于后续分析",
    )

    args = parser.parse_args()

    # 构建端侧LLM推理JSON模板
    json_template = build_baseline_json_template(args)

    # 加载 tokenizer（用于截断判断）
    tokenizer = None
    if args.tokenizer_path:
        print(f"正在从本地路径加载 tokenizer: {args.tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    else:
        print(f"正在加载 tokenizer: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 加载 chunks 信息
    print("\n步骤 1: 加载 chunks 信息")
    chunks_map = load_chunks_info(args.chunks_dir)
    if not chunks_map:
        print("❌ 没有找到 chunks 信息，退出")
        return

    # 加载端侧分数
    print("\n步骤 2: 加载端侧分数")
    scores_map = load_scores_csv(args.scores_csv)
    if not scores_map:
        print("❌ 没有找到端侧分数，退出")
        return

    # 加载原始数据集（获取 max_gen 信息）
    print("\n步骤 3: 加载原始数据集")
    samples = read_input_samples(args.input_file)
    print(f"✅ 共读取 {len(samples)} 个样本")

    # 构建 dataset → max_gen 的映射
    # 用于截断时确定每个数据集的最大生成长度
    sample_id_to_max_gen = {}
    for sample in samples:
        _id = sample.get("_id", 0)
        dataset = sample.get("dataset", "unknown")
        sample_id_to_max_gen[_id] = dataset2maxlen.get(dataset, 64)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    compressed_info_dir = os.path.join(args.output_dir, "compressed_info") if args.save_compressed else None

    # 处理所有样本
    print(f"\n步骤 4: topk筛选 → 拼接 → 生成LLM推理JSON (rate={args.rate})")
    success_count = 0
    skip_count = 0

    for sample_id, chunks_info in tqdm(chunks_map.items(), desc="处理样本"):
        # 检查是否有对应的端侧分数
        if sample_id not in scores_map:
            print(f"⚠️ 样本 {sample_id} 没有端侧分数，跳过")
            skip_count += 1
            continue

        # 获取该样本的 max_gen
        max_gen = sample_id_to_max_gen.get(sample_id, 64)

        try:
            success = process_sample(
                sample_id=sample_id,
                chunks_map_entry=chunks_info,
                scores_map_entry=scores_map[sample_id],
                rate=args.rate,
                json_template=json_template,
                output_dir=args.output_dir,
                tokenizer=tokenizer,
                max_ctx=args.max_ctx,
                max_gen=max_gen,
                save_compressed_info=args.save_compressed,
                compressed_info_dir=compressed_info_dir,
            )
            if success:
                success_count += 1
        except Exception as e:
            print(f"❌ 处理样本 {sample_id} 时出错: {e}")
            import traceback
            traceback.print_exc()

    # 统计
    print(f"\n{'=' * 60}")
    print(f"处理完成！")
    print(f"  成功: {success_count} 个样本")
    print(f"  跳过（无分数）: {skip_count} 个样本")
    print(f"  总 chunks 信息: {len(chunks_map)} 个样本")
    print(f"输出目录: {args.output_dir}")
    if args.save_compressed:
        print(f"压缩信息目录: {compressed_info_dir}")
    print(f"下一步请运行:")
    print(f"  ./run_baseline_device.sh -i {args.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()