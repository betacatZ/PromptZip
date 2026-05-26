import argparse
import json
import os
import re
import csv
import sys
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch

# 尝试导入 scipy，如果没有则提供替代实现
try:
    from scipy import stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️ 没有安装 scipy，将使用简化的 Spearman 秩相关系数计算")
    print("⚠️ 如果需要完整功能，请安装: pip install scipy")

# 添加 src 目录到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.compressor import RerankCompressor


def spearman_rank_correlation_simple(gpu_scores, device_scores):
    """
    简化的 Spearman 秩相关系数计算，不依赖 scipy
    """
    n = len(gpu_scores)

    # 得到每个分数的排名
    def get_ranks(arr):
        sorted_indices = sorted(range(n), key=lambda i: -arr[i])
        ranks = [0] * n
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank + 1
        return ranks

    gpu_ranks = get_ranks(gpu_scores)
    device_ranks = get_ranks(device_scores)

    # 计算平方差的和
    sum_squared_diff = sum((g - d) ** 2 for g, d in zip(gpu_ranks, device_ranks))

    # 计算 Spearman 系数
    spearman = 1 - (6 * sum_squared_diff) / (n * (n**2 - 1))

    return spearman


def load_chunks_info(input_dir: str) -> List[Dict]:
    """
    加载所有样本的 chunks 信息

    Args:
        input_dir: 包含 inputs/ 目录的父目录

    Returns:
        样本信息列表
    """
    samples = []
    inputs_dir = os.path.join(input_dir, "inputs")

    if not os.path.exists(inputs_dir):
        print(f"⚠️ 警告：找不到目录 {inputs_dir}")
        return samples

    # 遍历所有 *_chunks.json 文件
    for filename in os.listdir(inputs_dir):
        if filename.endswith("_chunks.json"):
            file_path = os.path.join(inputs_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    chunk_info = json.load(f)
                samples.append(chunk_info)
            except Exception as e:
                print(f"⚠️ 读取文件 {filename} 失败: {e}")

    print(f"✅ 加载了 {len(samples)} 个样本的 chunks 信息")
    return samples


def compute_scores_gpu(
    compressor, instruction: str, query: str, chunks: List[str], batch_size: int = 32
) -> List[float]:
    """
    在GPU上使用 RerankCompressor 计算得分

    Args:
        compressor: RerankCompressor 实例
        instruction: 指令
        query: 查询
        chunks: 文本块列表
        batch_size: 批处理大小

    Returns:
        得分列表
    """
    scores = []

    # 分批次处理
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]

        # 构建输入对
        batch_pairs = [compressor.format_instruction(instruction, query, chunk) for chunk in batch_chunks]

        # 处理输入
        batch_inputs = compressor.process_inputs(batch_pairs)

        # 模型推理
        with torch.no_grad():
            outputs = compressor.model(**batch_inputs)

        # 计算得分
        batch_scores = outputs.logits[:, -1, :]
        true_vector = batch_scores[:, compressor.token_true_id]
        false_vector = batch_scores[:, compressor.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        batch_scores = batch_scores[:, 1].exp().tolist()

        scores.extend(batch_scores)

    return scores


def parse_device_scores(device_outputs_dir: str, sample_id: str, num_chunks: int) -> Optional[List[float]]:
    """
    从 device_outputs/ 目录解析端侧推理得分

    Args:
        device_outputs_dir: device_outputs 目录的绝对路径
        sample_id: 样本ID（可以是字符串）
        num_chunks: chunks 的数量

    Returns:
        得分列表，如果解析失败返回None
    """
    if not os.path.exists(device_outputs_dir):
        print(f"⚠️ 找不到目录 {device_outputs_dir}")
        return None

    scores = []
    pattern = r"correlation coefficient:\s*([\d.]+)"

    for chunk_idx in range(num_chunks):
        filename = f"sample_{sample_id}_chunk_{chunk_idx}.txt"
        file_path = os.path.join(device_outputs_dir, filename)

        if not os.path.exists(file_path):
            print(f"⚠️ 找不到文件 {filename}")
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 使用正则表达式提取得分
            match = re.search(pattern, content)
            if match:
                score = float(match.group(1))
                scores.append(score)
            else:
                print(f"⚠️ 文件 {filename} 中找不到 correlation coefficient")
                return None
        except Exception as e:
            print(f"⚠️ 解析文件 {filename} 失败: {e}")
            return None

    return scores


def compute_top_k_agreement(gpu_scores: List[float], device_scores: List[float], k_values: List) -> Dict:
    """
    计算 Top-K 重合度，K可以是整数或百分比

    Args:
        gpu_scores: GPU得分
        device_scores: 端侧得分
        k_values: 要评估的k值列表，可以是整数或百分比字符串（例如 "10%"）

    Returns:
        每个k值对应的重合度
    """
    # 获取排序索引
    gpu_indices = np.argsort(gpu_scores)[::-1]  # 降序
    device_indices = np.argsort(device_scores)[::-1]
    num_chunks = len(gpu_indices)

    agreements = {}
    for k in k_values:
        # 解析k值：可以是整数或百分比字符串
        if isinstance(k, str) and k.endswith("%"):
            try:
                percentage = float(k[:-1]) / 100
                actual_k = max(1, int(num_chunks * percentage))
            except ValueError:
                continue
        elif isinstance(k, int):
            actual_k = k
        else:
            continue

        if actual_k > num_chunks:
            agreements[k] = None
            continue

        gpu_top = set(gpu_indices[:actual_k])
        device_top = set(device_indices[:actual_k])

        intersection = len(gpu_top & device_top)
        agreement = intersection / actual_k
        agreements[k] = agreement

    return agreements


def compute_ranking_metrics(gpu_scores: List[float], device_scores: List[float]) -> Dict:
    """
    计算排序相关指标

    Args:
        gpu_scores: GPU得分
        device_scores: 端侧得分

    Returns:
        指标字典
    """
    metrics = {}

    # Spearman秩相关系数
    if HAS_SCIPY:
        try:
            spearman_corr, spearman_p = stats.spearmanr(gpu_scores, device_scores)
            metrics["spearman_correlation"] = spearman_corr
            metrics["spearman_p_value"] = spearman_p
        except Exception as e:
            print(f"⚠️ 计算 Spearman 系数出错: {e}")
            metrics["spearman_correlation"] = spearman_rank_correlation_simple(gpu_scores, device_scores)
            metrics["spearman_p_value"] = float("nan")
    else:
        metrics["spearman_correlation"] = spearman_rank_correlation_simple(gpu_scores, device_scores)
        metrics["spearman_p_value"] = float("nan")

    # Kendall Tau相关系数
    if HAS_SCIPY:
        try:
            kendall_corr, kendall_p = stats.kendalltau(gpu_scores, device_scores)
            metrics["kendall_tau"] = kendall_corr
            metrics["kendall_p_value"] = kendall_p
        except Exception as e:
            print(f"⚠️ 计算 Kendall Tau 系数出错: {e}")
            metrics["kendall_tau"] = float("nan")
            metrics["kendall_p_value"] = float("nan")
    else:
        metrics["kendall_tau"] = float("nan")
        metrics["kendall_p_value"] = float("nan")

    # 得分差异指标
    gpu_array = np.array(gpu_scores)
    device_array = np.array(device_scores)
    diff_array = gpu_array - device_array

    metrics["mae"] = float(np.mean(np.abs(diff_array)))
    metrics["mse"] = float(np.mean(np.square(diff_array)))
    metrics["max_abs_error"] = float(np.max(np.abs(diff_array)))
    metrics["min_abs_error"] = float(np.min(np.abs(diff_array)))

    return metrics


def compare_sample(
    compressor, chunk_info: Dict, device_outputs_dir: str, batch_size: int, top_ks: List[int]
) -> Optional[Dict]:
    """
    对比单个样本的GPU和端侧结果

    Args:
        compressor: RerankCompressor 实例
        chunk_info: 样本的chunk信息
        device_outputs_dir: device_outputs 目录路径
        batch_size: 批大小
        top_ks: 要评估的top-k列表

    Returns:
        对比结果字典
    """
    sample_id = chunk_info["sample_id"]
    chunks = chunk_info["chunks"]
    instruction = chunk_info["instruction"]
    query = chunk_info["query"]

    print(f"\n处理样本 {sample_id} (num_chunks: {len(chunks)})")

    # GPU推理
    try:
        gpu_scores = compute_scores_gpu(compressor, instruction, query, chunks, batch_size)
        print(f"✅ GPU推理完成")
    except Exception as e:
        print(f"❌ GPU推理失败: {e}")
        import traceback

        traceback.print_exc()
        return None

    # 解析端侧得分
    device_scores = parse_device_scores(device_outputs_dir, sample_id, len(chunks))
    if device_scores is None:
        return None
    print(f"✅ 端侧得分解析完成")

    # 计算排序指标
    ranking_metrics = compute_ranking_metrics(gpu_scores, device_scores)

    # 计算Top-K重合度
    top_k_agreements = compute_top_k_agreement(gpu_scores, device_scores, top_ks)

    # 组合结果
    result = {
        "sample_id": sample_id,
        "dataset": chunk_info["dataset"],
        "num_chunks": len(chunks),
        "gpu_scores": gpu_scores,
        "device_scores": device_scores,
        "ranking_metrics": ranking_metrics,
        "top_k_agreements": top_k_agreements,
    }

    return result


def save_results_to_csv(results: List[Dict], output_file: str, top_ks: List):
    """
    保存对比结果到CSV文件

    Args:
        results: 对比结果列表
        output_file: 输出文件路径
        top_ks: top-k 列表（可以是整数或百分比）
    """
    # 构建CSV表头
    fieldnames = [
        "sample_id",
        "dataset",
        "num_chunks",
        "spearman_correlation",
        "spearman_p_value",
        "kendall_tau",
        "kendall_p_value",
        "mae",
        "mse",
        "max_abs_error",
        "min_abs_error",
    ]
    for k in top_ks:
        fieldnames.append(f"top_{k}_agreement")

    # 写文件
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            row = {
                "sample_id": result["sample_id"],
                "dataset": result["dataset"],
                "num_chunks": result["num_chunks"],
                "spearman_correlation": result["ranking_metrics"]["spearman_correlation"],
                "spearman_p_value": result["ranking_metrics"]["spearman_p_value"],
                "kendall_tau": result["ranking_metrics"]["kendall_tau"],
                "kendall_p_value": result["ranking_metrics"]["kendall_p_value"],
                "mae": result["ranking_metrics"]["mae"],
                "mse": result["ranking_metrics"]["mse"],
                "max_abs_error": result["ranking_metrics"]["max_abs_error"],
                "min_abs_error": result["ranking_metrics"]["min_abs_error"],
            }
            for k in top_ks:
                agreement = result["top_k_agreements"].get(k, None)
                row[f"top_{k}_agreement"] = agreement

            writer.writerow(row)

    print(f"✅ CSV结果已保存到 {output_file}")


def generate_summary_report(results: List[Dict], output_file: str, top_ks: List[int]):
    """
    生成统计摘要报告

    Args:
        results: 对比结果列表
        output_file: 输出文件路径
        top_ks: top-k 列表
    """
    num_samples = len(results)

    # 收集所有指标
    spearman_corrs = [r["ranking_metrics"]["spearman_correlation"] for r in results]
    kendall_taus = [r["ranking_metrics"]["kendall_tau"] for r in results]
    maes = [r["ranking_metrics"]["mae"] for r in results]
    mses = [r["ranking_metrics"]["mse"] for r in results]

    # 统计Top-K重合度
    top_k_avg_agreements = {}
    for k in top_ks:
        agreements = [
            r["top_k_agreements"].get(k, None) for r in results if r["top_k_agreements"].get(k, None) is not None
        ]
        if agreements:
            top_k_avg_agreements[k] = float(np.mean(agreements))

    # 写报告
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("GPU与端侧推理对比结果摘要报告\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"总样本数: {num_samples}\n\n")

        f.write("-" * 80 + "\n")
        f.write("排序相关性统计\n")
        f.write("-" * 80 + "\n")

        f.write(f"Spearman 秩相关系数:\n")
        f.write(f"  Mean: {np.mean(spearman_corrs):.6f}\n")
        f.write(f"  Std:  {np.std(spearman_corrs):.6f}\n")
        f.write(f"  Min:  {np.min(spearman_corrs):.6f}\n")
        f.write(f"  Max:  {np.max(spearman_corrs):.6f}\n\n")

        f.write(f"Kendall Tau 相关系数:\n")
        f.write(f"  Mean: {np.nanmean(kendall_taus):.6f}\n")
        f.write(f"  Std:  {np.nanstd(kendall_taus):.6f}\n")
        f.write(f"  Min:  {np.nanmin(kendall_taus):.6f}\n")
        f.write(f"  Max:  {np.nanmax(kendall_taus):.6f}\n\n")

        f.write("-" * 80 + "\n")
        f.write("得分差异统计\n")
        f.write("-" * 80 + "\n")

        f.write(f"MAE (平均绝对误差):\n")
        f.write(f"  Mean: {np.mean(maes):.6f}\n")
        f.write(f"  Std:  {np.std(maes):.6f}\n")
        f.write(f"  Min:  {np.min(maes):.6f}\n")
        f.write(f"  Max:  {np.max(maes):.6f}\n\n")

        f.write(f"MSE (均方误差):\n")
        f.write(f"  Mean: {np.mean(mses):.6f}\n")
        f.write(f"  Std:  {np.std(mses):.6f}\n")
        f.write(f"  Min:  {np.min(mses):.6f}\n")
        f.write(f"  Max:  {np.max(mses):.6f}\n\n")

        f.write("-" * 80 + "\n")
        f.write("Top-K 重合度统计\n")
        f.write("-" * 80 + "\n")

        for k in top_ks:
            if k in top_k_avg_agreements:
                avg_agreement = top_k_avg_agreements[k]
                f.write(f"Top-{k} 平均重合度: {avg_agreement:.4f}\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"✅ 摘要报告已保存到 {output_file}")


def main():
    parser = argparse.ArgumentParser(description="对比GPU与端侧推理结果")
    parser.add_argument("--input_dir", type=str, required=True, help="包含 inputs/ 目录的目录")
    parser.add_argument("--device_outputs_dir", type=str, required=True, help="包含 device_outputs/ 目录的目录")
    parser.add_argument("--output_dir", type=str, default="./comparison_output", help="输出目录")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Reranker-0.6B", help="Rerank模型名称")
    parser.add_argument("--device", type=str, default="cuda", help="使用的设备 (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=32, help="批处理大小")
    parser.add_argument(
        "--top_ks",
        type=str,
        default="10%,20%,50%",
        help="要评估的Top-K值，用逗号分隔，可以是整数或百分比 (例如: 1,3,5 或 10%,20%,50%)",
    )

    args = parser.parse_args()

    # 解析 Top-K 参数
    top_ks = []
    for k_str in args.top_ks.split(","):
        k_str = k_str.strip()
        if k_str.endswith("%"):
            top_ks.append(k_str)
        else:
            try:
                top_ks.append(int(k_str))
            except ValueError:
                print(f"⚠️ 无法解析参数 '{k_str}'，跳过")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载所有样本信息
    print("=" * 80)
    print("步骤 1: 加载样本信息")
    print("=" * 80)
    samples = load_chunks_info(args.input_dir)
    if not samples:
        print("❌ 没有找到样本信息，退出")
        return

    # 初始化 RerankCompressor
    print("\n" + "=" * 80)
    print("步骤 2: 初始化 RerankCompressor")
    print("=" * 80)

    try:
        compressor = RerankCompressor(
            model_name=args.model_name,
            device_map=args.device,
            engine="hf",  # 必须使用hf engine
            chunk_end_tokens=["。", "！", "？", ".", "!", "?", "\n", "。\n", "？\n", "！\n"],
        )
        print(f"✅ RerankCompressor 初始化成功")
    except Exception as e:
        print(f"❌ RerankCompressor 初始化失败: {e}")
        import traceback

        traceback.print_exc()
        return

    # 对比每个样本
    print("\n" + "=" * 80)
    print("步骤 3: 对比每个样本")
    print("=" * 80)

    all_results = []
    for chunk_info in tqdm(samples, desc="对比样本"):
        try:
            result = compare_sample(compressor, chunk_info, args.device_outputs_dir, args.batch_size, top_ks)
            if result is not None:
                all_results.append(result)
        except Exception as e:
            print(f"❌ 处理样本 {chunk_info.get('sample_id', 'unknown')} 时出错: {e}")
            import traceback

            traceback.print_exc()

    # 保存结果
    print("\n" + "=" * 80)
    print("步骤 4: 保存结果")
    print("=" * 80)

    csv_file = os.path.join(args.output_dir, "comparison_results.csv")
    save_results_to_csv(all_results, csv_file, top_ks)

    summary_file = os.path.join(args.output_dir, "summary_report.txt")
    generate_summary_report(all_results, summary_file, top_ks)

    print("\n" + "=" * 80)
    print("✅ 所有任务完成！")
    print(f"总处理样本数: {len(all_results)} / {len(samples)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
