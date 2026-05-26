import argparse
import os
import re
import csv
from pathlib import Path


def extract_correlation_coefficient(content):
    """
    从推理输出中提取 correlation coefficient
    """
    pattern = r"correlation coefficient:\s*([\d.]+)"
    match = re.search(pattern, content)
    if match:
        return float(match.group(1))
    return None


def parse_file_info_from_path(file_path):
    """
    从文件路径中解析样本 ID 和 chunk index
    假设文件命名格式类似：sample_1_chunk_0.txt 或 1_0.txt
    """
    filename = os.path.basename(file_path)
    name_without_ext = os.path.splitext(filename)[0]

    # 尝试多种模式解析
    # 模式 1: sample_1_chunk_0.txt 或 sample_abc_1_chunk_0.txt
    pattern1 = r"sample_([^_]+)_chunk_(\d+)"
    match1 = re.search(pattern1, name_without_ext)
    if match1:
        # sample_ 后面一定不是纯数字，直接返回字符串
        return match1.group(1), int(match1.group(2))

    # 模式 2: 1_0.txt
    pattern2 = r"(\d+)_(\d+)"
    match2 = re.search(pattern2, name_without_ext)
    if match2:
        return int(match2.group(1)), int(match2.group(2))

    # 如果不能解析，返回 None
    return None, None


def collect_results_from_dir(input_dir, output_file):
    """
    从指定目录收集所有结果并保存到 CSV 文件
    """
    results = []
    failed_files = []

    # 遍历目录中的所有 txt 文件
    txt_files = list(Path(input_dir).rglob("*.txt"))
    print(f"找到 {len(txt_files)} 个输出文件")

    for file_path in txt_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 提取 correlation coefficient
            coeff = extract_correlation_coefficient(content)

            if coeff is not None:
                # 解析样本 ID 和 chunk index
                sample_id, chunk_idx = parse_file_info_from_path(str(file_path))

                if sample_id is not None and chunk_idx is not None:
                    results.append({"sample_id": sample_id, "chunk_index": chunk_idx, "correlation_coefficient": coeff})
                else:
                    # 无法解析 ID 和 index，使用文件名作为标识
                    results.append({"sample_id": str(file_path), "chunk_index": -1, "correlation_coefficient": coeff})
                print(f"✓ {file_path}: {coeff}")
            else:
                failed_files.append(str(file_path))
                print(f"✗ {file_path}: 未找到 correlation coefficient")
        except Exception as e:
            failed_files.append(str(file_path))
            print(f"✗ {file_path}: 处理失败 - {e}")

    # 保存到 CSV
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["sample_id", "chunk_index", "correlation_coefficient"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    # 打印统计信息
    print(f"\n{'=' * 60}")
    print(f"成功处理: {len(results)} 个文件")
    print(f"失败: {len(failed_files)} 个文件")
    if failed_files:
        print(f"失败文件列表:")
        for f in failed_files[:10]:
            print(f"  - {f}")
        if len(failed_files) > 10:
            print(f"  ... 还有 {len(failed_files) - 10} 个文件")
    print(f"结果已保存到: {output_file}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="从端侧推理输出中收集 correlation coefficient")
    parser.add_argument("--input_dir", type=str, required=True, help="包含输出文件的目录")
    parser.add_argument("--output_file", type=str, default="correlation_results.csv", help="输出的 CSV 文件路径")

    args = parser.parse_args()

    collect_results_from_dir(args.input_dir, args.output_file)


if __name__ == "__main__":
    main()
