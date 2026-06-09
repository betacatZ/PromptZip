"""
解析端侧 baseline 推理输出，提取 <|im_start|>assistant 后的生成内容，
与数据集 ground truth 匹配，计算精度分数。

使用方式:
    python parse_baseline_output.py \
        --input_dir /path/to/device_output_txt_files \
        --dataset_dir /path/to/jsonl_datasets \
        --output_dir /path/to/results_output  (可选)

流程:
    1. 解析每个 .txt 文件，提取 [start of text] 和 [end of text] 之间
       <|im_start|>assistant 后的内容
    2. 从文件名获取 sample_id，从 JSONL 数据集加载对应的 ground truth
    3. 使用 eval_longbench.py 的 scorer() 和 metrics.py 计算精度
    4. 输出 result.json / score.json / score.csv
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

# 添加当前目录到 sys.path，确保可以 import metrics 和 eval_longbench
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_longbench import (
    scorer,
    write_score,
)


def parse_txt_file(txt_path):
    """
    解析端侧推理输出的 .txt 文件，提取 <|im_start|>assistant 后的生成内容。

    端侧输出格式：
        ... 初始化日志 ...
        ========== [start of text] ==========
        <|im_start|>system
        ...<|im_end|>
        <|im_start|>user
        ...<|im_end|>
        <|im_start|>assistant
        模型生成的回答内容...
        ========== [end of text] ==========
        ... 性能统计 ...

    Returns:
        str: 模型生成的回答内容（assistant 后的部分），如果解析失败返回空字符串
    """
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"  读取文件失败: {txt_path}, 错误: {e}")
        return ""

    # 1. 提取 [start of text] 和 [end of text] 之间的内容
    start_marker = "========== [start of text] =========="
    end_marker = "========== [end of text] =========="

    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)

    if start_idx == -1 or end_idx == -1:
        # 没有找到标记，尝试整个文件作为 text content
        print(f"  警告: 未找到 [start/end of text] 标记: {txt_path}")
        text_content = content
    else:
        text_content = content[start_idx + len(start_marker) : end_idx]

    # 2. 在 text_content 中找到 <|im_start|>assistant 后的内容
    assistant_marker = "<|im_start|>assistant"
    assistant_idx = text_content.find(assistant_marker)

    if assistant_idx == -1:
        print(f"  警告: 未找到 <|im_start|>assistant 标记: {txt_path}")
        # fallback: 返回整个 text_content（去除前后空白）
        return text_content.strip()

    # 提取 assistant 后的内容
    response = text_content[assistant_idx + len(assistant_marker) :]

    # 3. 清理: 去除开头的换行符
    response = response.lstrip("\n")

    # 4. 如果模型输出中出现了 <|im_end|> 或新的 <|im_start|>，
    #    表示模型开始生成新的对话轮次，应该截断
    #    取 <|im_end|> 或 <|im_start|> 之前的部分
    im_end_idx = response.find("<|im_end|>")
    im_start_idx = response.find("<|im_start|>")

    # 取最早出现的特殊 token 位置作为截断点
    truncate_idx = -1
    if im_end_idx != -1 and im_start_idx != -1:
        truncate_idx = min(im_end_idx, im_start_idx)
    elif im_end_idx != -1:
        truncate_idx = im_end_idx
    elif im_start_idx != -1:
        truncate_idx = im_start_idx

    if truncate_idx != -1:
        response = response[:truncate_idx]

    # 5. 去除尾部空白
    response = response.rstrip()

    return response


def get_sample_id_from_filename(filename):
    """
    从文件名中解析 sample_id。

    支持的文件名格式:
        - sample_{sample_id}_baseline.txt -> sample_id
        - {sample_id}.txt -> sample_id

    Returns:
        str: sample_id，如果无法解析返回 None
    """
    # 去除 .txt 后缀
    stem = filename.rsplit(".txt", 1)[0] if filename.endswith(".txt") else filename

    # 格式1: sample_{id}_baseline
    match = re.match(r"sample_(.+)_baseline", stem)
    if match:
        return match.group(1)

    # 格式2: 直接就是 id
    # 如果 stem 不包含 "sample_" 前缀，直接返回
    if not stem.startswith("sample_"):
        return stem

    return None


def load_ground_truth_from_jsonl(dataset_dir):
    """
    从 JSONL 数据集目录加载所有样本的 ground truth 信息。

    遍历 dataset_dir 下所有 .jsonl 文件，以 _id 为 key 构建 lookup dict。

    Args:
        dataset_dir: 包含各数据集 .jsonl 文件的目录

    Returns:
        dict: {sample_id: {"answers": [...], "dataset": "...", "all_classes": ..., "length": ..., "_id": "..."}}
    """
    ground_truth = {}

    if not os.path.isdir(dataset_dir):
        print(f"警告: 数据集目录不存在: {dataset_dir}")
        return ground_truth

    jsonl_files = [f for f in os.listdir(dataset_dir) if f.endswith(".jsonl")]

    if not jsonl_files:
        print(f"警告: 数据集目录中没有 .jsonl 文件: {dataset_dir}")
        return ground_truth

    print(f"  加载 {len(jsonl_files)} 个 JSONL 数据集文件...")
    for jsonl_file in jsonl_files:
        filepath = os.path.join(dataset_dir, jsonl_file)
        dataset_name = jsonl_file.rsplit(".jsonl", 1)[0]
        count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    sample_id = sample.get("_id", "")
                    if sample_id:
                        ground_truth[sample_id] = {
                            "answers": sample.get("answers", []),
                            "dataset": sample.get("dataset", dataset_name),
                            "all_classes": sample.get("all_classes", None),
                            "length": sample.get("length", 0),
                            "_id": sample_id,
                            "input": sample.get("input", ""),
                        }
                        count += 1
                except json.JSONDecodeError as e:
                    print(f"  JSONL 解析错误 ({jsonl_file}): {e}")
                    continue
        print(f"    {jsonl_file}: 加载 {count} 个样本")

    print(f"  共加载 {len(ground_truth)} 个样本的 ground truth")
    return ground_truth


def load_ground_truth_from_inputs(input_dir):
    """
    从 extract_dataset.py 生成的辅助文件加载 ground truth 信息。

    extract_dataset.py 在 --save_chunks 模式下会在 output_dir/inputs/ 下生成
    {sample_id}_baseline.json 文件，包含 dataset, query, context 信息。
    但 ground truth (answers) 不在这些辅助文件中，需要从 JSONL 数据集获取。

    这里只读取辅助文件中的 dataset 字段，用于确认样本对应的数据集名称。

    Args:
        input_dir: 包含辅助文件的目录（或其 parent 目录下的 inputs/ 子目录）

    Returns:
        dict: {sample_id: {"dataset": "...", ...}} 仅包含基本信息
    """
    info = {}

    # 尝试在 input_dir/inputs/ 下查找辅助文件
    inputs_dir = os.path.join(input_dir, "inputs")
    if not os.path.isdir(inputs_dir):
        # 也尝试直接在 input_dir 下查找
        inputs_dir = input_dir

    baseline_files = [f for f in os.listdir(inputs_dir) if f.endswith("_baseline.json")]
    for bf in baseline_files:
        filepath = os.path.join(inputs_dir, bf)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            sample_id = data.get("sample_id", "")
            if sample_id:
                info[sample_id] = {
                    "dataset": data.get("dataset", ""),
                    "query": data.get("query", ""),
                    "context": data.get("context", ""),
                }
        except Exception as e:
            print(f"  辅助文件解析失败: {filepath}, 错误: {e}")

    return info


def main():
    parser = argparse.ArgumentParser(description="解析端侧 baseline 输出并计算精度分数")
    parser.add_argument(
        "--input_dir", type=str, required=True, help="端侧输出 .txt 文件所在目录（如 GEWU_output/baseline/test/）"
    )
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="原始 JSONL 数据集所在目录（用于查找 ground truth）"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="结果输出目录（默认: input_dir 下创建 eval_results 子目录）"
    )

    args = parser.parse_args()

    # 设置输出目录
    if args.output_dir is None:
        output_dir = os.path.join(args.input_dir, "eval_results")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 50)
    print("端侧 Baseline 输出解析与精度计算")
    print("=" * 50)
    print(f"输入目录: {args.input_dir}")
    print(f"数据集目录: {args.dataset_dir}")
    print(f"输出目录: {output_dir}")
    print()

    # 1. 加载 ground truth
    print("加载 ground truth...")
    ground_truth = load_ground_truth_from_jsonl(args.dataset_dir)

    # 2. 尝试加载辅助文件（如果有）
    auxiliary_info = load_ground_truth_from_inputs(args.input_dir)
    if auxiliary_info:
        print(f"  从辅助文件加载了 {len(auxiliary_info)} 个样本的基本信息")

    # 3. 遍历所有 .txt 文件
    txt_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".txt")])
    print(f"\n找到 {len(txt_files)} 个 .txt 文件")

    results = {}
    unmatched = []

    for txt_file in txt_files:
        txt_path = os.path.join(args.input_dir, txt_file)

        # 解析 sample_id
        sample_id = get_sample_id_from_filename(txt_file)

        if sample_id is None:
            print(f"  警告: 无法解析文件名的 sample_id: {txt_file}")
            unmatched.append(txt_file)
            continue

        # 解析模型输出
        prediction = parse_txt_file(txt_path)

        if not prediction:
            print(f"  警告: 解析结果为空: {txt_file}")
            # 仍然记录，但 prediction 为空
            prediction = ""

        # 查找 ground truth
        gt = ground_truth.get(sample_id)
        if gt is None:
            # 尝试字符串匹配（有些 sample_id 格式可能不同）
            # 比如文件名中的 sample_id 是 hex string，ground truth 中的 _id 也是
            # 先尝试直接查找，如果失败，搜索相近匹配
            found = False
            for gt_id, gt_data in ground_truth.items():
                if str(gt_id) == str(sample_id) or gt_id.endswith(sample_id) or sample_id.endswith(str(gt_id)):
                    gt = gt_data
                    found = True
                    break
            if not found:
                # 也尝试在 auxiliary_info 中查找 dataset 信息
                aux = auxiliary_info.get(sample_id)
                if aux and aux.get("dataset"):
                    print(f"  警告: 找到辅助信息但无 ground truth: {txt_file} (dataset={aux['dataset']})")
                else:
                    print(f"  警告: 未找到 ground truth: {txt_file} (sample_id={sample_id})")
                unmatched.append(txt_file)
                continue

        # 构建 result entry（与 eval_longbench.py 的 result.json 格式一致）
        sample_key = f"{gt['dataset']}_{sample_id}"
        results[sample_key] = {
            "input": gt.get("input", ""),
            "pred": prediction,
            "answers": gt.get("answers", []),
            "task": gt.get("dataset", ""),
            "idx": sample_key,
            "all_classes": gt.get("all_classes", None),
            "length": gt.get("length", 0),
        }

    # 4. 保存 result.json
    result_json_path = os.path.join(output_dir, "result.json")
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"\n结果已保存: {result_json_path}")

    # 5. 计算精度
    if not results:
        print("没有有效的结果，无法计算精度")
        return

    print("\n计算精度分数...")
    task_scores = defaultdict(list)

    for key, data in results.items():
        task = data["task"]
        prediction = data["pred"]
        ground_truths = data["answers"]
        all_classes = data.get("all_classes", None)

        # 计算样本 score
        score = scorer(task, prediction, ground_truths, all_classes)

        # 写回 JSON
        results[key]["score"] = score

        # 保存 task 层面的 score
        task_scores[task].append(score)

    # 更新 result.json（加入 score 字段）
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # 6. 计算每个 task 的平均分
    score_dict = {}
    for task, s_list in task_scores.items():
        avg_task_score = sum(s_list) / len(s_list)
        score_dict[task] = {"score": round(avg_task_score * 100, 2), "num": len(s_list)}

    score_list = [s["score"] for s in score_dict.values()]
    score_dict["avg"] = round(sum(score_list) / len(score_list), 2) if score_list else 0

    # 7. 使用 eval_longbench.py 的 write_score 保存 score.json 和 score.csv
    write_score(output_dir, score_dict)

    # 8. 打印摘要
    print("\n" + "=" * 50)
    print("精度摘要")
    print("=" * 50)
    for task, info in score_dict.items():
        if isinstance(info, dict):
            print(f"  {task}: score={info['score']}, num={info['num']}")
        else:
            print(f"  {task}: {info}")

    if unmatched:
        print(f"\n未匹配的文件 ({len(unmatched)}):")
        for f in unmatched[:10]:
            print(f"  {f}")
        if len(unmatched) > 10:
            print(f"  ... 还有 {len(unmatched) - 10} 个")

    print(f"\n输出目录: {output_dir}")
    print("完成!")


if __name__ == "__main__":
    main()
