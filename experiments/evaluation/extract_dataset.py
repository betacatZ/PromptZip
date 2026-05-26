
import argparse
import json
import os
import sys
from tqdm import tqdm
from pathlib import Path

# 添加 src 目录到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.compressor import RerankCompressor
from transformers import AutoTokenizer

# 从 eval_longbench.py 导入所需的常量
INSTRUCTION = {
    "narrativeqa": "Find passages that answer the question.",
    "qasper": "Find passages that answer the question.",
    "multifieldqa_en": "Find passages that answer the question.",
    "multifieldqa_zh": "检索有助于回答问题的相关内容。",
    "hotpotqa": "Find passages that provide evidence useful for answering the question.",
    "2wikimqa": "Find passages that provide evidence useful for answering the question.",
    "musique": "Find passages that support multi-hop reasoning for the question.",
    "dureader": "检索有助于回答问题的相关内容。",
    "gov_report": "Find passages containing key findings, policy conclusions, major facts, and important statistics essential for summarizing.",
    "qmsum": "Find transcript segments relevant to the query.",
    "multi_news": "Find passages containing the most important events, key facts, main actors, and outcomes needed for summarizing.",
    "vcsum": "检索与会议总结相关的重要内容。",
    "trec": "Determine the type of the question.",
    "samsum": "Summarize the dialogue.",
    "lsht": "判断新闻的类别。",
    "passage_count": "Count the number of unique paragraphs.",
    "passage_retrieval_en": "Determine which paragraph the abstract belongs to.",
    "passage_retrieval_zh": "判断摘要属于哪个段落。",
    "lcc": "Complete the code.",
    "repobench-p": "Complete the code.",
}


def read_input_samples(input_file):
    """
    读取jsonl文件中的样本

    Args:
        input_file: 输入的jsonl文件路径

    Returns:
        样本列表
    """
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sample = json.loads(line)
                samples.append(sample)
    return samples


def build_text_messages(instruction, query, chunk):
    """
    构建text字段的三元组，严格按照用户提供的示例格式
    """
    return [
        '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n',
        f"<|im_start|>user\n<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {chunk}<|im_end|>\n",
        "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    ]


def save_inference_json(output_dir, sample_id, chunk_idx, text_messages, json_template):
    """
    保存推理JSON文件

    Args:
        output_dir: 输出目录
        sample_id: 样本ID
        chunk_idx: 块索引
        text_messages: text字段内容
        json_template: JSON模板
    """
    # 直接在输出目录中创建文件，使用清晰的命名方式，便于后续解析
    # 文件名格式：sample_{sample_id}_chunk_{chunk_idx}.json
    os.makedirs(output_dir, exist_ok=True)

    # 创建完整的JSON
    inference_json = dict(json_template)
    inference_json["text"] = text_messages

    # 保存文件，使用清晰的命名格式
    output_file = os.path.join(output_dir, f"sample_{sample_id}_chunk_{chunk_idx}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(inference_json, f, ensure_ascii=False, indent=4)


def save_sample_chunks(output_dir, sample_id, dataset, instruction, query, chunks):
    """
    保存样本的 chunks 信息，便于后续 GPU 推理和对比

    Args:
        output_dir: 输出目录
        sample_id: 样本ID
        dataset: 数据集名称
        instruction: 指令
        query: 查询
        chunks: 文本块列表
    """
    inputs_dir = os.path.join(output_dir, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)
    
    chunk_info = {
        "sample_id": sample_id,
        "dataset": dataset,
        "instruction": instruction,
        "query": query,
        "chunks": chunks
    }
    
    output_file = os.path.join(inputs_dir, f"{sample_id}_chunks.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunk_info, f, ensure_ascii=False, indent=2)


def process_sample(sample, compressor, chunk_size, json_template, output_dir, save_chunks=True):
    """
    处理单个样本，切分文本并生成推理JSON

    Args:
        sample: 输入样本
        compressor: RerankCompressor实例
        chunk_size: 块大小
        json_template: JSON模板
        output_dir: 输出目录
        save_chunks: 是否保存 chunks 信息用于后续对比
    """
    context = sample.get("context", "")
    query = sample.get("input", "")
    dataset = sample.get("dataset", "unknown")
    _id = sample.get("_id", 0)

    # 获取合适的instruction
    instruction = INSTRUCTION[dataset]

    # 切分文本
    chunks = compressor._chunk_context(context, compressor.chunk_end_tokens, chunk_size)
    
    # 保存 chunks 信息（如果需要）
    if save_chunks:
        save_sample_chunks(output_dir, _id, dataset, instruction, query, chunks)

    # 为每个chunk生成一个JSON文件
    for chunk_idx, chunk in enumerate(chunks):
        # 构建text字段
        text_messages = build_text_messages(instruction, query, chunk)
        # 保存JSON
        save_inference_json(
            output_dir=output_dir,
            sample_id=_id,
            chunk_idx=chunk_idx,
            text_messages=text_messages,
            json_template=json_template,
        )

    return len(chunks)


def main():
    parser = argparse.ArgumentParser(description="提取数据集并构建推理JSON文件")
    parser.add_argument("--input_file", type=str, required=True, help="输入的jsonl文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Reranker-0.6B", help="模型名称")
    parser.add_argument("--chunk_size", type=int, default=256, help="块大小")
    parser.add_argument("--device", type=str, default="cpu", help="设备 (cpu/cuda)")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="/data/qwen3/qwen3-reranker-0.6b/Q4_N_0_G128/tokenizer.json",
        help="tokenizer路径",
    )
    parser.add_argument(
        "--params_path", type=str, default="/data/qwen3/qwen3-reranker-0.6b/Q4_N_0_G128/params", help="参数路径"
    )
    parser.add_argument("--max_ctx", type=int, default=512, help="最大上下文")
    parser.add_argument("--save_chunks", action="store_true", default=True, help="保存 chunks 信息用于后续对比")

    args = parser.parse_args()

    # 创建JSON模板
    json_template = {
        "tokenizer": {"model_path": args.tokenizer_path},
        "params_path": args.params_path,
        "hparams": {
            "n_vocab": 151669,
            "n_embed": 1024,
            "n_head": 16,
            "head_dim": 128,
            "n_kv_head": 8,
            "n_layer": 28,
            "n_ffn": 3072,
            "rms_norm_eps": 1e-06,
            "rope_freq_base": 1000000.0,
            "rope_freq_scale": 1.0,
            "eos_token_id": 151645,
        },
        "esets": {
            "im_type": "fp16",
            "backend": "knpu",
            "comp_on_core": [-1, -1],
            "max_ctx": args.max_ctx,
            "n_batched": 128,
            "low_mem_m": 1,
            "low_mem_n": 0,
            "low_mem_k": 0,
            "load_embed_on_request": True,
            "io_on_core": 8,
            "iobackend": "sync",
            "iouring_on_core": -1,
            "direct_io": False,
            "cache_layout": "knvt",
            "cache_grow_policy": "static",
            "cache_grow_size": 0,
            "k_cache_type": "fp16",
            "v_cache_type": "fp16",
            "snapkv_config": {"use_snapkv": False, "window_size": 32, "max_prompt_capacity": 256, "kernel_size": 5},
        },
        "text": [],
    }

    # 初始化compressor
    print("正在初始化 RerankCompressor...")
    compressor = RerankCompressor(
        model_name=args.model_name,
        device_map=args.device,
        chunk_end_tokens=["。", "！", "？", ".", "!", "?", "\n", "。\n", "？\n", "！\n"],
    )

    # 读取输入样本
    print(f"正在读取输入文件: {args.input_file}")
    samples = read_input_samples(args.input_file)
    print(f"共读取 {len(samples)} 个样本")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 处理所有样本
    total_chunks = 0
    print("开始处理样本...")
    for sample in tqdm(samples, desc="处理样本"):
        try:
            n_chunks = process_sample(
                sample=sample,
                compressor=compressor,
                chunk_size=args.chunk_size,
                json_template=json_template,
                output_dir=args.output_dir,
                save_chunks=args.save_chunks,
            )
            total_chunks += n_chunks
        except Exception as e:
            sample_id = sample.get("_id", "unknown")
            print(f"处理样本 {sample_id} 时出错: {e}")

    print(f"处理完成！共生成 {total_chunks} 个 JSON 文件")


if __name__ == "__main__":
    main()

