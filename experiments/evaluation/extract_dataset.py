import argparse
import copy
import json
import os
import sys
from tqdm import tqdm

# 添加 src 目录到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.compressor import RerankCompressor
from transformers import AutoTokenizer

# Reranker指令（仅chunk模式使用）
INSTRUCTION = {
    "narrativeqa": "Find passages that answer the question.",
    "qasper": "Find passages that answer the question.",
    "multifieldqa_en": "Find passages that answer the question.",
    "multifieldqa_zh": "检索有助于回答问题的相关内容。",
    "hotpotqa": "Find passages that provide evidence useful for answering the question.",
    "2wikimqa": "Find passages that provide evidence useful for answering the question.",
    "musique": "Find passages that support multi-hop reasoning for the question.",
    "dureader": "检索有助于回答问题的相关内容。",
    "gov_report": "Find passages containing important information for summarization.",
    "qmsum": "Find transcript segments relevant to the query.",
    "multi_news": "Find passages containing key information for summarization.",
    "vcsum": "检索与会议总结相关的重要内容。",
}

# LLM问答prompt模板（仅baseline模式使用）
dataset2prompt = {
    "narrativeqa": [
        {
            "role": "system",
            "content": "You are a helpful assistant. You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible. Do not provide any explanation.",
        },
        {"role": "user", "content": "<content>{context}</content>\n\nQuestion: {input}"},
    ],
    "qasper": [
        {
            "role": "system",
            "content": 'You are a scientific assistant. Answer the question based on the article. If the question cannot be answered, write "unanswerable". For yes/no questions, respond with "yes", "no", or "unanswerable". Do not provide any explanations.',
        },
        {"role": "user", "content": "<content>{context}</content>\n\nQuestion: {input}"},
    ],
    "multifieldqa_en": [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the question based on the given text. Only provide the answer and do not give any additional words.",
        },
        {"role": "user", "content": "<content>{context}</content>\n\nQuestion: {input}"},
    ],
    "multifieldqa_zh": [
        {
            "role": "system",
            "content": "你是一个中文问答助手。请根据文章简洁地回答问题，只提供答案，不要额外解释。",
        },
        {"role": "user", "content": "<content>{context}</content>\n\n问题：{input}"},
    ],
    "hotpotqa": [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the question based on the given passages. Only give me the answer and do not output any other words.",
        },
        {
            "role": "user",
            "content": "The following are the given passages:\n<content>{context}</content>\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}",
        },
    ],
    "2wikimqa": [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the question based on the given passages. Only give me the answer and do not output any other words.",
        },
        {
            "role": "user",
            "content": "The following are the given passages:\n<content>{context}</content>\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}",
        },
    ],
    "musique": [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the question based on the given passages. Only give me the answer and do not output any other words..",
        },
        {
            "role": "user",
            "content": "The following are the given passages:\n<content>{context}</content>\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}",
        },
    ],
    "dureader": [
        {
            "role": "system",
            "content": "你是一个中文问答助手。请根据给定文章回答问题。只给出答案，不要解释。",
        },
        {"role": "user", "content": "<content>{context}</content>\n\n问题：{input}"},
    ],
    "gov_report": [
        {
            "role": "system",
            "content": "You are a helpful assistant. You are given a report by a government agency. Write a one-page summary of the report.",
        },
        {"role": "user", "content": "<content>{context}</content>\n\nSummary:"},
    ],
    "qmsum": [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the query based on the meeting transcript in one or more sentences.",
        },
        {"role": "user", "content": "<content>{context}</content>\n\nQuery: {input}"},
    ],
    "multi_news": [
        {
            "role": "system",
            "content": "You are a helpful assistant. Read the news passages and write a one-page summary. Do not provide additional explanations.",
        },
        {"role": "user", "content": "<content>{context}</content>\n\nPlease provide the summary below."},
    ],
    "vcsum": [
        {
            "role": "system",
            "content": "你是会议总结助手。请根据会议记录写一段总结，不要解释。",
        },
        {"role": "user", "content": "<content>{context}</content>\n\n会议总结："},
    ],
    "trec": [
        {
            "role": "system",
            "content": "You are a classification assistant. Determine the type of the question based on the given examples. Only output the type/category.",
        },
        {"role": "user", "content": "<content>{context}</content>\n\nQuestion: {input}"},
    ],
    "triviaqa": [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the question based on the given passage. Only give the answer and do not output any other words.",
        },
        {"role": "user", "content": "<content>{context}</content>\n\n{input}"},
    ],
    "samsum": [
        {
            "role": "system",
            "content": "You are a helpful assistant. Summarize the dialogue into a few short sentences. Do not provide explanations or extra words.",
        },
        {"role": "user", "content": "<content>{context}</content>\n\n{input}"},
    ],
    "lsht": [
        {
            "role": "system",
            "content": "你是中文分类助手。根据给定的新闻内容，判断新闻的类别。只输出类别名称。",
        },
        {"role": "user", "content": "<content>{context}</content>\n\n{input}"},
    ],
    "passage_count": [
        {
            "role": "system",
            "content": "You are a helpful assistant. Count how many unique paragraphs there are in the given set of paragraphs after removing duplicates. Only output the number.",
        },
        {"role": "user", "content": "<content>{context}</content>"},
    ],
    "passage_retrieval_en": [
        {
            "role": "system",
            "content": "You are a helpful assistant. Determine which paragraph the abstract belongs to. Only output the number of the paragraph. The answer format must be like 'Paragraph 1', 'Paragraph 2', etc.",
        },
        {
            "role": "user",
            "content": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like 'Paragraph 1', 'Paragraph 2', etc.\n\nThe answer is: ",
        },
    ],
    "passage_retrieval_zh": [
        {
            "role": "system",
            'content': '你是中文阅读助手。请根据摘要判断该段落属于哪个段落，答案格式为"段落1"、"段落2"',
        },
        {
            "role": "user",
            'content': '{context}\n\n下面是一个摘要:\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是"段落1"、"段落2"等格式\n\n答案是：',
        },
    ],
    "lcc": [
        {
            "role": "system",
            "content": "You are a coding assistant. Complete the code below accurately. Only output the next line of code without explanation.",
        },
        {"role": "user", "content": "{context}\nNext line of code:\n"},
    ],
    "repobench-p": [
        {
            "role": "system",
            "content": "You are a coding assistant. Complete the code below accurately based on the context. Only output the next line of code without explanation.",
        },
        {"role": "user", "content": "{context}{input}\nNext line of code:\n"},
    ],
}

dataset2maxlen = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64,
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


def build_text_messages(instruction, query, document):
    """
    构建chunk模式（reranker）的text字段三元组

    Args:
        instruction: 指令文本
        query: 查询文本
        document: 文档文本（chunk）

    Returns:
        list[str]: 由三段文本组成的三元组，用于推理引擎的 text 字段
    """
    return [
        '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n',
        f"<|im_start|>user\n<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}<|im_end|>\n",
        "<|im_start|>assistant\n<tool_call>\n\nRep\n\n",
    ]


def build_llm_text_messages(dataset, context, query):
    """
    构建baseline模式（LLM问答）的text字段，使用dataset2prompt模板

    Args:
        dataset: 数据集名称
        context: 文档文本（可能已截断）
        query: 查询文本

    Returns:
        list[str]: 用于推理引擎的 text 字段
    """
    messages = copy.deepcopy(dataset2prompt[dataset])
    # 对不同数据集格式化content
    if dataset in ["gov_report", "multi_news", "vcsum"]:
        messages[1]["content"] = messages[1]["content"].format(context=context)
    else:
        messages[1]["content"] = messages[1]["content"].format(context=context, input=query)

    # 拼接为 <|im_start|>...<|im_end|>\n 格式的字符串列表
    text_parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        text_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    text_parts.append("<|im_start|>assistant\n")
    return text_parts


def save_inference_json(output_dir, sample_id, chunk_idx, text_messages, json_template):
    """
    保存推理JSON文件（chunk模式）

    Args:
        output_dir: 输出目录
        sample_id: 样本ID
        chunk_idx: 块索引
        text_messages: text字段内容
        json_template: JSON模板
    """
    os.makedirs(output_dir, exist_ok=True)

    # 创建完整的JSON
    inference_json = dict(json_template)
    inference_json["text"] = text_messages

    # 保存文件，文件名格式：sample_{sample_id}_chunk_{chunk_idx}.json
    output_file = os.path.join(output_dir, f"sample_{sample_id}_chunk_{chunk_idx}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(inference_json, f, ensure_ascii=False, indent=4)


def save_baseline_json(output_dir, sample_id, text_messages, json_template):
    """
    保存推理JSON文件（baseline模式，完整文档不切分）

    Args:
        output_dir: 输出目录
        sample_id: 样本ID
        text_messages: text字段内容
        json_template: JSON模板
    """
    os.makedirs(output_dir, exist_ok=True)

    # 创建完整的JSON
    inference_json = dict(json_template)
    inference_json["text"] = text_messages

    # 保存文件，文件名格式：sample_{sample_id}_baseline.json
    output_file = os.path.join(output_dir, f"sample_{sample_id}_baseline.json")
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
        "chunks": chunks,
    }

    output_file = os.path.join(inputs_dir, f"{sample_id}_chunks.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunk_info, f, ensure_ascii=False, indent=2)


def save_sample_baseline(output_dir, sample_id, dataset, instruction, query, context):
    """
    保存样本的完整文档信息（baseline模式），便于后续 GPU 推理和对比

    Args:
        output_dir: 输出目录
        sample_id: 样本ID
        dataset: 数据集名称
        instruction: 指令
        query: 查询
        context: 完整文档文本
    """
    inputs_dir = os.path.join(output_dir, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)

    baseline_info = {
        "sample_id": sample_id,
        "dataset": dataset,
        "instruction": instruction,
        "query": query,
        "context": context,
    }

    output_file = os.path.join(inputs_dir, f"{sample_id}_baseline.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(baseline_info, f, ensure_ascii=False, indent=2)


def process_sample_chunk(sample, compressor, chunk_size, json_template, output_dir, save_chunks=True):
    """
    处理单个样本（chunk模式），切分文本并生成推理JSON

    Args:
        sample: 输入样本
        compressor: RerankCompressor实例
        chunk_size: 块大小
        json_template: JSON模板
        output_dir: 输出目录
        save_chunks: 是否保存 chunks 信息用于后续对比

    Returns:
        int: 生成的 chunk 数量
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


def process_sample_baseline(sample, json_template, output_dir, tokenizer, max_ctx, max_gen, save_baseline=True):
    """
    处理单个样本（baseline模式），传入完整文档不切分，生成推理JSON
    当文档token数超出(max_ctx - max_gen)限制时，保留前半和后半（中间截断）

    Args:
        sample: 输入样本
        json_template: JSON模板（Qwen2.5-7b-instruct）
        output_dir: 输出目录
        tokenizer: tokenizer实例，用于计算token数和截断
        max_ctx: 最大上下文token数限制
        max_gen: 最大生成token数（各数据集不同）
        save_baseline: 是否保存完整文档信息用于后续对比

    Returns:
        int: 固定返回 1（每个样本生成一个 baseline JSON）
    """
    context = sample.get("context", "")
    query = sample.get("input", "")
    dataset = sample.get("dataset", "unknown")
    _id = sample.get("_id", 0)

    # 截断超长context：保留前半和后半，与eval_longbench.py的做法一致
    # 可用空间 = max_ctx - max_gen
    token_ids = tokenizer.encode(context)
    if len(token_ids) > (max_ctx - max_gen):
        half = int((max_ctx - max_gen) / 2) - 1
        context = tokenizer.decode(token_ids[:half]) + tokenizer.decode(token_ids[-half:])
        print(
            f"  样本 {_id}: 原文 {len(token_ids)} tokens, 截断至 ~{2 * half} tokens (max_ctx={max_ctx}, max_gen={max_gen})"
        )

    # 保存完整文档信息（如果需要）
    if save_baseline:
        save_sample_baseline(output_dir, _id, dataset, "", query, context)

    # 构建text字段，使用LLM问答prompt模板
    text_messages = build_llm_text_messages(dataset, context, query)
    # 保存JSON
    save_baseline_json(
        output_dir=output_dir,
        sample_id=_id,
        text_messages=text_messages,
        json_template=json_template,
    )

    return 1


def build_chunk_json_template(args):
    """构建chunk模式（reranker）的JSON模板"""
    return {
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


def build_baseline_json_template(args):
    """构建baseline模式（LLM问答）的JSON模板"""
    return {
        "tokenizer": {"model_path": args.baseline_tokenizer_path},
        "params_path": args.baseline_params_path,
        "hparams": {
            "n_vocab": 152064,
            "n_embed": 3584,
            "n_head": 28,
            "head_dim": 128,
            "n_kv_head": 4,
            "n_layer": 28,
            "n_ffn": 18944,
            "rms_norm_eps": 1e-06,
            "rope_freq_base": 1000000.0,
            "rope_freq_scale": 1.0,
            "eos_token_id": 151645,
        },
        "esets": {
            "im_type": "fp16",
            "backend": "cpu",
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
            "cache_persist_policy": "full",
            "k_cache_type": "fp16",
            "v_cache_type": "fp16",
            "snapkv_config": {"use_snapkv": False, "window_size": 32, "max_prompt_capacity": 256, "kernel_size": 5},
        },
        "lora": {"type": "none", "path": "", "layers_selected": []},
        "text": [],
    }


def main():
    parser = argparse.ArgumentParser(description="提取数据集并构建推理JSON文件")
    parser.add_argument("--input_file", type=str, required=True, help="输入的jsonl文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--mode", type=str, choices=["chunk", "baseline"], default="chunk",
        help="运行模式：chunk=切分文档后逐块推理(reranker)；baseline=传入完整文档(LLM问答)，用于基线测试")

    # chunk模式参数
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Reranker-0.6B", help="模型名称（chunk模式用reranker，baseline模式用LLM）")
    parser.add_argument("--chunk_size", type=int, default=256, help="块大小（仅chunk模式生效）")
    parser.add_argument("--device", type=str, default="cpu", help="设备 (cpu/cuda)")
    parser.add_argument("--tokenizer_path", type=str,
        default="/data/qwen3/qwen3-reranker-0.6b/Q4_N_0_G128/tokenizer.json",
        help="tokenizer路径（chunk模式）")
    parser.add_argument("--params_path", type=str,
        default="/data/qwen3/qwen3-reranker-0.6b/Q4_N_0_G128/params",
        help="参数路径（chunk模式）")

    # baseline模式参数
    parser.add_argument("--baseline_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="baseline模式的LLM模型名称，用于tokenizer截断")
    parser.add_argument("--baseline_tokenizer_path", type=str,
        default="/data2/llm/qwen2.5-7b-instruct/Q4_0/tokenizer.json",
        help="tokenizer路径（baseline模式）")
    parser.add_argument("--baseline_params_path", type=str,
        default="/data2/llm/qwen2.5-7b-instruct/Q4_0/params",
        help="参数路径（baseline模式）")

    # 共用参数
    parser.add_argument("--max_ctx", type=int, default=512, help="最大上下文token数")
    parser.add_argument("--save_chunks", action="store_true", help="保存 chunks/baseline 信息用于后续对比")

    args = parser.parse_args()

    # 根据mode构建不同的JSON模板
    if args.mode == "chunk":
        json_template = build_chunk_json_template(args)
    else:
        json_template = build_baseline_json_template(args)

    # 初始化模式所需组件
    compressor = None
    tokenizer = None
    if args.mode == "chunk":
        print("正在初始化 RerankCompressor...")
        compressor = RerankCompressor(
            model_name=args.model_name,
            device_map=args.device,
            chunk_end_tokens=["。", "！", "？", ".", "!", "?", "\n", "。\n", "？\n", "！\n"],
        )
    elif args.mode == "baseline":
        print("正在初始化 tokenizer（baseline模式用于截断超长文档）...")
        tokenizer = AutoTokenizer.from_pretrained(args.baseline_model_name)

    # 读取输入样本
    print(f"正在读取输入文件: {args.input_file}")
    samples = read_input_samples(args.input_file)
    print(f"共读取 {len(samples)} 个样本")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 处理所有样本
    total_outputs = 0
    print(f"运行模式: {args.mode}")
    print("开始处理样本...")
    for sample in tqdm(samples, desc="处理样本"):
        try:
            if args.mode == "chunk":
                n = process_sample_chunk(
                    sample=sample,
                    compressor=compressor,
                    chunk_size=args.chunk_size,
                    json_template=json_template,
                    output_dir=args.output_dir,
                    save_chunks=args.save_chunks,
                )
            else:
                n = process_sample_baseline(
                    sample=sample,
                    json_template=json_template,
                    output_dir=args.output_dir,
                    tokenizer=tokenizer,
                    max_ctx=args.max_ctx,
                    max_gen=dataset2maxlen[sample.get("dataset", "unknown")],
                    save_baseline=args.save_chunks,
                )
            total_outputs += n
        except Exception as e:
            sample_id = sample.get("_id", "unknown")
            print(f"处理样本 {sample_id} 时出错: {e}")

    mode_label = "chunks" if args.mode == "chunk" else "baseline JSONs"
    print(f"处理完成！共生成 {total_outputs} 个 {mode_label}")


if __name__ == "__main__":
    main()