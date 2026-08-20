"""
LongBench-Pro 评测主脚本。

对齐 eval_longbench.py 的风格(YAML 配置驱动、reranker 压缩、vLLM 异步批量推理、
本地 JSON 结果 + 评分 + CSV 汇总),适配 LongBench-Pro 数据集:
- 数据集为单个 json 文件(1500 样本,非 per-task jsonl)
- 指标按 secondary_task 分派(见 lbpro_metrics.task_metric_config)
- BoN=1 单次推理,只算 average 指标
- 用 question_nonthinking + chat template(压缩后)推理,保留 [Answer]/[答案] 标记要求

用法:
    cd experiments/evaluation
    python eval_longbench_pro.py --config ../config/longbench_pro.yaml
    python eval_longbench_pro.py --config ../config/longbench_pro.yaml --debug   # 仅前20样本
"""

import argparse
import json
import os
import sys
import csv
import copy
import asyncio
import warnings
import multiprocessing
from collections import defaultdict

import urllib3
import yaml
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
from vllm import SamplingParams

# 添加 src 目录到 sys.path,与 eval_longbench.py 一致
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.compressor import RerankCompressor, EmbeddingCompressor
from utils import setup_logging, construct_llm
from lbpro_metrics import (
    calculate_metric,
    calculate_overall_metrics,
    calculate_dimension_metrics,
    DIMENSION_CONFIG,
    task_metric_config,
)

warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
os.environ["HF_ENDPOINT"] = "https://huggingface.co"

multiprocessing.set_start_method("spawn", force=True)

# LongBench-Pro 没有统一的 max_gen 映射,这里给一个默认生成上限(答案最多 55 个 component,多行)。
DEFAULT_MAX_NEW_TOKENS = 2048


def build_components(yaml_args):
    """构建 reranker(可选) + compressor(可选,本脚本暂仅支持 rerank/null) + llm。
    与 eval_longbench.py 的 build_components 结构对齐,适配 LongBench-Pro config。
    """
    ranker = None
    if yaml_args["reranker_config"].get("model_type") == "rerank":
        ranker = RerankCompressor(
            yaml_args["reranker_config"]["model_name"],
            f"cuda:{yaml_args['reranker_config']['device_id']}",
            chunk_end_tokens=[
                "。", "！", "？", ".", "!", "?", "\n", "。\n", "？\n", "！\n",
            ],
            engine=yaml_args["reranker_config"]["engine"],
        )
        ranker.max_position_embeddings = yaml_args["reranker_config"].get("max_position_embeddings", 32768)
    elif yaml_args["reranker_config"].get("model_type") == "embedding":
        ranker = EmbeddingCompressor(
            model_name=yaml_args["reranker_config"]["model_name"],
            device=f"cuda:{yaml_args['reranker_config']['device_id']}",
            chunk_end_tokens=[
                "。", "！", "？", ".", "!", "?", "\n", "。\n", "？\n", "！\n",
            ],
        )

    # LLM 放到 reranker 之外的 device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(yaml_args["llm_config"]["device_ids"])
    llm = construct_llm(yaml_args["llm_config"])

    return ranker, llm


def load_lbpro_dataset(dataset_path, enable_test=False):
    """加载 LongBench-Pro 数据集(单个 json 文件),展平为 bon_idx=1 的样本列表。
    数据集 schema:
        id, context, language, token_length, primary_task, secondary_task,
        contextual_requirement, question_nonthinking, question_thinking, answer, difficulty
    """
    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        # 回退到 HuggingFace(若本地 json 不存在)
        ds = load_dataset("caskcsg/LongBench-Pro", split="test")
        data = list(ds)

    samples = []
    for item in data:
        samples.append({
            "bon_idx": 1,
            "id": item["id"],
            "context": item["context"],
            "language": item["language"],
            "token_length": item["token_length"],
            "primary_task": item["primary_task"],
            "secondary_task": item["secondary_task"],
            "contextual_requirement": item["contextual_requirement"],
            "question_nonthinking": item["question_nonthinking"],
            "question_thinking": item["question_thinking"],
            "answer": item["answer"],
            "difficulty": item["difficulty"],
        })

    if enable_test:
        samples = samples[:20]
    return samples


async def predict(yaml_args, json_path, enable_test=False):
    """producer/consumer + queue 异步推理,对齐 eval_longbench.py。"""
    dataset_path = yaml_args["dataset_path"]
    samples = load_lbpro_dataset(dataset_path, enable_test=enable_test)
    print(f"共加载 {len(samples)} 个样本")

    results = {}
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            results = json.load(f)

    batch_size = yaml_args["exp_config"]["batch_size"]
    max_new_tokens = yaml_args["exp_config"].get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
    max_total_token = yaml_args["exp_config"]["max_total_token"]

    tokenizer = AutoTokenizer.from_pretrained(
        yaml_args["llm_config"]["llm"]["model_name"], trust_remote_code=True
    )

    reranker, llm = build_components(yaml_args)

    queue = asyncio.Queue(maxsize=2)

    async def producer():
        for i in tqdm(range(0, len(samples), batch_size), desc="producer"):
            batch = [samples[j] for j in range(i, min(i + batch_size, len(samples)))]

            # 跳过已处理样本(断点续跑)
            filtered_batch = []
            for sample in batch:
                sample_key = f"{sample['id']}_{sample['bon_idx']}"
                if sample_key not in results:
                    filtered_batch.append(sample)
                else:
                    print(f"Skipping already processed sample: {sample_key}")

            if not filtered_batch:
                continue

            batch_prompt = []
            for sample in filtered_batch:
                max_gen = max_new_tokens

                def run_compress():
                    context = sample["context"]
                    question = sample["question_nonthinking"]
                    if reranker is not None:
                        # reranker 用 question 作为 query 做相关性打分
                        query = question if question.strip() else "Summarize the document"
                        if (yaml_args["reranker_config"].get("model_type") == "embedding"
                                and type(reranker) is EmbeddingCompressor):
                            _, select_chunks, _ = reranker.compress(
                                context, None, query,
                                run_config["reranker_config"]["chunk_size"],
                                run_config["reranker_config"]["rate"],
                                chunk_method="bypunc",
                                selection_mode=run_config["reranker_config"].get("selection_mode", "topk"),
                            )
                        elif (yaml_args["reranker_config"].get("model_type") == "rerank"
                              and type(reranker) is RerankCompressor):
                            _, select_chunks, _ = reranker.compress(
                                context, None, query,
                                run_config["reranker_config"]["chunk_size"],
                                run_config["reranker_config"]["rate"],
                                dataset=f"{sample['id']}_{sample['bon_idx']}",
                                chunk_method="bypunc",
                                selection_mode=run_config["reranker_config"].get("selection_mode", "topk"),
                                result_path=run_save_dir,
                                coverage_ratio=run_config["reranker_config"].get("coverage_ratio", 1 / 3),
                                coverage_chunk_size=run_config["reranker_config"].get("coverage_chunk_size"),
                            )
                        context = "".join(select_chunks)

                    # 压缩后用 chat template:单条 user message,内容为 context + 4 换行 + question
                    # (保留官方的 4 换行分隔,但包进 chat template,适配 Qwen3-8B-Instruct)
                    user_content = f"{context}\n\n\n\n{question}"
                    messages = [{"role": "user", "content": user_content}]
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                    # 超长截断:复用 eval_longbench.py 的前后半截断逻辑
                    token_ids = tokenizer.encode(prompt)
                    if len(token_ids) > (max_total_token - max_gen):
                        half = int((max_total_token - max_gen) / 2) - 1
                        prompt = tokenizer.decode(token_ids[:half]) + tokenizer.decode(token_ids[-half:])
                    return prompt

                prompt = await asyncio.to_thread(run_compress)
                batch_prompt.append(prompt)

            await queue.put((i, filtered_batch, batch_prompt))
        await queue.put(None)

    async def consumer():
        while True:
            item = await queue.get()
            if item is None:
                break
            i, batch, batch_prompt = item

            sampling_dict = dict(yaml_args["llm_config"]["sampling"])
            sampling_dict["max_tokens"] = max_new_tokens
            sampling_params = SamplingParams(
                temperature=sampling_dict.get("temperature", 0.0),
                max_tokens=sampling_dict.get("max_tokens", DEFAULT_MAX_NEW_TOKENS),
                top_p=sampling_dict.get("top_p", 1.0),
            )

            def run_llm():
                with torch.cuda.device("cuda:0"):
                    return llm.generate(batch_prompt, sampling_params)

            preds = await asyncio.to_thread(run_llm)

            for idx, sample in enumerate(batch):
                sample_key = f"{sample['id']}_{sample['bon_idx']}"
                pred_text = preds[idx].outputs[0].text
                # 原文 token 数(用于按 token 长度分桶;这里算压缩前原文的 token 数)
                context_tok_len = len(tokenizer.encode(sample["context"], add_special_tokens=False))
                results[sample_key] = {
                    "id": sample["id"],
                    "bon_idx": sample["bon_idx"],
                    "pred": pred_text,
                    "prediction": pred_text,  # 官方字段名,指标读取 prediction
                    "answer": sample["answer"],
                    "secondary_task": sample["secondary_task"],
                    "primary_task": sample["primary_task"],
                    "language": sample["language"],
                    "token_length": sample["token_length"],
                    "difficulty": sample["difficulty"],
                    "contextual_requirement": sample["contextual_requirement"],
                    "context_tok_len": context_tok_len,
                }

            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

    prod_task = asyncio.create_task(producer())
    cons_task = asyncio.create_task(consumer())
    await asyncio.gather(prod_task, cons_task)


def eval(json_path, embedding_model=None):
    """对结果 JSON 评分:按 secondary_task 分派指标,计算总体平均 + 各维度平均。
    BoN=1,不做 best-of-n / pass@n。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    metric_results = []  # 含 metric + 各 dimension 字段
    fail_samples_num = 0

    for key, data in results.items():
        secondary_task = data["secondary_task"]
        is_zh = data["language"] == "Chinese"
        answer = data["answer"]
        prediction = data.get("prediction", data.get("pred", ""))

        if "metric" in data:
            # 已评分(增量复跑跳过的旧样本),直接复用,不重算
            metric_value = data["metric"]
            success = True
        else:
            success, metric_value = calculate_metric(
                secondary_task, answer, prediction, is_zh, embedding_model=embedding_model
            )
            data["metric"] = metric_value
            if not success:
                fail_samples_num += 1

        metric_results.append({**data, "metric": metric_value})

    # 总体平均
    overall = calculate_overall_metrics(metric_results)

    # 各维度平均
    dimension_metrics = {}
    for dim, sort_keys in DIMENSION_CONFIG.items():
        dimension_metrics[f"average_{dim}_metric"] = calculate_dimension_metrics(
            metric_results, dim, sort_keys
        )

    summary = {
        "total_samples_num": len(metric_results),
        "fail_samples_num": fail_samples_num,
        "average_overall_metric": overall,
        **dimension_metrics,
    }

    # 写回 metric 到结果 JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    return summary


def write_score(run_save_dir, summary, score_dict_per_sample=None):
    """写 summary.json + score.csv(各维度分组),对齐 eval_longbench.py 的输出风格。"""
    # summary.json
    summary_path = os.path.join(run_save_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
    print(f"结果写入: {summary_path}")

    # score.csv:总体 + 各维度
    score_path = os.path.join(run_save_dir, "score.csv")
    with open(score_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric_type", "dimension", "value", "count"])
        writer.writerow(["overall", "-", round(summary["average_overall_metric"] * 100, 2),
                         summary["total_samples_num"]])
        for dim_key, label in [
            ("average_token_length_metric", "token_length"),
            ("average_contextual_requirement_metric", "contextual_requirement"),
            ("average_difficulty_metric", "difficulty"),
            ("average_primary_task_metric", "primary_task"),
            ("average_secondary_task_metric", "secondary_task"),
            ("average_language_metric", "language"),
        ]:
            for sub_key, value in summary.get(dim_key, {}).items():
                writer.writerow([label, sub_key, round(value * 100, 2), ""])
            writer.writerow([f"Avg ({label})", "", round(np.mean(list(summary.get(dim_key, {}).values())) * 100, 2)
                             if summary.get(dim_key) else "", ""])
    print(f"结果写入: {score_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output", default="../output")
    parser.add_argument("--only_eval", action="store_true", help="仅评分(跳过推理)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    base_exp_name = base_config["exp_config"]["name"]

    reranker_conf = base_config.get("reranker_config", {}) or {}
    has_reranker = reranker_conf.get("model_type") is not None
    chunk_sizes = reranker_conf.get("chunk_size", [None]) if has_reranker else [None]
    rerank_rates = reranker_conf.get("rate", [None]) if has_reranker else [None]

    # Summary 指标需要 embedding 模型;若配置则加载,否则 Summary 任务得 0 分
    embedding_model = None
    emb_conf = base_config.get("embedding_config")
    if emb_conf and emb_conf.get("model_name"):
        try:
            from sentence_transformers import SentenceTransformer
            emb_device_id = emb_conf.get("device_id", 0)
            emb_device = f"cuda:{emb_device_id}"
            print(f"正在加载 embedding 模型: {emb_conf['model_name']} (device={emb_device})")
            embedding_model = SentenceTransformer(
                emb_conf["model_name"],
                device=emb_device,
                tokenizer_kwargs={"padding_side": "left"},
            )
        except Exception as e:
            print(f"[main] embedding 模型加载失败,Summary 任务将得 0 分: {e}")

    # 无 reranker:单次运行
    if not has_reranker:
        run_name = base_exp_name + ("-debug" if args.debug else "")
        run_save_dir = os.path.join(args.output, run_name)
        os.makedirs(run_save_dir, exist_ok=True)

        result_json = os.path.join(run_save_dir, "result.json")
        config_yaml = os.path.join(run_save_dir, "config.yaml")
        with open(config_yaml, "w", encoding="utf-8") as f:
            yaml.dump(base_config, f, allow_unicode=True, sort_keys=False)

        if not args.only_eval:
            asyncio.run(predict(base_config, result_json, enable_test=args.debug))

        summary = eval(result_json, embedding_model=embedding_model)
        write_score(run_save_dir, summary)
        sys.exit(0)

    # 有 reranker:按 chunk_size × rate 循环运行
    for chunk_size in chunk_sizes:
        for rr in rerank_rates:
            print("-" * 20, f"chunk_size={chunk_size}, reranker rate={rr}", "-" * 20)
            run_config = copy.deepcopy(base_config)
            parts = []
            if chunk_size is not None:
                run_config["reranker_config"]["chunk_size"] = chunk_size
                run_config["reranker_config"]["rate"] = rr
                parts.append(f"chunksize-{chunk_size}-rate-{rr}")
            if args.debug:
                parts.append("debug")

            run_name = "_".join(parts) if parts else "base"
            run_save_dir = os.path.join(args.output, base_exp_name, run_name)
            os.makedirs(run_save_dir, exist_ok=True)

            result_json = os.path.join(run_save_dir, "result.json")
            config_yaml = os.path.join(run_save_dir, "config.yaml")
            with open(config_yaml, "w", encoding="utf-8") as f:
                yaml.dump(run_config, f, allow_unicode=True, sort_keys=False)

            if not args.only_eval:
                asyncio.run(predict(run_config, result_json, enable_test=args.debug))

            summary = eval(result_json, embedding_model=embedding_model)
            write_score(run_save_dir, summary)

            # reranker 实际压缩率(rate.csv)聚合
            if has_reranker and run_config["reranker_config"].get("selection_mode") in [
                "cluster", "topp", "cluster-zscore", "topk", "threshold", "pure-topk", "shunt",
            ]:
                rate_csv = os.path.join(run_save_dir, "rate.csv")
                if os.path.exists(rate_csv):
                    avg_rate_csv = os.path.join(run_save_dir, "avg_rate.csv")
                    # 简单平均(无分组,LongBench-Pro 无 task group 概念)
                    rates = []
                    with open(rate_csv, "r", newline="", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        next(reader, None)  # 表头
                        for row in reader:
                            if len(row) >= 2:
                                try:
                                    rates.append(float(row[1]))
                                except ValueError:
                                    continue
                    with open(avg_rate_csv, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(["dataset", "average_chunk_rate"])
                        writer.writerow(["Avg (Overall)", f"{np.mean(rates):.6f}" if rates else "N/A"])
                    print(f"平均压缩率结果写入: {avg_rate_csv}")
