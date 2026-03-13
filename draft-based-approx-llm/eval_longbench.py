import argparse
import json
import sys
import os
import urllib3
import warnings
import csv
from pathlib import Path
from typing import Any, cast

import yaml
from datasets import load_dataset
from collections import defaultdict
import numpy as np
from draft_approx_llm import SpecPCConfig, patch_model

from dataset.longbench.metrics import (
    classification_score,
    count_score,
    retrieval_score,
    retrieval_zh_score,
    code_sim_score,
    qa_f1_score,
    qa_f1_zh_score,
    rouge_score,
    rouge_zh_score,
)


from tqdm import tqdm
import multiprocessing
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
os.environ["HF_ENDPOINT"] = "https://huggingface.co"

multiprocessing.set_start_method("spawn", force=True)

# MARK dataset
dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "trec": classification_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


# dataset2prompt = {
#     "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
#     "qasper": 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:',
#     "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
#     "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
#     "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
#     "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
#     "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
#     "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
#     "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
#     "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
#     "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
#     "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
#     "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
#     "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
#     "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
#     "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
#     "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
#     "passage_retrieval_en": 'Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\nThe answer is: ',
#     "passage_retrieval_zh": '以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是"段落1"，"段落2"等格式\n\n答案是：',
#     "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
#     "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
# }

dataset2prompt = {
    "narrativeqa": "<|im_start|>system\nYou are a helpful assistant. You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n<|im_end|>\n\n<|im_start|>user\n<context>\n{context}\n</context>\n\nQuestion: {input}\n<|im_end|>\n\n<|im_start|>assistant\n",
    "qasper": "<|im_start|>system\nYou are a scientific assistant. Answer the question based on the article. If the question cannot be answered, write 'unanswerable'. For yes/no questions, respond with 'yes', 'no', or 'unanswerable'. Do not provide any explanations.\n<|im_end|>\n\n<|im_start|>user\n<context>\n{context}\n</context>\n\nQuestion: {input}\n<|im_end|>\n\n<|im_start|>assistant\n",
    "multifieldqa_en": "<|im_start|>system\nYou are a helpful assistant. Answer the question based on the given text. Only provide the answer and do not give any additional words.\n<|im_end|>\n\n<|im_start|>user\n<context>\n{context}\n</context>\n\nQuestion: {input}\n<|im_end|>\n\n<|im_start|>assistant\n",
    "multifieldqa_zh": "<|im_start|>system\n你是一个中文问答助手。请根据文章简洁地回答问题，只提供答案，不要额外解释。\n<|im_end|>\n\n<|im_start|>user\n<context>\n{context}\n</context>\n\n问题：{input}\n<|im_end|>\n\n<|im_start|>assistant\n",
    "hotpotqa": "<|im_start|>system\nYou are a helpful assistant. Answer the question based on the given passages. Only give me the answer and do not output any other words.\n<|im_end|>\n\n<|im_start|>user\nThe following are the given passages:\n<context>\n{context}\n</context>\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\n<|im_end|>\n\n<|im_start|>assistant\n",
    "2wikimqa": "<|im_start|>system\nYou are a helpful assistant. Answer the question based on the given passages. Only give me the answer and do not output any other words.\n<|im_end|>\n\n<|im_start|>user\nThe following are the given passages:\n<context>\n{context}\n</context>\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\n<|im_end|>\n\n<|im_start|>assistant\n",
    "musique": "<|im_start|>system\nYou are a helpful assistant. Answer the question based on the given passages. Only give me the answer and do not output any other words..\n<|im_end|>\n\n<|im_start|>user\nThe following are the given passages:\n<context>\n{context}\n</context>\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\n<|im_end|>\n\n<|im_start|>assistant\n",
    "dureader": "<|im_start|>system\n你是一个中文问答助手。请根据给定文章回答问题。只给出答案，不要解释。\n<|im_end|>\n\n<|im_start|>user\n<context>\n{context}\n</context>\n\n问题：{input}\n<|im_end|>\n\n<|im_start|>assistant\n",
    "gov_report": "<|im_start|>system\nYou are a helpful assistant. You are given a report by a government agency. Write a one-page summary of the report.\n<|im_end|>\n\n<|im_start|>user\n<context>\n{context}\n</context>\n\nSummary:\n<|im_end|>\n\n<|im_start|>assistant\n",
    "qmsum": "<|im_start|>system\nYou are a helpful assistant. Answer the query based on the meeting transcript in one or more sentences.\n<|im_end|>\n\n<|im_start|>user\n<context>\n{context}\n</context>\n\nQuery: {input}\n<|im_end|>\n\n<|im_start|>assistant\n",
    "multi_news": "<|im_start|>system\nYou are a helpful assistant. Read the news passages and write a one-page summary. Do not provide additional explanations.\n<|im_end|>\n\n<|im_start|>user\n<context>\n{context}\n</context>\n\nPlease provide the summary below.\n<|im_end|>\n\n<|im_start|>assistant\n",
    "vcsum": "<|im_start|>system\n你是会议总结助手。请根据会议记录写一段总结，不要解释。\n<|im_end|>\n\n<|im_start|>user\n<context>\n{context}\n</context>\n\n会议总结：\n<|im_end|>\n\n<|im_start|>assistant\n",
    "trec": "<|im_start|>system\nYou are a classification assistant. Determine the type of the question based on the given examples. Only output the type/category.\n<|im_end|>\n\n<|im_start|>user\n<context>\n{context}\n</context>\n\nQuestion: {input}\n<|im_end|>\n\n<|im_start|>assistant\n",
    "triviaqa": "<|im_start|>system\nYou are a helpful assistant. Answer the question based on the given passage. Only give the answer and do not output any other words.\n<|im_end|>\n\n<|im_start|>user\n{context}\n\n{input}\n<|im_end|>\n\n<|im_start|>assistant\n",
    "samsum": "<|im_start|>system\nYou are a helpful assistant. Summarize the dialogue into a few short sentences. Do not provide explanations or extra words.\n<|im_end|>\n\n<|im_start|>user\n{context}\n\n{input}\n<|im_end|>\n\n<|im_start|>assistant\n",
    "lsht": "<|im_start|>system\n你是中文分类助手。根据给定的新闻内容，判断新闻的类别。只输出类别名称。\n<|im_end|>\n\n<|im_start|>user\n{context}\n\n{input}\n<|im_end|>\n\n<|im_start|>assistant\n",
    "passage_count": "<|im_start|>system\nYou are a helpful assistant. Count how many unique paragraphs there are in the given set of paragraphs after removing duplicates. Only output the number.\n<|im_end|>\n\n<|im_start|>user\n{context}\n<|im_end|>\n\n<|im_start|>assistant\n",
    "passage_retrieval_en": "<|im_start|>system\nYou are a helpful assistant. Determine which paragraph the abstract belongs to. Only output the number of the paragraph. The answer format must be like 'Paragraph 1', 'Paragraph 2', etc.\n<|im_end|>\n\n<|im_start|>user\nHere are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like 'Paragraph 1', 'Paragraph 2', etc.\n\nThe answer is: \n<|im_end|>\n\n<|im_start|>assistant\n",
    "passage_retrieval_zh": "<|im_start|>system\n你是中文阅读助手。请根据摘要判断该段落属于哪个段落，答案格式为“段落1”、“段落2”\n<|im_end|>\n\n<|im_start|>user\n{context}\n\n下面是一个摘要:\n{input}\n<|im_end|>\n\n<|im_start|>assistant\n",
    "lcc": "<|im_start|>system\nYou are a coding assistant. Complete the code below accurately. Only output the next line of code without explanation.\n<|im_end|>\n\n<|im_start|>user\n{context}\nNext line of code:\n<|im_end|>\n\n<|im_start|>assistant\n",
    "repobench-p": "<|im_start|>system\nYou are a coding assistant. Complete the code below accurately based on the context. Only output the next line of code without explanation.\n<|im_end|>\n\n<|im_start|>user\n{context}{input}\nNext line of code:\n<|im_end|>\n\n<|im_start|>assistant\n",
}


QUESTIONS = {
    "narrativeqa": "Now, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": 'Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:',
    "multifieldqa_en": "Now, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "gov_report": "Now, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "Now, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "Now, write a one-page summary of all the news.\n\nSummary:",
    "trec": "{input}",
    "triviaqa": "{input}",
    "samsum": "{input}",
    "passage_count": "Please enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": 'The following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\nThe answer is: ',
    "lcc": "Next line of code:\n",
    "repobench-p": "{input}Next line of code:\n",
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


def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for prediction, ground_truths, length in zip(predictions, answers, lengths):
        score = 0.0
        if dataset in ["triviaqa", "samsum"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(
                score,
                dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes),
            )
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    return {key: round(100 * np.mean(values), 2) for key, values in scores.items()}


def scorer(dataset, prediction, ground_truths, all_classes):
    """计算单个样本的 score"""
    # 特定数据集需要清理 prediction
    if dataset in [
        "triviaqa",
        "samsum",
        "narrativeqa",
        "qasper",
        "multifieldqa_en",
        "multifieldqa_zh",
        "hotpotqa",
        "2wikimqa",
        "musique",
        "dureader",
        "vcsum",
    ]:
        prediction = prediction.lstrip("\n").split("\n")[0]

    score = 0.0
    for ground_truth in ground_truths:
        score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
    return score


def eval(json_path):
    # 读取 JSON
    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # 按 task 分类收集每个 task 的 scores
    task_scores = defaultdict(list)

    # 遍历每个样本
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

    # 计算每个 task 的平均分
    scores = {}
    for task, s_list in task_scores.items():
        avg_task_score = sum(s_list) / len(s_list)
        scores[task] = {"score": round(avg_task_score * 100, 2), "num": len(s_list)}

    score_list = [s["score"] for s in scores.values()]

    # 总体平均分
    scores["avg"] = round((sum(score_list) / len(score_list)), 2)

    # out_path = None
    # # 保存更新后的 JSON
    # if out_path is None:
    #     base, ext = os.path.splitext(json_path)
    #     out_path = base + "_with_scores.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    return scores


# def scorer(dataset, predictions, answers, all_classes):
#     total_score = 0.0
#     for prediction, ground_truths in zip(predictions, answers):
#         score = 0.0
#         if dataset in [
#             "triviaqa",
#             "samsum",
#             "narrativeqa",
#             "qasper",
#             "multifieldqa_en",
#             "multifieldqa_zh",
#             "hotpotqa",
#             "2wikimqa",
#             "musique",
#             "dureader",
#             "vcsum",
#         ]:
#             prediction = prediction.lstrip("\n").split("\n")[0]
#         for ground_truth in ground_truths:
#             score = max(
#                 score,
#                 dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes),
#             )
#         total_score += score
#     print(f"datasets:{dataset} ,score: {round(100 * total_score / len(predictions), 2)}")
#     return round(100 * total_score / len(predictions), 2)


# def eval(json_path):
#     results = {}
#     if os.path.exists(json_path):
#         with open(json_path, "r", encoding="utf-8") as f:
#             results = json.load(f)

#     predictions, answers, lengths = (
#         defaultdict(list),
#         defaultdict(list),
#         defaultdict(list),
#     )

#     all_classes = {}
#     for key, data in results.items():
#         predictions[data["task"]].append(data["pred"])
#         answers[data["task"]].append(data["answers"])
#         all_classes[data["task"]] = data["all_classes"]
#         if "length" in data:
#             lengths[data["task"]].append(data["length"])
#     scores = {}
#     for task in predictions.keys():
#         pred_list, ans_list, length_list = (
#             predictions[task],
#             answers[task],
#             lengths[task],
#         )
#         score = scorer(task, pred_list, ans_list, all_classes[task])
#         scores[task] = {"score": score, "num": len(pred_list)}
#     score_list = [s["score"] for s in scores.values()]
#     scores["avg"] = sum(score_list) / len(score_list)
#     return scores


def load_model(cfg, tokenizer):
    cfg = {**cfg}
    model_name_or_path = cfg.pop("model_name_or_path")
    dtype = cfg.pop("dtype", "auto")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        **cfg,
    )
    generation_config = cast(Any, model.generation_config)
    generation_config.pad_token_id = (
        tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    )
    return model


def build_components(yaml_args):
    tokenizer = AutoTokenizer.from_pretrained(yaml_args["target_model"]["model_name_or_path"], trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    draft_model = load_model(yaml_args["draft_model"], tokenizer)
    target_model = load_model(yaml_args["target_model"], tokenizer)

    specpc_config = SpecPCConfig(
        max_capacity_prompt=1024,
        window_size=64,
        pool_type="max",
        kernel_size=64,
        reduction_type="max",
        lookahead_tokens=1,
        neighbor_tokens=64,
        starting_layer_index=8,
        weighted_query=True,
    )
    return tokenizer, target_model, draft_model, specpc_config


def resolve_tasks(yaml_args):
    task = yaml_args.get("dataset", {}).get("task")
    if task is None:
        return list(dataset2metric.keys())
    if isinstance(task, list):
        return task
    return [task]


def resolve_batch_size(yaml_args):
    return yaml_args.get("batch_size", yaml_args.get("exp_config", {}).get("batch_size", 1))


def resolve_max_total_token(yaml_args, tokenizer):
    explicit = yaml_args.get("max_total_token", yaml_args.get("exp_config", {}).get("max_total_token"))
    if explicit is not None:
        return int(explicit)

    model_max_length = getattr(tokenizer, "model_max_length", None)
    if isinstance(model_max_length, int) and 0 < model_max_length < 10**7:
        return model_max_length

    return 32768


def resolve_sampling(yaml_args):
    sampling_cfg = yaml_args.get("sampling", yaml_args.get("llm_config", {}).get("sampling", {}))
    return {
        "temperature": sampling_cfg.get("temperature", 0.0),
        "top_p": sampling_cfg.get("top_p", 1.0),
    }


def resolve_specpc_rate(yaml_args):
    sparse_cfg = yaml_args.get("sparse_config", {})
    rate = sparse_cfg.get("rate", yaml_args.get("rate"))
    if rate is None:
        raise ValueError("Missing SpecPC rate. Please set sparse_config.rate in yaml.")

    rate = float(rate)
    if not (0 < rate <= 1):
        raise ValueError(f"Invalid sparse_config.rate={rate}. Expected 0 < rate <= 1.")
    return rate


def resolve_dataset_dir(yaml_args):
    dataset_cfg = yaml_args.get("dataset", {})
    return yaml_args.get("dataset_dir", dataset_cfg.get("dataset_dir", dataset_cfg.get("data_dir")))


def load_longbench_dataset(yaml_args, task):
    dataset_dir = resolve_dataset_dir(yaml_args)
    if dataset_dir:
        dataset_path = os.path.join(dataset_dir, f"{task}.jsonl")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"LongBench dataset file not found: {dataset_path}")
        return cast(
            Any,
            load_dataset(
                "json",
                data_files={"test": dataset_path},
                split="test",
            ),
        )

    return cast(Any, load_dataset("THUDM/LongBench", task, split="test", trust_remote_code=True))


def get_model_device(model):
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def generate_batch(target_model, draft_model, specpc_config, tokenizer, prompts, max_new_tokens, sampling_cfg, rate):
    preds = []
    temperature = sampling_cfg.get("temperature", 0.0)

    for prompt in prompts:
        inputs = tokenizer([prompt], padding=True, return_tensors="pt")
        input_len = int(inputs["attention_mask"].sum().item())
        specpc_config.max_capacity_prompt = max(1, int(rate * input_len))

        model = patch_model(target_model, draft_model, specpc_config)
        device = get_model_device(model)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        generate_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "return_dict_in_generate": True,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if temperature and temperature > 0:
            generate_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": sampling_cfg.get("top_p", 1.0),
                }
            )
        else:
            generate_kwargs.update(
                {
                    "do_sample": False,
                    "temperature": None,
                    "top_p": None,
                }
            )

        outputs = model.generate(**generate_kwargs)
        if hasattr(outputs, "output_ids"):
            pred = tokenizer.decode(outputs.output_ids[0], skip_special_tokens=True)
        else:
            input_len = int(inputs["attention_mask"].sum().item())
            pred = tokenizer.decode(outputs.sequences[0][input_len:], skip_special_tokens=True)
        preds.append(pred)

    return preds


def predict(yaml_args, json_path, enable_test=False):
    tasks = resolve_tasks(yaml_args)
    tokenizer, target_model, draft_model, specpc_config = build_components(yaml_args)
    batch_size = resolve_batch_size(yaml_args)
    max_total_token = resolve_max_total_token(yaml_args, tokenizer)
    sampling_cfg = resolve_sampling(yaml_args)
    specpc_rate = resolve_specpc_rate(yaml_args)

    results = {}
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            results = json.load(f)

    for task in tasks:
        dataset = load_longbench_dataset(yaml_args, task)
        if enable_test:
            dataset = dataset.select(range(min(20, len(dataset))))

        print(len(dataset))
        print("Task: ", task)

        for start in tqdm(range(0, len(dataset), batch_size)):
            batch = [cast(dict[str, Any], dataset[idx]) for idx in range(start, min(start + batch_size, len(dataset)))]

            filtered_batch = []
            batch_prompt = []
            for sample_idx, sample in enumerate(batch):
                sample_key = f"{sample['dataset']}_{sample.get('_id', start + sample_idx)}"
                if sample_key in results:
                    print(f"Skipping already processed sample: {sample_key}")
                    continue

                prompt = dataset2prompt[sample["dataset"]].format(
                    context=sample["context"],
                    input=sample["input"],
                )
                max_gen = int(dataset2maxlen[sample["dataset"]])
                token_ids = tokenizer(prompt, padding=False, return_attention_mask=False)["input_ids"]
                if len(token_ids) > (max_total_token - max_gen):
                    half = int((max_total_token - max_gen) / 2) - 1
                    prompt = tokenizer.decode(token_ids[:half]) + tokenizer.decode(token_ids[-half:])

                filtered_batch.append((sample_key, sample))
                batch_prompt.append(prompt)

            if not filtered_batch:
                continue

            preds = generate_batch(
                target_model,
                draft_model,
                specpc_config,
                tokenizer,
                batch_prompt,
                max_new_tokens=int(dataset2maxlen[filtered_batch[0][1]["dataset"]]),
                sampling_cfg=sampling_cfg,
                rate=specpc_rate,
            )

            for (sample_key, sample), pred in zip(filtered_batch, preds):
                results[sample_key] = {
                    "input": sample["input"],
                    "pred": pred,
                    "answers": sample["answers"],
                    "task": sample["dataset"],
                    "idx": sample_key,
                    "all_classes": sample["all_classes"],
                    "length": sample["length"],
                }

            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)


def compute_rate_averages(file_path, output_path, rate_column="chunk_rate"):
    """
    从 CSV 文件读取 dataset 列的前缀（如 narrativeqa_xxx）并按分组计算平均值，
    每组写入 CSV 后带组均值，最后写整体平均值。

    Args:
        file_path (str): 输入 CSV 文件路径，必须包含 'dataset' 列。
        output_path (str): 输出 CSV 文件路径。
        rate_column (str): 要计算均值的列名，默认 'chunk_rate'。
    """

    # 读取 CSV 并聚合
    dataset_values = {}
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "dataset" not in fieldnames:
            raise ValueError("输入文件必须包含列名 'dataset'。")
        if rate_column not in fieldnames:
            raise ValueError(f"输入文件必须包含列名 '{rate_column}'。")

        for row in reader:
            ds_full = row["dataset"].strip()
            if "_" not in ds_full:
                continue
            ds_prefix = ds_full.rsplit("_", 1)[0]
            try:
                value = float(row[rate_column])
            except ValueError:
                continue
            dataset_values.setdefault(ds_prefix, []).append(value)

    # 计算每个数据集均值
    dataset_avg = {ds: np.mean(vals) for ds, vals in dataset_values.items()}

    # 定义分组（可根据需求修改）
    groups = [
        ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh"],
        ["hotpotqa", "2wikimqa", "musique", "dureader"],
        ["gov_report", "qmsum", "multi_news", "vcsum"],
        ["triviaqa", "samsum", "trec", "lsht"],
        ["passage_retrieval_en", "passage_count", "passage_retrieval_zh"],
        ["lcc", "repobench-p"],
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "average_chunk_rate"])

        # 按组写入每个数据集和组均值
        for group in groups:
            group_scores = []
            for ds in group:
                if ds in dataset_avg:
                    avg = dataset_avg[ds]
                    writer.writerow([ds, f"{avg:.6f}"])
                    group_scores.append(avg)
                else:
                    writer.writerow([ds, "N/A"])
            if group_scores:
                group_avg = np.mean(group_scores)
                writer.writerow(["Avg", f"{group_avg:.6f}"])
            else:
                writer.writerow(["Avg", "N/A"])

        # 整体平均值
        all_values = [v for vals in dataset_values.values() for v in vals]
        overall_avg = np.mean(all_values) if all_values else None
        writer.writerow(["Avg (Overall)", f"{overall_avg:.6f}" if overall_avg is not None else "N/A"])

    print(f"平均压缩率结果写入: {output_path}")


def write_score(run_save_dir, score_dict):
    """
    将分数数据写入 JSON 和 CSV 文件，并计算每个指定分组的均值。

    Args:
        run_save_dir (str): 结果保存的目录路径。
        score_dict (dict): 包含每个数据集得分的字典，格式为 {dataset: {"score": score, "num": num}}。
    """

    # 分组定义
    groups = [
        ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh"],  # 第0到3组（4个）
        ["hotpotqa", "2wikimqa", "musique", "dureader"],  # 第4到8组（4个）
        ["gov_report", "qmsum", "multi_news", "vcsum"],  # 第9到13组（4个）
        ["triviaqa", "samsum", "trec", "lsht"],  # 第14到18组（4个）
        ["passage_retrieval_en", "passage_count", "passage_retrieval_zh"],  # 第19到21组（3个）
        ["lcc", "repobench-p"],  # 第22到24组（2个）
    ]

    # 计算每个组的平均值
    group_averages = []
    for group in groups:
        group_scores = [score_dict[dataset]["score"] for dataset in group if dataset in score_dict]
        group_avg = sum(group_scores) / len(group_scores) if group_scores else 0
        group_averages.append((group, group_avg))

    # 写入 score.json
    score_json = os.path.join(run_save_dir, "score.json")
    with open(score_json, "w", encoding="utf-8") as f:
        json.dump(score_dict, f, ensure_ascii=False, indent=4)

    # 写入 CSV 文件
    output_path = os.path.join(run_save_dir, "score.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "score", "num"])  # 写入表头

        # 按照数据集写入每个数据集的分数和数量
        for group, _ in group_averages:
            # 先写入组的数据集分数
            for dataset in group:
                if dataset in score_dict:
                    writer.writerow([dataset, score_dict[dataset]["score"], score_dict[dataset]["num"]])

            group_avg = next(avg for g, avg in group_averages if g == group)  # 获取该组的均值
            group_avg = round(group_avg, 2)
            writer.writerow(["Avg", group_avg, ""])  # 每个组后面写上均值

        # 计算总的平均值（所有数据集的均值）
        all_scores = [
            score_dict[dataset]["score"]
            for dataset in score_dict
            if isinstance(score_dict[dataset], dict) and "score" in score_dict[dataset]
        ]
        overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0
        writer.writerow(["Avg (Overall)", overall_avg, ""])

    print(f"结果写入: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output", default="../output")

    args = parser.parse_args()
    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        run_config = yaml.safe_load(f)

    run_name = run_config.get("name", config_path.stem)
    if args.debug:
        run_name = f"{run_name}-debug"

    run_save_dir = os.path.join(args.output, run_name)
    os.makedirs(run_save_dir, exist_ok=True)

    result_json = os.path.join(run_save_dir, "result.json")
    config_yaml = os.path.join(run_save_dir, "config.yaml")
    with open(config_yaml, "w", encoding="utf-8") as f:
        yaml.dump(run_config, f, allow_unicode=True, sort_keys=False)

    predict(run_config, result_json, enable_test=args.debug)

    score_dict = eval(result_json)
    write_score(run_save_dir, score_dict)
    sys.exit(0)
