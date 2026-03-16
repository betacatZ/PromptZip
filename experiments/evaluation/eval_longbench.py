import argparse
import json
from pdb import run
import sys
import os
import urllib3
import warnings
import csv
import copy
import yaml
import asyncio
import torch
from datasets import load_dataset
from collections import defaultdict
from transformers import AutoTokenizer
import numpy as np
from copy import deepcopy
from utils import (
    setup_logging,
    construct_llm,
)

# from llmlingua import PromptCompressor
from metrics import (
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

from src.compressor import (
    PPLCompressor,
    RerankCompressor,
    LongLLMLinguaTokenCompressor,
    PromptCompressor,
)


from tqdm import tqdm
from vllm import SamplingParams
import multiprocessing

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
    "narrativeqa": [
        {
            "role": "system",
            "content": "You are a helpful assistant. You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible. Do not provide any explanation.",
        },
        {"role": "user", "content": "{context}\n\nQuestion: {input}"},
    ],
    "qasper": [
        {
            "role": "system",
            "content": 'You are a scientific assistant. Answer the question based on the article. If the question cannot be answered, write "unanswerable". For yes/no questions, respond with "yes", "no", or "unanswerable". Do not provide any explanations.',
        },
        {"role": "user", "content": "{context}\n\nQuestion: {input}"},
    ],
    "multifieldqa_en": [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the question based on the given text. Only provide the answer and do not give any additional words.",
        },
        {"role": "user", "content": "{context}\n\nQuestion: {input}"},
    ],
    "multifieldqa_zh": [
        {
            "role": "system",
            "content": "你是一个中文问答助手。请根据文章简洁地回答问题，只提供答案，不要额外解释。",
        },
        {"role": "user", "content": "{context}\n\n问题：{input}"},
    ],
    "hotpotqa": [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the question based on the given passages. Only give me the answer and do not output any other words.",
        },
        {
            "role": "user",
            "content": "The following are the given passages:\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}",
        },
    ],
    "2wikimqa": [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the question based on the given passages. Only give me the answer and do not output any other words.",
        },
        {
            "role": "user",
            "content": "The following are the given passages:\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}",
        },
    ],
    "musique": [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the question based on the given passages. Only give me the answer and do not output any other words..",
        },
        {
            "role": "user",
            "content": "The following are the given passages:\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}",
        },
    ],
    "dureader": [
        {
            "role": "system",
            "content": "你是一个中文问答助手。请根据给定文章回答问题。只给出答案，不要解释。",
        },
        {"role": "user", "content": "{context}\n\n问题：{input}"},
    ],
    "gov_report": [
        {
            "role": "system",
            "content": "You are a helpful assistant. You are given a report by a government agency. Write a one-page summary of the report.",
        },
        {"role": "user", "content": "{context}\n\nSummary:"},
    ],
    "qmsum": [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the query based on the meeting transcript in one or more sentences.",
        },
        {"role": "user", "content": "{context}\n\nQuery: {input}"},
    ],
    "multi_news": [
        {
            "role": "system",
            "content": "You are a helpful assistant. Read the news passages and write a one-page summary. Do not provide additional explanations.",
        },
        {"role": "user", "content": "{context}\n\nPlease provide the summary below."},
    ],
    "vcsum": [
        {
            "role": "system",
            "content": "你是会议总结助手。请根据会议记录写一段总结，不要解释。",
        },
        {"role": "user", "content": "{context}\n\n会议总结："},
    ],
    "trec": [
        {
            "role": "system",
            "content": "You are a classification assistant. Determine the type of the question based on the given examples. Only output the type/category.",
        },
        {"role": "user", "content": "{context}\n\nQuestion: {input}"},
    ],
    "triviaqa": [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the question based on the given passage. Only give the answer and do not output any other words.",
        },
        {"role": "user", "content": "{context}\n\n{input}"},
    ],
    "samsum": [
        {
            "role": "system",
            "content": "You are a helpful assistant. Summarize the dialogue into a few short sentences. Do not provide explanations or extra words.",
        },
        {"role": "user", "content": "{context}\n\n{input}"},
    ],
    "lsht": [
        {
            "role": "system",
            "content": "你是中文分类助手。根据给定的新闻内容，判断新闻的类别。只输出类别名称。",
        },
        {"role": "user", "content": "{context}\n\n{input}"},
    ],
    "passage_count": [
        {
            "role": "system",
            "content": "You are a helpful assistant. Count how many unique paragraphs there are in the given set of paragraphs after removing duplicates. Only output the number.",
        },
        {"role": "user", "content": "{context}"},
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
            "content": "你是中文阅读助手。请根据摘要判断该段落属于哪个段落，答案格式为“段落1”、“段落2”",
        },
        {
            "role": "user",
            "content": "{context}\n\n下面是一个摘要:\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是“段落1”、“段落2”等格式\n\n答案是：",
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

# dataset2prompt = {
#     "narrativeqa": "<|im_start|>system\nYou are a helpful assistant. You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n<|im_end|>\n\n<|im_start|>user\n<context>\n{context}\n</context>\n\nQuestion: {input}\n<|im_end|>\n\n<|im_start|>assistant\n",
#     "qasper": "<|im_start|>system\nYou are a scientific assistant. Answer the question based on the article. If the question cannot be answered, write 'unanswerable'. For yes/no questions, respond with 'yes', 'no', or 'unanswerable'. Do not provide any explanations.\n<|im_end|>\n\n<|im_start|>user\n<context>\n{context}\n</context>\n\nQuestion: {input}\n<|im_end|>\n\n<|im_start|>assistant\n",
#     "multifieldqa_en": "<|im_start|>system\nYou are a helpful assistant. Answer the question based on the given text. Only provide the answer and do not give any additional words.\n<|im_end|>\n\n<|im_start|>user\n<context>\n{context}\n</context>\n\nQuestion: {input}\n<|im_end|>\n\n<|im_start|>assistant\n",
#     "multifieldqa_zh": "<|im_start|>system\n你是一个中文问答助手。请根据文章简洁地回答问题，只提供答案，不要额外解释。\n<|im_end|>\n\n<|im_start|>user\n<context>\n{context}\n</context>\n\n问题：{input}\n<|im_end|>\n\n<|im_start|>assistant\n",
#     "hotpotqa": "<|im_start|>system\nYou are a helpful assistant. Answer the question based on the given passages. Only give me the answer and do not output any other words.\n<|im_end|>\n\n<|im_start|>user\nThe following are the given passages:\n<context>\n{context}\n</context>\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\n<|im_end|>\n\n<|im_start|>assistant\n",
#     "2wikimqa": "<|im_start|>system\nYou are a helpful assistant. Answer the question based on the given passages. Only give me the answer and do not output any other words.\n<|im_end|>\n\n<|im_start|>user\nThe following are the given passages:\n<context>\n{context}\n</context>\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\n<|im_end|>\n\n<|im_start|>assistant\n",
#     "musique": "<|im_start|>system\nYou are a helpful assistant. Answer the question based on the given passages. Only give me the answer and do not output any other words..\n<|im_end|>\n\n<|im_start|>user\nThe following are the given passages:\n<context>\n{context}\n</context>\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\n<|im_end|>\n\n<|im_start|>assistant\n",
#     "dureader": "<|im_start|>system\n你是一个中文问答助手。请根据给定文章回答问题。只给出答案，不要解释。\n<|im_end|>\n\n<|im_start|>user\n<context>\n{context}\n</context>\n\n问题：{input}\n<|im_end|>\n\n<|im_start|>assistant\n",
#     "gov_report": "<|im_start|>system\nYou are a helpful assistant. You are given a report by a government agency. Write a one-page summary of the report.\n<|im_end|>\n\n<|im_start|>user\n<context>\n{context}\n</context>\n\nSummary:\n<|im_end|>\n\n<|im_start|>assistant\n",
#     "qmsum": "<|im_start|>system\nYou are a helpful assistant. Answer the query based on the meeting transcript in one or more sentences.\n<|im_end|>\n\n<|im_start|>user\n<context>\n{context}\n</context>\n\nQuery: {input}\n<|im_end|>\n\n<|im_start|>assistant\n",
#     "multi_news": "<|im_start|>system\nYou are a helpful assistant. Read the news passages and write a one-page summary. Do not provide additional explanations.\n<|im_end|>\n\n<|im_start|>user\n<context>\n{context}\n</context>\n\nPlease provide the summary below.\n<|im_end|>\n\n<|im_start|>assistant\n",
#     "vcsum": "<|im_start|>system\n你是会议总结助手。请根据会议记录写一段总结，不要解释。\n<|im_end|>\n\n<|im_start|>user\n<context>\n{context}\n</context>\n\n会议总结：\n<|im_end|>\n\n<|im_start|>assistant\n",
#     "trec": "<|im_start|>system\nYou are a classification assistant. Determine the type of the question based on the given examples. Only output the type/category.\n<|im_end|>\n\n<|im_start|>user\n<context>\n{context}\n</context>\n\nQuestion: {input}\n<|im_end|>\n\n<|im_start|>assistant\n",
#     "triviaqa": "<|im_start|>system\nYou are a helpful assistant. Answer the question based on the given passage. Only give the answer and do not output any other words.\n<|im_end|>\n\n<|im_start|>user\n{context}\n\n{input}\n<|im_end|>\n\n<|im_start|>assistant\n",
#     "samsum": "<|im_start|>system\nYou are a helpful assistant. Summarize the dialogue into a few short sentences. Do not provide explanations or extra words.\n<|im_end|>\n\n<|im_start|>user\n{context}\n\n{input}\n<|im_end|>\n\n<|im_start|>assistant\n",
#     "lsht": "<|im_start|>system\n你是中文分类助手。根据给定的新闻内容，判断新闻的类别。只输出类别名称。\n<|im_end|>\n\n<|im_start|>user\n{context}\n\n{input}\n<|im_end|>\n\n<|im_start|>assistant\n",
#     "passage_count": "<|im_start|>system\nYou are a helpful assistant. Count how many unique paragraphs there are in the given set of paragraphs after removing duplicates. Only output the number.\n<|im_end|>\n\n<|im_start|>user\n{context}\n<|im_end|>\n\n<|im_start|>assistant\n",
#     "passage_retrieval_en": "<|im_start|>system\nYou are a helpful assistant. Determine which paragraph the abstract belongs to. Only output the number of the paragraph. The answer format must be like 'Paragraph 1', 'Paragraph 2', etc.\n<|im_end|>\n\n<|im_start|>user\nHere are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like 'Paragraph 1', 'Paragraph 2', etc.\n\nThe answer is: \n<|im_end|>\n\n<|im_start|>assistant\n",
#     "passage_retrieval_zh": "<|im_start|>system\n你是中文阅读助手。请根据摘要判断该段落属于哪个段落，答案格式为“段落1”、“段落2”\n<|im_end|>\n\n<|im_start|>user\n{context}\n\n下面是一个摘要:\n{input}\n<|im_end|>\n\n<|im_start|>assistant\n",
#     "lcc": "<|im_start|>system\nYou are a coding assistant. Complete the code below accurately. Only output the next line of code without explanation.\n<|im_end|>\n\n<|im_start|>user\n{context}\nNext line of code:\n<|im_end|>\n\n<|im_start|>assistant\n",
#     "repobench-p": "<|im_start|>system\nYou are a coding assistant. Complete the code below accurately based on the context. Only output the next line of code without explanation.\n<|im_end|>\n\n<|im_start|>user\n{context}{input}\nNext line of code:\n<|im_end|>\n\n<|im_start|>assistant\n",
# }
QUERY = {
    "gov_report": """Identify passages that contain key findings, policy conclusions, major facts, and important statistics that are essential for writing a comprehensive summary of a government report.""",
    "multi_news": """Select passages that contain the most important events, key facts, main actors,and outcomes needed to write a comprehensive summary of multiple news articles.""",
    "vcsum": """请找出会议记录中包含关键信息的段落，例如：会议讨论的主要议题、重要决策、关键观点和结论，这些信息将用于生成会议总结。""",
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
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores


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


def build_components(yaml_args):
    # 1) get reranker
    ranker = None
    if yaml_args["reranker_config"].get("model_type") == "rerank":
        ranker = RerankCompressor(
            yaml_args["reranker_config"]["model_name"],
            f"cuda:{yaml_args['reranker_config']['device_id']}",
            chunk_end_tokens=[
                "。",
                "！",
                "？",
                ".",
                "!",
                "?",
                "\n",
                "。\n",
                "？\n",
                "！\n",
            ],
        )
        ranker.max_position_embeddings = yaml_args["reranker_config"]["max_position_embeddings"]

    # 2) get compressor
    compressor = None
    comp_type = yaml_args["compressor_config"].get("model_type")
    if comp_type == "PPL":
        compressor = PPLCompressor(
            yaml_args["compressor_config"]["model_name"],
            f"cuda:{yaml_args['compressor_config']['device_id']}",
        )
        compressor.max_position_embeddings = yaml_args["compressor_config"]["max_position_embeddings"]

    elif comp_type == "longllmlingua":
        compressor = LongLLMLinguaTokenCompressor(
            yaml_args["compressor_config"]["model_name"],
            f"cuda:{yaml_args['compressor_config']['device_id']}",
        )

    elif comp_type == "llmlingua2":
        model_name = yaml_args["compressor_config"].get("model_name", None)
        quant = yaml_args["compressor_config"].get("quant", False)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(yaml_args["compressor_config"]["device_id"])

        compressor = PromptCompressor(model_name=model_name, use_llmlingua2=True, device_map="cuda:0", quant=quant)

    # 3) get llm
    os.environ["CUDA_VISIBLE_DEVICES"] = str(yaml_args["llm_config"]["device_ids"])
    llm = construct_llm(yaml_args["llm_config"])

    return ranker, compressor, llm


async def predict(yaml_args, json_path, enable_test=False):
    all_datasets = []
    for task in dataset2metric:
        # ds = load_dataset("THUDM/LongBench", task, split="test")
        dataset_dir = yaml_args["dataset_dir"]
        ds = load_dataset(
            "json",
            data_files={"test": os.path.join(dataset_dir, f"{task}.jsonl")},
            split="test",
        )
        if enable_test:
            ds = ds.select(range(min(20, len(ds))))
        all_datasets.append(ds)

    # dataset = concatenate_datasets(all_datasets)
    # save_path = generate_path(yaml_args["exp_config"]["name"] + ("-debug" if enable_test else ""))
    # json_path = save_path + "result.json"

    results = {}
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            results = json.load(f)

    batch_size = yaml_args["exp_config"]["batch_size"]

    tokenizer = AutoTokenizer.from_pretrained(yaml_args["llm_config"]["llm"]["model_name"], trust_remote_code=True)

    reranker, compressor, llm = build_components(yaml_args)

    queue = asyncio.Queue(maxsize=2)

    async def producer(dataset):
        dataset_len = len(dataset)

        # 选择分词器

        for i in tqdm(range(0, dataset_len, batch_size)):
            batch = [dataset[j] for j in range(i, min(i + batch_size, dataset_len))]

            # 跳过已经处理过的数据
            filtered_batch = []
            for sample in batch:
                sample_key = f"{sample['dataset']}_{sample['_id']}"
                if sample_key not in results:
                    filtered_batch.append(sample)
                else:
                    print(f"Skipping already processed sample: {sample_key}")

            if not filtered_batch:
                continue

            batch_prompt = []
            for sample in filtered_batch:
                # prompt_format = dataset2prompt[sample["dataset"]]
                max_gen = int(dataset2maxlen[sample["dataset"]])

                def run_compress():
                    context = sample["context"]
                    compressed_context = context
                    if reranker is not None:
                        if sample["dataset"] in ["gov_report", "multi_news", "vcsum"]:
                            sample["input"] = QUERY[sample["dataset"]]

                        _, select_chunks, _ = reranker.compress(
                            context,
                            None,
                            sample["input"],
                            run_config["reranker_config"]["chunk_size"],
                            run_config["reranker_config"]["rate"],
                            dataset=f"{sample['dataset']}_{sample['_id']}",
                            chunk_method="bypunc",
                            selection_mode=run_config["reranker_config"].get("selection_mode", "topk"),
                            result_path=run_save_dir,
                        )
                        context = "".join(select_chunks)
                        compressed_context = context

                    if compressor is not None:
                        if yaml_args["compressor_config"].get("model_type") == "longllmlingua":
                            if sample["dataset"] in [
                                "gov_report",
                                "multi_news",
                            ]:
                                compressed_context = compressor.compress(
                                    context=select_chunks,
                                    question=QUESTIONS[sample["dataset"]],
                                    rate=run_config["compressor_config"]["rate"],
                                    threshold=run_config["compressor_config"]["threshold"],
                                    compress_level=run_config["compressor_config"]["compress_level"],
                                )
                                compressed_context = "".join(compressed_context)
                            else:
                                compressed_context = compressor.compress(
                                    context=select_chunks,
                                    question=QUESTIONS[sample["dataset"]].format(input=sample["input"]),
                                    rate=run_config["compressor_config"]["rate"],
                                    threshold=run_config["compressor_config"]["threshold"],
                                    compress_level=run_config["compressor_config"]["compress_level"],
                                )
                                compressed_context = "".join(compressed_context)
                        if yaml_args["compressor_config"].get("model_type") == "llmlingua2":
                            if yaml_args["compressor_config"].get("compress_level") == "sentence":
                                if yaml_args["compressor_config"].get("rate"):
                                    compressed_context = compressor.compress_prompt(
                                        context,
                                        rate=yaml_args["compressor_config"]["rate"],
                                        drop_consecutive=True,
                                        use_token_level_filter=False,
                                        use_sentence_level_filter=True,
                                    )
                            if yaml_args["compressor_config"].get("compress_level") == "token":
                                # force_tokens = [
                                #     "!",
                                #     ".",
                                #     "?",
                                #     "。",
                                #     "？",
                                #     "！",
                                #     "\n",
                                #     "{{",
                                #     "}}",
                                #     "#",
                                #     "##",
                                #     "mediaItems",
                                #     "Image:",
                                #     "Image Caption:",
                                #     "Image Alt Text:",
                                # ]
                                force_tokens = ["!", ".", "?", "\n", ","]
                                if yaml_args["compressor_config"].get("rate"):
                                    compressed_context = compressor.compress_prompt(
                                        context,
                                        rate=yaml_args["compressor_config"]["rate"],
                                        force_tokens=force_tokens,
                                        use_token_level_filter=True,
                                        use_sentence_level_filter=False,
                                    )
                                elif yaml_args["compressor_config"].get("threshold"):
                                    compressed_context = compressor.compress_prompt(
                                        context,
                                        target_token=yaml_args["compressor_config"]["threshold"],
                                        force_tokens=force_tokens,
                                    )
                                compressed_context = compressed_context["compressed_prompt"]

                    return compressed_context

                result = await asyncio.to_thread(run_compress)

                # chatML msg
                # prompt_templete = deepcopy(dataset2messages[sample["dataset"]])
                # for msg in prompt_templete:
                #     msg["content"] = msg["content"].format(context=result, input=sample["input"])
                # prompt = tokenizer.apply_chat_template(prompt_templete, tokenize=False, add_generation_prompt=True)

                messages = copy.deepcopy(dataset2prompt[sample["dataset"]])

                if sample["dataset"] in ["gov_report", "multi_news", "vcsum"]:
                    messages[1]["content"] = messages[1]["content"].format(context=result)
                else:
                    messages[1]["content"] = messages[1]["content"].format(context=result, input=sample["input"])
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                # prompt = prompt_format.format(
                #     context=result,
                #     input=sample["input"],
                # )
                # model_inputs = tokenizer(prompt, return_tensors="pt")

                # if len(token_ids) > (yaml_args["exp_config"]["max_total_token"] - max_gen):
                #     half = int((yaml_args["exp_config"]["max_total_token"] - max_gen) / 2) - 1
                #     prompt = tokenizer.decode(token_ids[:half]) + tokenizer.decode(token_ids[-half:])

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
            sampling_dict["max_tokens"] = int(dataset2maxlen[dataset[0]["dataset"]])
            sampling_params = SamplingParams(
                temperature=sampling_dict.get("temperature", 0.0),
                max_tokens=sampling_dict.get("max_tokens", 128),
                top_p=sampling_dict.get("top_p", 1.0),
            )

            def run_llm():
                # with torch.cuda.device(f"cuda:{yaml_args['llm_config']['device_ids']}"):
                with torch.cuda.device("cuda:0"):
                    return llm.generate(batch_prompt, sampling_params)

            preds = await asyncio.to_thread(run_llm)

            for idx, sample in enumerate(batch):
                sample_key = f"{sample['dataset']}_{sample['_id']}"
                results[sample_key] = {
                    "input": sample["input"],
                    "pred": preds[idx].outputs[0].text,
                    "answers": sample["answers"],
                    "task": sample["dataset"],
                    "idx": sample_key,
                    "all_classes": sample["all_classes"],
                    "length": sample["length"],
                }

            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

    for dataset in all_datasets:
        print(len(dataset))
        print("Task: ", dataset[0]["dataset"])
        prod_task = asyncio.create_task(producer(dataset))
        cons_task = asyncio.create_task(consumer())
        await asyncio.gather(prod_task, cons_task)


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
        if "dataset" not in reader.fieldnames:
            raise ValueError("输入文件必须包含列名 'dataset'。")
        if rate_column not in reader.fieldnames:
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
    with open(args.config, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    base_exp_name = base_config["exp_config"]["name"]

    reranker_conf = base_config.get("reranker_config", {}) or {}
    compressor_conf = base_config.get("compressor_config", {}) or {}

    has_reranker = reranker_conf.get("model_type") is not None
    has_compressor = compressor_conf.get("model_type") is not None
    chunk_sizes = reranker_conf.get("chunk_size", [None]) if has_reranker else [None]

    # reranker rate
    rerank_rates = reranker_conf.get("rate", [None]) if has_reranker else [None]
    # compressor 可能用 rate 或 threshold
    if has_compressor:
        if compressor_conf.get("rate") not in (None, [], [None]):
            comp_key = "rate"
            comp_values = compressor_conf["rate"]
        elif compressor_conf.get("threshold") not in (None, [], [None]):
            comp_key = "threshold"
            comp_values = compressor_conf["threshold"]
        else:
            comp_key = None
            comp_values = [None]
    else:
        comp_key = None
        comp_values = [None]

    # only run llm
    if not has_reranker and not has_compressor:
        run_config = deepcopy(base_config)
        run_name = base_exp_name + ("-debug" if args.debug else "")
        run_save_dir = os.path.join(args.output, run_name)
        os.makedirs(run_save_dir, exist_ok=True)

        result_json = os.path.join(run_save_dir, "result.json")
        config_yaml = os.path.join(run_save_dir, "config.yaml")
        with open(config_yaml, "w", encoding="utf-8") as f:
            yaml.dump(run_config, f, allow_unicode=True, sort_keys=False)

        asyncio.run(predict(run_config, result_json, enable_test=args.debug))

        score_dict = eval(result_json)
        write_score(run_save_dir, score_dict)

        sys.exit(0)  #  阻止继续进入后面的多重循环

    # ------------------------ rate 一一对应处理 ------------------------
    if has_reranker and has_compressor:
        if len(rerank_rates) != len(comp_values):
            raise ValueError("reranker rate 和 compressor rate/threshold 数量不一致，不能一一对应")
        rate_pairs = list(zip(rerank_rates, comp_values))
    else:
        rate_pairs = [(r, c) for r in rerank_rates for c in comp_values]

    # ------------------------ 循环执行 ------------------------
    for chunk_size in chunk_sizes:
        for rr, cr in rate_pairs:
            print(
                "-" * 20,
                ",chunk_size=",
                chunk_size,
                ",reranker rate=",
                rr,
                ",compressor rate=",
                cr,
                "-" * 20,
            )
            run_config = deepcopy(base_config)
            parts = []
            # chunk_size
            if has_reranker and chunk_size is not None:
                run_config["reranker_config"]["chunk_size"] = chunk_size
                run_config["reranker_config"]["rate"] = rr
                parts.append(f"chunksize-{chunk_size}-rate-{rr}")

            # compressor
            if has_compressor:
                if comp_key == "rate":
                    run_config["compressor_config"]["rate"] = cr
                    run_config["compressor_config"]["threshold"] = None
                    parts.append(f"compress-rate-{cr}")
                elif comp_key == "threshold":
                    run_config["compressor_config"]["threshold"] = cr
                    run_config["compressor_config"]["rate"] = None
                    parts.append(f"compress-thr-{cr}")
                else:
                    run_config["compressor_config"]["rate"] = None
                    run_config["compressor_config"]["threshold"] = None

            if not parts:
                parts.append("base")

            if args.debug:
                parts.append("debug")

            run_name = "_".join(parts)
            run_save_dir = os.path.join(args.output, base_exp_name, run_name)
            os.makedirs(run_save_dir, exist_ok=True)

            result_json = os.path.join(run_save_dir, "result.json")
            config_yaml = os.path.join(run_save_dir, "config.yaml")
            with open(config_yaml, "w", encoding="utf-8") as f:
                yaml.dump(run_config, f, allow_unicode=True, sort_keys=False)

            asyncio.run(predict(run_config, result_json, enable_test=args.debug))

            score_dict = eval(result_json)
            write_score(run_save_dir, score_dict)

            if (
                has_reranker
                and has_compressor
                and run_config["reranker_config"].get("selection_mode") in ["cluster", "topp", "cluster-zscore"]
            ):
                cluster_ratio_txt = os.path.join(run_save_dir, "rate.csv")
                avg_ratio_txt = os.path.join(run_save_dir, "avg_rate.csv")
                compute_rate_averages(cluster_ratio_txt, avg_ratio_txt)
