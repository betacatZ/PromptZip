from datasets import load_dataset
from compressor import RerankCompressor
from tqdm import tqdm

dataset2prompt = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:',
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": 'Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\nThe answer is: ',
    "passage_retrieval_zh": '以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是"段落1"，"段落2"等格式\n\n答案是：',
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}

all_datasets = []
for task, prompt_format in dataset2prompt.items():
    ds = load_dataset(
        "json",
        data_files={
            "test": f"/data3/zhangdeming/cache/huggingface/datasets/zai-org/LongBench/data/{task}.jsonl"
        },
        split="test",
    )
    all_datasets.append(ds)

ranker = RerankCompressor("Qwen/Qwen3-Reranker-0.6B", "cuda:2")
batch_size = 32
global_total_chunk_length = 0
global_total_chunks = 0

dataset_chunk_sizes = {}  # To store the average chunk sizes for each dataset

# Loop over all datasets
import statistics

dataset_chunk_stats = {}  # To store the average chunk sizes and variance for each dataset

global_chunk_sizes = []  # 全局所有 chunk 的长度

# Loop over all datasets
for dataset in all_datasets:
    dataset_len = len(dataset)
    dataset_name = dataset[0]["dataset"]
    print(dataset_name)
    chunk_sizes = []  # 当前数据集所有 chunk 的长度

    # Loop over dataset in batches
    for i in tqdm(range(0, dataset_len, batch_size)):
        batch = [dataset[j] for j in range(i, min(i + batch_size, dataset_len))]
        for sample in batch:
            chunk_end_tokens = ["。", "！", "？", ".", "!", "?", "\n"]

            # Split context into chunks
            chunks = ranker._chunk_context(sample["context"], chunk_end_tokens, 512)

            # Count total chunk length and total chunks
            for chunk in chunks:
                ids = ranker.tokenizer.tokenize(chunk)
                size = len(ids)
                chunk_sizes.append(size)
                global_chunk_sizes.append(size)
    print(chunk_sizes)
    # Calculate average and variance for the dataset
    if len(chunk_sizes) > 0:
        avg_size = statistics.mean(chunk_sizes)
        std_size = statistics.pstdev(chunk_sizes)  # 总体标准差
    else:
        avg_size = 0
        std_size = 0

    dataset_chunk_stats[dataset_name] = (avg_size, std_size)

# Print the average and variance for each dataset
for dataset, (avg_size, std_size) in dataset_chunk_stats.items():
    print(f"{dataset}: average = {avg_size:.2f}, variance = {std_size:.2f}")


# 计算整体平均 & 方差
if len(global_chunk_sizes) > 0:
    overall_avg = statistics.mean(global_chunk_sizes)
    overall_std = statistics.pstdev(global_chunk_sizes)
else:
    overall_avg = 0
    overall_std = 0

print(f"\nOverall average chunk size: {overall_avg:.2f}")
print(f"Overall std  of chunk size: {overall_std:.2f}")