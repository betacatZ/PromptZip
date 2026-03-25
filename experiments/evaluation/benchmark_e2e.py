# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse

import gc
import time
import json
import sys
import os
import resource

from tqdm import tqdm
import yaml
import torch
from transformers import AutoTokenizer

from datasets import load_dataset
from copy import deepcopy

from vllm import LLM
from vllm.distributed.parallel_state import destroy_model_parallel

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../src")))
# from llmlingua import PromptCompressor
from compressor import (
    PPLCompressor,
    RerankCompressor,
    LongLLMLinguaTokenCompressor,
    # LLMLingua2PromptCompressor,
)

from vllm import SamplingParams
import multiprocessing

os.environ["HF_ENDPOINT"] = "https://huggingface.co"
import urllib3
import warnings

warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)

multiprocessing.set_start_method("spawn", force=True)

dataset2prompt = {
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
}

dataset2maxlen = {
    "multifieldqa_en": 1,
}


def _bytes_to_gib(num_bytes):
    return num_bytes / (1024**3)


def _get_gpu_mem_used_total(device_id):
    if not torch.cuda.is_available():
        return None

    try:
        device = torch.device(f"cuda:{int(device_id)}")
        torch.cuda.synchronize(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        used_bytes = total_bytes - free_bytes
        return used_bytes, total_bytes
    except Exception:
        return None


def _get_process_rss_mb():
    # Linux ru_maxrss unit is KB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run_target_length(m, yaml_args, reranker, compressor, llm):
    res = {}
    ds = load_dataset(
        "json",
        data_files={"test": "/data8/zhangdeming/PromptZip/experiments/evaluation/prompt.jsonl"},
        split="test",
    )
    dataset_len = len(ds)
    for i in tqdm(range(0, dataset_len, 1)):
        sample = ds[i]
        prompt_format = dataset2prompt[sample["dataset"]]
        max_gen = int(dataset2maxlen[sample["dataset"]])
        context = sample["context"]
        llm_tokenizer = AutoTokenizer.from_pretrained(
            yaml_args["llm_config"]["llm"]["model_name"], trust_remote_code=True
        )
        input_ids = llm_tokenizer(context)["input_ids"]
        n = len(input_ids)
        b = m // n + 1
        new_input_ids = (input_ids * b)[:m]
        # context = llm_tokenizer.decode(new_input_ids)
        s = 0
        T = 10

        sampling_dict = dict(yaml_args["llm_config"]["sampling"])
        sampling_params = SamplingParams(
            temperature=sampling_dict.get("temperature", 0.0),
            max_tokens=max_gen,
            min_tokens=max_gen,
            top_p=sampling_dict.get("top_p", 1.0),
        )

        reranker_time = 0.0
        compressor_time = 0.0
        llm_time = 0.0
        total_time = 0.0

        for _ in range(T + 1):
            context = llm_tokenizer.decode(new_input_ids)
            compressed_context = context
            reranker_elapsed = 0.0
            compressor_elapsed = 0.0
            if reranker:
                torch.cuda.synchronize()
                start_reranker = time.time()
                _, select_chunks, _ = reranker.compress(
                    context,
                    prompt_format,
                    sample["input"],
                    yaml_args["reranker_config"]["chunk_size"],
                    yaml_args["reranker_config"]["rate"],
                    chunk_method="bypunc",
                    selection_mode=yaml_args["reranker_config"].get("selection_mode", "topk"),
                )
                torch.cuda.synchronize()
                end_reranker = time.time()
                reranker_elapsed = end_reranker - start_reranker
                compressed_context = "".join(select_chunks)
            if compressor:
                if yaml_args["compressor_config"].get("model_type") == "llmlingua2":
                    torch.cuda.synchronize()
                    start_compressor = time.time()
                    compressed_context = compressor.compress_prompt(
                        compressed_context,
                        rate=yaml_args["compressor_config"]["rate"],
                        force_tokens=[
                            "!",
                            ".",
                            "?",
                            "。",
                            "？",
                            "！",
                            "\n",
                            "{{",
                            "}}",
                            "#",
                            "##",
                            "mediaItems",
                            "Image:",
                            "Image Caption:",
                            "Image Alt Text:",
                        ],
                        drop_consecutive=True,
                        use_token_level_filter=True,
                        use_sentence_level_filter=False,
                    )
                    torch.cuda.synchronize()
                    end_compressor = time.time()
                    compressed_context = compressed_context["compressed_prompt"]
                    compressed_context = "".join(compressed_context)
                    compressor_elapsed = end_compressor - start_compressor

            torch.cuda.synchronize()
            start_llm = time.time()

            outputs = llm.generate([compressed_context], sampling_params)
            torch.cuda.synchronize()
            end_llm = time.time()
            llm_elapsed = end_llm - start_llm
            if _:
                reranker_time += reranker_elapsed
                compressor_time += compressor_elapsed
                llm_time += llm_elapsed
                total_time += reranker_elapsed + compressor_elapsed + llm_elapsed

        print(
            f"{len(outputs[0].outputs[0].token_ids)} {m} reranker: {reranker_time / T:.4f}s,compressor: {compressor_time / T:.4f}s, llm: {llm_time / T:.4f}s, total: {total_time / T:.4f}s"
        )
        res[len(outputs[0].outputs[0].token_ids)] = (
            f"reranker: {reranker_time / T:.4f}s, compressor: {compressor_time / T:.4f}s, llm: {llm_time / T:.4f}s, total: {total_time / T:.4f}s"
        )

    res = dict(sorted(res.items(), key=lambda x: x[0]))
    return res


def build_components(yaml_args):
    # 1) get reranker
    ranker = None
    if yaml_args["reranker_config"].get("model_type") == "rerank":
        reranker_device_id = yaml_args["reranker_config"]["device_id"]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(reranker_device_id)
        gpu_mem_before = _get_gpu_mem_used_total(reranker_device_id)
        rss_before = _get_process_rss_mb()
        ranker = RerankCompressor(
            yaml_args["reranker_config"]["model_name"],
            f"cuda:{reranker_device_id}",
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

        gpu_mem_after = _get_gpu_mem_used_total(reranker_device_id)
        rss_after = _get_process_rss_mb()
        if gpu_mem_before is not None and gpu_mem_after is not None:
            used_before, total_bytes = gpu_mem_before
            used_after, _ = gpu_mem_after
            delta_bytes = used_after - used_before
            print(
                "[MEM][reranker] "
                f"GPU{reranker_device_id} used={_bytes_to_gib(used_after):.3f} GiB / {_bytes_to_gib(total_bytes):.3f} GiB, "
                f"delta={_bytes_to_gib(delta_bytes):.3f} GiB"
            )
        print(f"[MEM][reranker] process_rss={rss_after:.1f} MB, delta={rss_after - rss_before:.1f} MB")

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

    # elif comp_type == "llmlingua2":
    #     os.environ["CUDA_VISIBLE_DEVICES"] = str(yaml_args["compressor_config"]["device_id"])
    #     compressor = LLMLingua2PromptCompressor(
    #         model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    #         use_llmlingua2=True,
    #         device_map="cuda:0",
    #     )

    # 3) get llm
    os.environ["CUDA_VISIBLE_DEVICES"] = str(yaml_args["llm_config"]["device_ids"])
    llm_args = yaml_args["llm_config"]["llm"]
    llm = LLM(
        model=llm_args["model_name"],
        tensor_parallel_size=1,
        trust_remote_code=True,
        enforce_eager=True,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
    )

    return ranker, compressor, llm


def run_benchmark(run_config, result_json):
    TARGET_LENS = [int(l * 1000) for l in [2.5, 4, 8, 16, 32]]
    latency = {}
    reranker, compressor, llm = build_components(run_config)

    for l in TARGET_LENS:
        res = run_target_length(l, run_config, reranker, compressor, llm)
        latency[l] = res

    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(latency, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-c", "--config", type=str)
    args.add_argument("--context_window", type=int, default=100_000)
    args.add_argument("--run_benchmark", action="store_true")
    args.add_argument("--output", default="../output")
    args = args.parse_args()

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

    result_json_file = "e2e_result.json"
    config_yaml_file = "e2e_config.yaml"
    # only run llm
    if not has_reranker and not has_compressor:
        run_config = deepcopy(base_config)
        run_name = base_exp_name
        run_save_dir = os.path.join(args.output, run_name)
        os.makedirs(run_save_dir, exist_ok=True)

        result_json = os.path.join(run_save_dir, result_json_file)
        config_yaml = os.path.join(run_save_dir, config_yaml_file)
        with open(config_yaml, "w", encoding="utf-8") as f:
            yaml.dump(run_config, f, allow_unicode=True, sort_keys=False)
        os.makedirs(os.path.dirname(result_json), exist_ok=True)
        run_benchmark(run_config, result_json)

        sys.exit(0)
    if has_reranker and has_compressor:
        if len(rerank_rates) != len(comp_values):
            raise ValueError("reranker rate 和 compressor rate/threshold 数量不一致，不能一一对应")
        rate_pairs = list(zip(rerank_rates, comp_values))
    else:
        rate_pairs = [(r, c) for r in rerank_rates for c in comp_values]

    for chunk_size in chunk_sizes:
        for rr, cr in rate_pairs:
            print(
                "-" * 20,
                f"chunk_size={chunk_size}, reranker rate={rr}, compressor rate={cr}",
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

            run_name = "_".join(parts)
            run_save_dir = os.path.join(args.output, base_exp_name, run_name)
            os.makedirs(run_save_dir, exist_ok=True)

            engine = run_config.get("reranker_config", {}).get("engine")
            if engine:
                result_json = os.path.join(run_save_dir, f"e2e_result_{engine}.json")
            else:
                result_json = os.path.join(run_save_dir, result_json_file)

            config_yaml = os.path.join(run_save_dir, config_yaml_file)
            with open(config_yaml, "w", encoding="utf-8") as f:
                yaml.dump(run_config, f, allow_unicode=True, sort_keys=False)
            run_benchmark(run_config, result_json)
