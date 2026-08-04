"""评测 reranker 在 OpenMLRL/BFCL-V4-Parallel-Multi-Turn 上的精度（压缩 function 文档）。

两种模式：
- baseline：原样喂全部 function 文档
- compress_func：reranker 给每个工具文档（整条 JSON 为 1 个 chunk）打分，纯 top-k 保留高相关工具

判分用 bfcl_metrics（移植自官方 ast_checker 的 parallel_function_checker_no_order 口径）。

用法:
    python eval_bfcl_parallel_multi_turn.py -c ../config/bfcl_parallel_multi_turn.yaml [--debug]
"""

import argparse
import asyncio
import copy
import csv
import json
import os
import sys
import warnings

import yaml
import torch
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# 引入项目源码与评测工具
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.append(os.path.dirname(__file__))

from utils import setup_logging, construct_llm  # noqa: E402
from src.compressor import RerankCompressor  # noqa: E402
import bfcl_metrics  # noqa: E402

# BFCL dataset id
BFCL_DATASET = "OpenMLRL/BFCL-V4-Parallel-Multi-Turn"

# reranker 默认指令（与 web search 检索语义一致，Qwen3-Reranker 原生 instruction）
DEFAULT_INSTRUCTION = "Given a user request, retrieve the most relevant tool functions that should be called to fulfill the request."

# 期望模型输出的工具调用 JSON 数组格式说明
# 注：当 tokenizer 支持 tools= 时，工具文档由 chat template 自动渲染（Hermes 风格），
# 此 SYSTEM_PROMPT 仅在不支持 tools= 的 fallback 场景使用。
SYSTEM_PROMPT_TEMPLATE = (
    "You are a function calling assistant. Based on the user request and the available tools below, "
    "output the tool calls needed for the CURRENT user turn.\n\n"
    "Return ONLY a JSON array of objects, each with \"name\" and \"arguments\" keys, e.g.:\n"
    '[{{"name": "<func_name>", "arguments": {{"<param>": <value>}}}}]\n\n'
    "Do not output any text outside the JSON array. If no tool call is needed for this turn, return [].\n\n"
    "Available tools:\n{tools}"
)

# 当用 tools= 渲染时的 system 提示（数据集 user_prompt 已含「Return only the tool calls...」指令，
# 这里只补充输出格式约定，避免与数据集指令冲突）
SYSTEM_WITH_TOOLS = (
    "You are a function calling assistant. For the CURRENT user turn, decide which tools to call. "
    'Output the tool calls as a JSON array: [{"name": "<func>", "arguments": {<param>: <value>}}]. '
    "If no tool call is needed, return []. Return only the JSON array."
)


def _bfcl_to_openai_tools(func_list: list) -> list:
    """把 BFCL 工具格式转成 OpenAI/JSON Schema 标准工具格式，供 chat template 的 tools= 使用。

    BFCL: {"name","description","parameters":{"type":"dict","properties":{...},"required":[...]},"response":{...}}
    OpenAI: {"type":"function","function":{"name","description","parameters":{"type":"object","properties":{...},"required":[...]}}}
    主要差异：dict->object，外层包一层 function，去掉 response。
    """
    out = []
    for f in func_list:
        params = copy.deepcopy(f.get("parameters") or {})  # 深拷贝，避免污染原数据集对象
        # 递归把 type: dict -> object（嵌套 properties 里也可能有）
        def _fix(node):
            if isinstance(node, dict):
                if node.get("type") == "dict":
                    node["type"] = "object"
                for v in node.values():
                    _fix(v)
            elif isinstance(node, list):
                for v in node:
                    _fix(v)
            return node

        params = _fix(params)
        # 兜底：OpenAI 规范要求 parameters 顶层 type=object
        if "type" not in params:
            params["type"] = "object"
        out.append(
            {
                "type": "function",
                "function": {
                    "name": f["name"],
                    "description": f.get("description", ""),
                    "parameters": params,
                },
            }
        )
    return out


def _build_prompt(tokenizer, func_list: list, user_prompt: str, max_total_token: int, max_gen: int) -> str:
    """构造推理 prompt：优先用 chat template 的 tools= 渲染工具文档。

    成功走 tools= 路径（模型训练时见过的格式）→ 输出更可靠；
    失败则 fallback 到纯文本 tools 塞 system。
    返回 (prompt, used_tools_param)：used_tools_param 标记是否走了 tools= 路径。
    """
    messages = [
        {"role": "system", "content": SYSTEM_WITH_TOOLS},
        {"role": "user", "content": user_prompt},
    ]
    tools = _bfcl_to_openai_tools(func_list)
    used_tools_param = True
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True
        )
    except (TypeError, ValueError):
        # 该 tokenizer 不支持 tools= 参数，fallback 到纯文本
        used_tools_param = False
        tools_text = _build_tools_for_prompt(func_list)
        messages[0]["content"] = SYSTEM_PROMPT_TEMPLATE.format(tools=tools_text)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 超长截断（保留首尾）
    token_ids = tokenizer.encode(prompt)
    if len(token_ids) > (max_total_token - max_gen):
        half = int((max_total_token - max_gen) / 2) - 1
        prompt = tokenizer.decode(token_ids[:half]) + tokenizer.decode(token_ids[-half:])
    return prompt, used_tools_param
    return prompt


def load_bfcl_eval(dataset_dir: str | None = None):
    """加载 BFCL V4 Parallel Multi-Turn eval 集。

    优先用本地 eval.jsonl（dataset_dir 指定）；否则用 datasets 库从 HF 拉取。
    返回 list[dict]，每行含 id/official_category/task_type/user_prompt/function/ground_truth/turn_index。
    """
    if dataset_dir:
        local = os.path.join(dataset_dir, "eval.jsonl")
        if os.path.exists(local):
            with open(local, "r", encoding="utf-8") as f:
                return [json.loads(l) for l in f if l.strip()]
    # 从 HF 拉
    from datasets import load_dataset
    ds = load_dataset(BFCL_DATASET, split="eval")
    return [dict(r) for r in ds]


def _extract_current_turn(user_prompt: str) -> str:
    """从 user_prompt 里抽 CURRENT user turn 的自然语言作为 reranker 的 query。

    user_prompt 结构：... Initial environment state: <JSON> ... CURRENT user turn: <text> ... Return only ...
    """
    marker = "CURRENT user turn:"
    if marker in user_prompt:
        rest = user_prompt.split(marker, 1)[1]
        # 截到 "Return only" 之前
        if "Return only" in rest:
            rest = rest.split("Return only", 1)[0]
        return rest.strip()
    # fallback：取后半段
    return user_prompt[-500:].strip()


def build_components(yaml_args):
    """构造 reranker 与 llm。复刻 eval_longbench 的 build_components 精简版。"""
    ranker = None
    rc = yaml_args.get("reranker_config", {}) or {}
    if rc.get("model_type") == "rerank":
        ranker = RerankCompressor(
            rc["model_name"],
            f"cuda:{rc['device_id']}",
            chunk_end_tokens=[
                "。", "！", "？", ".", "!", "?", "\n",
                "。\n", "？\n", "！\n",
            ],
            engine=rc["engine"],
        )
        ranker.max_position_embeddings = rc.get("max_position_embeddings", 32768)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(yaml_args["llm_config"]["device_ids"])
    llm = construct_llm(yaml_args["llm_config"])
    return ranker, llm


def _build_tools_for_prompt(func_list: list) -> str:
    """把工具列表序列化成 prompt 里展示的文本（每个工具一个 JSON 块）。"""
    return "\n".join(json.dumps(f, ensure_ascii=False) for f in func_list)


def _compress_tools(ranker, func_list: list, query: str, rate, instruction: str, dataset_tag: str, result_path: str):
    """用 reranker 对工具文档做纯 top-k 选择，返回 (保留的工具列表, 保留的函数名集合)。"""
    if not func_list or ranker is None:
        return func_list, {f["name"] for f in func_list}

    # 每个工具的完整 JSON 文档作为一个 chunk
    chunks = [json.dumps(f, ensure_ascii=False) for f in func_list]
    _, selected_chunks, _ = ranker.compress_chunks(
        chunks,
        instruction=instruction,
        query=query,
        rate=rate,
        dataset=dataset_tag,
        result_path=result_path,
    )

    # 反序列化回工具 dict
    kept = []
    for c in selected_chunks:
        try:
            kept.append(json.loads(c))
        except json.JSONDecodeError:
            continue
    return kept, {f["name"] for f in kept}


async def predict(yaml_args, mode: str, rate, json_path: str, enable_test: bool = False):
    """对 eval 集逐条生成函数调用。mode ∈ {baseline, compress_func}。"""
    rows = load_bfcl_eval(yaml_args.get("dataset_dir"))
    if enable_test:
        rows = rows[: min(20, len(rows))]

    ranker, llm = build_components(yaml_args)
    tokenizer = AutoTokenizer.from_pretrained(
        yaml_args["llm_config"]["llm"]["model_name"], trust_remote_code=True
    )

    instruction = yaml_args["reranker_config"].get("instruction") or DEFAULT_INSTRUCTION
    max_gen = yaml_args["llm_config"]["sampling"].get("max_tokens", 512)
    max_total_token = yaml_args["exp_config"].get("max_total_token", 32768)

    results = {}
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            results = json.load(f)

    batch_size = yaml_args["exp_config"].get("batch_size", 8)
    queue = asyncio.Queue(maxsize=2)

    async def producer():
        for i in tqdm(range(0, len(rows), batch_size), desc=f"[{mode} rate={rate}]"):
            batch = rows[i : i + batch_size]
            filtered = [r for r in batch if r["id"] not in results]
            if not filtered:
                continue

            batch_prompt = []
            meta = []  # 与 batch_prompt 对齐的附加信息
            for sample in filtered:
                query = _extract_current_turn(sample["user_prompt"])
                func_list = sample["function"]
                gt_funcs = {list(it.keys())[0] for it in sample["ground_truth"]} if sample["ground_truth"] else set()

                if mode == "compress_func" and ranker is not None:
                    dataset_tag = f"{sample['id']}"
                    kept_funcs, kept_names = _compress_tools(
                        ranker, func_list, query, rate, instruction, dataset_tag, yaml_args.get("_result_path", "")
                    )
                else:
                    kept_funcs = func_list
                    kept_names = {f["name"] for f in func_list}

                # 用 chat template 的 tools= 渲染工具文档（模型训练时见过的格式）
                prompt, used_tools_param = _build_prompt(
                    tokenizer, kept_funcs, sample["user_prompt"], max_total_token, max_gen
                )

                # 第一条样本打印诊断：看 tools= 是否生效 + prompt 前缀
                if not getattr(predict, "_diag_done", False):
                    print("=" * 60)
                    print(f"[诊断] mode={mode} sample_id={sample['id']}")
                    print(f"[诊断] tools= 生效: {used_tools_param}")
                    print(f"[诊断] prompt 前 600 字符:")
                    print(prompt[:600])
                    print(f"[诊断] prompt 总 token 数: {len(tokenizer.encode(prompt))}")
                    print("=" * 60)
                    predict._diag_done = True

                batch_prompt.append(prompt)
                meta.append(
                    {
                        "id": sample["id"],
                        "official_category": sample["official_category"],
                        "task_type": sample["task_type"],
                        "turn_index": sample["turn_index"],
                        "n_tools_total": len(func_list),
                        "n_tools_kept": len(kept_funcs),
                        "kept_funcs": sorted(kept_names),
                        "gt_funcs": sorted(gt_funcs),
                        "tool_recall": (
                            len(kept_names & gt_funcs) / len(gt_funcs) if gt_funcs else 1.0
                        ),
                        # 判分用：模型实际可见的工具列表（压缩后）+ ground_truth
                        "function": kept_funcs,
                        "ground_truth": sample["ground_truth"],
                    }
                )

            await queue.put((batch_prompt, meta))
        await queue.put(None)

    async def consumer():
        while True:
            item = await queue.get()
            if item is None:
                break
            batch_prompt, meta = item

            sampling_params = SamplingParams(
                temperature=yaml_args["llm_config"]["sampling"].get("temperature", 0.0),
                max_tokens=max_gen,
                top_p=yaml_args["llm_config"]["sampling"].get("top_p", 1.0),
            )

            def run_llm():
                with torch.cuda.device("cuda:0"):
                    return llm.generate(batch_prompt, sampling_params)

            preds = await asyncio.to_thread(run_llm)

            for idx, m in enumerate(meta):
                results[m["id"]] = {
                    "mode": mode,
                    "rate": rate,
                    "pred": preds[idx].outputs[0].text,
                    **m,
                }

            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    await asyncio.gather(producer(), consumer())
    return results


def score(run_save_dir: str, json_path: str):
    """读 result.json，逐条调 bfcl_metrics 判分，写 score.json / score.csv / recall.csv。"""
    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    judged = []
    for rid, data in results.items():
        res = bfcl_metrics.evaluate_row(
            data["function"],
            data["pred"],
            data["ground_truth"],
        )
        judged.append(
            {
                "id": rid,
                "mode": data["mode"],
                "rate": data["rate"],
                "official_category": data["official_category"],
                "task_type": data["task_type"],
                "valid": res["valid"],
                "error_type": res["error_type"],
                "n_tools_total": data["n_tools_total"],
                "n_tools_kept": data["n_tools_kept"],
                "tool_recall": data["tool_recall"],
                "pred_parsed": res["model_output"],
            }
        )

    # 写 per-row 判分明细
    with open(os.path.join(run_save_dir, "judged.json"), "w", encoding="utf-8") as f:
        json.dump(judged, f, ensure_ascii=False, indent=2)

    # 按 (mode, rate) 分组算 accuracy
    by_group = defaultdict(list)
    for r in judged:
        by_group[(r["mode"], r["rate"])].append(r)

    score_dict = {}
    for (mode, rate), items in by_group.items():
        key = f"{mode}_rate-{rate}"
        score_dict[key] = bfcl_metrics.compute_accuracy(items)
        score_dict[key]["mode"] = mode
        score_dict[key]["rate"] = rate

    with open(os.path.join(run_save_dir, "score.json"), "w", encoding="utf-8") as f:
        json.dump(score_dict, f, ensure_ascii=False, indent=2)

    # CSV: 每行一个 (mode, rate, category) 切片，列结构清晰，Excel 可直接打开
    # UTF-8 with BOM：让 Excel/Numbers 正确识别中文 category 名（如 long_context）
    csv_path = os.path.join(run_save_dir, "score.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "mode", "rate", "category", "overall_acc", "num",
            "n_tools_avg", "tool_recall_avg", "top_error_type",
        ])
        for key, sd in score_dict.items():
            items = by_group[(sd["mode"], sd["rate"])]
            n_avg = sum(r["n_tools_kept"] for r in items) / len(items) if items else 0
            recall_avg = sum(r["tool_recall"] for r in items) / len(items) if items else 0
            # 主行：overall
            top_err = max(sd["error_type_distribution"].items(), key=lambda x: x[1])[0] if sd["error_type_distribution"] else "-"
            writer.writerow([
                sd["mode"],
                sd["rate"] if sd["rate"] is not None else "-",
                "OVERALL",
                f"{sd['overall_accuracy']:.4f}",
                sd["num_samples"],
                f"{n_avg:.2f}",
                f"{recall_avg:.4f}",
                top_err,
            ])
            # 子行：per official_category
            for cat, cd in sd["by_category"].items():
                writer.writerow([
                    sd["mode"],
                    sd["rate"] if sd["rate"] is not None else "-",
                    cat,
                    f"{cd['accuracy']:.4f}",
                    cd["num"],
                    "",  # n_tools 仅 overall 有意义
                    "",  # tool_recall 仅 overall 有意义
                    "",
                ])

    print(f"[score] 写入 {run_save_dir}/score.json 与 score.csv")
    # 打印摘要
    for key, sd in score_dict.items():
        print(
            f"  {key}: overall_acc={sd['overall_accuracy']:.4f} "
            f"({sum(1 for r in by_group[(sd['mode'],sd['rate'])] if r['valid'])}/{sd['num_samples']})"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output", default="../output")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    base_exp_name = base_config["exp_config"]["name"]
    modes = base_config["exp_config"].get("modes", ["baseline"])
    rates = base_config["reranker_config"].get("rate", [None])

    for mode in modes:
        rate_list = rates if mode == "compress_func" else [None]
        for rate in rate_list:
            run_config = copy.deepcopy(base_config)
            run_config["reranker_config"]["rate"] = rate
            parts = [mode]
            if mode == "compress_func":
                parts.append(f"rate-{rate}")
            if args.debug:
                parts.append("debug")
            run_name = "_".join(str(p) for p in parts)

            run_save_dir = os.path.join(args.output, base_exp_name, run_name)
            os.makedirs(run_save_dir, exist_ok=True)
            run_config["_result_path"] = run_save_dir

            config_yaml = os.path.join(run_save_dir, "config.yaml")
            with open(config_yaml, "w", encoding="utf-8") as f:
                yaml.dump(run_config, f, allow_unicode=True, sort_keys=False)

            result_json = os.path.join(run_save_dir, "result.json")
            print("-" * 40, mode, "rate=", rate, "-" * 40)
            asyncio.run(predict(run_config, mode, rate, result_json, enable_test=args.debug))
            score(run_save_dir, result_json)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
