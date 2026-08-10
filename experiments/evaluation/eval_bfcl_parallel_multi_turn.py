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

# 当用 tools= 渲染时的 system 提示。
# 对照 BFCL 官方 _DEFAULT_SYSTEM_PROMPT（constants/default_prompts.py），保留关键指令：
#  - 缺函数/缺参数要指出（对应 miss_func/miss_param 类别）
#  - 「Continue to output functions to call until you have fulfilled the user's request」
#    这是防止模型少调用的关键指令（BFCL multi-turn 必备）
# 输出格式（Hermes tool_call 标签）由 chat template 自动注入，不在此重复。
SYSTEM_WITH_TOOLS = (
    "You are an expert in composing functions. You are given a question and a set of possible functions. "
    "Based on the question, you will need to make one or more function/tool calls to achieve the purpose. "
    "If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out. "
    "You should only return the function calls in your response.\n\n"
    "At each turn, you should try your best to complete the tasks requested by the user within the current turn. "
    "Continue to output functions to call until you have fulfilled the user's request to the best of your ability. "
    "Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task."
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
            messages, tools=tools, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except (TypeError, ValueError):
        # 该 tokenizer 不支持 tools= 参数，fallback 到纯文本
        used_tools_param = False
        tools_text = _build_tools_for_prompt(func_list)
        messages[0]["content"] = SYSTEM_PROMPT_TEMPLATE.format(tools=tools_text)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )

    # 超长截断（保留首尾）
    token_ids = tokenizer.encode(prompt)
    if len(token_ids) > (max_total_token - max_gen):
        half = int((max_total_token - max_gen) / 2) - 1
        prompt = tokenizer.decode(token_ids[:half]) + tokenizer.decode(token_ids[-half:])
    return prompt, used_tools_param


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
                pred_text = preds[idx].outputs[0].text
                # 端到端判分：这里拿到的是 reranker（压缩工具文档）+ LLM 正常生成的完整输出。
                # 同时跑两套判分，各自用途不同：
                #   1) evaluate_row —— 官方无序并行口径（|P|须等于|G|且全匹配才算 valid），
                #      主要用于拿 error_type 做错误类型分布统计（哪类错最多）。
                #   2) compute_ast_metrics —— 5 种 AST 口径（exact/superset/subset/top1/top3），
                #      口径更全：exact 即官方口径，superset 不罚多调，subset 不罚少调，
                #      top1/top3 看至少命中几个。供 score() 汇总 5 维准确率。
                # valid 字段改用 AST 的 exact_match（与 evaluate_row 的无序并行口径等价），
                # 这样 compute_accuracy 算的 overall_acc 与 AST 的 exact_acc 保持一致。
                res = bfcl_metrics.evaluate_row(m["function"], pred_text, m["ground_truth"])
                ast = bfcl_metrics.compute_ast_metrics(m["function"], pred_text, m["ground_truth"])
                results[m["id"]] = {
                    "mode": mode,
                    "rate": rate,
                    "pred": pred_text,
                    "valid": ast["exact_match"],   # = 官方无序并行口径（最严格）
                    "error_type": res["error_type"],
                    "error": res["error"],
                    "parsed_calls": res["model_output"],
                    "ast": ast,                    # 5 种口径 + n_matched/n_gt/n_pred
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
        # 复用 result.json 里已写入的判分（consumer 已判过）；旧 result 缺字段则重算
        if "valid" in data and "error_type" in data:
            valid = data["valid"]
            error_type = data["error_type"]
            pred_parsed = data.get("parsed_calls", [])
        else:
            # 旧 result 无判分字段，需重算；若也缺 function（新 result 不存），则报错提示重跑
            if "function" not in data:
                raise RuntimeError(
                    f"result.json 中 {rid} 既无判分字段又无 function 字段，无法判分。请重新跑 predict 生成完整 result。"
                )
            res = bfcl_metrics.evaluate_row(data["function"], data["pred"], data["ground_truth"])
            valid = res["valid"]
            error_type = res["error_type"]
            pred_parsed = res["model_output"]
        # AST 5 口径：优先用 result 里 consumer 已写入的 ast；旧 result 缺则重算并回写
        if "ast" in data:
            ast = data["ast"]
        else:
            if "function" not in data:
                raise RuntimeError(
                    f"result.json 中 {rid} 缺 ast 字段又无 function 字段，无法重算 AST。请重新跑 predict 生成完整 result。"
                )
            ast = bfcl_metrics.compute_ast_metrics(data["function"], data["pred"], data["ground_truth"])
            results[rid]["ast"] = ast  # 回写，下面统一 dump，下次跑可直接复用
        judged.append(
            {
                "id": rid,
                "mode": data["mode"],
                "rate": data["rate"],
                "official_category": data["official_category"],
                "task_type": data["task_type"],
                "valid": valid,            # = exact_match（最严格口径，见 consumer 注释）
                "error_type": error_type,
                "ast": ast,                 # 5 种口径 + n_matched/n_gt/n_pred
                "n_tools_total": data["n_tools_total"],
                "n_tools_kept": data["n_tools_kept"],
                "tool_recall": data["tool_recall"],
                "pred_parsed": pred_parsed,
            }
        )

    # 回写补算的 ast 字段到 result.json（若上面有重算），保持 result 与 judged 一致
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 写 per-row 判分明细（含 5 种口径逐条对错，便于定位哪些样本在哪个口径上挂了）
    with open(os.path.join(run_save_dir, "judged.json"), "w", encoding="utf-8") as f:
        json.dump(judged, f, ensure_ascii=False, indent=2)

    # 按 (mode, rate) 分组算准确率
    by_group = defaultdict(list)
    for r in judged:
        by_group[(r["mode"], r["rate"])].append(r)

    score_dict = {}
    for (mode, rate), items in by_group.items():
        key = f"{mode}_rate-{rate}"
        # compute_accuracy：单一 overall_acc（基于 valid=exact_match）+ 错误类型分布
        # compute_ast_accuracy：5 种口径（exact/superset/subset/top1/top3）准确率，维度更全
        score_dict[key] = bfcl_metrics.compute_accuracy(items)
        score_dict[key]["ast"] = bfcl_metrics.compute_ast_accuracy(items)
        score_dict[key]["mode"] = mode
        score_dict[key]["rate"] = rate

    with open(os.path.join(run_save_dir, "score.json"), "w", encoding="utf-8") as f:
        json.dump(score_dict, f, ensure_ascii=False, indent=2)

    # CSV: 每行一个 (mode, rate, category) 切片，列结构清晰，Excel 可直接打开
    # UTF-8 with BOM：让 Excel/Numbers 正确识别中文 category 名（如 long_context）
    # 列说明：
    #   overall_acc  = exact_match 准确率（最严格，与 valid 一致）
    #   exact        = exact_match   准确率（P=G，完全一致，不多不少）
    #   superset     = superset_match 准确率（G⊆P，GT 全命中，不罚多调）
    #   subset       = subset_match  准确率（P⊆G，输出全对，不罚少调）
    #   top1         = top1_match    准确率（至少命中 1 个 GT 调用）
    #   top3         = top3_match    准确率（至少命中 min(3,|G|) 个 GT 调用）
    #   n_tools_avg  = 平均保留工具数（压缩效果）
    #   tool_recall_avg = reranker 工具召回率平均（GT 工具被保留的比例，压缩前后对比）
    #   top_error_type = 数量最多的错误类型（错误分布诊断）
    csv_path = os.path.join(run_save_dir, "score.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "mode", "rate", "category", "overall_acc",
            "exact", "superset", "subset", "top1", "top3",
            "num",
            "n_tools_avg", "tool_recall_avg", "top_error_type",
        ])
        for key, sd in score_dict.items():
            items = by_group[(sd["mode"], sd["rate"])]
            n_avg = sum(r["n_tools_kept"] for r in items) / len(items) if items else 0
            recall_avg = sum(r["tool_recall"] for r in items) / len(items) if items else 0
            # 主行：overall（全样本的 5 种口径 + overall_acc + 错误分布）
            top_err = max(sd["error_type_distribution"].items(), key=lambda x: x[1])[0] if sd["error_type_distribution"] else "-"
            ast_ov = sd["ast"]["overall"]
            writer.writerow([
                sd["mode"],
                sd["rate"] if sd["rate"] is not None else "-",
                "OVERALL",
                f"{sd['overall_accuracy']:.4f}",
                f"{ast_ov['exact_match']:.4f}",
                f"{ast_ov['superset_match']:.4f}",
                f"{ast_ov['subset_match']:.4f}",
                f"{ast_ov['top1_match']:.4f}",
                f"{ast_ov['top3_match']:.4f}",
                sd["num_samples"],
                f"{n_avg:.2f}",
                f"{recall_avg:.4f}",
                top_err,
            ])
            # 子行：per official_category（5 种口径的类内准确率）
            for cat, cd in sd["ast"]["by_category"].items():
                writer.writerow([
                    sd["mode"],
                    sd["rate"] if sd["rate"] is not None else "-",
                    cat,
                    "",  # overall_acc 仅 overall 行有意义
                    f"{cd['exact_match']:.4f}",
                    f"{cd['superset_match']:.4f}",
                    f"{cd['subset_match']:.4f}",
                    f"{cd['top1_match']:.4f}",
                    f"{cd['top3_match']:.4f}",
                    cd["num"],
                    "",  # n_tools 仅 overall 有意义
                    "",  # tool_recall 仅 overall 有意义
                    "",
                ])

    print(f"[score] 写入 {run_save_dir}/score.json 与 score.csv")
    # 打印摘要：overall_acc（=exact_match，最严格）+ AST 5 口径 + 通过数/总数
    # 通过观察 superset vs subset 的差异可判断模型偏「多调」（superset 低→多调无关）还是
    # 偏「少调」（subset 低→漏调）；top1/top3 反映最低命中门槛下的通过率。
    for key, sd in score_dict.items():
        ast_ov = sd["ast"]["overall"]
        print(
            f"  {key}: overall_acc={sd['overall_accuracy']:.4f} "
            f"AST[exact={ast_ov['exact_match']:.4f} superset={ast_ov['superset_match']:.4f} "
            f"subset={ast_ov['subset_match']:.4f} top1={ast_ov['top1_match']:.4f} "
            f"top3={ast_ov['top3_match']:.4f}] "
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
