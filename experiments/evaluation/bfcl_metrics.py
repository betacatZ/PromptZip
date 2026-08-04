"""BFCL V4 Parallel Multi-Turn 判分器。

移植自 gorilla 官方 ast_checker.py 的核心口径（Python only）：
- standardize_string / string_checker / list_checker / dict_checker
- simple_function_checker（单函数：函数名 + required 参数 + 值合法 + optional 处理）
- parallel_function_checker_no_order（无序并行：数量相等 + 贪心消除匹配）

数据集 OpenMLRL/BFCL-V4-Parallel-Multi-Turn 的 ground_truth 字段正是
官方 possible_answers 的「无序并行」格式（list[{func: {param: [可接受值]}}]），
可直接用 parallel_function_checker_no_order 判分。

不依赖 bfcl_eval 包，独立实现，避免引入额外依赖。
"""

import json
import re
from typing import Any

# Python 类型映射：BFCL 工具 schema 里的 type 字符串 -> Python 类型
PYTHON_TYPE_MAPPING = {
    "string": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "bool": bool,
    "array": list,
    "dict": dict,
    "object": dict,
    "tuple": list,  # tuple 经 json 序列化后变 list，按 list 对待
}
PYTHON_NESTED_TYPE_CHECK_LIST = ["array"]  # 仅 array 需要 items.type 嵌套类型检查


def standardize_string(input_string: str) -> str:
    """标准化字符串：去空格和 ,./-_*^ 标点、lowercase、单引号转双引号。

    与官方 ast_checker.standardize_string 一致，用于字符串值的宽松比对
    （如 "April 1, 2024" == "April 1 2024"）。
    """
    return re.sub(r"[ \,\./\-\_\*\^]", "", input_string).lower().replace("'", '"')


def _get_possible_answer_type(possible_answer: list):
    """从 possible_answer 列表推断期望类型（取第一个非 "" 的项）。"""
    for answer in possible_answer:
        if answer != "":  # "" 表示 optional 参数
            return type(answer)
    return None


def type_checker(
    param: str,
    value: Any,
    possible_answer: list,
    expected_type_description: str,
    expected_type_converted: type,
    nested_type_converted: type | None,
) -> dict:
    """检查参数值的类型是否合法（Python only）。

    与官方 type_checker 对齐：以 possible_answer 的类型为准；嵌套类型（array）递归检查元素。
    """
    possible_answer_type = _get_possible_answer_type(possible_answer)
    if possible_answer_type is not None:
        # float 期望允许 int 自动转 float（与官方一致）
        if possible_answer_type == float and type(value) == int:
            value = float(value)
        if type(value) != possible_answer_type:
            return {
                "valid": False,
                "error": f"Incorrect type for parameter {repr(param)}. Expected {possible_answer_type.__name__}, got {type(value).__name__}. Value: {repr(value)}.",
                "error_type": "type_error:simple",
                "is_variable": False,
            }

    is_variable = False  # 该数据集无 variable 概念，恒 False

    # 嵌套类型检查（array of X）
    if expected_type_description in PYTHON_NESTED_TYPE_CHECK_LIST and nested_type_converted is not None:
        if not isinstance(value, list):
            return {
                "valid": False,
                "error": f"Parameter {repr(param)} expected array, got {type(value).__name__}.",
                "error_type": "type_error:nested",
                "is_variable": is_variable,
            }
        # 元素类型逐一检查（与官方一致：每个元素至少匹配一个 possible answer 类型）
        for possible_answer_item in possible_answer:
            if type(possible_answer_item) == list:
                # 嵌套的 possible answer 项本身是列表，递归
                for sub_item in possible_answer_item:
                    for v in value:
                        if type(v) != type(sub_item):
                            return {
                                "valid": False,
                                "error": f"Nested type mismatch for parameter {repr(param)}.",
                                "error_type": "type_error:nested",
                                "is_variable": is_variable,
                            }

    return {"valid": True, "error": [], "error_type": "", "is_variable": is_variable}


def string_checker(param: str, model_output: str, possible_answer: list) -> dict:
    """字符串值检查：standardize 后看 model_output 是否在 possible_answer 集合里（大小写/标点不敏感）。"""
    standardize_possible_answer = []
    for item in possible_answer:
        if type(item) == str:
            standardize_possible_answer.append(standardize_string(item))
    standardize_model_output = standardize_string(model_output) if type(model_output) == str else model_output

    if standardize_model_output not in standardize_possible_answer:
        return {
            "valid": False,
            "error": f"Invalid value for parameter {repr(param)}: {repr(model_output)}. Expected one of {possible_answer}. Case insensitive.",
            "error_type": "value_error:string",
        }
    return {"valid": True, "error": []}


def list_checker(param: str, model_output: list, possible_answer: list) -> dict:
    """列表值检查：standardize 每个元素后，model_output 是否等于某个 possible_answer 项。"""
    standardize_possible_answer = []
    for item in possible_answer:
        standardize_possible_answer.append([])
        for sub in item:
            if type(sub) == str:
                standardize_possible_answer[-1].append(standardize_string(sub))
            else:
                standardize_possible_answer[-1].append(sub)

    standardize_model_output = []
    for v in model_output:
        if type(v) == str:
            standardize_model_output.append(standardize_string(v))
        else:
            standardize_model_output.append(v)

    if standardize_model_output not in standardize_possible_answer:
        return {
            "valid": False,
            "error": f"Invalid value for parameter {repr(param)}: {repr(model_output)}. Expected one of {possible_answer}.",
            "error_type": "value_error:list",
        }
    return {"valid": True, "error": []}


def dict_checker(param: str, model_output: dict, possible_answers: list) -> dict:
    """字典值检查：遍历每个 possible_answer（单 dict），看是否有一个完全匹配。"""
    result = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}
    for possible_answer in possible_answers:
        flag = True
        for key, value in model_output.items():
            if key not in possible_answer:
                result["valid"] = False
                result["error"] = [f"Unexpected dict key parameter: {repr(key)}."]
                result["error_type"] = "value_error:dict_key"
                flag = False
                break
            standardize_value = standardize_string(value) if type(value) == str else value
            standardize_possible = [
                standardize_string(p) if type(p) == str else p for p in possible_answer[key]
            ]
            if standardize_value not in standardize_possible:
                result["valid"] = False
                result["error"] = [f"Invalid value for dict key {repr(key)}: {repr(value)}. Expected one of {possible_answer[key]}."]
                result["error_type"] = "value_error:dict_value"
                flag = False
                break
        for key, value in possible_answer.items():
            if key not in model_output and "" not in value:
                result["valid"] = False
                result["error"] = [f"Missing dict key parameter: {repr(key)}."]
                result["error_type"] = "value_error:dict_key"
                flag = False
                break
        if flag:
            return {"valid": True, "error": []}
    return result


def _find_description(func_descriptions: list, name: str) -> dict | None:
    """按函数名从工具列表里找对应的工具描述。"""
    for desc in func_descriptions:
        if desc.get("name") == name:
            return desc
    return None


def simple_function_checker(
    func_description: dict,
    model_output: dict,
    possible_answer: dict,
) -> dict:
    """单函数判分：函数名匹配 + required 参数齐全 + 每参数类型与值合法 + optional 处理。

    与官方 simple_function_checker 对齐（Python only）。model_output 形如 {"func": {param: val}}。
    """
    possible_answer = list(possible_answer.values())[0]
    func_name = func_description["name"]
    param_details = func_description["parameters"]["properties"]
    required_params = func_description["parameters"].get("required", [])

    result = {"valid": True, "error": [], "error_type": "simple_function_checker:unclear"}

    # 函数名匹配（model_output 形如 {func_name: {param: val}}）
    if func_name not in model_output:
        return {
            "valid": False,
            "error": [f"Function name {repr(func_name)} not found in model output."],
            "error_type": "simple_function_checker:wrong_func_name",
        }

    model_params = model_output[func_name]

    # required 参数必须在
    for param in required_params:
        if param not in model_params:
            return {
                "valid": False,
                "error": [f"Missing required parameter: {repr(param)}."],
                "error_type": "simple_function_checker:missing_required",
            }

    # 逐参数：类型 + 值
    for param, value in model_params.items():
        if param not in param_details or param not in possible_answer:
            return {
                "valid": False,
                "error": [f"Unexpected parameter: {repr(param)}."],
                "error_type": "simple_function_checker:unexpected_param",
            }

        full_param_details = param_details[param]
        expected_type_description = full_param_details["type"]
        expected_type_converted = PYTHON_TYPE_MAPPING.get(expected_type_description, object)
        nested_type_converted = None
        if expected_type_description in PYTHON_NESTED_TYPE_CHECK_LIST:
            nested_type = full_param_details.get("items", {}).get("type")
            if nested_type:
                nested_type_converted = PYTHON_TYPE_MAPPING.get(nested_type, object)

        # tuple -> list 兼容
        if expected_type_description == "tuple" and type(value) == tuple:
            value = list(value)
        # int -> float 自动转换
        if expected_type_description == "float" and type(value) == int:
            value = float(value)

        type_check_result = type_checker(
            param,
            value,
            possible_answer[param],
            expected_type_description,
            expected_type_converted,
            nested_type_converted,
        )
        if not type_check_result["valid"]:
            return type_check_result
        is_variable = type_check_result["is_variable"]

        if not is_variable:
            # dict 值
            if expected_type_converted == dict:
                r = dict_checker(param, value, possible_answer[param])
                if not r["valid"]:
                    return r
                continue
            # 字符串值
            if expected_type_converted == str:
                r = string_checker(param, value, possible_answer[param])
                if not r["valid"]:
                    return r
                continue
            # 列表值
            if expected_type_converted == list:
                r = list_checker(param, value, possible_answer[param])
                if not r["valid"]:
                    return r
                continue

        # 其他类型：直接看值是否在 possible_answer 里
        if value not in possible_answer[param]:
            return {
                "valid": False,
                "error": f"Invalid value for parameter {repr(param)}: {repr(value)}. Expected one of {possible_answer[param]}.",
                "error_type": "value_error:others",
            }

    # optional 参数未提供但未标记 optional
    for param in possible_answer:
        if param not in model_params and "" not in possible_answer[param]:
            return {
                "valid": False,
                "error": [f"Optional parameter {repr(param)} not provided and not marked as optional."],
                "error_type": "simple_function_checker:missing_optional",
            }

    return result


def parallel_function_checker_no_order(
    func_descriptions: list,
    model_output: list,
    possible_answers: list,
) -> dict:
    """无序并行判分：函数数量相等 + 每个 ground_truth 函数在剩余模型输出里找到匹配（贪心消除）。

    与官方 parallel_function_checker_no_order 对齐。
    model_output: list[{func_name: {param: val}}]
    possible_answers: list[{func_name: {param: [可接受值]}}]
    """
    if len(model_output) != len(possible_answers):
        return {
            "valid": False,
            "error": [f"Wrong number of functions. Model: {len(model_output)}, GT: {len(possible_answers)}."],
            "error_type": "parallel_function_checker_no_order:wrong_count",
        }

    matched_indices = []
    for i in range(len(possible_answers)):
        func_name_expected = list(possible_answers[i].keys())[0]
        func_description = _find_description(func_descriptions, func_name_expected)
        if func_description is None:
            return {
                "valid": False,
                "error": [f"Function {repr(func_name_expected)} not found in provided function descriptions."],
                "error_type": "parallel_function_checker_no_order:func_not_in_desc",
            }

        all_errors = []
        result = None
        for index in range(len(model_output)):
            if index in matched_indices:
                continue
            result = simple_function_checker(
                func_description,
                model_output[index],
                possible_answers[i],
            )
            if result["valid"]:
                matched_indices.append(index)
                break
            else:
                all_errors.append(
                    {
                        f"Model Result Index {index}": {
                            "sub_error": result["error"],
                            "sub_error_type": result["error_type"],
                            "model_output_item": model_output[index],
                            "possible_answer_item": possible_answers[i],
                        }
                    }
                )

        if not result["valid"]:
            considered = [j for j in range(len(model_output)) if j not in matched_indices]
            all_errors.insert(
                0,
                f"Could not find a matching function among index {considered} of model output for index {i} of possible answers.",
            )
            return {
                "valid": False,
                "error": all_errors,
                "error_type": "parallel_function_checker_no_order:cannot_find_match",
            }

    return {"valid": True, "error": []}


# -------------------- 模型输出解析 --------------------


def parse_model_output(text: str) -> list[dict]:
    """从模型生成文本里解析出函数调用列表，归一化为 [{func_name: {param: val}}]。

    支持三种模型输出形态：
      1. Hermes/Qwen tool_call 标签（可能有多个），或特殊 token 形式
         <|tool_call|>{...}<|/tool_call|>
      2. BFCL 原生 JSON 数组：[{"name":"cd","arguments":{...}}, ...]
      3. 单个 JSON 对象 / markdown fence / 前置自然语言等容错。

    返回统一为 [{func_name: {param: val}}] 形式（与 official ast_checker 输入一致）。
    """
    if not text:
        return []

    s = text.strip()
    normalized = []

    # ---- 1) 优先解析 tool_call 标签（Hermes/Qwen 格式，可能有多个）----
    # 用普通字符串变量拼标签，避免源码出现裸标签字面量
    open_tag = chr(60) + "tool" + chr(62)        # <tool>
    close_tag = chr(60) + "/tool" + chr(62)     # </tool>
    tag_re = re.compile(re.escape(open_tag) + r"\s*(.*?)\s*" + re.escape(close_tag), re.DOTALL)
    bodies = [m.group(1) for m in tag_re.finditer(s)]
    if not bodies:
        # <|tool_call|> ... <|/tool_call|> 形式
        ot1 = "<|tool_call|>"
        ct1 = "<|/tool_call|>"
        tag_re2 = re.compile(re.escape(ot1) + r"\s*(.*?)\s*(?:" + re.escape(ct1) + r"|$)", re.DOTALL)
        bodies = [m.group(1) for m in tag_re2.finditer(s)]
    for body in bodies:
        obj = _parse_single_json_object(body)
        if obj is not None:
            _append_normalized(normalized, obj)
    if normalized:
        return normalized

    # ---- 2/3) fallback：去 markdown fence 后找 JSON 数组/对象 ----
    fence_match = re.search(r"```(?:json|python|tool|function)?\s*(.*?)```", s, re.DOTALL)
    if fence_match:
        s = fence_match.group(1).strip()

    # 找第一个 JSON 数组 [ ... ]
    arr_match = re.search(r"\[.*\]", s, re.DOTALL)
    candidate = arr_match.group(0) if arr_match else s

    parsed = None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        # 容错：找第一个 JSON 对象 { ... }
        obj_match = re.search(r"\{.*\}", s, re.DOTALL)
        if obj_match:
            parsed = _parse_single_json_object(obj_match.group(0))

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []

    for item in parsed:
        _append_normalized(normalized, item)

    return normalized


def _parse_single_json_object(text: str) -> dict | None:
    """容错解析单个 JSON 对象，处理尾随逗号等小问题。"""
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        cleaned = text.strip().rstrip(",").strip()
        try:
            obj = json.loads(cleaned)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None


def _append_normalized(normalized: list, item: dict) -> None:
    """把一个工具调用对象归一化成 {func_name: {param: val}} 并追加到 normalized。"""
    if not isinstance(item, dict):
        return
    # 兼容 name/function + arguments/parameters
    func_name = item.get("name") or item.get("function")
    args = item.get("arguments")
    if args is None:
        args = item.get("parameters")
    if args is None:
        args = {}
    if func_name is None:
        # 也许整个 item 就是 {func_name: {param: val}} 形式
        if len(item) == 1:
            func_name = list(item.keys())[0]
            args = item[func_name]
        else:
            return
    if not isinstance(args, dict):
        args = {}
    normalized.append({func_name: args})


# -------------------- 行级判分入口 --------------------


def evaluate_row(
    func_descriptions: list,
    model_output_text: str,
    ground_truth: list,
) -> dict:
    """对一条 eval row 判分。

    Args:
        func_descriptions: 该行 function 字段（工具列表）。
        model_output_text: 模型生成的原始文本。
        ground_truth: 该行 ground_truth 字段，list[{func: {param: [可接受值]}}]，
                      空列表表示该轮应不输出函数调用。

    Returns:
        {valid: bool, error_type: str, error: list/str, model_output: list}
    """
    model_output = parse_model_output(model_output_text)

    # 空 ground_truth：模型应不输出任何函数调用
    if not ground_truth:
        if not model_output:
            return {"valid": True, "error_type": "", "error": [], "model_output": model_output}
        return {
            "valid": False,
            "error_type": "irrelevance_error:decoder_success",
            "error": ["Model outputs valid function calls when it should not."],
            "model_output": model_output,
        }

    if not model_output:
        return {
            "valid": False,
            "error_type": "empty_model_response",
            "error": ["Model response is empty or no valid function call parsed."],
            "model_output": model_output,
        }

    result = parallel_function_checker_no_order(func_descriptions, model_output, ground_truth)
    return {
        "valid": result["valid"],
        "error_type": result.get("error_type", ""),
        "error": result.get("error", []),
        "model_output": model_output,
    }


def compute_accuracy(results: list[dict]) -> dict:
    """从 per-row 判分结果计算 overall + per-official_category + per-task_type 准确率。

    results: list[{valid, official_category, task_type, ...}]
    """
    from collections import defaultdict

    def _acc(items):
        if not items:
            return 0.0
        return sum(1 for r in items if r.get("valid")) / len(items)

    by_cat = defaultdict(list)
    by_tt = defaultdict(list)
    for r in results:
        by_cat[r.get("official_category", "unknown")].append(r)
        by_tt[r.get("task_type", "unknown")].append(r)

    # 错误类型分布
    err_dist = defaultdict(int)
    for r in results:
        if not r.get("valid"):
            err_dist[r.get("error_type", "unknown")] += 1

    return {
        "overall_accuracy": _acc(results),
        "num_samples": len(results),
        "by_category": {k: {"accuracy": _acc(v), "num": len(v)} for k, v in by_cat.items()},
        "by_task_type": {k: {"accuracy": _acc(v), "num": len(v)} for k, v in by_tt.items()},
        "error_type_distribution": dict(err_dist),
    }
