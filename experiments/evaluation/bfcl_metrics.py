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

    支持多种模型输出形态：
      1. Hermes/Qwen tool_call 标签（可能有多个），或特殊 token 形式
         <|tool_call|>{...}<|/tool_call|>（Qwen2.5 / Qwen3 模板）
      2. Qwen3.5 工具调用格式（chat template 的 qwen3_coder 口径）：
         标签内为 <function=NAME><parameter=KEY>VALUE</parameter></function> 的
         XML 式结构，参数值是原始文本（非 JSON）；复杂类型参数值由模板 tojson 成 JSON。
      3. BFCL 原生 JSON 数组：[{"name":"cd","arguments":{...}}, ...]
      4. 单个 JSON 对象 / markdown fence / 前置自然语言等容错。

    返回统一为 [{func_name: {param: val}}] 形式（与 official ast_checker 输入一致）。
    """
    if not text:
        return []

    s = text.strip()
    normalized = []

    # ---- 1) 优先解析 tool_call 标签（Qwen2.5/Qwen3 Hermes 模板格式，可能有多个）----
    # 标签名含下划线，用 chr 拼接构造，避免源码出现裸标签字面量
    _tc = "tool" + chr(95) + "call"  # tool_call
    open_tag = chr(60) + _tc + chr(62)
    close_tag = chr(60) + chr(47) + _tc + chr(62)
    tag_re = re.compile(re.escape(open_tag) + r"\s*(.*?)\s*" + re.escape(close_tag), re.DOTALL)
    bodies = [m.group(1) for m in tag_re.finditer(s)]
    if not bodies:
        # 特殊 token 形式
        ot1 = "<|" + _tc + "|>"
        ct1 = "<|/" + _tc + "|>"
        tag_re2 = re.compile(re.escape(ot1) + r"\s*(.*?)\s*(?:" + re.escape(ct1) + r"|$)", re.DOTALL)
        bodies = [m.group(1) for m in tag_re2.finditer(s)]
    for body in bodies:
        # 先按 Qwen2.5/Qwen3 的 JSON 对象解析（body 是 {"name":..,"arguments":..}）
        obj = _parse_single_json_object(body)
        if obj is not None:
            _append_normalized(normalized, obj)
        else:
            # 退化到 Qwen3.5 的 XML 式 <function=..><parameter=..>..</parameter></function>
            for call_obj in _parse_qwen35_function_blocks(body):
                _append_normalized(normalized, call_obj)
    if normalized:
        return normalized

    # ---- 2) 去 markdown fence ----
    fence_match = re.search(r"```(?:json|python|tool|function)?\s*(.*?)```", s, re.DOTALL)
    if fence_match:
        s = fence_match.group(1).strip()

    # ---- 3) 先尝试整体解析为 JSON 数组（模型输出干净时直接成功）----
    arr_match = re.search(r"\[.*\]", s, re.DOTALL)
    if arr_match:
        try:
            parsed = json.loads(arr_match.group(0))
            if isinstance(parsed, list):
                for item in parsed:
                    _append_normalized(normalized, item)
                if normalized:
                    return normalized
        except json.JSONDecodeError:
            pass  # 数组被噪声破坏，走逐对象提取

    # ---- 4) 逐个提取顶层 {...} 对象（括号配平），跳过噪声文本 ----
    # 适配样本2：[\nnoise\n{obj1}\nnoise\n{obj2}\n] 这种对象本身完整但数组被噪声破坏的情况
    for obj_str in _iter_top_level_json_objects(s):
        obj = _parse_single_json_object(obj_str)
        if obj is not None:
            _append_normalized(normalized, obj)
    if normalized:
        return normalized

    # ---- 5) 最后兜底：单个 {.*} 贪心匹配 ----
    obj_match = re.search(r"\{.*\}", s, re.DOTALL)
    if obj_match:
        obj = _parse_single_json_object(obj_match.group(0))
        if obj is not None:
            _append_normalized(normalized, obj)

    return normalized


def _iter_top_level_json_objects(s: str):
    """用括号配平从字符串里逐个提取顶层 {...} JSON 对象，跳过非 JSON 噪声。

    正确处理嵌套大括号、字符串内的括号、转义。遇到 { 开始计数，配平到 0 即一个完整对象。
    """
    objects = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] == "{":
            depth = 0
            in_str = False
            esc = False
            start_i = i
            while i < n:
                c = s[i]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                else:
                    if c == '"':
                        in_str = True
                    elif c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            objects.append(s[start_i : i + 1])
                            i += 1
                            break
                i += 1
        else:
            i += 1
    return objects


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


# Qwen3.5 工具调用格式的标签名（chat template qwen3_coder 口径）。
# 用 chr 拼接构造，避免源码出现裸标签字面量被工具误吞。
_FUNC_OPEN_PREFIX = chr(60) + "function="   # <function=
_FUNC_CLOSE = chr(60) + chr(47) + "function" + chr(62)  # </function>
_PARAM_OPEN_PREFIX = chr(60) + "parameter="  # <parameter=
_PARAM_CLOSE = chr(60) + chr(47) + "parameter" + chr(62)  # </parameter>


def _parse_qwen35_function_blocks(body: str) -> list[dict]:
    """解析 Qwen3.5 工具调用格式，返回 list[{func_name: {param: val}}]。

    Qwen3.5 的 chat template（qwen3_coder 口径）让模型在 tool_call 标签里输出：
        <function=NAME>
        <parameter=KEY>VALUE</parameter>
        <parameter=KEY2>VALUE2（可跨多行）</parameter>
        </function>
    可能在一个 body 里出现多个 <function=..>..</function> 块（并行调用）。

    参数值是原始文本（非 JSON）；但模板对 mapping/sequence 类型参数值会先
    tojson 再写入，故值若是合法 JSON 也解析为对象/列表，否则保留原始字符串。
    """
    calls: list[dict] = []
    # 逐个匹配 <function=NAME> ... </function> 块
    func_re = re.compile(
        re.escape(_FUNC_OPEN_PREFIX) + r'([^>\s]+)\s*' + chr(62)  # <function=NAME>
        + r"(.*?)"                                                # body（非贪婪）
        + re.escape(_FUNC_CLOSE),                                  # </function>
        re.DOTALL,
    )
    for fm in func_re.finditer(body):
        func_name = fm.group(1).strip()
        inner = fm.group(2)
        params: dict = {}
        # 逐个匹配 <parameter=KEY>VALUE</parameter>
        param_re = re.compile(
            re.escape(_PARAM_OPEN_PREFIX) + r'([^>\s]+)\s*' + chr(62)  # <parameter=KEY>
            + r"(.*?)"                                                  # value（非贪婪）
            + re.escape(_PARAM_CLOSE),                                   # </parameter>
            re.DOTALL,
        )
        for pm in param_re.finditer(inner):
            key = pm.group(1).strip()
            raw_val = pm.group(2).strip()
            params[key] = _coerce_param_value(raw_val)
        calls.append({func_name: params})
    return calls


def _coerce_param_value(raw_val: str):
    """把 Qwen3.5 参数原始文本值尽量转成 JSON 类型。

    简单标量（int/float/bool/null/str）与复杂类型（object/array）若能被
    json.loads 解析则取解析结果，否则保留原始字符串。这样既兼容模板对
    mapping/sequence 参数的 tojson 渲染，也兼容纯文本值（如城市名、日期）。
    """
    if not raw_val:
        return raw_val
    try:
        return json.loads(raw_val)
    except (json.JSONDecodeError, ValueError):
        return raw_val


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


# -------------------- AST 多口径判分 --------------------
#
# 本组函数在「端到端输出」上计算 5 种 AST 匹配口径。
#
# 「端到端」指：reranker 选工具 → LLM 正常生成函数调用 → 解析 LLM 输出。
# 这里只负责最后一步：拿 LLM 的原始输出文本，解析成函数调用集合 P，
# 与 ground truth G 对比，按 5 种宽松程度递增的口径判分。
#
# 记号：
#   G = ground_truth，即该样本期望的函数调用集合，|G| = n_gt
#   P = LLM 输出解析出的函数调用集合，|P| = n_pred
#   P ∩ G = P 中能和 G 某个调用匹配上的对数，n_matched（函数名+参数完全一致才算匹配）
#
# 匹配算法用「贪心消除」：遍历 G 的每个调用，在剩余未匹配的 P 里找一个能对上的，
# 找到就配对并标记该 P 项已用，继续下一个 G 调用。最终配上的对数即 n_matched = |P ∩ G|。
# 这与官方 parallel_function_checker_no_order 同口径，但放宽了「|P| 必须等于 |G|」的硬约束，
# 因为 superset/subset/top-k 口径允许 P 与 G 数量不等。
#
# 5 种口径（宽松度从高到低不严格单调，但大体如此）：
#   exact_match    P = G              完全一致：数量相等且全部命中，不多不少
#   superset_match G ⊆ P             超集匹配：G 的调用全被命中即可，允许 P 多输出无关调用
#   subset_match   P ⊆ G             子集匹配：P 的调用全在 G 里即可，允许 P 遗漏部分 G 调用
#   top1_match     |P ∩ G| ≥ 1       Top-1：至少命中 1 个 GT 调用
#   top3_match     |P ∩ G| ≥ min(3, |G|)  Top-3：至少命中 3 个；若 |G| < 3 则要求全部命中
#
# 集合关系说明：
#   superset (G ⊆ P) 等价于 n_matched == n_gt（G 的每个调用都被匹配）
#   subset   (P ⊆ G) 等价于 n_matched == n_pred（P 的每个调用都被匹配）
#   exact    (P = G)  等价于 n_matched == n_gt == n_pred（既超集又子集）


def _greedy_match_count(func_descriptions: list, model_output: list, ground_truth: list) -> int:
    """贪心消除：对 G 的每个函数调用，在剩余未匹配的 P 里找匹配，返回匹配数 |P ∩ G|。

    复用 simple_function_checker 做单次匹配判定（函数名 + required 参数齐全 + 每参数类型/值合法），
    单次匹配通过即配对，标记该 P 项已用，不再参与后续匹配。

    与 parallel_function_checker_no_order 的区别：后者要求 |P| == |G|（数量必须相等），
    本函数不要求，只数能配上的对数，因此能支持 superset（P 比 G 多）/subset（P 比 G 少）口径。

    Args:
        func_descriptions: 该样本可见的工具列表（含 name/parameters/required 等），
                           用于查函数描述 + 参数类型检查。
        model_output: 解析后的 LLM 输出函数调用列表，元素形如 {func_name: {param: val}}。
        ground_truth: GT 函数调用列表，元素形如 {func_name: {param: [可接受值]}}。

    Returns:
        int: 成功配对的对数，即 |P ∩ G|。
    """
    matched_pred = set()  # 已被配掉的 P 项下标，避免重复使用
    matched = 0
    for gt_item in ground_truth:
        if not isinstance(gt_item, dict) or not gt_item:
            continue
        func_name = list(gt_item.keys())[0]
        func_desc = _find_description(func_descriptions, func_name)
        if func_desc is None:
            # GT 函数不在可见工具列表里，无法匹配，跳过（该 G 调用必然失配）
            continue
        for j in range(len(model_output)):
            if j in matched_pred:
                continue
            if simple_function_checker(func_desc, model_output[j], gt_item)["valid"]:
                matched_pred.add(j)
                matched += 1
                break  # 该 G 调用已配上，处理下一个 G 调用
    return matched


def compute_ast_metrics(func_descriptions: list, model_output_text: str, ground_truth: list) -> dict:
    """对一条 eval row 计算 5 种 AST 匹配口径（端到端：reranker + LLM 输出）。

    设 G = ground_truth 函数调用集合（|G| = n_gt），P = LLM 输出解析出的函数调用集合（|P| = n_pred），
    匹配要求函数名 + 参数（AST）完全一致，n_matched = |P ∩ G| 由贪心消除算出。

    五种口径：
      exact_match:    P = G                  完全一致，数量相等且全部命中（不多不少）。
                                             判定：n_matched == n_gt 且 n_pred == n_gt。
                                             语义：与官方 parallel_function_checker_no_order 一致，
                                                   是最严格的口径，要求模型不多调、不少调、参数全对。
      superset_match: G ⊆ P                  超集匹配，G 的每个调用都被 P 命中即可，允许 P 多输出无关调用。
                                             判定：n_matched == n_gt。
                                             语义：关心「该调的是不是都调了」，不罚多调（多调常无害，
                                                   如多查一次日历、多打一次日志）。
      subset_match:   P ⊆ G                  子集匹配，P 的每个调用都属于 G 即可，允许 P 遗漏部分 G 调用。
                                             判定：n_matched == n_pred。
                                             语义：关心「调出来的有没有调错的」，不罚少调（少调可后续补，
                                                   但调错往往更危险）。
      top1_match:     |P ∩ G| ≥ 1            Top-1，P 中至少有 1 个调用命中 G。
                                             判定：n_matched >= 1。
                                             语义：最宽松的「有没有命中」口径，只要模型在正确方向上调用了
                                                   至少一个函数就算对。
      top3_match:     |P ∩ G| ≥ min(3, |G|)  Top-3，P 中至少命中 3 个 G 调用；若 |G| < 3 则要求全部命中。
                                             判定：n_matched >= min(3, n_gt)。
                                             语义：多函数调用场景下，衡量「覆盖率」是否过半到 3 个。
                                                   |G| 不足 3 时退化为全命中，避免 GT 本身就少于 3 个时
                                                   top3 永远算不出的退化。

    边界处理：
      - 空 GT（n_gt == 0，该轮应不输出函数调用）：
          P 也为空 → exact/superset/subset 均为 True（正确地什么都没调）；
          P 非空 → exact/superset 为 False（不该调却调了），subset 仍 True（P 非空不属于空 G，
                  故实际为 False——见下条修正）；top1/top3 因无 GT 可命中恒 False。
          注：空 GT 的 subset_match 语义为「P 的输出全在 G 里」，P 非空时 P⊄G，应判 False。
              为避免空集恒真带来的误导，n_gt==0 时 subset_match 仅当 n_pred==0 才 True。
      - 有 GT 但 P 空（n_pred == 0，模型没调出任何函数）：
          exact/superset/top1/top3 均 False（该调的没调），subset True（空集 ⊆ 任意集合恒成立，
          即「没调错」成立，但显然「漏调」了）。

    Args:
        func_descriptions: 该样本可见的工具列表（reranker 压缩后剩下的 + 原始参数 schema）。
        model_output_text: LLM 生成的原始文本（含 tool_call 标签或 JSON 数组）。
        ground_truth: 该样本 GT，list[{func: {param: [可接受值]}}]，空列表表示该轮应不调用。

    Returns:
        dict: {exact_match, superset_match, subset_match, top1_match, top3_match,
               n_matched, n_gt, n_pred}，5 个 bool + 3 个 int（供调试与统计）。
    """
    model_output = parse_model_output(model_output_text)
    n_gt = len(ground_truth) if ground_truth else 0
    n_pred = len(model_output)

    # ---- 边界 1：空 GT（该轮应不输出函数调用）----
    if n_gt == 0:
        # 正确行为是 P 也为空；P 非空 = 不该调却调了（irrelevance）
        p_empty = n_pred == 0
        return {
            "exact_match": p_empty,        # P=G=∅
            "superset_match": p_empty,      # G=∅ ⊆ P 当且仅当 P=∅：G 空时 G⊆P 数学上恒真，
                                            # 但这里对齐「该不调就不调」的语义，仅在 P 也空时记 True，
                                            # 否则模型乱调了无关函数不该算 G 被「全覆盖」。
            "subset_match": p_empty,        # P ⊆ G=∅ 当且仅当 P=∅：P 非空则 P 不属于空 G，
                                            # 即调出了 G 里根本不存在的函数，subset 应判 False。
            "top1_match": False,            # 无 GT 可命中
            "top3_match": False,
            "n_matched": 0,
            "n_gt": 0,
            "n_pred": n_pred,
        }

    # ---- 边界 2：有 GT 但 P 空（模型没调出任何函数）----
    if n_pred == 0:
        return {
            "exact_match": False,          # P=∅ ≠ G（漏调）
            "superset_match": False,       # G 的调用没被命中
            "subset_match": True,          # P=∅ ⊆ G 恒成立（没调错，但漏调）
            "top1_match": False,           # 0 个命中
            "top3_match": False,
            "n_matched": 0,
            "n_gt": n_gt,
            "n_pred": 0,
        }

    # ---- 一般情况：贪心消除算 |P ∩ G| ----
    n_matched = _greedy_match_count(func_descriptions, model_output, ground_truth)
    return {
        # exact：既超集又子集 = 数量相等且全命中
        "exact_match": n_matched == n_gt and n_pred == n_gt,
        # superset (G ⊆ P)：G 的每个调用都被命中（不罚 P 多调）
        "superset_match": n_matched == n_gt,
        # subset (P ⊆ G)：P 的每个调用都命中了 G（不罚 P 少调）
        "subset_match": n_matched == n_pred,
        # top1：至少命中 1 个
        "top1_match": n_matched >= 1,
        # top3：至少命中 3 个；|G| < 3 时退化为全命中（min(3, n_gt)）
        "top3_match": n_matched >= min(3, n_gt),
        "n_matched": n_matched,
        "n_gt": n_gt,
        "n_pred": n_pred,
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


def compute_ast_accuracy(results: list[dict]) -> dict:
    """从 per-row AST 判分结果计算 5 种口径的准确率（overall + per-category + per-task_type）。

    与 compute_accuracy（只算单一 valid/exact 准确率）并列输出，本函数把 5 种口径都算出来，
    便于在 score.json / score.csv 里横向对比不同宽松度下模型的通过率。

    每种口径的准确率 = 该组样本里该口径判 True 的比例，即：
        exact_acc     = Σ(exact_match) / N
        superset_acc  = Σ(superset_match) / N
        subset_acc    = Σ(subset_match) / N
        top1_acc      = Σ(top1_match) / N
        top3_acc      = Σ(top3_match) / N

    口径宽松度（一般情形，非严格单调）：
        exact ≤ superset, exact ≤ subset ≤ top1 ≤ ... ；superset 与 subset 不可比。
    实践中 top1 ≥ top3 ≥ ... ≥ exact，superset/subset 介于其间，具体看模型偏「多调」还是「少调」。

    Args:
        results: list[dict]，每个元素须含 ast 子字典（由 compute_ast_metrics 产出）+
                 official_category + task_type 用于分组。

    Returns:
        dict:
          num_samples: 样本总数
          overall: {5 种口径: accuracy} 全局准确率
          by_category: {category: {5 种口径: accuracy, num}} 按 official_category 分组
          by_task_type: {task_type: {5 种口径: accuracy, num}} 按 task_type 分组
    """
    from collections import defaultdict

    # 5 种口径 key，与 compute_ast_metrics 返回字段一一对应
    METRICS = ["exact_match", "superset_match", "subset_match", "top1_match", "top3_match"]

    def _ast_acc(items, key):
        """算一组样本在指定口径上的准确率（判 True 的占比）。"""
        if not items:
            return 0.0
        return sum(1 for r in items if r.get("ast", {}).get(key, False)) / len(items)

    by_cat = defaultdict(list)
    by_tt = defaultdict(list)
    for r in results:
        by_cat[r.get("official_category", "unknown")].append(r)
        by_tt[r.get("task_type", "unknown")].append(r)

    out = {
        "num_samples": len(results),
        # overall：5 种口径各算一个全局准确率
        "overall": {k: _ast_acc(results, k) for k in METRICS},
        # 按 official_category（如 long_context / simple / multi_hop 等）分组
        "by_category": {
            k: {mk: _ast_acc(v, mk) for mk in METRICS} | {"num": len(v)}
            for k, v in by_cat.items()
        },
        # 按 task_type 分组（如 single_hop / multi_hop / long_context）
        "by_task_type": {
            k: {mk: _ast_acc(v, mk) for mk in METRICS} | {"num": len(v)}
            for k, v in by_tt.items()
        },
    }
    return out
