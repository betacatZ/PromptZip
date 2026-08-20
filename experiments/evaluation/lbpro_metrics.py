"""
LongBench-Pro 评测指标。

从官方仓库 caskcsg/longcontext (LongBench-Pro/modules/utils.py + evaluation.py) 逐字移植,
不改变评分口径。仅做最小适配(独立成模块,不依赖官方 modules 包)。

指标函数签名与官方一致:
    metric(answers: List[str], prediction: str) -> float   (Summary 另需 embedding_model/is_zh)
所有返回值在 [0, 1] 区间。

依赖: jieba, rouge, pytrec_eval, sentence_transformers —— 均为 pyproject 已有或需新增(pytrec_eval)。
"""

from typing import List, Dict, Set
from collections import defaultdict
from itertools import combinations
import math


# ======================== 归一化 helpers (逐字移植) ========================

def get_answer_area(text: str) -> str:
    """若 prediction 含 [Answer] / [答案] 标记,只取最后一个标记之后的部分作为答案区。
    这是官方评分的关键逻辑:模型被要求输出 [Answer] 标识,评分只看标识后的内容。
    """
    if "[Answer]" in text or "[答案]" in text:
        if "[Answer]" in text:
            last_answer_start: int = text.rfind("[Answer]")
            if last_answer_start != -1:
                text = text[last_answer_start + 8:]
        else:
            last_answer_start: int = text.rfind("[答案]")
            if last_answer_start != -1:
                text = text[last_answer_start + 4:]
    return text.strip()


def lower(text: str) -> str:
    return text.lower()


def split_by_new_line(text: str) -> List[str]:
    return text.split("\n")


def fix_space(text: str) -> str:
    """不能移除所有空格: "1 11" != "11 1" 但 "111" == "111",故只规整连续空白为单空格。"""
    return " ".join(text.split())


def normalize_answers(answers: List[str]) -> List[str]:
    return [fix_space(lower(a).strip()) for a in answers]


def normalize_prediction(prediction: str) -> List[str]:
    """prediction 先取答案区(get_answer_area),再 lower,再按换行切成 component 列表。"""
    return [fix_space(p.strip()) for p in split_by_new_line(lower(get_answer_area(prediction)))]


def normalize_prediction_abstract(abstract: str) -> str:
    """Summary 任务用:prediction 不切行,整体归一化为一个字符串。"""
    return fix_space(lower(abstract).strip())


# ======================== 指标函数 (逐字移植) ========================

def Accuracy(answers: List[str], prediction: str) -> float:
    answers = normalize_answers(answers)
    predictions = normalize_prediction(prediction)

    if len(answers) == 0 or len(predictions) == 0:
        return 0.0

    if answers[0] == predictions[0]:
        return 1.0
    else:
        return 0.0


def F1_Score(answers: List[str], prediction: str) -> float:
    answers = normalize_answers(answers)
    predictions = normalize_prediction(prediction)

    answer_set: Set[str] = set(answers)
    prediction_set: Set[str] = set(predictions)

    common: Set[str] = answer_set & prediction_set
    if len(common) == 0 or len(prediction_set) == 0 or len(answer_set) == 0:
        return 0.0

    precision: float = len(common) / len(prediction_set)
    recall: float = len(common) / len(answer_set)

    if precision + recall == 0:
        return 0.0

    f1: float = (2 * precision * recall) / (precision + recall)
    return f1


def SubEM(answers: List[str], prediction: str) -> float:
    """子串精确匹配:每个 answer 若作为某条 prediction 的子串出现则计 1,取平均。"""
    answers = normalize_answers(answers)
    predictions = normalize_prediction(prediction)

    if len(answers) == 0 or len(predictions) == 0:
        return 0.0

    score: float = 0.0
    for a in answers:
        if a in predictions:
            score += 1.0
    return score / len(answers)


# Rouge: https://github.com/pltrdy/rouge
def Summary_Max_Rouge_L(answers: List[str], prediction: str, is_zh: bool) -> float:
    if is_zh:
        import jieba
        from rouge import Rouge

        answers = [" ".join(list(jieba.cut(a, cut_all=False))) for a in answers]
        prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))

    rouge_evaluator = Rouge()
    try:
        scores = rouge_evaluator.get_scores([prediction] * len(answers), answers, avg=False)
    except Exception:
        return 0.0

    return max([score["rouge-l"]["f"] for score in scores])


def Summary_Max_Semantic_Similarity(embedding_model, answers: List[str], prediction: str) -> float:
    answer_embeddings = embedding_model.encode(answers)
    prediction_embeddings = embedding_model.encode([prediction])

    # 计算 answer 与 prediction embedding 的余弦相似度
    similarity = embedding_model.similarity(answer_embeddings, prediction_embeddings)  # n * 1
    return float(similarity.max().cpu().numpy())


def Summary(embedding_model, answers: List[str], prediction: str, is_zh: bool, alpha: float = 0.5, beta: float = 0.5) -> float:
    """Summary 任务综合分:0.5 * 语义相似度 + 0.5 * Rouge-L。
    embedding_model 需为 sentence-transformers 模型(支持 .encode / .similarity)。
    """
    answers = normalize_answers(answers)
    prediction = normalize_prediction_abstract(prediction)

    if len(answers) == 0 or not prediction:
        return 0.0

    return alpha * Summary_Max_Semantic_Similarity(embedding_model, answers, prediction) + beta * Summary_Max_Rouge_L(answers, prediction, is_zh)


# NDCG@k: https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/evaluation.py#L67
# 官方用 pytrec_eval。pytrec_eval 是 C 扩展,缺编译环境时装不上;
# 这里优先用 pytrec_eval(与官方结果逐位一致),不可用时退回纯 Python 实现 ndcg_cut.k,
# 公式与 pytrec_eval 完全一致(IDCG 用理想排序的 DCG),只是无二进制依赖。
def _ndcg_cut_pytreceval(rel_answers: dict, run_predictions: dict, k: int) -> float:
    import pytrec_eval

    ndcg_string = "ndcg_cut." + str(k)
    evaluator = pytrec_eval.RelevanceEvaluator(rel_answers, {ndcg_string})
    scores = evaluator.evaluate(run_predictions)
    ndcg = 0.0
    for query_id in scores.keys():
        ndcg += scores[query_id]["ndcg_cut_" + str(k)]
    return ndcg / len(scores)


def _ndcg_cut_pure(rel_answers: dict, run_predictions: dict, k: int) -> float:
    """纯 Python 实现 pytrec_eval 的 ndcg_cut.k,公式一致。
    rel_answers: {query_id: {doc_id: rel}};run_predictions: {query_id: {doc_id: score}}
    DCG@k  = sum(rel_i / log2(i+1)), i 从 1..k(按 prediction 的 score 降序取前 k)
    IDCG@k = 同理但按 rel 降序的理想排序
    NDCG@k = DCG@k / IDCG@k
    """
    def _dcg(rels_in_rank_order, k):
        dcg = 0.0
        for i, rel in enumerate(rels_in_rank_order[:k]):
            dcg += rel / math.log2(i + 2)  # i+2 因为 log2(1)=0,位置 1 对应 log2(2)
        return dcg

    total = 0.0
    n_queries = 0
    for qid, rel_dict in rel_answers.items():
        n_queries += 1
        pred = run_predictions.get(qid, {})
        # 按 prediction score 降序排,取 rel;未被预测的 doc rel=0
        ranked_docs = sorted(pred.keys(), key=lambda d: pred[d], reverse=True)
        gain_seq = [rel_dict.get(d, 0) for d in ranked_docs]
        dcg = _dcg(gain_seq, k)
        # ideal: 所有 rel 降序(含未被预测的,因它们 rel 仍贡献理想排序)
        ideal_seq = sorted(rel_dict.values(), reverse=True)
        idcg = _dcg(ideal_seq, k)
        total += dcg / idcg if idcg > 0 else 0.0
    return total / n_queries if n_queries > 0 else 0.0


def NDCG(answers: List[str], prediction: str) -> float:
    answers = normalize_answers(answers)
    predictions = normalize_prediction(prediction)

    if len(answers) == 0 or len(predictions) == 0:
        return 0.0

    k_value = len(answers)

    # 与官方一致的 relevance 构造:answer 顺序赋予递减 rel(首位最高)
    rel_answers = {
        "query": {a: len(answers) - i for i, a in enumerate(answers)}
    }
    # prediction 顺序赋予递减 score(首位最高),pytrec_eval 据此排序
    run_predictions = {
        "query": {p: len(predictions) - i for i, p in enumerate(predictions)}
    }

    try:
        return _ndcg_cut_pytreceval(rel_answers, run_predictions, k_value)
    except ImportError:
        return _ndcg_cut_pure(rel_answers, run_predictions, k_value)


def Pairwise_Accuracy(answers: List[str], prediction: str) -> float:
    """排序任务:对 answer 中所有两两组合,检查 prediction 中的相对顺序是否一致。"""
    answers = normalize_answers(answers)
    predictions = normalize_prediction(prediction)

    if len(answers) == 0 or len(answers) == 1 or len(predictions) == 0 or len(predictions) == 1:
        return 0.0

    n_total: int = len(predictions) * (len(predictions) - 1) // 2  # prediction 所有可能两两组合数
    prediction_indices: Dict[str, int] = {p: i for i, p in enumerate(predictions)}
    n_correct: int = 0

    for a, b in combinations(answers, 2):
        if a in prediction_indices and b in prediction_indices:
            if prediction_indices[a] < prediction_indices[b]:
                n_correct += 1

    return n_correct / n_total


# ======================== 指标分派 (逐字移植自官方 evaluation.py) ========================

# 23 个 secondary_task -> 指标名。与官方 Evaluator.task_metric_config 完全一致。
task_metric_config: Dict[str, str] = {
    "T1.1 Global Cohesive Retrieval": "NDCG",
    "T1.2 Key-Snippet Retrieval": "NDCG",
    "T2.1 Global Timeline Reconstruction": "Pairwise_Accuracy",
    "T2.2 Local Causal Chain Sorting": "Pairwise_Accuracy",
    "T3.1 Multi-Doc Integration QA": "Accuracy",
    "T3.2 Single-Hop Fact QA": "Accuracy",
    "T4.1 Global-Coverage Constrained Summary": "Summary",
    "T4.2 Query-Focused Summary": "Summary",
    "T5.1 Full-Sentence Citation Alignment": "F1_Score",
    "T5.2 Key-Statement Citation Alignment": "F1_Score",
    "T6.1 Large-Scale Document Clustering": "SubEM",
    "T6.2 Targeted Subset Cluster Identification": "F1_Score",
    "T6.3 Global Frequency Analysis": "Pairwise_Accuracy",
    "T7.1 Global Conflict & Inconsistency Localization": "F1_Score",
    "T7.2 Targeted Rule or Condition Violation Detection": "F1_Score",
    "T7.3 Comprehensive Error & Anomaly Sweep": "F1_Score",
    "T8.1 Structured Multi-Source Consistency Verification": "SubEM",
    "T8.2 Single-Source Targeted Aggregation": "SubEM",
    "T8.3 Long-Context Procedural State Tracking": "SubEM",
    "T9.1 Dependency-Aware Multi-Version Impact Analysis": "F1_Score",
    "T9.2 Localized Interface Change Detection": "F1_Score",
    "T10.1 Large-Scale In-Context Rule Induction": "SubEM",
    "T10.2 Targeted Example-Based Rule Induction": "SubEM",
    "T11.1 Long-Range Entity & Commitment Tracking": "Accuracy",
    "T11.2 Short-Range Reference Resolution & State Query": "Accuracy",
}


def calculate_metric(secondary_task: str, answer: List[str], prediction: str, is_zh: bool, embedding_model=None):
    """计算单样本指标,返回 (success: bool, metric_value: float)。
    与官方 Evaluator.calculate_metric 口径一致;prediction 为空串时返回 (False, 0.0)。
    embedding_model 仅 Summary 任务需要;为 None 时 Summary 任务会失败(返回 0.0)。
    """
    try:
        if prediction == "":
            return False, 0.0

        metric_name: str = task_metric_config[secondary_task]

        if metric_name == "NDCG":
            metric_value = NDCG(answer, prediction)
        elif metric_name == "Pairwise_Accuracy":
            metric_value = Pairwise_Accuracy(answer, prediction)
        elif metric_name == "Accuracy":
            metric_value = Accuracy(answer, prediction)
        elif metric_name == "F1_Score":
            metric_value = F1_Score(answer, prediction)
        elif metric_name == "SubEM":
            metric_value = SubEM(answer, prediction)
        elif metric_name == "Summary":
            if embedding_model is None:
                return False, 0.0
            metric_value = Summary(embedding_model, answer, prediction, is_zh)
        else:
            return False, 0.0

        assert 0.0 <= metric_value <= 1.0, f"Metric {metric_value} is not in [0, 1]"
        return True, metric_value

    except Exception:
        return False, 0.0


# ======================== 聚合 helpers (逐字移植) ========================

def calculate_overall_metrics(metric_results) -> float:
    """总体平均分。metric_results 为含 'metric' 字段的 dict 列表。"""
    if not metric_results:
        return 0.0
    metrics = [m["metric"] for m in metric_results]
    return sum(metrics) / len(metrics)


def calculate_dimension_metrics(metric_results, dimension: str, sort_keys) -> Dict[str, float]:
    """按某维度(token_length/difficulty/primary_task/contextual_requirement/language)分组求平均。
    sort_keys 指定输出顺序;不在结果中的 key 不出现。
    """
    dimension_groups = defaultdict(list)
    for m in metric_results:
        value = m[dimension]
        dimension_groups[value].append(m["metric"])

    results = {}
    for value, metrics in dimension_groups.items():
        results[value] = sum(metrics) / len(metrics)

    return {key: results[key] for key in sort_keys if key in results}


# 维度及其排序键(逐字移植自官方 evaluate_configs)
DIMENSION_CONFIG = {
    "token_length": ["8k", "16k", "32k", "64k", "128k", "256k"],
    "contextual_requirement": ["Full", "Partial"],
    "difficulty": ["Easy", "Moderate", "Hard", "Extreme"],
    "primary_task": [
        "T1. Retrieval & Ranking",
        "T2. Sequencing & Structure Reconstruction",
        "T3. Evidence-Grounded QA",
        "T4. Summarization & Synthesis",
        "T5. Attribution & Citation Alignment",
        "T6. Aggregation & Clustering",
        "T7. Consistency & Compliance Checking",
        "T8. Structured & Numeric Reasoning",
        "T9. Version & Code Diff Analysis",
        "T10. Rule Induction & In-Context Learning",
        "T11. Dialogue Memory & Long-Horizon Tracking",
    ],
    "secondary_task": list(task_metric_config.keys()),
    "language": ["Chinese", "English"],
}
