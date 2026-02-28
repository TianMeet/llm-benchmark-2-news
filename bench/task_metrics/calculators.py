"""指标计算函数库 — 8 种指标计算器，由 GenericTask.metrics() 通过 compute_metric() 统一分发。

每个 calc_* 函数接受 (rule, sample, parsed) 并返回 dict[str, Any]，
详细参数规范见 docs/eval_data_contract.md §2 指标类型详解。

指标对应关系（与 unified_answer_contract.md 对齐）：
  choice   -> exact_match
  number   -> numeric_error (MAE + within_tolerance)
  list     -> list_overlap  (precision / recall / f1)
  text     -> reference_rouge (ROUGE-1 F1)
  reasoning -> reasoning_consistency + claim_evidence_overlap

公共辅助函数：
- normalize_value()       : 标准化值（降低大小写/空白格式噪声）
- safe_float()            : 尽力转为 float，不可转换返回 None
- extract_text_fragments(): 递归提取文本片段
- as_set_of_str()         : 将各类形态转为字符串集合
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 公共辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def normalize_value(value: Any) -> Any:
    """标准化值，降低字符串比较中的大小写/空白格式噪声。"""
    if isinstance(value, str):
        return " ".join(value.strip().lower().split())
    return value


def safe_float(value: Any) -> float | None:
    """尽力将值转为 float，不可转换返回 None。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_text_fragments(value: Any) -> list[str]:
    """递归提取文本片段，用于 reasoning 一致性判断。"""
    texts: list[str] = []
    if isinstance(value, str):
        if value.strip():
            texts.append(value)
    elif isinstance(value, list):
        for item in value:
            texts.extend(extract_text_fragments(item))
    elif isinstance(value, dict):
        for key in ("text", "summary", "rationale", "analysis", "content", "conclusion", "reason", "claim"):
            if key in value:
                texts.extend(extract_text_fragments(value.get(key)))
    return texts


def as_set_of_str(values: Any, *, normalize: bool = True) -> set[str]:
    """将值转换为字符串集合，支持 list[str] / list[dict(id)] / str。"""
    items: set[str] = set()
    if isinstance(values, str):
        candidate = normalize_value(values) if normalize else values
        if isinstance(candidate, str) and candidate:
            items.add(candidate)
        return items
    if not isinstance(values, list):
        return items
    for v in values:
        if isinstance(v, str):
            candidate = normalize_value(v) if normalize else v
            if isinstance(candidate, str) and candidate:
                items.add(candidate)
        elif isinstance(v, dict) and "id" in v:
            raw = str(v.get("id", ""))
            candidate = normalize_value(raw) if normalize else raw
            if isinstance(candidate, str) and candidate:
                items.add(candidate)
    return items


# ─────────────────────────────────────────────────────────────────────────────
# 各类型指标计算函数
# ─────────────────────────────────────────────────────────────────────────────

def calc_field_completeness(rule: dict, sample: dict, parsed: dict | None) -> dict[str, Any]:
    """计算 parse 结果中指定字段的填充率。

    注：sample 参数未使用，保留是为了与其他 calc_* 函数签名统一，
    消除 compute_metric() 分发层的 if/else 特殊分支。
    """
    fields = rule.get("fields", [])
    if not fields or not parsed:
        return {"field_completeness": 0.0}
    non_empty = 0
    for k in fields:
        val = parsed.get(k)
        if isinstance(val, str):
            non_empty += int(bool(val.strip()))
        elif isinstance(val, list):
            non_empty += int(bool(val))
        else:
            non_empty += int(val is not None)
    return {"field_completeness": round(non_empty / len(fields), 4)}


def calc_reference_rouge(rule: dict, sample: dict, parsed: dict | None) -> dict[str, Any]:
    """计算 ROUGE-1 F1 分数，需要 rouge-score 包。"""
    out_field = rule.get("output_field", "summary")
    ref_field = rule.get("reference_field", "reference_summary")
    hypothesis = (parsed or {}).get(out_field, "") if parsed else ""
    reference = sample.get(ref_field, "")
    if not hypothesis or not reference:
        return {"rouge1": 0.0}
    try:
        from rouge_score import rouge_scorer  # type: ignore
        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=False)
        scores = scorer.score(reference, hypothesis)
        return {"rouge1": round(scores["rouge1"].fmeasure, 4)}
    except ImportError:
        logger.warning("rouge_score not installed; skipping rouge metric. pip install rouge-score")
        return {"rouge1": None}


def calc_custom_ratio(rule: dict, sample: dict, parsed: dict | None) -> dict[str, Any]:
    """计算任意分子/分母字段之比（归一化指标）。"""
    name = rule.get("name", "ratio")
    numerator_field = rule.get("numerator_field", "")
    if "denominator_fixed" in rule:
        denom = float(rule["denominator_fixed"] or 1)
    else:
        denom_field = rule.get("denominator_sample_field", "")
        denom_raw = sample.get(denom_field, [])
        denom = float(len(denom_raw)) if isinstance(denom_raw, list) else float(denom_raw or 1)
    raw_num = (parsed or {}).get(numerator_field, 0) if parsed else 0
    try:
        numerator = abs(float(raw_num))
    except (TypeError, ValueError):
        numerator = 0.0
    return {name: round(numerator / max(denom, 1), 4)}


def calc_exact_match(rule: dict, sample: dict, parsed: dict | None) -> dict[str, Any]:
    """计算分类/标签预测准确率（0 或 1）。"""
    pred_field = rule.get("pred_field", "")
    label_field = rule.get("label_field", "")
    name = rule.get("name", f"{pred_field}_exact_match")
    do_normalize = bool(rule.get("normalize", True))

    pred_val = (parsed or {}).get(pred_field) if parsed else None
    label_val = sample.get(label_field)

    if do_normalize:
        pred_val = normalize_value(pred_val)
        label_val = normalize_value(label_val)
    return {name: 1.0 if pred_val == label_val else 0.0}


def calc_numeric_error(rule: dict, sample: dict, parsed: dict | None) -> dict[str, Any]:
    """计算数值 MAE + 容差命中率。"""
    pred_field = rule.get("pred_field", "")
    label_field = rule.get("label_field", "")
    name = rule.get("name", pred_field or "numeric")
    tolerance = rule.get("tolerance")

    pred_val = safe_float((parsed or {}).get(pred_field) if parsed else None)
    label_val = safe_float(sample.get(label_field))

    result: dict[str, Any] = {}
    if pred_val is None or label_val is None:
        result[f"{name}_mae"] = None
        if tolerance is not None:
            result[f"{name}_within_tol"] = 0.0
        return result

    err = abs(pred_val - label_val)
    result[f"{name}_mae"] = round(err, 6)
    if tolerance is not None:
        tol = max(0.0, float(tolerance))
        result[f"{name}_within_tol"] = 1.0 if err <= tol else 0.0
    return result


def calc_list_overlap(rule: dict, sample: dict, parsed: dict | None) -> dict[str, Any]:
    """计算列表重叠 precision / recall / F1。"""
    pred_field = rule.get("pred_field", "")
    label_field = rule.get("label_field", "")
    name = rule.get("name", pred_field or "list")
    do_normalize = bool(rule.get("normalize", True))

    pred_list_raw = (parsed or {}).get(pred_field, []) if parsed else []
    label_list_raw = sample.get(label_field, [])
    pred_list = pred_list_raw if isinstance(pred_list_raw, list) else []
    label_list = label_list_raw if isinstance(label_list_raw, list) else []

    if do_normalize:
        pred_set = {normalize_value(v) for v in pred_list}
        label_set = {normalize_value(v) for v in label_list}
    else:
        pred_set = set(pred_list)
        label_set = set(label_list)

    if not pred_set and not label_set:
        return {f"{name}_precision": 1.0, f"{name}_recall": 1.0, f"{name}_f1": 1.0}

    inter = len(pred_set & label_set)
    precision = inter / len(pred_set) if pred_set else 0.0
    recall = inter / len(label_set) if label_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        f"{name}_precision": round(precision, 4),
        f"{name}_recall": round(recall, 4),
        f"{name}_f1": round(f1, 4),
    }


def calc_reasoning_consistency(rule: dict, sample: dict, parsed: dict | None) -> dict[str, Any]:
    """检查最终答案是否在推理支撑文本中出现（一致性）。"""
    name = rule.get("name", "reasoning")
    final_field = rule.get("final_answer_field", "final_answer")
    support_fields = list(rule.get("support_fields", ["steps", "claims", "rationale"]))
    do_normalize = bool(rule.get("normalize", True))

    final_raw = (parsed or {}).get(final_field) if parsed else None
    if isinstance(final_raw, dict):
        final_raw = final_raw.get("value", "")
    final_text = str(final_raw or "").strip()

    if not final_text:
        return {f"{name}_consistency": 0.0, f"{name}_support_coverage": 0.0}

    support_texts: list[str] = []
    for sf in support_fields:
        support_texts.extend(extract_text_fragments((parsed or {}).get(sf) if parsed else None))

    if not support_texts:
        return {f"{name}_consistency": 0.0, f"{name}_support_coverage": 0.0}

    target = normalize_value(final_text) if do_normalize else final_text
    hits = 0
    for text in support_texts:
        src = normalize_value(text) if do_normalize else text
        if isinstance(src, str) and isinstance(target, str) and target and target in src:
            hits += 1
    coverage = hits / len(support_texts) if support_texts else 0.0
    return {
        f"{name}_consistency": 1.0 if hits > 0 else 0.0,
        f"{name}_support_coverage": round(coverage, 4),
    }


def calc_claim_evidence_overlap(rule: dict, sample: dict, parsed: dict | None) -> dict[str, Any]:
    """计算 claim 中引用的 evidence 与标注 evidence 的重叠 precision/recall/F1。"""
    name = rule.get("name", "evidence")
    claims_field = rule.get("claims_field", "claims")
    claim_evidence_key = rule.get("claim_evidence_key", "evidence")
    label_field = rule.get("label_field", "gt_evidence_ids")
    do_normalize = bool(rule.get("normalize", True))

    claims = (parsed or {}).get(claims_field, []) if parsed else []
    pred_evidence: set[str] = set()
    if isinstance(claims, list):
        for claim in claims:
            if isinstance(claim, dict):
                pred_evidence |= as_set_of_str(claim.get(claim_evidence_key, []), normalize=do_normalize)
            elif isinstance(claim, str):
                pred_evidence |= as_set_of_str([claim], normalize=do_normalize)
    gt_evidence = as_set_of_str(sample.get(label_field, []), normalize=do_normalize)

    if not pred_evidence and not gt_evidence:
        return {f"{name}_precision": 1.0, f"{name}_recall": 1.0, f"{name}_f1": 1.0}

    inter = len(pred_evidence & gt_evidence)
    precision = inter / len(pred_evidence) if pred_evidence else 0.0
    recall = inter / len(gt_evidence) if gt_evidence else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        f"{name}_precision": round(precision, 4),
        f"{name}_recall": round(recall, 4),
        f"{name}_f1": round(f1, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 统一分发入口
# ─────────────────────────────────────────────────────────────────────────────

_CALC_REGISTRY: dict[str, Any] = {
    "field_completeness": calc_field_completeness,
    "reference_rouge": calc_reference_rouge,
    "custom_ratio": calc_custom_ratio,
    "exact_match": calc_exact_match,
    "numeric_error": calc_numeric_error,
    "list_overlap": calc_list_overlap,
    "reasoning_consistency": calc_reasoning_consistency,
    "claim_evidence_overlap": calc_claim_evidence_overlap,
}


def compute_metric(
    mtype: str,
    rule: dict,
    sample: dict,
    parsed: dict | None,
    task_name: str = "",
) -> dict[str, Any]:
    """按指标类型分发到对应计算函数，返回指标结果 dict。

    未知类型记录 warning 并返回空 dict。
    """
    calc = _CALC_REGISTRY.get(mtype)
    if calc is None:
        logger.warning("Unknown metric type '%s' in task '%s'", mtype, task_name)
        return {}

    # 所有 calc_* 函数统一签名 (rule, sample, parsed)，无需按类型分支。
    return calc(rule, sample, parsed)
