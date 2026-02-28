"""
ScoreCard 组装逻辑 — 将散装指标 dict 包装为统一 ScoreCard。

仅在需要 ScoreCard 输出（如回归门禁比较）时调用。
GenericTask.metrics() 的默认返回仍为 dict（保持向后兼容）。

主指标查找策略（三级）：
1. 精确匹配 primary_metric 键名
2. 后缀匹配 — 如 "within_tol" 匹配 "value_within_tol"
3. 名称反查 — preset 中 type == primary_metric 的 metric，用其 name 字段做查找
"""

from __future__ import annotations

from typing import Any

from bench.contracts.scoring import ScoreCard, ScorerPreset
from bench.task_metrics.profile import load_profile


def build_score_card(
    raw_scores: dict[str, Any],
    answer_type: str,
    profile_path: str | None = None,
) -> ScoreCard:
    """将计算器返回的散装 dict 包装为 ScoreCard。"""
    profile = load_profile(profile_path)
    preset = profile.get_preset(answer_type)

    if preset is None:
        # 无预设 → 取第一个指标作为 primary
        first_key = next(iter(raw_scores), "")
        return ScoreCard(
            primary_metric=first_key,
            primary_score=float(raw_scores.get(first_key, 0.0) or 0.0),
            passed=False,
            sub_scores=raw_scores,
        )

    primary_score = _find_primary_score(raw_scores, preset)
    return ScoreCard(
        primary_metric=preset.primary_metric,
        primary_score=primary_score,
        passed=primary_score >= preset.pass_threshold,
        sub_scores=raw_scores,
    )


def _find_primary_score(
    raw_scores: dict[str, Any],
    preset: ScorerPreset,
) -> float:
    """在散装指标 dict 中定位主指标值。

    支持三种查找方式（按优先级）：
    1. 精确匹配 primary_metric 键名
    2. 模糊匹配 — 键名以 primary_metric 结尾（适配 {name}_{metric} 命名模式）
    3. 名称反查 — 查找 preset 中 type == primary_metric 的 metric，用其 name 做查找
       （适配 profile 中 primary_metric 是 type 而输出键是 name 的场景）
    """
    key = preset.primary_metric
    if key in raw_scores:
        return float(raw_scores[key] or 0.0)

    # 后缀匹配：如 primary_metric="within_tol" → 匹配 "value_within_tol"
    # 仅匹配 "_{key}" 后缀，避免无分隔符的误匹配（如 key="f1" 匹配 "stuff1"）
    suffix = f"_{key}"
    for k, v in raw_scores.items():
        if k.endswith(suffix):
            return float(v or 0.0)

    # 名称反查：primary_metric="exact_match"(type) → metric.name="choice_acc"(output key)
    for m in preset.metrics:
        if m.type == key and m.name and m.name in raw_scores:
            return float(raw_scores[m.name] or 0.0)

    return 0.0
