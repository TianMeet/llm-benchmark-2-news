"""ScoringProfile 加载与指标解析。

职责：
1. 从 YAML 加载 ScoringProfile（lru_cache 缓存）
2. 加载时校验所有 metric type 是否在 _CALC_REGISTRY 中已注册
3. resolve_metrics() 三层合并：answer_type / task 级 cfg / profile 预设
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from bench.contracts.scoring import ScoringProfile

logger = logging.getLogger(__name__)

_DEFAULT_PROFILE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "configs"
    / "eval_contract"
    / "scoring_profile.template.yaml"
)


@lru_cache(maxsize=4)
def load_profile(path: str | None = None) -> ScoringProfile:
    """加载并缓存 ScoringProfile。

    使用 lru_cache 替代全局可变单例 —— 线程安全、无副作用、可用
    load_profile.cache_clear() 在测试中重置。
    """
    resolved = Path(path) if path else _DEFAULT_PROFILE_PATH
    with resolved.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    profile = ScoringProfile.from_dict(raw)
    _validate_profile(profile)
    return profile


def _validate_profile(profile: ScoringProfile) -> None:
    """启动时校验 profile 中的 metric type 是否都已注册。"""
    from bench.task_metrics.calculators import _CALC_REGISTRY

    for atype, preset in profile.scorers.items():
        for m in preset.metrics:
            if m.type not in _CALC_REGISTRY:
                logger.warning(
                    "ScoringProfile: answer_type '%s' 引用了未注册的指标类型 '%s'",
                    atype,
                    m.type,
                )


def resolve_metrics(
    answer_type: str | None,
    task_metrics_cfg: list[dict[str, Any]],
    profile_path: str | None = None,
) -> list[dict[str, Any]]:
    """解析最终生效的 metrics 列表。

    策略（三层合并）：
    1. answer_type 为 None → 直接返回 task_metrics_cfg（向后兼容）
    2. answer_type 有值、task_metrics_cfg 非空 → task 级配置覆盖 profile 预设
    3. answer_type 有值、task_metrics_cfg 为空 → 使用 profile 预设
    4. answer_type 在 profile 中不存在 → 回退到 task_metrics_cfg 并 WARNING
    """
    if answer_type is None:
        return task_metrics_cfg

    profile = load_profile(profile_path)
    preset = profile.get_preset(answer_type)
    if preset is None:
        logger.warning(
            "answer_type '%s' 在 scoring_profile 中无预设，回退到任务级配置",
            answer_type,
        )
        return task_metrics_cfg

    # 任务级显式配置优先
    if task_metrics_cfg:
        return task_metrics_cfg

    # 使用 profile 预设
    return [m.to_rule_dict() for m in preset.metrics]
