"""统一评分契约模型。

与 configs/eval_contract/canonical_answer.schema.json 和
configs/eval_contract/scoring_profile.template.yaml 对齐，
使用 frozen dataclasses 保持项目风格一致（无额外依赖）。

核心类型：
- CanonicalAnswer：统一答案对象，跨任务回归时使用。
- ScoreCard：统一评分输出，包含主指标、子指标、通过判定。
- MetricRule / ScorerPreset / ScoringProfile：评分配置的三层结构，
  支持从 YAML 模板构建，实现 answer_type -> 预设指标集合 的映射。
- NormalizationConfig：值标准化配置（大小写/空白/单位同义词）。

设计参考：docs/design/unified_answer_contract.md
"""

from __future__ import annotations

__all__ = [
    "AnswerType",
    "CanonicalAnswer",
    "ScoreCard",
    "MetricRule",
    "ScorerPreset",
    "NormalizationConfig",
    "ScoringProfile",
]

from dataclasses import dataclass, field
from typing import Any, Literal

AnswerType = Literal["choice", "number", "list", "text", "reasoning"]


@dataclass(frozen=True, slots=True)
class CanonicalAnswer:
    """统一答案对象 — 仅在需要跨任务回归或 ScoreCard 输出时构建。"""

    task_id: str
    sample_id: str
    answer_type: AnswerType
    prediction: dict[str, Any]
    ground_truth: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ScoreCard:
    """统一评分输出 — 与 docs/design/unified_answer_contract.md §4 对齐。

    不 frozen — 允许 build 后补充 error_tags / explain。
    """

    primary_metric: str
    primary_score: float
    passed: bool
    sub_scores: dict[str, Any]
    error_tags: list[str] = field(default_factory=list)
    explain: str = ""


@dataclass(frozen=True, slots=True)
class MetricRule:
    """单条指标规则 — scoring_profile 中 metrics 列表的每一项。"""

    type: str
    name: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def to_rule_dict(self) -> dict[str, Any]:
        """转为 _CALC_REGISTRY 计算器所需的 rule dict。"""
        d: dict[str, Any] = {"type": self.type}
        if self.name:
            d["name"] = self.name
        d.update(self.extras)
        return d


@dataclass(frozen=True, slots=True)
class ScorerPreset:
    """某个 answer_type 对应的评分预设。"""

    primary_metric: str
    pass_threshold: float
    metrics: tuple[MetricRule, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScorerPreset:
        """从 YAML 风格 dict 构建 ScorerPreset（纯函数，不修改输入）。"""
        threshold = float(data.get("pass_threshold", 0.5))
        metric_rules: list[MetricRule] = []
        for m in data.get("metrics", []):
            mtype = m["type"]
            mname = m.get("name", "")
            extras = {k: v for k, v in m.items() if k not in ("type", "name")}
            metric_rules.append(MetricRule(type=mtype, name=mname, extras=extras))
        return cls(
            primary_metric=data["primary_metric"],
            pass_threshold=threshold,
            metrics=tuple(metric_rules),
        )


@dataclass(frozen=True, slots=True)
class NormalizationConfig:
    """标准化配置 — 与 scoring_profile.yaml normalization 节对齐。"""

    lowercase: bool = True
    trim_whitespace: bool = True
    collapse_spaces: bool = True
    unit_aliases: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScoringProfile:
    """评分配置文件完整模型。"""

    version: str
    normalization: NormalizationConfig
    scorers: dict[str, ScorerPreset]  # key = answer_type

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ScoringProfile:
        """从 YAML 解析结果构建 ScoringProfile（纯函数，不修改原始 dict）。"""
        norm_raw = raw.get("normalization", {})
        norm = NormalizationConfig(
            lowercase=norm_raw.get("lowercase", True),
            trim_whitespace=norm_raw.get("trim_whitespace", True),
            collapse_spaces=norm_raw.get("collapse_spaces", True),
            unit_aliases=norm_raw.get("unit_aliases", {}),
        )
        scorers: dict[str, ScorerPreset] = {}
        for atype, scorer_data in raw.get("scorers", {}).items():
            scorers[atype] = ScorerPreset.from_dict(scorer_data)
        return cls(
            version=raw.get("version", "v1"),
            normalization=norm,
            scorers=scorers,
        )

    def get_preset(self, answer_type: str) -> ScorerPreset | None:
        """获取指定 answer_type 的评分预设，不存在返回 None。"""
        return self.scorers.get(answer_type)
