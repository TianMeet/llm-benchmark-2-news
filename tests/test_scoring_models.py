"""bench/contracts/scoring.py 数据模型单元测试。

验收标准：
- 所有 dataclass 可正确创建 / frozen 不可变
- MetricRule.to_rule_dict() 正确桥接计算器签名
- ScorerPreset.from_dict() 正确解析 YAML 风格 dict
- ScoringProfile.from_dict() 正确做完整配置解析
- NormalizationConfig 默认值正确
"""

from __future__ import annotations

import copy
import dataclasses
import pytest

from bench.contracts.scoring import (
    AnswerType,
    CanonicalAnswer,
    MetricRule,
    NormalizationConfig,
    ScoreCard,
    ScorerPreset,
    ScoringProfile,
)


# ── CanonicalAnswer ──────────────────────────────────────────────────

class TestCanonicalAnswer:
    def test_create_basic(self) -> None:
        a = CanonicalAnswer(
            task_id="ie_json",
            sample_id="S001",
            answer_type="choice",
            prediction={"value": "positive"},
            ground_truth={"value": "positive"},
        )
        assert a.task_id == "ie_json"
        assert a.sample_id == "S001"
        assert a.answer_type == "choice"
        assert a.meta == {}  # 默认值

    def test_frozen_immutable(self) -> None:
        a = CanonicalAnswer(
            task_id="t", sample_id="s", answer_type="number",
            prediction={}, ground_truth={},
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            a.task_id = "modified"  # type: ignore[misc]

    def test_is_dataclass(self) -> None:
        assert dataclasses.is_dataclass(CanonicalAnswer)

    def test_with_meta(self) -> None:
        a = CanonicalAnswer(
            task_id="t", sample_id="s", answer_type="text",
            prediction={}, ground_truth={},
            meta={"schema_version": "v1", "prompt_version": "ie_json.v1"},
        )
        assert a.meta["schema_version"] == "v1"


# ── ScoreCard ────────────────────────────────────────────────────────

class TestScoreCard:
    def test_create_basic(self) -> None:
        sc = ScoreCard(
            primary_metric="exact_match",
            primary_score=1.0,
            passed=True,
            sub_scores={"exact_match": 1.0, "f1": 0.8},
        )
        assert sc.primary_score == 1.0
        assert sc.passed is True
        assert sc.error_tags == []  # 默认值
        assert sc.explain == ""     # 默认值

    def test_is_dataclass(self) -> None:
        assert dataclasses.is_dataclass(ScoreCard)

    def test_mutable_for_post_update(self) -> None:
        """ScoreCard 不 frozen — 允许 build 后补充 error_tags / explain。"""
        sc = ScoreCard(
            primary_metric="f1", primary_score=0.5,
            passed=False, sub_scores={},
        )
        sc.error_tags = ["wrong_unit"]
        sc.explain = "unit mismatch"
        assert sc.error_tags == ["wrong_unit"]


# ── MetricRule ───────────────────────────────────────────────────────

class TestMetricRule:
    def test_to_rule_dict_basic(self) -> None:
        rule = MetricRule(type="exact_match", name="sentiment_acc", extras={
            "pred_field": "sentiment",
            "label_field": "gt_sentiment",
        })
        d = rule.to_rule_dict()
        assert d == {
            "type": "exact_match",
            "name": "sentiment_acc",
            "pred_field": "sentiment",
            "label_field": "gt_sentiment",
        }

    def test_to_rule_dict_no_name(self) -> None:
        rule = MetricRule(type="field_completeness", extras={"fields": ["f1"]})
        d = rule.to_rule_dict()
        assert d == {"type": "field_completeness", "fields": ["f1"]}
        assert "name" not in d

    def test_frozen(self) -> None:
        rule = MetricRule(type="exact_match")
        with pytest.raises(dataclasses.FrozenInstanceError):
            rule.type = "other"  # type: ignore[misc]


# ── ScorerPreset ─────────────────────────────────────────────────────

class TestScorerPreset:
    def test_from_dict_basic(self) -> None:
        raw = {
            "primary_metric": "exact_match",
            "pass_threshold": 0.8,
            "metrics": [
                {"type": "exact_match", "name": "acc", "pred_field": "v", "label_field": "gt_v"},
            ],
        }
        preset = ScorerPreset.from_dict(raw)
        assert preset.primary_metric == "exact_match"
        assert preset.pass_threshold == 0.8
        assert len(preset.metrics) == 1
        assert preset.metrics[0].type == "exact_match"
        assert preset.metrics[0].name == "acc"
        assert preset.metrics[0].extras == {"pred_field": "v", "label_field": "gt_v"}

    def test_from_dict_default_threshold(self) -> None:
        raw = {"primary_metric": "f1", "metrics": []}
        preset = ScorerPreset.from_dict(raw)
        assert preset.pass_threshold == 0.5  # 默认值

    def test_metrics_is_tuple(self) -> None:
        """metrics 应为 tuple（不可变），不允许运行时追加。"""
        raw = {"primary_metric": "f1", "metrics": [{"type": "list_overlap"}]}
        preset = ScorerPreset.from_dict(raw)
        assert isinstance(preset.metrics, tuple)


# ── NormalizationConfig ──────────────────────────────────────────────

class TestNormalizationConfig:
    def test_defaults(self) -> None:
        nc = NormalizationConfig()
        assert nc.lowercase is True
        assert nc.trim_whitespace is True
        assert nc.collapse_spaces is True
        assert nc.unit_aliases == {}

    def test_custom_values(self) -> None:
        nc = NormalizationConfig(lowercase=False, unit_aliases={"percent": ["%", "pct"]})
        assert nc.lowercase is False
        assert nc.unit_aliases["percent"] == ["%", "pct"]


# ── ScoringProfile ───────────────────────────────────────────────────

class TestScoringProfile:
    @pytest.fixture
    def raw_profile(self) -> dict:
        """模拟 scoring_profile.template.yaml 结构。"""
        return {
            "version": "v1",
            "normalization": {
                "lowercase": True,
                "trim_whitespace": True,
                "collapse_spaces": True,
                "unit_aliases": {"percent": ["%", "pct"]},
            },
            "scorers": {
                "choice": {
                    "primary_metric": "exact_match",
                    "pass_threshold": 0.5,
                    "metrics": [
                        {"type": "exact_match", "name": "choice_acc",
                         "pred_field": "value", "label_field": "value"},
                    ],
                },
                "number": {
                    "primary_metric": "within_tol",
                    "pass_threshold": 0.5,
                    "metrics": [
                        {"type": "numeric_error", "name": "value",
                         "pred_field": "value", "label_field": "value",
                         "tolerance": 0.01},
                    ],
                },
            },
        }

    def test_from_dict(self, raw_profile: dict) -> None:
        profile = ScoringProfile.from_dict(raw_profile)
        assert profile.version == "v1"
        assert profile.normalization.lowercase is True
        assert "choice" in profile.scorers
        assert "number" in profile.scorers

    def test_get_preset_exists(self, raw_profile: dict) -> None:
        profile = ScoringProfile.from_dict(raw_profile)
        preset = profile.get_preset("choice")
        assert preset is not None
        assert preset.primary_metric == "exact_match"

    def test_get_preset_missing(self, raw_profile: dict) -> None:
        profile = ScoringProfile.from_dict(raw_profile)
        assert profile.get_preset("nonexistent") is None

    def test_from_dict_does_not_mutate_input(self, raw_profile: dict) -> None:
        """from_dict 不应修改原始 dict。"""
        original = copy.deepcopy(raw_profile)
        ScoringProfile.from_dict(raw_profile)
        assert raw_profile == original

    def test_from_dict_default_normalization(self) -> None:
        """缺少 normalization 字段时应使用默认值。"""
        profile = ScoringProfile.from_dict({"version": "v1", "scorers": {}})
        assert profile.normalization.lowercase is True

    def test_from_dict_default_version(self) -> None:
        """缺少 version 字段时应使用 'v1'。"""
        profile = ScoringProfile.from_dict({"scorers": {}})
        assert profile.version == "v1"
