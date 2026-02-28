"""bench/task_metrics/scoring.py 单元测试。

验收标准：
- build_score_card() 能正确组装 ScoreCard
- _find_primary_score() 精确匹配 + 后缀模糊匹配
- 无 preset 时取第一个指标作为 primary
- passed 判定基于 pass_threshold
"""

from __future__ import annotations

import pytest
from pathlib import Path

from conftest import SCORING_PROFILE_PATH as PROFILE_PATH

from bench.contracts.scoring import MetricRule, ScoreCard, ScorerPreset
from bench.task_metrics.profile import load_profile
from bench.task_metrics.scoring import _find_primary_score, build_score_card


class TestFindPrimaryScore:
    """测试 _find_primary_score 的匹配逻辑。"""

    def test_exact_match(self) -> None:
        preset = ScorerPreset(
            primary_metric="exact_match",
            pass_threshold=0.5,
            metrics=(MetricRule(type="exact_match"),),
        )
        raw = {"exact_match": 1.0, "f1": 0.8}
        assert _find_primary_score(raw, preset) == 1.0

    def test_suffix_match(self) -> None:
        """primary_metric="within_tol" 应匹配 "value_within_tol"。"""
        preset = ScorerPreset(
            primary_metric="within_tol",
            pass_threshold=0.5,
            metrics=(MetricRule(type="numeric_error"),),
        )
        raw = {"value_mae": 0.05, "value_within_tol": 1.0}
        assert _find_primary_score(raw, preset) == 1.0

    def test_suffix_match_f1(self) -> None:
        """primary_metric="f1" 应匹配 "list_f1"。"""
        preset = ScorerPreset(
            primary_metric="f1",
            pass_threshold=0.5,
            metrics=(MetricRule(type="list_overlap"),),
        )
        raw = {"list_precision": 0.8, "list_recall": 0.7, "list_f1": 0.75}
        assert _find_primary_score(raw, preset) == 0.75

    def test_no_match_returns_zero(self) -> None:
        preset = ScorerPreset(
            primary_metric="nonexistent",
            pass_threshold=0.5,
            metrics=(),
        )
        raw = {"exact_match": 1.0}
        assert _find_primary_score(raw, preset) == 0.0

    def test_none_value_treated_as_zero(self) -> None:
        """值为 None 时转为 0.0 而非崩溃。"""
        preset = ScorerPreset(
            primary_metric="rouge1",
            pass_threshold=0.3,
            metrics=(MetricRule(type="reference_rouge"),),
        )
        raw = {"rouge1": None}
        assert _find_primary_score(raw, preset) == 0.0


class TestBuildScoreCard:
    """测试 build_score_card() 完整组装逻辑。"""

    def setup_method(self) -> None:
        load_profile.cache_clear()

    def test_choice_type(self) -> None:
        raw = {"choice_acc": 1.0}
        sc = build_score_card(raw, answer_type="choice", profile_path=str(PROFILE_PATH))
        assert isinstance(sc, ScoreCard)
        assert sc.primary_metric == "exact_match"
        assert sc.primary_score == 1.0
        assert sc.passed is True
        assert sc.sub_scores == raw

    def test_number_type_pass(self) -> None:
        raw = {"value_mae": 0.005, "value_within_tol": 1.0}
        sc = build_score_card(raw, answer_type="number", profile_path=str(PROFILE_PATH))
        assert sc.primary_metric == "within_tol"
        assert sc.primary_score == 1.0
        assert sc.passed is True

    def test_number_type_fail(self) -> None:
        raw = {"value_mae": 5.0, "value_within_tol": 0.0}
        sc = build_score_card(raw, answer_type="number", profile_path=str(PROFILE_PATH))
        assert sc.primary_score == 0.0
        assert sc.passed is False

    def test_unknown_answer_type_fallback(self) -> None:
        """无 preset 时使用第一个指标。"""
        raw = {"custom_metric": 0.9, "other": 0.5}
        sc = build_score_card(raw, answer_type="nonexistent", profile_path=str(PROFILE_PATH))
        assert sc.primary_metric == "custom_metric"
        assert sc.primary_score == 0.9
        assert sc.passed is False  # 无 preset → passed=False

    def test_empty_raw_scores(self) -> None:
        sc = build_score_card({}, answer_type="choice", profile_path=str(PROFILE_PATH))
        assert sc.primary_score == 0.0
        assert sc.passed is False

    def test_scorecard_error_tags_default_empty(self) -> None:
        sc = build_score_card({"x": 1.0}, answer_type="choice", profile_path=str(PROFILE_PATH))
        assert sc.error_tags == []
        assert sc.explain == ""
