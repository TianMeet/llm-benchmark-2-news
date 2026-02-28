"""bench/task_metrics/regression.py 单元测试。

验收标准：
- check_regression() 正确判断退化 / 正常 / 提升
- tolerance 参数控制容忍区间
- RegressionResult 包含完整的比较信息
"""

from __future__ import annotations

import pytest

from bench.contracts.scoring import ScoreCard
from bench.task_metrics.regression import RegressionResult, check_regression


class TestCheckRegression:
    def _make_card(self, metric: str, score: float, passed: bool = True) -> ScoreCard:
        return ScoreCard(
            primary_metric=metric,
            primary_score=score,
            passed=passed,
            sub_scores={metric: score},
        )

    def test_no_regression(self) -> None:
        """分数提升 → 不退化。"""
        baseline = self._make_card("f1", 0.80)
        current = self._make_card("f1", 0.85)
        result = check_regression(current, baseline)
        assert result.regressed is False
        assert result.delta == pytest.approx(0.05, abs=1e-6)
        assert result.details == "OK"

    def test_regression_detected(self) -> None:
        """分数大幅下降 → 退化。"""
        baseline = self._make_card("f1", 0.80)
        current = self._make_card("f1", 0.70)
        result = check_regression(current, baseline)
        assert result.regressed is True
        assert result.delta == pytest.approx(-0.10, abs=1e-6)
        assert "REGRESSION" in result.details

    def test_within_tolerance(self) -> None:
        """分数微降但在容忍范围内 → 不退化。"""
        baseline = self._make_card("f1", 0.80)
        current = self._make_card("f1", 0.79)
        result = check_regression(current, baseline, tolerance=0.02)
        assert result.regressed is False

    def test_exact_boundary(self) -> None:
        """delta 恰好等于 -tolerance → 不退化（含边界）。"""
        baseline = self._make_card("f1", 0.80)
        current = self._make_card("f1", 0.78)
        result = check_regression(current, baseline, tolerance=0.02)
        assert result.regressed is False

    def test_just_beyond_tolerance(self) -> None:
        """略超容忍范围 → 退化。"""
        baseline = self._make_card("f1", 0.80)
        current = self._make_card("f1", 0.7799)
        result = check_regression(current, baseline, tolerance=0.02)
        assert result.regressed is True

    def test_custom_tolerance(self) -> None:
        """自定义大容忍范围。"""
        baseline = self._make_card("f1", 0.80)
        current = self._make_card("f1", 0.70)
        result = check_regression(current, baseline, tolerance=0.15)
        assert result.regressed is False

    def test_result_has_all_fields(self) -> None:
        baseline = self._make_card("f1", 0.80)
        current = self._make_card("f1", 0.85)
        result = check_regression(current, baseline)
        assert isinstance(result, RegressionResult)
        assert result.baseline_score == 0.80
        assert result.current_score == 0.85

    def test_same_score_no_regression(self) -> None:
        """分数相同 → 不退化。"""
        baseline = self._make_card("f1", 0.80)
        current = self._make_card("f1", 0.80)
        result = check_regression(current, baseline)
        assert result.regressed is False
        assert result.delta == pytest.approx(0.0)
