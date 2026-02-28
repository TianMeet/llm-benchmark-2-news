"""回归门禁：对比当前 run 与 baseline run 的 ScoreCard。

提供 check_regression() 纯函数，判断主指标是否退化超出容忍范围。

规则：
- delta = current.primary_score - baseline.primary_score
- delta < -tolerance（默认 0.02）视为退化（不含边界，精度 6 位小数）

使用场景：
  CI 中对比固定 baseline run，确保新提示词/模型/评分者不导致指标退化。
"""

from __future__ import annotations

__all__ = ["RegressionResult", "check_regression"]

from dataclasses import dataclass

from bench.contracts.scoring import ScoreCard


@dataclass(slots=True)
class RegressionResult:
    """回归比较结果。"""

    baseline_score: float
    current_score: float
    delta: float
    regressed: bool
    details: str


def check_regression(
    current: ScoreCard,
    baseline: ScoreCard,
    tolerance: float = 0.02,
) -> RegressionResult:
    """判断主指标是否退化超出容忍范围。

    规则：delta < -tolerance 视为退化（不含边界，精度到 6 位小数）。
    """
    delta = round(current.primary_score - baseline.primary_score, 6)
    regressed = delta < -tolerance
    return RegressionResult(
        baseline_score=baseline.primary_score,
        current_score=current.primary_score,
        delta=delta,
        regressed=regressed,
        details=(
            f"REGRESSION: {current.primary_metric} dropped by {abs(delta):.4f} "
            f"(tolerance={tolerance})"
            if regressed
            else "OK"
        ),
    )
