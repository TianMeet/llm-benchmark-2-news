"""Structured exception hierarchy for benchmark pipeline."""

from __future__ import annotations

__all__ = [
    "EvalBenchError",
    "PromptBuildError",
    "GatewayCallError",
    "ParseOutputError",
    "MetricsComputeError",
    "UpstreamDependencyError",
]


class EvalBenchError(Exception):
    """Base exception with stable error metadata."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str,
        error_stage: str,
        error_code: str,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.error_stage = error_stage
        self.error_code = error_code


class PromptBuildError(EvalBenchError):
    def __init__(self, message: str) -> None:
        super().__init__(
            f"build_prompt_error: {message}",
            error_type="prompt_build_failure",
            error_stage="build_prompt",
            error_code="build_prompt_error",
        )


class GatewayCallError(EvalBenchError):
    def __init__(self, message: str, *, error_code: str = "gateway_error") -> None:
        super().__init__(
            f"{error_code}: {message}",
            error_type="gateway_failure",
            error_stage="gateway",
            error_code=error_code,
        )


class ParseOutputError(EvalBenchError):
    def __init__(self, message: str) -> None:
        super().__init__(
            f"parse_error: {message}",
            error_type="parse_failure",
            error_stage="parse",
            error_code="parse_error",
        )


class MetricsComputeError(EvalBenchError):
    def __init__(self, message: str) -> None:
        super().__init__(
            f"metrics_error: {message}",
            error_type="metrics_failure",
            error_stage="metrics",
            error_code="metrics_error",
        )


class UpstreamDependencyError(EvalBenchError):
    def __init__(self, message: str, *, missing: bool) -> None:
        code = "upstream_missing" if missing else "upstream_parse_failed"
        super().__init__(
            f"{code}: {message}",
            error_type="upstream_missing" if missing else "upstream_parse_failed",
            error_stage="upstream",
            error_code=code,
        )
