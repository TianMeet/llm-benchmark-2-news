"""评测管线结构化异常层级。

所有评测专用异常都继承自 EvalBenchError，每个异常自带
error_type / error_stage / error_code 三个元数据字段，
用于在 ResultRow 和报告中做结构化错误聚合。

异常层级：
  EvalBenchError (base)
  ├── PromptBuildError       → build_prompt 阶段
  ├── GatewayCallError       → gateway/API 调用阶段
  ├── ParseOutputError       → 输出解析阶段
  ├── MetricsComputeError    → 指标计算阶段
  └── UpstreamDependencyError → workflow 上游依赖缺失/解析失败
"""

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
    """评测管线基础异常，携带稳定的错误元数据 (type/stage/code)。"""

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
    """build_prompt 阶段异常：模板渲染失败、变量缺失等。"""

    def __init__(self, message: str) -> None:
        super().__init__(
            f"build_prompt_error: {message}",
            error_type="prompt_build_failure",
            error_stage="build_prompt",
            error_code="build_prompt_error",
        )


class GatewayCallError(EvalBenchError):
    """gateway/API 调用阶段异常：超时、限流、HTTP 错误等。"""

    def __init__(self, message: str, *, error_code: str = "gateway_error") -> None:
        super().__init__(
            f"{error_code}: {message}",
            error_type="gateway_failure",
            error_stage="gateway",
            error_code=error_code,
        )


class ParseOutputError(EvalBenchError):
    """输出解析阶段异常：JSON 解析失败、字段校验不通过等。"""

    def __init__(self, message: str) -> None:
        super().__init__(
            f"parse_error: {message}",
            error_type="parse_failure",
            error_stage="parse",
            error_code="parse_error",
        )


class MetricsComputeError(EvalBenchError):
    """指标计算阶段异常：缺少必要字段、类型不匹配等。"""

    def __init__(self, message: str) -> None:
        super().__init__(
            f"metrics_error: {message}",
            error_type="metrics_failure",
            error_stage="metrics",
            error_code="metrics_error",
        )


class UpstreamDependencyError(EvalBenchError):
    """workflow 上游依赖异常：上游步骤缺失或解析失败。"""

    def __init__(self, message: str, *, missing: bool) -> None:
        code = "upstream_missing" if missing else "upstream_parse_failed"
        super().__init__(
            f"{code}: {message}",
            error_type="upstream_missing" if missing else "upstream_parse_failed",
            error_stage="upstream",
            error_code=code,
        )
