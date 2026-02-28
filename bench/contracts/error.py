"""结构化错误分类辅助模块。

评测管线中的错误需要被归一化为 (error_type, error_stage, error_code) 三元组，
以支持按类型聚合统计、报告中的失败分布表、以及错误根因分析。

两个入口：
- classify_error()：从错误文本关键词推断错误分类（兜底路径）。
- derive_error_fields()：优先从结构化异常(EvalBenchError)提取，否则回退到文本分类。

错误阶段(error_stage)枚举：
  build_prompt → gateway → parse → metrics → upstream → unknown
"""

from __future__ import annotations

__all__ = ["classify_error", "derive_error_fields"]

from typing import Any

from bench.contracts.exceptions import EvalBenchError


def classify_error(error_message: str) -> tuple[str, str, str]:
    """从错误文本关键词推断 (error_type, error_stage, error_code) 三元组。

    这是兜底分类路径——当调用方没有传入结构化异常时使用。
    关键词匹配顺序决定优先级：prompt > gateway > parse > metrics > upstream。
    """
    text = (error_message or "").strip()
    lower = text.lower()
    if not text:
        return ("", "", "")
    if "build_prompt_error" in lower:
        return ("prompt_build_failure", "build_prompt", "build_prompt_error")
    if "gateway_error" in lower:
        if "timeout" in lower:
            return ("timeout", "gateway", "gateway_timeout")
        if "rate limit" in lower or "429" in lower:
            return ("rate_limit", "gateway", "gateway_rate_limit")
        if "refusal" in lower or "safety" in lower:
            return ("refusal", "gateway", "gateway_refusal")
        return ("gateway_failure", "gateway", "gateway_error")
    if "parse_error" in lower:
        return ("parse_failure", "parse", "parse_error")
    if "metrics_error" in lower:
        return ("metrics_failure", "metrics", "metrics_error")
    if "upstream_" in lower and "_missing" in lower:
        return ("upstream_missing", "upstream", "upstream_missing")
    if "upstream_" in lower and "_parse_failed" in lower:
        return ("upstream_parse_failed", "upstream", "upstream_parse_failed")
    return ("unknown_error", "unknown", "unknown_error")


def derive_error_fields(
    *,
    success: bool,
    parse_applicable: bool,
    parse_success: bool,
    error_message: str,
    error_type: str = "",
    error_stage: str = "",
    error_code: str = "",
    error_exc: EvalBenchError | None = None,
) -> dict[str, Any]:
    """Return normalized error fields for a result row."""
    if error_exc is not None:
        error_type = error_exc.error_type
        error_stage = error_exc.error_stage
        error_code = error_exc.error_code
    elif not (error_type and error_stage and error_code):
        error_type, error_stage, error_code = classify_error(error_message)
    if not success and not error_type:
        error_type, error_stage, error_code = ("unknown_error", "unknown", "unknown_error")
    # Successful API call but parsing failed/empty structured output.
    if success and parse_applicable and not parse_success and not error_type:
        error_type, error_stage, error_code = ("schema_mismatch", "parse", "parse_empty_or_invalid")
    return {
        "error_type": error_type,
        "error_stage": error_stage,
        "error_code": error_code,
        "error_detail": error_message or "",
    }
