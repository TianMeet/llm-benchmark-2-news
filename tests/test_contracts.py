"""bench.contracts 模块契约测试（P1: ResultRow 强类型化）。"""

from __future__ import annotations

import dataclasses
import pytest
from bench.contracts.result import UnifiedCallResult, ResultRow


# ── ResultRow 强类型契约 ──────────────────────────────────────────────

def test_result_row_is_dataclass() -> None:
    """P1: ResultRow 必须是 dataclass，保证字段有类型注解且可做静态分析。"""
    assert dataclasses.is_dataclass(ResultRow), "ResultRow 应是 @dataclass"


def test_result_row_required_fields_present() -> None:
    """P1: ResultRow 应包含评测系统所有核心字段。"""
    fields = {f.name for f in dataclasses.fields(ResultRow)}
    required = {
        "schema_version",
        "run_id",
        "mode",
        "sample_id",
        "task_name",
        "task_version",
        "step_id",
        "prompt_version",
        "model_id",
        "success",
        "error",
        "parse_applicable",
        "parse_success",
        "latency_ms",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "retry_count",
        "cost_estimate",
        "from_cache",
        "output_text",
    }
    missing = required - fields
    assert not missing, f"ResultRow 缺少字段：{missing}"


def test_result_row_to_dict_round_trips() -> None:
    """P1: ResultRow 可以通过 asdict() 转为 dict，字段无丢失。"""
    row = ResultRow(
        schema_version="result_row.v1",
        run_id="test_run",
        mode="task",
        sample_id="S001",
        task_name="ie_json",
        task_version="v1",
        step_id="__task__",
        prompt_version="v1",
        model_id="m1",
        params={},
        repeat_idx=0,
        success=True,
        error="",
        parse_applicable=True,
        parse_success=True,
        latency_ms=123.4,
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        retry_count=0,
        cost_estimate=0.0,
        from_cache=False,
        output_text='{"ticker":"AAPL"}',
        parsed={"ticker": "AAPL"},
        task_metrics={"field_completeness": 1.0},
    )
    d = dataclasses.asdict(row)
    assert d["task_name"] == "ie_json"
    assert d["success"] is True
    assert d["cost_estimate"] == 0.0


def test_result_row_from_cache_cost_invariant() -> None:
    """P0: from_cache=True 的 ResultRow cost_estimate 必须为 0（不变量）。

    这个测试确认业务规则已被编码到数据契约级别。
    如果将来有任何代码路径将 from_cache=True + cost>0 写入，此测试应当失败。
    """
    # __post_init__ 在构造时自动触发 validate()，直接在构造处捕获异常。
    with pytest.raises((ValueError, AssertionError)):
        ResultRow(
            schema_version="result_row.v1",
            run_id="r",
            mode="task",
            sample_id="s",
            task_name="ie_json",
            task_version="v1",
            step_id="__task__",
            prompt_version="v1",
            model_id="m1",
            params={},
            repeat_idx=0,
            success=True,
            error="",
            parse_applicable=True,
            parse_success=True,
            latency_ms=0.5,
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            retry_count=0,
            cost_estimate=0.001,   # ← 非零成本
            from_cache=True,       # ← 同时 from_cache=True：违反不变量
            output_text="{}",
            parsed={},
            task_metrics={},
        )


def test_result_row_valid_invariant_no_error() -> None:
    """from_cache=False + cost>0 是合法的，validate() 不应抛错。"""
    row = ResultRow(
        schema_version="result_row.v1",
        run_id="r",
        mode="task",
        sample_id="s",
        task_name="ie_json",
        task_version="v1",
        step_id="__task__",
        prompt_version="v1",
        model_id="m1",
        params={},
        repeat_idx=0,
        success=True,
        error="",
        parse_applicable=True,
        parse_success=True,
        latency_ms=100.0,
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        retry_count=0,
        cost_estimate=0.001,
        from_cache=False,
        output_text="{}",
        parsed={},
        task_metrics={},
    )
    row.validate()   # 不应抛错
