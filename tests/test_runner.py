"""Runner 端到端集成测试（全部使用 mock 模式，不发起真实 API 调用）。

覆盖范围：
- task 模式基础运行、task_metrics 进入 summary（R2）
- workflow 正常路径、e2e cost 累积（R3）
- workflow upstream parse 失败触发 skip（R1）
- API key 未解析时前置报错（R4）
- 完整 CLI run() 入口，产物目录结构验证
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from eval.contracts.result import UnifiedCallResult
from eval.cli.runner import run
from eval.execution.gateway import ModelGateway
from eval.execution.task_runner import run_task_mode as _run_task_mode
from eval.execution.workflow_runner import run_workflow_mode as _run_workflow_mode
from eval.metrics.aggregate import aggregate_records
from eval.registry import ModelRegistry
from eval.io.store import RunStore
from eval.workflow import load_workflow
from eval.workflow import WorkflowSpec, WorkflowStep


# ── 辅助：返回 from_cache=True 的 Gateway ────────────────────────────
class CachedResponseGateway(ModelGateway):
    """所有调用均返回 from_cache=True，且 usage 非零，用于验证成本归零逻辑。"""

    def __init__(self, content_map: dict[str, str]) -> None:
        self.content_map = content_map

    async def call(
        self, *, model_id: str, task_name: str, sample: dict, sample_cache_id: str,
        messages: list, params_override: dict,
    ) -> UnifiedCallResult:
        content = self.content_map.get(task_name, self.content_map.get("*", "{}"))
        return UnifiedCallResult(
            content=content,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            latency_ms=0.5,
            raw_response={},
            success=True,
            error="",
            retry_count=0,
            from_cache=True,          # ← 关键：命中缓存
        )


# ── 最小测试注册表（无需真实 API key）────────────────────────────────

_REGISTRY_DATA = {
    "active": "m1",
    "models": {
        "m1": {
            "provider": "openai_compatible",
            "model": "test",
            "api_key": "test-key",
            "base_url": "http://localhost",
            "temperature": 0.0,
            "max_tokens": 256,
        },
        "m2": {
            "provider": "openai_compatible",
            "model": "test",
            "api_key": "test-key",
            "base_url": "http://localhost",
            "temperature": 0.0,
            "max_tokens": 256,
        },
        "m3": {
            "provider": "openai_compatible",
            "model": "test",
            "api_key": "test-key",
            "base_url": "http://localhost",
            "temperature": 0.0,
            "max_tokens": 256,
        },
    },
}

_SAMPLE: dict[str, Any] = {
    "sample_id": "T001",
    "ticker": "AAPL",
    "cutoff_time": "2026-01-01T00:00:00Z",
    "news_items": [
        {
            "title": "Apple Q4 beats expectations",
            "content": "Strong earnings.",
            "publish_time": "2026-01-01T00:00:00Z",
            "source": "Reuters",
        }
    ],
}


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture()
def registry(tmp_path: Path) -> ModelRegistry:
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(_REGISTRY_DATA), encoding="utf-8")
    return ModelRegistry(path)


@pytest.fixture()
def store(tmp_path: Path) -> RunStore:
    return RunStore(tmp_path / "runs")


def _make_workflow_json(tmp_path: Path) -> Path:
    wf = {
        "name": "test_pipeline",
        "version": "v1",
        "steps": [
            {"id": "step1_dedup", "task": "news_dedup", "model_id": "m1", "input_from": "sample"},
            {"id": "step2_extract", "task": "ie_json", "model_id": "m2", "input_from": "step1_dedup"},
            {"id": "step3_score", "task": "stock_score", "model_id": "m3", "input_from": "step2_extract"},
        ],
    }
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps(wf), encoding="utf-8")
    return path


# ── 辅助 Gateway ──────────────────────────────────────────────────────

class FixedGateway(ModelGateway):
    """按 task_name 返回固定内容的测试 gateway；"*" 作为通配键。"""

    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses

    async def call(
        self, *, model_id: str, task_name: str, sample: dict, sample_cache_id: str,
        messages: list, params_override: dict,
    ) -> UnifiedCallResult:
        content = self.responses.get(task_name, self.responses.get("*", "{}"))
        return UnifiedCallResult(
            content=content,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            latency_ms=5.0,
            raw_response={},
            success=True,
            error="",
            retry_count=0,
            from_cache=False,
        )


class BadParseGateway(ModelGateway):
    """step1(news_dedup) 返回无法解析的内容，用于测试 R1 upstream 失败处理。"""

    async def call(
        self, *, model_id: str, task_name: str, sample: dict, sample_cache_id: str,
        messages: list, params_override: dict,
    ) -> UnifiedCallResult:
        content = "NOT VALID JSON AT ALL" if task_name == "news_dedup" else "{}"
        return UnifiedCallResult(
            content=content,
            usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
            latency_ms=3.0,
            raw_response={},
            success=True,
            error="",
            retry_count=0,
            from_cache=False,
        )


class SlowTrackingGateway(ModelGateway):
    """用于验证 workflow 样本级并发是否生效。"""

    def __init__(self, delay_s: float = 0.02) -> None:
        self.delay_s = delay_s
        self.inflight = 0
        self.max_inflight = 0

    async def call(
        self, *, model_id: str, task_name: str, sample: dict, sample_cache_id: str,
        messages: list, params_override: dict,
    ) -> UnifiedCallResult:
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            await asyncio.sleep(self.delay_s)
        finally:
            self.inflight -= 1

        content = json.dumps({"deduped_titles": ["t"], "removed_count": 0})
        return UnifiedCallResult(
            content=content,
            usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
            latency_ms=3.0,
            raw_response={},
            success=True,
            error="",
            retry_count=0,
            from_cache=False,
        )


class _ParseErrorTask:
    name = "parse_error_task"
    version = "v1"
    parse_applicable = True

    def build_prompt(self, sample, context=None):
        return [{"role": "user", "content": "x"}], "v1"

    def parse(self, output):
        raise RuntimeError("parse exploded")

    def metrics(self, sample, output, parsed):
        return {}


# ── Task 模式测试 ─────────────────────────────────────────────────────

def test_task_mode_record_count(registry: ModelRegistry, store: RunStore) -> None:
    """task 模式：1 sample × 2 models = 2 条记录。"""
    ie_resp = json.dumps({
        "ticker": "AAPL", "event_type": "earnings",
        "sentiment": "positive", "impact_score": 3, "summary": "beat",
    })
    records = asyncio.run(
        _run_task_mode(
            run_id=store.run_id,
            store=store,
            dataset=[_SAMPLE],
            registry=registry,
            task_name="ie_json",
            model_ids=["m1", "m2"],
            params_override={},
            repeats=1,
            gateway=FixedGateway({"ie_json": ie_resp}),
        )
    )
    assert len(records) == 2
    assert all(r["success"] for r in records)
    assert all(r["parse_success"] for r in records)
    assert all(r["task_metrics"].get("field_completeness") == 1.0 for r in records)
    assert all(r["schema_version"] == "result_row.v1" for r in records)


def test_task_mode_parse_exception_isolated(monkeypatch: pytest.MonkeyPatch, registry: ModelRegistry, store: RunStore) -> None:
    """task 模式 parse 异常不应中断 run，应落盘失败记录。"""
    monkeypatch.setattr("eval.execution.task_runner.build_task", lambda _: _ParseErrorTask())
    rows = asyncio.run(
        _run_task_mode(
            run_id=store.run_id,
            store=store,
            dataset=[_SAMPLE],
            registry=registry,
            task_name="ie_json",
            model_ids=["m1"],
            params_override={},
            repeats=1,
            gateway=FixedGateway({"*": "{}"}),
        )
    )
    assert len(rows) == 1
    assert rows[0]["success"] is False
    assert rows[0]["parse_success"] is False
    assert "parse_error" in rows[0]["error"]


def test_task_metrics_appear_in_summary(registry: ModelRegistry, store: RunStore) -> None:
    """R2 验证：aggregate_records 应将 task_metrics 聚合为 tm_* 列。"""
    ie_resp = json.dumps({
        "ticker": "AAPL", "event_type": "earnings",
        "sentiment": "positive", "impact_score": 3, "summary": "beat",
    })
    records = asyncio.run(
        _run_task_mode(
            run_id=store.run_id,
            store=store,
            dataset=[_SAMPLE],
            registry=registry,
            task_name="ie_json",
            model_ids=["m1"],
            params_override={},
            repeats=1,
            gateway=FixedGateway({"ie_json": ie_resp}),
        )
    )
    summary = aggregate_records(records)
    assert len(summary) == 1
    assert "tm_field_completeness_avg" in summary[0], "task_metrics should be aggregated into summary"
    assert summary[0]["tm_field_completeness_avg"] == 1.0


# ── Workflow 模式测试 ─────────────────────────────────────────────────

def test_workflow_happy_path_row_counts(tmp_path: Path, registry: ModelRegistry, store: RunStore) -> None:
    """workflow 正常路径：3 个 step 行 + 1 个 e2e 行，e2e_success=True。"""
    dedup_resp = json.dumps({"deduped_titles": ["Apple Q4 beats expectations"], "removed_count": 0})
    ie_resp = json.dumps({"ticker": "AAPL", "event_type": "earnings", "sentiment": "positive", "impact_score": 3, "summary": "beat"})
    score_resp = json.dumps({"score": 70, "rationale": "strong"})
    gateway = FixedGateway({"news_dedup": dedup_resp, "ie_json": ie_resp, "stock_score": score_resp})

    wf = load_workflow(_make_workflow_json(tmp_path))
    records = asyncio.run(
        _run_workflow_mode(
            run_id=store.run_id,
            store=store,
            dataset=[_SAMPLE],
            registry=registry,
            workflow=wf,
            global_params={},
            gateway=gateway,
        )
    )
    step_rows = [r for r in records if r["mode"] == "workflow_step"]
    e2e_rows = [r for r in records if r["mode"] == "workflow_e2e"]
    assert len(step_rows) == 3
    assert len(e2e_rows) == 1
    assert e2e_rows[0]["success"] is True
    assert all(r["schema_version"] == "result_row.v1" for r in records)


def test_workflow_parse_exception_isolated_as_failed_step(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    registry: ModelRegistry,
    store: RunStore,
) -> None:
    """workflow 中 step parse 异常不应中断执行，应失败并触发下游 skip。"""
    monkeypatch.setattr("eval.execution.workflow_runner.build_task", lambda _: _ParseErrorTask())
    wf = load_workflow(_make_workflow_json(tmp_path))
    rows = asyncio.run(
        _run_workflow_mode(
            run_id=store.run_id,
            store=store,
            dataset=[_SAMPLE],
            registry=registry,
            workflow=wf,
            global_params={},
            gateway=FixedGateway({"*": "{}"}),
        )
    )
    step_rows = [r for r in rows if r["mode"] == "workflow_step"]
    e2e = [r for r in rows if r["mode"] == "workflow_e2e"][0]
    assert step_rows[0]["success"] is False
    assert "parse_error" in step_rows[0]["error"]
    assert any("upstream_" in r["error"] for r in step_rows[1:])
    assert e2e["success"] is False


def test_workflow_e2e_cost_accumulates(tmp_path: Path, store: RunStore) -> None:
    """R3 验证：e2e_row 的 cost_estimate 应为各 step cost 之和（>0）。"""
    reg_with_price = {
        "active": "m1",
        "models": {
            k: {**v, "price_config": {"input_per_1k": 0.001, "output_per_1k": 0.002}}
            for k, v in _REGISTRY_DATA["models"].items()
        },
    }
    reg_path = tmp_path / "registry_price.json"
    reg_path.write_text(json.dumps(reg_with_price), encoding="utf-8")
    registry_priced = ModelRegistry(reg_path)

    dedup_resp = json.dumps({"deduped_titles": ["title"], "removed_count": 0})
    ie_resp = json.dumps({"ticker": "AAPL", "event_type": "earnings", "sentiment": "positive", "impact_score": 1, "summary": "ok"})
    score_resp = json.dumps({"score": 50, "rationale": "ok"})
    gateway = FixedGateway({"news_dedup": dedup_resp, "ie_json": ie_resp, "stock_score": score_resp})

    wf = load_workflow(_make_workflow_json(tmp_path))
    records = asyncio.run(
        _run_workflow_mode(
            run_id=store.run_id,
            store=store,
            dataset=[_SAMPLE],
            registry=registry_priced,
            workflow=wf,
            global_params={},
            gateway=gateway,
        )
    )
    e2e_rows = [r for r in records if r["mode"] == "workflow_e2e"]
    assert e2e_rows[0]["cost_estimate"] > 0, "R3: e2e cost_estimate should accumulate step costs"


def test_workflow_upstream_parse_fail_skips_downstream(tmp_path: Path, registry: ModelRegistry, store: RunStore) -> None:
    """R1 验证：step1 parse 失败时，依赖它的 step2/step3 应被跳过，e2e_success=False。"""
    wf = load_workflow(_make_workflow_json(tmp_path))
    records = asyncio.run(
        _run_workflow_mode(
            run_id=store.run_id,
            store=store,
            dataset=[_SAMPLE],
            registry=registry,
            workflow=wf,
            global_params={},
            gateway=BadParseGateway(),
        )
    )
    step_rows = {r["step_id"]: r for r in records if r["mode"] == "workflow_step"}
    e2e_rows = [r for r in records if r["mode"] == "workflow_e2e"]

    # step1 运行但解析失败
    assert step_rows["step1_dedup"]["parse_success"] is False

    # step2/step3 应被跳过，error 含 upstream 标记
    assert "upstream" in step_rows["step2_extract"]["error"], "R1: step2 should have upstream error"
    assert "upstream" in step_rows["step3_score"]["error"], "R1: step3 should have upstream error"
    assert step_rows["step2_extract"]["success"] is False
    assert step_rows["step3_score"]["success"] is False

    # e2e 整体失败
    assert e2e_rows[0]["success"] is False, "R1: e2e_success should be False when steps are skipped"


def test_workflow_e2e_parse_success_reflects_last_step_parse(
    tmp_path: Path,
    registry: ModelRegistry,
    store: RunStore,
) -> None:
    """最后一步 parse 失败时，e2e parse_success 应为 False，parsed 应为空。"""
    dedup_resp = json.dumps({"deduped_titles": ["title"], "removed_count": 0})
    ie_resp = json.dumps({"ticker": "AAPL", "event_type": "earnings", "sentiment": "positive", "impact_score": 1, "summary": "ok"})
    # stock_score 期望 JSON；这里故意返回不可解析文本，验证不会沿用上一步 parsed
    bad_score_resp = "NOT_JSON"
    gateway = FixedGateway({"news_dedup": dedup_resp, "ie_json": ie_resp, "stock_score": bad_score_resp})

    wf = load_workflow(_make_workflow_json(tmp_path))
    records = asyncio.run(
        _run_workflow_mode(
            run_id=store.run_id,
            store=store,
            dataset=[_SAMPLE],
            registry=registry,
            workflow=wf,
            global_params={},
            gateway=gateway,
        )
    )
    e2e_row = [r for r in records if r["mode"] == "workflow_e2e"][0]
    assert e2e_row["success"] is False
    assert e2e_row["parse_success"] is False
    assert e2e_row["parsed"] is None


def test_workflow_skip_row_respects_task_parse_applicable(
    monkeypatch: pytest.MonkeyPatch,
    registry: ModelRegistry,
    store: RunStore,
) -> None:
    """skip 行应继承被跳过 task 的 parse_applicable，而不是写死 True。"""

    class _NoParseTask:
        name = "no_parse"
        version = "v1"
        parse_applicable = False

        def build_prompt(self, sample, context=None):
            return [], "v1"

        def parse(self, output):
            return None

        def metrics(self, sample, output, parsed):
            return {}

    from eval.tasks import build_task as _real_build_task

    def _patched_build_task(task_name: str):
        if task_name == "no_parse":
            return _NoParseTask()
        return _real_build_task(task_name)

    monkeypatch.setattr("eval.execution.workflow_runner.build_task", _patched_build_task)

    wf = WorkflowSpec(
        name="skip_parse_applicable_test",
        version="v1",
        steps=[
            WorkflowStep(id="step1", task="news_dedup", model_id="m1", input_from="sample"),
            WorkflowStep(id="step2", task="no_parse", model_id="m2", input_from="step1"),
        ],
    )
    records = asyncio.run(
        _run_workflow_mode(
            run_id=store.run_id,
            store=store,
            dataset=[_SAMPLE],
            registry=registry,
            workflow=wf,
            global_params={},
            gateway=BadParseGateway(),
        )
    )
    step_rows = {r["step_id"]: r for r in records if r["mode"] == "workflow_step"}
    assert step_rows["step2"]["success"] is False
    assert step_rows["step2"]["parse_applicable"] is False


def test_workflow_supports_sample_level_concurrency(registry: ModelRegistry, store: RunStore) -> None:
    """workflow_concurrency>1 时应出现并行调用（max_inflight >= 2）。"""
    wf = WorkflowSpec(
        name="concurrency_test",
        version="v1",
        steps=[
            WorkflowStep(id="step1", task="news_dedup", model_id="m1", input_from="sample"),
        ],
    )
    dataset = [{**_SAMPLE, "sample_id": f"T{i:03d}"} for i in range(6)]
    gateway = SlowTrackingGateway(delay_s=0.02)

    records = asyncio.run(
        _run_workflow_mode(
            run_id=store.run_id,
            store=store,
            dataset=dataset,
            registry=registry,
            workflow=wf,
            global_params={},
            gateway=gateway,
            workflow_concurrency=3,
        )
    )

    assert gateway.max_inflight >= 2
    assert len(records) == 12  # 6 * (1 step + 1 e2e)


# ── Registry API key 校验测试 ─────────────────────────────────────────

def test_registry_raises_on_unresolved_api_key(tmp_path: Path) -> None:
    """R4 验证：api_key 含未解析占位符时应在 resolve_config 时抛出 ValueError。"""
    _UNSET_VAR = "NONEXISTENT_TEST_VAR_XYZ_12345"
    os.environ.pop(_UNSET_VAR, None)

    reg_data = {
        "active": "bad-model",
        "models": {
            "bad-model": {
                "provider": "openai_compatible",
                "model": "test",
                "api_key": f"${{{_UNSET_VAR}}}",
                "base_url": "http://localhost",
                "temperature": 0.0,
                "max_tokens": 256,
            }
        },
    }
    reg_path = tmp_path / "bad_registry.json"
    reg_path.write_text(json.dumps(reg_data), encoding="utf-8")
    registry = ModelRegistry(reg_path)
    with pytest.raises(ValueError, match="unresolved env var"):
        registry.resolve_config("bad-model")


# ── 完整 CLI run() 端到端验证 ─────────────────────────────────────────

def test_full_run_task_mode_mock_produces_artifacts(tmp_path: Path) -> None:
    """端到端：通过 run() 入口在 mock 模式验证产物目录完整，summary 含 tm_ 列。"""
    reg_path = tmp_path / "registry.json"
    reg_path.write_text(json.dumps(_REGISTRY_DATA), encoding="utf-8")

    dataset_path = tmp_path / "data.jsonl"
    dataset_path.write_text(json.dumps(_SAMPLE) + "\n", encoding="utf-8")

    args = Namespace(
        task="ie_json",
        workflow=None,
        dataset=str(dataset_path),
        models="m1,m2",
        model_registry=str(reg_path),
        out=str(tmp_path / "runs"),
        params=None,
        repeats=1,
        cache_dir=str(tmp_path / ".cache"),
        no_cache=True,
        max_samples=None,
        mock=True,
        concurrency=4,
    )
    run_dir = asyncio.run(run(args))

    assert (run_dir / "config.json").exists()
    assert (run_dir / "results.jsonl").exists()
    assert (run_dir / "summary.csv").exists()
    assert (run_dir / "report.md").exists()

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["scorer_version"] == "v1"
    assert config["result_schema_version"] == "result_row.v1"
    assert config["summary_schema_version"] == "summary_row.v1"

    # R2 验证：summary.csv 应包含 tm_* 列
    with (run_dir / "summary.csv").open(encoding="utf-8") as f:
        fieldnames = csv.DictReader(f).fieldnames or []
    assert any(fn.startswith("tm_") for fn in fieldnames), (
        "summary.csv should contain task_metrics columns (tm_*)"
    )
    assert "schema_version" in fieldnames


# ── from_cache 成本归零测试（P0 Bug 修复）────────────────────────────

def _make_priced_registry(tmp_path: Path) -> ModelRegistry:
    """构造含 price_config 的注册表，确保默认模型有成本估算配置。"""
    reg_with_price = {
        "active": "m1",
        "models": {
            k: {**v, "price_config": {"input_per_1k": 0.001, "output_per_1k": 0.002}}
            for k, v in _REGISTRY_DATA["models"].items()
        },
    }
    path = tmp_path / "registry_priced.json"
    path.write_text(json.dumps(reg_with_price), encoding="utf-8")
    return ModelRegistry(path)


def test_task_mode_cached_result_cost_is_zero(tmp_path: Path, store: RunStore) -> None:
    """P0 Bug: from_cache=True 的结果 cost_estimate 必须为 0，不应误计 API 费用。"""
    registry = _make_priced_registry(tmp_path)
    ie_resp = json.dumps({
        "ticker": "AAPL", "event_type": "earnings",
        "sentiment": "positive", "impact_score": 3, "summary": "beat",
    })
    records = asyncio.run(
        _run_task_mode(
            run_id=store.run_id,
            store=store,
            dataset=[_SAMPLE],
            registry=registry,
            task_name="ie_json",
            model_ids=["m1"],
            params_override={},
            repeats=1,
            gateway=CachedResponseGateway({"ie_json": ie_resp}),
        )
    )
    assert len(records) == 1
    # P0 Bug：修复前此行会 > 0（用 usage token 算出一个"幻象成本"）
    assert records[0]["cost_estimate"] == 0.0, (
        "from_cache=True 的结果不应计算 API 费用，cost_estimate 应为 0.0"
    )


def test_workflow_mode_cached_steps_cost_is_zero(tmp_path: Path, store: RunStore) -> None:
    """P0 Bug: workflow 中每一步 from_cache=True 时，step 和 e2e 的 cost_estimate 均应为 0。"""
    registry = _make_priced_registry(tmp_path)
    dedup_resp = json.dumps({"deduped_titles": ["title"], "removed_count": 0})
    ie_resp = json.dumps({"ticker": "AAPL", "event_type": "earnings", "sentiment": "positive", "impact_score": 1, "summary": "ok"})
    score_resp = json.dumps({"score": 50, "rationale": "ok"})
    gateway = CachedResponseGateway({
        "news_dedup": dedup_resp, "ie_json": ie_resp, "stock_score": score_resp,
    })
    wf = load_workflow(_make_workflow_json(tmp_path))
    records = asyncio.run(
        _run_workflow_mode(
            run_id=store.run_id,
            store=store,
            dataset=[_SAMPLE],
            registry=registry,
            workflow=wf,
            global_params={},
            gateway=gateway,
        )
    )
    step_rows = [r for r in records if r["mode"] == "workflow_step"]
    e2e_rows  = [r for r in records if r["mode"] == "workflow_e2e"]
    for row in step_rows:
        assert row["cost_estimate"] == 0.0, (
            f"step '{row['step_id']}': from_cache=True 不应计入成本"
        )
    assert e2e_rows[0]["cost_estimate"] == 0.0, "e2e 汇总行成本应为各 step 之和，全部为 0"
