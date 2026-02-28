"""多步 Workflow 评测执行器。

处理多步骤流水线评测，如 news_pipeline.yaml 定义的：
  step1_dedup → step2_extract → step3_score

核心逻辑：
1. 每个样本按步骤顺序执行，上游解析结果通过 step_outputs 传递。
2. 上游失败时写入 skipped 记录并继续后续样本（不阻塞整个批次）。
3. 每个样本完成后输出一行 e2e 汇总行，汇总总耗时/token/成本。
4. 支持样本级并发（--workflow-concurrency），单样本内 step 仍顺序执行。
5. 采用 FIRST_COMPLETED 流式回收，避免全量 gather 带来的 OOM 风险。

模型解析优先级：CLI step_model_map > 步骤声明 > default_model_id（$active 回退）。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from bench.contracts.constants import STEP_ID_E2E
from bench.contracts.exceptions import (
    EvalBenchError,
    GatewayCallError,
    MetricsComputeError,
    ParseOutputError,
    PromptBuildError,
    UpstreamDependencyError,
)
from bench.contracts.error import derive_error_fields
from bench.contracts.result import ResultRow
from bench.execution.gateway import LLMGateway
from bench.execution.utils import RESULT_ROW_SCHEMA_VERSION
from bench.io.store import RunStore
from bench.registry import ModelRegistry, estimate_cost
from bench.tasks import build_task
from bench.workflow import WorkflowSpec

__all__ = ["run_workflow_mode"]

logger = logging.getLogger(__name__)


def _result_row_dict(
    row: ResultRow,
    *,
    workflow_name: str,
    workflow_version: str,
) -> dict[str, Any]:
    out = asdict(row)
    out["timestamp"] = datetime.now(UTC).isoformat()
    out["workflow_name"] = workflow_name
    out["workflow_version"] = workflow_version
    return out


def _build_workflow_error_row(
    *,
    run_id: str,
    sample_id: str,
    workflow_name: str,
    workflow_version: str,
    task_name: str,
    task_version: str,
    step_id: str,
    prompt_version: str,
    model_id: str,
    params: dict[str, Any],
    parse_applicable: bool,
    error_message: str,
    error_exc: EvalBenchError | None = None,
) -> dict[str, Any]:
    fields = derive_error_fields(
        success=False,
        parse_applicable=parse_applicable,
        parse_success=False,
        error_message=error_message,
        error_exc=error_exc,
    )
    row = ResultRow(
        schema_version=RESULT_ROW_SCHEMA_VERSION,
        run_id=run_id,
        mode="workflow_step",
        sample_id=sample_id,
        task_name=task_name,
        task_version=task_version,
        step_id=step_id,
        prompt_version=prompt_version,
        model_id=model_id,
        params=params,
        success=False,
        error=error_message,
        parse_applicable=parse_applicable,
        parse_success=False,
        latency_ms=0.0,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        retry_count=0,
        cost_estimate=0.0,
        from_cache=False,
        output_text="",
        parsed=None,
        task_metrics={},
        **fields,
    )
    return _result_row_dict(row, workflow_name=workflow_name, workflow_version=workflow_version)


def _build_workflow_step_row(
    *,
    run_id: str,
    sample_id: str,
    workflow_name: str,
    workflow_version: str,
    task_name: str,
    task_version: str,
    step_id: str,
    prompt_version: str,
    model_id: str,
    params: dict[str, Any],
    row_success: bool,
    error_message: str,
    parse_applicable: bool,
    parse_success: bool,
    latency_ms: float,
    usage: dict[str, int],
    retry_count: int,
    cost_estimate: float,
    from_cache: bool,
    output_text: str,
    parsed: dict[str, Any] | None,
    task_metrics: dict[str, Any],
    error_type: str = "",
    error_stage: str = "",
    error_code: str = "",
    error_exc: EvalBenchError | None = None,
) -> dict[str, Any]:
    fields = derive_error_fields(
        success=row_success,
        parse_applicable=parse_applicable,
        parse_success=parse_success,
        error_message=error_message or "",
        error_type=error_type,
        error_stage=error_stage,
        error_code=error_code,
        error_exc=error_exc,
    )
    row = ResultRow(
        schema_version=RESULT_ROW_SCHEMA_VERSION,
        run_id=run_id,
        mode="workflow_step",
        sample_id=sample_id,
        task_name=task_name,
        task_version=task_version,
        step_id=step_id,
        prompt_version=prompt_version,
        model_id=model_id,
        params=params,
        success=row_success,
        error=error_message,
        parse_applicable=parse_applicable,
        parse_success=parse_success,
        latency_ms=round(latency_ms, 2),
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        total_tokens=usage["total_tokens"],
        retry_count=retry_count,
        cost_estimate=round(cost_estimate, 8),
        from_cache=from_cache,
        output_text=output_text,
        parsed=parsed,
        task_metrics=task_metrics,
        **fields,
    )
    return _result_row_dict(row, workflow_name=workflow_name, workflow_version=workflow_version)


def _build_workflow_e2e_row(
    *,
    run_id: str,
    sample_id: str,
    workflow_name: str,
    workflow_version: str,
    model_id: str,
    params: dict[str, Any],
    success: bool,
    parse_applicable: bool,
    parse_success: bool,
    latency_ms: float,
    total_tokens: int,
    retry_count: int,
    cost_estimate: float,
    parsed: dict[str, Any] | None,
) -> dict[str, Any]:
    error_message = "" if success else "workflow_step_failed"
    fields = derive_error_fields(
        success=success,
        parse_applicable=parse_applicable,
        parse_success=parse_success,
        error_message=error_message,
    )
    row = ResultRow(
        schema_version=RESULT_ROW_SCHEMA_VERSION,
        run_id=run_id,
        mode="workflow_e2e",
        sample_id=sample_id,
        task_name=workflow_name,
        task_version=workflow_version,
        step_id=STEP_ID_E2E,
        prompt_version="workflow",
        model_id=model_id,
        params=params,
        success=success,
        error=error_message,
        parse_applicable=parse_applicable,
        parse_success=parse_success,
        latency_ms=round(latency_ms, 2),
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=total_tokens,
        retry_count=retry_count,
        cost_estimate=round(cost_estimate, 8),
        from_cache=False,
        output_text="",
        parsed=parsed,
        task_metrics={},
        **fields,
    )
    return _result_row_dict(row, workflow_name=workflow_name, workflow_version=workflow_version)


# ---------------------------------------------------------------------------
# _SampleRunner: 将原 ~230 行的 _run_one_sample 拆分为可独立测试的方法
# ---------------------------------------------------------------------------

@dataclass
class _StepExecResult:
    """Successfully executed step result."""
    row: dict[str, Any]
    output_entry: dict[str, Any]
    step_ok: bool
    latency_ms: float
    total_tokens: int
    retry_count: int
    cost: float
    parsed: dict[str, Any] | None
    parse_applicable: bool


class _StepExecError(Exception):
    """Wraps a prompt-build or gateway-call error with audit context."""

    def __init__(self, exc: EvalBenchError, prompt_version: str = "error"):
        super().__init__(str(exc))
        self.cause = exc
        self.prompt_version = prompt_version


class _SampleRunner:
    """对单个样本执行全部 workflow steps 的状态机。

    将 _run_one_sample 的逻辑拆分为：
      _resolve_model_id / _merge_params / _resolve_step_input /
      _fail_row / _execute_step / run
    """

    __slots__ = (
        "_run_id", "_store", "_gateway", "_registry", "_wf",
        "_global_params", "_default_model_id", "_step_map",
    )

    def __init__(
        self,
        *,
        run_id: str,
        store: RunStore,
        gateway: LLMGateway,
        registry: ModelRegistry,
        workflow: WorkflowSpec,
        global_params: dict[str, Any],
        default_model_id: str,
        step_map: dict[str, str],
    ) -> None:
        self._run_id = run_id
        self._store = store
        self._gateway = gateway
        self._registry = registry
        self._wf = workflow
        self._global_params = global_params
        self._default_model_id = default_model_id
        self._step_map = step_map

    # ─── 辅助方法 ────────────────────────────────────────────────

    def _resolve_model_id(self, step: Any) -> str:
        raw = self._step_map.get(step.id) or (
            self._default_model_id if step.model_id == "$active" else step.model_id
        )
        if not raw:
            raise ValueError(
                f"Cannot resolve model for step '{step.id}': "
                f"no --models supplied and no active model in registry."
            )
        return raw

    @staticmethod
    def _merge_params(task: Any, step: Any, global_params: dict[str, Any]) -> dict[str, Any]:
        merged = dict(getattr(task, "default_params", {}) or {})
        merged.update(global_params)
        merged.update(step.params_override)
        return merged

    def _resolve_step_input(
        self,
        step: Any,
        sample: dict[str, Any],
        step_outputs: dict[str, dict[str, Any]],
    ) -> tuple[Any, UpstreamDependencyError | None]:
        """返回 ``(step_input, error_or_none)``。upstream 失败时 step_input 为 None。"""
        if step.input_from == "sample":
            return sample, None
        prev = step_outputs.get(step.input_from)
        upstream_failed = (
            prev is None
            or not prev.get("success")
            or (prev.get("parse_applicable", True) and not prev.get("parse_success"))
        )
        if upstream_failed:
            reason = (
                f"upstream_{step.input_from}_parse_failed"
                if prev is not None
                else f"upstream_{step.input_from}_missing"
            )
            return None, UpstreamDependencyError(reason, missing=(prev is None))
        step_input = prev["parsed"] if prev.get("parse_applicable", True) else prev.get("raw", "")
        return step_input, None

    def _fail_row(
        self,
        *,
        sample_id: str,
        step: Any,
        task: Any,
        model_id: str,
        merged_params: dict[str, Any],
        error_exc: EvalBenchError,
        prompt_version: str = "skipped",
        task_version: str | None = None,
    ) -> dict[str, Any]:
        """Create, persist and return an error row for a failed step."""
        row = _build_workflow_error_row(
            run_id=self._run_id,
            sample_id=sample_id,
            workflow_name=self._wf.name,
            workflow_version=self._wf.version,
            task_name=getattr(task, "name", step.task),
            task_version=task_version if task_version is not None else getattr(task, "version", "skipped"),
            step_id=step.id,
            prompt_version=prompt_version,
            model_id=model_id,
            params=merged_params,
            error_message=str(error_exc),
            parse_applicable=getattr(task, "parse_applicable", True),
            error_exc=error_exc,
        )
        self._store.append_result(row)
        return row

    @staticmethod
    def _failure_output_entry(task: Any) -> dict[str, Any]:
        return {
            "parsed": None,
            "raw": "",
            "success": False,
            "parse_success": False,
            "parse_applicable": task.parse_applicable,
        }

    # ─── 核心执行 ────────────────────────────────────────────────

    async def _execute_step(
        self,
        *,
        sample_id: str,
        sample: dict[str, Any],
        step: Any,
        task: Any,
        model_id: str,
        merged_params: dict[str, Any],
        step_input: Any,
        step_outputs: dict[str, dict[str, Any]],
    ) -> _StepExecResult:
        """执行 prompt→call→parse→metrics 流水线。

        Raises:
            _StepExecError: 当 prompt 构建或 gateway 调用抛出异常。
        """
        # ── prompt ──
        try:
            if self._gateway.mock:
                messages: list[Any] = []
                prompt_version = "mock"
            else:
                messages, prompt_version = task.build_prompt(
                    sample, context={"input": step_input, "step_outputs": step_outputs},
                )
        except Exception as exc:
            raise _StepExecError(PromptBuildError(str(exc)), "error") from exc

        # ── gateway call ──
        try:
            call = await self._gateway.call(
                model_id=model_id,
                task_name=task.name,
                sample=sample,
                sample_cache_id=f"{sample_id}:{step.id}:{task.name}",
                messages=messages,
                params_override=merged_params,
            )
        except Exception as exc:
            gw_exc = exc if isinstance(exc, EvalBenchError) else GatewayCallError(str(exc))
            raise _StepExecError(gw_exc, prompt_version) from exc

        # ── parse (non-fatal) ──
        parsed = None
        parse_success = False
        task_metrics: dict[str, Any] = {}
        row_success = bool(call.success)
        error_message = call.error
        derived_exc: EvalBenchError | None = None

        if call.success:
            try:
                parsed = task.parse(call.content)
                parse_success = parsed is not None
            except Exception as exc:
                row_success = False
                parse_success = False
                parse_exc = ParseOutputError(str(exc))
                derived_exc = parse_exc
                error_message = f"{error_message}; {parse_exc}" if error_message else str(parse_exc)
        else:
            parse_success = False

        # ── metrics (non-fatal) ──
        try:
            task_metrics = task.metrics(sample, call.content, parsed)
        except Exception as exc:
            row_success = False
            metrics_exc = MetricsComputeError(str(exc))
            derived_exc = metrics_exc
            error_message = f"{error_message}; {metrics_exc}" if error_message else str(metrics_exc)
            task_metrics = {}

        spec = self._registry.get(model_id)
        cost = 0.0 if call.from_cache else estimate_cost(call.usage, spec.price_config)

        step_row = _build_workflow_step_row(
            run_id=self._run_id,
            sample_id=sample_id,
            workflow_name=self._wf.name,
            workflow_version=self._wf.version,
            task_name=task.name,
            task_version=task.version,
            step_id=step.id,
            prompt_version=prompt_version,
            model_id=model_id,
            params=merged_params,
            row_success=row_success,
            error_message=error_message,
            parse_applicable=task.parse_applicable,
            parse_success=parse_success,
            latency_ms=call.latency_ms,
            usage=call.usage,
            retry_count=call.retry_count,
            cost_estimate=cost,
            from_cache=call.from_cache,
            output_text=call.content,
            parsed=parsed,
            task_metrics=task_metrics,
            error_type=call.error_type,
            error_stage=call.error_stage,
            error_code=call.error_code,
            error_exc=derived_exc,
        )
        self._store.append_result(step_row)

        output_entry = {
            "parsed": parsed,
            "raw": call.content,
            "success": call.success,
            "parse_success": parse_success,
            "parse_applicable": task.parse_applicable,
        }
        step_ok = row_success and (parse_success or not task.parse_applicable)

        return _StepExecResult(
            row=step_row,
            output_entry=output_entry,
            step_ok=step_ok,
            latency_ms=call.latency_ms,
            total_tokens=call.usage["total_tokens"],
            retry_count=call.retry_count,
            cost=cost,
            parsed=parsed,
            parse_applicable=task.parse_applicable,
        )

    # ─── 样本入口 ────────────────────────────────────────────────

    async def run(self, sample: dict[str, Any]) -> list[dict[str, Any]]:
        """对单个样本执行全部 workflow steps，返回 step 级 + e2e 级结果行。"""
        sample_records: list[dict[str, Any]] = []
        sample_id = str(sample.get("sample_id", ""))
        step_outputs: dict[str, dict[str, Any]] = {}
        e2e_success = True
        e2e_latency = 0.0
        e2e_tokens = 0
        e2e_retry = 0
        e2e_cost = 0.0
        final_output: dict[str, Any] | None = None
        last_step_pa = True

        for step in self._wf.steps:
            model_id = self._resolve_model_id(step)
            task = build_task(step.task)
            merged_params = self._merge_params(task, step, self._global_params)

            # ── upstream 检查 ──
            step_input, upstream_err = self._resolve_step_input(step, sample, step_outputs)
            if upstream_err is not None:
                logger.warning(
                    "workflow step '%s' skipped: %s (sample=%s)",
                    step.id, upstream_err, sample_id,
                )
                sample_records.append(self._fail_row(
                    sample_id=sample_id, step=step, task=task,
                    model_id=model_id, merged_params=merged_params,
                    error_exc=upstream_err, task_version="skipped",
                ))
                step_outputs[step.id] = self._failure_output_entry(task)
                final_output = None
                e2e_success = False
                continue

            # ── 执行 prompt → call → parse → metrics ──
            try:
                result = await self._execute_step(
                    sample_id=sample_id, sample=sample, step=step, task=task,
                    model_id=model_id, merged_params=merged_params,
                    step_input=step_input, step_outputs=step_outputs,
                )
            except _StepExecError as e:
                logger.warning(
                    "workflow step '%s' failed: %s (sample=%s)",
                    step.id, e, sample_id,
                )
                sample_records.append(self._fail_row(
                    sample_id=sample_id, step=step, task=task,
                    model_id=model_id, merged_params=merged_params,
                    error_exc=e.cause, prompt_version=e.prompt_version,
                ))
                step_outputs[step.id] = self._failure_output_entry(task)
                final_output = None
                e2e_success = False
                continue

            sample_records.append(result.row)
            step_outputs[step.id] = result.output_entry
            e2e_success = e2e_success and result.step_ok
            e2e_latency += result.latency_ms
            e2e_tokens += result.total_tokens
            e2e_retry += result.retry_count
            e2e_cost += result.cost
            final_output = result.parsed
            last_step_pa = result.parse_applicable

        # ── e2e 汇总行 ──
        e2e_row = _build_workflow_e2e_row(
            run_id=self._run_id,
            sample_id=sample_id,
            workflow_name=self._wf.name,
            workflow_version=self._wf.version,
            model_id="+".join(
                self._step_map.get(s.id) or (
                    self._default_model_id if s.model_id == "$active" else s.model_id
                )
                for s in self._wf.steps
            ),
            params=self._global_params,
            success=e2e_success,
            parse_applicable=last_step_pa,
            parse_success=final_output is not None if last_step_pa else False,
            latency_ms=e2e_latency,
            total_tokens=e2e_tokens,
            retry_count=e2e_retry,
            cost_estimate=e2e_cost,
            parsed=final_output,
        )
        self._store.append_result(e2e_row)
        sample_records.append(e2e_row)

        return sample_records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_workflow_mode(
    *,
    run_id: str,
    store: RunStore,
    dataset: list[dict[str, Any]],
    registry: ModelRegistry,
    workflow: WorkflowSpec,
    global_params: dict[str, Any],
    gateway: LLMGateway,
    default_model_id: str = "",
    step_model_map: dict[str, str] | None = None,
    workflow_concurrency: int = 1,
    completed_keys: set[tuple[str, str, str, str, int]] | None = None,
) -> list[dict[str, Any]]:
    """执行 workflow 模式，输出 step 级与 e2e 级结果。

    step_model_map 优先覆盖步骤自身的 model_id；
    步骤 model_id 为 ``$active`` 时回退到 default_model_id。
    """
    _step_map = step_model_map or {}
    completed = completed_keys or set()
    records: list[dict[str, Any]] = []
    filtered_dataset = [
        sample
        for sample in dataset
        if not any(
            key[0] == str(sample.get("sample_id", ""))
            and key[1] == workflow.name
            and key[2] == STEP_ID_E2E
            for key in completed
        )
    ]

    runner = _SampleRunner(
        run_id=run_id,
        store=store,
        gateway=gateway,
        registry=registry,
        workflow=workflow,
        global_params=global_params,
        default_model_id=default_model_id,
        step_map=_step_map,
    )

    if workflow_concurrency <= 1:
        for sample in filtered_dataset:
            records.extend(await runner.run(sample))
        return records

    sem = asyncio.Semaphore(workflow_concurrency if workflow_concurrency > 0 else 10_000)

    async def _bounded_run(sample: dict[str, Any]) -> list[dict[str, Any]]:
        async with sem:
            return await runner.run(sample)

    pending: set[asyncio.Task[list[dict[str, Any]]]] = set()

    async def _drain_one() -> None:
        nonlocal pending
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task_fut in done:
            try:
                rows = task_fut.result()
            except Exception as exc:
                logger.error("Unexpected workflow_runner coroutine exception: %s", exc, exc_info=True)
                continue
            records.extend(rows)

    for sample in filtered_dataset:
        pending.add(asyncio.create_task(_bounded_run(sample)))
        if len(pending) >= max(1, workflow_concurrency):
            await _drain_one()

    while pending:
        await _drain_one()
    return records
