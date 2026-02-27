import asyncio
import json
from pathlib import Path

from eval.contracts import UnifiedCallResult
from eval.registry import ModelRegistry
from eval.runner import _run_workflow_mode
from eval.store import RunStore
from eval.workflow import WorkflowSpec, WorkflowStep


class _FixedGateway:
    mock = False

    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    async def call(
        self,
        *,
        model_id: str,
        task_name: str,
        sample: dict,
        sample_cache_id: str,
        messages: list,
        params_override: dict,
    ) -> UnifiedCallResult:
        content = self.mapping.get(task_name, "{}")
        return UnifiedCallResult(
            content=content,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            latency_ms=1.0,
            raw_response={"ok": True},
            success=True,
            error="",
            retry_count=0,
            from_cache=False,
        )


def test_e2e_parse_follows_last_step(tmp_path: Path) -> None:
    reg_path = tmp_path / "registry.json"
    reg_path.write_text(
        json.dumps(
            {
                "active": "m1",
                "models": {
                    "m1": {
                        "provider": "openai_compatible",
                        "model": "x",
                        "api_key": "k",
                        "base_url": "http://localhost",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    registry = ModelRegistry(reg_path)
    store = RunStore(tmp_path / "runs")

    wf = WorkflowSpec(
        name="wf",
        version="v1",
        steps=(
            WorkflowStep(id="s1", task="news_dedup", model_id="m1", input_from="sample"),
            WorkflowStep(id="s2", task="ie_json", model_id="m1", input_from="s1"),
            WorkflowStep(id="s3", task="stock_score", model_id="m1", input_from="s2"),
        ),
    )
    gateway = _FixedGateway(
        {
            "news_dedup": json.dumps({"deduped_titles": ["a"], "removed_count": 0}),
            "ie_json": json.dumps(
                {
                    "ticker": "AAPL",
                    "event_type": "earnings",
                    "sentiment": "positive",
                    "impact_score": 2,
                    "summary": "ok",
                }
            ),
            # 最后一步故意不可解析，确保 e2e 不会沿用上一步 parsed。
            "stock_score": "NOT_JSON",
        }
    )
    rows = asyncio.run(
        _run_workflow_mode(
            run_id=store.run_id,
            store=store,
            dataset=[{"sample_id": "S1", "news_items": []}],
            registry=registry,
            workflow=wf,
            global_params={},
            gateway=gateway,
        )
    )
    e2e = [r for r in rows if r["mode"] == "workflow_e2e"][0]
    assert e2e["success"] is False
    assert e2e["parse_success"] is False
    assert e2e["parsed"] is None
