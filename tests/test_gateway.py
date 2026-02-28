import asyncio

from bench.execution.gateway import LLMGateway
from bench.registry import ModelRegistry


def test_gateway_mock_call() -> None:
    registry = ModelRegistry("configs/llm_providers.json")
    gateway = LLMGateway(registry=registry, cache=None, mock=True)

    result = asyncio.run(
        gateway.call(
            model_id="deepseek-v3",
            task_name="ie_json",
            sample={"ticker": "AAPL", "news_items": []},
            sample_cache_id="s1",
            messages=[],
            params_override={},
        )
    )
    assert result.success is True
    assert result.usage["total_tokens"] == 30
    assert "ticker" in result.content


class _FakeLLMResp:
    def __init__(self) -> None:
        self.text = "{}"
        self.model = "fake-model"
        self.input_tokens = 1
        self.output_tokens = 1
        self.latency_ms = 1.0
        self.success = True
        self.error_message = ""
        self.retry_count = 0


class _FakeClient:
    async def complete(self, messages):
        return _FakeLLMResp()


class _FakeRegistry:
    def __init__(self) -> None:
        self.create_client_calls = 0

    def create_client(self, model_id: str, params_override=None):
        self.create_client_calls += 1
        return _FakeClient()


def test_gateway_reuses_client_for_same_model_and_params() -> None:
    registry = _FakeRegistry()
    gateway = LLMGateway(registry=registry, cache=None, mock=False)

    async def _run_twice() -> None:
        await gateway.call(
            model_id="m1",
            task_name="ie_json",
            sample={"sample_id": "s1"},
            sample_cache_id="s1",
            messages=[{"role": "user", "content": "x"}],
            params_override={"temperature": 0.1},
        )
        await gateway.call(
            model_id="m1",
            task_name="ie_json",
            sample={"sample_id": "s2"},
            sample_cache_id="s2",
            messages=[{"role": "user", "content": "y"}],
            params_override={"temperature": 0.1},
        )

    asyncio.run(_run_twice())
    assert registry.create_client_calls == 1


def test_gateway_does_not_reuse_client_when_params_differ() -> None:
    registry = _FakeRegistry()
    gateway = LLMGateway(registry=registry, cache=None, mock=False)

    async def _run_with_different_params() -> None:
        await gateway.call(
            model_id="m1",
            task_name="ie_json",
            sample={"sample_id": "s1"},
            sample_cache_id="s1",
            messages=[{"role": "user", "content": "x"}],
            params_override={"temperature": 0.1},
        )
        await gateway.call(
            model_id="m1",
            task_name="ie_json",
            sample={"sample_id": "s2"},
            sample_cache_id="s2",
            messages=[{"role": "user", "content": "y"}],
            params_override={"temperature": 0.2},
        )

    asyncio.run(_run_with_different_params())
    assert registry.create_client_calls == 2


# ── Mock 响应解耦测试（P1）───────────────────────────────────────────

def test_task_has_mock_response_method() -> None:
    """P1: GenericTask 必须暴露 mock_response(sample) 方法（解耦前不存在）。"""
    from bench.tasks import build_task
    task = build_task("ie_json")
    assert hasattr(task, "mock_response"), (
        "GenericTask 应提供 mock_response() 方法，使 Gateway 不再需要硬编码任务名"
    )


def test_task_mock_response_returns_valid_json_string() -> None:
    """P1: mock_response() 应返回可 JSON 解析的字符串，内容来自 YAML 定义。"""
    import json
    from bench.tasks import build_task

    sample = {"ticker": "TSLA", "news_items": []}
    for task_name in ("ie_json", "stock_score", "news_dedup"):
        task = build_task(task_name)
        content = task.mock_response(sample)
        assert isinstance(content, str), f"{task_name}.mock_response() 应返回 str"
        parsed = json.loads(content)
        assert isinstance(parsed, dict), f"{task_name}.mock_response() 应返回可 JSON 解析的内容"


def test_task_mock_response_ie_json_has_ticker_field() -> None:
    """P1: ie_json 的 mock_response 内容必须包含 ticker 字段（覆盖现有 gateway 测试预期）。"""
    import json
    from bench.tasks import build_task

    task = build_task("ie_json")
    content = task.mock_response({"ticker": "AAPL"})
    parsed = json.loads(content)
    assert "ticker" in parsed, "ie_json mock_response 应包含 ticker 字段"


def test_gateway_mock_delegates_to_task_not_hardcoded() -> None:
    """P1: mock=True 时 gateway 应委托 task.mock_response()，不得在 gateway 内保留 if/elif 任务名判断。"""
    import inspect
    from bench.execution.gateway import LLMGateway

    source = inspect.getsource(LLMGateway._mock_response)
    # 修复后 gateway 内不应再有形如 task_name == "ie_json" 的硬编码
    assert 'task_name == "ie_json"' not in source, (
        "Gateway._mock_response 不应硬编码任务名，应委托给 task.mock_response()"
    )
    assert 'task_name == "stock_score"' not in source, (
        "Gateway._mock_response 不应硬编码任务名，应委托给 task.mock_response()"
    )

