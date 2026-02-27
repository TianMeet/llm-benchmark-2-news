import asyncio

from llm_core.base_client import BaseLLMClient


class _RetryableError(Exception):
    pass


class _AlwaysRetryableClient(BaseLLMClient):
    RATE_LIMIT_ERRORS = ()
    RETRYABLE_ERRORS = (_RetryableError,)
    NON_RETRYABLE_ERRORS = ()

    async def _do_complete(self, messages: list[dict]):
        raise _RetryableError("boom")


def test_no_sleep_on_last_failed_attempt(monkeypatch) -> None:
    sleeps: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    client = _AlwaysRetryableClient({"max_retries": 0, "retry_delay": 0.2})
    resp = asyncio.run(client.complete([]))

    assert resp.success is False
    assert sleeps == []


def test_sleep_only_between_attempts(monkeypatch) -> None:
    sleeps: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    client = _AlwaysRetryableClient({"max_retries": 2, "retry_delay": 0.5})
    resp = asyncio.run(client.complete([]))

    assert resp.success is False
    assert sleeps == [0.5, 1.0]
