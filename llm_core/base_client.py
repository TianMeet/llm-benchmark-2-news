"""
LLM 客户端抽象基类 + 标准化响应数据类
"""

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLM API 调用的标准化返回结果。"""
    text: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error_message: str = ""
    retry_count: int = 0
    error_type: str = ""
    error_stage: str = ""
    error_code: str = ""


class BaseLLMClient(ABC):
    """LLM 客户端抽象基类，所有 provider 实现此接口。"""

    RATE_LIMIT_ERRORS: tuple = ()
    RETRYABLE_ERRORS: tuple = ()
    NON_RETRYABLE_ERRORS: tuple = ()

    def __init__(self, config: dict):
        self.model = config.get("model", "")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 1024)
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 2)
        self.retry_delay = config.get("retry_delay", 2.0)
        self.retry_jitter = float(config.get("retry_jitter", 0.2))
        self.seed: int | None = config.get("seed", None)

    def _with_jitter(self, delay: float) -> float:
        if self.retry_jitter <= 0:
            return max(0.0, delay)
        jitter = random.uniform(0.0, delay * self.retry_jitter)
        return max(0.0, delay + jitter)

    async def complete(self, messages: list[dict]) -> LLMResponse:
        """
        发送消息列表到 LLM，返回标准化响应。

        统一的重试循环 + 指数退避 + 延迟测量。
        子类通过 _do_complete() 实现单次 API 调用。
        """
        last_error = None
        last_error_code = "unknown_error"
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = await self._do_complete(messages)
                resp.retry_count = attempt
                return resp

            except self.RATE_LIMIT_ERRORS as e:
                last_error = str(e)
                last_error_code = "rate_limit"
                delay = self._with_jitter(self.retry_delay * (2 ** attempt) + 3.0)
                if attempt < self.max_retries:
                    logger.warning(
                        "Rate limited (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, self.max_retries + 1, delay, e,
                    )
                    await asyncio.sleep(delay)

            except self.RETRYABLE_ERRORS as e:
                last_error = str(e)
                last_error_code = "retryable_api_error"
                delay = self._with_jitter(self.retry_delay * (2 ** attempt))
                if attempt < self.max_retries:
                    logger.warning(
                        "API error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, self.max_retries + 1, delay, e,
                    )
                    await asyncio.sleep(delay)

            except self.NON_RETRYABLE_ERRORS as e:
                latency_ms = (time.perf_counter() - t0) * 1000
                logger.error("API error (non-retryable): %s", e)
                return LLMResponse(
                    success=False,
                    error_message=str(e),
                    model=self.model,
                    latency_ms=latency_ms,
                    retry_count=attempt,
                    error_type="gateway_failure",
                    error_stage="gateway",
                    error_code="non_retryable_api_error",
                )

        return LLMResponse(
            success=False,
            error_message=f"All {self.max_retries + 1} attempts failed. Last error: {last_error}",
            model=self.model,
            latency_ms=0.0,
            retry_count=self.max_retries,
            error_type="rate_limit" if last_error_code == "rate_limit" else "gateway_failure",
            error_stage="gateway",
            error_code=f"{last_error_code}_max_retries_exceeded",
        )

    @abstractmethod
    async def _do_complete(self, messages: list[dict]) -> LLMResponse:
        """执行单次 LLM API 调用（无重试）。子类实现此方法。"""
        ...
