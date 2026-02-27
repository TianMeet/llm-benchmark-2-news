"""
OpenAI 兼容 LLM 客户端 — 适用于 DeepSeek、Kimi、及所有 OpenAI 协议兼容端点
"""

import logging
import time

import openai

from llm_core.base_client import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)


class OpenAICompatibleClient(BaseLLMClient):
    """基于 openai.AsyncOpenAI 的通用客户端，支持所有 OpenAI 协议兼容 API。"""

    RATE_LIMIT_ERRORS = (openai.RateLimitError,)
    RETRYABLE_ERRORS = (openai.APITimeoutError, openai.InternalServerError)
    NON_RETRYABLE_ERRORS = (openai.APIError,)

    def __init__(self, config: dict):
        super().__init__(config)
        api_key = config.get("api_key", "")
        if not api_key:
            raise ValueError("API key is required in config (llm.api_key)")
        base_url = config.get("base_url", "")
        if not base_url:
            raise ValueError("base_url is required in config (llm.base_url)")
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.timeout,
        )
        logger.info(
            "OpenAI-compatible client initialized (model=%s, base_url=%s)",
            self.model, base_url,
        )

    async def _do_complete(self, messages: list[dict]) -> LLMResponse:
        """执行单次 Chat Completions API 调用。"""
        t0 = time.perf_counter()
        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=messages,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        text = response.choices[0].message.content if response.choices else ""
        return LLMResponse(
            text=text,
            model=response.model,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            latency_ms=latency_ms,
            success=True,
        )
