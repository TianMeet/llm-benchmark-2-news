"""
OpenAI 兼容 LLM 客户端 — 适用于 DeepSeek、Kimi、及所有 OpenAI 协议兼容端点。

基于 openai.AsyncOpenAI，支持参数：
  model, temperature, max_tokens, top_p, stop, response_format, seed

配合 base_client.BaseLLMClient 的重试机制：
  RateLimitError   → 可重试（加长退避）
  APITimeoutError  → 可重试（标准退避）
  InternalServerError → 可重试
  APIError         → 不可重试
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
    # 精确列出不可重试的错误子类，避免父类 APIError 过宽匹配
    # （若 openai 库新增可重试子类不会被误捕获为不可重试）。
    NON_RETRYABLE_ERRORS = (
        openai.AuthenticationError,
        openai.PermissionDeniedError,
        openai.BadRequestError,
        openai.NotFoundError,
        openai.UnprocessableEntityError,
        openai.ConflictError,
    )

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
        self.top_p = config.get("top_p")
        self.stop = config.get("stop")
        self.response_format = config.get("response_format")

    async def _do_complete(self, messages: list[dict]) -> LLMResponse:
        """执行单次 Chat Completions API 调用。"""
        t0 = time.perf_counter()
        kwargs: dict = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=messages,
        )
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.stop is not None:
            kwargs["stop"] = self.stop
        if self.response_format is not None:
            kwargs["response_format"] = self.response_format
        response = await self.client.chat.completions.create(**kwargs)
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
