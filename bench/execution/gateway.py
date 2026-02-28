"""模型调用门面：封装真实 API 调用、缓存与 mock 返回。

这是评测系统与底层 LLM 服务之间的唯一接触点，责任包括：
1. SHA256 缓存键生成（model + params + messages + sample_id）
2. 缓存命中/写入（仅缓存成功结果，避免错误响应污染）
3. 客户端连接池（相同 model_id + params 复用 client）
4. Mock 模式（离线开发/CI 中不触发真实 API）
5. 统一错误包装为 GatewayCallError

显式抽象接口 ModelGateway 便于测试和后续插入新的调用策略。
"""

from __future__ import annotations

__all__ = ["ModelGateway", "LLMGateway"]

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any

from bench.contracts.result import UnifiedCallResult
from bench.contracts.exceptions import GatewayCallError
from bench.io.cache import EvalCache
from bench.registry import ModelRegistry


class ModelGateway(ABC):
    """模型调用抽象接口，便于后续替换 provider 或并发策略。"""

    mock: bool = False  # 子类或测试 gateway 可覆盖此属性

    @abstractmethod
    async def call(
        self,
        *,
        model_id: str,
        task_name: str,
        sample: dict[str, Any],
        sample_cache_id: str,
        messages: list[dict[str, Any]],
        params_override: dict[str, Any],
    ) -> UnifiedCallResult:
        """发起一次模型调用，返回统一结果结构。"""
        ...


class LLMGateway(ModelGateway):
    """基于现有 llm_core 的调用实现，支持可复现缓存。"""

    # 连接池默认最大容量——超过时淘汰最早的条目。
    _DEFAULT_POOL_MAXSIZE: int = 32

    def __init__(
        self,
        registry: ModelRegistry,
        cache: EvalCache | None = None,
        mock: bool = False,
        pool_maxsize: int | None = None,
    ):
        self.registry = registry
        self.cache = cache
        self.mock = mock
        self._pool_maxsize = pool_maxsize or self._DEFAULT_POOL_MAXSIZE
        # 使用 OrderedDict 实现简易 LRU 淘汰。
        from collections import OrderedDict
        self._client_pool: OrderedDict[str, Any] = OrderedDict()
        self._client_pool_lock = asyncio.Lock()

    @staticmethod
    def _client_pool_key(model_id: str, params_override: dict[str, Any]) -> str:
        payload = {
            "model_id": model_id,
            "params": params_override,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    async def _get_or_create_client(self, model_id: str, params_override: dict[str, Any]) -> Any:
        key = self._client_pool_key(model_id, params_override)
        client = self._client_pool.get(key)
        if client is not None:
            # 访问时移动到末尾（LRU）。
            self._client_pool.move_to_end(key)
            return client

        async with self._client_pool_lock:
            # 双重检查，避免并发下重复创建。
            client = self._client_pool.get(key)
            if client is not None:
                self._client_pool.move_to_end(key)
                return client
            client = self.registry.create_client(model_id, params_override=params_override)
            self._client_pool[key] = client
            # 淘汰最早的条目。
            while len(self._client_pool) > self._pool_maxsize:
                self._client_pool.popitem(last=False)
            return client

    async def call(
        self,
        *,
        model_id: str,
        task_name: str,
        sample: dict[str, Any],
        sample_cache_id: str,
        messages: list[dict[str, Any]],
        params_override: dict[str, Any],
    ) -> UnifiedCallResult:
        # 本地离线模式：不触发真实 API 调用。
        if self.mock:
            return self._mock_response(task_name, sample)

        cache_key = ""
        if self.cache is not None:
            # 缓存键包含模型、模型默认参数、消息与样本标识，保证配置变更后自动失效。
            model_default_params = dict(self.registry.get(model_id).default_params)
            cache_key = self.cache.make_key(
                {
                    "model_id": model_id,
                    "model_params": model_default_params,
                    "params": params_override,
                    "messages": messages,
                    "sample_id": sample_cache_id,
                }
            )
            cached = self.cache.get(cache_key)
            if cached:
                cached.pop("from_cache", None)
                return UnifiedCallResult(**cached, from_cache=True)

        # 命中失败后走真实模型调用。
        try:
            client = await self._get_or_create_client(model_id, params_override)
            llm_resp = await client.complete(messages)
        except Exception as exc:
            raise GatewayCallError(str(exc)) from exc
        usage = {
            "prompt_tokens": int(llm_resp.input_tokens),
            "completion_tokens": int(llm_resp.output_tokens),
            "total_tokens": int(llm_resp.input_tokens + llm_resp.output_tokens),
        }
        result = UnifiedCallResult(
            content=llm_resp.text,
            usage=usage,
            latency_ms=float(llm_resp.latency_ms),
            raw_response={
                "model": llm_resp.model,
                "success": llm_resp.success,
                "error_message": llm_resp.error_message,
            },
            success=bool(llm_resp.success),
            error=str(llm_resp.error_message or ""),
            retry_count=int(getattr(llm_resp, "retry_count", 0)),
            error_type=str(getattr(llm_resp, "error_type", "") or ""),
            error_stage=str(getattr(llm_resp, "error_stage", "") or ""),
            error_code=str(getattr(llm_resp, "error_code", "") or ""),
        )
        if self.cache is not None and cache_key and result.success:
            # 只缓存成功结果，避免错误响应污染缓存。
            self.cache.set(cache_key, asdict(result))
        return result

    @staticmethod
    def _mock_response(task_name: str, sample: dict[str, Any]) -> UnifiedCallResult:
        """从任务自身的 mock_response() 获取响应内容，不在 Gateway 中硬编码任务名。

        若任务未知或构建失败，则回退到空 JSON 对象 "{}"。
        """
        try:
            from bench.tasks import build_task
            content = build_task(task_name).mock_response(sample)
        except Exception:
            content = "{}"
        return UnifiedCallResult(
            content=content,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            latency_ms=1.0,
            raw_response={"model": "mock", "success": True, "error_message": ""},
            success=True,
            error="",
            retry_count=0,
            from_cache=False,
        )
