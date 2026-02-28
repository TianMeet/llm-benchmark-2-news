"""模型调用门面：封装真实调用、缓存与 mock 返回。"""

from __future__ import annotations

__all__ = ["ModelGateway", "LLMGateway"]

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any

from eval.cache import EvalCache
from eval.contracts import UnifiedCallResult
from eval.registry import ModelRegistry


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

    def __init__(self, registry: ModelRegistry, cache: EvalCache | None = None, mock: bool = False):
        self.registry = registry
        self.cache = cache
        self.mock = mock
        self._client_pool: dict[str, Any] = {}
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
            return client

        async with self._client_pool_lock:
            # 双重检查，避免并发下重复创建。
            client = self._client_pool.get(key)
            if client is not None:
                return client
            client = self.registry.create_client(model_id, params_override=params_override)
            self._client_pool[key] = client
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
        client = await self._get_or_create_client(model_id, params_override)
        llm_resp = await client.complete(messages)
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
            from eval.tasks import build_task
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
