"""
llm_core — 通用 LLM 调用基础设施（零业务逻辑）
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_core.async_helpers import run_async
    from llm_core.base_client import BaseLLMClient, LLMResponse
    from llm_core.batch import call_llm_batch
    from llm_core.config import load_provider_registry, load_yaml
    from llm_core.openai_client import OpenAICompatibleClient
    from llm_core.prompt_renderer import PromptRenderer
    from llm_core.response_parser import (
        JsonParseMeta,
        parse_llm_json,
        parse_llm_json_with_meta,
        validate_enum_field,
        validate_int_range,
        validate_list_field,
        validate_str_field,
    )

# ── 惰性导入映射 ─────────────────────────────────────────────────────────
_SYMBOLS: dict[str, tuple[str, str]] = {
    "BaseLLMClient": ("llm_core.base_client", "BaseLLMClient"),
    "LLMResponse": ("llm_core.base_client", "LLMResponse"),
    "run_async": ("llm_core.async_helpers", "run_async"),
    "JsonParseMeta": ("llm_core.response_parser", "JsonParseMeta"),
    "parse_llm_json": ("llm_core.response_parser", "parse_llm_json"),
    "parse_llm_json_with_meta": ("llm_core.response_parser", "parse_llm_json_with_meta"),
    "validate_str_field": ("llm_core.response_parser", "validate_str_field"),
    "validate_list_field": ("llm_core.response_parser", "validate_list_field"),
    "validate_enum_field": ("llm_core.response_parser", "validate_enum_field"),
    "validate_int_range": ("llm_core.response_parser", "validate_int_range"),
}

# 可选依赖：导入失败时返回 None 或抛出更友好的错误
_OPTIONAL_NONE: dict[str, tuple[str, str]] = {
    "PromptRenderer": ("llm_core.prompt_renderer", "PromptRenderer"),
    "OpenAICompatibleClient": ("llm_core.openai_client", "OpenAICompatibleClient"),
}

_OPTIONAL_STUB: dict[str, tuple[str, str, str]] = {
    # name -> (module, attr, install_hint)
    "call_llm_batch": ("llm_core.batch", "call_llm_batch", "pip install openai"),
    "load_yaml": ("llm_core.config", "load_yaml", "pip install pyyaml"),
    "load_provider_registry": ("llm_core.config", "load_provider_registry", "pip install pyyaml"),
}


def __getattr__(name: str):  # noqa: ANN201
    if name in _SYMBOLS:
        mod_path, attr = _SYMBOLS[name]
        return getattr(importlib.import_module(mod_path), attr)
    if name in _OPTIONAL_NONE:
        mod_path, attr = _OPTIONAL_NONE[name]
        try:
            return getattr(importlib.import_module(mod_path), attr)
        except ModuleNotFoundError:  # pragma: no cover
            return None
    if name in _OPTIONAL_STUB:
        mod_path, attr, hint = _OPTIONAL_STUB[name]
        try:
            return getattr(importlib.import_module(mod_path), attr)
        except ModuleNotFoundError:  # pragma: no cover
            def _stub(*args, _n=name, _h=hint, **kwargs):  # type: ignore[no-redef]
                raise ModuleNotFoundError(f"{_n} requires additional packages. Install with: {_h}")
            return _stub
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ── Provider 注册表 ──────────────────────────────────────────────────────
# 新增 provider 只需在此处注册，无需修改工厂函数本身（开闭原则）。
# 键名与 llm_providers.yaml 中 provider 字段对应。
# 注册表在首次使用时延迟初始化，避免提前触发子模块导入。
_PROVIDER_REGISTRY: dict[str, type] = {}
_registry_initialized: bool = False


def _ensure_registry() -> None:
    """首次调用时填充 _PROVIDER_REGISTRY（惰性初始化）。"""
    global _registry_initialized
    if _registry_initialized:
        return
    _registry_initialized = True
    try:
        from llm_core.openai_client import OpenAICompatibleClient as _OAI  # noqa: PLC0415
        _PROVIDER_REGISTRY["openai_compatible"] = _OAI
        # 兼容别名
        _PROVIDER_REGISTRY["openai"] = _OAI
    except ModuleNotFoundError:  # pragma: no cover
        pass


def create_llm_client(config: dict) -> "BaseLLMClient":
    """
    根据配置创建 LLM 客户端实例。

    provider 字段决定使用哪个客户端类；默认 "openai_compatible"。
    扩展新 provider 只需向 _PROVIDER_REGISTRY 注册，符合开闭原则。
    """
    _ensure_registry()
    provider = str(config.get("provider", "openai_compatible"))
    client_cls = _PROVIDER_REGISTRY.get(provider)
    if client_cls is None:
        available = sorted(_PROVIDER_REGISTRY.keys())
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Available providers: {available}. "
            f"Register new providers via llm_core._PROVIDER_REGISTRY."
        )
    return client_cls(config)


__all__ = [
    # 核心类型
    "LLMResponse",
    "BaseLLMClient",
    "OpenAICompatibleClient",
    # 工厂
    "create_llm_client",
    # 批量调用
    "call_llm_batch",
    # 响应解析
    "parse_llm_json",
    "parse_llm_json_with_meta",
    "JsonParseMeta",
    "validate_str_field",
    "validate_list_field",
    "validate_enum_field",
    "validate_int_range",
    # 模板渲染
    "PromptRenderer",
    # 配置
    "load_yaml",
    "load_provider_registry",
    # 异步
    "run_async",
]
