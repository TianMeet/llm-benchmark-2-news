"""
llm_core — 通用 LLM 调用基础设施（零业务逻辑）
"""

from llm_core.base_client import BaseLLMClient, LLMResponse
from llm_core.async_helpers import run_async
try:
    from llm_core.batch import call_llm_batch
except ModuleNotFoundError:  # pragma: no cover
    def call_llm_batch(*args, **kwargs):  # type: ignore[no-redef]
        raise ModuleNotFoundError("openai package is required. Install with: pip install openai")
from llm_core.response_parser import (
    JsonParseMeta,
    parse_llm_json,
    parse_llm_json_with_meta,
    validate_str_field,
    validate_list_field,
    validate_enum_field,
    validate_int_range,
)
try:
    from llm_core.prompt_renderer import PromptRenderer
except ModuleNotFoundError:  # pragma: no cover
    PromptRenderer = None  # type: ignore[assignment]

try:
    from llm_core.config import load_yaml, load_provider_registry
except ModuleNotFoundError:  # pragma: no cover
    def load_yaml(*args, **kwargs):  # type: ignore[no-redef]
        raise ModuleNotFoundError("pyyaml is required. Install with: pip install pyyaml")

    def load_provider_registry(*args, **kwargs):  # type: ignore[no-redef]
        raise ModuleNotFoundError("pyyaml is required. Install with: pip install pyyaml")

try:
    from llm_core.openai_client import OpenAICompatibleClient
except ModuleNotFoundError:  # pragma: no cover
    OpenAICompatibleClient = None  # type: ignore[assignment]


# ── Provider 注册表 ──────────────────────────────────────────────────────
# 新增 provider 只需在此处注册，无需修改工厂函数本身（开闭原则）。
# 键名与 llm_providers.yaml 中 provider 字段对应。
_PROVIDER_REGISTRY: dict[str, type] = {}

if OpenAICompatibleClient is not None:
    _PROVIDER_REGISTRY["openai_compatible"] = OpenAICompatibleClient
    # 兼容别名
    _PROVIDER_REGISTRY["openai"] = OpenAICompatibleClient


def create_llm_client(config: dict) -> BaseLLMClient:
    """
    根据配置创建 LLM 客户端实例。

    provider 字段决定使用哪个客户端类；默认 "openai_compatible"。
    扩展新 provider 只需向 _PROVIDER_REGISTRY 注册，符合开闭原则。
    """
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
