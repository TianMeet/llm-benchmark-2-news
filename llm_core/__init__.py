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
    parse_llm_json,
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


def create_llm_client(config: dict) -> BaseLLMClient:
    """
    根据配置创建 LLM 客户端实例。

    所有 provider 统一使用 OpenAI 兼容协议（openai SDK）。
    """
    if OpenAICompatibleClient is None:
        raise ModuleNotFoundError(
            "openai package is required to create LLM client. Install with: pip install openai"
        )
    return OpenAICompatibleClient(config)


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
