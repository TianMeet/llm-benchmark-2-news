"""模型注册表抽象：统一读取模型配置并创建客户端。"""

from __future__ import annotations

__all__ = ["ModelSpec", "ModelRegistry", "estimate_cost"]

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm_core.base_client import BaseLLMClient


@dataclass(frozen=True)
class ModelSpec:
    """模型配置快照（评测时使用的解析结果）。"""

    model_id: str
    provider: str
    default_params: dict[str, Any]
    price_config: dict[str, float]


class ModelRegistry:
    """加载并解析模型注册表（YAML/JSON）。"""

    def __init__(self, registry_path: str | Path):
        self.registry_path = Path(registry_path)
        raw = self._load_registry(self.registry_path)
        models = raw.get("models", {})
        self.active = raw.get("active")
        self._models: dict[str, ModelSpec] = {}
        for model_id, cfg in models.items():
            cfg = dict(cfg)
            # default_params 保留完整配置，便于后续透传与覆盖。
            self._models[model_id] = ModelSpec(
                model_id=model_id,
                provider=str(cfg.get("provider", "unknown")),
                default_params=cfg,
                price_config=dict(cfg.get("price_config", {}) or {}),
            )

    @staticmethod
    def _load_registry(path: Path) -> dict[str, Any]:
        """根据后缀自动加载 JSON 或 YAML 注册表（均展开 ${ENV_VAR}）。"""
        from llm_core.config import load_yaml, _resolve_env_vars, _load_dotenv_file

        if path.suffix.lower() == ".json":
            # 与 YAML 路径保持一致：优先加载配置同级 .env
            _load_dotenv_file(path.parent / ".env", override=False)
            raw = json.loads(path.read_text(encoding="utf-8"))
            return _resolve_env_vars(raw)
        return load_yaml(path)

    def list_model_ids(self) -> list[str]:
        """返回可用 model_id 列表。"""
        return sorted(self._models.keys())

    def get(self, model_id: str) -> ModelSpec:
        """按 model_id 获取模型配置，不存在则抛错。"""
        if model_id not in self._models:
            raise ValueError(
                f"Unknown model_id '{model_id}' in registry '{self.registry_path}'. "
                f"Available: {self.list_model_ids()}"
            )
        return self._models[model_id]

    def resolve_config(self, model_id: str, params_override: dict[str, Any] | None = None) -> dict[str, Any]:
        """合并默认参数与运行时覆盖参数。"""
        spec = self.get(model_id)
        merged = dict(spec.default_params)
        if params_override:
            merged.update(params_override)
        # 提前检测未解析的环境变量占位符，避免深层 API 报错难以定位。
        api_key = merged.get("api_key", "")
        if isinstance(api_key, str) and "${" in api_key:
            raise ValueError(
                f"api_key for model '{model_id}' contains unresolved env var '{api_key}'. "
                f"Please set the corresponding environment variable."
            )
        return merged

    def create_client(self, model_id: str, params_override: dict[str, Any] | None = None) -> "BaseLLMClient":
        """按 model_id 创建并返回对应的 LLM 客户端实例。

        懒加载 llm_core，可降低单元测试对 openai 依赖的耦合。
        """
        from llm_core import create_llm_client

        return create_llm_client(self.resolve_config(model_id, params_override))


def estimate_cost(usage: dict[str, int], price_config: dict[str, float] | None) -> float:
    """按每千 token 单价估算成本；未配置价格时返回 0。"""
    if not price_config:
        return 0.0
    in_per_1k = float(price_config.get("input_per_1k", 0.0))
    out_per_1k = float(price_config.get("output_per_1k", 0.0))
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    return (prompt_tokens / 1000.0) * in_per_1k + (completion_tokens / 1000.0) * out_per_1k
