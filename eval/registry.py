"""Model registry abstraction for evaluation runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    """Resolved model definition from provider registry."""

    model_id: str
    provider: str
    default_params: dict[str, Any]
    price_config: dict[str, float]


class ModelRegistry:
    """Loads and resolves model configuration from YAML registry."""

    def __init__(self, registry_path: str | Path):
        self.registry_path = Path(registry_path)
        raw = self._load_registry(self.registry_path)
        models = raw.get("models", {})
        self.active = raw.get("active")
        self._models: dict[str, ModelSpec] = {}
        for model_id, cfg in models.items():
            cfg = dict(cfg)
            self._models[model_id] = ModelSpec(
                model_id=model_id,
                provider=str(cfg.get("provider", "unknown")),
                default_params=cfg,
                price_config=dict(cfg.get("price_config", {}) or {}),
            )

    @staticmethod
    def _load_registry(path: Path) -> dict[str, Any]:
        if path.suffix.lower() == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        from llm_core.config import load_yaml

        return load_yaml(path)

    def list_model_ids(self) -> list[str]:
        return sorted(self._models.keys())

    def get(self, model_id: str) -> ModelSpec:
        if model_id not in self._models:
            raise ValueError(f"Unknown model_id '{model_id}'. Available: {self.list_model_ids()}")
        return self._models[model_id]

    def resolve_config(self, model_id: str, params_override: dict[str, Any] | None = None) -> dict[str, Any]:
        spec = self.get(model_id)
        merged = dict(spec.default_params)
        if params_override:
            merged.update(params_override)
        return merged

    def create_client(self, model_id: str, params_override: dict[str, Any] | None = None):
        # Lazy import keeps non-network/unit-test paths independent of openai package.
        from llm_core import create_llm_client

        return create_llm_client(self.resolve_config(model_id, params_override))


def estimate_cost(usage: dict[str, int], price_config: dict[str, float] | None) -> float:
    """Estimate cost by per-1k token pricing; returns 0.0 when missing config."""
    if not price_config:
        return 0.0
    in_per_1k = float(price_config.get("input_per_1k", 0.0))
    out_per_1k = float(price_config.get("output_per_1k", 0.0))
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    return (prompt_tokens / 1000.0) * in_per_1k + (completion_tokens / 1000.0) * out_per_1k
