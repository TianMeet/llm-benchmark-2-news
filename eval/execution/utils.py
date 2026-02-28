"""runner 共享常量与辅助函数 — 供 runner_task / runner_workflow / runner 共同使用。"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from eval.registry import ModelRegistry

__all__ = [
    "RESULT_ROW_SCHEMA_VERSION",
    "SUMMARY_ROW_SCHEMA_VERSION",
    "SCORER_VERSION",
    "load_dataset",
    "git_commit_hash",
    "parse_params",
    "parse_workflow_models",
    "resolve_concurrency",
]

RESULT_ROW_SCHEMA_VERSION = "result_row.v1"
SUMMARY_ROW_SCHEMA_VERSION = "summary_row.v1"
SCORER_VERSION = "v1"


def load_dataset(path: str | Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    """加载 JSONL 数据集，可选限制样本数量。"""
    if max_samples == 0:
        return []
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples and len(rows) >= max_samples:
                break
    return rows


def git_commit_hash() -> str:
    """读取当前 git commit，失败时返回 unknown。"""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out
    except Exception:
        return "unknown"


def parse_params(params_str: str | None) -> dict[str, Any]:
    """解析 CLI 传入的 JSON 参数覆盖。"""
    if not params_str:
        return {}
    return dict(json.loads(params_str))


def parse_workflow_models(
    models_str: str,
    registry: ModelRegistry,
) -> tuple[str, dict[str, str]]:
    """解析 workflow 模式下的 --models 参数，返回 (default_model_id, step_model_map)。

    支持两种格式：
    - 单一模型 ID：``kimi-personal``  →  所有 $active 步骤均使用该模型
    - 步骤映射：``step1_dedup=kimi-personal,step2_extract=deepseek-v3``  →  按步骤 ID 单独指定
      未出现在映射中的 $active 步骤回退到 registry.active
    """
    if not models_str:
        default = registry.active or ""
        return default, {}
    if "=" in models_str:
        step_map: dict[str, str] = {}
        for part in models_str.split(","):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                step_map[k.strip()] = v.strip()
        return registry.active or "", step_map
    return models_str.strip(), {}


def resolve_concurrency(
    model_ids: list[str],
    registry: ModelRegistry,
    user_value: int | None,
) -> dict[str, int]:
    """返回每个 model_id 的独立并发数。

    规则（依优先级）：
    - CLI --concurrency 不为 None 时：所有模型统一设为该值
    - 否则各模型独立读取自身配置中的 ``concurrency`` 字段，缺省 8
    0 = 无限制
    """
    result: dict[str, int] = {}
    for mid in model_ids:
        if user_value is not None:
            result[mid] = user_value
        else:
            cfg_val = int(registry.get(mid).default_params.get("concurrency", 8))
            result[mid] = cfg_val
    return result
