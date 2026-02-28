"""
runner 共享常量与辅助函数。

供 task_runner / workflow_runner / cli.runner 共同使用，
包含：
- 版本常量：RESULT_ROW_SCHEMA_VERSION / SCORER_VERSION
- 数据集加载：load_dataset()
- Git 信息采集：git_commit_hash() / git_is_dirty()
- 数据集指纹：sha256_file() / dataset_fingerprint()
- 运行时元数据：runtime_metadata()  # python 版本、平台
- 敏感字段脱敏：redact_sensitive_fields()  # api_key 等替换为 ***
- CLI 参数解析：parse_params() / parse_workflow_models() / resolve_concurrency()
"""

from __future__ import annotations

import json
import subprocess
import hashlib
import platform
import sys
from pathlib import Path
from typing import Any

from bench.contracts.constants import STEP_ID_TASK
from bench.metrics.aggregate import SUMMARY_ROW_SCHEMA_VERSION
from bench.registry import ModelRegistry

__all__ = [
    "RESULT_ROW_SCHEMA_VERSION",
    "SUMMARY_ROW_SCHEMA_VERSION",
    "SCORER_VERSION",
    "STEP_ID_TASK",
    "load_dataset",
    "git_commit_hash",
    "parse_params",
    "parse_workflow_models",
    "resolve_concurrency",
    "git_is_dirty",
    "sha256_file",
    "dataset_fingerprint",
    "runtime_metadata",
    "redact_sensitive_fields",
]

RESULT_ROW_SCHEMA_VERSION = "result_row.v1"
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


def git_is_dirty() -> bool:
    """读取当前仓库是否存在未提交改动。"""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:
        return False


def sha256_file(path: str | Path) -> str:
    """计算文件 SHA256 指纹。"""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_fingerprint(path: str | Path) -> dict[str, Any]:
    """返回数据集路径/行数/SHA256，用于 run 可审计。"""
    p = Path(path)
    rows = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows += 1
    return {
        "dataset_path": str(p),
        "dataset_rows": rows,
        "dataset_sha256": sha256_file(p),
    }


def runtime_metadata() -> dict[str, Any]:
    """返回运行时环境元信息。"""
    return {
        "python_version": sys.version.split(" ")[0],
        "platform": platform.platform(),
    }


def redact_sensitive_fields(payload: Any) -> Any:
    """Recursively redact sensitive credential-like fields.

    使用精确的敏感字段名列表匹配，避免过度脱敏（如误命中 cache_key、primary_key 等）。
    匹配规则：字段名完全等于、或以 ``_`` 分隔后包含敏感词根。
    """
    # 完整匹配这些字段名，或字段名以 _ 分隔后任一段匹配
    _SENSITIVE_EXACT = frozenset({
        "api_key", "apikey", "secret", "secret_key", "secretkey",
        "password", "passwd", "token", "access_token", "refresh_token",
        "credential", "auth", "authorization", "auth_token",
    })
    _SENSITIVE_SUFFIXES = ("_key", "_secret", "_token", "_password", "_credential")
    if isinstance(payload, dict):
        out: dict[str, Any] = {}
        for k, v in payload.items():
            key_lower = str(k).lower()
            is_sensitive = (
                key_lower in _SENSITIVE_EXACT
                or any(key_lower.endswith(suf) for suf in _SENSITIVE_SUFFIXES)
            )
            if is_sensitive:
                out[k] = "***"
            else:
                out[k] = redact_sensitive_fields(v)
        return out
    if isinstance(payload, list):
        return [redact_sensitive_fields(x) for x in payload]
    return payload


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
