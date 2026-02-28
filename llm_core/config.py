"""配置加载工具 — YAML 加载 + 环境变量解析 + .env 自动加载（零业务依赖）。

核心功能：
- load_yaml(path)：加载 YAML 并递归解析 ${ENV_VAR}
- load_provider_registry(path)：从模型注册表选取活跃模型配置
- .env 加载：自实现（无第三方依赖），支持 `export KEY=VALUE` 和引号值
- 环境变量解析：字符串中的 ${VAR} 替换为 os.environ[VAR]，
  缺失时保留原始占位符（供下游检测报错）
"""

import os
import re
from pathlib import Path

import yaml

_ENV_VAR_PATTERN = re.compile(r'\$\{(\w+)\}')
_DOTENV_LOADED = False


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    """解析 .env 单行，支持 `export KEY=VALUE` 与引号值。"""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export "):].strip()
    if "=" not in stripped:
        return None
    key, raw_value = stripped.split("=", 1)
    key = key.strip()
    if not key:
        return None
    value = raw_value.strip()
    if len(value) >= 2 and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
        value = value[1:-1]
    return key, value


def _load_dotenv_file(path: Path, *, override: bool = False) -> None:
    """从指定文件加载环境变量。"""
    if not path.exists() or not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_dotenv_line(line)
        if not parsed:
            continue
        key, value = parsed
        if override or key not in os.environ:
            os.environ[key] = value


def _ensure_dotenv_loaded() -> None:
    """懒加载 .env（默认当前工作目录）。"""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _load_dotenv_file(Path.cwd() / ".env", override=False)
    _DOTENV_LOADED = True


def _resolve_env_vars(value):
    """
    递归解析配置值中的 ${ENV_VAR} 引用。

    - 字符串: 替换 ${VAR} 为 os.environ[VAR]，缺失时保留原始占位符
    - dict/list: 递归处理
    - 其他类型: 原样返回
    """
    _ensure_dotenv_loaded()
    if isinstance(value, str):
        def _replace(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))
        return _ENV_VAR_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


def load_yaml(config_path: str | Path) -> dict:
    """加载 YAML 文件并解析 ${ENV_VAR} 引用。

    Raises
    ------
    FileNotFoundError
        配置文件不存在时抛出。
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    # 优先加载配置文件同级 .env，覆盖 cwd 默认行为的目录差异。
    _load_dotenv_file(config_path.parent / ".env", override=False)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return _resolve_env_vars(config)


def load_provider_registry(
    providers_path: str | Path,
    active_override: str | None = None,
) -> dict:
    """
    从模型注册表选取活跃 provider 配置。

    Parameters
    ----------
    providers_path : str | Path
        模型注册表 YAML 路径（如 llm_providers.yaml）。
    active_override : str | None
        强制指定活跃模型名，为 None 时使用注册表中 active 字段。

    Returns
    -------
    dict
        选中模型的配置字典（含 provider/model/api_key/base_url 等）。

    Raises
    ------
    FileNotFoundError
        注册表文件不存在。
    ValueError
        指定的模型名不在注册表中。
    """
    registry = load_yaml(providers_path)
    models = registry.get("models", {})
    active = active_override or registry.get("active", "")

    if not active:
        raise ValueError(
            f"No active model specified. Available: {list(models.keys())}"
        )
    if active not in models:
        raise ValueError(
            f"Model '{active}' not found in {providers_path}. "
            f"Available: {list(models.keys())}"
        )

    return models[active]
