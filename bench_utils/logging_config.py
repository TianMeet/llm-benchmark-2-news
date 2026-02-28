"""
统一日志配置模块。

==============
职责
==============
1. 为整个评测中台提供统一的日志格式与级别。
2. 支持可选的文件日志输出（run 目录下的 run.log）。
3. 确保重复调用 setup_logging() 是幂等的（不会叠加 handler）。

==============
使用方式
==============
在 CLI 入口（bench/cli/runner.py）的 main() 中调用一次：

    from bench_utils.logging_config import setup_logging
    setup_logging()                           # 仅 stderr
    setup_logging(log_file="runs/.../run.log")  # stderr + 文件

其他模块正常使用标准 logging：

    import logging
    logger = logging.getLogger(__name__)
    logger.info("...")

==============
设计说明
==============
- 全局仅配置 root logger，各子模块通过 getLogger(__name__) 自动继承。
- 文件 handler 使用 UTF-8 编码，便于包含中文日志。
- 控制台输出到 stderr（Python logging 默认行为）。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

__all__ = ["setup_logging", "LOG_FORMAT"]

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"

# 防止重复调用时叠加 handler
_INITIALIZED = False


def setup_logging(
    level: int = logging.INFO,
    *,
    log_file: str | Path | None = None,
) -> None:
    """初始化全局日志配置（幂等）。

    Parameters
    ----------
    level : int
        全局日志级别，默认 INFO。
    log_file : str | Path | None
        可选的日志文件路径。传入后会额外添加一个写入文件的 handler，
        通常设为 ``{run_dir}/run.log``，便于事后排查历史 run 的问题。
    """
    global _INITIALIZED

    root = logging.getLogger()

    if not _INITIALIZED:
        # ── 首次初始化：设置级别 + stderr handler ────────────────────
        root.setLevel(level)

        console = logging.StreamHandler(sys.stderr)
        console.setLevel(level)
        console.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(console)

        _INITIALIZED = True

    if log_file is not None:
        # ── 可选：添加文件 handler（每次 run 调用一次）──────────────
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # 避免对同一文件重复添加 handler
        existing = [
            h for h in root.handlers
            if isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_path.resolve()
        ]
        if not existing:
            fh = logging.FileHandler(str(log_path), encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter(LOG_FORMAT))
            root.addHandler(fh)
