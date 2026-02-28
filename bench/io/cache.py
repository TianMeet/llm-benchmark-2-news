"""本地文件缓存：用于可复现评测与节省 API 调用成本。

缓存策略：
- 缓存键 = SHA256(model_id + model_params + params + messages + sample_id)，
  任何配置变更都会自动失效。
- 仅缓存成功结果（success=True），错误响应不写入缓存。
- 坐缓存文件损坏时按未命中处理，不中断评测流程。
- 默认缓存目录：.eval_cache/，可通过 --cache-dir 覆盖。
- --no-cache 模式下不创建缓存实例。

线程安全：使用 RLock 保护文件读写，支持同一线程重入。
"""

from __future__ import annotations

__all__ = ["EvalCache"]

import hashlib
import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EvalCache:
    """基于 JSON 文件的持久化缓存。

    线程安全设计：使用 threading.RLock 保护文件读写，
    防止并发场景下文件截断（write_text truncate→write）与读取竞争导致 JSON 解析失败。
    """

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()  # RLock 支持同一线程重入

    @staticmethod
    def make_key(payload: dict[str, Any]) -> str:
        """对请求负载做稳定序列化并计算哈希键。"""
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _path_for(self, key: str) -> Path:
        """将缓存键映射为两级子目录路径，避免单目录文件过多导致文件系统性能下降。

        结构：{cache_dir}/{hash[:2]}/{hash[2:4]}/{hash}.json
        """
        sub = self.cache_dir / key[:2] / key[2:4]
        sub.mkdir(parents=True, exist_ok=True)
        return sub / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        """读取缓存命中结果；未命中返回 None。"""
        with self._lock:
            path = self._path_for(key)
            if not path.exists():
                return None
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                # 坏缓存按未命中处理，避免中断整个评测流程。
                logger.warning("cache read failed, treat as miss (path=%s): %s", path, exc)
                return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        """原子写入缓存：先写临时文件再 rename，保证文件要么完整要么不存在。

        在 POSIX 系统上 rename 是原子操作，即使进程中途崩溃也不会产生
        空文件或截断文件。Windows 上 os.replace 也能保证覆盖原子性。
        """
        with self._lock:
            target = self._path_for(key)
            try:
                fd, tmp_path = tempfile.mkstemp(
                    dir=str(target.parent), suffix=".tmp", prefix="cache_",
                )
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(value, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, str(target))
            except Exception:
                # 清理残留临时文件
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
