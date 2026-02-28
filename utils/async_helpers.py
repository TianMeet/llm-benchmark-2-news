"""异步辅助工具 — 与 llm_core/async_helpers.py 功能相同。

保留此副本以兼容已有数据管线脚本的 `from utils.async_helpers import run_async` 引用。
"""

import asyncio
import concurrent.futures


def run_async(coro):
    """
    在同步上下文安全地运行协程，兼容已有事件循环（如 Jupyter）。

    - 无事件循环 → asyncio.run()
    - 已有事件循环 → 在独立线程中执行，避免 RuntimeError
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            return pool.submit(asyncio.run, coro).result()
