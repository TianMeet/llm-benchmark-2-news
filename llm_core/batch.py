"""
LLM 并发批量调用 — Semaphore 控制并发度 + 延迟统计。

主入口：call_llm_batch()
- 按原始顺序返回结果（缺失位置用 _FAILED_SENTINEL 哨兵值填充）
- 延迟分布概览（avg / p50 / p99 / min / max）
- 支持外部传入预构建 client（避免重复创建）

注意：在评测中台场景下，task_runner 和 workflow_runner 有各自的并发调度逻辑，
本函数主要用于独立脚本或数据管线场景。
"""

import asyncio
import logging
import statistics

from llm_core.base_client import BaseLLMClient, LLMResponse
from llm_core.openai_client import OpenAICompatibleClient

logger = logging.getLogger(__name__)

# LLM 调用失败时的哨兵响应，避免 None 导致 AttributeError
_FAILED_SENTINEL = LLMResponse(
    success=False,
    error_message="Task result missing (internal error)",
    model="",
)


def _create_client(llm_cfg: dict) -> BaseLLMClient:
    """内部工厂：从配置创建客户端（避免循环导入 __init__）。"""
    return OpenAICompatibleClient(llm_cfg)


async def call_llm_batch(
    all_messages: list[list[dict]],
    llm_cfg: dict,
    concurrency: int,
    *,
    label: str = "item",
    client: BaseLLMClient | None = None,
) -> list[LLMResponse]:
    """
    使用 Semaphore 控制并发批量调用 LLM。

    Parameters
    ----------
    all_messages : list[list[dict]]
        每个元素是一组 messages（即一次 LLM 调用的输入）。
    llm_cfg : dict
        LLM 配置字典，传给 create_llm_client()。
    concurrency : int
        最大并发数。
    label : str
        日志中的条目名称（如 "row"、"stock"），仅影响日志输出。
    client : BaseLLMClient | None
        预构建的 LLM 客户端实例，传入时忽略 llm_cfg 直接使用。

    Returns
    -------
    list[LLMResponse]
        与 all_messages 等长、顺序对齐的响应列表。
    """
    if client is None:
        client = _create_client(llm_cfg)
    semaphore = asyncio.Semaphore(concurrency)

    async def _call_one(idx: int, messages: list[dict]) -> tuple[int, LLMResponse]:
        async with semaphore:
            response = await client.complete(messages)
            if not response.success:
                logger.warning("LLM call failed for %s %d: %s", label, idx, response.error_message)
            return idx, response

    tasks = [_call_one(i, msgs) for i, msgs in enumerate(all_messages)]
    results_unordered = await asyncio.gather(*tasks)

    # 按原始顺序排列，缺失位置使用哨兵值
    results: list[LLMResponse] = [_FAILED_SENTINEL] * len(all_messages)
    for idx, resp in results_unordered:
        results[idx] = resp

    # ---- 统计 ----
    success_count = sum(1 for r in results if r.success)
    total_input = sum(r.input_tokens for r in results if r.success)
    total_output = sum(r.output_tokens for r in results if r.success)
    logger.info(
        "LLM batch done — %d/%d success, tokens: %d in / %d out",
        success_count, len(results), total_input, total_output,
    )

    # 延迟分布（仅成功的调用）
    latencies = [r.latency_ms for r in results if r.success and r.latency_ms > 0]
    if latencies:
        avg_ms = statistics.mean(latencies)
        p50_ms = statistics.median(latencies)
        p99_ms = sorted(latencies)[min(int(len(latencies) * 0.99), len(latencies) - 1)]
        logger.info(
            "LLM latency — avg: %.0fms, p50: %.0fms, p99: %.0fms, min: %.0fms, max: %.0fms",
            avg_ms, p50_ms, p99_ms, min(latencies), max(latencies),
        )

    return results
