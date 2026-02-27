"""
管线性能计时器 — 复用于 ODS / DWD / DWS / ARCHIVE 管线
"""

import logging
import time

import pandas as pd

logger = logging.getLogger(__name__)


class PipelineTimer:
    """管线步骤计时与性能汇总。"""

    def __init__(self):
        self.timings: list[dict] = []

    @staticmethod
    def _count_rows(result) -> int:
        """从步骤返回值中推断行数。"""
        if isinstance(result, pd.DataFrame):
            return len(result)
        if isinstance(result, list):
            return len(result)
        if isinstance(result, tuple) and len(result) > 0:
            # 多返回值（如 (df_llm, df_skip)），取第一个 DataFrame 的行数
            first = result[0]
            if isinstance(first, pd.DataFrame):
                return len(first)
            if isinstance(first, list):
                return len(first)
        return 0

    def timed(self, name: str, func, *args):
        """执行一个管线步骤并记录耗时。"""
        t0 = time.perf_counter()
        result = func(*args)
        elapsed = time.perf_counter() - t0
        rows = self._count_rows(result)
        self.timings.append({
            "step": name,
            "elapsed_sec": round(elapsed, 3),
            "output_rows": rows,
        })
        logger.info("TIMER  %-25s  %.2fs  (%d rows)", name, elapsed, rows)
        return result

    def add_step(self, name: str, elapsed_sec: float = 0.0, output_rows: int = 0):
        """手动添加一个步骤记录（用于不通过 timed() 执行的步骤）。"""
        self.timings.append({
            "step": name,
            "elapsed_sec": elapsed_sec,
            "output_rows": output_rows,
        })

    def print_summary(self):
        """输出管线性能汇总表。"""
        total = sum(s["elapsed_sec"] for s in self.timings)
        logger.info("=" * 65)
        logger.info("  %-25s  %8s  %6s  %s", "STEP", "TIME", "PCT", "ROWS")
        logger.info("-" * 65)
        for s in self.timings:
            pct = s["elapsed_sec"] / total * 100 if total > 0 else 0
            logger.info(
                "  %-25s  %7.2fs  %5.1f%%  %d",
                s["step"], s["elapsed_sec"], pct, s["output_rows"],
            )
        logger.info("-" * 65)
        logger.info("  %-25s  %7.2fs  %5s", "TOTAL", total, "100%")
        logger.info("=" * 65)
