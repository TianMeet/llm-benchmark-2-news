"""从 Xuangubao SQLite 数据库生成「概念板块→股票联想」基准测试数据集。

用途：测试 LLM 对 A 股概念板块的核心关联股票召回能力。
输入：data/xuangubao.db
输出：datasets/stock_association_eval.jsonl

筛选规则：
  - 仅保留核心股票数（core_flag=100）在 [MIN_CORE, MAX_CORE] 范围内的板块
  - 去除无 desc 或 desc 过短的板块

每条样本结构：
  {
    "sample_id": "SA_<plate_id>",
    "plate_name": "稀土磁材",
    "plate_desc": "...",
    "chains": ["上游原矿", "中游冶炼", "下游应用"],
    "core_stocks_count": 26,
    "gt_stocks": ["000758", "000795", ...]
  }
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── 配置 ─────────────────────────────────────────────────────────────────────
DB_PATH = Path("data/xuangubao.db")
OUTPUT_PATH = Path("datasets/stock_association_eval.jsonl")
MIN_CORE = 5    # 板块核心股最少数目
MAX_CORE = 50   # 板块核心股最多数目
MIN_DESC_LEN = 10  # 板块描述最小字符数


def strip_html(html: str) -> str:
    """去除 HTML 标签，保留纯文本。"""
    text = re.sub(r"<[^>]+>", "", html)
    text = re.sub(r"&[a-zA-Z]+;", "", text)
    return text.strip()


def load_plates(conn: sqlite3.Connection) -> list[dict]:
    """加载所有板块及其核心股票数。"""
    conn.row_factory = sqlite3.Row
    cur = conn.execute("""
        SELECT p.id, p.name, p.desc,
               SUM(CASE WHEN s.core_flag = 100 THEN 1 ELSE 0 END) AS core_count,
               COUNT(s.symbol) AS total_count
        FROM plates p
        LEFT JOIN stocks s ON p.id = s.plate_id
        GROUP BY p.id
        HAVING core_count BETWEEN ? AND ?
        ORDER BY p.name
    """, (MIN_CORE, MAX_CORE))
    return [dict(row) for row in cur.fetchall()]


def load_core_stocks(conn: sqlite3.Connection, plate_id: int) -> list[str]:
    """加载板块的核心股票代码列表。"""
    cur = conn.execute(
        "SELECT symbol FROM stocks WHERE plate_id = ? AND core_flag = 100 ORDER BY symbol",
        (plate_id,),
    )
    return [row[0] for row in cur.fetchall()]


def load_chains(conn: sqlite3.Connection, plate_id: int) -> list[str]:
    """加载板块的产业链名称（按 order 排列）。"""
    cur = conn.execute(
        "SELECT name FROM industrial_chains WHERE plate_id = ? ORDER BY \"order\"",
        (plate_id,),
    )
    return [row[0] for row in cur.fetchall()]


def load_qa_intro(conn: sqlite3.Connection, plate_id: int) -> str | None:
    """从 question_answers 中提取板块介绍文本（纯文本，去 HTML）。"""
    cur = conn.execute(
        "SELECT html_answer FROM question_answers WHERE plate_id = ? AND question = '板块介绍' LIMIT 1",
        (plate_id,),
    )
    row = cur.fetchone()
    if row and row[0]:
        text = strip_html(row[0])
        # 截断过长的介绍（保留前 500 字）
        return text[:500] if len(text) > 500 else text
    return None


def build_dataset() -> None:
    """主流程：读取 DB → 筛选 → 组装 → 写入 JSONL。"""
    if not DB_PATH.exists():
        logger.error("Database not found: %s", DB_PATH)
        return

    conn = sqlite3.connect(str(DB_PATH))
    plates = load_plates(conn)
    logger.info("Plates matching criteria (%d-%d core stocks): %d", MIN_CORE, MAX_CORE, len(plates))

    samples = []
    skipped = 0

    for plate in plates:
        plate_id = plate["id"]
        name = plate["name"]
        desc = (plate["desc"] or "").strip()

        # 跳过描述过短的板块
        if len(desc) < MIN_DESC_LEN:
            logger.debug("Skipping '%s': desc too short (%d chars)", name, len(desc))
            skipped += 1
            continue

        core_stocks = load_core_stocks(conn, plate_id)
        chains = load_chains(conn, plate_id)
        qa_intro = load_qa_intro(conn, plate_id)

        sample = {
            "sample_id": f"SA_{plate_id}",
            "plate_name": name,
            "plate_desc": desc,
            "chains": chains,
            "core_stocks_count": len(core_stocks),
            "gt_stocks": core_stocks,
        }

        # 如果有 QA 板块介绍，作为额外上下文
        if qa_intro:
            sample["plate_intro"] = qa_intro

        samples.append(sample)

    conn.close()

    # 写入 JSONL
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info("Generated %d samples (skipped %d with short desc)", len(samples), skipped)
    logger.info("Output: %s", OUTPUT_PATH)

    # 统计摘要
    if samples:
        counts = [s["core_stocks_count"] for s in samples]
        chain_counts = [len(s["chains"]) for s in samples]
        intro_counts = sum(1 for s in samples if "plate_intro" in s)
        logger.info(
            "Core stocks per sample: min=%d, max=%d, avg=%.1f",
            min(counts), max(counts), sum(counts) / len(counts),
        )
        logger.info("Samples with chains: %d, with plate_intro: %d", sum(1 for c in chain_counts if c > 0), intro_counts)


if __name__ == "__main__":
    build_dataset()
