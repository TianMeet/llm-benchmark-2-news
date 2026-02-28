"""将 data/test_news_dataset.parquet 转换为系统标准 JSONL 格式。

字段映射：
  news_id             → sample_id
  ticker              → ticker
  sec_short_name      → sec_short_name
  exchange_cd         → exchange_cd
  trade_date          → trade_date
  news_publish_time   → news_publish_time
  news_title          → news_title
  news_body           → news_body          （模型输入）
  news_summary        → gt_summary         （参考摘要，作为 ground truth）
  sentiment           → gt_sentiment_int   （原始整数: -1/0/1）
                      → gt_sentiment       （文本: negative/neutral/positive）
  sentiment_score     → gt_sentiment_score （置信度，可供 numeric_error 分析）

用法：
    python scripts/convert_parquet_to_jsonl.py \\
        --input data/test_news_dataset.parquet \\
        --output datasets/news_summary_eval.jsonl \\
        [--max-rows 100]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SENTIMENT_MAP = {-1: "negative", 0: "neutral", 1: "positive"}


def convert(input_path: str, output_path: str, max_rows: int | None = None) -> int:
    """转换 parquet → JSONL，返回写入行数。"""
    try:
        import duckdb
    except ImportError as e:
        raise ImportError("请先安装 duckdb: pip install duckdb") from e

    conn = duckdb.connect()
    limit_clause = f"LIMIT {max_rows}" if max_rows else ""
    sql = f"""
        SELECT
            news_id,
            trade_date,
            ticker,
            sec_short_name,
            exchange_cd,
            news_publish_time,
            news_title,
            news_body,
            news_summary,
            sentiment,
            sentiment_score
        FROM read_parquet('{input_path}')
        WHERE news_body IS NOT NULL AND news_body != ''
          AND news_summary IS NOT NULL AND news_summary != ''
        ORDER BY news_id
        {limit_clause}
    """
    rows = conn.execute(sql).fetchall()
    cols = ["news_id", "trade_date", "ticker", "sec_short_name", "exchange_cd",
            "news_publish_time", "news_title", "news_body", "news_summary",
            "sentiment", "sentiment_score"]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            d = dict(zip(cols, row))
            sentiment_int = int(d["sentiment"]) if d["sentiment"] is not None else 0
            record = {
                "sample_id":          str(d["news_id"]),
                "ticker":             d["ticker"] or "",
                "sec_short_name":     d["sec_short_name"] or "",
                "exchange_cd":        d["exchange_cd"] or "",
                "trade_date":         str(d["trade_date"]) if d["trade_date"] else "",
                "news_publish_time":  str(d["news_publish_time"]) if d["news_publish_time"] else "",
                "news_title":         d["news_title"] or "",
                "news_body":          d["news_body"] or "",
                # ---- ground truth ----
                "gt_summary":         d["news_summary"] or "",
                "gt_sentiment":       SENTIMENT_MAP.get(sentiment_int, "neutral"),
                "gt_sentiment_score": round(float(d["sentiment_score"]), 6) if d["sentiment_score"] is not None else 0.0,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"✓ 写入 {written} 条样本 → {out}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Parquet → JSONL 数据集转换工具")
    parser.add_argument("--input", default="data/test_news_dataset.parquet")
    parser.add_argument("--output", default="datasets/news_summary_eval.jsonl")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    convert(args.input, args.output, args.max_rows)


if __name__ == "__main__":
    main()
