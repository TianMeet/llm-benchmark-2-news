#!/usr/bin/env python3
"""
爬取选股宝板块数据并保存到 SQLite。

Usage:
    python scripts/fetch_xuangubao_data.py [--db-path data/xuangubao.db] [--interval 3.0]
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from other.xuangubao import XuangubaoClient, ALL_PLATE_IDS


def setup_logging():
    """配置日志输出"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    parser = argparse.ArgumentParser(
        description="爬取选股宝板块数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python scripts/fetch_xuangubao_data.py
    python scripts/fetch_xuangubao_data.py --db-path data/xuangubao.db --interval 2.0
    python scripts/fetch_xuangubao_data.py --max-plates 100
        """,
    )
    parser.add_argument(
        "--db-path",
        default="data/xuangubao.db",
        help="SQLite 数据库保存路径 (默认: data/xuangubao.db)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=3.0,
        help="请求间隔秒数，实际会随机波动为 0.8~2.5 倍 (默认: 3.0)",
    )
    parser.add_argument(
        "--max-plates",
        type=int,
        default=None,
        help="最大爬取板块数量，默认爬取全部",
    )

    args = parser.parse_args()
    setup_logging()

    # 确保目录存在
    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 选择板块 ID 列表
    plate_ids = ALL_PLATE_IDS
    if args.max_plates:
        plate_ids = plate_ids[: args.max_plates]
        logging.info(f"将爬取前 {args.max_plates} 个板块（共 {len(ALL_PLATE_IDS)} 个）")
    else:
        logging.info(f"将爬取全部 {len(plate_ids)} 个板块")

    # 创建客户端并开始爬取
    client = XuangubaoClient(db_path=str(db_path))
    logging.info(f"数据库路径: {db_path.absolute()}")
    logging.info(f"请求间隔: {args.interval}s (含随机波动)")
    print("-" * 50)

    try:
        stats = client.fetch_all_plates(
            plate_ids=plate_ids,
            interval=args.interval,
        )

        print("-" * 50)
        logging.info("爬取完成!")
        logging.info(f"成功: {stats['success']} 个")
        logging.info(f"失败: {stats['fail']} 个")

        if stats["fail_ids"]:
            logging.warning(f"失败的板块 ID: {stats['fail_ids']}")

        # 显示数据库统计
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        tables = ["plates", "stocks", "industrial_chains", "question_answers"]
        print("\n数据库统计:")
        for table in tables:
            count = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table}: {count} 条记录")

        conn.close()

    except KeyboardInterrupt:
        print("\n")
        logging.warning("用户中断，已保存的数据可在下次运行时断点续爬")
        sys.exit(1)
    except Exception as e:
        logging.error(f"爬取过程出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
