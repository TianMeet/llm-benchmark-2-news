#!/usr/bin/env python3
"""
监控选股宝数据爬取进度

Usage:
    python scripts/monitor_xuangubao.py [--interval 10] [--output-file PATH]
"""
import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_progress(output_text: str) -> dict:
    """解析输出文本中的进度信息"""
    lines = output_text.strip().split('\n')

    # 查找最后一条进度记录
    current = 0
    total = 589
    latest_plate = ""
    stock_count = 0

    for line in reversed(lines):
        # 匹配进度行: [123/589] 板块名(id=12345) 成分股=67
        match = re.search(r'\[(\d+)/(\d+)\]\s+(.+?)\(id=(\d+)\)\s+成分股=(\d+)', line)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            latest_plate = match.group(3)
            stock_count = int(match.group(5))
            break

    # 计算统计信息
    success_count = len([l for l in lines if '成分股=' in l and '失败' not in l])
    fail_count = len([l for l in lines if '失败' in l])

    return {
        'current': current,
        'total': total,
        'latest_plate': latest_plate,
        'stock_count': stock_count,
        'success': success_count,
        'fail': fail_count,
    }


def format_eta(elapsed_seconds: float, current: int, total: int) -> str:
    """计算预计剩余时间"""
    if current == 0:
        return "计算中..."

    avg_time_per_item = elapsed_seconds / current
    remaining_items = total - current
    eta_seconds = avg_time_per_item * remaining_items

    eta_min = int(eta_seconds // 60)
    eta_sec = int(eta_seconds % 60)

    if eta_min > 60:
        eta_hour = eta_min // 60
        eta_min = eta_min % 60
        return f"{eta_hour}h {eta_min}m"
    return f"{eta_min}m {eta_sec}s"


def monitor(output_file: str, interval: int):
    """监控进度"""
    output_path = Path(output_file)
    start_time = time.time()

    print("=" * 60)
    print("选股宝数据爬取监控")
    print("=" * 60)
    print(f"输出文件: {output_file}")
    print(f"刷新间隔: {interval}秒")
    print("=" * 60)
    print()

    try:
        while True:
            # 读取输出文件
            if output_path.exists():
                content = output_path.read_text(encoding='utf-8')
            else:
                content = ""

            # 解析进度
            progress = parse_progress(content)

            # 计算统计
            elapsed = time.time() - start_time
            elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"

            if progress['current'] > 0:
                percent = (progress['current'] / progress['total']) * 100
                eta = format_eta(elapsed, progress['current'], progress['total'])
                bar_len = 30
                filled = int(bar_len * progress['current'] / progress['total'])
                bar = '█' * filled + '░' * (bar_len - filled)

                # 清空屏幕（ANSI escape code）
                print('\033[2J\033[H', end='')

                print("=" * 60)
                print("选股宝数据爬取监控")
                print("=" * 60)
                print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"运行时间: {elapsed_str}")
                print()
                print(f"进度: [{bar}] {percent:.1f}%")
                print(f"      {progress['current']} / {progress['total']} 板块")
                print()
                print(f"当前板块: {progress['latest_plate']}")
                print(f"成分股数: {progress['stock_count']} 只")
                print()
                print(f"成功: {progress['success']} | 失败: {progress['fail']}")
                print(f"预计剩余: {eta}")
                print("=" * 60)
                print()
                print("按 Ctrl+C 停止监控")

                # 检查是否完成
                if progress['current'] >= progress['total']:
                    print()
                    print("🎉 爬取完成！")
                    break
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 等待任务开始...")

            time.sleep(interval)

    except KeyboardInterrupt:
        print()
        print()
        print("监控已停止")


def main():
    parser = argparse.ArgumentParser(
        description="监控选股宝数据爬取进度",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="刷新间隔秒数 (默认: 10)",
    )
    parser.add_argument(
        "--output-file",
        default="/private/tmp/claude-501/-Users-xingkong-Desktop-llm-benchmark-2-news/tasks/blvoawn34.output",
        help="爬取任务的输出文件路径",
    )

    args = parser.parse_args()
    monitor(args.output_file, args.interval)


if __name__ == "__main__":
    main()
