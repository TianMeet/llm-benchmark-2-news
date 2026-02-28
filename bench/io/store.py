"""运行产物持久化存储层。

负责每次评测 run 的全部文件产物管理：
- config.json             : 本次运行配置快照
- run_meta.json           : 运行环境元信息（git/python/platform）
- dataset_fingerprint.json: 数据集 SHA256 指纹
- model_snapshot.json     : 模型配置快照（已脱敏）
- results.jsonl           : 逐条追加的样本级结果
- summary.csv             : 聚合指标
- by_model/<task>/<model>.jsonl : 按任务/模型分区的结果

线程安全：append_result 使用 threading.Lock 保护，
防止 asyncio + executor 场景下行内容交错写入。

断点续跑：existing_run_dir 参数支持复用既有 run 目录，
load_completed_keys() 加载已完成记录键。
"""

from __future__ import annotations

__all__ = ["RunStore"]

import csv
import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RunStore:
    """基于文件的运行存储，满足 MVP 可复现与可迁移要求。

    线程安全设计：append_result 使用 threading.Lock 保护，
    防止多线程（或 asyncio + executor）场景下行内容交错写入。
    """

    def __init__(
        self,
        out_dir: str | Path,
        *,
        existing_run_dir: str | Path | None = None,
    ):
        self.out_dir = Path(out_dir)
        if existing_run_dir is not None:
            self.run_dir = Path(existing_run_dir)
            self.run_id = self.run_dir.name
            self.run_dir.mkdir(parents=True, exist_ok=True)
        else:
            # run_id 使用 UTC 时间戳，便于排序和追踪。
            self.run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
            self.run_dir = self.out_dir / self.run_id
            self.run_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.run_dir / "results.jsonl"
        self._append_lock = threading.Lock()
        # 持有文件 writer，避免每次 append 都 open/close
        self._results_writer: Any | None = None

    def write_config(self, payload: dict[str, Any]) -> None:
        """写入本次评测配置快照。"""
        (self.run_dir / "config.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def write_run_meta(self, payload: dict[str, Any]) -> None:
        """写入运行元数据（环境、版本、审计字段）。"""
        (self.run_dir / "run_meta.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def write_dataset_fingerprint(self, payload: dict[str, Any]) -> None:
        """写入数据集指纹。"""
        (self.run_dir / "dataset_fingerprint.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def write_model_snapshot(self, payload: dict[str, Any]) -> None:
        """写入本次运行实际使用模型的配置快照（可脱敏）。"""
        (self.run_dir / "model_snapshot.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def append_result(self, row: dict[str, Any]) -> None:
        """逐条追加样本结果（加锁保护，保证多线程下行原子性）。

        内部持有打开的文件句柄，避免每次 append 都 open/close 带来的
        系统调用开销。每次写入后立即 flush 保证数据可见性。
        """
        with self._append_lock:
            if self._results_writer is None:
                self._results_writer = self.results_path.open("a", encoding="utf-8")
            self._results_writer.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._results_writer.flush()

    def close(self) -> None:
        """关闭持有的文件句柄。

        调用方应在 run 结束后调用，或使用 ``with`` 上下文管理器。
        """
        with self._append_lock:
            if self._results_writer is not None:
                self._results_writer.close()
                self._results_writer = None

    def __enter__(self) -> "RunStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def write_partitioned_results(self, records: list[dict[str, Any]]) -> None:
        """按 task_name / model_id 分目录写出结果。

        产物目录结构：
            runs/<id>/by_model/<task_name>/<model_id>.jsonl
        不影响原有的平铺 results.jsonl。
        """
        from collections import defaultdict
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in records:
            key = (str(row.get("task_name", "_")), str(row.get("model_id", "_")))
            groups[key].append(row)

        base = self.run_dir / "by_model"
        for (task_name, model_id), rows in groups.items():
            # model_id 可能包含 "/" 或 ":"，用 "-" 替换保证文件名安全
            safe_model = model_id.replace("/", "-").replace(":", "-")
            out_dir = base / task_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{safe_model}.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def write_summary(self, rows: list[dict[str, Any]]) -> None:
        """将聚合指标写入 summary.csv。

        使用所有行的 key 并集作为 fieldnames，避免 workflow 模式下
        各 step task_metrics 字段不一致导致 DictWriter 报错。
        """
        if not rows:
            return
        summary_path = self.run_dir / "summary.csv"
        # 收集所有行的字段联合，保持首行字段顺序以便阅读
        seen: dict[str, None] = {}
        for row in rows:
            for k in row:
                seen[k] = None
        fieldnames = list(seen.keys())
        with summary_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
            writer.writeheader()
            writer.writerows(rows)

    def load_result_rows(self) -> list[dict[str, Any]]:
        """读取既有结果行；用于 resume。"""
        if not self.results_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.results_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def load_completed_keys(self) -> set[tuple[str, str, str, str, int]]:
        """加载已完成记录键，键格式为(sample_id, task_name, step_id, model_id, repeat_idx)。"""
        done: set[tuple[str, str, str, str, int]] = set()
        for row in self.load_result_rows():
            try:
                repeat_idx = int(row.get("repeat_idx", 0))
            except (ValueError, TypeError):
                logger.warning(
                    "invalid repeat_idx %r in row sample_id=%s, defaulting to 0",
                    row.get("repeat_idx"), row.get("sample_id", "?"),
                )
                repeat_idx = 0
            done.add(
                (
                    str(row.get("sample_id", "")),
                    str(row.get("task_name", "")),
                    str(row.get("step_id", "")),
                    str(row.get("model_id", "")),
                    repeat_idx,
                )
            )
        return done
