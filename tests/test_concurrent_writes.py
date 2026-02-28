"""并发写入竞态回归测试 — 保护 EvalCache 和 RunStore 的线程安全实现。"""

from __future__ import annotations

import json
import threading

import pytest

from eval.io.cache import EvalCache
from eval.io.store import RunStore


# ─────────────────────────────────────────────────────────────────────────────
# EvalCache 并发写入
# ─────────────────────────────────────────────────────────────────────────────

class TestEvalCacheConcurrentWrite:
    """多线程并发写入 EvalCache，确保文件不截断且读取正确。"""

    def test_concurrent_set_distinct_keys(self, tmp_path) -> None:
        """N 个线程同时写入不同 key，全部应可正确读回。"""
        cache = EvalCache(tmp_path)
        n = 50
        errors: list[Exception] = []

        def write(i: int) -> None:
            try:
                key = cache.make_key({"thread": i})
                cache.set(key, {"value": i, "data": "x" * 200})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"写入过程中出现异常: {errors}"

        # 验证所有 key 可正确读回
        for i in range(n):
            key = cache.make_key({"thread": i})
            result = cache.get(key)
            assert result is not None, f"thread={i} 的缓存未找到"
            assert result["value"] == i

    def test_concurrent_set_same_key(self, tmp_path) -> None:
        """多线程对同一 key 并发写入，最终应能正确 JSON 读取（无截断）。"""
        cache = EvalCache(tmp_path)
        key = cache.make_key({"shared": True})
        n = 30
        errors: list[Exception] = []

        def write(i: int) -> None:
            try:
                cache.set(key, {"value": i, "payload": "y" * 300})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"并发写入同一 key 出现异常: {errors}"
        result = cache.get(key)
        # 最终值必须可读，且为某一次写入的完整结果
        assert result is not None, "同一 key 并发写入后无法读取"
        assert "value" in result


# ─────────────────────────────────────────────────────────────────────────────
# RunStore 并发写入
# ─────────────────────────────────────────────────────────────────────────────

class TestRunStoreConcurrentAppend:
    """多线程并发调用 append_result，确保行完整且顺序不交错。"""

    def test_concurrent_append_result(self, tmp_path) -> None:
        """N 个线程同时追加结果行，最终 results.jsonl 每行都应为合法 JSON。"""
        store = RunStore(tmp_path)
        n = 100
        errors: list[Exception] = []

        def append(i: int) -> None:
            try:
                store.append_result({"index": i, "data": "z" * 150})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=append, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"并发写入 RunStore 出现异常: {errors}"

        lines = store.results_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == n, f"期望 {n} 行，实际 {len(lines)} 行"

        parsed_indices = set()
        for lineno, line in enumerate(lines, 1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                pytest.fail(f"第 {lineno} 行 JSON 解析失败（行内容截断？）: {e}\n内容: {line!r}")
            parsed_indices.add(obj["index"])

        assert parsed_indices == set(range(n)), "部分写入结果丢失"

    def test_concurrent_append_does_not_interleave_lines(self, tmp_path) -> None:
        """大数据量追加时，每行 newline 分隔应保持完整，不出现行内交错。"""
        store = RunStore(tmp_path)
        n = 200
        payload = {"key": "value", "long_field": "A" * 500}

        threads = [
            threading.Thread(target=store.append_result, args=(dict(payload, index=i),))
            for i in range(n)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        raw = store.results_path.read_text(encoding="utf-8")
        lines = [l for l in raw.splitlines() if l.strip()]
        assert len(lines) == n

        for line in lines:
            obj = json.loads(line)  # 不应抛出 JSONDecodeError
            assert "index" in obj
