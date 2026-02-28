import json

from eval.store import RunStore


def test_run_store_writes_files(tmp_path) -> None:
    store = RunStore(tmp_path)
    store.write_config({"run_id": store.run_id})
    store.append_result({"sample_id": "s1", "success": True})
    store.write_summary([{"task_name": "t", "model_id": "m", "step_id": "x"}])

    assert (store.run_dir / "config.json").exists()
    assert (store.run_dir / "results.jsonl").exists()
    assert (store.run_dir / "summary.csv").exists()

    config = json.loads((store.run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["run_id"] == store.run_id


def test_store_concurrent_append_no_row_loss(tmp_path) -> None:
    """P0: 多线程并发写入时不应有行丢失（行总数必须等于写入次数）。"""
    import threading

    store = RunStore(tmp_path)
    n_threads = 20
    rows_per_thread = 50
    errors: list[Exception] = []

    def _append_rows(thread_id: int) -> None:
        for i in range(rows_per_thread):
            try:
                store.append_result({"thread": thread_id, "idx": i, "sample_id": f"t{thread_id}_{i}"})
            except Exception as exc:
                errors.append(exc)

    threads = [threading.Thread(target=_append_rows, args=(tid,)) for tid in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"并发写入时出现异常：{errors}"

    lines = [
        line for line in store.results_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    expected = n_threads * rows_per_thread
    assert len(lines) == expected, (
        f"期望 {expected} 行，实际 {len(lines)} 行 — 并发写入存在行丢失或行交错"
    )
    # 每行都应是合法 JSON
    for line in lines:
        row = json.loads(line)
        assert "sample_id" in row


def test_store_concurrent_append_no_interleaved_json(tmp_path) -> None:
    """P0: 并发写入时每行必须是独立完整的 JSON，不应出现行内容交错。"""
    import threading

    store = RunStore(tmp_path)
    big_payload = "data_" + "x" * 512

    def _write(_: int) -> None:
        for i in range(20):
            store.append_result({"sample_id": f"s{i}", "big": big_payload})

    threads = [threading.Thread(target=_write, args=(tid,)) for tid in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for line in store.results_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)          # 若行交错则 JSON 解析失败
            assert row["big"] == big_payload
