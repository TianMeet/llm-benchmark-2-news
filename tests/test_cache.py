from bench.io.cache import EvalCache


def test_cache_roundtrip(tmp_path) -> None:
    cache = EvalCache(tmp_path / "cache")
    payload = {"model_id": "m1", "sample_id": "s1", "messages": [{"role": "user", "content": "x"}]}
    key = cache.make_key(payload)

    assert cache.get(key) is None
    cache.set(key, {"content": "ok"})
    assert cache.get(key) == {"content": "ok"}


def test_cache_concurrent_write_threadsafe(tmp_path) -> None:
    """P0: 多线程并发写同一 key 不应导致文件损坏（JSON 不可解析）。"""
    import json
    import threading

    cache = EvalCache(tmp_path / "cache")
    key = cache.make_key({"id": "concurrent"})
    errors: list[Exception] = []

    def _write(value: int) -> None:
        try:
            # 每个线程写自己的数据，完成后读回应仍能 JSON 解析
            cache.set(key, {"v": value, "padding": "x" * 256})
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_write, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"写入时出现异常：{errors}"
    # 最终文件应仍可解析（不应被截断/交错写入）
    result = cache.get(key)
    assert result is not None, "并发写入后缓存文件不可读"
    assert "v" in result, "写入数据结构损坏"


def test_cache_concurrent_read_write_threadsafe(tmp_path) -> None:
    """P0: 读写并发时不应抛出异常也不应返回损坏数据。"""
    import threading

    cache = EvalCache(tmp_path / "cache")
    key = cache.make_key({"id": "rw_concurrent"})
    cache.set(key, {"status": "initial"})

    errors: list[Exception] = []

    def _reader() -> None:
        for _ in range(30):
            try:
                val = cache.get(key)
                # 要么读到 None（写入过程中），要么是合法 dict
                assert val is None or isinstance(val, dict)
            except Exception as exc:
                errors.append(exc)

    def _writer() -> None:
        for i in range(30):
            try:
                cache.set(key, {"status": f"update_{i}", "padding": "y" * 128})
            except Exception as exc:
                errors.append(exc)

    threads = (
        [threading.Thread(target=_reader) for _ in range(5)]
        + [threading.Thread(target=_writer) for _ in range(5)]
    )
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"读写并发时出现异常：{errors}"


def test_cache_get_treats_corrupted_json_as_miss(tmp_path) -> None:
    cache = EvalCache(tmp_path / "cache")
    key = cache.make_key({"id": "broken"})
    broken_path = (tmp_path / "cache") / f"{key}.json"
    broken_path.write_text("{not-json", encoding="utf-8")

    assert cache.get(key) is None
