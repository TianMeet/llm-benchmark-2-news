from eval.cache import EvalCache


def test_cache_roundtrip(tmp_path) -> None:
    cache = EvalCache(tmp_path / "cache")
    payload = {"model_id": "m1", "sample_id": "s1", "messages": [{"role": "user", "content": "x"}]}
    key = cache.make_key(payload)

    assert cache.get(key) is None
    cache.set(key, {"content": "ok"})
    assert cache.get(key) == {"content": "ok"}
