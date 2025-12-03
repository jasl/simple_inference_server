import numpy as np

from app.embedding_cache import EmbeddingCache, embed_with_cache

EXPECTED_EVICTION_CALLS = 4


def _toy_encoder(texts: list[str], calls: dict[str, int]) -> np.ndarray:
    calls["count"] += 1
    # Encode each string as a single float (its length) for easy assertions.
    return np.array([[float(len(t))] for t in texts])


def test_cached_embeddings_reuse_previous_results() -> None:
    cache = EmbeddingCache(max_size=4)
    calls: dict[str, int] = {"count": 0}

    first = embed_with_cache(["a", "bb"], lambda t: _toy_encoder(t, calls), cache, "dummy")
    second = embed_with_cache(["bb", "a"], lambda t: _toy_encoder(t, calls), cache, "dummy")

    assert calls["count"] == 1  # second call served entirely from cache
    np.testing.assert_array_equal(first[::-1], second)


def test_cache_eviction_respects_lru_order() -> None:
    cache = EmbeddingCache(max_size=2)
    calls: dict[str, int] = {"count": 0}

    embed_with_cache(["first"], lambda t: _toy_encoder(t, calls), cache, "dummy")
    embed_with_cache(["second"], lambda t: _toy_encoder(t, calls), cache, "dummy")
    embed_with_cache(["third"], lambda t: _toy_encoder(t, calls), cache, "dummy")

    # "first" should be evicted after "third" enters
    embed_with_cache(["first"], lambda t: _toy_encoder(t, calls), cache, "dummy")

    assert calls["count"] == EXPECTED_EVICTION_CALLS  # required re-computing "first" after eviction


def test_batch_deduplicates_missing_texts() -> None:
    cache = EmbeddingCache(max_size=2)
    calls: dict[str, int] = {"count": 0}

    embed_with_cache(["repeat", "repeat"], lambda t: _toy_encoder(t, calls), cache, "dummy")

    # Only one compute invocation despite two identical texts
    assert calls["count"] == 1
