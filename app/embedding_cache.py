from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import cast

import numpy as np
import xxhash

from app.monitoring.metrics import record_cache_usage


def _hash_key(text: str) -> str:
    """Compute a fast hash of the text for cache keying.

    Using xxhash.xxh64 provides:
    - Extremely fast hashing (faster than MD5/SHA by 10-50x)
    - Fixed-size keys (16 hex chars) reducing memory for long inputs
    - Very low collision probability for typical text lengths
    """
    return xxhash.xxh64(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Thread-safe LRU cache for embedding vectors keyed by input text.

    Internally uses xxhash for fast, fixed-size cache keys.
    """

    def __init__(self, max_size: int) -> None:
        self.max_size = max(0, max_size)
        self._lock = threading.Lock()
        self._store: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, text: str) -> np.ndarray | None:
        if self.max_size == 0:
            return None
        key = _hash_key(text)
        with self._lock:
            vec = self._store.get(key)
            if vec is None:
                return None
            # Mark as recently used
            self._store.move_to_end(key)
            return vec

    def set(self, text: str, vector: np.ndarray) -> None:
        if self.max_size == 0:
            return
        key = _hash_key(text)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = vector
            if len(self._store) > self.max_size:
                # Evict least-recently-used item
                self._store.popitem(last=False)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._store)


def embed_with_cache(
    texts: list[str],
    compute: Callable[[list[str]], np.ndarray],
    cache: EmbeddingCache | None,
    model_name: str,
) -> np.ndarray:
    """
    Compute embeddings with an optional LRU cache.

    Args:
        texts: Input strings to embed.
        compute: Function that embeds a list of texts (without caching).
        cache: LRU cache instance; if None or size=0, caching is bypassed.
        model_name: Used for cache metrics labeling.
    """

    if not texts:
        return np.empty((0, 0))

    if cache is None or cache.max_size == 0:
        vectors = compute(texts)
        record_cache_usage(model_name, hits=0, misses=len(texts))
        return vectors

    cached_vectors: list[np.ndarray | None] = [None] * len(texts)
    missing_indices: dict[str, list[int]] = {}
    hits = 0
    misses = 0

    for idx, text in enumerate(texts):
        vec = cache.get(text)
        if vec is not None:
            cached_vectors[idx] = vec
            hits += 1
            continue

        misses += 1
        missing_indices.setdefault(text, []).append(idx)

    if missing_indices:
        missing_texts = list(missing_indices.keys())
        computed = compute(missing_texts)

        for text, vec in zip(missing_texts, computed, strict=False):
            cache.set(text, vec)
            for idx in missing_indices[text]:
                cached_vectors[idx] = vec

    record_cache_usage(model_name, hits=hits, misses=misses)

    # All slots should now be populated; stack into a single ndarray.
    if any(vec is None for vec in cached_vectors):  # pragma: no cover - defensive
        raise RuntimeError("Embedding cache missing vector; this is a bug")

    return np.stack(cast(list[np.ndarray], cached_vectors))
