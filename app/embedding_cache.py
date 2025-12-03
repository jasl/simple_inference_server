from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import cast

import numpy as np

from app.monitoring.metrics import record_cache_usage


class EmbeddingCache:
    """Thread-safe LRU cache for embedding vectors keyed by input text."""

    def __init__(self, max_size: int) -> None:
        self.max_size = max(0, max_size)
        self._lock = threading.Lock()
        self._store: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, text: str) -> np.ndarray | None:
        if self.max_size == 0:
            return None
        with self._lock:
            vec = self._store.get(text)
            if vec is None:
                return None
            # Mark as recently used
            self._store.move_to_end(text)
            return vec

    def set(self, text: str, vector: np.ndarray) -> None:
        if self.max_size == 0:
            return
        with self._lock:
            if text in self._store:
                self._store.move_to_end(text)
            self._store[text] = vector
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
