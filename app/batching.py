from __future__ import annotations

import asyncio
import contextlib
import functools
import os
import threading
import time
from typing import Any

import numpy as np

from app.models.registry import ModelRegistry
from app.monitoring.metrics import observe_embedding_batch_wait
from app.threadpool import get_embedding_executor


class EmbeddingBatchQueueTimeoutError(Exception):
    """Raised when waiting to enqueue into the embedding batch queue times out."""


class _BatchItem:
    def __init__(self, texts: list[str], future: asyncio.Future[np.ndarray], cancel_event: threading.Event | None) -> None:
        self.texts = texts
        self.future = future
        self.enqueue_time = asyncio.get_running_loop().time()
        self.cancel_event = cancel_event


class ModelBatcher:
    def __init__(self, model: Any, max_batch: int, window_ms: float, queue_size: int, queue_timeout: float) -> None:
        self.model = model
        self.max_batch = max_batch
        self.window = max(window_ms / 1000.0, 0.0)
        # Bounded queue prevents unbounded memory growth under bursty load.
        self.queue: asyncio.Queue[_BatchItem] = asyncio.Queue(max(queue_size, 1))
        self._task: asyncio.Task[None] | None = None
        self.queue_timeout = max(queue_timeout, 0.0)

    async def enqueue(self, texts: list[str], cancel_event: threading.Event | None = None) -> np.ndarray:
        loop = asyncio.get_running_loop()
        if self._task is None:
            # Lazily start worker on first request to bind to the running loop.
            self._task = loop.create_task(self._worker())

        fut: asyncio.Future[np.ndarray] = loop.create_future()
        try:
            await asyncio.wait_for(self.queue.put(_BatchItem(texts, fut, cancel_event)), timeout=self.queue_timeout)
        except TimeoutError as exc:
            raise EmbeddingBatchQueueTimeoutError("Embedding batch queue wait exceeded") from exc
        return await fut

    async def _worker(self) -> None:  # noqa: PLR0912
        loop = asyncio.get_running_loop()
        executor = get_embedding_executor()
        while True:
            item = await self.queue.get()
            batch_items = [item]
            total = len(item.texts)
            start = time.perf_counter()

            # Collect within window and max_batch
            while total < self.max_batch:
                remaining = self.window - (time.perf_counter() - start)
                if remaining <= 0:
                    break
                try:
                    nxt = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                except TimeoutError:
                    break
                batch_items.append(nxt)
                total += len(nxt.texts)

            # Prepare batch
            active_items: list[_BatchItem] = []
            texts: list[str] = []
            sizes: list[int] = []
            for bi in batch_items:
                # Drop canceled items before running expensive work.
                if bi.cancel_event is not None and bi.cancel_event.is_set():
                    if not bi.future.done():
                        bi.future.set_exception(asyncio.CancelledError())
                    continue
                active_items.append(bi)
                texts.extend(bi.texts)
                sizes.append(len(bi.texts))

            if not texts:
                continue

            # Merge cancel signals so a single event can be passed down to the model.
            cancel_events = [bi.cancel_event for bi in active_items if bi.cancel_event is not None]
            cancel_event = _merge_cancel_events(cancel_events)

            try:
                vectors = await loop.run_in_executor(
                    executor,
                    functools.partial(self.model.embed, texts, cancel_event=cancel_event),
                )
            except Exception as exc:  # pragma: no cover - defensive
                for bi in active_items:
                    if not bi.future.done():
                        bi.future.set_exception(exc)
                continue

            # Split outputs per request
            offset = 0
            for bi, size in zip(active_items, sizes, strict=True):
                with contextlib.suppress(Exception):
                    observe_embedding_batch_wait(getattr(self.model, "name", "unknown"), loop.time() - bi.enqueue_time)
                if not bi.future.done():
                    bi.future.set_result(vectors[offset : offset + size])
                offset += size

    async def stop(self) -> None:
        if self._task is not None:
            while not self.queue.empty():
                pending = self.queue.get_nowait()
                if not pending.future.done():
                    pending.future.set_exception(asyncio.CancelledError())
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task


class _AggregateCancel:
    """Aggregate cancel signal that triggers when any source event is set.

    This avoids spawning background threads per batch while still honoring
    cancellations from any request in the batch. The model's cancel check
    calls is_set() which polls all source events.
    """

    __slots__ = ("_events",)

    def __init__(self, events: list[threading.Event]) -> None:
        self._events = events

    def is_set(self) -> bool:
        return any(e.is_set() for e in self._events)

    def set(self) -> None:
        for e in self._events:
            e.set()

    def wait(self, timeout: float | None = None) -> bool:
        # Simple polling fallback; not used in hot path but keeps interface compatible.
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self.is_set():
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.01)


def _merge_cancel_events(events: list[threading.Event]) -> _AggregateCancel | threading.Event | None:
    """Return an aggregate cancel signal for the batch.

    When multiple requests are batched together, we need to detect if *any* of
    them has been cancelled. This returns an _AggregateCancel wrapper that
    polls all source events on is_set().
    """
    if not events:
        return None
    if len(events) == 1:
        return events[0]
    return _AggregateCancel(events)


class BatchingService:
    def __init__(  # noqa: PLR0913 - config-rich initializer
        self,
        registry: ModelRegistry,
        enabled: bool = True,
        max_batch_size: int | None = None,
        window_ms: float | None = None,
        queue_size: int | None = None,
        queue_timeout_sec: float | None = None,
    ) -> None:
        self.enabled = enabled
        self.max_batch_size = max_batch_size or int(
            os.getenv("EMBEDDING_BATCH_WINDOW_MAX_SIZE", os.getenv("MAX_BATCH_SIZE", "32"))
        )
        self.window_ms = window_ms if window_ms is not None else float(
            os.getenv("EMBEDDING_BATCH_WINDOW_MS", "6")
        )
        self.queue_size = queue_size if queue_size is not None else int(
            os.getenv("EMBEDDING_BATCH_QUEUE_SIZE", os.getenv("MAX_QUEUE_SIZE", "64"))
        )
        self.queue_timeout_sec = queue_timeout_sec if queue_timeout_sec is not None else float(
            os.getenv("EMBEDDING_BATCH_QUEUE_TIMEOUT_SEC", os.getenv("QUEUE_TIMEOUT_SEC", "2.0"))
        )
        self._batchers: dict[str, ModelBatcher] = {}
        for name in registry.list_models():
            model = registry.get(name)
            if "text-embedding" in getattr(model, "capabilities", []):
                self._batchers[name] = ModelBatcher(
                    model,
                    self.max_batch_size,
                    self.window_ms,
                    self.queue_size,
                    self.queue_timeout_sec,
                )

    async def enqueue(self, model_name: str, texts: list[str], cancel_event: threading.Event | None = None) -> np.ndarray:
        if not self.enabled:
            raise RuntimeError("Batching disabled")
        if model_name not in self._batchers:
            raise KeyError(f"Model {model_name} not registered")
        return await self._batchers[model_name].enqueue(texts, cancel_event=cancel_event)

    async def stop(self) -> None:
        for batcher in self._batchers.values():
            await batcher.stop()

    def queue_stats(self) -> dict[str, tuple[int, int | None]]:
        """Return current queue sizes for each embedding batcher."""

        stats: dict[str, tuple[int, int | None]] = {}
        for name, batcher in self._batchers.items():
            size = batcher.queue.qsize()
            # asyncio.Queue default maxsize of 0 means unbounded; expose None instead.
            max_size = batcher.queue.maxsize or None
            stats[name] = (size, max_size)
        return stats
