from __future__ import annotations

import asyncio
import contextlib
import os
import time
from typing import Any

import numpy as np

from app.models.registry import ModelRegistry
from app.threadpool import get_embedding_executor


class _BatchItem:
    def __init__(self, texts: list[str], future: asyncio.Future[np.ndarray]) -> None:
        self.texts = texts
        self.future = future


class ModelBatcher:
    def __init__(self, model: Any, max_batch: int, window_ms: float) -> None:
        self.model = model
        self.max_batch = max_batch
        self.window = max(window_ms / 1000.0, 0.0)
        self.queue: asyncio.Queue[_BatchItem] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    async def enqueue(self, texts: list[str]) -> np.ndarray:
        loop = asyncio.get_running_loop()
        if self._task is None:
            # Lazily start worker on first request to bind to the running loop.
            self._task = loop.create_task(self._worker())

        fut: asyncio.Future[np.ndarray] = loop.create_future()
        await self.queue.put(_BatchItem(texts, fut))
        return await fut

    async def _worker(self) -> None:
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
                # If there is no backlog, don't hold the first request just to wait out the window.
                if self.queue.empty():
                    break
                try:
                    nxt = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                except TimeoutError:
                    break
                batch_items.append(nxt)
                total += len(nxt.texts)

            # Prepare batch
            texts: list[str] = []
            sizes: list[int] = []
            for bi in batch_items:
                texts.extend(bi.texts)
                sizes.append(len(bi.texts))

            try:
                vectors = await loop.run_in_executor(executor, self.model.embed, texts)
            except Exception as exc:  # pragma: no cover - defensive
                for bi in batch_items:
                    if not bi.future.done():
                        bi.future.set_exception(exc)
                continue

            # Split outputs per request
            offset = 0
            for bi, size in zip(batch_items, sizes, strict=False):
                if not bi.future.done():
                    bi.future.set_result(vectors[offset : offset + size])
                offset += size

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task


class BatchingService:
    def __init__(
        self,
        registry: ModelRegistry,
        enabled: bool = True,
        max_batch_size: int | None = None,
        window_ms: float | None = None,
    ) -> None:
        self.enabled = enabled
        self.max_batch_size = max_batch_size or int(
            os.getenv("BATCH_WINDOW_MAX_SIZE", os.getenv("BATCH_MAX_SIZE", os.getenv("MAX_BATCH_SIZE", "32")))
        )
        self.window_ms = window_ms if window_ms is not None else float(os.getenv("BATCH_WINDOW_MS", "0"))
        self._batchers: dict[str, ModelBatcher] = {}
        for name in registry.list_models():
            model = registry.get(name)
            if "text-embedding" in getattr(model, "capabilities", []):
                self._batchers[name] = ModelBatcher(model, self.max_batch_size, self.window_ms)

    async def enqueue(self, model_name: str, texts: list[str]) -> np.ndarray:
        if not self.enabled:
            raise RuntimeError("Batching disabled")
        if model_name not in self._batchers:
            raise KeyError(f"Model {model_name} not registered")
        return await self._batchers[model_name].enqueue(texts)

    async def stop(self) -> None:
        for batcher in self._batchers.values():
            await batcher.stop()
