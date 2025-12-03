from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import torch

from app.monitoring.metrics import (
    observe_chat_batch_size,
    record_chat_batch_oom_retry,
    record_chat_batch_queue,
)
from app.threadpool import get_chat_executor

logger = logging.getLogger(__name__)
_COUNT_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="chat-count")


@dataclass
class _ChatBatchItem:
    messages: Sequence[dict[str, Any]]
    max_new_tokens: int
    temperature: float
    top_p: float
    stop: tuple[str, ...]
    future: asyncio.Future[Any]

    @property
    def config_key(self) -> tuple[float, float, int, tuple[str, ...]]:
        return (self.temperature, self.top_p, self.max_new_tokens, self.stop)


class ChatBatcher:
    """Per-model chat batcher using a single worker to keep the model thread-safe."""

    def __init__(self, model: Any, *, max_batch: int, window_ms: float, max_prompt_tokens: int, max_new_tokens_ceiling: int, queue_size: int) -> None:  # noqa: PLR0913 - explicit config args keep callsites readable
        self.model = model
        self.max_batch = max_batch
        self.window = max(window_ms / 1000.0, 0.0)
        self.max_prompt_tokens = max_prompt_tokens
        self.max_new_tokens_ceiling = max_new_tokens_ceiling
        self.model_name = getattr(model, "name", "unknown")
        # Bounded queue to avoid unbounded RAM growth under slow/backlogged models.
        self.queue: asyncio.Queue[_ChatBatchItem] = asyncio.Queue(maxsize=max(queue_size, 1))
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self.oom_retries = 0

    async def enqueue(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: Sequence[str] | None,
    ) -> Any:
        if self._stopping:
            raise RuntimeError("Chat batcher is stopping")
        # Clamp overly large requests instead of failing so the API can fall back gracefully.
        max_new_tokens = min(max_new_tokens, self.max_new_tokens_ceiling)

        loop = asyncio.get_running_loop()

        # Enforce prompt length; use a lightweight dedicated executor to avoid starving the main pool.
        prompt_tokens = await loop.run_in_executor(_COUNT_EXECUTOR, self.model.count_tokens, messages)
        if prompt_tokens > self.max_prompt_tokens:
            raise ValueError(f"Prompt too long; max {self.max_prompt_tokens} tokens")

        if self._task is None:
            self._task = loop.create_task(self._worker(), name=f"chat-batcher-{getattr(self.model, 'name', 'model')}")

        fut: asyncio.Future[Any] = loop.create_future()
        item = _ChatBatchItem(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=tuple(stop or []),
            future=fut,
        )
        try:
            self.queue.put_nowait(item)
        except asyncio.QueueFull as exc:
            raise RuntimeError("Chat batch queue is full") from exc
        return await fut

    async def _worker(self) -> None:  # noqa: PLR0912 - batching loop keeps several branches for fairness/backpressure
        loop = asyncio.get_running_loop()
        executor = get_chat_executor()
        try:
            while True:
                first = await self.queue.get()
                candidates = [first]
                start = time.perf_counter()
                # Pull up to max_batch items within the window, regardless of config.
                while len(candidates) < self.max_batch:
                    remaining = self.window - (time.perf_counter() - start)
                    if remaining <= 0:
                        break
                    try:
                        candidates.append(await asyncio.wait_for(self.queue.get(), timeout=remaining))
                    except TimeoutError:
                        break

                # Partition by config_key; process the largest-compatible bucket; push back the rest.
                buckets: dict[tuple[float, float, int, tuple[str, ...]], list[_ChatBatchItem]] = {}
                for item in candidates:
                    buckets.setdefault(item.config_key, []).append(item)
                batch_items = max(buckets.values(), key=len)
                leftover: list[_ChatBatchItem] = [it for bucket in buckets.values() for it in bucket if it not in batch_items]
                for pending in leftover:
                    try:
                        self.queue.put_nowait(pending)
                    except asyncio.QueueFull:
                        if not pending.future.done():
                            pending.future.set_exception(asyncio.QueueFull("Chat batch queue full"))

                pending_size = self.queue.qsize()
                record_chat_batch_queue(self.model_name, pending_size + len(leftover))
                observe_chat_batch_size(self.model_name, len(batch_items))

                try:
                    run_batch = functools.partial(self._generate_batch, list(batch_items))
                    results = await loop.run_in_executor(executor, run_batch)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception(
                        "chat_batch_failed",
                        extra={
                            "model": getattr(self.model, "name", "unknown"),
                            "batch_size": len(batch_items),
                        },
                    )
                    for bi in batch_items:
                        if not bi.future.done():
                            bi.future.set_exception(exc)
                    continue

                for bi, gen in zip(batch_items, results, strict=False):
                    if not bi.future.done():
                        bi.future.set_result(gen)
        except asyncio.CancelledError:
            # Propagate cancellation to pending futures.
            while not self.queue.empty():
                pending = self.queue.get_nowait()
                if not pending.future.done():
                    pending.future.set_exception(asyncio.CancelledError())
            raise

    # --- helpers ------------------------------------------------------------
    def _generate_batch(self, batch_items: list[_ChatBatchItem]) -> list[Any]:
        """Run batched generation if the model supports it; fall back to per-request."""

        stop_list = list(batch_items[0].stop)
        max_new_tokens = batch_items[0].max_new_tokens
        temperature = batch_items[0].temperature
        top_p = batch_items[0].top_p

        try:
            if hasattr(self.model, "batched_generate"):
                return self.model.batched_generate(
                    [bi.messages for bi in batch_items],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop_list,
                )
            # Fallback: run sequentially (still reduces queue contention)
            return [
                self.model.generate(
                    bi.messages,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop_list,
                )
                for bi in batch_items
            ]
        except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):  # pragma: no cover - OOM guard
            self.oom_retries += 1
            record_chat_batch_oom_retry(self.model_name)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if len(batch_items) == 1:
                raise
            # Retry sequentially to reduce peak memory.
            return [
                self.model.generate(
                    bi.messages,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop_list,
                )
                for bi in batch_items
            ]

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None


class ChatBatchingService:
    """Manages chat batchers per model (text-only for now)."""

    def __init__(  # noqa: PLR0913 - config-rich initializer kept explicit for clarity
        self,
        registry: Any,
        *,
        enabled: bool,
        max_batch_size: int,
        window_ms: float,
        max_prompt_tokens: int,
        max_new_tokens_ceiling: int,
        queue_size: int,
        allow_vision: bool = False,
    ) -> None:
        self.enabled = enabled
        self.max_batch_size = max_batch_size
        self.window_ms = window_ms
        self.max_prompt_tokens = max_prompt_tokens
        self.max_new_tokens_ceiling = max_new_tokens_ceiling
        self.allow_vision = allow_vision
        self._batchers: dict[str, ChatBatcher] = {}

        if not enabled:
            return

        for name in registry.list_models():
            model = registry.get(name)
            capabilities = getattr(model, "capabilities", [])
            if "chat-completion" not in capabilities:
                continue
            if not allow_vision and "vision" in capabilities:
                continue
            self._batchers[name] = ChatBatcher(
                model,
                max_batch=max_batch_size,
                window_ms=window_ms,
                max_prompt_tokens=max_prompt_tokens,
                max_new_tokens_ceiling=max_new_tokens_ceiling,
                queue_size=queue_size,
            )

    def is_supported(self, model_name: str) -> bool:
        return self.enabled and model_name in self._batchers

    async def enqueue(  # noqa: PLR0913 - API mirrors model.generate parameters
        self,
        model_name: str,
        messages: Sequence[dict[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: Sequence[str] | None,
    ) -> Any:
        if not self.enabled:
            raise RuntimeError("Chat batching disabled")
        if model_name not in self._batchers:
            raise KeyError(f"Model {model_name} not registered for chat batching")
        return await self._batchers[model_name].enqueue(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )

    async def stop(self) -> None:
        for batcher in self._batchers.values():
            await batcher.stop()
