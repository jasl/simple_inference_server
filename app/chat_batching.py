from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import os
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import torch

from app.monitoring.metrics import (
    observe_chat_batch_size,
    observe_chat_batch_wait,
    record_chat_batch_degraded_max_size,
    record_chat_batch_oom_retry,
    record_chat_batch_queue,
    record_chat_batch_queue_rejection,
    record_chat_batch_requeue,
    record_chat_count_pool_size,
)
from app.threadpool import get_chat_executor

logger = logging.getLogger(__name__)
_COUNT_EXECUTOR_REF: dict[str, ThreadPoolExecutor | None] = {"value": None}
_COUNT_EXECUTOR_LOCK = threading.Lock()
_REQUEUE_RETRIES = max(0, int(os.getenv("CHAT_REQUEUE_RETRIES", "3")))
_REQUEUE_BASE_DELAY_SEC = max(0.001, float(os.getenv("CHAT_REQUEUE_BASE_DELAY_MS", "5")) / 1000.0)
_REQUEUE_MAX_DELAY_SEC = float(os.getenv("CHAT_REQUEUE_MAX_DELAY_MS", "100")) / 1000.0
_REQUEUE_MAX_WAIT_SEC = float(os.getenv("CHAT_REQUEUE_MAX_WAIT_MS", "2000")) / 1000.0
_QUEUE_MAX_WAIT_SEC = float(os.getenv("CHAT_QUEUE_MAX_WAIT_MS", "2000")) / 1000.0
_REQUEUE_MAX_TASKS = int(os.getenv("CHAT_REQUEUE_MAX_TASKS", "64"))
_PREPARE_TIMEOUT_SEC = float(os.getenv("CHAT_PREPARE_TIMEOUT_SEC", "10"))
_GENERATE_TIMEOUT_SEC = float(os.getenv("CHAT_GENERATE_TIMEOUT_SEC", "60"))
_OOM_COOLDOWN_SEC = float(os.getenv("CHAT_OOM_COOLDOWN_SEC", "300"))  # 5 minutes default
_EXECUTOR_GRACE_PERIOD_SEC = float(os.getenv("EXECUTOR_GRACE_PERIOD_SEC", "2.0"))
_CANCELLED_SENTINEL = object()


@dataclass
class _ChatBatchItem:
    messages: Sequence[dict[str, Any]]
    max_new_tokens: int
    temperature: float
    top_p: float
    stop: tuple[str, ...]
    prompt_tokens: int | None
    prepared_inputs: dict[str, Any] | None
    future: asyncio.Future[Any]
    enqueue_time: float
    deadline: float | None
    cancel_event: threading.Event

    @property
    def config_key(self) -> tuple[float, float, int, tuple[str, ...]]:
        return (self.temperature, self.top_p, self.max_new_tokens, self.stop)


class ChatBatchQueueFullError(Exception):
    """Raised when the chat batch queue is full."""


class ChatBatchQueueTimeoutError(Exception):
    """Raised when a chat request waits too long in the batch queue."""


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
        # Lazily created on first enqueue to bind to the running loop and avoid
        # multiple workers being spawned under concurrent first use.
        self._start_lock: asyncio.Lock | None = None
        self._stopping = False
        self.oom_retries = 0
        self._requeue_tasks: dict[asyncio.Task[None], _ChatBatchItem] = {}
        self._pending_requeue_cleanup = False
        # OOM graceful degradation state
        self._current_max_batch = max_batch
        self._oom_cooldown_until: float | None = None

    async def enqueue(  # noqa: PLR0913 - explicit params keep batching contract clear
        self,
        messages: Sequence[dict[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: Sequence[str] | None,
        prompt_tokens: int | None = None,
        prepared_inputs: dict[str, Any] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Any:
        if self._stopping:
            raise RuntimeError("Chat batcher is stopping")
        # Clamp overly large requests instead of failing so the API can fall back gracefully.
        max_new_tokens = min(max_new_tokens, self.max_new_tokens_ceiling)

        loop = asyncio.get_running_loop()

        # Enforce prompt length; optionally reuse caller-provided count.
        if prompt_tokens is None:
            # Default to a dedicated counting pool to avoid head-of-line blocking chat generation threads.
            use_shared = os.getenv("CHAT_COUNT_USE_CHAT_EXECUTOR", "0") != "0"
            count_executor = _get_count_executor(use_chat_executor=use_shared)
            try:
                prompt_tokens = await asyncio.wait_for(
                    loop.run_in_executor(count_executor, self.model.count_tokens, messages),
                    timeout=_PREPARE_TIMEOUT_SEC,
                )
            except TimeoutError as exc:  # pragma: no cover - defensive
                raise ValueError("Prompt processing timed out") from exc
        if prompt_tokens is None:
            raise ValueError("Prompt token counting failed")
        prompt_tokens = int(prompt_tokens)
        if prompt_tokens > self.max_prompt_tokens:
            raise ValueError(f"Prompt too long; max {self.max_prompt_tokens} tokens")

        if self._task is None:
            # Guard worker creation with a per-instance lock so concurrent first
            # use does not accidentally start multiple batcher workers.
            if self._start_lock is None:
                self._start_lock = asyncio.Lock()
            async with self._start_lock:
                if self._task is None:
                    self._task = loop.create_task(
                        self._worker(),
                        name=f"chat-batcher-{getattr(self.model, 'name', 'model')}",
                    )

        fut: asyncio.Future[Any] = loop.create_future()
        item = _ChatBatchItem(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=tuple(stop or []),
            prompt_tokens=prompt_tokens,
            prepared_inputs=prepared_inputs,
            future=fut,
            enqueue_time=loop.time(),
            deadline=loop.time() + _REQUEUE_MAX_WAIT_SEC if _REQUEUE_MAX_WAIT_SEC > 0 else None,
            cancel_event=cancel_event or threading.Event(),
        )
        try:
            self.queue.put_nowait(item)
        except asyncio.QueueFull as exc:
            record_chat_batch_queue_rejection(self.model_name)
            logger.warning(
                "chat_batch_queue_full",
                extra={
                    "model": self.model_name,
                    "queue_size": self.queue.qsize(),
                    "max_queue_size": self.queue.maxsize,
                },
            )
            raise ChatBatchQueueFullError("Chat batch queue is full") from exc
        return await fut

    def _check_oom_recovery(self) -> None:
        """Check if OOM cooldown has expired and restore original batch size."""
        if self._oom_cooldown_until is None:
            return
        if time.perf_counter() >= self._oom_cooldown_until:
            logger.info(
                "chat_batch_oom_recovery",
                extra={
                    "model": self.model_name,
                    "restored_max_batch": self.max_batch,
                    "previous_max_batch": self._current_max_batch,
                },
            )
            self._current_max_batch = self.max_batch
            self._oom_cooldown_until = None
            record_chat_batch_degraded_max_size(self.model_name, self._current_max_batch)

    def _handle_oom_degradation(self) -> None:
        """Halve the batch size on OOM and set cooldown."""
        previous = self._current_max_batch
        self._current_max_batch = max(1, self._current_max_batch // 2)
        self._oom_cooldown_until = time.perf_counter() + _OOM_COOLDOWN_SEC
        logger.warning(
            "chat_batch_oom_degradation",
            extra={
                "model": self.model_name,
                "previous_max_batch": previous,
                "degraded_max_batch": self._current_max_batch,
                "cooldown_sec": _OOM_COOLDOWN_SEC,
            },
        )
        record_chat_batch_degraded_max_size(self.model_name, self._current_max_batch)
        self._pending_requeue_cleanup = True

    def _prune_requeue_tasks(self, now: float, *, reason: str) -> None:
        """Cancel stale requeue attempts to avoid unbounded growth."""

        expired: list[tuple[asyncio.Task[None], _ChatBatchItem]] = []
        done: list[asyncio.Task[None]] = []

        for task, item in list(self._requeue_tasks.items()):
            if task.done():
                done.append(task)
                continue

            if item.deadline is not None and now >= item.deadline:
                expired.append((task, item))

        for task in done:
            self._requeue_tasks.pop(task, None)

        for task, item in expired:
            task.cancel()
            self._requeue_tasks.pop(task, None)
            if not item.future.done():
                item.future.set_exception(ChatBatchQueueTimeoutError("Chat batch requeue wait exceeded"))
            item.cancel_event.set()
            record_chat_batch_queue_rejection(self.model_name)
            logger.warning(
                "chat_batch_requeue_pruned",
                extra={
                    "model": self.model_name,
                    "queue_size": self.queue.qsize(),
                    "reason": reason,
                    "requeue_tasks": len(self._requeue_tasks),
                    "waited_sec": now - item.enqueue_time,
                },
            )

    async def _worker(self) -> None:  # noqa: PLR0912, PLR0915 - batching loop keeps several branches for fairness/backpressure
        loop = asyncio.get_running_loop()
        executor = get_chat_executor()
        try:
            while True:
                prune_reason = "worker_loop"
                if self._pending_requeue_cleanup:
                    prune_reason = "oom"
                    self._pending_requeue_cleanup = False
                self._prune_requeue_tasks(loop.time(), reason=prune_reason)

                first = await self.queue.get()
                # Check OOM recovery before processing new batch
                self._check_oom_recovery()
                candidates = [first]
                start = time.perf_counter()
                # Pull up to _current_max_batch items within the window (may be reduced due to OOM).
                while len(candidates) < self._current_max_batch:
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
                    self._schedule_requeue(pending)

                pending_size = self.queue.qsize()
                record_chat_batch_queue(self.model_name, pending_size)
                observe_chat_batch_size(self.model_name, len(batch_items))
                now = loop.time()
                for bi in batch_items:
                    with contextlib.suppress(Exception):
                        observe_chat_batch_wait(self.model_name, now - bi.enqueue_time)

                # Drop items that waited too long in queue
                if _QUEUE_MAX_WAIT_SEC > 0:
                    alive_items: list[_ChatBatchItem] = []
                    for bi in batch_items:
                        if now - bi.enqueue_time > _QUEUE_MAX_WAIT_SEC:
                            if not bi.future.done():
                                bi.future.set_exception(
                                    ChatBatchQueueTimeoutError("Chat batch queue wait exceeded")
                                )
                            bi.cancel_event.set()
                            record_chat_batch_queue_rejection(self.model_name)
                        else:
                            alive_items.append(bi)
                    if not alive_items:
                        oldest_wait = max(now - bi.enqueue_time for bi in batch_items)
                        logger.warning(
                            "chat_batch_items_dropped_due_to_queue_wait",
                            extra={
                                "model": self.model_name,
                                "dropped": len(batch_items),
                                "queue_size": pending_size,
                                "max_queue_size": self.queue.maxsize,
                                "queue_max_wait_sec": _QUEUE_MAX_WAIT_SEC,
                                "oldest_wait_sec": oldest_wait,
                            },
                        )
                        continue
                    batch_items = alive_items

                run_batch = functools.partial(self._generate_batch, list(batch_items))
                run_future = asyncio.wrap_future(loop.run_in_executor(executor, run_batch))
                try:
                    results = await asyncio.wait_for(run_future, timeout=_GENERATE_TIMEOUT_SEC)
                except TimeoutError:  # pragma: no cover - defensive
                    for bi in batch_items:
                        bi.cancel_event.set()
                        if not bi.future.done():
                            bi.future.set_exception(ChatBatchQueueTimeoutError("Chat generation timed out"))
                    record_chat_batch_queue_rejection(self.model_name)
                    await _await_future_grace(run_future, reason="chat_generate_timeout")
                    continue
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
                    if gen is _CANCELLED_SENTINEL:
                        if not bi.future.done():
                            bi.future.set_exception(asyncio.CancelledError())
                        continue
                    if not bi.future.done():
                        bi.future.set_result(gen)
        except asyncio.CancelledError:
            # Propagate cancellation to pending futures.
            while not self.queue.empty():
                pending = self.queue.get_nowait()
                if not pending.future.done():
                    pending.future.set_exception(asyncio.CancelledError())
                pending.cancel_event.set()
            raise

    # --- helpers ------------------------------------------------------------
    def _generate_batch(self, batch_items: list[_ChatBatchItem]) -> list[Any]:
        """Run batched generation if the model supports it; fall back to per-request."""

        stop_list = list(batch_items[0].stop)
        max_new_tokens = batch_items[0].max_new_tokens
        temperature = batch_items[0].temperature
        top_p = batch_items[0].top_p
        cancel_events = [bi.cancel_event for bi in batch_items]

        try:
            prepared_inputs = [bi.prepared_inputs for bi in batch_items]
            has_prepared = all(pi is not None for pi in prepared_inputs)

            if has_prepared and hasattr(self.model, "batched_generate_prepared"):
                return self.model.batched_generate_prepared(
                    prepared_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop_list,
                    cancel_events=cancel_events,
                )

            if hasattr(self.model, "batched_generate"):
                return self.model.batched_generate(
                    [bi.messages for bi in batch_items],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop_list,
                    cancel_events=cancel_events,
                )

            # Fallback: run sequentially (still reduces queue contention and keeps
            # behaviour consistent with non-batched paths).
            outputs: list[Any] = []
            for bi in batch_items:
                if bi.cancel_event.is_set():
                    outputs.append(_CANCELLED_SENTINEL)
                    continue
                outputs.append(
                    self._generate_single(bi, stop_list, max_new_tokens, temperature, top_p)
                )
            return outputs
        except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):  # pragma: no cover - OOM guard
            self.oom_retries += 1
            record_chat_batch_oom_retry(self.model_name)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Apply graceful degradation: halve batch size for future batches
            self._handle_oom_degradation()
            if len(batch_items) == 1:
                raise
            # Retry sequentially to reduce peak memory.
            return [
                self._generate_single(bi, stop_list, max_new_tokens, temperature, top_p)
                for bi in batch_items
            ]

    def _generate_single(
        self,
        item: _ChatBatchItem,
        stop_list: list[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Any:
        if item.prepared_inputs is not None and hasattr(self.model, "generate_prepared"):
            return self.model.generate_prepared(
                item.prepared_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop_list,
                cancel_event=item.cancel_event,
            )

        return self.model.generate(
            item.messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_list,
            cancel_event=item.cancel_event,
        )

    def _schedule_requeue(self, item: _ChatBatchItem) -> None:
        """Requeue leftovers without blocking the main worker loop."""

        # Drop if deadline passed
        if item.deadline is not None and asyncio.get_running_loop().time() >= item.deadline:
            if not item.future.done():
                item.future.set_exception(ChatBatchQueueTimeoutError("Chat batch queue deadline exceeded"))
            item.cancel_event.set()
            record_chat_batch_queue_rejection(self.model_name)
            logger.warning(
                "chat_batch_requeue_deadline_exceeded",
                extra={
                    "model": self.model_name,
                    "queue_size": self.queue.qsize(),
                    "max_queue_size": self.queue.maxsize,
                    "requeue_max_wait_sec": _REQUEUE_MAX_WAIT_SEC,
                },
            )
            return

        try:
            self.queue.put_nowait(item)
            record_chat_batch_requeue(self.model_name)
            return
        except asyncio.QueueFull:
            pass

        # Avoid unbounded background retries under sustained pressure
        if len(self._requeue_tasks) >= _REQUEUE_MAX_TASKS:
            if not item.future.done():
                item.future.set_exception(ChatBatchQueueTimeoutError("Chat batch requeue backlog exceeded"))
            item.cancel_event.set()
            record_chat_batch_queue_rejection(self.model_name)
            logger.warning(
                "chat_batch_requeue_backlog_exceeded",
                extra={
                    "model": self.model_name,
                    "queue_size": self.queue.qsize(),
                    "max_queue_size": self.queue.maxsize,
                    "requeue_tasks": len(self._requeue_tasks),
                    "requeue_max_tasks": _REQUEUE_MAX_TASKS,
                },
            )
            return

        async def _retry() -> None:
            delay = _REQUEUE_BASE_DELAY_SEC
            loop = asyncio.get_running_loop()
            deadline = loop.time() + _REQUEUE_MAX_WAIT_SEC if _REQUEUE_MAX_WAIT_SEC > 0 else None

            while True:
                try:
                    if deadline is None:
                        await self.queue.put(item)
                    else:
                        remaining = deadline - loop.time()
                        if remaining <= 0:
                            raise TimeoutError
                        await asyncio.wait_for(self.queue.put(item), timeout=remaining)
                    record_chat_batch_requeue(self.model_name)
                    return
                except TimeoutError:
                    break
                await asyncio.sleep(delay)
                delay = min(delay * 2, _REQUEUE_MAX_DELAY_SEC)

            if not item.future.done():
                item.future.set_exception(ChatBatchQueueTimeoutError("Chat batch queue full"))
            item.cancel_event.set()
            record_chat_batch_queue_rejection(self.model_name)
            logger.warning(
                "chat_batch_requeue_timeout",
                extra={
                    "model": self.model_name,
                    "queue_size": self.queue.qsize(),
                    "max_queue_size": self.queue.maxsize,
                    "requeue_max_wait_sec": _REQUEUE_MAX_WAIT_SEC,
                    "waited_sec": (loop.time() - item.enqueue_time),
                },
            )
            return

        task = asyncio.create_task(_retry())
        self._requeue_tasks[task] = item

        def _cleanup(_t: asyncio.Task[Any]) -> None:
            self._requeue_tasks.pop(_t, None)

        task.add_done_callback(_cleanup)

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            while not self.queue.empty():
                pending = self.queue.get_nowait()
                if not pending.future.done():
                    pending.future.set_exception(asyncio.CancelledError())
                pending.cancel_event.set()
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        requeue_tasks = list(self._requeue_tasks.items())
        for task, item in requeue_tasks:
            item.cancel_event.set()
            task.cancel()
        if requeue_tasks:
            with contextlib.suppress(Exception):
                await asyncio.gather(*(task for task, _ in requeue_tasks), return_exceptions=True)
        self._requeue_tasks.clear()


def shutdown_count_executor() -> None:
    executor = _COUNT_EXECUTOR_REF.get("value")
    _COUNT_EXECUTOR_REF["value"] = None
    if executor is not None:
        executor.shutdown(wait=True)


def _get_count_executor(*, use_chat_executor: bool) -> ThreadPoolExecutor:
    if use_chat_executor:
        executor = get_chat_executor()
        workers = max(1, getattr(executor, "_max_workers", 1))
        record_chat_count_pool_size(workers)
        return executor

    with _COUNT_EXECUTOR_LOCK:
        cached_executor: ThreadPoolExecutor | None = _COUNT_EXECUTOR_REF.get("value")
        if cached_executor is None:
            workers = max(1, int(os.getenv("CHAT_COUNT_MAX_WORKERS", "2")))
            count_executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="chat-count")
            _COUNT_EXECUTOR_REF["value"] = count_executor
            record_chat_count_pool_size(workers)
            return count_executor
        record_chat_count_pool_size(max(1, getattr(cached_executor, "_max_workers", 1)))
        return cached_executor


def get_count_executor(*, use_chat_executor: bool = False) -> ThreadPoolExecutor:
    """Public helper to reuse the chat token counting pool."""

    return _get_count_executor(use_chat_executor=use_chat_executor)


async def _await_future_grace(fut: asyncio.Future[Any], *, reason: str) -> None:
    """Wait briefly for an executor future after signalling cancellation.

    Mirrors the API-layer grace period handling so batched generation doesn't
    leave long-running work pinned to the chat executor after timeouts.
    """

    if fut.done():
        return

    try:
        await asyncio.wait_for(asyncio.shield(fut), timeout=_EXECUTOR_GRACE_PERIOD_SEC)
    except TimeoutError:
        logger.warning(
            "chat_executor_overrun",
            extra={"reason": reason, "grace_period_sec": _EXECUTOR_GRACE_PERIOD_SEC},
        )
    except Exception:  # noqa: S110 - best-effort cleanup
        return


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
        prompt_tokens: int | None = None,
        prepared_inputs: dict[str, Any] | None = None,
        cancel_event: threading.Event | None = None,
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
            prompt_tokens=prompt_tokens,
            prepared_inputs=prepared_inputs,
            cancel_event=cancel_event,
        )

    async def stop(self) -> None:
        for batcher in self._batchers.values():
            await batcher.stop()

    def queue_stats(self) -> dict[str, tuple[int, int]]:
        stats: dict[str, tuple[int, int]] = {}
        for name, batcher in self._batchers.items():
            stats[name] = (batcher.queue.qsize(), batcher.queue.maxsize)
        return stats
