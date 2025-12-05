import asyncio
import contextlib
import contextvars
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.monitoring.metrics import GENERIC_LABEL_WARN, observe_queue_wait, record_queue_rejection


class QueueFullError(Exception):
    """Raised when the request queue is full."""


class QueueTimeoutError(Exception):
    """Raised when waiting for an available worker times out."""


class ShuttingDownError(Exception):
    """Raised when service is draining and not accepting new work."""


# Global defaults
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "4"))
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "64"))
QUEUE_TIMEOUT_SEC = float(os.getenv("QUEUE_TIMEOUT_SEC", "2.0"))

# Per-capability settings (fall back to global if not set)
EMBEDDING_MAX_CONCURRENT = int(os.getenv("EMBEDDING_MAX_CONCURRENT", str(MAX_CONCURRENT)))
EMBEDDING_MAX_QUEUE_SIZE = int(os.getenv("EMBEDDING_MAX_QUEUE_SIZE", str(MAX_QUEUE_SIZE)))
EMBEDDING_QUEUE_TIMEOUT_SEC = float(os.getenv("EMBEDDING_QUEUE_TIMEOUT_SEC", str(QUEUE_TIMEOUT_SEC)))

CHAT_MAX_CONCURRENT = int(os.getenv("CHAT_MAX_CONCURRENT", str(MAX_CONCURRENT)))
CHAT_MAX_QUEUE_SIZE = int(os.getenv("CHAT_MAX_QUEUE_SIZE", str(MAX_QUEUE_SIZE)))
CHAT_QUEUE_TIMEOUT_SEC = float(os.getenv("CHAT_QUEUE_TIMEOUT_SEC", str(QUEUE_TIMEOUT_SEC)))

# Shared state for shutdown coordination
_state = {"accepting": True}
_queue_label: contextvars.ContextVar[str] = contextvars.ContextVar("queue_label", default="generic")

# Embedding limiter state
_embedding_semaphore: asyncio.Semaphore = asyncio.Semaphore(EMBEDDING_MAX_CONCURRENT)
_embedding_queue: asyncio.Queue[int] = asyncio.Queue(EMBEDDING_MAX_QUEUE_SIZE)
_embedding_in_flight_state = {"count": 0}
_embedding_in_flight_lock = asyncio.Lock()

# Chat limiter state
_chat_semaphore: asyncio.Semaphore = asyncio.Semaphore(CHAT_MAX_CONCURRENT)
_chat_queue: asyncio.Queue[int] = asyncio.Queue(CHAT_MAX_QUEUE_SIZE)
_chat_in_flight_state = {"count": 0}
_chat_in_flight_lock = asyncio.Lock()


def set_queue_label(label: str) -> contextvars.Token[str]:
    return _queue_label.set(label)


def reset_queue_label(token: contextvars.Token[str]) -> None:
    with contextlib.suppress(Exception):
        _queue_label.reset(token)


@asynccontextmanager
async def embedding_limiter() -> AsyncIterator[None]:
    """Per-capability concurrency guard for embedding work.

    Provides a dedicated pool for embeddings so bursty chat traffic doesn't
    starve embedding requests. Falls back to global settings if per-capability
    envs are not set.
    """
    if not _state["accepting"]:
        raise ShuttingDownError("Service is shutting down")
    queued = False
    label = _queue_label.get()
    if label == "generic":
        GENERIC_LABEL_WARN.inc()
    try:
        _embedding_queue.put_nowait(1)
        queued = True
    except asyncio.QueueFull as exc:
        record_queue_rejection()
        raise QueueFullError("Embedding request queue is full") from exc

    acquired = False
    start_wait = asyncio.get_running_loop().time()
    try:
        try:
            await asyncio.wait_for(_embedding_semaphore.acquire(), timeout=EMBEDDING_QUEUE_TIMEOUT_SEC)
            acquired = True
            observe_queue_wait(label, asyncio.get_running_loop().time() - start_wait)
        except TimeoutError as exc:
            record_queue_rejection()
            raise QueueTimeoutError("Timed out waiting for embedding worker") from exc

        async with _embedding_in_flight_lock:
            _embedding_in_flight_state["count"] += 1
        try:
            yield
        finally:
            async with _embedding_in_flight_lock:
                _embedding_in_flight_state["count"] = max(0, _embedding_in_flight_state["count"] - 1)
    finally:
        if acquired:
            _embedding_semaphore.release()
        if queued:
            _embedding_queue.get_nowait()
            _embedding_queue.task_done()


@asynccontextmanager
async def chat_limiter() -> AsyncIterator[None]:
    """Per-capability concurrency guard for chat work.

    Provides a dedicated pool for chat so bursty embedding traffic doesn't
    starve chat requests. Falls back to global settings if per-capability
    envs are not set.
    """
    if not _state["accepting"]:
        raise ShuttingDownError("Service is shutting down")
    queued = False
    label = _queue_label.get()
    if label == "generic":
        GENERIC_LABEL_WARN.inc()
    try:
        _chat_queue.put_nowait(1)
        queued = True
    except asyncio.QueueFull as exc:
        record_queue_rejection()
        raise QueueFullError("Chat request queue is full") from exc

    acquired = False
    start_wait = asyncio.get_running_loop().time()
    try:
        try:
            await asyncio.wait_for(_chat_semaphore.acquire(), timeout=CHAT_QUEUE_TIMEOUT_SEC)
            acquired = True
            observe_queue_wait(label, asyncio.get_running_loop().time() - start_wait)
        except TimeoutError as exc:
            record_queue_rejection()
            raise QueueTimeoutError("Timed out waiting for chat worker") from exc

        async with _chat_in_flight_lock:
            _chat_in_flight_state["count"] += 1
        try:
            yield
        finally:
            async with _chat_in_flight_lock:
                _chat_in_flight_state["count"] = max(0, _chat_in_flight_state["count"] - 1)
    finally:
        if acquired:
            _chat_semaphore.release()
        if queued:
            _chat_queue.get_nowait()
            _chat_queue.task_done()


def stop_accepting() -> None:
    """Block new work from entering the queue."""
    _state["accepting"] = False


async def wait_for_drain(timeout: float = 5.0) -> None:
    """Wait for in-flight work across all limiters to finish, with a timeout."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        # Check all limiter states
        async with _embedding_in_flight_lock:
            embedding_active = _embedding_in_flight_state["count"]
        async with _chat_in_flight_lock:
            chat_active = _chat_in_flight_state["count"]
        queue_backlog = _embedding_queue.qsize() + _chat_queue.qsize()
        total_active = embedding_active + chat_active
        if total_active == 0 and queue_backlog == 0:
            break
        if loop.time() >= deadline:
            break
        await asyncio.sleep(0.05)
