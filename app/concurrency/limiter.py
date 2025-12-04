import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.monitoring.metrics import record_queue_rejection


class QueueFullError(Exception):
    """Raised when the request queue is full."""


class QueueTimeoutError(Exception):
    """Raised when waiting for an available worker times out."""


class ShuttingDownError(Exception):
    """Raised when service is draining and not accepting new work."""


MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "4"))
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "64"))
QUEUE_TIMEOUT_SEC = float(os.getenv("QUEUE_TIMEOUT_SEC", "2.0"))

_semaphore: asyncio.Semaphore = asyncio.Semaphore(MAX_CONCURRENT)
_queue: asyncio.Queue[int] = asyncio.Queue(MAX_QUEUE_SIZE)
_state = {"accepting": True}
_in_flight_count = 0
_drain_event: asyncio.Event = asyncio.Event()
_drain_event.set()


def _update_drain_event() -> None:
    if _queue.qsize() == 0 and _in_flight_count == 0:
        _drain_event.set()
    else:
        _drain_event.clear()


@asynccontextmanager
async def limiter() -> AsyncIterator[None]:
    global _in_flight_count
    if not _state["accepting"]:
        raise ShuttingDownError("Service is shutting down")
    try:
        _queue.put_nowait(1)
        _update_drain_event()
    except asyncio.QueueFull as exc:  # queue already at capacity
        record_queue_rejection()
        raise QueueFullError("Request queue is full") from exc

    try:
        try:
            await asyncio.wait_for(_semaphore.acquire(), timeout=QUEUE_TIMEOUT_SEC)
        except TimeoutError as exc:  # waited too long
            record_queue_rejection()
            raise QueueTimeoutError("Timed out waiting for worker") from exc

        try:
            _in_flight_count += 1
            _update_drain_event()
            yield
        finally:
            _in_flight_count -= 1
            _semaphore.release()
            _update_drain_event()
    finally:
        _queue.get_nowait()
        _queue.task_done()
        _update_drain_event()


def stop_accepting() -> None:
    """Block new work from entering the queue."""
    _state["accepting"] = False


async def wait_for_drain(timeout: float = 5.0) -> None:
    """Wait for in-flight work to finish, with a timeout."""
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        if _drain_event.is_set():
            break
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            break
        try:
            await asyncio.wait_for(_drain_event.wait(), timeout=remaining)
        except asyncio.TimeoutError:
            break
