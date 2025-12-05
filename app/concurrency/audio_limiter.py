import asyncio
import contextlib
import contextvars
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.monitoring.metrics import AUDIO_GENERIC_LABEL_WARN, observe_audio_queue_wait, record_queue_rejection


class AudioQueueFullError(Exception):
    """Raised when the audio request queue is full."""


class AudioQueueTimeoutError(Exception):
    """Raised when waiting for an available audio worker times out."""


class AudioShuttingDownError(Exception):
    """Raised when service is draining and not accepting new audio work."""


_MAX_CONCURRENT_DEFAULT = os.getenv("MAX_CONCURRENT", "4")
MAX_CONCURRENT = int(os.getenv("AUDIO_MAX_CONCURRENT", _MAX_CONCURRENT_DEFAULT))
_MAX_QUEUE_DEFAULT = os.getenv("MAX_QUEUE_SIZE", "64")
MAX_QUEUE_SIZE = int(os.getenv("AUDIO_MAX_QUEUE_SIZE", _MAX_QUEUE_DEFAULT))
_TIMEOUT_DEFAULT = os.getenv("QUEUE_TIMEOUT_SEC", "2.0")
QUEUE_TIMEOUT_SEC = float(os.getenv("AUDIO_QUEUE_TIMEOUT_SEC", _TIMEOUT_DEFAULT))

_semaphore: asyncio.Semaphore = asyncio.Semaphore(MAX_CONCURRENT)
_queue: asyncio.Queue[int] = asyncio.Queue(MAX_QUEUE_SIZE)
_state = {"accepting": True}
_in_flight_state = {"count": 0}
_in_flight_lock = asyncio.Lock()
# Use a distinct sentinel so that "audio" can be a legitimate fallback label
# (for requests without a concrete model name) without triggering warnings.
_queue_label: contextvars.ContextVar[str] = contextvars.ContextVar(
    "audio_queue_label",
    default="audio_unset",
)


async def _change_in_flight(delta: int) -> None:
    async with _in_flight_lock:
        new_count = _in_flight_state["count"] + delta
        _in_flight_state["count"] = max(0, new_count)


async def _get_in_flight() -> int:
    async with _in_flight_lock:
        return _in_flight_state["count"]


@asynccontextmanager
async def limiter() -> AsyncIterator[None]:
    """Dedicated concurrency guard for audio/Whisper work.

    Mirrors the main limiter but uses AUDIO_* settings so that long-running
    audio jobs cannot starve chat/embedding traffic.
    """
    if not _state["accepting"]:
        raise AudioShuttingDownError("Service is shutting down")
    queued = False
    label = _queue_label.get()
    if label == "audio_unset":
        AUDIO_GENERIC_LABEL_WARN.inc()
    try:
        _queue.put_nowait(1)
        queued = True
    except asyncio.QueueFull as exc:  # queue already at capacity
        record_queue_rejection()
        raise AudioQueueFullError("Audio request queue is full") from exc

    acquired = False
    start_wait = asyncio.get_running_loop().time()
    try:
        try:
            await asyncio.wait_for(_semaphore.acquire(), timeout=QUEUE_TIMEOUT_SEC)
            acquired = True
            observe_audio_queue_wait(label, asyncio.get_running_loop().time() - start_wait)
        except TimeoutError as exc:  # waited too long
            record_queue_rejection()
            raise AudioQueueTimeoutError("Timed out waiting for audio worker") from exc

        await _change_in_flight(1)
        try:
            yield
        finally:
            try:
                await asyncio.shield(_change_in_flight(-1))
            except asyncio.CancelledError:
                raise
    finally:
        if acquired:
            _semaphore.release()
        if queued:
            _queue.get_nowait()
            _queue.task_done()


def set_queue_label(label: str) -> contextvars.Token[str]:
    return _queue_label.set(label)


def reset_queue_label(token: contextvars.Token[str]) -> None:
    with contextlib.suppress(Exception):
        _queue_label.reset(token)


def stop_accepting() -> None:
    """Block new audio work from entering the queue."""
    _state["accepting"] = False


def start_accepting() -> None:
    """Allow new audio work after a prior shutdown."""
    _state["accepting"] = True


async def wait_for_drain(timeout: float = 5.0) -> None:
    """Wait for in-flight audio work to finish, with a timeout."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        active = await _get_in_flight()
        queue_backlog = _queue.qsize()
        if active == 0 and queue_backlog == 0:
            break
        if loop.time() >= deadline:
            break
        await asyncio.sleep(0.05)
