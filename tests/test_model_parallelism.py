import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from app.utils.executor_context import run_in_executor_with_context_limited

EVENT_WAIT_TIMEOUT_SEC = 1.0
RELEASE_WAIT_TIMEOUT_SEC = 5.0
SLEEP_GRACE_SEC = 0.2
TASK_WAIT_TIMEOUT_SEC = 2.0


class _LimitedModel:
    max_parallelism = 1


def _block_until(release: threading.Event, started: threading.Event) -> int:
    started.set()
    release.wait(timeout=RELEASE_WAIT_TIMEOUT_SEC)
    return 1


async def _wait_for_set(ev: threading.Event, *, timeout: float = EVENT_WAIT_TIMEOUT_SEC) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not ev.is_set():
        if loop.time() >= deadline:
            raise AssertionError("timed out waiting for event to be set")
        await asyncio.sleep(0.01)


async def test_model_max_parallelism_gates_executor_submission() -> None:
    model = _LimitedModel()
    release = threading.Event()
    started1 = threading.Event()
    started2 = threading.Event()

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=2) as executor:
        t1 = asyncio.create_task(
            run_in_executor_with_context_limited(loop, executor, lambda: _block_until(release, started1), model=model)
        )
        await _wait_for_set(started1)

        t2 = asyncio.create_task(
            run_in_executor_with_context_limited(loop, executor, lambda: _block_until(release, started2), model=model)
        )

        # If the limiter is working, the second call should not have been submitted yet,
        # because submission happens only after acquiring the per-model semaphore.
        await asyncio.sleep(SLEEP_GRACE_SEC)
        assert not started2.is_set()

        release.set()
        await asyncio.wait_for(t1, timeout=TASK_WAIT_TIMEOUT_SEC)
        await asyncio.wait_for(t2, timeout=TASK_WAIT_TIMEOUT_SEC)
        assert started2.is_set()

