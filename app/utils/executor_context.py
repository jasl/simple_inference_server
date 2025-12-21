from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Callable
from concurrent.futures import Executor

_MODEL_PARALLELISM_SEM_ATTR = "_model_parallelism_semaphore"

def run_in_executor_with_context[T](
    loop: asyncio.AbstractEventLoop,
    executor: Executor | None,
    func: Callable[[], T],
) -> asyncio.Future[T]:
    """Run a callable in an executor while propagating contextvars.

    Python does not automatically propagate `contextvars` into executor threads.
    This helper captures the current context and runs the callable inside it.
    """

    ctx = contextvars.copy_context()
    return loop.run_in_executor(executor, ctx.run, func)


def _get_max_parallelism(model: object) -> int | None:
    max_parallelism = getattr(model, "max_parallelism", None)
    if max_parallelism is None:
        return None
    if isinstance(max_parallelism, bool):
        return None
    if not isinstance(max_parallelism, int):
        try:
            max_parallelism = int(max_parallelism)
        except (TypeError, ValueError):
            return None
    return max_parallelism if max_parallelism > 0 else None


def _get_or_create_model_semaphore(model: object, *, max_parallelism: int) -> asyncio.Semaphore:
    existing = getattr(model, _MODEL_PARALLELISM_SEM_ATTR, None)
    if isinstance(existing, asyncio.Semaphore):
        return existing
    sem = asyncio.Semaphore(max_parallelism)
    setattr(model, _MODEL_PARALLELISM_SEM_ATTR, sem)
    return sem


async def run_in_executor_with_context_limited[T](
    loop: asyncio.AbstractEventLoop,
    executor: Executor | None,
    func: Callable[[], T],
    *,
    model: object | None = None,
) -> T:
    """Run an executor callable with contextvars and optional per-model parallelism.

    If the provided `model` exposes `max_parallelism: int`, this function limits the
    number of in-flight executor calls for that model. The limit is enforced *before*
    scheduling work onto the executor to avoid wasting threads waiting on internal locks.

    Note: cancellation of the awaiting coroutine does not cancel the underlying executor
    work; it only stops waiting. We use `asyncio.shield` so the release hook runs when the
    executor work actually finishes.
    """

    if model is None:
        return await run_in_executor_with_context(loop, executor, func)

    max_parallelism = _get_max_parallelism(model)
    if max_parallelism is None:
        return await run_in_executor_with_context(loop, executor, func)

    sem = _get_or_create_model_semaphore(model, max_parallelism=max_parallelism)
    await sem.acquire()

    fut = run_in_executor_with_context(loop, executor, func)
    fut.add_done_callback(lambda _fut: sem.release())
    return await asyncio.shield(fut)

