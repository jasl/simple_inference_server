from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Callable
from concurrent.futures import Executor


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

