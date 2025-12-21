from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from fastapi import Request

from app.config import settings

EXECUTOR_GRACE_PERIOD = settings.executor_grace_period_sec


class _WorkTimeoutError(Exception):
    """Internal marker for executor work timeouts."""


class _RequestCancelledError(Exception):
    """Internal marker when work is cancelled before completion."""


class _ClientDisconnectedError(Exception):
    """Internal marker when the client disconnects first."""


async def _cancel_on_disconnect(request: Request, event: threading.Event) -> None:
    """Set cancel event if client disconnects."""

    try:
        while True:
            if await request.is_disconnected():
                event.set()
                return
            await asyncio.sleep(0.05)
    except Exception:
        return


async def _await_executor_cleanup(
    work_task: asyncio.Future[Any],
    grace_period: float,
    reason: str,
) -> None:
    """Wait briefly for executor work to finish after cancellation."""

    if work_task.done():
        return

    try:
        await asyncio.wait_for(asyncio.shield(work_task), timeout=grace_period)
    except TimeoutError:
        try:
            from app import api as api_module  # noqa: PLC0415 - local import to avoid circular import

            log_obj = getattr(api_module, "logger", logging.getLogger(__name__))
        except Exception:
            log_obj = logging.getLogger(__name__)
        log_obj.warning(
            "executor_work_overran_grace_period",
            extra={
                "reason": reason,
                "grace_period_sec": grace_period,
            },
        )
    except Exception:  # noqa: S110 - intentionally silencing; work completion is enough
        pass


async def _run_work_with_client_cancel(  # noqa: D401
    request: Request,
    work_task: asyncio.Future[Any],
    cancel_event: threading.Event,
    timeout: float,
) -> Any:
    """Race a background executor task against client disconnect and timeout."""

    disconnect_task: asyncio.Task[None] = asyncio.create_task(_cancel_on_disconnect(request, cancel_event))
    try:
        done, _pending = await asyncio.wait(
            {work_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
            timeout=timeout,
        )
        if not done:
            cancel_event.set()
            work_task.cancel()
            await _await_executor_cleanup(work_task, EXECUTOR_GRACE_PERIOD, "timeout")
            raise _WorkTimeoutError()

        if work_task in done:
            try:
                return work_task.result()
            except (asyncio.CancelledError, RuntimeError) as exc:
                cancel_event.set()
                raise _RequestCancelledError() from exc

        cancel_event.set()
        work_task.cancel()
        await _await_executor_cleanup(work_task, EXECUTOR_GRACE_PERIOD, "client_disconnect")
        raise _ClientDisconnectedError()
    finally:
        disconnect_task.cancel()
        # Best-effort cleanup: do not block request completion on the disconnect watcher.
        # In some test client / portal configurations, awaiting this task can deadlock.
