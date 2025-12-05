"""Tests for cancellation handling across all paths."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any
from unittest import mock

import numpy as np
import pytest

from app.api import _await_executor_cleanup
from app.batching import ModelBatcher, _AggregateCancel, _merge_cancel_events
from app.concurrency.limiter import (
    ShuttingDownError,
    _state,
    chat_limiter,
    embedding_limiter,
)


class TestAggregateCancelEvent:
    """Tests for _AggregateCancel in embedding batching."""

    def test_aggregate_cancel_is_set_when_any_event_set(self) -> None:
        e1 = threading.Event()
        e2 = threading.Event()
        e3 = threading.Event()

        agg = _AggregateCancel([e1, e2, e3])
        assert not agg.is_set()

        # Setting any source event should trigger the aggregate
        e2.set()
        assert agg.is_set()

    def test_aggregate_cancel_set_propagates_to_all(self) -> None:
        e1 = threading.Event()
        e2 = threading.Event()

        agg = _AggregateCancel([e1, e2])
        agg.set()

        assert e1.is_set()
        assert e2.is_set()

    def test_aggregate_cancel_wait_returns_on_any_set(self) -> None:
        e1 = threading.Event()
        e2 = threading.Event()
        agg = _AggregateCancel([e1, e2])

        # Set one event in a background thread after a delay
        def delayed_set() -> None:
            time.sleep(0.05)
            e1.set()

        t = threading.Thread(target=delayed_set)
        t.start()

        result = agg.wait(timeout=1.0)
        assert result is True
        t.join()

    def test_aggregate_cancel_wait_timeout(self) -> None:
        e1 = threading.Event()
        agg = _AggregateCancel([e1])

        result = agg.wait(timeout=0.05)
        assert result is False

    def test_merge_cancel_events_single_event(self) -> None:
        e1 = threading.Event()
        result = _merge_cancel_events([e1])
        # Single event should be returned as-is
        assert result is e1

    def test_merge_cancel_events_multiple_returns_aggregate(self) -> None:
        e1 = threading.Event()
        e2 = threading.Event()
        result = _merge_cancel_events([e1, e2])

        assert isinstance(result, _AggregateCancel)

    def test_merge_cancel_events_empty_returns_none(self) -> None:
        result = _merge_cancel_events([])
        assert result is None


class DummyEmbeddingModel:
    """Test model that tracks cancellation checks."""

    def __init__(self, delay: float = 0.0) -> None:
        self.name = "test-embed"
        self.calls = 0
        self.delay = delay
        self.cancel_checks: list[bool] = []

    def embed(self, texts: list[str], cancel_event: threading.Event | None = None) -> np.ndarray:
        self.calls += 1
        if self.delay > 0:
            time.sleep(self.delay)
        if cancel_event:
            self.cancel_checks.append(cancel_event.is_set())
        return np.zeros((len(texts), 4))


@pytest.mark.asyncio
async def test_batched_embed_aggregate_cancel_propagates() -> None:
    """Multiple batched requests with different cancel events should all be checkable."""
    # Use longer window so we can set the cancel event before embed is called
    model = DummyEmbeddingModel(delay=0.0)
    batcher = ModelBatcher(model, max_batch=8, window_ms=200, queue_size=16, queue_timeout=2.0)

    e1 = threading.Event()
    e2 = threading.Event()
    e3 = threading.Event()

    # Pre-set the middle event before enqueuing
    e2.set()

    # Enqueue multiple requests with different cancel events
    t1 = asyncio.create_task(batcher.enqueue(["a"], cancel_event=e1))
    await asyncio.sleep(0.01)
    t2 = asyncio.create_task(batcher.enqueue(["b"], cancel_event=e2))
    await asyncio.sleep(0.01)
    t3 = asyncio.create_task(batcher.enqueue(["c"], cancel_event=e3))

    results = await asyncio.gather(t1, t2, t3, return_exceptions=True)

    # Request with pre-set cancel should have been dropped
    # and raised CancelledError; other two should succeed
    cancelled_count = sum(1 for r in results if isinstance(r, asyncio.CancelledError))
    success_count = sum(1 for r in results if not isinstance(r, BaseException))
    expected_success = 2

    assert cancelled_count == 1  # e2 was pre-cancelled
    assert success_count == expected_success  # e1 and e3 succeeded

    await batcher.stop()


@pytest.mark.asyncio
async def test_aggregate_cancel_detected_during_embed() -> None:
    """When cancel is set mid-batch, aggregate should detect it."""
    e1 = threading.Event()
    e2 = threading.Event()
    e3 = threading.Event()

    agg = _AggregateCancel([e1, e2, e3])

    # Initially not set
    assert not agg.is_set()

    # Set one of the source events
    e2.set()

    # Aggregate should now report as set
    assert agg.is_set()


class TestExecutorTimeoutGracePeriod:
    """Tests for grace period handling when executor work times out."""

    @pytest.mark.asyncio
    async def test_await_executor_cleanup_logs_on_overrun(self) -> None:
        """Verify logging when executor work overruns grace period."""
        # Create a future that never completes
        loop = asyncio.get_running_loop()
        never_done: asyncio.Future[Any] = loop.create_future()

        with mock.patch("app.api.logger") as mock_logger:
            await _await_executor_cleanup(never_done, grace_period=0.05, reason="test_timeout")

            # Should have logged a warning
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args
            assert call_args[0][0] == "executor_work_overran_grace_period"
            assert call_args[1]["extra"]["reason"] == "test_timeout"

    @pytest.mark.asyncio
    async def test_await_executor_cleanup_no_log_when_fast(self) -> None:
        """Verify no logging when executor work finishes within grace period."""
        loop = asyncio.get_running_loop()
        fast_future: asyncio.Future[str] = loop.create_future()
        fast_future.set_result("done")

        with mock.patch("app.api.logger") as mock_logger:
            await _await_executor_cleanup(fast_future, grace_period=1.0, reason="test")

            # Should not have logged anything
            mock_logger.warning.assert_not_called()

    @pytest.mark.asyncio
    async def test_await_executor_cleanup_handles_exception(self) -> None:
        """Verify cleanup handles futures that raise exceptions."""
        loop = asyncio.get_running_loop()
        error_future: asyncio.Future[str] = loop.create_future()
        error_future.set_exception(ValueError("test error"))

        with mock.patch("app.api.logger") as mock_logger:
            # Should not raise
            await _await_executor_cleanup(error_future, grace_period=1.0, reason="test")
            mock_logger.warning.assert_not_called()


class TestPerCapabilityLimiters:
    """Tests for separate embedding/chat limiters."""

    @pytest.mark.asyncio
    async def test_embedding_limiter_independent_from_chat(self) -> None:
        """Embedding and chat limiters should have separate capacity."""
        # Both should be acquirable simultaneously
        async with embedding_limiter(), chat_limiter():
            # If we get here, both limiters are independent
            pass

    @pytest.mark.asyncio
    async def test_embedding_limiter_respects_shutdown(self) -> None:
        """Embedding limiter should respect shutdown state."""
        original = _state["accepting"]
        try:
            _state["accepting"] = False
            with pytest.raises(ShuttingDownError):
                async with embedding_limiter():
                    pass
        finally:
            _state["accepting"] = original

    @pytest.mark.asyncio
    async def test_chat_limiter_respects_shutdown(self) -> None:
        """Chat limiter should respect shutdown state."""
        original = _state["accepting"]
        try:
            _state["accepting"] = False
            with pytest.raises(ShuttingDownError):
                async with chat_limiter():
                    pass
        finally:
            _state["accepting"] = original

