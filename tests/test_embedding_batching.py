import asyncio
import threading

import numpy as np
import pytest

from app.batching import ModelBatcher


class DummyEmbeddingModel:
    def __init__(self) -> None:
        self.name = "dummy-embed"
        self.calls = 0
        self.last_texts: list[str] | None = None
        self.last_cancel_event: threading.Event | None = None

    def embed(self, texts: list[str], cancel_event: threading.Event | None = None) -> np.ndarray:
        self.calls += 1
        self.last_texts = list(texts)
        self.last_cancel_event = cancel_event
        return np.zeros((len(texts), 4))


@pytest.mark.asyncio
async def test_model_batcher_batches_requests_within_window() -> None:
    model = DummyEmbeddingModel()
    batcher = ModelBatcher(model, max_batch=8, window_ms=50, queue_size=16, queue_timeout=0.1)

    t1 = asyncio.create_task(batcher.enqueue(["a"]))
    await asyncio.sleep(0.01)
    t2 = asyncio.create_task(batcher.enqueue(["b"]))

    await asyncio.gather(t1, t2)

    # Both requests should have been served by a single embed() call.
    assert model.calls == 1
    assert model.last_texts == ["a", "b"]

    await batcher.stop()


@pytest.mark.asyncio
async def test_model_batcher_respects_cancel_event_before_enqueue() -> None:
    model = DummyEmbeddingModel()
    batcher = ModelBatcher(model, max_batch=4, window_ms=10, queue_size=8, queue_timeout=0.1)

    cancel_event = threading.Event()
    cancel_event.set()

    with pytest.raises(asyncio.CancelledError):
        await batcher.enqueue(["x"], cancel_event=cancel_event)

    # Cancelled items should be dropped before expensive work is scheduled.
    assert model.calls == 0

    await batcher.stop()

