import asyncio
import sys
import types
from typing import Any

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")

    class _DummyCuda:
        OutOfMemoryError = type("CudaOOM", (Exception,), {})

        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def empty_cache() -> None:
            return None

    torch_stub.cuda = _DummyCuda()  # type: ignore[attr-defined]
    torch_stub.OutOfMemoryError = type("TorchOOM", (Exception,), {})  # type: ignore[attr-defined]
    sys.modules["torch"] = torch_stub

if "torchaudio" not in sys.modules:
    torchaudio_stub = types.ModuleType("torchaudio")

    class _DummyInfo:
        def __init__(self) -> None:
            self.num_frames = 0
            self.sample_rate = 0

    torchaudio_stub.info = lambda _path: _DummyInfo()  # type: ignore[attr-defined]
    sys.modules["torchaudio"] = torchaudio_stub

from app import api
from app.chat_batching import ChatBatcher, ChatBatchingService, ChatBatchQueueFullError
from app.dependencies import get_model_registry
from app.models.base import ChatGeneration
from app.monitoring import metrics

HTTP_OK = 200


class DummyChatModel:
    def __init__(self) -> None:
        self.name = "dummy-chat"
        self.device = "cpu"
        self.capabilities = ["chat-completion"]
        self.batched_calls: list[int] = []
        self.generate_calls = 0

    def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        return 5

    def batched_generate(self, batch_messages: list[list[dict[str, Any]]], **_: object) -> list[ChatGeneration]:
        self.batched_calls.append(len(batch_messages))
        return [
            ChatGeneration(text=f"resp-{idx}", prompt_tokens=1, completion_tokens=1)
            for idx, _ in enumerate(batch_messages)
        ]

    def generate(self, messages: list[dict[str, Any]], **_: object) -> ChatGeneration:
        self.generate_calls += 1
        return ChatGeneration(text="solo", prompt_tokens=1, completion_tokens=1)


class DummyRegistry:
    def __init__(self, models: dict[str, DummyChatModel]) -> None:
        self._models = models

    def get(self, name: str) -> DummyChatModel:
        if name not in self._models:
            raise KeyError
        return self._models[name]

    def list_models(self) -> list[str]:
        return list(self._models.keys())


@pytest.mark.asyncio
async def test_chat_batcher_batches_within_window() -> None:
    model = DummyChatModel()
    batcher = ChatBatcher(
        model,
        max_batch=4,
        window_ms=50,
        max_prompt_tokens=32,
        max_new_tokens_ceiling=64,
        queue_size=16,
    )

    t1 = asyncio.create_task(
        batcher.enqueue(
            [{"role": "user", "content": "hi"}],
            max_new_tokens=16,
            temperature=0.7,
            top_p=0.9,
            stop=[],
        )
    )
    await asyncio.sleep(0.01)
    t2 = asyncio.create_task(
        batcher.enqueue(
            [{"role": "user", "content": "there"}],
            max_new_tokens=16,
            temperature=0.7,
            top_p=0.9,
            stop=[],
        )
    )
    await asyncio.gather(t1, t2)
    assert model.batched_calls == [2]
    assert model.generate_calls == 0
    await batcher.stop()


@pytest.mark.asyncio
async def test_chat_batcher_separates_incompatible_configs() -> None:
    model = DummyChatModel()
    batcher = ChatBatcher(
        model,
        max_batch=4,
        window_ms=50,
        max_prompt_tokens=32,
        max_new_tokens_ceiling=64,
        queue_size=16,
    )

    t1 = asyncio.create_task(
        batcher.enqueue(
            [{"role": "user", "content": "hi"}],
            max_new_tokens=16,
            temperature=0.7,
            top_p=0.9,
            stop=[],
        )
    )
    await asyncio.sleep(0.01)
    t2 = asyncio.create_task(
        batcher.enqueue(
            [{"role": "user", "content": "there"}],
            max_new_tokens=16,
            temperature=0.5,  # different temperature, should not batch together
            top_p=0.9,
            stop=[],
        )
    )
    await asyncio.gather(t1, t2)
    # Should have produced two separate batches of size 1
    assert model.batched_calls == [1, 1]
    await batcher.stop()


def test_chat_api_prefers_batcher_when_available() -> None:
    model = DummyChatModel()
    registry = DummyRegistry({"dummy-chat": model})
    chat_batching_service = ChatBatchingService(
        registry,
        enabled=True,
        max_batch_size=4,
        window_ms=0,
        max_prompt_tokens=64,
        max_new_tokens_ceiling=64,
        queue_size=16,
        allow_vision=True,
    )

    app = FastAPI()
    app.include_router(api.router)
    app.state.model_registry = registry
    app.state.chat_batching_service = chat_batching_service
    app.dependency_overrides[get_model_registry] = lambda: registry

    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "dummy-chat", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == HTTP_OK
    assert model.batched_calls == [1]
    assert model.generate_calls == 0

    # Clean up the background worker to avoid leaking tasks
    loop = asyncio.get_event_loop()
    loop.run_until_complete(chat_batching_service.stop())


@pytest.mark.asyncio
async def test_chat_batcher_queue_full_records_metrics() -> None:
    class FailingQueue(asyncio.Queue):
        def put_nowait(self, item: object) -> None:  # noqa: D401
            raise asyncio.QueueFull

    model = DummyChatModel()
    batcher = ChatBatcher(
        model,
        max_batch=1,
        window_ms=0,
        max_prompt_tokens=32,
        max_new_tokens_ceiling=64,
        queue_size=1,
    )
    batcher.queue = FailingQueue(maxsize=1)

    counter = metrics.CHAT_BATCH_QUEUE_REJECTIONS.labels(model=model.name)
    before = counter._value.get()

    with pytest.raises(ChatBatchQueueFullError):
        await batcher.enqueue(
            [{"role": "user", "content": "hi"}],
            max_new_tokens=16,
            temperature=0.7,
            top_p=0.9,
            stop=[],
        )

    after = counter._value.get()
    assert after == before + 1
    await batcher.stop()


def test_chat_api_returns_429_on_batch_queue_full(monkeypatch: pytest.MonkeyPatch) -> None:
    model = DummyChatModel()
    registry = DummyRegistry({"dummy-chat": model})
    chat_batching_service = ChatBatchingService(
        registry,
        enabled=True,
        max_batch_size=1,
        window_ms=0,
        max_prompt_tokens=64,
        max_new_tokens_ceiling=64,
        queue_size=1,
        allow_vision=True,
    )

    batcher = chat_batching_service._batchers["dummy-chat"]

    async def _raise_queue_full(*_: object, **__: object) -> None:
        raise ChatBatchQueueFullError("full")

    monkeypatch.setattr(batcher, "enqueue", _raise_queue_full)

    app = FastAPI()
    app.include_router(api.router)
    app.state.model_registry = registry
    app.state.chat_batching_service = chat_batching_service
    app.dependency_overrides[get_model_registry] = lambda: registry

    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "dummy-chat", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert resp.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert resp.headers.get("Retry-After") == str(int(api.QUEUE_TIMEOUT_SEC))

    # Clean up the background worker to avoid leaking tasks
    loop = asyncio.get_event_loop()
    loop.run_until_complete(chat_batching_service.stop())
