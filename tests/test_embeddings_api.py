from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import api
from app.concurrency.limiter import QueueFullError
from app.dependencies import get_model_registry

HTTP_OK = 200
HTTP_NOT_FOUND = 404
HTTP_BAD_REQUEST = 400
HTTP_TOO_MANY = 429
BATCH_TWO = 2


class DummyModel:
    name = "dummy"
    dim = 3

    def embed(self, texts: list[str], cancel_event: object | None = None) -> np.ndarray:
        arr = np.array([[1.0, 2.0, 3.0]])
        return np.repeat(arr, len(texts), axis=0)

    def count_tokens(self, texts: list[str]) -> int:
        return sum(len(t) for t in texts)


class DummyRegistry:
    def get(self, name: str) -> DummyModel:
        if name != "dummy":
            raise KeyError
        return DummyModel()

    def list_models(self) -> list[str]:
        return ["dummy"]


def create_app() -> FastAPI:
    app = FastAPI()
    app.include_router(api.router)
    app.dependency_overrides[get_model_registry] = lambda: DummyRegistry()
    return app


def test_single_text_embedding() -> None:
    client = TestClient(create_app())
    resp = client.post("/v1/embeddings", json={"model": "dummy", "input": "hello"})
    assert resp.status_code == HTTP_OK
    payload = resp.json()
    assert payload["model"] == "dummy"
    assert len(payload["data"]) == 1
    assert payload["data"][0]["embedding"] == [1.0, 2.0, 3.0]


def test_batch_text_embedding() -> None:
    client = TestClient(create_app())
    resp = client.post(
        "/v1/embeddings", json={"model": "dummy", "input": ["a", "b"]}
    )
    assert resp.status_code == HTTP_OK
    payload = resp.json()
    assert len(payload["data"]) == BATCH_TWO
    assert payload["data"][1]["embedding"] == [1.0, 2.0, 3.0]


def test_model_not_found_returns_404() -> None:
    client = TestClient(create_app())
    resp = client.post("/v1/embeddings", json={"model": "missing", "input": "x"})
    assert resp.status_code == HTTP_NOT_FOUND


def test_invalid_encoding_format_returns_400() -> None:
    client = TestClient(create_app())
    resp = client.post(
        "/v1/embeddings",
        json={"model": "dummy", "input": "x", "encoding_format": "base64"},
    )
    assert resp.status_code == HTTP_BAD_REQUEST


def test_queue_full_returns_429(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(create_app())

    # Force embedding_limiter to raise QueueFullError immediately
    @asynccontextmanager
    async def fail_limiter() -> AsyncIterator[None]:
        raise QueueFullError
        yield

    monkeypatch.setattr(api, "embedding_limiter", fail_limiter)

    resp = client.post("/v1/embeddings", json={"model": "dummy", "input": "x"})
    assert resp.status_code == HTTP_TOO_MANY


def test_batch_too_large(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_BATCH_SIZE", "1")
    client = TestClient(create_app())
    resp = client.post("/v1/embeddings", json={"model": "dummy", "input": ["a", "b"]})
    assert resp.status_code == HTTP_BAD_REQUEST


def test_text_too_long(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_TEXT_CHARS", "5")
    client = TestClient(create_app())
    resp = client.post("/v1/embeddings", json={"model": "dummy", "input": "123456"})
    assert resp.status_code == HTTP_BAD_REQUEST


def test_list_models_returns_loaded_models() -> None:
    client = TestClient(create_app())
    resp = client.get("/v1/models")
    assert resp.status_code == HTTP_OK
    payload = resp.json()
    assert payload["object"] == "list"
    assert len(payload["data"]) == 1
    assert payload["data"][0]["id"] == "dummy"
    assert payload["data"][0]["embedding_dimensions"] == DummyModel.dim
