from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import api
from app.concurrency.limiter import QueueFullError
from app.dependencies import get_model_registry
from app.models.base import RerankResult
from app.routes import rerank as rerank_module
from app.routes.common import _WorkTimeoutError

HTTP_OK = 200
HTTP_NOT_FOUND = 404
HTTP_BAD_REQUEST = 400
HTTP_TOO_MANY = 429
HTTP_GATEWAY_TIMEOUT = 504
EXPECTED_TWO = 2


class DummyRerankModel:
    name = "dummy-rerank"
    capabilities = ["rerank"]

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int | None = None,
        cancel_event: Any | None = None,
    ) -> list[RerankResult]:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("cancelled")
        scores = list(range(len(documents)))
        ranked = list(enumerate(scores))
        if top_k is not None:
            ranked = ranked[:top_k]
        return [RerankResult(index=idx, relevance_score=float(score), document=documents[idx]) for idx, score in ranked]


class DummyRegistry:
    def get(self, name: str) -> DummyRerankModel:
        if name != "dummy-rerank":
            raise KeyError
        return DummyRerankModel()

    def list_models(self) -> list[str]:
        return ["dummy-rerank"]


def create_app() -> FastAPI:
    app = FastAPI()
    app.include_router(api.router)
    app.dependency_overrides[get_model_registry] = lambda: DummyRegistry()
    return app


def test_rerank_success() -> None:
    client = TestClient(create_app())
    resp = client.post(
        "/v1/rerank",
        json={"model": "dummy-rerank", "query": "q", "documents": ["a", "b", "c"], "top_n": 2},
    )
    assert resp.status_code == HTTP_OK
    body = resp.json()
    assert body["model"] == "dummy-rerank"
    assert len(body["results"]) == EXPECTED_TWO
    assert body["results"][0]["relevance_score"] == 0.0


def test_rerank_model_not_found() -> None:
    client = TestClient(create_app())
    resp = client.post("/v1/rerank", json={"model": "missing", "query": "q", "documents": ["a"]})
    assert resp.status_code == HTTP_NOT_FOUND


def test_rerank_missing_capability() -> None:
    client = TestClient(create_app())

    class NoRerankModel:
        name = "dummy-rerank"
        capabilities: list[str] = []

        def rerank(self, *args: Any, **kwargs: Any) -> list[RerankResult]:
            return []

    def _get_missing_cap_model() -> Any:
        class _Registry(DummyRegistry):
            def get(self, name: str) -> Any:
                if name != "dummy-rerank":
                    raise KeyError
                return NoRerankModel()

        return _Registry()

    client.app.dependency_overrides[get_model_registry] = lambda: _get_missing_cap_model()
    resp = client.post("/v1/rerank", json={"model": "dummy-rerank", "query": "q", "documents": ["a"]})
    assert resp.status_code == HTTP_BAD_REQUEST


def test_rerank_queue_full(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(create_app())

    @asynccontextmanager
    async def fail_limiter() -> AsyncIterator[None]:
        raise QueueFullError
        yield

    monkeypatch.setattr(api, "embedding_limiter", fail_limiter)
    resp = client.post("/v1/rerank", json={"model": "dummy-rerank", "query": "q", "documents": ["a"]})
    assert resp.status_code == HTTP_TOO_MANY


def test_rerank_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(create_app())

    async def _timeout(*args: Any, **kwargs: Any) -> Any:
        raise _WorkTimeoutError()

    monkeypatch.setattr(rerank_module, "_run_work_with_client_cancel", _timeout)
    resp = client.post("/v1/rerank", json={"model": "dummy-rerank", "query": "q", "documents": ["a"]})
    assert resp.status_code == HTTP_GATEWAY_TIMEOUT
