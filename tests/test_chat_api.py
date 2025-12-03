from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import api
from app.dependencies import get_model_registry
from app.models.base import ChatGeneration

HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_NOT_FOUND = 404


class DummyChatModel:
    def __init__(self, capabilities: list[str]) -> None:
        self.name = "dummy-chat"
        self.device = "cpu"
        self.capabilities = capabilities

    def generate(
        self,
        messages: list[dict[str, object]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None = None,
    ) -> ChatGeneration:
        return ChatGeneration(
            text="hello world",
            prompt_tokens=3,
            completion_tokens=2,
            finish_reason="stop",
        )

    def count_tokens(self, messages: list[dict[str, object]]) -> int:
        return 5


class DummyRegistry:
    def __init__(self, models: dict[str, DummyChatModel]) -> None:
        self._models = models

    def get(self, name: str) -> DummyChatModel:
        if name not in self._models:
            raise KeyError
        return self._models[name]

    def list_models(self) -> list[str]:
        return list(self._models.keys())


def create_app(models: dict[str, DummyChatModel]) -> FastAPI:
    app = FastAPI()
    app.include_router(api.router)
    app.dependency_overrides[get_model_registry] = lambda: DummyRegistry(models)
    return app


def test_chat_completion_basic() -> None:
    client = TestClient(create_app({"dummy-chat": DummyChatModel(["chat-completion", "vision"])}))
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy-chat",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == HTTP_OK
    payload = resp.json()
    assert payload["choices"][0]["message"]["content"] == "hello world"
    assert payload["usage"]["prompt_tokens"] == PROMPT_TOKENS
    assert payload["usage"]["completion_tokens"] == COMPLETION_TOKENS


def test_chat_completion_stream_not_supported() -> None:
    client = TestClient(create_app({"dummy-chat": DummyChatModel(["chat-completion", "vision"])}))
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "dummy-chat", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert resp.status_code == HTTP_BAD_REQUEST


def test_chat_completion_model_not_found() -> None:
    client = TestClient(create_app({"dummy-chat": DummyChatModel(["chat-completion", "vision"])}))
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "missing", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == HTTP_NOT_FOUND


def test_image_rejected_when_model_not_vision() -> None:
    client = TestClient(create_app({"dummy-chat": DummyChatModel(["chat-completion"])}))
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy-chat",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                        {"type": "text", "text": "describe"},
                    ],
                }
            ],
        },
    )
    assert resp.status_code == HTTP_BAD_REQUEST
PROMPT_TOKENS = 3
COMPLETION_TOKENS = 2
