from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import api
from app.dependencies import get_model_registry
from app.models.base import ChatGeneration

HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_NOT_FOUND = 404
HTTP_SERVER_ERROR = 500


class DummyChatModel:
    def __init__(self, capabilities: list[str], *, supports_structured_outputs: bool = False) -> None:
        self.name = "dummy-chat"
        self.device = "cpu"
        self.capabilities = capabilities
        self.supports_structured_outputs = supports_structured_outputs

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


class FixedResponseChatModel(DummyChatModel):
    def __init__(
        self,
        capabilities: list[str],
        *,
        response_text: str,
        supports_structured_outputs: bool = False,
    ) -> None:
        super().__init__(capabilities, supports_structured_outputs=supports_structured_outputs)
        self._response_text = response_text

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
            text=self._response_text,
            prompt_tokens=3,
            completion_tokens=2,
            finish_reason="stop",
        )


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


def test_chat_completion_response_format_rejected_when_model_not_supports_structured_outputs() -> None:
    client = TestClient(
        create_app({"dummy-chat": DummyChatModel(["chat-completion", "vision"], supports_structured_outputs=False)})
    )
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy-chat",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_object"},
        },
    )
    assert resp.status_code == HTTP_BAD_REQUEST
    assert "does not support response_format" in resp.json()["detail"]


def test_chat_completion_response_format_json_object_success() -> None:
    client = TestClient(
        create_app(
            {
                "dummy-chat": FixedResponseChatModel(
                    ["chat-completion", "vision"],
                    response_text='{"ok":true}',
                    supports_structured_outputs=True,
                )
            }
        )
    )
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy-chat",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_object"},
        },
    )
    assert resp.status_code == HTTP_OK
    payload = resp.json()
    assert payload["choices"][0]["message"]["content"] == '{"ok":true}'


def test_chat_completion_response_format_json_object_invalid_json() -> None:
    client = TestClient(
        create_app(
            {
                "dummy-chat": FixedResponseChatModel(
                    ["chat-completion", "vision"],
                    response_text="not json",
                    supports_structured_outputs=True,
                )
            }
        )
    )
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy-chat",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_object"},
        },
    )
    assert resp.status_code >= HTTP_SERVER_ERROR


def test_chat_completion_response_format_json_schema_strict_success() -> None:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    client = TestClient(
        create_app(
            {
                "dummy-chat": FixedResponseChatModel(
                    ["chat-completion", "vision"],
                    response_text='{"answer":"ok"}',
                    supports_structured_outputs=True,
                )
            }
        )
    )
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy-chat",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema, "strict": True},
            },
        },
    )
    assert resp.status_code == HTTP_OK
    payload = resp.json()
    assert payload["choices"][0]["message"]["content"] == '{"answer":"ok"}'


def test_chat_completion_response_format_json_schema_strict_validation_error() -> None:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    client = TestClient(
        create_app(
            {
                "dummy-chat": FixedResponseChatModel(
                    ["chat-completion", "vision"],
                    response_text='{"answer":1}',
                    supports_structured_outputs=True,
                )
            }
        )
    )
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy-chat",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema, "strict": True},
            },
        },
    )
    assert resp.status_code >= HTTP_SERVER_ERROR


PROMPT_TOKENS = 3
COMPLETION_TOKENS = 2
