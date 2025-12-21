from __future__ import annotations

import asyncio
import inspect
import json
import logging
import threading
import time
from collections.abc import Sequence
from typing import Annotated, Any, Literal
from uuid import uuid4

import jsonschema
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.chat_batching import ChatBatchQueueFullError, ChatBatchQueueTimeoutError, get_count_executor
from app.concurrency.limiter import (
    CHAT_QUEUE_TIMEOUT_SEC,
    QUEUE_TIMEOUT_SEC,
    VISION_QUEUE_TIMEOUT_SEC,
    QueueFullError,
    QueueTimeoutError,
    ShuttingDownError,
    chat_limiter,
    reset_queue_label,
    set_queue_label,
    vision_limiter,
)
from app.config import settings
from app.dependencies import get_model_registry
from app.models.base import ChatGeneration
from app.models.registry import ModelRegistry
from app.monitoring.metrics import (
    observe_chat_latency,
    record_chat_request,
)
from app.routes.common import (
    _ClientDisconnectedError,
    _RequestCancelledError,
    _run_work_with_client_cancel,
    _WorkTimeoutError,
)
from app.threadpool import get_chat_executor, get_vision_executor

logger = logging.getLogger(__name__)
router = APIRouter()

_STRUCTURED_OUTPUT_ERROR_MAX_CHARS = 400
_STRUCTURED_OUTPUT_PREVIOUS_OUTPUT_MAX_CHARS = 4000


class ImageURL(BaseModel):
    url: str
    detail: Literal["auto", "low", "high"] | None = Field(default=None)


class ChatContentPart(BaseModel):
    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: ImageURL | None = None

    @staticmethod
    def _assert_valid(part: ChatContentPart) -> ChatContentPart:
        if part.type == "text" and part.text is None:
            raise ValueError("text content part requires 'text'")
        if part.type == "image_url" and (part.image_url is None or not part.image_url.url):
            raise ValueError("image_url content part requires 'image_url.url'")
        return part

    def model_post_init(self, __context: object) -> None:  # noqa: D401
        self._assert_valid(self)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str | list[ChatContentPart]


class ChatCompletionMessage(BaseModel):
    role: Literal["assistant"]
    content: str


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: str = "stop"


class ChatResponseFormatJsonSchema(BaseModel):
    """OpenAI-compatible json_schema response_format payload."""

    model_config = ConfigDict(populate_by_name=True)

    name: str | None = None
    schema_: dict[str, Any] = Field(alias="schema")
    strict: bool | None = None


class ChatResponseFormat(BaseModel):
    """OpenAI-compatible response_format for chat completions.

    Supported:
      - {"type": "text"} (default)
      - {"type": "json_object"} (best-effort; server enforces valid JSON object)
      - {"type": "json_schema", "json_schema": {"name": "...", "schema": {...}, "strict": true}}
    """

    type: Literal["text", "json_object", "json_schema"]
    json_schema: ChatResponseFormatJsonSchema | None = None
    # Compatibility shim: some clients may send `strict` alongside `type`, but
    # OpenAI's canonical shape nests it under `json_schema`.
    strict: bool | None = None

    @model_validator(mode="after")
    def _validate_payload(self) -> ChatResponseFormat:  # noqa: D401
        if self.type == "json_schema" and self.json_schema is None:
            raise ValueError("response_format.json_schema is required when type='json_schema'")
        if self.type != "json_schema" and self.json_schema is not None:
            raise ValueError("response_format.json_schema is only allowed when type='json_schema'")
        return self


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None, description="Max new tokens to generate")
    temperature: float | None = None
    top_p: float | None = None
    n: int = 1
    stream: bool = False
    stop: str | list[str] | None = None
    response_format: ChatResponseFormat | None = None
    user: str | None = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: int | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


def _contains_image_content(messages: Sequence[dict[str, Any]]) -> bool:
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


def _normalize_stop(stop: str | list[str] | None) -> list[str] | None:
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    return [s for s in stop if s]


def _effective_json_schema_strict(response_format: ChatResponseFormat) -> bool:
    if response_format.type != "json_schema":
        return False
    if response_format.json_schema is not None and response_format.json_schema.strict is not None:
        return response_format.json_schema.strict
    return bool(response_format.strict)


def _structured_output_system_prompt(response_format: ChatResponseFormat) -> str:
    """Build a system instruction for OpenAI-style structured output."""

    if response_format.type == "json_object":
        return (
            "You MUST respond with a single JSON object.\n"
            "- Output ONLY valid JSON (no markdown, no code fences).\n"
            "- Do not include any explanatory text.\n"
        )
    if response_format.type == "json_schema":
        if response_format.json_schema is None:
            return ""
        schema_json = json.dumps(response_format.json_schema.schema_, ensure_ascii=False, separators=(",", ":"))
        name = response_format.json_schema.name or "response"
        strict = _effective_json_schema_strict(response_format)
        strict_label = "true" if strict else "false"
        return (
            "You MUST respond with a single JSON value that matches the provided JSON Schema.\n"
            "- Output ONLY valid JSON (no markdown, no code fences).\n"
            "- Do not include any explanatory text.\n"
            f"- schema_name: {name}\n"
            f"- strict: {strict_label}\n"
            f"JSON Schema: {schema_json}\n"
        )
    return ""


def _inject_system_message(messages: list[dict[str, Any]], instruction: str) -> list[dict[str, Any]]:
    """Insert a system instruction after any existing system messages."""

    if not instruction:
        return messages

    prefix: list[dict[str, Any]] = []
    idx = 0
    while idx < len(messages) and messages[idx].get("role") == "system":
        prefix.append(messages[idx])
        idx += 1

    return [*prefix, {"role": "system", "content": instruction}, *messages[idx:]]


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    # Drop the opening fence line (e.g. ```json).
    newline = stripped.find("\n")
    if newline != -1:
        stripped = stripped[newline + 1 :]

    # Drop the trailing fence.
    stripped = stripped.strip()
    if stripped.endswith("```"):
        stripped = stripped[: stripped.rfind("```")].strip()
    return stripped


def _extract_first_json_value(text: str) -> Any:
    """Extract the first JSON value from text (tolerant of code fences and leading prose)."""

    cleaned = _strip_code_fences(text)
    start = next((i for i, ch in enumerate(cleaned) if ch in "{["), None)
    if start is None:
        raise ValueError("Model output did not contain a JSON value")

    candidate = cleaned[start:]
    decoder = json.JSONDecoder()
    try:
        value, _end = decoder.raw_decode(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc.msg}") from exc
    return value


def _validate_json_schema_strict(instance: Any, schema: dict[str, Any]) -> None:
    try:
        jsonschema.validate(instance=instance, schema=schema)
    except jsonschema.SchemaError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid json_schema.schema: {exc}",
        ) from exc
    except jsonschema.ValidationError as exc:
        raise ValueError(f"JSON does not match schema: {exc.message}") from exc


def _coerce_structured_output(text: str, response_format: ChatResponseFormat) -> str:
    value = _extract_first_json_value(text)

    if response_format.type == "json_object":
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object for response_format.type='json_object'")
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    if response_format.type == "json_schema":
        if response_format.json_schema is None:
            raise ValueError("Missing response_format.json_schema")
        if _effective_json_schema_strict(response_format):
            _validate_json_schema_strict(value, response_format.json_schema.schema_)
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    return text.strip()


def _build_structured_output_retry_messages(
    base_messages: list[dict[str, Any]],
    *,
    last_output: str,
    error: str,
) -> list[dict[str, Any]]:
    error_msg = error.strip()
    if len(error_msg) > _STRUCTURED_OUTPUT_ERROR_MAX_CHARS:
        error_msg = f"{error_msg[:_STRUCTURED_OUTPUT_ERROR_MAX_CHARS]}..."

    previous = last_output.strip()
    if len(previous) > _STRUCTURED_OUTPUT_PREVIOUS_OUTPUT_MAX_CHARS:
        previous = f"{previous[:_STRUCTURED_OUTPUT_PREVIOUS_OUTPUT_MAX_CHARS]}..."

    return [
        *base_messages,
        {"role": "assistant", "content": previous},
        {
            "role": "user",
            "content": (f"Your previous response was invalid. Error: {error_msg}. Return corrected JSON only."),
        },
    ]


def _resolve_generation_params(
    req: ChatCompletionRequest,
    model: Any,
) -> tuple[int, float, float]:
    defaults = getattr(model, "generation_defaults", {}) or {}
    max_tokens_default = defaults.get("max_tokens") or settings.max_new_tokens
    temperature_default = defaults.get("temperature", 0.7)
    top_p_default = defaults.get("top_p", 0.9)

    max_tokens = req.max_tokens or max_tokens_default
    temperature = req.temperature if req.temperature is not None else temperature_default
    top_p = req.top_p if req.top_p is not None else top_p_default

    return max_tokens, temperature, top_p


def _build_generation_kwargs(  # noqa: PLR0913
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: list[str] | None,
    cancel_event: threading.Event | None,
    accepts_cancel: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop": stop,
    }
    if accepts_cancel and cancel_event is not None:
        kwargs["cancel_event"] = cancel_event
    return kwargs


async def _prepare_chat_request(
    model: Any,
    messages: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, int]:
    loop = asyncio.get_running_loop()
    prepare_timeout = settings.chat_prepare_timeout_sec
    count_executor = get_count_executor(use_chat_executor=settings.chat_count_use_chat_executor)
    if hasattr(model, "prepare_inputs"):
        try:
            prepared, prompt_tokens = await asyncio.wait_for(
                loop.run_in_executor(
                    count_executor,
                    lambda: model.prepare_inputs(messages, add_generation_prompt=True),
                ),
                timeout=prepare_timeout,
            )
            return prepared, prompt_tokens
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "chat_prepare_inputs_failed",
                extra={"model": getattr(model, "name", "unknown"), "error": str(exc)},
            )

    prompt_tokens = await asyncio.wait_for(
        loop.run_in_executor(count_executor, lambda: model.count_tokens(messages)),
        timeout=prepare_timeout,
    )
    return None, int(prompt_tokens)


async def _run_chat_generation(  # noqa: PLR0915
    *,
    req: ChatCompletionRequest,
    registry: ModelRegistry,
    request: Request,
    raw_messages: list[dict[str, Any]],
    has_images: bool,
) -> tuple[ChatGeneration, int, int]:
    model = _resolve_chat_model_and_caps(registry, req.model, has_images=has_images)
    max_tokens, temperature, top_p = _resolve_generation_params(req, model)

    if max_tokens <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="max_tokens must be positive",
        )

    loop = asyncio.get_running_loop()
    executor = get_vision_executor() if has_images else get_chat_executor()
    max_prompt_tokens = settings.chat_max_prompt_tokens
    prepared_inputs, prompt_tokens = await _prepare_chat_request(model, raw_messages)
    if prompt_tokens > max_prompt_tokens:
        record_chat_request(req.model, "400")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prompt too long; max {max_prompt_tokens} tokens",
        )
    cancel_event = threading.Event()
    gen_timeout = settings.chat_generate_timeout_sec
    generate_accepts_cancel = "cancel_event" in inspect.signature(model.generate).parameters
    generate_prepared_accepts_cancel = (
        hasattr(model, "generate_prepared") and "cancel_event" in inspect.signature(model.generate_prepared).parameters
    )
    batcher = getattr(request.app.state, "chat_batching_service", None)
    stop = _normalize_stop(req.stop)

    async def _run_generation() -> ChatGeneration:
        if prepared_inputs is not None and hasattr(model, "generate_prepared"):
            kwargs = _build_generation_kwargs(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                cancel_event=cancel_event,
                accepts_cancel=generate_prepared_accepts_cancel,
            )
            return await loop.run_in_executor(
                executor,
                lambda: model.generate_prepared(prepared_inputs, **kwargs),
            )

        kwargs = _build_generation_kwargs(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            cancel_event=cancel_event,
            accepts_cancel=generate_accepts_cancel,
        )
        return await loop.run_in_executor(
            executor,
            lambda: model.generate(raw_messages, **kwargs),
        )

    async def _run_batched_or_fallback() -> ChatGeneration:
        if batcher is not None and getattr(batcher, "is_supported", lambda _m: False)(req.model) and not has_images:
            try:
                return await batcher.enqueue(
                    req.model,
                    raw_messages,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    prompt_tokens=prompt_tokens,
                    prepared_inputs=prepared_inputs,
                    cancel_event=cancel_event,
                )
            except ValueError as exc:
                record_chat_request(req.model, "400")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(exc),
                ) from exc
            except ChatBatchQueueFullError as exc:
                record_chat_request(req.model, "429")
                logger.info("chat_batch_queue_full", extra={"model": req.model, "status": 429})
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Chat batch queue full",
                    headers={"Retry-After": str(int(QUEUE_TIMEOUT_SEC))},
                ) from exc
            except ChatBatchQueueTimeoutError as exc:
                record_chat_request(req.model, "429")
                logger.info("chat_batch_queue_timeout", extra={"model": req.model, "status": 429})
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Chat batch queue wait exceeded",
                    headers={"Retry-After": str(int(QUEUE_TIMEOUT_SEC))},
                ) from exc
            except QueueFullError as exc:
                record_chat_request(req.model, "429")
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Chat request queue full",
                    headers={"Retry-After": str(int(CHAT_QUEUE_TIMEOUT_SEC))},
                ) from exc
            except QueueTimeoutError as exc:
                record_chat_request(req.model, "429")
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Timed out waiting for chat worker",
                    headers={"Retry-After": str(int(CHAT_QUEUE_TIMEOUT_SEC))},
                ) from exc
            except ShuttingDownError as exc:
                record_chat_request(req.model, "503")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Service is shutting down",
                ) from exc
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning(
                    "chat_batcher_failed_falling_back",
                    extra={"model": req.model, "error": str(exc)},
                )
                label_token = set_queue_label(req.model or "chat")
                try:
                    async with chat_limiter():
                        return await _run_generation()
                finally:
                    reset_queue_label(label_token)

        return await _run_generation()

    work_task = asyncio.ensure_future(_run_batched_or_fallback())
    try:
        generation = await _run_work_with_client_cancel(
            request=request,
            work_task=work_task,
            cancel_event=cancel_event,
            timeout=gen_timeout,
        )
    except _WorkTimeoutError as exc:
        cancel_event.set()
        record_chat_request(req.model, "504")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Chat generation timed out",
        ) from exc
    except _RequestCancelledError as exc:
        cancel_event.set()
        record_chat_request(req.model, "499")
        raise HTTPException(
            status_code=status.HTTP_499_CLIENT_CLOSED_REQUEST,
            detail="Request cancelled",
        ) from exc
    except _ClientDisconnectedError as exc:
        cancel_event.set()
        record_chat_request(req.model, "499")
        raise HTTPException(
            status_code=status.HTTP_499_CLIENT_CLOSED_REQUEST,
            detail="Client disconnected",
        ) from exc
    except (QueueFullError, QueueTimeoutError, ShuttingDownError):
        raise
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - unexpected runtime failure
        cancel_event.set()
        record_chat_request(req.model, "500")
        logger.exception(
            "chat_generation_failed",
            extra={"model": req.model, "max_tokens": max_tokens, "status": 500},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chat generation failed",
        ) from exc

    return generation, int(prompt_tokens), int(max_tokens)


def _resolve_chat_model_and_caps(
    registry: ModelRegistry,
    model_name: str,
    *,
    has_images: bool,
) -> Any:
    try:
        model = registry.get(model_name)
    except KeyError as exc:
        record_chat_request(model_name, "404")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_name} not found",
        ) from exc

    capabilities = getattr(model, "capabilities", [])
    if "chat-completion" not in capabilities:
        record_chat_request(model_name, "400")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model {model_name} does not support chat/completions",
        )
    if has_images and "vision" not in capabilities:
        record_chat_request(model_name, "400")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model {model_name} does not support image inputs",
        )

    return model


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completions(  # noqa: PLR0912, PLR0915
    req: ChatCompletionRequest,
    registry: Annotated[ModelRegistry, Depends(get_model_registry)],
    _request: Request,
) -> ChatCompletionResponse:
    if req.stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Streaming responses are not supported yet",
        )
    if req.n != 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only n=1 is supported",
        )

    start = time.perf_counter()
    label_token = None
    has_images = False
    try:
        raw_messages = [msg.model_dump(mode="python") for msg in req.messages]
        has_images = _contains_image_content(raw_messages)

        batcher = getattr(_request.app.state, "chat_batching_service", None)
        use_batcher = bool(
            batcher is not None and getattr(batcher, "is_supported", lambda _m: False)(req.model) and not has_images
        )

        response_format = req.response_format
        base_messages = raw_messages
        if response_format is not None and response_format.type != "text":
            structured_output = True
            base_messages = _inject_system_message(raw_messages, _structured_output_system_prompt(response_format))
        else:
            structured_output = False

        if not use_batcher:
            label_token = set_queue_label(req.model or "chat")

        attempt_messages = base_messages
        max_retries = settings.chat_structured_output_max_retries if structured_output else 0

        for attempt in range(max_retries + 1):
            if not use_batcher:
                limiter = vision_limiter if has_images else chat_limiter
                async with limiter():
                    generation, prompt_tokens, max_tokens = await _run_chat_generation(
                        req=req,
                        registry=registry,
                        request=_request,
                        raw_messages=attempt_messages,
                        has_images=has_images,
                    )
            else:
                generation, prompt_tokens, max_tokens = await _run_chat_generation(
                    req=req,
                    registry=registry,
                    request=_request,
                    raw_messages=attempt_messages,
                    has_images=has_images,
                )

            if not structured_output:
                break

            try:
                if response_format is None:  # pragma: no cover - defensive
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Internal error: response_format missing",
                    )
                generation.text = _coerce_structured_output(generation.text, response_format)
                break
            except ValueError as exc:
                if attempt >= max_retries:
                    record_chat_request(req.model, "500")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Structured output validation failed: {exc}",
                    ) from exc
                attempt_messages = _build_structured_output_retry_messages(
                    base_messages,
                    last_output=generation.text,
                    error=str(exc),
                )
    except QueueFullError as exc:
        record_chat_request(req.model, "429")
        retry_after = VISION_QUEUE_TIMEOUT_SEC if has_images else CHAT_QUEUE_TIMEOUT_SEC
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Vision request queue full" if has_images else "Chat request queue full",
            headers={"Retry-After": str(int(retry_after))},
        ) from exc
    except ShuttingDownError as exc:
        record_chat_request(req.model, "503")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is shutting down",
        ) from exc
    except QueueTimeoutError as exc:
        record_chat_request(req.model, "429")
        retry_after = VISION_QUEUE_TIMEOUT_SEC if has_images else CHAT_QUEUE_TIMEOUT_SEC
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Timed out waiting for vision worker" if has_images else "Timed out waiting for chat worker",
            headers={"Retry-After": str(int(retry_after))},
        ) from exc
    finally:
        if label_token is not None:
            reset_queue_label(label_token)

    latency = time.perf_counter() - start
    observe_chat_latency(req.model, latency)
    record_chat_request(req.model, "200")
    logger.info(
        "chat_request",
        extra={
            "model": req.model,
            "latency_ms": round(latency * 1000, 2),
            "status": 200,
            "max_tokens": max_tokens,
        },
    )

    completion_tokens = generation.completion_tokens or 0
    prompt_tokens = generation.prompt_tokens if generation.prompt_tokens is not None else prompt_tokens
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    choice = ChatCompletionChoice(
        index=0,
        message=ChatCompletionMessage(role="assistant", content=generation.text),
        finish_reason=generation.finish_reason or "stop",
    )
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid4().hex}",
        created=int(time.time()),
        model=req.model,
        choices=[choice],
        usage=usage,
    )
