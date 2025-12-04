import asyncio
import contextlib
import logging
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import uuid4

import torchaudio
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

from app.chat_batching import ChatBatchQueueFullError
from app.chat_batching import ChatBatchQueueTimeoutError
from app.concurrency.audio_limiter import (
    QUEUE_TIMEOUT_SEC as AUDIO_QUEUE_TIMEOUT_SEC,
    AudioQueueFullError,
    AudioQueueTimeoutError,
    AudioShuttingDownError,
    reset_queue_label as reset_audio_queue_label,
    set_queue_label as set_audio_queue_label,
    limiter as audio_limiter,
)
from app.concurrency.limiter import (
    QUEUE_TIMEOUT_SEC,
    QueueFullError,
    QueueTimeoutError,
    ShuttingDownError,
    reset_queue_label,
    set_queue_label,
    limiter,
)
from app.dependencies import get_model_registry
from app.models.base import ChatGeneration
from app.models.registry import ModelRegistry
from app.monitoring.metrics import (
    observe_audio_latency,
    observe_chat_latency,
    observe_latency,
    record_audio_request,
    record_chat_request,
    record_request,
)
from app.state import WarmupStatus, warmup_status as global_warmup_status
from app.threadpool import get_audio_executor, get_chat_executor, get_embedding_executor
from app.utils.uploads import chunked_upload_to_tempfile

router = APIRouter()
logger = logging.getLogger(__name__)
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(25 * 1024 * 1024)))  # default 25MB
UPLOAD_CHUNK_BYTES = 1024 * 1024  # 1MB chunks
_chat_generation_locks: dict[str, asyncio.Lock] = {}


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: str | None = Field(default="float", description="Only 'float' is supported")
    user: str | None = Field(default=None, description="OpenAI compatibility placeholder")


class EmbeddingObject(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float]


class Usage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: int | None = None


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingObject]
    model: str
    usage: Usage


# ---- Chat completions (OpenAI compatible) ------------------------------------


class ImageURL(BaseModel):
    url: str
    detail: Literal["auto", "low", "high"] | None = Field(default=None)


class ChatContentPart(BaseModel):
    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: ImageURL | None = None

    @staticmethod
    def _assert_valid(part: "ChatContentPart") -> "ChatContentPart":
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


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None, description="Max new tokens to generate")
    temperature: float | None = None
    top_p: float | None = None
    n: int = 1
    stream: bool = False
    stop: str | list[str] | None = None
    user: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


# ---- Audio transcription/translation (Whisper-compatible) --------------------


class TranscriptionSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str


class TranscriptionVerboseResponse(BaseModel):
    text: str
    language: str | None = None
    duration: float | None = None
    segments: list[TranscriptionSegment] | None = None


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


async def _prepare_chat_request(
    model: Any,
    messages: list[dict[str, Any]],
    executor: Any,
) -> tuple[dict[str, Any] | None, int]:
    """Tokenize once for both counting and generation when supported by the model.

    Runs heavy tokenizer work inside the chat executor to avoid blocking the event loop.
    """

    loop = asyncio.get_running_loop()
    prepare_timeout = float(os.getenv("CHAT_PREPARE_TIMEOUT_SEC", "10"))
    if hasattr(model, "prepare_inputs"):
        try:
            prepared, prompt_tokens = await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
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
        loop.run_in_executor(executor, lambda: model.count_tokens(messages)),
        timeout=prepare_timeout,
    )
    return None, int(prompt_tokens)


async def _save_upload(file: UploadFile, max_bytes: int = MAX_AUDIO_BYTES) -> tuple[str, int]:
    """Persist UploadFile to a temp file and enforce size guard."""
    suffix = Path(file.filename or "").suffix or ".wav"
    return await chunked_upload_to_tempfile(
        file,
        chunk_size=UPLOAD_CHUNK_BYTES,
        max_bytes=max_bytes,
        suffix=suffix,
        on_exceed=lambda limit, size: HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio file too large; max {limit} bytes",
        ),
    )


def _probe_duration(path: str) -> float | None:
    try:
        info = torchaudio.info(path)
        if info.num_frames and info.sample_rate:
            return float(info.num_frames) / float(info.sample_rate)
    except Exception:  # pragma: no cover - best effort only
        return None
    return None


def _resolve_warmup_status(request: Request | None) -> WarmupStatus:
    if request is not None:
        status_obj = getattr(request.app.state, "warmup_status", None)
        if isinstance(status_obj, WarmupStatus):
            return status_obj

    if isinstance(global_warmup_status, WarmupStatus):
        return global_warmup_status

    return WarmupStatus()


def _select_granularity(values: list[str] | None) -> Literal["word", "segment", None]:
    if not values:
        return None
    lowered = [v.lower() for v in values]
    if "word" in lowered:
        return "word"
    if "segment" in lowered:
        return "segment"
    return None


def _get_chat_lock(model: str) -> asyncio.Lock:
    lock = _chat_generation_locks.get(model)
    if lock is None:
        lock = asyncio.Lock()
        _chat_generation_locks[model] = lock
    return lock


def _format_ts(seconds: float, *, sep: str = ",") -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", sep)


def _srt_from_segments(segments: list[dict[str, float | str]]) -> str:
    lines: list[str] = []
    for idx, seg in enumerate(segments, start=1):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        text = str(seg.get("text", "")).strip()
        lines.append(str(idx))
        lines.append(f"{_format_ts(start)} --> {_format_ts(end)}")
        lines.append(text)
        lines.append("")  # blank line between entries
    return "\n".join(lines).rstrip() + "\n"


def _vtt_from_segments(segments: list[dict[str, float | str]]) -> str:
    lines: list[str] = ["WEBVTT", ""]
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        text = str(seg.get("text", "")).strip()
        lines.append(f"{_format_ts(start, sep='.')} --> {_format_ts(end, sep='.')}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "local"
    embedding_dimensions: int | None = None


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class WarmupDetails(BaseModel):
    required: bool
    completed: bool
    failures: list[str] | None = None
    ok_models: list[str] | None = None
    capabilities: dict[str, dict[str, bool]] | None = None


class QueueDepth(BaseModel):
    model: str
    size: int
    max_size: int | None = None


class HealthResponse(BaseModel):
    status: str
    models: list[str] | None = None
    warmup_failures: list[str] | None = None
    warmup: WarmupDetails | None = None
    chat_batch_queues: list[QueueDepth] | None = None
    embedding_batch_queues: list[QueueDepth] | None = None


@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    req: EmbeddingRequest,
    registry: Annotated[ModelRegistry, Depends(get_model_registry)],
    request: Request,
) -> EmbeddingResponse:
    if req.encoding_format not in (None, "float"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only 'float' encoding_format is supported",
        )

    texts = [req.input] if isinstance(req.input, str) else list(req.input)
    max_batch = int(os.getenv("MAX_BATCH_SIZE", "32"))
    if len(texts) > max_batch:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch too large; max {max_batch} items",
        )

    max_text_chars = int(os.getenv("MAX_TEXT_CHARS", "20000"))
    for idx, t in enumerate(texts):
        if len(t) > max_text_chars:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Input at index {idx} exceeds max length {max_text_chars} chars",
            )

    start = time.perf_counter()
    label_token = set_queue_label(req.model or "embedding")
    try:
        async with limiter():
            try:
                model = registry.get(req.model)
            except KeyError as exc:
                record_request(req.model, "404")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model {req.model} not found",
                ) from exc

            try:
                batcher = getattr(request.app.state, "batching_service", None)
                if batcher is not None and getattr(batcher, "enabled", False):
                    vectors = await batcher.enqueue(req.model, texts)
                else:
                    loop = asyncio.get_running_loop()
                    executor = get_embedding_executor()
                    vectors = await loop.run_in_executor(executor, model.embed, texts)
            except Exception as exc:  # pragma: no cover - unexpected runtime failure
                record_request(req.model, "500")
                logger.exception(
                    "embedding_failed",
                    extra={
                        "model": req.model,
                        "batch_size": len(texts),
                        "device": getattr(model, "device", None),
                    },
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Embedding generation failed",
                ) from exc
    except QueueFullError as exc:
        record_request(req.model, "429")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Request queue full",
            headers={"Retry-After": str(int(QUEUE_TIMEOUT_SEC))},
        ) from exc
    except ShuttingDownError as exc:
        record_request(req.model, "503")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is shutting down",
        ) from exc
    except QueueTimeoutError as exc:
        record_request(req.model, "429")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Timed out waiting for worker",
            headers={"Retry-After": str(int(QUEUE_TIMEOUT_SEC))},
        ) from exc
    finally:
        reset_queue_label(label_token)

    latency = time.perf_counter() - start
    observe_latency(req.model, latency)
    record_request(req.model, "200")
    logger.info(
        "embedding_request",
        extra={
            "model": req.model,
            "latency_ms": round(latency * 1000, 2),
            "batch_size": len(texts),
            "status": 200,
        },
    )

    data = [
        EmbeddingObject(index=i, embedding=vec.tolist()) for i, vec in enumerate(vectors)
    ]
    try:
        prompt_tokens = model.count_tokens(texts)
    except Exception:
        prompt_tokens = 0
    usage = Usage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens, completion_tokens=None)
    return EmbeddingResponse(data=data, model=req.model, usage=usage)


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completions(  # noqa: PLR0915, PLR0912
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

    raw_messages = [msg.model_dump(mode="python") for msg in req.messages]
    has_images = _contains_image_content(raw_messages)
    stop = _normalize_stop(req.stop)

    start = time.perf_counter()
    label_token = set_queue_label(req.model or "chat")
    try:
        async with limiter():
            try:
                model = registry.get(req.model)
            except KeyError as exc:
                record_chat_request(req.model, "404")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model {req.model} not found",
                ) from exc

            capabilities = getattr(model, "capabilities", [])
            if "chat-completion" not in capabilities:
                record_chat_request(req.model, "400")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Model {req.model} does not support chat/completions",
                )
            if has_images and "vision" not in capabilities:
                record_chat_request(req.model, "400")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Model {req.model} does not support image inputs",
                )

            defaults = getattr(model, "generation_defaults", {}) or {}
            max_tokens_default = defaults.get("max_tokens") or int(os.getenv("MAX_NEW_TOKENS", "512"))
            temperature_default = defaults.get("temperature", 0.7)
            top_p_default = defaults.get("top_p", 0.9)

            max_tokens = req.max_tokens or max_tokens_default
            if max_tokens <= 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="max_tokens must be positive",
                )
            temperature = req.temperature if req.temperature is not None else temperature_default
            top_p = req.top_p if req.top_p is not None else top_p_default

            loop = asyncio.get_running_loop()
            executor = get_chat_executor()
            max_prompt_tokens = int(os.getenv("CHAT_MAX_PROMPT_TOKENS", "4096"))
            prepared_inputs, prompt_tokens = await _prepare_chat_request(model, raw_messages, executor)
            if prompt_tokens > max_prompt_tokens:
                record_chat_request(req.model, "400")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Prompt too long; max {max_prompt_tokens} tokens",
                )
            generation: ChatGeneration | None = None
            batcher = getattr(_request.app.state, "chat_batching_service", None)
            if (
                batcher is not None
                and getattr(batcher, "is_supported", lambda _m: False)(req.model)
                and not has_images
            ):
                try:
                    generation = await batcher.enqueue(
                        req.model,
                        raw_messages,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop=stop,
                        prompt_tokens=prompt_tokens,
                        prepared_inputs=prepared_inputs,
                    )
                except ValueError as exc:
                    record_chat_request(req.model, "400")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=str(exc),
                    ) from exc
                except ChatBatchQueueFullError as exc:
                    record_chat_request(req.model, "429")
                    logger.info(
                        "chat_batch_queue_full", extra={"model": req.model, "status": 429}
                    )
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Chat batch queue full",
                        headers={"Retry-After": str(int(QUEUE_TIMEOUT_SEC))},
                    ) from exc
                except ChatBatchQueueTimeoutError as exc:
                    record_chat_request(req.model, "429")
                    logger.info(
                        "chat_batch_queue_timeout", extra={"model": req.model, "status": 429}
                    )
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Chat batch queue wait exceeded",
                        headers={"Retry-After": str(int(QUEUE_TIMEOUT_SEC))},
                    ) from exc
                except Exception as exc:  # pragma: no cover - defensive fallback
                    logger.warning(
                        "chat_batcher_failed_falling_back",
                        extra={"model": req.model, "error": str(exc)},
                    )
                    generation = None

            if generation is None:
                try:
                    lock = _get_chat_lock(req.model)
                    async with lock:
                        if prepared_inputs is not None and hasattr(model, "generate_prepared"):
                            generation = await loop.run_in_executor(
                                executor,
                                lambda: model.generate_prepared(
                                    prepared_inputs,
                                    max_new_tokens=max_tokens,
                                    temperature=temperature,
                                    top_p=top_p,
                                    stop=stop,
                                ),
                            )
                        else:
                            generation = await loop.run_in_executor(
                                executor,
                                lambda: model.generate(
                                    raw_messages,
                                    max_new_tokens=max_tokens,
                                    temperature=temperature,
                                    top_p=top_p,
                                    stop=stop,
                                ),
                            )
                except Exception as exc:  # pragma: no cover - unexpected runtime failure
                    record_chat_request(req.model, "500")
                    logger.exception(
                        "chat_generation_failed",
                        extra={"model": req.model, "max_tokens": max_tokens, "status": 500},
                    )
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Chat generation failed",
                    ) from exc
            if generation is None:
                record_chat_request(req.model, "500")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Chat generation failed",
                )
    except QueueFullError as exc:
        record_chat_request(req.model, "429")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Request queue full",
            headers={"Retry-After": str(int(QUEUE_TIMEOUT_SEC))},
        ) from exc
    except ShuttingDownError as exc:
        record_chat_request(req.model, "503")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is shutting down",
        ) from exc
    except QueueTimeoutError as exc:
        record_chat_request(req.model, "429")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Timed out waiting for worker",
            headers={"Retry-After": str(int(QUEUE_TIMEOUT_SEC))},
        ) from exc
    finally:
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
    prompt_tokens = (
        generation.prompt_tokens
        if generation.prompt_tokens is not None
        else prompt_tokens  # fall back to pre-check count
    )
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


ALLOWED_AUDIO_RESPONSE_FORMATS = {"json", "text", "srt", "verbose_json", "vtt"}


async def _handle_audio_request(  # noqa: PLR0912, PLR0913, PLR0915
    *,
    file: UploadFile,
    model_name: str,
    registry: ModelRegistry,
    task: Literal["transcribe", "translate"],
    language: str | None,
    prompt: str | None,
    response_format: str,
    temperature: float | None,
    timestamp_granularities: list[str] | None,
) -> Response:
    if response_format not in ALLOWED_AUDIO_RESPONSE_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid response_format '{response_format}'",
        )

    # Whisper is deterministic with temperature=0; default to that unless user overrides.
    effective_temperature = 0.0 if temperature is None else temperature
    granularity = _select_granularity(timestamp_granularities)
    need_segments = response_format in {"verbose_json", "srt", "vtt"} or granularity is not None

    start = time.perf_counter()
    temp_path: str | None = None
    size_bytes = 0
    duration: float | None = None
    label_token = set_audio_queue_label(model_name or "audio")

    try:
        temp_path, size_bytes = await _save_upload(file)
        duration = _probe_duration(temp_path)

        async with audio_limiter():
            try:
                model = registry.get(model_name)
            except KeyError as exc:
                record_audio_request(model_name, "404")
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model {model_name} not found") from exc

            capabilities = getattr(model, "capabilities", [])
            if "audio-transcription" not in capabilities:
                record_audio_request(model_name, "400")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Model {model_name} does not support audio transcription",
                )

            loop = asyncio.get_running_loop()
            executor = get_audio_executor()
            try:
                result = await loop.run_in_executor(
                    executor,
                    lambda: model.transcribe(
                        temp_path,
                        language=language,
                        prompt=prompt,
                        temperature=effective_temperature,
                        task=task,
                        timestamp_granularity=granularity if need_segments else None,
                    ),
                )
            except Exception as exc:  # pragma: no cover - runtime failure
                record_audio_request(model_name, "500")
                logger.exception(
                    "audio_transcription_failed",
                    extra={"model": model_name, "task": task, "status": 500},
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Audio processing failed",
                ) from exc
    except AudioQueueFullError as exc:
        record_audio_request(model_name, "429")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Request queue full",
            headers={"Retry-After": str(int(AUDIO_QUEUE_TIMEOUT_SEC))},
        ) from exc
    except AudioShuttingDownError as exc:
        record_audio_request(model_name, "503")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is shutting down",
        ) from exc
    except AudioQueueTimeoutError as exc:
        record_audio_request(model_name, "429")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Timed out waiting for worker",
            headers={"Retry-After": str(int(AUDIO_QUEUE_TIMEOUT_SEC))},
        ) from exc
    finally:
        reset_audio_queue_label(label_token)
        if temp_path:
            with contextlib.suppress(Exception):
                Path(temp_path).unlink(missing_ok=True)

    latency = time.perf_counter() - start
    observe_audio_latency(model_name, latency)
    record_audio_request(model_name, "200")
    logger.info(
        "audio_request",
        extra={
            "model": model_name,
            "task": task,
            "latency_ms": round(latency * 1000, 2),
            "bytes": size_bytes,
            "duration": duration,
            "response_format": response_format,
        },
    )

    # Build response content
    text = getattr(result, "text", "") or ""
    language_out = getattr(result, "language", None) or language
    duration_out = getattr(result, "duration", None) or duration
    segments = getattr(result, "segments", None) or []

    if response_format == "text":
        return PlainTextResponse(text)

    if response_format == "json":
        return JSONResponse({"text": text})

    if response_format == "verbose_json":
        verbose = TranscriptionVerboseResponse(
            text=text,
            language=language_out,
            duration=duration_out,
            segments=[
                TranscriptionSegment(id=s.id, start=s.start, end=s.end, text=s.text)
                for s in segments
            ]
            if segments
            else None,
        )
        return JSONResponse(verbose.model_dump(mode="json", exclude_none=True))

    seg_dicts: list[dict[str, float | str]] = [
        {"id": s.id, "start": s.start, "end": s.end, "text": s.text} for s in segments
    ]
    if not seg_dicts:
        seg_dicts = [{"id": 0, "start": 0.0, "end": duration_out or 0.0, "text": text}]

    if response_format == "srt":
        return PlainTextResponse(_srt_from_segments(seg_dicts), media_type="application/x-subrip")

    if response_format == "vtt":
        return PlainTextResponse(_vtt_from_segments(seg_dicts), media_type="text/vtt")

    # Should never reach here because of earlier validation
    return JSONResponse({"text": text})


@router.post("/v1/audio/transcriptions")
async def create_transcription(  # noqa: PLR0913
    registry: Annotated[ModelRegistry, Depends(get_model_registry)],
    file: UploadFile = File(...),  # noqa: B008
    model: str = Form(...),  # noqa: B008
    language: str | None = Form(default=None),  # noqa: B008
    prompt: str | None = Form(default=None),  # noqa: B008
    response_format: str = Form(default="json"),  # noqa: B008
    temperature: float | None = Form(default=None),  # noqa: B008
    timestamp_granularities: list[str] | None = Form(default=None),  # noqa: B008
) -> Response:
    return await _handle_audio_request(
        file=file,
        model_name=model,
        registry=registry,
        task="transcribe",
        language=language,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        timestamp_granularities=timestamp_granularities,
    )


@router.post("/v1/audio/translations")
async def create_translation(  # noqa: PLR0913
    registry: Annotated[ModelRegistry, Depends(get_model_registry)],
    file: UploadFile = File(...),  # noqa: B008
    model: str = Form(...),  # noqa: B008
    prompt: str | None = Form(default=None),  # noqa: B008
    response_format: str = Form(default="json"),  # noqa: B008
    temperature: float | None = Form(default=None),  # noqa: B008
    timestamp_granularities: list[str] | None = Form(default=None),  # noqa: B008
) -> Response:
    return await _handle_audio_request(
        file=file,
        model_name=model,
        registry=registry,
        task="translate",
        language="en",  # OpenAI translate outputs English
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        timestamp_granularities=timestamp_granularities,
    )

@router.get("/v1/models", response_model=ModelsResponse)
async def list_models(
    registry: Annotated[ModelRegistry, Depends(get_model_registry)]
) -> ModelsResponse:
    models: list[ModelInfo] = []
    for name in registry.list_models():
        model = registry.get(name)
        dim = getattr(model, "dim", None)
        models.append(ModelInfo(id=name, embedding_dimensions=dim))
    return ModelsResponse(data=models)


@router.get("/health", response_model=HealthResponse)
async def health(
    request: Request,
    registry: Annotated[ModelRegistry | None, Depends(get_model_registry, use_cache=False)] = None,
) -> HealthResponse | JSONResponse:
    if registry is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model registry not initialized")
    try:
        models = registry.list_models()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Registry unavailable") from exc

    warmup_status = _resolve_warmup_status(request)
    warmup_failures = list(warmup_status.failures)
    warmup_details = WarmupDetails(
        required=warmup_status.required,
        completed=warmup_status.completed,
        failures=warmup_failures or None,
        ok_models=warmup_status.ok_models or None,
        capabilities=warmup_status.capabilities or None,
    )

    chat_queue_depths: list[QueueDepth] | None = None
    embed_queue_depths: list[QueueDepth] | None = None

    chat_batcher = getattr(request.app.state, "chat_batching_service", None)
    if chat_batcher and getattr(chat_batcher, "queue_stats", None):
        chat_queue_depths = [
            QueueDepth(model=name, size=size, max_size=max_size)
            for name, (size, max_size) in chat_batcher.queue_stats().items()
        ]

    embed_batcher = getattr(request.app.state, "batching_service", None)
    if embed_batcher and getattr(embed_batcher, "queue_stats", None):
        embed_queue_depths = [
            QueueDepth(model=name, size=size, max_size=max_size)
            for name, (size, max_size) in embed_batcher.queue_stats().items()
        ]

    health_status = "ok"
    http_status = status.HTTP_200_OK
    if warmup_status.required and (not warmup_status.completed or warmup_failures):
        health_status = "unhealthy"
        http_status = status.HTTP_503_SERVICE_UNAVAILABLE

    response = HealthResponse(
        status=health_status,
        models=models,
        warmup_failures=warmup_failures or None,
        warmup=warmup_details,
        chat_batch_queues=chat_queue_depths,
        embedding_batch_queues=embed_queue_depths,
    )

    if http_status != status.HTTP_200_OK:
        return JSONResponse(status_code=http_status, content=response.model_dump())

    return response
