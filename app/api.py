import asyncio
import contextlib
import inspect
import logging
import os
import threading
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

from app.batching import EmbeddingBatchQueueTimeoutError
from app.chat_batching import (
    ChatBatchQueueFullError,
    ChatBatchQueueTimeoutError,
    get_count_executor,
)
from app.concurrency.audio_limiter import (
    QUEUE_TIMEOUT_SEC as AUDIO_QUEUE_TIMEOUT_SEC,
    AudioQueueFullError,
    AudioQueueTimeoutError,
    AudioShuttingDownError,
    limiter as audio_limiter,
    reset_queue_label as reset_audio_queue_label,
    set_queue_label as set_audio_queue_label,
)
from app.concurrency.limiter import (
    CHAT_QUEUE_TIMEOUT_SEC,
    EMBEDDING_QUEUE_TIMEOUT_SEC,
    QUEUE_TIMEOUT_SEC,
    QueueFullError,
    QueueTimeoutError,
    ShuttingDownError,
    chat_limiter,
    embedding_limiter,
    reset_queue_label,
    set_queue_label,
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
from app.state import WarmupStatus
from app.threadpool import get_audio_executor, get_chat_executor, get_embedding_count_executor, get_embedding_executor
from app.utils.uploads import chunked_upload_to_tempfile

router = APIRouter()
logger = logging.getLogger(__name__)
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(25 * 1024 * 1024)))  # default 25MB
UPLOAD_CHUNK_BYTES = 1024 * 1024  # 1MB chunks


class _WorkTimeoutError(Exception):
    """Internal marker for executor work timeouts."""


class _RequestCancelledError(Exception):
    """Internal marker when work is cancelled before completion."""


class _ClientDisconnectedError(Exception):
    """Internal marker when the client disconnects first."""


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


def _resolve_generation_params(
    req: "ChatCompletionRequest",
    model: Any,
) -> tuple[int, float, float]:
    """Resolve max_tokens, temperature, top_p from request and model defaults.

    Returns (max_tokens, temperature, top_p).
    """
    defaults = getattr(model, "generation_defaults", {}) or {}
    max_tokens_default = defaults.get("max_tokens") or int(os.getenv("MAX_NEW_TOKENS", "512"))
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
    """Build kwargs dict for model.generate or model.generate_prepared."""
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
    """Tokenize once for both counting and generation when supported by the model.

    Runs heavy tokenizer work inside the chat executor to avoid blocking the event loop.
    """

    loop = asyncio.get_running_loop()
    prepare_timeout = float(os.getenv("CHAT_PREPARE_TIMEOUT_SEC", "10"))
    count_executor = get_count_executor(
        use_chat_executor=os.getenv("CHAT_COUNT_USE_CHAT_EXECUTOR", "0") != "0"
    )
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
    """Run chat generation with optional batching and shared timeout/cancel logic.

    Returns the ChatGeneration together with (prompt_tokens, max_tokens) used
    so the caller can build usage and logging without re-threading details.
    """

    model = _resolve_chat_model_and_caps(registry, req.model, has_images=has_images)
    max_tokens, temperature, top_p = _resolve_generation_params(req, model)

    if max_tokens <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="max_tokens must be positive",
        )

    loop = asyncio.get_running_loop()
    executor = get_chat_executor()
    max_prompt_tokens = int(os.getenv("CHAT_MAX_PROMPT_TOKENS", "4096"))
    prepared_inputs, prompt_tokens = await _prepare_chat_request(model, raw_messages)
    if prompt_tokens > max_prompt_tokens:
        record_chat_request(req.model, "400")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prompt too long; max {max_prompt_tokens} tokens",
        )
    cancel_event = threading.Event()
    gen_timeout = float(os.getenv("CHAT_GENERATE_TIMEOUT_SEC", "60"))
    generate_accepts_cancel = "cancel_event" in inspect.signature(model.generate).parameters
    generate_prepared_accepts_cancel = hasattr(model, "generate_prepared") and "cancel_event" in inspect.signature(model.generate_prepared).parameters
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
        # Prefer chat batching when supported and no vision inputs; on
        # internal errors fall back to per-request generation so the
        # request can still succeed.
        if (
            batcher is not None
            and getattr(batcher, "is_supported", lambda _m: False)(req.model)
            and not has_images
        ):
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
                # Fall back to per-request generation below.
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
    except HTTPException:
        # Propagate HTTPExceptions produced inside the generation helpers
        # unchanged so status codes and metrics stay local.
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
    if request is None:
        return WarmupStatus()

    status_obj = getattr(request.app.state, "warmup_status", None)
    if isinstance(status_obj, WarmupStatus):
        return status_obj

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


EXECUTOR_GRACE_PERIOD = float(os.getenv("EXECUTOR_GRACE_PERIOD_SEC", "2.0"))


async def _await_executor_cleanup(
    work_task: "asyncio.Future[Any]",
    grace_period: float,
    reason: str,
) -> None:
    """Wait briefly for executor work to finish after cancellation.

    This prevents background work from piling up silently after timeouts or
    disconnects. If work doesn't complete within the grace period, we log a
    warning but don't block further.
    """
    if work_task.done():
        return

    try:
        await asyncio.wait_for(asyncio.shield(work_task), timeout=grace_period)
    except TimeoutError:
        logger.warning(
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
    work_task: "asyncio.Future[Any]",
    cancel_event: threading.Event,
    timeout: float,
) -> Any:
    """Wait for work or client disconnect, propagating timeouts and cancellations.

    This helper encapsulates the common pattern of racing a background executor
    task against client disconnect and a hard timeout. It does not translate
    errors into HTTP responses so that callers can keep metrics and status code
    handling local to each endpoint.

    After setting the cancel_event on timeout or disconnect, we wait briefly
    (EXECUTOR_GRACE_PERIOD) for the executor work to finish and log if it
    overruns. This prevents background work from piling up silently.
    """

    disconnect_task: asyncio.Task[None] = asyncio.create_task(
        _cancel_on_disconnect(request, cancel_event)
    )
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

        # Client disconnect task completed first.
        cancel_event.set()
        work_task.cancel()
        await _await_executor_cleanup(work_task, EXECUTOR_GRACE_PERIOD, "client_disconnect")
        raise _ClientDisconnectedError()
    finally:
        disconnect_task.cancel()


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
    runtime_config: dict[str, Any] | None = None


def _normalize_embedding_texts(req: EmbeddingRequest) -> list[str]:
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

    return texts


async def _build_embedding_usage(model: Any, texts: list[str]) -> Usage:
    """Compute OpenAI-style usage for embeddings, optionally skipping token count."""

    # High-QPS deployments can trade usage token accounting for lower latency by
    # skipping a second tokenizer pass when EMBEDDING_USAGE_DISABLE_TOKEN_COUNT=1.
    disable_usage_tokens = os.getenv("EMBEDDING_USAGE_DISABLE_TOKEN_COUNT", "0") != "0"
    if disable_usage_tokens:
        prompt_tokens = 0
    else:
        try:
            loop = asyncio.get_running_loop()
            prompt_tokens = await loop.run_in_executor(
                get_embedding_count_executor(),
                lambda: model.count_tokens(texts),
            )
        except Exception:
            prompt_tokens = 0

    return Usage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens, completion_tokens=None)


async def _run_embedding_generation(  # noqa: PLR0913
    *,
    registry: ModelRegistry,
    model_name: str,
    texts: list[str],
    request: Request,
    cancel_event: threading.Event,
    timeout: float,
) -> Any:
    """Resolve model and run embedding generation with batching/timeout/cancellation.

    This keeps the embedding route focused on request validation, metrics, and
    queue-level errors while centralising the execution and error mapping here.
    """

    try:
        model = registry.get(model_name)
    except KeyError as exc:
        record_request(model_name, "404")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_name} not found",
        ) from exc

    try:
        batcher = getattr(request.app.state, "batching_service", None)
        loop = asyncio.get_running_loop()
        if batcher is not None and getattr(batcher, "enabled", False):
            work_task = asyncio.ensure_future(
                batcher.enqueue(model_name, texts, cancel_event=cancel_event)
            )
        else:
            executor = get_embedding_executor()
            work_task = asyncio.ensure_future(
                loop.run_in_executor(
                    executor,
                    lambda: model.embed(texts, cancel_event=cancel_event),
                )
            )
        return await _run_work_with_client_cancel(
            request=request,
            work_task=work_task,
            cancel_event=cancel_event,
            timeout=timeout,
        )
    except _WorkTimeoutError as exc:
        cancel_event.set()
        record_request(model_name, "504")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Embedding generation timed out",
        ) from exc
    except _RequestCancelledError as exc:
        cancel_event.set()
        record_request(model_name, "499")
        raise HTTPException(
            status_code=status.HTTP_499_CLIENT_CLOSED_REQUEST,
            detail="Request cancelled",
        ) from exc
    except _ClientDisconnectedError as exc:
        cancel_event.set()
        record_request(model_name, "499")
        raise HTTPException(
            status_code=status.HTTP_499_CLIENT_CLOSED_REQUEST,
            detail="Client disconnected",
        ) from exc
    except EmbeddingBatchQueueTimeoutError as exc:
        record_request(model_name, "429")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Embedding batch queue wait exceeded",
            headers={"Retry-After": str(int(QUEUE_TIMEOUT_SEC))},
        ) from exc
    except Exception as exc:  # pragma: no cover - unexpected runtime failure
        record_request(model_name, "500")
        logger.exception(
            "embedding_failed",
            extra={
                "model": model_name,
                "batch_size": len(texts),
                # Model may not expose device attribute; use getattr defensively.
                "device": getattr(registry.get(model_name), "device", None)
                if model_name in registry.list_models()
                else None,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Embedding generation failed",
        ) from exc


@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(  # noqa: PLR0912
    req: EmbeddingRequest,
    registry: Annotated[ModelRegistry, Depends(get_model_registry)],
    request: Request,
) -> EmbeddingResponse:
    texts = _normalize_embedding_texts(req)
    start = time.perf_counter()
    embed_timeout = float(os.getenv("EMBEDDING_GENERATE_TIMEOUT_SEC", "60"))
    cancel_event = threading.Event()
    label_token = set_queue_label(req.model or "embedding")
    try:
        async with embedding_limiter():
            vectors = await _run_embedding_generation(
                registry=registry,
                model_name=req.model,
                texts=texts,
                request=request,
                cancel_event=cancel_event,
                timeout=embed_timeout,
            )
    except QueueFullError as exc:
        record_request(req.model, "429")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Embedding request queue full",
            headers={"Retry-After": str(int(EMBEDDING_QUEUE_TIMEOUT_SEC))},
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
            detail="Timed out waiting for embedding worker",
            headers={"Retry-After": str(int(EMBEDDING_QUEUE_TIMEOUT_SEC))},
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
    # Reuse the same underlying model instance for usage token accounting.
    usage_model = registry.get(req.model)
    usage = await _build_embedding_usage(usage_model, texts)
    return EmbeddingResponse(data=data, model=req.model, usage=usage)


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completions(  # noqa: PLR0912
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
    label_token = set_queue_label(req.model or "chat")
    try:
        async with chat_limiter():
            raw_messages = [msg.model_dump(mode="python") for msg in req.messages]
            has_images = _contains_image_content(raw_messages)
            generation, prompt_tokens, max_tokens = await _run_chat_generation(
                req=req,
                registry=registry,
                request=_request,
                raw_messages=raw_messages,
                has_images=has_images,
            )
    except QueueFullError as exc:
        record_chat_request(req.model, "429")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Chat request queue full",
            headers={"Retry-After": str(int(CHAT_QUEUE_TIMEOUT_SEC))},
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
            detail="Timed out waiting for chat worker",
            headers={"Retry-After": str(int(CHAT_QUEUE_TIMEOUT_SEC))},
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


def _validate_audio_params(
    response_format: str,
    timestamp_granularities: list[str] | None,
    temperature: float | None,
) -> tuple[float, Literal["word", "segment", None], bool]:
    """Validate and resolve audio request parameters.

    Returns:
        tuple of (effective_temperature, granularity, need_segments)

    Raises:
        HTTPException if response_format is invalid.
    """
    if response_format not in ALLOWED_AUDIO_RESPONSE_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid response_format '{response_format}'",
        )

    # Whisper is deterministic with temperature=0; default to that unless user overrides.
    effective_temperature = 0.0 if temperature is None else temperature
    granularity = _select_granularity(timestamp_granularities)
    need_segments = response_format in {"verbose_json", "srt", "vtt"} or granularity is not None

    return effective_temperature, granularity, need_segments


def _format_audio_response(
    *,
    response_format: str,
    result: Any,
    language: str | None,
    duration: float | None,
) -> Response:
    """Build the appropriate Response based on response_format.

    Args:
        response_format: One of json, text, srt, verbose_json, vtt
        result: SpeechResult from the model
        language: Requested or detected language
        duration: Audio duration in seconds
    """
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


async def _execute_transcription(  # noqa: PLR0913
    *,
    temp_path: str,
    model: Any,
    model_name: str,
    request: Request,
    task: Literal["transcribe", "translate"],
    language: str | None,
    prompt: str | None,
    effective_temperature: float,
    granularity: Literal["word", "segment", None],
    need_segments: bool,
    cancel_event: threading.Event,
    audio_timeout: float,
) -> Any:
    """Execute the transcription/translation via the model.

    Handles timeout, cancellation, and client disconnect.

    Returns:
        SpeechResult from the model

    Raises:
        HTTPException on timeout, cancellation, disconnect, or internal errors.
    """
    loop = asyncio.get_running_loop()
    executor = get_audio_executor()

    work_task: asyncio.Future[Any] = asyncio.ensure_future(
        loop.run_in_executor(
            executor,
            lambda: model.transcribe(
                temp_path,
                language=language,
                prompt=prompt,
                temperature=effective_temperature,
                task=task,
                timestamp_granularity=granularity if need_segments else None,
                cancel_event=cancel_event,
            ),
        )
    )
    try:
        return await _run_work_with_client_cancel(
            request=request,
            work_task=work_task,
            cancel_event=cancel_event,
            timeout=audio_timeout,
        )
    except _WorkTimeoutError as exc:
        cancel_event.set()
        record_audio_request(model_name, "504")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Audio processing timed out",
        ) from exc
    except _RequestCancelledError as exc:
        cancel_event.set()
        record_audio_request(model_name, "499")
        raise HTTPException(
            status_code=status.HTTP_499_CLIENT_CLOSED_REQUEST,
            detail="Request cancelled",
        ) from exc
    except _ClientDisconnectedError as exc:
        cancel_event.set()
        record_audio_request(model_name, "499")
        raise HTTPException(
            status_code=status.HTTP_499_CLIENT_CLOSED_REQUEST,
            detail="Client disconnected",
        ) from exc
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


def _resolve_audio_model_and_caps(registry: ModelRegistry, model_name: str) -> Any:
    """Lookup and validate an audio-capable model for Whisper-style endpoints."""

    try:
        model = registry.get(model_name)
    except KeyError as exc:
        record_audio_request(model_name, "404")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_name} not found",
        ) from exc

    capabilities = getattr(model, "capabilities", [])
    if "audio-transcription" not in capabilities:
        record_audio_request(model_name, "400")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model {model_name} does not support audio transcription",
        )

    return model


def _resolve_chat_model_and_caps(
    registry: ModelRegistry,
    model_name: str,
    *,
    has_images: bool,
) -> Any:
    """Lookup and validate a chat-capable model, including optional vision support."""

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


async def _handle_audio_request(  # noqa: PLR0913
    *,
    file: UploadFile,
    model_name: str,
    registry: ModelRegistry,
    request: Request,
    task: Literal["transcribe", "translate"],
    language: str | None,
    prompt: str | None,
    response_format: str,
    temperature: float | None,
    timestamp_granularities: list[str] | None,
) -> Response:
    """Process audio transcription/translation requests.

    Orchestrates parameter validation, file upload, model execution, and response formatting.
    """
    effective_temperature, granularity, need_segments = _validate_audio_params(
        response_format, timestamp_granularities, temperature
    )

    start = time.perf_counter()
    temp_path: str | None = None
    size_bytes = 0
    duration: float | None = None
    audio_timeout = float(os.getenv("AUDIO_PROCESS_TIMEOUT_SEC", "180"))
    cancel_event = threading.Event()
    label_token = set_audio_queue_label(model_name or "audio")

    try:
        async with audio_limiter():
            temp_path, size_bytes = await _save_upload(file)
            loop = asyncio.get_running_loop()
            executor = get_audio_executor()
            duration = await loop.run_in_executor(executor, lambda: _probe_duration(temp_path))
            model = _resolve_audio_model_and_caps(registry, model_name)

            result = await _execute_transcription(
                temp_path=temp_path,
                model=model,
                model_name=model_name,
                request=request,
                task=task,
                language=language,
                prompt=prompt,
                effective_temperature=effective_temperature,
                granularity=granularity,
                need_segments=need_segments,
                cancel_event=cancel_event,
                audio_timeout=audio_timeout,
            )
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

    return _format_audio_response(
        response_format=response_format,
        result=result,
        language=language,
        duration=duration,
    )


@router.post("/v1/audio/transcriptions")
async def create_transcription(  # noqa: PLR0913
    registry: Annotated[ModelRegistry, Depends(get_model_registry)],
    request: Request,
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
        request=request,
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
    request: Request,
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
        request=request,
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

    runtime_cfg: dict[str, Any] | None = getattr(request.app.state, "runtime_config", None)

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
        runtime_config=runtime_cfg,
    )

    if http_status != status.HTTP_200_OK:
        return JSONResponse(status_code=http_status, content=response.model_dump())

    return response
