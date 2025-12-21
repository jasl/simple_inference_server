from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from pathlib import Path
from typing import Annotated, Any, Literal

import torchaudio
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel

from app.concurrency.audio_limiter import (
    QUEUE_TIMEOUT_SEC as AUDIO_QUEUE_TIMEOUT_SEC,
    AudioQueueFullError,
    AudioQueueTimeoutError,
    AudioShuttingDownError,
    limiter as audio_limiter,
    reset_queue_label as reset_audio_queue_label,
    set_queue_label as set_audio_queue_label,
)
from app.config import settings
from app.dependencies import get_model_registry
from app.models.registry import ModelRegistry
from app.monitoring.metrics import (
    observe_audio_latency,
    record_audio_request,
)
from app.routes.common import (
    _ClientDisconnectedError,
    _RequestCancelledError,
    _run_work_with_client_cancel,
    _WorkTimeoutError,
)
from app.threadpool import get_audio_executor
from app.utils.executor_context import run_in_executor_with_context, run_in_executor_with_context_limited
from app.utils.uploads import chunked_upload_to_tempfile

router = APIRouter()
logger = logging.getLogger(__name__)
MAX_AUDIO_BYTES = settings.max_audio_bytes
UPLOAD_CHUNK_BYTES = 1024 * 1024  # 1MB chunks


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


ALLOWED_AUDIO_RESPONSE_FORMATS = {"json", "text", "srt", "verbose_json", "vtt"}


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
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _vtt_from_segments(segments: list[dict[str, float | str]]) -> str:
    lines: list[str] = ["WEBVTT", ""]
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        text = str(seg.get("text", "")).strip()
        lines.append(f"{_format_ts(start, sep='.')} --> {_format_ts(end, sep='.')}".replace(",", "."))
        lines.append(text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _normalize_stop_audio(response_format: str) -> None:
    if response_format not in ALLOWED_AUDIO_RESPONSE_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid response_format '{response_format}'",
        )


def _select_granularity(values: list[str] | None) -> Literal["word", "segment", None]:
    if not values:
        return None
    lowered = [v.lower() for v in values]
    if "word" in lowered:
        return "word"
    if "segment" in lowered:
        return "segment"
    return None


async def _save_upload(file: UploadFile, max_bytes: int = MAX_AUDIO_BYTES) -> tuple[str, int]:
    suffix = Path(file.filename or "").suffix or ".wav"
    try:
        from app import api as api_module  # noqa: PLC0415 - local import to avoid circular import

        chunk_size = getattr(api_module, "UPLOAD_CHUNK_BYTES", UPLOAD_CHUNK_BYTES)
    except Exception:
        chunk_size = UPLOAD_CHUNK_BYTES
    return await chunked_upload_to_tempfile(
        file,
        chunk_size=chunk_size,
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


def _validate_audio_params(
    response_format: str,
    timestamp_granularities: list[str] | None,
    temperature: float | None,
) -> tuple[float, Literal["word", "segment", None], bool]:
    _normalize_stop_audio(response_format)
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
            segments=[TranscriptionSegment(id=s.id, start=s.start, end=s.end, text=s.text) for s in segments]
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
    loop = asyncio.get_running_loop()
    executor = get_audio_executor()

    async def _run_transcribe() -> Any:
        return await run_in_executor_with_context_limited(
            loop,
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
            model=model,
        )

    work_task: asyncio.Future[Any] = asyncio.ensure_future(_run_transcribe())
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
    effective_temperature, granularity, need_segments = _validate_audio_params(
        response_format, timestamp_granularities, temperature
    )

    start = time.perf_counter()
    temp_path: str | None = None
    size_bytes = 0
    duration: float | None = None
    audio_timeout = settings.audio_process_timeout_sec
    cancel_event = threading.Event()
    label_token = set_audio_queue_label(model_name or "audio")

    try:
        # Allow tests to monkeypatch api._save_upload
        try:
            from app import api as api_module  # noqa: PLC0415 - local import for test patching

            save_fn = getattr(api_module, "_save_upload", _save_upload)
        except Exception:
            save_fn = _save_upload

        temp_path, size_bytes = await save_fn(file)
        loop = asyncio.get_running_loop()
        executor = get_audio_executor()
        duration = await run_in_executor_with_context(loop, executor, lambda: _probe_duration(temp_path))

        async with audio_limiter():
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
        language="en",
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        timestamp_granularities=timestamp_granularities,
    )
