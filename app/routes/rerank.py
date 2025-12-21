from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.concurrency.limiter import (
    EMBEDDING_QUEUE_TIMEOUT_SEC,
    QueueFullError,
    QueueTimeoutError,
    ShuttingDownError,
    embedding_limiter,
    reset_queue_label,
    set_queue_label,
)
from app.config import settings
from app.dependencies import get_model_registry
from app.models.registry import ModelRegistry
from app.monitoring.metrics import observe_rerank_latency, record_rerank_request
from app.routes.common import (
    _ClientDisconnectedError,
    _RequestCancelledError,
    _run_work_with_client_cancel,
    _WorkTimeoutError,
)
from app.threadpool import get_embedding_executor
from app.utils.executor_context import run_in_executor_with_context

router = APIRouter()
logger = logging.getLogger(__name__)


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: list[str]
    top_n: int | None = Field(default=None, description="Top N results to return")


class RerankResponseResult(BaseModel):
    index: int
    relevance_score: float
    document: str | None = None


class RerankResponse(BaseModel):
    model: str
    results: list[RerankResponseResult]
    usage: dict[str, int]


@router.post("/v1/rerank", response_model=RerankResponse)
async def create_rerank(  # noqa: PLR0915
    req: RerankRequest,
    registry: Annotated[ModelRegistry, Depends(get_model_registry)],
    request: Request,
) -> RerankResponse:
    model_name = req.model
    start = time.perf_counter()
    cancel_event = threading.Event()
    label_token = set_queue_label(model_name or "rerank")

    try:
        try:
            model = registry.get(model_name)
        except KeyError as exc:
            record_rerank_request(model_name, "404")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_name} not found",
            ) from exc

        if "rerank" not in getattr(model, "capabilities", []):
            record_rerank_request(model_name, "400")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model {model_name} does not support rerank",
            )

        async def _run_rerank_work() -> list[RerankResponseResult]:
            # Rerank uses direct executor path (batching not implemented for rerank)
            executor = get_embedding_executor()
            loop = asyncio.get_running_loop()
            return await run_in_executor_with_context(
                loop,
                executor,
                lambda: model.rerank(req.query, req.documents, top_k=req.top_n, cancel_event=cancel_event),
            )

        limiter_cm = _get_rerank_limiter()
        limiter_ctx = limiter_cm() if callable(limiter_cm) else limiter_cm
        async with limiter_ctx:
            work_task = asyncio.ensure_future(_run_rerank_work())
            try:
                results = await _run_work_with_client_cancel(
                    request=request,
                    work_task=work_task,
                    cancel_event=cancel_event,
                    timeout=settings.embedding_generate_timeout_sec,
                )
            except _WorkTimeoutError as exc:
                cancel_event.set()
                record_rerank_request(model_name, "504")
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Rerank timed out",
                ) from exc
            except _RequestCancelledError as exc:
                cancel_event.set()
                record_rerank_request(model_name, "499")
                raise HTTPException(
                    status_code=status.HTTP_499_CLIENT_CLOSED_REQUEST,
                    detail="Request cancelled",
                ) from exc
            except _ClientDisconnectedError as exc:
                cancel_event.set()
                record_rerank_request(model_name, "499")
                raise HTTPException(
                    status_code=status.HTTP_499_CLIENT_CLOSED_REQUEST,
                    detail="Client disconnected",
                ) from exc
            except HTTPException:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                record_rerank_request(model_name, "500")
                logger.exception("Rerank failed", extra={"model": model_name})
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Rerank failed",
                ) from exc
    except QueueFullError as exc:
        record_rerank_request(model_name, "429")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rerank request queue full",
            headers={"Retry-After": str(int(EMBEDDING_QUEUE_TIMEOUT_SEC))},
        ) from exc
    except ShuttingDownError as exc:
        record_rerank_request(model_name, "503")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is shutting down",
        ) from exc
    except QueueTimeoutError as exc:
        record_rerank_request(model_name, "429")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Timed out waiting for rerank worker",
            headers={"Retry-After": str(int(EMBEDDING_QUEUE_TIMEOUT_SEC))},
        ) from exc
    finally:
        reset_queue_label(label_token)

    latency = time.perf_counter() - start
    observe_rerank_latency(model_name, latency)
    record_rerank_request(model_name, "200")
    logger.info(
        "rerank_request",
        extra={
            "model": model_name,
            "latency_ms": round(latency * 1000, 2),
            "status": 200,
        },
    )

    return RerankResponse(
        model=model_name,
        results=[
            RerankResponseResult(
                index=r.index,
                relevance_score=r.relevance_score,
                document=r.document,
            )
            for r in results
        ],
        usage={"total_tokens": 0, "prompt_tokens": 0},
    )


def _get_rerank_limiter() -> Any:
    """Indirection kept for tests to monkeypatch the limiter."""

    try:
        from app import api as api_module  # noqa: PLC0415 - local import for test patching

        return getattr(api_module, "embedding_limiter", embedding_limiter)
    except Exception:
        return embedding_limiter
