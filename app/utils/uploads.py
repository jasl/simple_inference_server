from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from fastapi import HTTPException, UploadFile, status


class _ExceptionFactory(Protocol):
    def __call__(self, limit: int, size: int) -> Exception:  # pragma: no cover - protocol
        ...


async def chunked_upload_to_tempfile(
    file: UploadFile,
    *,
    chunk_size: int,
    max_bytes: int,
    suffix: str,
    on_exceed: _ExceptionFactory | None = None,
) -> tuple[str, int]:
    """Stream an UploadFile into a temporary file with size enforcement."""

    exception_factory: Callable[[int, int], Exception]

    if on_exceed is None:
        def _default_exception(limit: int, size: int) -> Exception:  # pragma: no cover - default
            return HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File too large; max {limit} bytes (received {size})",
            )

        exception_factory = _default_exception
    else:
        exception_factory = on_exceed

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        try:
            size = 0
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                size += len(chunk)
                if max_bytes and size > max_bytes:
                    raise exception_factory(max_bytes, size)
                tmp.write(chunk)
            tmp.flush()
            return tmp.name, size
        except Exception:
            Path(tmp.name).unlink(missing_ok=True)
            raise
