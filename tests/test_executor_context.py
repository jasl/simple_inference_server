import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from app.middleware.request_id import REQUEST_ID_HEADER, RequestIDMiddleware, get_request_id as _get_request_id
from app.utils.executor_context import run_in_executor_with_context


def test_request_id_contextvars_propagate_into_executor_threads() -> None:
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    with ThreadPoolExecutor(max_workers=1) as executor:

        @app.get("/_executor_request_id")
        async def _executor_request_id() -> dict[str, str | None]:
            loop = asyncio.get_running_loop()
            rid = await run_in_executor_with_context(loop, executor, _get_request_id)
            return {"request_id": rid}

        with TestClient(app) as client:
            resp = client.get("/_executor_request_id", headers={REQUEST_ID_HEADER: "req-123"})
            assert resp.status_code == status.HTTP_200_OK
            assert resp.json()["request_id"] == "req-123"

