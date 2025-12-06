"""Request ID middleware for tracing requests across logs and responses.

Usage:
    1. Add the middleware to your FastAPI app:
       app.add_middleware(RequestIDMiddleware)

    2. Access the current request ID in your code:
       from app.middleware import get_request_id
       request_id = get_request_id()

    3. The ID is automatically included in:
       - Response headers (X-Request-ID)
       - Logs (via JsonFormatter integration)
"""

from __future__ import annotations

import contextvars
import uuid
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

# Context variable to hold the current request ID
_request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)

# Header name for request ID (following common conventions)
REQUEST_ID_HEADER = "X-Request-ID"


def get_request_id() -> str | None:
    """Get the current request ID from context.

    Returns:
        The request ID if set, None otherwise.
    """
    return _request_id_var.get()


def request_id_context() -> dict[str, Any]:
    """Get the request ID as a dict suitable for logging extras.

    Returns:
        Dict with request_id key if set, empty dict otherwise.
    """
    request_id = _request_id_var.get()
    if request_id:
        return {"request_id": request_id}
    return {}


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware that assigns and propagates request IDs.

    - If the incoming request has an X-Request-ID header, that value is used.
    - Otherwise, a new UUID is generated.
    - The request ID is stored in a context variable for access throughout the request.
    - The request ID is included in the response headers.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Get existing request ID from header or generate a new one
        request_id = request.headers.get(REQUEST_ID_HEADER)
        if not request_id:
            request_id = uuid.uuid4().hex

        # Store in context variable
        token = _request_id_var.set(request_id)
        try:
            # Process the request
            response = await call_next(request)

            # Add request ID to response headers
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            # Reset the context variable
            _request_id_var.reset(token)



