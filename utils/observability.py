"""
Observability middleware — request IDs, timing, and structured tracing.

Adds a unique request_id to every HTTP request for log correlation.
"""

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject a unique request_id into every request for traceability."""

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        start = time.time()

        response: Response = await call_next(request)

        elapsed = time.time() - start
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{elapsed:.3f}s"
        return response
