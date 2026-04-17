"""
API key authentication middleware.
Checks x-api-key header against the API_KEY env var.
"""

import os

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


API_KEY = os.getenv("API_KEY", "")


class ApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip auth if no API_KEY configured
        if not API_KEY:
            return await call_next(request)

        # Skip preflight
        if request.method == "OPTIONS":
            return await call_next(request)

        # Skip auth for the root docs page and warmup ping
        if request.url.path in ("", "/", "/api/warmup"):
            return await call_next(request)

        provided_key = request.headers.get("x-api-key", "")
        if provided_key != API_KEY:
            return JSONResponse(
                status_code=401,
                content={"error": "Unauthorized: invalid or missing API key."},
            )

        return await call_next(request)
