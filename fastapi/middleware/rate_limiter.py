"""
Simple sliding-window rate limiter (in-memory, per-process).
Matches the Node.js implementation: 20 requests per 10 seconds.
"""

import time

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimiterMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = 20, window_ms: int = 10_000):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_ms = window_ms
        self.timestamps: list[float] = []

    async def dispatch(self, request: Request, call_next):
        # Skip preflight and root
        if request.method == "OPTIONS" or request.url.path in ("/", ""):
            return await call_next(request)

        now = time.time() * 1000
        cutoff = now - self.window_ms

        # Remove expired timestamps
        while self.timestamps and self.timestamps[0] <= cutoff:
            self.timestamps.pop(0)

        if len(self.timestamps) >= self.max_requests:
            return JSONResponse(
                status_code=429,
                content={"error": "Too many requests. Please try again shortly."},
                headers={"Retry-After": "10"},
            )

        self.timestamps.append(now)
        return await call_next(request)
