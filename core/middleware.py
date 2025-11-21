from typing import Callable
from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from starlette.middleware.cors import CORSMiddleware


class RequestIDMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, header_name: str = "X-Request-ID") -> None:
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next: Callable):
        req_id = request.headers.get(self.header_name) or str(uuid4())
        request.state.request_id = req_id
        response = await call_next(request)
        response.headers[self.header_name] = req_id
        return response


def apply_middlewares(app: ASGIApp, settings) -> None:
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=getattr(settings, "CORS_ALLOW_ORIGINS", ["*"]),
        allow_credentials=getattr(settings, "CORS_ALLOW_CREDENTIALS", True),
        allow_methods=getattr(settings, "CORS_ALLOW_METHODS", ["*"]),
        allow_headers=getattr(settings, "CORS_ALLOW_HEADERS", ["*"]),
    )
    # Request ID
    app.add_middleware(RequestIDMiddleware, header_name=getattr(settings, "REQUEST_ID_HEADER", "X-Request-ID"))
