"""FileLinker HTTP download service — binds to Tailscale IP only."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

import aiofiles
from loguru import logger
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response, StreamingResponse
from starlette.routing import Route

if TYPE_CHECKING:
    from summerclaw.filelinker.service import FileLinkerService

_CHUNK_SIZE = 8192  # 8 KB streaming chunks


# ── Streaming helper ──────────────────────────────────────────────────────────

async def _stream_file(file_path: Path, chunk_size: int = _CHUNK_SIZE):
    """Async generator that yields file chunks."""
    async with aiofiles.open(file_path, "rb") as f:
        while True:
            data = await f.read(chunk_size)
            if not data:
                break
            yield data


# ── Route handlers ────────────────────────────────────────────────────────────

def _make_download_handler(service: FileLinkerService):
    """Create the download route handler bound to *service*."""

    async def download(request: Request) -> Response:
        token = request.path_params["token"]
        filename = request.path_params["filename"]

        # 1. Validate token
        ftoken = await service.validate_token(token)
        if ftoken is None:
            return PlainTextResponse("403 Forbidden — invalid or expired token", status_code=403)

        # 2. Security: filename must match
        if filename != ftoken.original_name:
            return PlainTextResponse("403 Forbidden — filename mismatch", status_code=403)

        # 3. Security: realpath must be under storage_dir
        real_file = Path(ftoken.file_path).resolve()
        real_storage = service.storage_dir.resolve()
        try:
            real_file.relative_to(real_storage)
        except ValueError:
            return PlainTextResponse("403 Forbidden — path traversal blocked", status_code=403)

        if not real_file.is_file():
            return PlainTextResponse("404 Not Found", status_code=404)

        # 4. Record download
        await service.record_download(token)

        # 5. Stream response
        return StreamingResponse(
            content=_stream_file(real_file),
            status_code=200,
            headers={
                "Content-Type": ftoken.content_type,
                "Content-Disposition": f'attachment; filename="{ftoken.original_name}"',
                "Content-Length": str(ftoken.file_size),
            },
        )

    return download


def _make_health_handler(service: FileLinkerService):
    async def health(request: Request) -> Response:
        from starlette.responses import JSONResponse

        return JSONResponse({
            "status": "ok",
            "tailscale_ip": service.tailscale_ip,
            "active_tokens": len(service._tokens),
        })

    return health


# ── App factory ───────────────────────────────────────────────────────────────

def create_filelinker_app(service: FileLinkerService) -> Starlette:
    """Create a Starlette app for the FileLinker HTTP server."""
    routes = [
        Route("/dl/{token}/{filename}", _make_download_handler(service), methods=["GET"]),
        Route("/health", _make_health_handler(service), methods=["GET"]),
    ]
    return Starlette(routes=routes)


# ── Server lifecycle ──────────────────────────────────────────────────────────

class FileLinkerHTTPServer:
    """Manages the FileLinker HTTP server lifecycle (uvicorn)."""

    def __init__(self, service: FileLinkerService):
        self.service = service
        self._server: object | None = None  # uvicorn.Server

    async def start(self) -> None:
        host = self.service.tailscale_ip or "127.0.0.1"
        port = self.service.port

        app = create_filelinker_app(self.service)

        try:
            import uvicorn
        except ImportError:
            logger.warning("uvicorn not installed — FileLinker HTTP server not started")
            return

        config = uvicorn.Config(app=app, host=host, port=port, log_level="info")
        self._server = uvicorn.Server(config)
        logger.info("FileLinker HTTP server starting on {}:{}", host, port)
        await self._server.serve()

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
            logger.info("FileLinker HTTP server stopping")
