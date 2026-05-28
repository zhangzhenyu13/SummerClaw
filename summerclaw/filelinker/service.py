"""FileLinker core service — token lifecycle, file management, persistence."""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import secrets
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiofiles
from loguru import logger

from summerclaw.filelinker.models import FileLinkToken
from summerclaw.filelinker.tailscale import TailscaleHelper

if TYPE_CHECKING:
    pass  # FileLinkerConfig imported at runtime via duck-typing


_INDEX_FILE = ".index.json"


class FileLinkerService:
    """FileLinker core service managing token lifecycle and file storage."""

    def __init__(self, config: Any, workspace: Path):
        """
        Args:
            config: ``FileLinkerConfig`` instance (duck-typed to avoid import cycle).
            workspace: Project workspace root directory.
        """
        self.config = config
        self.storage_dir = (
            Path(config.storage_dir)
            if getattr(config, "storage_dir", "")
            else workspace / "filelinker_storage"
        )
        self._tokens: dict[str, FileLinkToken] = {}
        self._tailscale_ip: str | None = getattr(config, "tailscale_ip", "") or None
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None
        self._started = False

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def tailscale_ip(self) -> str | None:
        return self._tailscale_ip

    @property
    def port(self) -> int:
        return getattr(self.config, "port", 8090)

    def exceeds_threshold(self, channel: str, file_path: str) -> bool:
        """Return ``True`` if *file_path* exceeds the channel's size threshold.

        Unlike ``should_use_link``, this checks size only — it does NOT require
        Tailscale to be available.  Used by the middleware to decide whether to
        fall back to a warning message when Tailscale is down.
        """
        if not getattr(self.config, "enabled", False):
            return False
        try:
            size = os.path.getsize(file_path)
        except OSError:
            return False
        thresholds: dict[str, int] = getattr(self.config, "channel_thresholds", {})
        threshold = thresholds.get(channel, thresholds.get("default", 800_000))
        return size >= threshold

    def should_use_link(self, channel: str, file_path: str) -> bool:
        """Return ``True`` if *file_path* exceeds the channel's size threshold."""
        if not getattr(self.config, "enabled", False):
            return False
        if not self._tailscale_ip:
            return False
        return self.exceeds_threshold(channel, file_path)

    def get_file_size(self, file_path: str) -> int:
        try:
            return os.path.getsize(file_path)
        except OSError:
            return 0

    async def create_link(
        self,
        file_path: str,
        original_name: str,
        channel: str,
        chat_id: str,
        max_downloads: int = 0,
    ) -> str:
        """
        Create a P2P download link.

        1. Copy the file into ``storage_dir/{token}/{original_name}``.
        2. Generate a secure random token.
        3. Record ``FileLinkToken`` metadata.
        4. Return the P2P URL: ``http://{tailscale_ip}:{port}/dl/{token}/{filename}``
        """
        src = Path(file_path)
        if not src.is_file():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        max_size_bytes = getattr(self.config, "max_file_size_mb", 500) * 1024 * 1024
        if src.stat().st_size > max_size_bytes:
            raise ValueError(
                f"File {original_name} ({src.stat().st_size} bytes) exceeds "
                f"max_file_size_mb limit ({self.config.max_file_size_mb} MB)"
            )

        token_str = secrets.token_urlsafe(24)
        token_dir = self.storage_dir / token_str
        token_dir.mkdir(parents=True, exist_ok=True)
        dest = token_dir / original_name

        # Async copy
        async with aiofiles.open(src, "rb") as fsrc, aiofiles.open(dest, "wb") as fdst:
            while chunk := await fsrc.read(1024 * 1024):
                await fdst.write(chunk)

        content_type = mimetypes.guess_type(original_name)[0] or "application/octet-stream"
        now = time.time()
        ttl_seconds = getattr(self.config, "token_ttl_hours", 24) * 3600

        ftoken = FileLinkToken(
            token=token_str,
            file_path=str(dest),
            original_name=original_name,
            file_size=src.stat().st_size,
            content_type=content_type,
            channel=channel,
            chat_id=chat_id,
            created_at=now,
            expires_at=now + ttl_seconds,
            max_downloads=max_downloads,
        )

        async with self._lock:
            self._tokens[token_str] = ftoken

        await self.save_index()

        ip = self._tailscale_ip or "127.0.0.1"
        url = f"http://{ip}:{self.port}/dl/{token_str}/{original_name}"
        logger.info("FileLinker link created: {} -> {}", original_name, url)
        return url

    async def validate_token(self, token: str) -> FileLinkToken | None:
        """Validate token existence, expiry, and download count."""
        async with self._lock:
            ftoken = self._tokens.get(token)
        if ftoken is None:
            return None
        if ftoken.is_expired:
            logger.debug("FileLinker token expired: {}", token[:8])
            return None
        if ftoken.is_download_exhausted():
            logger.debug("FileLinker token download exhausted: {}", token[:8])
            return None
        return ftoken

    async def record_download(self, token: str) -> None:
        async with self._lock:
            ftoken = self._tokens.get(token)
            if ftoken:
                ftoken.download_count += 1
        await self.save_index()

    async def cleanup_expired(self) -> int:
        """Remove expired tokens and their files. Returns count of cleaned entries."""
        now = time.time()
        expired_tokens: list[str] = []

        async with self._lock:
            for tok, meta in self._tokens.items():
                if meta.expires_at < now:
                    expired_tokens.append(tok)

            for tok in expired_tokens:
                meta = self._tokens.pop(tok, None)
                if meta:
                    token_dir = Path(meta.file_path).parent
                    if token_dir.exists() and token_dir != self.storage_dir:
                        shutil.rmtree(token_dir, ignore_errors=True)
                        logger.debug("Cleaned expired token dir: {}", token_dir.name)

        if expired_tokens:
            await self.save_index()
            logger.info("FileLinker cleaned {} expired tokens", len(expired_tokens))

        return len(expired_tokens)

    # ── Persistence ───────────────────────────────────────────────────────────

    async def save_index(self) -> None:
        """Persist the token index to ``.index.json``."""
        index_path = self.storage_dir / _INDEX_FILE
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        data = {tok: meta.to_dict() for tok, meta in self._tokens.items()}
        async with aiofiles.open(index_path, "w") as f:
            await f.write(json.dumps(data, indent=2))

    async def load_index(self) -> None:
        """Restore unexpired tokens from ``.index.json``."""
        index_path = self.storage_dir / _INDEX_FILE
        if not index_path.exists():
            return
        try:
            async with aiofiles.open(index_path) as f:
                raw = await f.read()
            data = json.loads(raw)
            now = time.time()
            async with self._lock:
                for tok, meta_dict in data.items():
                    meta = FileLinkToken.from_dict(meta_dict)
                    if meta.expires_at > now:
                        self._tokens[tok] = meta
            logger.info("FileLinker restored {} active tokens from index", len(self._tokens))
        except Exception as exc:
            logger.warning("FileLinker index load failed: {}", exc)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the service: create storage dir, load index, detect IP, start cleanup loop."""
        if self._started:
            return

        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Detect Tailscale IP if not configured
        if not self._tailscale_ip:
            self._tailscale_ip = TailscaleHelper.get_tailscale_ip()
            if self._tailscale_ip:
                logger.info("FileLinker Tailscale IP detected: {}", self._tailscale_ip)
            else:
                logger.warning("FileLinker disabled — Tailscale IP not available")

        await self.load_index()
        await self.cleanup_expired()

        # Periodic cleanup
        interval_s = getattr(self.config, "cleanup_interval_hours", 6) * 3600
        self._cleanup_task = asyncio.create_task(self._cleanup_loop(interval_s))
        self._started = True
        logger.info("FileLinker service started (storage: {})", self.storage_dir)

    async def stop(self) -> None:
        """Persist state and stop the cleanup loop."""
        if not self._started:
            return
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        await self.save_index()
        self._started = False
        logger.info("FileLinker service stopped")

    async def _cleanup_loop(self, interval_s: int) -> None:
        while True:
            try:
                await asyncio.sleep(interval_s)
                await self.cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("FileLinker cleanup loop error: {}", exc)
