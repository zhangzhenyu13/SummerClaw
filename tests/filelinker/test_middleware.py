"""Tests for summerclaw.filelinker.middleware — FileLinkerMiddleware."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from summerclaw.bus.events import OutboundMessage
from summerclaw.channels.base import BaseChannel
from summerclaw.filelinker.middleware import FileLinkerMiddleware, format_size
from summerclaw.filelinker.service import FileLinkerService


# ── Stub channel ──────────────────────────────────────────────────────────────

class StubChannel(BaseChannel):
    name = "telegram"
    display_name = "Telegram"

    async def start(self): pass
    async def stop(self): pass
    async def send(self, msg): pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_config(
    enabled=True,
    tailscale_ip="100.64.0.1",
    port=8090,
    token_ttl_hours=24,
    max_file_size_mb=500,
    storage_dir="",
    cleanup_interval_hours=6,
    channel_thresholds=None,
) -> SimpleNamespace:
    return SimpleNamespace(
        enabled=enabled,
        tailscale_ip=tailscale_ip,
        port=port,
        token_ttl_hours=token_ttl_hours,
        max_file_size_mb=max_file_size_mb,
        storage_dir=storage_dir,
        cleanup_interval_hours=cleanup_interval_hours,
        channel_thresholds=channel_thresholds or {"default": 800_000, "telegram": 800_000},
    )


@pytest.fixture(autouse=True)
def _reset_middleware():
    """Ensure middleware service is cleared between tests."""
    FileLinkerMiddleware.set_service(None)
    yield
    FileLinkerMiddleware.set_service(None)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestFormatSize:
    def test_bytes(self):
        assert format_size(512) == "512 B"

    def test_kilobytes(self):
        assert format_size(2048) == "2.0 KB"

    def test_megabytes(self):
        assert format_size(15 * 1024 * 1024) == "15.0 MB"

    def test_gigabytes(self):
        assert format_size(2 * 1024 ** 3) == "2.00 GB"


class TestIntercept:
    @pytest.mark.asyncio
    async def test_no_service_passthrough(self):
        msg = OutboundMessage(channel="telegram", chat_id="c1", content="hi", media=["/f.txt"])
        ch = StubChannel(config={}, bus=None)
        result = await FileLinkerMiddleware.intercept(msg, ch)
        assert result is msg

    @pytest.mark.asyncio
    async def test_disabled_passthrough(self, tmp_path):
        cfg = _make_config(enabled=False)
        svc = FileLinkerService(cfg, tmp_path)
        FileLinkerMiddleware.set_service(svc)

        msg = OutboundMessage(channel="telegram", chat_id="c1", content="hi", media=["/f.txt"])
        ch = StubChannel(config={}, bus=None)
        result = await FileLinkerMiddleware.intercept(msg, ch)
        assert result is msg

    @pytest.mark.asyncio
    async def test_no_media_passthrough(self, tmp_path):
        cfg = _make_config()
        svc = FileLinkerService(cfg, tmp_path)
        svc._tailscale_ip = "100.64.0.1"
        FileLinkerMiddleware.set_service(svc)

        msg = OutboundMessage(channel="telegram", chat_id="c1", content="hi")
        ch = StubChannel(config={}, bus=None)
        result = await FileLinkerMiddleware.intercept(msg, ch)
        assert result is msg

    @pytest.mark.asyncio
    async def test_small_file_passthrough(self, tmp_path):
        cfg = _make_config(storage_dir=str(tmp_path / "storage"))
        svc = FileLinkerService(cfg, tmp_path)
        svc._tailscale_ip = "100.64.0.1"
        FileLinkerMiddleware.set_service(svc)

        small = tmp_path / "small.txt"
        small.write_bytes(b"x" * 100)

        msg = OutboundMessage(channel="telegram", chat_id="c1", content="see attached", media=[str(small)])
        ch = StubChannel(config={}, bus=None)
        result = await FileLinkerMiddleware.intercept(msg, ch)
        assert result.media == [str(small)]
        assert result.content == "see attached"

    @pytest.mark.asyncio
    async def test_large_file_replaced_with_link(self, tmp_path):
        cfg = _make_config(storage_dir=str(tmp_path / "storage"))
        svc = FileLinkerService(cfg, tmp_path)
        svc._tailscale_ip = "100.64.0.1"
        FileLinkerMiddleware.set_service(svc)

        big = tmp_path / "big.apk"
        big.write_bytes(b"x" * 900_000)

        msg = OutboundMessage(channel="telegram", chat_id="c1", content="done", media=[str(big)])
        ch = StubChannel(config={}, bus=None)
        result = await FileLinkerMiddleware.intercept(msg, ch)

        assert result.media == []
        assert "📎" in result.content
        assert "big.apk" in result.content
        assert "http://100.64.0.1:8090/dl/" in result.content
        assert "done" in result.content

    @pytest.mark.asyncio
    async def test_mixed_files(self, tmp_path):
        cfg = _make_config(storage_dir=str(tmp_path / "storage"))
        svc = FileLinkerService(cfg, tmp_path)
        svc._tailscale_ip = "100.64.0.1"
        FileLinkerMiddleware.set_service(svc)

        small = tmp_path / "small.txt"
        small.write_bytes(b"x" * 100)
        big = tmp_path / "big.zip"
        big.write_bytes(b"x" * 900_000)

        msg = OutboundMessage(
            channel="telegram", chat_id="c1",
            content="results", media=[str(small), str(big)],
        )
        ch = StubChannel(config={}, bus=None)
        result = await FileLinkerMiddleware.intercept(msg, ch)

        assert str(small) in result.media
        assert str(big) not in result.media
        assert "big.zip" in result.content

    @pytest.mark.asyncio
    async def test_empty_content_with_link(self, tmp_path):
        cfg = _make_config(storage_dir=str(tmp_path / "storage"))
        svc = FileLinkerService(cfg, tmp_path)
        svc._tailscale_ip = "100.64.0.1"
        FileLinkerMiddleware.set_service(svc)

        big = tmp_path / "data.bin"
        big.write_bytes(b"x" * 900_000)

        msg = OutboundMessage(channel="telegram", chat_id="c1", content="", media=[str(big)])
        ch = StubChannel(config={}, bus=None)
        result = await FileLinkerMiddleware.intercept(msg, ch)

        # When original content is empty, should NOT prepend \n\n
        assert result.content.startswith("📎")
        assert result.media == []
