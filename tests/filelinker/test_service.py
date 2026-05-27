"""Tests for summerclaw.filelinker.service — FileLinkerService."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from summerclaw.filelinker.service import FileLinkerService


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


@pytest.fixture
def tmp_workspace(tmp_path):
    """Provide a temporary workspace directory."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def service(tmp_workspace):
    """Create a FileLinkerService with a temp workspace."""
    cfg = _make_config()
    svc = FileLinkerService(cfg, tmp_workspace)
    svc._tailscale_ip = "100.64.0.1"  # bypass auto-detection
    return svc


class TestShouldUseLink:
    def test_disabled_returns_false(self, tmp_workspace):
        cfg = _make_config(enabled=False)
        svc = FileLinkerService(cfg, tmp_workspace)
        svc._tailscale_ip = "100.64.0.1"
        f = tmp_workspace / "small.txt"
        f.write_bytes(b"x" * 900_000)
        assert svc.should_use_link("telegram", str(f)) is False

    def test_no_tailscale_ip_returns_false(self, tmp_workspace):
        cfg = _make_config(tailscale_ip="")
        svc = FileLinkerService(cfg, tmp_workspace)
        svc._tailscale_ip = None
        f = tmp_workspace / "big.txt"
        f.write_bytes(b"x" * 900_000)
        assert svc.should_use_link("telegram", str(f)) is False

    def test_small_file_returns_false(self, service, tmp_workspace):
        f = tmp_workspace / "small.txt"
        f.write_bytes(b"x" * 100)
        assert service.should_use_link("telegram", str(f)) is False

    def test_large_file_returns_true(self, service, tmp_workspace):
        f = tmp_workspace / "big.apk"
        f.write_bytes(b"x" * 900_000)
        assert service.should_use_link("telegram", str(f)) is True

    def test_channel_specific_threshold(self, tmp_workspace):
        cfg = _make_config(channel_thresholds={"discord": 500_000, "default": 800_000})
        svc = FileLinkerService(cfg, tmp_workspace)
        svc._tailscale_ip = "100.64.0.1"
        f = tmp_workspace / "mid.zip"
        f.write_bytes(b"x" * 600_000)
        assert svc.should_use_link("discord", str(f)) is True
        assert svc.should_use_link("unknown_ch", str(f)) is False

    def test_nonexistent_file_returns_false(self, service):
        assert service.should_use_link("telegram", "/no/such/file.txt") is False


class TestCreateLink:
    @pytest.mark.asyncio
    async def test_creates_link_and_copies_file(self, service, tmp_workspace):
        src = tmp_workspace / "output.apk"
        src.write_bytes(b"fake-apk-data" * 100)

        url = await service.create_link(
            file_path=str(src),
            original_name="output.apk",
            channel="telegram",
            chat_id="chat123",
        )

        assert url.startswith("http://100.64.0.1:8090/dl/")
        assert url.endswith("/output.apk")
        assert len(service._tokens) == 1

        token_meta = next(iter(service._tokens.values()))
        assert token_meta.original_name == "output.apk"
        assert token_meta.channel == "telegram"
        assert Path(token_meta.file_path).exists()

    @pytest.mark.asyncio
    async def test_file_not_found_raises(self, service):
        with pytest.raises(FileNotFoundError):
            await service.create_link(
                file_path="/no/such/file.txt",
                original_name="file.txt",
                channel="telegram",
                chat_id="c1",
            )

    @pytest.mark.asyncio
    async def test_exceeds_max_size_raises(self, service, tmp_workspace):
        service.config.max_file_size_mb = 1  # 1 MB limit
        src = tmp_workspace / "huge.bin"
        src.write_bytes(b"x" * (1024 * 1024 + 1))
        with pytest.raises(ValueError, match="exceeds"):
            await service.create_link(str(src), "huge.bin", "telegram", "c1")


class TestValidateToken:
    @pytest.mark.asyncio
    async def test_valid_token(self, service, tmp_workspace):
        src = tmp_workspace / "f.txt"
        src.write_bytes(b"data")
        url = await service.create_link(str(src), "f.txt", "discord", "c1")
        token = url.split("/dl/")[1].split("/")[0]

        meta = await service.validate_token(token)
        assert meta is not None
        assert meta.original_name == "f.txt"

    @pytest.mark.asyncio
    async def test_expired_token_returns_none(self, service, tmp_workspace):
        src = tmp_workspace / "f.txt"
        src.write_bytes(b"data")
        url = await service.create_link(str(src), "f.txt", "discord", "c1")
        token = url.split("/dl/")[1].split("/")[0]

        # Force expiry
        service._tokens[token].expires_at = time.time() - 1
        assert await service.validate_token(token) is None

    @pytest.mark.asyncio
    async def test_exhausted_downloads_returns_none(self, service, tmp_workspace):
        src = tmp_workspace / "f.txt"
        src.write_bytes(b"data")
        url = await service.create_link(str(src), "f.txt", "discord", "c1", max_downloads=1)
        token = url.split("/dl/")[1].split("/")[0]

        await service.record_download(token)
        assert await service.validate_token(token) is None

    @pytest.mark.asyncio
    async def test_unknown_token_returns_none(self, service):
        assert await service.validate_token("nonexistent-token") is None


class TestCleanupExpired:
    @pytest.mark.asyncio
    async def test_cleanup_removes_expired(self, service, tmp_workspace):
        src = tmp_workspace / "a.txt"
        src.write_bytes(b"aa")
        await service.create_link(str(src), "a.txt", "telegram", "c1")

        # Force expiry
        for tok in service._tokens:
            service._tokens[tok].expires_at = time.time() - 1

        count = await service.cleanup_expired()
        assert count == 1
        assert len(service._tokens) == 0

    @pytest.mark.asyncio
    async def test_cleanup_keeps_valid(self, service, tmp_workspace):
        src = tmp_workspace / "b.txt"
        src.write_bytes(b"bb")
        await service.create_link(str(src), "b.txt", "telegram", "c1")

        count = await service.cleanup_expired()
        assert count == 0
        assert len(service._tokens) == 1


class TestPersistence:
    @pytest.mark.asyncio
    async def test_save_and_load_index(self, service, tmp_workspace):
        src = tmp_workspace / "p.txt"
        src.write_bytes(b"persist")
        await service.create_link(str(src), "p.txt", "discord", "c2")

        await service.save_index()
        index_path = service.storage_dir / ".index.json"
        assert index_path.exists()

        # Create a new service and load
        svc2 = FileLinkerService(service.config, tmp_workspace)
        svc2._tailscale_ip = "100.64.0.1"
        await svc2.load_index()
        assert len(svc2._tokens) == 1
        tok = next(iter(svc2._tokens.values()))
        assert tok.original_name == "p.txt"

    @pytest.mark.asyncio
    async def test_load_skips_expired(self, service, tmp_workspace):
        src = tmp_workspace / "exp.txt"
        src.write_bytes(b"exp")
        await service.create_link(str(src), "exp.txt", "discord", "c2")

        # Force expiry before save
        for tok in service._tokens:
            service._tokens[tok].expires_at = time.time() - 1
        await service.save_index()

        svc2 = FileLinkerService(service.config, tmp_workspace)
        await svc2.load_index()
        assert len(svc2._tokens) == 0
