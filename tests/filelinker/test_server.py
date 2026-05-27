"""Tests for summerclaw.filelinker.server — HTTP download endpoint."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from summerclaw.filelinker.server import create_filelinker_app
from summerclaw.filelinker.service import FileLinkerService


def _make_config(
    enabled=True,
    tailscale_ip="127.0.0.1",
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
        channel_thresholds=channel_thresholds or {"default": 800_000},
    )


@pytest.fixture
def svc_and_client(tmp_path):
    """Create a service + test client pair."""
    ws = tmp_path / "ws"
    ws.mkdir()
    cfg = _make_config(storage_dir=str(ws / "storage"))
    svc = FileLinkerService(cfg, ws)
    svc._tailscale_ip = "127.0.0.1"
    svc.storage_dir.mkdir(parents=True, exist_ok=True)
    app = create_filelinker_app(svc)
    client = TestClient(app)
    return svc, client, ws


class TestDownloadEndpoint:
    @pytest.mark.asyncio
    async def test_successful_download(self, svc_and_client):
        svc, client, ws = svc_and_client
        src = ws / "report.pdf"
        src.write_bytes(b"PDF-content-here" * 100)

        url = await svc.create_link(str(src), "report.pdf", "telegram", "c1")
        token = url.split("/dl/")[1].split("/")[0]

        resp = client.get(f"/dl/{token}/report.pdf")
        assert resp.status_code == 200
        assert resp.content == b"PDF-content-here" * 100
        assert "attachment" in resp.headers.get("content-disposition", "")

    @pytest.mark.asyncio
    async def test_invalid_token_403(self, svc_and_client):
        _, client, _ = svc_and_client
        resp = client.get("/dl/fake-token/file.txt")
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_expired_token_403(self, svc_and_client):
        svc, client, ws = svc_and_client
        src = ws / "old.txt"
        src.write_bytes(b"old-data")

        url = await svc.create_link(str(src), "old.txt", "telegram", "c1")
        token = url.split("/dl/")[1].split("/")[0]

        svc._tokens[token].expires_at = time.time() - 10

        resp = client.get(f"/dl/{token}/old.txt")
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_filename_mismatch_403(self, svc_and_client):
        svc, client, ws = svc_and_client
        src = ws / "data.csv"
        src.write_bytes(b"csv,data")

        url = await svc.create_link(str(src), "data.csv", "telegram", "c1")
        token = url.split("/dl/")[1].split("/")[0]

        resp = client.get(f"/dl/{token}/wrong_name.csv")
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self, svc_and_client):
        svc, client, ws = svc_and_client
        src = ws / "safe.txt"
        src.write_bytes(b"safe")

        url = await svc.create_link(str(src), "safe.txt", "telegram", "c1")
        token = url.split("/dl/")[1].split("/")[0]

        # Tamper file_path to point outside storage
        svc._tokens[token].file_path = "/etc/passwd"
        resp = client.get(f"/dl/{token}/safe.txt")
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_download_count_incremented(self, svc_and_client):
        svc, client, ws = svc_and_client
        src = ws / "counter.txt"
        src.write_bytes(b"count-me")

        url = await svc.create_link(str(src), "counter.txt", "telegram", "c1")
        token = url.split("/dl/")[1].split("/")[0]

        client.get(f"/dl/{token}/counter.txt")
        client.get(f"/dl/{token}/counter.txt")

        assert svc._tokens[token].download_count == 2


class TestHealthEndpoint:
    def test_health_ok(self, svc_and_client):
        _, client, _ = svc_and_client
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
