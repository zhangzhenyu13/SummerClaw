"""Tests for summerclaw.filelinker.tailscale — TailscaleHelper."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest

from summerclaw.filelinker.tailscale import TailscaleHelper


class TestGetTailscaleIp:
    """TailscaleHelper.get_tailscale_ip() detection tests."""

    def test_env_variable_wins(self):
        with patch.dict(os.environ, {"TAILSCALE_IP": "100.64.0.99"}):
            assert TailscaleHelper.get_tailscale_ip() == "100.64.0.99"

    def test_env_variable_empty_string(self):
        with patch.dict(os.environ, {"TAILSCALE_IP": ""}):
            # Should fall through to CLI / interface
            with patch.object(TailscaleHelper, "_detect_via_cli", return_value=None), \
                 patch.object(TailscaleHelper, "_detect_via_interface", return_value=None):
                assert TailscaleHelper.get_tailscale_ip() is None

    def test_cli_detection(self):
        with patch.dict(os.environ, {"TAILSCALE_IP": ""}, clear=False):
            with patch.object(TailscaleHelper, "_detect_via_cli", return_value="100.64.0.5"):
                assert TailscaleHelper.get_tailscale_ip() == "100.64.0.5"

    def test_interface_detection(self):
        with patch.dict(os.environ, {"TAILSCALE_IP": ""}, clear=False):
            with patch.object(TailscaleHelper, "_detect_via_cli", return_value=None), \
                 patch.object(TailscaleHelper, "_detect_via_interface", return_value="100.64.0.7"):
                assert TailscaleHelper.get_tailscale_ip() == "100.64.0.7"

    def test_all_fail_returns_none(self):
        with patch.dict(os.environ, {"TAILSCALE_IP": ""}, clear=False):
            with patch.object(TailscaleHelper, "_detect_via_cli", return_value=None), \
                 patch.object(TailscaleHelper, "_detect_via_interface", return_value=None):
                assert TailscaleHelper.get_tailscale_ip() is None

    def test_cli_subprocess_success(self):
        """Test actual _detect_via_cli with mocked subprocess."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "100.64.0.12\n"
        with patch("summerclaw.filelinker.tailscale.shutil.which", return_value="/usr/bin/tailscale"), \
             patch("summerclaw.filelinker.tailscale.subprocess.run", return_value=mock_result):
            assert TailscaleHelper._detect_via_cli() == "100.64.0.12"

    def test_cli_no_binary(self):
        with patch("summerclaw.filelinker.tailscale.shutil.which", return_value=None):
            assert TailscaleHelper._detect_via_cli() is None


class TestHealthCheck:
    """TailscaleHelper.health_check() tests."""

    @pytest.mark.asyncio
    async def test_no_binary_returns_not_running(self):
        with patch("summerclaw.filelinker.tailscale.shutil.which", return_value=None):
            result = await TailscaleHelper.health_check()
        assert result == {"running": False, "ip": None, "peers": 0}

    @pytest.mark.asyncio
    async def test_successful_status(self):
        status_json = b'{"Self": {"TailscaleIPs": ["100.64.0.1", "fd7a:115c::1"]}, "Peer": {"p1": {"Online": true}, "p2": {"Online": false}}}'
        mock_proc = MagicMock()
        mock_proc.returncode = 0

        async def mock_communicate():
            return (status_json, b"")

        mock_proc.communicate = mock_communicate

        with patch("summerclaw.filelinker.tailscale.shutil.which", return_value="/usr/bin/tailscale"), \
             patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await TailscaleHelper.health_check()

        assert result["running"] is True
        assert result["ip"] == "100.64.0.1"
        assert result["peers"] == 1

    @pytest.mark.asyncio
    async def test_command_failure(self):
        mock_proc = MagicMock()
        mock_proc.returncode = 1

        async def mock_communicate():
            return (b"", b"error")

        mock_proc.communicate = mock_communicate

        with patch("summerclaw.filelinker.tailscale.shutil.which", return_value="/usr/bin/tailscale"), \
             patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await TailscaleHelper.health_check()

        assert result["running"] is False
