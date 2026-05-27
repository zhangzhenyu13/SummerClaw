"""Tailscale integration utilities — IP detection and health checks."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from loguru import logger


class TailscaleHelper:
    """Tailscale integration helper providing IP auto-detection and health checks."""

    # ── IP Detection ────────────────────────────────────────────────────────

    @staticmethod
    def get_tailscale_ip() -> str | None:
        """
        Auto-detect the local Tailscale IPv4 address.

        Detection order:
          1. ``TAILSCALE_IP`` environment variable
          2. ``tailscale ip -4`` CLI command
          3. Network interface name matching ``tailscale`` prefix

        Returns ``None`` when Tailscale is not running or unreachable.
        """
        # 1. Environment variable
        env_ip = os.environ.get("TAILSCALE_IP", "").strip()
        if env_ip:
            logger.debug("Tailscale IP from env: {}", env_ip)
            return env_ip

        # 2. CLI command
        cli_ip = TailscaleHelper._detect_via_cli()
        if cli_ip:
            logger.debug("Tailscale IP from CLI: {}", cli_ip)
            return cli_ip

        # 3. Network interface scan
        iface_ip = TailscaleHelper._detect_via_interface()
        if iface_ip:
            logger.debug("Tailscale IP from interface scan: {}", iface_ip)
            return iface_ip

        logger.warning("Tailscale IP not detected — FileLinker will be disabled")
        return None

    @staticmethod
    def _detect_via_cli() -> str | None:
        tailscale_bin = shutil.which("tailscale")
        if not tailscale_bin:
            return None
        try:
            result = subprocess.run(
                [tailscale_bin, "ip", "-4"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                ip = result.stdout.strip()
                if ip:
                    return ip
        except (subprocess.TimeoutExpired, OSError):
            pass
        return None

    @staticmethod
    def _detect_via_interface() -> str | None:
        """Scan network interfaces whose name starts with ``tailscale``."""
        try:
            import socket
            for iface_name, addrs in _get_interface_addresses().items():
                if iface_name.lower().startswith("tailscale"):
                    for addr in addrs:
                        if addr.family == socket.AF_INET:
                            return addr.address
        except Exception:
            pass
        return None

    # ── Health Check ─────────────────────────────────────────────────────────

    @staticmethod
    async def health_check() -> dict:
        """
        Check Tailscale connectivity.

        Returns a dict:
            {"running": bool, "ip": str | None, "peers": int}
        """
        tailscale_bin = shutil.which("tailscale")
        if not tailscale_bin:
            return {"running": False, "ip": None, "peers": 0}

        try:
            proc = await asyncio.create_subprocess_exec(
                tailscale_bin, "status", "--json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            if proc.returncode != 0:
                return {"running": False, "ip": None, "peers": 0}

            import json
            data = json.loads(stdout.decode())

            # Extract self IP
            self_node = data.get("Self", {})
            ip = None
            for addr in self_node.get("TailscaleIPs", []):
                if "." in addr:  # IPv4
                    ip = addr
                    break

            # Count online peers
            peers = sum(
                1
                for node in data.get("Peer", {}).values()
                if node.get("Online", False)
            )

            return {"running": True, "ip": ip, "peers": peers}
        except Exception as exc:
            logger.debug("Tailscale health check failed: {}", exc)
            return {"running": False, "ip": None, "peers": 0}


def _get_interface_addresses() -> dict:
    """Return {interface_name: [addr, ...]} using the ``netifaces``-free stdlib approach."""
    import socket
    result: dict[str, list] = {}
    try:
        # Python 3.11+ has socket.if_nameindex()
        for _, name in socket.if_nameindex():
            try:
                infos = socket.getaddrinfo(name, None, socket.AF_INET)
                result[name] = [
                    type("Addr", (), {"family": info[0], "address": info[4][0]})()
                    for info in infos
                ]
            except (socket.gaierror, OSError):
                continue
    except (AttributeError, OSError):
        pass
    return result
