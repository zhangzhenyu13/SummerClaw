"""Training dashboard — FastAPI + React WebUI.

Provides:
  - Real-time training progress visualization (React frontend)
  - REST API for all dashboard operations
  - WebSocket for real-time log/status streaming
  - Tailscale Funnel for optional public URL access

The dashboard runs as a background web server started by the ``/train``
channel command. The URL is returned to the channel for the user to
open in a browser.

This module is intentionally thin — all API logic lives in ``api.py``
and task utilities in ``task_utils.py``.
"""
from __future__ import annotations

import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from loguru import logger

from summerclaw.agent_trainer.engine.trainer import TrainerEngine
from summerclaw.agent_trainer.dashboard.task_utils import _default_train_root
from summerclaw.agent_trainer.dashboard.api import _create_api


# -- Helpers ----------------------------------------------------------------

def _get_local_ip() -> str:
    """Return the machine's LAN IP address (e.g. 192.168.x.x)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _get_frontend_dist() -> Path | None:
    """Locate the React frontend build output directory.

    Search order:
      1. ``<dashboard_dir>/frontend/dist``
      2. ``<dashboard_dir>/frontend/build``
    """
    dashboard_dir = Path(__file__).resolve().parent
    for name in ("dist", "build"):
        p = dashboard_dir / "frontend" / name
        if p.is_dir() and (p / "index.html").is_file():
            return p
    return None


def _start_tailscale_funnel(port: int) -> str | None:
    """Start a Tailscale Funnel and return the public HTTPS URL.

    Returns ``None`` if Tailscale is not available or Funnel fails.
    """
    if not shutil.which("tailscale"):
        logger.debug("tailscale CLI not found in PATH")
        return None

    try:
        result = subprocess.run(
            ["tailscale", "funnel", "--bg", str(port)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning("tailscale funnel failed: {}", result.stderr.strip())
            return None

        # Parse the URL from stdout/stderr
        # Typical output: "Available on the internet:\n\nhttps://hostname.tail1234.ts.net\n..."
        output = result.stdout + result.stderr
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("https://") and ".ts.net" in line:
                url = line.rstrip("/")
                logger.info("Tailscale Funnel URL: {}", url)
                return url

        # If no URL found in output, try `tailscale status --json` to get hostname
        try:
            status_result = subprocess.run(
                ["tailscale", "status", "--json"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if status_result.returncode == 0:
                import json
                status = json.loads(status_result.stdout)
                self_node = status.get("Self", {})
                dns_name = self_node.get("DNSName", "")
                if dns_name:
                    # DNSName has trailing dot, e.g. "hostname.tail1234.ts.net."
                    hostname = dns_name.rstrip(".")
                    url = f"https://{hostname}"
                    logger.info("Tailscale Funnel URL (from status): {}", url)
                    return url
        except Exception:
            pass

        logger.warning("Could not parse Tailscale Funnel URL from output")
        return None

    except FileNotFoundError:
        logger.debug("tailscale command not found")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("tailscale funnel timed out")
        return None
    except Exception as exc:
        logger.warning("Tailscale Funnel error: {}", exc)
        return None


def _stop_tailscale_funnel() -> None:
    """Stop the Tailscale Funnel (best-effort)."""
    try:
        subprocess.run(
            ["tailscale", "funnel", "--bg", "off"],
            capture_output=True,
            timeout=10,
        )
    except Exception:
        pass


# -- Dashboard server -------------------------------------------------------

class DashboardServer:
    """Manages the FastAPI + React dashboard server.

    The server runs in a background thread. Both the local (LAN) URL and
    the optional public (Tailscale Funnel) URL are returned for the channel
    and console to display.
    """

    def __init__(
        self,
        engine: TrainerEngine,
        host: str = "0.0.0.0",
        port: int = 443,
        share: bool = True,
        train_root: Path | None = None,
        active_sessions: dict | None = None,
    ):
        self.engine = engine
        self.host = host
        self.port = port
        self.share = share
        self.train_root = train_root or _default_train_root()
        self.active_sessions = active_sessions or {}
        self._thread: threading.Thread | None = None
        self._local_url: str | None = None
        self._share_url: str | None = None
        self._share_done: bool = False
        self._ready: bool = False
        self._app = None

    def start(self) -> str:
        """Start the dashboard server in a background thread.

        Returns the best available URL (share URL if available, else local).
        """
        if self._thread and self._thread.is_alive():
            return self._share_url or self._local_url or "unknown"

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        # Wait for server to be ready
        for _ in range(30):
            if self._ready:
                break
            time.sleep(0.5)

        # If share is enabled, wait for funnel URL (but bail out early on failure)
        if self.share and not self._share_url:
            for _ in range(40):  # up to 20s
                if self._share_url or self._share_done:
                    break
                time.sleep(0.5)

        return self._share_url or self._local_url or f"http://{self.host}:{self.port}"

    def _run(self) -> None:
        """Run the dashboard server (in background thread).

        Uses FastAPI + uvicorn to serve both the REST API and the React
        frontend static files. Optionally starts a Tailscale Funnel for
        public URL access.
        """
        try:
            from contextlib import asynccontextmanager
            from fastapi import FastAPI
            from fastapi.middleware.cors import CORSMiddleware
            from fastapi.staticfiles import StaticFiles
            from fastapi.responses import FileResponse
            import uvicorn
        except ImportError as exc:
            logger.error("FastAPI/uvicorn not installed: {}", exc)
            return

        lan_ip = _get_local_ip()
        local_url = f"http://{lan_ip}:{self.port}"

        # -- Build the FastAPI app ----------------------------------------
        server_ready = threading.Event()

        # Mount REST API endpoints
        api_router, _state = _create_api(
            self.engine,
            train_root=self.train_root,
            active_sessions=self.active_sessions,
        )

        # -- Lifespan: start scheduler inside the async event loop --------
        _scheduler_ref = getattr(_state, "scheduler", None)

        @asynccontextmanager
        async def _lifespan(app):
            self._local_url = local_url
            self._ready = True
            server_ready.set()
            if _scheduler_ref is not None:
                _scheduler_ref.start()
            try:
                yield
            finally:
                if _scheduler_ref is not None:
                    _scheduler_ref.stop()

        api_app = FastAPI(lifespan=_lifespan, title="Agent Trainer Dashboard")

        # CORS
        api_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        if api_router:
            api_app.include_router(api_router)

        # Serve React frontend
        dist_dir = _get_frontend_dist()
        if dist_dir:
            # Serve static assets (js, css, images, etc.)
            assets_dir = dist_dir / "assets"
            if assets_dir.is_dir():
                api_app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

            # Catch-all: serve index.html for SPA client-side routing
            @api_app.get("/{full_path:path}")
            async def _serve_spa(full_path: str):
                # If the path looks like a file that exists in dist, serve it
                file_path = dist_dir / full_path
                if full_path and file_path.is_file():
                    return FileResponse(str(file_path))
                # Otherwise return index.html for client-side routing
                return FileResponse(str(dist_dir / "index.html"))
        else:
            logger.warning(
                "React frontend not built. Run 'cd dashboard/frontend && npm run build' "
                "to create the UI. API-only mode is still functional."
            )

            @api_app.get("/")
            async def _api_only_root():
                return {
                    "message": "Dashboard API is running. Frontend not built.",
                    "docs": "/docs",
                }

        # -- Launch uvicorn -----------------------------------------------
        # Custom log config: only WARNING+ for all uvicorn loggers
        _uvicorn_log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "()": "uvicorn.logging.DefaultFormatter",
                    "fmt": "%(levelprefix)s %(message)s",
                    "use_colors": None,
                },
                "access": {
                    "()": "uvicorn.logging.AccessFormatter",
                    "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)d',
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "access": {
                    "formatter": "access",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "loggers": {
                "uvicorn": {"handlers": ["default"], "level": "WARNING", "propagate": False},
                "uvicorn.error": {"handlers": ["default"], "level": "WARNING", "propagate": False},
                "uvicorn.access": {"handlers": ["access"], "level": "WARNING", "propagate": False},
            },
        }

        config = uvicorn.Config(
            api_app,
            host=self.host,
            port=self.port,
            log_config=_uvicorn_log_config,
            access_log=False,
        )
        server = uvicorn.Server(config)
        self._app = server

        # -- Tailscale Funnel thread --------------------------------------
        def _funnel_thread():
            server_ready.wait(timeout=15)
            if not server_ready.is_set():
                return
            time.sleep(1)
            print(f"  * Running on local URL:  {local_url}")
            logger.info("Dashboard local URL: {}", local_url)
            if self.share:
                self._share_url = _start_tailscale_funnel(self.port)
                self._share_done = True
                if self._share_url:
                    print(f"  * Running on public URL: {self._share_url}")
                    logger.info("Dashboard public URL: {}", self._share_url)
                else:
                    print(f"  * Tailscale Funnel unavailable — use local URL: {local_url}")
                    logger.warning(
                        "Tailscale Funnel not available. Dashboard is still "
                        "accessible via local URL: {}",
                        local_url,
                    )

        if self.share:
            threading.Thread(target=_funnel_thread, daemon=True).start()
        else:
            def _print_local():
                server_ready.wait(timeout=15)
                if server_ready.is_set():
                    time.sleep(1)
                    print(f"  * Running on local URL:  {local_url}")
                    logger.info("Dashboard local URL: {}", local_url)
            threading.Thread(target=_print_local, daemon=True).start()

        # server.run() creates its own event loop and blocks until shutdown
        try:
            server.run()
        except Exception as exc:
            logger.exception("Dashboard uvicorn launch failed: {}", exc)

    def stop(self) -> None:
        """Stop the dashboard server."""
        if self._app:
            try:
                if hasattr(self._app, "should_exit"):
                    self._app.should_exit = True
                else:
                    self._app.close()
            except Exception:
                pass
        # Clean up Tailscale Funnel
        if self._share_url:
            _stop_tailscale_funnel()
        self._app = None
        self._local_url = None
        self._share_url = None
        self._share_done = False
        self._ready = False

    @property
    def url(self) -> str | None:
        """Best available URL (share URL preferred)."""
        return self._share_url or self._local_url

    @property
    def local_url(self) -> str | None:
        """Local (LAN) URL."""
        return self._local_url

    @property
    def share_url(self) -> str | None:
        """Public share-tunnel URL (None if share is disabled or unavailable)."""
        return self._share_url
