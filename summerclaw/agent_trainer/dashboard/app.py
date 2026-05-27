"""Training dashboard — Gradio WebUI + FastAPI status endpoints.

Provides:
  - Real-time training progress visualization
  - Manual control (pause, cancel, deploy)
  - REST API for external status queries

The dashboard runs as a background web server started by the ``/train``
channel command. The URL is returned to the channel for the user to
open in a browser.

This module is intentionally thin — all logic lives in sibling modules:
  - ``task_utils.py``: task scanning / caching
  - ``api.py``: FastAPI REST endpoints
  - ``ui_state.py``: UI state & callbacks
  - ``ui_data.py``: Data Tab layout
  - ``ui.py``: main Gradio Blocks layout
"""
from __future__ import annotations

import socket
import threading
import time
from pathlib import Path
from typing import Any

from loguru import logger

from summerclaw.agent_trainer.engine.trainer import TrainerEngine
from summerclaw.agent_trainer.dashboard.task_utils import _default_train_root
from summerclaw.agent_trainer.dashboard.api import _create_api
from summerclaw.agent_trainer.dashboard.ui import create_gradio_app


# ── Helpers ──────────────────────────────────────────────────────────────

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


# ── Dashboard server ─────────────────────────────────────────────────────

class DashboardServer:
    """Manages the Gradio + FastAPI dashboard server.

    The server runs in a background thread. Both the local (LAN) URL and
    the optional public (share-tunnel) URL are returned for the channel
    and console to display.
    """

    def __init__(
        self,
        engine: TrainerEngine,
        host: str = "0.0.0.0",
        port: int = 7860,
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

        # If share is enabled, wait for tunnel URL (but bail out early on failure)
        if self.share and not self._share_url:
            for _ in range(40):  # up to 20s
                if self._share_url or self._share_done:
                    break
                time.sleep(0.5)

        return self._share_url or self._local_url or f"http://{self.host}:{self.port}"

    def _run(self) -> None:
        """Run the dashboard server (in background thread).

        Uses ``mount_gradio_app`` + uvicorn to bind ``0.0.0.0`` (LAN accessible),
        then starts a Gradio share tunnel via ``setup_tunnel`` in a separate
        thread for public URL access.

        Calls ``uvicorn.Server.run()`` directly (same pattern as validated in
        the integration test) to avoid event-loop management pitfalls.
        """
        try:
            import gradio as gr
        except ImportError:
            logger.error("Gradio not installed; cannot start dashboard")
            return

        gradio_app = create_gradio_app(
            self.engine,
            train_root=self.train_root,
            active_sessions=self.active_sessions,
        )

        if gradio_app is None:
            logger.error("Failed to create Gradio app")
            return

        lan_ip = _get_local_ip()
        local_url = f"http://{lan_ip}:{self.port}"

        # ── Mount Gradio onto FastAPI + uvicorn ────────────────────────
        try:
            from contextlib import asynccontextmanager
            from fastapi import FastAPI as _FastAPI
            from gradio import mount_gradio_app
            import uvicorn

            server_ready = threading.Event()

            @asynccontextmanager
            async def _lifespan(app):
                # Runs inside the uvicorn event loop after startup
                self._local_url = local_url
                self._ready = True
                server_ready.set()
                yield

            api_app = _FastAPI(lifespan=_lifespan)

            # Add CORS middleware for external API access
            try:
                from fastapi.middleware.cors import CORSMiddleware
                api_app.add_middleware(
                    CORSMiddleware,
                    allow_origins=["*"],
                    allow_methods=["*"],
                    allow_headers=["*"],
                )
            except ImportError:
                pass

            # Mount REST API endpoints (status, logs, history, etc.)
            api_router = _create_api(self.engine)
            if api_router:
                api_app.include_router(api_router)

            mount_gradio_app(
                api_app,
                gradio_app,
                path="/",
                server_name=lan_ip,
                server_port=self.port,
            )

            config = uvicorn.Config(
                api_app,
                host=self.host,
                port=self.port,
                log_level="warning",
                access_log=False,
            )
            server = uvicorn.Server(config)
            self._app = server

            # ── Share tunnel thread ──────────────────────────────────────
            def _share_tunnel_thread():
                server_ready.wait(timeout=15)
                if not server_ready.is_set():
                    return
                time.sleep(1)  # brief delay for accept
                print(f"  * Running on local URL:  {local_url}")
                logger.info("Dashboard local URL: {}", local_url)
                if self.share:
                    self._share_url = self._start_share_tunnel()
                    self._share_done = True
                    if self._share_url:
                        print(f"  * Running on public URL: {self._share_url}")
                        logger.info("Dashboard public URL: {}", self._share_url)
                    else:
                        print(f"  * Share tunnel unavailable — use local URL: {local_url}")
                        logger.warning(
                            "Gradio share tunnel failed (network unreachable or "
                            "api.gradio.app not resolvable). Dashboard is still "
                            "accessible via local URL: {}", local_url,
                        )

            if self.share:
                threading.Thread(target=_share_tunnel_thread, daemon=True).start()
            else:
                # No share — just print local URL once ready
                def _print_local():
                    server_ready.wait(timeout=15)
                    if server_ready.is_set():
                        time.sleep(1)
                        print(f"  * Running on local URL:  {local_url}")
                        logger.info("Dashboard local URL: {}", local_url)
                threading.Thread(target=_print_local, daemon=True).start()

            # server.run() creates its own event loop and blocks until shutdown
            server.run()
        except Exception as exc:
            logger.exception("Dashboard uvicorn launch failed: {}", exc)
            return

    def _start_share_tunnel(self) -> str | None:
        """Start a Gradio share tunnel and return the public URL.

        Calls ``gradio.networking.setup_tunnel()`` with the correct
        ``local_host`` / ``local_port`` parameter names (Gradio 6.x).
        ``share_token`` must be a non-None string (used as frpc CLI arg).
        """
        import secrets
        share_token = secrets.token_urlsafe(32)
        try:
            from gradio.networking import setup_tunnel
            url = setup_tunnel(
                local_host="127.0.0.1",
                local_port=self.port,
                share_token=share_token,
                share_server_address=None,
                share_server_tls_certificate=None,
            )
            return url
        except TypeError as exc:
            # Gradio API signature changed — introspect and retry
            logger.debug("setup_tunnel signature mismatch, retrying: {}", exc)
            try:
                import inspect
                sig = inspect.signature(setup_tunnel)
                params = list(sig.parameters.keys())
                kwargs: dict[str, object] = {}
                for p in params:
                    if p in ("local_host", "server_name"):
                        kwargs[p] = "127.0.0.1"
                    elif p in ("local_port", "server_port"):
                        kwargs[p] = self.port
                    elif p in ("share_token",):
                        kwargs[p] = share_token
                    elif p in ("share_server_address",):
                        kwargs[p] = None
                    elif p in ("share_server_tls_certificate",):
                        kwargs[p] = None
                    else:
                        kwargs[p] = None
                url = setup_tunnel(**kwargs)
                return url
            except Exception:
                logger.debug("setup_tunnel retry also failed", exc_info=True)
                return None
        except Exception:
            logger.debug("setup_tunnel call failed", exc_info=True)
            return None

    def stop(self) -> None:
        """Stop the dashboard server."""
        if self._app:
            try:
                if hasattr(self._app, 'should_exit'):
                    # uvicorn Server
                    self._app.should_exit = True
                else:
                    self._app.close()
            except Exception:
                pass
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
