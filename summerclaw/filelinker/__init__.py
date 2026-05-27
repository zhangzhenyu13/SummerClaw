"""FileLinker — P2P large file direct transfer via Tailscale."""

from __future__ import annotations

__all__ = ["FileLinkerService", "FileLinkerMiddleware"]


def __getattr__(name: str):
    """Lazy imports to avoid circular dependencies and optional-dep errors."""
    if name == "FileLinkerService":
        from summerclaw.filelinker.service import FileLinkerService
        return FileLinkerService
    if name == "FileLinkerMiddleware":
        from summerclaw.filelinker.middleware import FileLinkerMiddleware
        return FileLinkerMiddleware
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
