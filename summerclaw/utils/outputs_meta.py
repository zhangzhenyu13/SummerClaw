"""Outputs meta.json management — track every write_file operation in outputs/.

Provides a lightweight, thread-safe record of agent outputs with source
context (channel, chat_id, session_key) and timestamps.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from loguru import logger

# Periodically compact meta.json when it exceeds this many entries
_MAX_META_ENTRIES = 10_000
# Drop oldest entries beyond this soft cap
_COMPACT_KEEP = 5_000


class OutputMetaManager:
    """Singleton manager for outputs/meta.json.

    Thread-safe: all reads and writes are serialised via a re-entrant lock.

    Usage::

        mgr = OutputMetaManager(workspace)

        # Record a new file write
        mgr.record_entry(
            relative_path="outputs/my-project/index.html",
            channel="telegram",
            chat_id="user123",
        )

        # Read all entries (e.g. for a status command)
        for entry in mgr.get_entries():
            print(entry["path"], entry["created_at"])
    """

    _instances: dict[Path, "OutputMetaManager"] = {}
    _lock = threading.Lock()

    def __new__(cls, workspace: Path) -> "OutputMetaManager":
        key = workspace.resolve()
        with cls._lock:
            if key not in cls._instances:
                inst = super().__new__(cls)
                cls._instances[key] = inst
            return cls._instances[key]

    def __init__(self, workspace: Path) -> None:
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self._workspace = workspace.resolve()
        self._outputs_dir = self._workspace / "outputs"
        self._meta_path = self._outputs_dir / "meta.json"
        self._rwlock = threading.RLock()
        # Bootstrap: create directory + empty meta.json if missing
        self._bootstrap()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def record_entry(
        self,
        relative_path: str,
        *,
        channel: str | None = None,
        chat_id: str | None = None,
        session_key: str | None = None,
        size_bytes: int = 0,
    ) -> dict[str, Any]:
        """Record a file write in meta.json and return the entry dict."""
        entry: dict[str, Any] = {
            "id": uuid.uuid4().hex[:12],
            "path": relative_path,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "size_bytes": size_bytes,
        }
        if channel:
            entry["channel"] = channel
        if chat_id:
            entry["chat_id"] = chat_id
        if session_key:
            entry["session_key"] = session_key

        with self._rwlock:
            data = self._load()
            entries: list[dict[str, Any]] = data.get("entries", [])
            entries.append(entry)
            # Compact when oversized
            if len(entries) > _MAX_META_ENTRIES:
                logger.info(
                    "Outputs meta.json reached {} entries; compacting to {}",
                    len(entries), _COMPACT_KEEP,
                )
                entries = entries[-_COMPACT_KEEP:]
            data["entries"] = entries
            self._save(data)

        logger.debug(
            "OutputMeta: recorded {} ({} chars) for channel={} chat_id={}",
            relative_path, size_bytes, channel, chat_id,
        )
        return entry

    def get_entries(self) -> list[dict[str, Any]]:
        """Return all meta entries (most recent last)."""
        with self._rwlock:
            return list(self._load().get("entries", []))

    def get_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the *limit* most recent entries."""
        with self._rwlock:
            entries = self._load().get("entries", [])
            return entries[-limit:]

    def get_by_channel(self, channel: str) -> list[dict[str, Any]]:
        """Return entries filtered by channel."""
        with self._rwlock:
            return [
                e for e in self._load().get("entries", [])
                if e.get("channel") == channel
            ]

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _bootstrap(self) -> None:
        self._outputs_dir.mkdir(parents=True, exist_ok=True)
        if not self._meta_path.exists():
            self._save({"version": "1.0", "entries": []})

    def _load(self) -> dict[str, Any]:
        try:
            raw = self._meta_path.read_text(encoding="utf-8")
            return json.loads(raw)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"version": "1.0", "entries": []}

    def _save(self, data: dict[str, Any]) -> None:
        tmp = self._meta_path.with_name(f".meta.{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self._meta_path)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)