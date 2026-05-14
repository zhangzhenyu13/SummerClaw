"""ReMe memory store — adapter wrapping ReMeLight's storage layer."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.utils.gitstore import GitStore


class ReMeStore:
    """Adapter that wraps ReMeLight storage and provides the MemoryStore interface.

    ReMeLight manages its own dialog files internally; this adapter maintains a
    companion JSONL history file so that nanobot's cursor-based Dream and
    Consolidator pipelines work unchanged.
    """

    _DEFAULT_MAX_HISTORY = 1000

    def __init__(
        self,
        reme_light: Any,
        workspace: Path,
        max_history_entries: int = _DEFAULT_MAX_HISTORY,
        algo_name: str | None = None,
    ):
        self.reme_light = reme_light
        self.workspace = workspace
        self.max_history_entries = max_history_entries

        if algo_name:
            self._algo_name = algo_name
            self.memory_dir = workspace / "memory" / algo_name
        else:
            self._algo_name = None
            self.memory_dir = workspace / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._history_file = self.memory_dir / "remem_history.jsonl"
        self._cursor_file = self.memory_dir / ".remem_cursor"
        self._dream_cursor_file = self.memory_dir / ".remem_dream_cursor"

        # File paths for SOUL.md and USER.md (interop with Dream pipeline)
        if algo_name:
            self.soul_file = self.memory_dir / "SOUL.md"
            self.user_file = self.memory_dir / "USER.md"
            # MEMORY.md is also in algorithm dir
            self._memory_file = self.memory_dir / "MEMORY.md"
        else:
            self.soul_file = workspace / "SOUL.md"
            self.user_file = workspace / "USER.md"
            self._memory_file = self.workspace / "MEMORY.md"

        self.memory_file = self._memory_file
        self.history_file = self._history_file

        # Git integration for line age tracking and auto-commit
        self._git = GitStore(
            workspace,
            tracked_files=[
                f"memory/{algo_name}/SOUL.md" if algo_name else "SOUL.md",
                f"memory/{algo_name}/USER.md" if algo_name else "USER.md",
                f"memory/{algo_name}/MEMORY.md" if algo_name else "MEMORY.md",
            ],
        )

        # Migrate legacy shared files if needed
        if algo_name:
            self._migrate_from_legacy()

    def _migrate_from_legacy(self) -> None:
        """Migrate data from the legacy shared location to the algorithm-specific dir."""
        from nanobot.memory.migrate import maybe_migrate_legacy_files
        old_memory_dir = self.workspace / "memory"
        maybe_migrate_legacy_files(
            memory_dir=self.memory_dir,
            old_memory_dir=old_memory_dir,
            old_workspace=self.workspace,
            files=[
                "remem_history.jsonl",
                "MEMORY.md",
                "SOUL.md",
                "USER.md",
                ".remem_cursor",
                ".remem_dream_cursor",
            ],
        )

    @property
    def git(self) -> GitStore:
        return self._git

    # -- MEMORY.md (long-term facts) -----------------------------------------

    def read_memory(self) -> str:
        """Read the long-term memory file (MEMORY.md)."""
        try:
            return self._memory_file.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def write_memory(self, content: str) -> None:
        """Write the long-term memory file (MEMORY.md)."""
        self._memory_file.write_text(content, encoding="utf-8")

    # -- SOUL.md -------------------------------------------------------------

    def read_soul(self) -> str:
        """Read the agent personality file (SOUL.md)."""
        return self._read_file(self.soul_file)

    def write_soul(self, content: str) -> None:
        """Write the agent personality file (SOUL.md)."""
        self.soul_file.write_text(content, encoding="utf-8")

    # -- USER.md -------------------------------------------------------------

    def read_user(self) -> str:
        """Read the user profile file (USER.md)."""
        return self._read_file(self.user_file)

    def write_user(self, content: str) -> None:
        """Write the user profile file (USER.md)."""
        self.user_file.write_text(content, encoding="utf-8")

    # -- generic file reading ------------------------------------------------

    @staticmethod
    def _read_file(path: Path) -> str:
        """Read a text file, returning '' if not found."""
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    # -- history.jsonl — append-only, JSONL format ---------------------------

    def append_history(self, entry: str) -> int:
        """Append *entry* to companion history JSONL and return its cursor."""
        cursor = self._next_cursor()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        record = {
            "cursor": cursor,
            "timestamp": ts,
            "content": entry,
        }
        with open(self._history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._cursor_file.write_text(str(cursor), encoding="utf-8")
        return cursor

    def _next_cursor(self) -> int:
        """Read the current cursor counter and return next value."""
        if self._cursor_file.exists():
            try:
                return int(self._cursor_file.read_text(encoding="utf-8").strip()) + 1
            except (ValueError, OSError):
                pass
        last = self._read_last_entry()
        if last and last.get("cursor"):
            return last["cursor"] + 1
        return 1

    def read_unprocessed_history(self, since_cursor: int) -> list[dict[str, Any]]:
        """Return history entries with cursor > *since_cursor*."""
        return [e for e in self._read_entries() if e.get("cursor", 0) > since_cursor]

    def compact_history(self) -> None:
        """Drop oldest entries if the file exceeds *max_history_entries*."""
        if self.max_history_entries <= 0:
            return
        entries = self._read_entries()
        if len(entries) <= self.max_history_entries:
            return
        kept = entries[-self.max_history_entries :]
        self._write_entries(kept)

    # -- JSONL helpers -------------------------------------------------------

    def _read_entries(self) -> list[dict[str, Any]]:
        """Read all entries from history JSONL."""
        entries: list[dict[str, Any]] = []
        try:
            with open(self._history_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except FileNotFoundError:
            pass
        return entries

    def _read_last_entry(self) -> dict[str, Any] | None:
        """Read the last entry from the JSONL file efficiently."""
        try:
            with open(self._history_file, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                if size == 0:
                    return None
                read_size = min(size, 4096)
                f.seek(size - read_size)
                data = f.read().decode("utf-8")
                lines = [line for line in data.split("\n") if line.strip()]
                if not lines:
                    return None
                return json.loads(lines[-1])
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            return None

    def _write_entries(self, entries: list[dict[str, Any]]) -> None:
        """Overwrite history JSONL with the given entries."""
        with open(self._history_file, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # -- dream cursor --------------------------------------------------------

    def get_last_dream_cursor(self) -> int:
        if self._dream_cursor_file.exists():
            try:
                return int(self._dream_cursor_file.read_text(encoding="utf-8").strip())
            except (ValueError, OSError):
                pass
        return 0

    def set_last_dream_cursor(self, cursor: int) -> None:
        self._dream_cursor_file.write_text(str(cursor), encoding="utf-8")

    # -- context injection ---------------------------------------------------

    def get_memory_context(self) -> str:
        """Return long-term memory formatted for context injection."""
        long_term = self.read_memory()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    # -- message formatting utility ------------------------------------------

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        lines = []
        for message in messages:
            if not message.get("content"):
                continue
            tools = (
                f" [tools: {', '.join(message['tools_used'])}]"
                if message.get("tools_used")
                else ""
            )
            lines.append(
                f"[{message.get('timestamp', '?')[:16]}] "
                f"{message['role'].upper()}{tools}: {message['content']}"
            )
        return "\n".join(lines)

    def raw_archive(self, messages: list[dict]) -> None:
        """Fallback: dump raw messages to history JSONL without LLM summarization."""
        self.append_history(
            f"[RAW] {len(messages)} messages\n"
            f"{self._format_messages(messages)}"
        )
        logger.warning(
            "Memory consolidation degraded: raw-archived {} messages", len(messages)
        )
