"""MastraOM store — file I/O for OBSERVATIONS.md, history.jsonl, om-ops.jsonl, SOUL.md, USER.md.

Based on Mastra's Observational Memory architecture:
- OBSERVATIONS.md: the observation log (replaces raw message history as it grows)
- history.jsonl: raw conversation history (append-only, used by Dream for analysis)
- om-ops.jsonl: OM operation log (Observer/Reflector/Buffer activation summaries)
- SOUL.md / USER.md: permanent persona files

The Observer agent converts raw messages → observations; the Reflector condenses them.
This store is the pure I/O layer — no LLM logic.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.utils.helpers import ensure_dir, strip_think
from nanobot.utils.gitstore import GitStore
from nanobot.memory.migrate import maybe_migrate_legacy_files


class MastraOMStore:
    """File I/O layer for MastraOM memory files.

    Files managed:
    - memory/OBSERVATIONS.md  — dense observation log
    - memory/history.jsonl     — raw conversation history (append-only JSONL)
    - memory/om-ops.jsonl      — OM operation log (summary entries)
    - SOUL.md / USER.md        — persona files
    - memory/.cursor           — history cursor
    - memory/.dream_cursor     — Dream processing cursor
    - memory/.obs_cursor       — observation processing cursor
    """

    _DEFAULT_MAX_HISTORY = 1000
    _LEGACY_ENTRY_START_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}[^\]]*)\]\s*")
    _LEGACY_TIMESTAMP_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]\s*")
    _LEGACY_RAW_MESSAGE_RE = re.compile(
        r"^\[\d{4}-\d{2}-\d{2}[^\]]*\]\s+[A-Z][A-Z0-9_]*(?:\s+\[tools:\s*[^\]]+\])?:"
    )

    def __init__(
        self,
        workspace: Path,
        max_history_entries: int = _DEFAULT_MAX_HISTORY,
        algo_name: str | None = None,
    ):
        self.workspace = workspace
        self.max_history_entries = max_history_entries

        if algo_name:
            self._algo_name = algo_name
            self.memory_dir = ensure_dir(workspace / "memory" / algo_name)
        else:
            self._algo_name = None
            self.memory_dir = ensure_dir(workspace / "memory")

        self.observations_file = self.memory_dir / "OBSERVATIONS.md"
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "history.jsonl"
        self.om_ops_file = self.memory_dir / "om-ops.jsonl"
        self.legacy_history_file = self.memory_dir / "HISTORY.md"
        if algo_name:
            self.soul_file = self.memory_dir / "SOUL.md"
            self.user_file = self.memory_dir / "USER.md"
        else:
            self.soul_file = workspace / "SOUL.md"
            self.user_file = workspace / "USER.md"
        self._cursor_file = self.memory_dir / ".cursor"
        self._dream_cursor_file = self.memory_dir / ".dream_cursor"
        self._obs_cursor_file = self.memory_dir / ".obs_cursor"
        self._generation_file = self.memory_dir / ".om_generation"
        self._git = GitStore(
            workspace,
            tracked_files=[
                f"memory/{algo_name}/SOUL.md" if algo_name else "SOUL.md",
                f"memory/{algo_name}/USER.md" if algo_name else "USER.md",
                f"memory/{algo_name}/OBSERVATIONS.md" if algo_name else "memory/OBSERVATIONS.md",
                f"memory/{algo_name}/MEMORY.md" if algo_name else "memory/MEMORY.md",
            ],
        )

        if algo_name:
            self._migrate_from_legacy()

        self._maybe_migrate_legacy_history()

    def _migrate_from_legacy(self) -> None:
        """Migrate data from the legacy shared location to the algorithm-specific dir."""
        old_memory_dir = self.workspace / "memory"
        old_workspace = self.workspace
        maybe_migrate_legacy_files(
            memory_dir=self.memory_dir,
            old_memory_dir=old_memory_dir,
            old_workspace=old_workspace,
            files=[
                "OBSERVATIONS.md",
                "history.jsonl",
                "HISTORY.md",
                "MEMORY.md",
                "SOUL.md",
                "USER.md",
                ".cursor",
                ".dream_cursor",
                ".obs_cursor",
                ".om_generation",
            ],
        )

    @property
    def git(self) -> GitStore:
        return self._git

    # -- generic helpers -----------------------------------------------------

    @staticmethod
    def read_file(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def _maybe_migrate_legacy_history(self) -> None:
        """One-time upgrade from legacy HISTORY.md to history.jsonl."""
        if not self.legacy_history_file.exists():
            return
        if self.history_file.exists() and self.history_file.stat().st_size > 0:
            return

        try:
            legacy_text = self.legacy_history_file.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            logger.exception("Failed to read legacy HISTORY.md for migration")
            return

        entries = self._parse_legacy_history(legacy_text)
        try:
            if entries:
                self._write_entries(entries)
                last_cursor = entries[-1]["cursor"]
                self._cursor_file.write_text(str(last_cursor), encoding="utf-8")
                self._dream_cursor_file.write_text(str(last_cursor), encoding="utf-8")

            backup_path = self._next_legacy_backup_path()
            self.legacy_history_file.replace(backup_path)
            logger.info(
                "Migrated legacy HISTORY.md to history.jsonl ({} entries)",
                len(entries),
            )
        except Exception:
            logger.exception("Failed to migrate legacy HISTORY.md")

    def _parse_legacy_history(self, text: str) -> list[dict[str, Any]]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return []

        fallback_timestamp = self._legacy_fallback_timestamp()
        entries: list[dict[str, Any]] = []
        chunks = self._split_legacy_history_chunks(normalized)

        for cursor, chunk in enumerate(chunks, start=1):
            timestamp = fallback_timestamp
            content = chunk
            match = self._LEGACY_TIMESTAMP_RE.match(chunk)
            if match:
                timestamp = match.group(1)
                remainder = chunk[match.end():].lstrip()
                if remainder:
                    content = remainder

            entries.append({
                "cursor": cursor,
                "timestamp": timestamp,
                "content": content,
            })
        return entries

    def _split_legacy_history_chunks(self, text: str) -> list[str]:
        lines = text.split("\n")
        chunks: list[str] = []
        current: list[str] = []
        saw_blank_separator = False

        for line in lines:
            if saw_blank_separator and line.strip() and current:
                chunks.append("\n".join(current).strip())
                current = [line]
                saw_blank_separator = False
                continue
            if self._should_start_new_legacy_chunk(line, current):
                chunks.append("\n".join(current).strip())
                current = [line]
                saw_blank_separator = False
                continue
            current.append(line)
            saw_blank_separator = not line.strip()

        if current:
            chunks.append("\n".join(current).strip())
        return [chunk for chunk in chunks if chunk]

    def _should_start_new_legacy_chunk(self, line: str, current: list[str]) -> bool:
        if not current:
            return False
        if not self._LEGACY_ENTRY_START_RE.match(line):
            return False
        if self._is_raw_legacy_chunk(current) and self._LEGACY_RAW_MESSAGE_RE.match(line):
            return False
        return True

    def _is_raw_legacy_chunk(self, lines: list[str]) -> bool:
        first_nonempty = next((line for line in lines if line.strip()), "")
        match = self._LEGACY_TIMESTAMP_RE.match(first_nonempty)
        if not match:
            return False
        return first_nonempty[match.end():].lstrip().startswith("[RAW]")

    def _legacy_fallback_timestamp(self) -> str:
        try:
            return datetime.fromtimestamp(
                self.legacy_history_file.stat().st_mtime,
            ).strftime("%Y-%m-%d %H:%M")
        except OSError:
            return datetime.now().strftime("%Y-%m-%d %H:%M")

    def _next_legacy_backup_path(self) -> Path:
        candidate = self.memory_dir / "HISTORY.md.bak"
        suffix = 2
        while candidate.exists():
            candidate = self.memory_dir / f"HISTORY.md.bak.{suffix}"
            suffix += 1
        return candidate

    # -- OBSERVATIONS.md (observation log) -----------------------------------

    def read_observations(self) -> str:
        """Read the current observation log."""
        content = self.read_file(self.observations_file)
        if content.strip():
            logger.debug(
                "[OM:store] read {} chars from OBSERVATIONS.md (gen={})",
                len(content), self.get_generation_count(),
            )
        return content

    def write_observations(self, content: str) -> None:
        """Write observation log content."""
        self.observations_file.write_text(content, encoding="utf-8")
        logger.debug("[OM:store] wrote {} chars to OBSERVATIONS.md", len(content))

    def append_observations(self, new_observations: str, cycle_id: str | None = None) -> str:
        """Append new observations to the observation log.

        Returns the cycle_id used (generated if not provided).
        """
        cycle_id = cycle_id or str(uuid.uuid4())
        existing = self.read_observations()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")

        header = f"\n\n## Observation Cycle {cycle_id[:8]} — {ts}\n"
        new_block = header + new_observations.strip()

        if existing.strip():
            new_content = existing.rstrip() + new_block
        else:
            new_content = f"# Observational Memory\n{new_block.strip()}"

        self.observations_file.write_text(new_content, encoding="utf-8")
        logger.info(
            "[OM:store] appended {} chars observations (cycle={}, total={} chars)",
            len(new_observations), cycle_id[:8], len(new_content),
        )
        return cycle_id

    def replace_observations(self, content: str) -> None:
        """Replace the entire observation log (used by Reflector condensation)."""
        self.observations_file.write_text(content, encoding="utf-8")
        logger.info(
            "[OM:store] replaced observations — Reflector condensed to {} chars (gen={})",
            len(content), self.get_generation_count() + 1,  # gen will be incremented after
        )

    # -- Generation tracking (Reflector cycle counter) -----------------------

    def get_generation_count(self) -> int:
        """Return the current reflection generation number."""
        if self._generation_file.exists():
            try:
                return int(self._generation_file.read_text(encoding="utf-8").strip())
            except (ValueError, OSError):
                pass
        return 0

    def increment_generation(self) -> int:
        """Increment and return the new generation count."""
        gen = self.get_generation_count() + 1
        self._generation_file.write_text(str(gen), encoding="utf-8")
        return gen

    # -- MEMORY.md (Dream output) --------------------------------------------

    def read_memory(self) -> str:
        memory_file = self.memory_dir / "MEMORY.md"
        content = self.read_file(memory_file)
        if content.strip():
            logger.debug("[OM:store] read {} chars from MEMORY.md", len(content))
        return content

    def write_memory(self, content: str) -> None:
        memory_file = self.memory_dir / "MEMORY.md"
        memory_file.write_text(content, encoding="utf-8")

    # -- SOUL.md -------------------------------------------------------------

    def read_soul(self) -> str:
        return self.read_file(self.soul_file)

    def write_soul(self, content: str) -> None:
        self.soul_file.write_text(content, encoding="utf-8")

    # -- USER.md -------------------------------------------------------------

    def read_user(self) -> str:
        return self.read_file(self.user_file)

    def write_user(self, content: str) -> None:
        self.user_file.write_text(content, encoding="utf-8")

    # -- context injection (used by context.py) ------------------------------

    def get_memory_context(self) -> str:
        """Return observations + MEMORY.md as combined message context.

        Observations are presented as message-like records rather than an opaque
        block. This allows any consumer (Dream, AgentLoop context, etc.) to
        treat observations as part of the message stream without knowing about
        the Observer/Reflector internals.
        """
        observations = self.read_observations()
        long_term = self.read_memory()

        parts: list[str] = []
        if observations:
            obs_records = self._observations_as_records(observations)
            if obs_records.strip():
                parts.append(
                    "## Past Conversation Records\n\n"
                    "The following records capture key insights distilled from past "
                    "conversations. Treat them as remembered conversation history — "
                    "they carry higher informational priority than raw messages:\n\n"
                    f"{obs_records}"
                )
        if long_term:
            parts.append(f"## Long-term Memory\n{long_term}")

        combined = "\n\n".join(parts) if parts else ""
        if combined:
            logger.info(
                "[OM:context] injecting memory context: {} chars (obs={} chars, MEMORY.md={} chars, gen={})",
                len(combined),
                len(observations) if observations else 0,
                len(long_term) if long_term else 0,
                self.get_generation_count(),
            )
        return combined

    @staticmethod
    def _observations_as_records(observations_text: str) -> str:
        """Convert raw OBSERVATIONS.md text into message-like records.

        Strips markdown headers and observation group tags, keeps only the
        fact lines (starting with *), and presents them as conversational
        memory records.
        """
        lines: list[str] = []
        for line in observations_text.split("\n"):
            stripped = line.strip()
            # Skip markdown headers, empty lines, and observation group tags
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            if stripped.startswith("<observation-group") or stripped.startswith("</observation-group"):
                continue
            # Keep fact lines (starting with *)
            if stripped.startswith("*"):
                # Remove the leading "* " and present as a record
                fact = stripped[1:].strip()
                if fact:
                    lines.append(fact)
        return "\n".join(lines) if lines else ""

    # -- history.jsonl — append-only, JSONL format ---------------------------

    def append_history(self, entry: str) -> int:
        """Append a raw conversation entry to history.jsonl and return its cursor.

        This stores raw/full message content, used by Dream for deep analysis.
        OM pipeline operation summaries go to om-ops.jsonl via append_om_ops()."""
        cursor = self._next_cursor()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        record = {
            "cursor": cursor,
            "timestamp": ts,
            "content": strip_think(entry.rstrip()) or entry.rstrip(),
        }
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._cursor_file.write_text(str(cursor), encoding="utf-8")
        logger.debug("[OM:store] appended history entry cursor={}", cursor)
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
        kept = entries[-self.max_history_entries:]
        self._write_entries(kept)

    # -- JSONL helpers -------------------------------------------------------

    def _read_entries(self) -> list[dict[str, Any]]:
        """Read all entries from history.jsonl."""
        entries: list[dict[str, Any]] = []
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
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
            with open(self.history_file, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                if size == 0:
                    return None
                read_size = min(size, 4096)
                f.seek(size - read_size)
                data = f.read().decode("utf-8")
                lines = [l for l in data.split("\n") if l.strip()]
                if not lines:
                    return None
                return json.loads(lines[-1])
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            return None

    def _write_entries(self, entries: list[dict[str, Any]]) -> None:
        """Overwrite history.jsonl with the given entries."""
        with open(self.history_file, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # -- om-ops.jsonl — OM operation log (Observer/Reflector/Buffer summaries) --

    def append_om_ops(self, entry: str) -> None:
        """Append an OM operation summary entry to om-ops.jsonl.

        Unlike history.jsonl, om-ops.jsonl entries have no cursor and are
        purely for debugging/tracking OM pipeline operations.
        """
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        record = {
            "timestamp": ts,
            "content": strip_think(entry.rstrip()) or entry.rstrip(),
        }
        with open(self.om_ops_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.debug("[OM:store] appended om-ops entry: {}", entry[:80])

    def read_om_ops(self) -> list[dict[str, Any]]:
        """Read all entries from om-ops.jsonl."""
        entries: list[dict[str, Any]] = []
        try:
            with open(self.om_ops_file, "r", encoding="utf-8") as f:
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

    # -- observation cursor --------------------------------------------------

    def get_last_obs_cursor(self) -> int:
        """Return the last processed observation cursor (for Consolidator)."""
        if self._obs_cursor_file.exists():
            try:
                return int(self._obs_cursor_file.read_text(encoding="utf-8").strip())
            except (ValueError, OSError):
                pass
        return 0

    def set_last_obs_cursor(self, cursor: int) -> None:
        """Update the last processed observation cursor."""
        self._obs_cursor_file.write_text(str(cursor), encoding="utf-8")

    # -- message formatting utility ------------------------------------------

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        lines = []
        for message in messages:
            if not message.get("content"):
                continue
            tools = f" [tools: {', '.join(message['tools_used'])}]" if message.get("tools_used") else ""
            lines.append(
                f"[{message.get('timestamp', '?')[:16]}] {message['role'].upper()}{tools}: {message['content']}"
            )
        return "\n".join(lines)

    def raw_archive(self, messages: list[dict]) -> None:
        """Fallback: dump raw messages to history.jsonl without LLM summarization."""
        self.append_history(
            f"[RAW] {len(messages)} messages\n"
            f"{self._format_messages(messages)}"
        )
        self.append_om_ops(
            f"[RAW-ARCHIVE] {len(messages)} messages raw-dumped (Observer failed/degenerate)"
        )
        logger.warning(
            "MastraOM consolidation degraded: raw-archived {} messages", len(messages)
        )
