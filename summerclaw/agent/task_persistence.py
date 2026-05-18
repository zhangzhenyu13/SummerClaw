"""Persistent task registry for subagent and main-session task recovery.

On spawn, a PersistedTask is written to {workspace}/task_registry/{task_id}.json.
On normal completion or cancellation, the file is deleted.
On restart, any remaining files represent tasks that were interrupted mid-flight.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger


@dataclass
class PersistedTask:
    """Metadata for a task that should survive process restarts."""

    task_id: str
    task_type: str       # "subagent" | "main_session"
    label: str
    description: str     # original task text / user message
    origin_channel: str
    origin_chat_id: str
    session_key: str | None
    depth: int           # subagent recursion depth (0 for main_session tasks)
    started_at: str      # ISO 8601 datetime string

    def elapsed_description(self) -> str:
        """Return a human-readable description of how long ago this task started."""
        try:
            started = datetime.fromisoformat(self.started_at)
            # Normalise to UTC-aware if naive
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            now = datetime.now(tz=timezone.utc)
            delta = now - started
            seconds = int(delta.total_seconds())
            if seconds < 60:
                return f"{seconds}s ago"
            minutes = seconds // 60
            if minutes < 60:
                return f"{minutes}m ago"
            hours = minutes // 60
            return f"{hours}h {minutes % 60}m ago"
        except Exception:
            return self.started_at

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PersistedTask":
        return cls(
            task_id=data["task_id"],
            task_type=data.get("task_type", "subagent"),
            label=data["label"],
            description=data["description"],
            origin_channel=data["origin_channel"],
            origin_chat_id=data["origin_chat_id"],
            session_key=data.get("session_key"),
            depth=data.get("depth", 0),
            started_at=data["started_at"],
        )


class TaskRegistry:
    """Disk-backed registry of active/interrupted tasks.

    One JSON file per task lives at ``{workspace}/task_registry/{task_id}.json``.
    A file's mere existence signals that the task was never completed normally.
    """

    def __init__(self, workspace: Path) -> None:
        self._dir = workspace / "task_registry"
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write / delete
    # ------------------------------------------------------------------

    def save(self, task: PersistedTask) -> None:
        """Persist a task to disk.  Called immediately after spawning."""
        path = self._dir / f"{task.task_id}.json"
        try:
            path.write_text(
                json.dumps(task.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("TaskRegistry.save failed for task {}: {}", task.task_id, e)

    def remove(self, task_id: str) -> None:
        """Delete a task file.  Called when the task finishes normally."""
        path = self._dir / f"{task_id}.json"
        try:
            if path.exists():
                path.unlink()
        except Exception as e:
            logger.warning("TaskRegistry.remove failed for task {}: {}", task_id, e)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, task_id: str) -> PersistedTask | None:
        """Load a single task by ID, or ``None`` if not found."""
        path = self._dir / f"{task_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return PersistedTask.from_dict(data)
        except Exception as e:
            logger.warning("TaskRegistry.get failed for task {}: {}", task_id, e)
            return None

    def list_all(self) -> list[PersistedTask]:
        """Return all persisted tasks (i.e., all tasks interrupted during the last run)."""
        tasks: list[PersistedTask] = []
        for path in sorted(self._dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                tasks.append(PersistedTask.from_dict(data))
            except Exception as e:
                logger.warning("TaskRegistry: failed to parse {}: {}", path.name, e)
        return tasks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def make_task_id() -> str:
        """Generate a short 8-character task ID."""
        return str(uuid.uuid4())[:8]

    @staticmethod
    def now_iso() -> str:
        """Return current UTC time as ISO 8601 string."""
        return datetime.now(tz=timezone.utc).isoformat()
