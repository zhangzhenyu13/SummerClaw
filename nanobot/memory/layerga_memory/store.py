"""Layerga memory store — L0-L4 layered file I/O for GenericAgent-style memory.

Extends the naive MemoryStore with layered file management:
  L0: layerga/constitution.md (meta-rules, copied from template on first init)
  L1: memory/layer_insight.txt (≤30 line minimal index)
  L2: memory/layer_facts.txt (environment-specific facts, ## [SECTION] blocks)
  L3: memory/sop/ (task SOPs and utility scripts)
  L4: memory/archives/ (compressed session archives)
"""

from __future__ import annotations

import json
import re as _re
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.memory.naive_memory.store import MemoryStore
from nanobot.utils.gitstore import GitStore
from nanobot.utils.helpers import ensure_dir


# L1 hard constraint
_L1_MAX_LINES = 30


class LayergaStore(MemoryStore):
    """Extended file I/O for the L0-L4 layered memory system.

    Inherits all standard memory operations (MEMORY.md, history.jsonl,
    SOUL.md, USER.md) and adds layered file management.
    """

    _TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

    def __init__(
        self,
        workspace: Path,
        max_history_entries: int = 1000,
        l1_max_lines: int = _L1_MAX_LINES,
    ):
        super().__init__(workspace, max_history_entries)
        self.l1_max_lines = l1_max_lines

        # L0 constitution
        self._constitution_dir = ensure_dir(workspace / "layerga")
        self._constitution_file = self._constitution_dir / "constitution.md"

        # L1 insight index
        self._insight_file = self.memory_dir / "layer_insight.txt"

        # L2 facts
        self._facts_file = self.memory_dir / "layer_facts.txt"

        # L3 SOP directory
        self._sop_dir = ensure_dir(self.memory_dir / "sop")

        # L4 archives directory
        self._archives_dir = ensure_dir(self.memory_dir / "archives")
        self._all_histories_file = self._archives_dir / "all_histories.txt"

        # Ensure L0-L2 files exist on first init
        self._ensure_layered_files()

    # ------------------------------------------------------------------
    # L0 — Meta-Rules Constitution
    # ------------------------------------------------------------------

    def read_constitution(self) -> str:
        """Read the L0 constitution from layerga/constitution.md."""
        return self.read_file(self._constitution_file)

    def ensure_constitution(self) -> None:
        """Copy the constitution template to the workspace on first init."""
        if not self._constitution_file.exists():
            template = self._TEMPLATE_DIR / "constitution.md"
            if template.exists():
                self._constitution_file.write_text(
                    template.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
                logger.info("L0 constitution initialized at {}", self._constitution_file)

    def get_constitution_summary(self) -> str:
        """Return a compact summary of L0 axioms for system prompt injection.

        Extracts the four core axioms and the decision tree quick-reference.
        """
        full = self.read_constitution()
        if not full:
            return ""
        # Extract axioms section
        axioms_match = _re.search(
            r"(## 0\. Core Axioms.*?)(?=## .*Layer Architecture)",
            full, _re.DOTALL,
        )
        if axioms_match:
            return axioms_match.group(1).strip()
        return full[:2000]

    # ------------------------------------------------------------------
    # L1 — Minimal Insight Index
    # ------------------------------------------------------------------

    def read_insight(self) -> str:
        """Read the L1 insight index from memory/layer_insight.txt."""
        return self.read_file(self._insight_file)

    def write_insight(self, content: str) -> None:
        """Overwrite the L1 insight index."""
        self._insight_file.write_text(content, encoding="utf-8")
        self._warn_if_over_l1_limit()

    def patch_insight(self, old_content: str, new_content: str) -> bool:
        """Perform a minimal local patch on L1 insight.

        Returns True if the patch was applied successfully.
        """
        current = self.read_insight()
        if old_content not in current:
            logger.warning("L1 patch: old_content not found in insight")
            return False
        updated = current.replace(old_content, new_content, 1)
        self.write_insight(updated)
        return True

    def validate_l1_lines(self) -> int:
        """Check if L1 is within the ≤30 line limit. Returns current line count."""
        content = self.read_insight()
        lines = [l for l in content.splitlines() if l.strip()]
        return len(lines)

    def _warn_if_over_l1_limit(self) -> None:
        """Warn if L1 exceeds the line limit."""
        line_count = self.validate_l1_lines()
        if line_count > self.l1_max_lines:
            logger.warning(
                "L1 insight has {} lines (limit: {}). Consider cleanup.",
                line_count, self.l1_max_lines,
            )

    def _ensure_insight(self) -> None:
        """Initialize empty L1 from template if it doesn't exist."""
        if not self._insight_file.exists():
            template = self._TEMPLATE_DIR / "insight.txt"
            if template.exists():
                self._insight_file.write_text(
                    template.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )

    # ------------------------------------------------------------------
    # L2 — Fact Base
    # ------------------------------------------------------------------

    def read_facts(self) -> str:
        """Read the L2 fact base from memory/layer_facts.txt."""
        return self.read_file(self._facts_file)

    def write_facts(self, content: str) -> None:
        """Overwrite the L2 fact base."""
        self._facts_file.write_text(content, encoding="utf-8")

    def patch_facts(self, old_content: str, new_content: str) -> bool:
        """Perform a minimal local patch on L2 facts.

        Returns True if the patch was applied successfully.
        """
        current = self.read_facts()
        if old_content not in current:
            logger.warning("L2 patch: old_content not found in facts")
            return False
        updated = current.replace(old_content, new_content, 1)
        self.write_facts(updated)
        return True

    def get_fact_sections(self) -> list[str]:
        """Return all ## [SECTION] names from L2 facts."""
        content = self.read_facts()
        sections = _re.findall(r"^##\s+\[(.+?)\]", content, _re.MULTILINE)
        return sections

    def get_recent_fact_sections(self, limit: int = 3) -> list[str]:
        """Return recently active L2 section names for context injection."""
        all_sections = self.get_fact_sections()
        return all_sections[-limit:] if len(all_sections) > limit else all_sections

    def _ensure_facts(self) -> None:
        """Initialize empty L2 from template if it doesn't exist."""
        if not self._facts_file.exists():
            template = self._TEMPLATE_DIR / "facts.txt"
            if template.exists():
                self._facts_file.write_text(
                    template.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )

    # ------------------------------------------------------------------
    # L3 — Task Records (SOPs + Scripts)
    # ------------------------------------------------------------------

    def list_sops(self) -> list[Path]:
        """List all SOP files in memory/sop/."""
        return sorted(
            p for p in self._sop_dir.iterdir()
            if p.is_file() and p.suffix == ".md"
        )

    def read_sop(self, name: str) -> str:
        """Read a SOP file by name (with or without .md extension)."""
        if not name.endswith(".md"):
            name = f"{name}.md"
        path = self._sop_dir / name
        return self.read_file(path)

    def write_sop(self, name: str, content: str) -> None:
        """Write a SOP file. Creates the sop/ dir if needed."""
        if not name.endswith(".md"):
            name = f"{name}.md"
        path = self._sop_dir / name
        path.write_text(content, encoding="utf-8")

    def patch_sop(self, name: str, old_content: str, new_content: str) -> bool:
        """Patch a SOP file minimally."""
        if not name.endswith(".md"):
            name = f"{name}.md"
        path = self._sop_dir / name
        if not path.exists():
            return False
        current = path.read_text(encoding="utf-8")
        if old_content not in current:
            return False
        path.write_text(current.replace(old_content, new_content, 1), encoding="utf-8")
        return True

    def list_scripts(self) -> list[Path]:
        """List all utility scripts in memory/sop/."""
        return sorted(
            p for p in self._sop_dir.iterdir()
            if p.is_file() and p.suffix == ".py"
        )

    def list_all_l3(self) -> list[str]:
        """Return all L3 entries as human-readable names for L1 index sync."""
        entries: list[str] = []
        for sop in self.list_sops():
            entries.append(sop.stem)
        for script in self.list_scripts():
            entries.append(script.name)
        return sorted(entries)

    # ------------------------------------------------------------------
    # L4 — Session Archives
    # ------------------------------------------------------------------

    def append_archive(self, summary: str) -> None:
        """Append a session summary to the L4 archive."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f"[{ts}] {summary.strip()}\n"
        with open(self._all_histories_file, "a", encoding="utf-8") as f:
            f.write(entry)

    def read_archive(self, month: str | None = None) -> str:
        """Read L4 archives. If *month* is given (YYYY-MM), read only that month."""
        if month:
            monthly_file = self._archives_dir / f"archive_{month}.zip"
            if monthly_file.exists():
                return f"[Archive {month} exists: {monthly_file}]"
            return ""
        return self.read_file(self._all_histories_file)

    def compact_archives(self, months_keep: int = 6) -> int:
        """Compress old L4 archives, keeping the most recent *months_keep* months.

        Returns the number of months archived.
        """
        # For simplicity, just trim all_histories.txt to last N months
        if not self._all_histories_file.exists():
            return 0
        content = self._all_histories_file.read_text(encoding="utf-8")
        if not content.strip():
            return 0
        # Keep last 50K entries (simple approach)
        lines = content.strip().split("\n")
        if len(lines) <= months_keep * 1000:
            return 0
        kept = lines[-(months_keep * 1000):]
        self._all_histories_file.write_text("\n".join(kept) + "\n", encoding="utf-8")
        trimmed = len(lines) - len(kept)
        logger.info("L4 archives compacted: trimmed {} entries", trimmed)
        return trimmed

    # ------------------------------------------------------------------
    # Index synchronization
    # ------------------------------------------------------------------

    def sync_l1_index(self) -> bool:
        """Synchronize L1 insight with current L2 sections and L3 entries.

        Scans L2 sections + L3 files → updates L1 navigation entries.
        New entries default to Tier 2 (keyword only).

        Returns True if L1 was modified.
        """
        l1 = self.read_insight()
        l2_sections = self.get_fact_sections()
        l3_entries = self.list_all_l3()

        modified = False

        # Add L2 sections not yet in L1
        for section in l2_sections:
            if section not in l1:
                # Add as Tier 2 entry (keyword only)
                lines = l1.splitlines()
                # Find insertion point before [RULES]
                insert_idx = len(lines)
                for i, line in enumerate(lines):
                    if line.strip().startswith("[RULES]"):
                        insert_idx = i
                        break
                lines.insert(insert_idx, f"L2: {section}")
                l1 = "\n".join(lines) + "\n"
                modified = True

        # Add L3 entries not yet in L1
        for entry in l3_entries:
            if entry not in l1:
                lines = l1.splitlines()
                insert_idx = len(lines)
                for i, line in enumerate(lines):
                    if line.strip().startswith("[RULES]"):
                        insert_idx = i
                        break
                lines.insert(insert_idx, f"L3: {entry}")
                l1 = "\n".join(lines) + "\n"
                modified = True

        if modified:
            self.write_insight(l1)
            logger.info("L1 index synced: {} L2 sections, {} L3 entries",
                       len(l2_sections), len(l3_entries))

        return modified

    # ------------------------------------------------------------------
    # Context injection (called by ContextBuilder)
    # ------------------------------------------------------------------

    def get_memory_context(self) -> str:
        """Return layered memory for context injection.

        Injects L1 insight (full, ≤30 lines) and L2 section pointers.
        Does NOT inject full L2/L3 content — Agent uses read_file to access them.
        """
        parts: list[str] = []

        # L0 constitution summary (first time only, via system prompt)
        # (handled separately by ContextBuilder)

        # L1 insight — always inject (≤30 lines, minimal token cost)
        l1 = self.read_insight()
        if l1.strip():
            parts.append(f"## Memory Insight (L1 Index)\n```\n{l1.strip()}\n```")

        # L2 section pointers — existence only (not content)
        l2_sections = self.get_recent_fact_sections(limit=5)
        if l2_sections:
            parts.append(
                "## Environment Facts (L2)\n"
                + "\n".join(f"- [{s}](file://memory/layer_facts.txt)" for s in l2_sections)
                + "\n(Use read_file to access full content)"
            )

        # L3 SOP pointers — existence only
        l3_sops = self.list_sops()
        if l3_sops:
            sop_names = [s.stem for s in l3_sops[:10]]
            parts.append(
                "## Task SOPs (L3)\n"
                + "\n".join(f"- {n}" for n in sop_names)
            )

        # Also include standard long-term memory as fallback
        long_term = self.read_memory()
        if long_term.strip():
            parts.append(f"## Long-term Memory\n{long_term}")

        return "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------
    # Action-verified fact logging
    # ------------------------------------------------------------------

    def log_verified_fact(
        self, source_tool: str, fact: str, args: dict[str, Any] | None = None
    ) -> None:
        """Log an action-verified fact with its source tool for audit trail."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = {
            "timestamp": ts,
            "source_tool": source_tool,
            "args": args or {},
            "fact": fact[:500],
        }
        audit_file = self.memory_dir / ".verified_facts.jsonl"
        with open(audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _ensure_layered_files(self) -> None:
        """Ensure all layered memory files exist on first initialization."""
        self.ensure_constitution()
        self._ensure_insight()
        self._ensure_facts()
        self._sop_dir.mkdir(parents=True, exist_ok=True)
        self._archives_dir.mkdir(parents=True, exist_ok=True)
