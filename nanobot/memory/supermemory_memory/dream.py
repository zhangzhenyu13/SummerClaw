"""Supermemory Dream — heavyweight cron-scheduled memory consolidation with graph updates.

Extends the naive Dream with Supermemory-specific processing:
- Temporal grounding: extracts both documentDate (conversation time) and eventDate
  (the actual time described in the conversation)
- Relational versioning: detects updates/extends/derives across history entries
- Chunk-based deep analysis: processes history in semantic chunks for better context
- Hybrid search context: injects source chunks alongside memory nodes in the prompt
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.memory.supermemory_memory.store import SupermemoryStore
from nanobot.utils.prompt_templates import render_template

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider


_STALE_THRESHOLD_DAYS = 14


class SupermemoryDream:
    """Two-phase memory processor with Supermemory graph integration.

    Phase 1: Analyze history.jsonl entries, extract facts with temporal grounding,
             detect relationships with existing memory graph nodes.

    Phase 2: Delegate to AgentRunner with read_file / edit_file tools to update
             MEMORY.md and other files based on the analysis.
    """

    def __init__(
        self,
        store: SupermemoryStore,
        provider: LLMProvider,
        model: str,
        max_batch_size: int = 20,
        max_iterations: int = 10,
        max_tool_result_chars: int = 16_000,
        annotate_line_ages: bool = True,
        algo_name: str = "supermemory_memory",
    ) -> None:
        self.store = store
        self.provider = provider
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_iterations = max_iterations
        self.max_tool_result_chars = max_tool_result_chars
        self.annotate_line_ages = annotate_line_ages
        self._algo_name = algo_name
        self._runner = AgentRunner(provider)
        self._tools = self._build_tools()

    # ------------------------------------------------------------------
    # Tool registry
    # ------------------------------------------------------------------

    def _build_tools(self) -> ToolRegistry:
        """Build a minimal tool registry for the Dream agent."""
        from nanobot.agent.skills import BUILTIN_SKILLS_DIR
        from nanobot.agent.tools.filesystem import (
            EditFileTool,
            ReadFileTool,
            SkillPrefixWriteFileTool,
        )

        tools = ToolRegistry()
        workspace = self.store.workspace
        extra_read = [BUILTIN_SKILLS_DIR] if BUILTIN_SKILLS_DIR.exists() else None
        tools.register(ReadFileTool(
            workspace=workspace,
            allowed_dir=workspace,
            extra_allowed_dirs=extra_read,
        ))
        tools.register(EditFileTool(workspace=workspace, allowed_dir=workspace))
        skills_dir = workspace / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        tools.register(SkillPrefixWriteFileTool(
            skill_prefix=f"dreamed--{self._algo_name}",
            workspace=workspace,
            allowed_dir=skills_dir,
        ))
        return tools

    # ------------------------------------------------------------------
    # Skill listing
    # ------------------------------------------------------------------

    def _list_existing_skills(self) -> list[str]:
        """List existing skills for dedup context."""
        import re as _re

        from nanobot.agent.skills import BUILTIN_SKILLS_DIR

        _DESC_RE = _re.compile(r"^description:\s*(.+)$", _re.MULTILINE | _re.IGNORECASE)
        entries: dict[str, str] = {}
        for base in (self.store.workspace / "skills", BUILTIN_SKILLS_DIR):
            if not base.exists():
                continue
            for d in base.iterdir():
                if not d.is_dir():
                    continue
                skill_md = d / "SKILL.md"
                if not skill_md.exists():
                    continue
                if d.name in entries and base == BUILTIN_SKILLS_DIR:
                    continue
                content = skill_md.read_text(encoding="utf-8")[:500]
                m = _DESC_RE.search(content)
                desc = m.group(1).strip() if m else "(no description)"
                entries[d.name] = desc
        return [f"{name} — {desc}" for name, desc in sorted(entries.items())]

    # ------------------------------------------------------------------
    # Age annotation
    # ------------------------------------------------------------------

    def _annotate_with_ages(self, content: str) -> str:
        """Append per-line age suffixes to MEMORY.md content."""
        file_path = f"memory/{self._algo_name}/MEMORY.md"
        try:
            ages = self.store.git.line_ages(file_path)
        except Exception:
            logger.debug("line_ages failed for {}", file_path)
            return content
        if not ages:
            return content

        had_trailing = content.endswith("\n")
        lines = content.splitlines()
        if len(lines) != len(ages):
            logger.debug(
                "line_ages length mismatch for {} (lines={}, ages={}); skipping annotation",
                file_path, len(lines), len(ages),
            )
            return content

        annotated: list[str] = []
        for line, age in zip(lines, ages):
            if not line.strip():
                annotated.append(line)
                continue
            if age.age_days > _STALE_THRESHOLD_DAYS:
                annotated.append(f"{line}  \u2190 {age.age_days}d")
            else:
                annotated.append(line)
        result = "\n".join(annotated)
        if had_trailing:
            result += "\n"
        return result

    # ------------------------------------------------------------------
    # Memory graph context builder
    # ------------------------------------------------------------------

    def _build_graph_context(self) -> str:
        """Build a summary of the memory graph for Phase 1 analysis."""
        nodes = self.store.get_latest_nodes()
        if not nodes:
            return "(empty memory graph)"

        lines = ["## Memory Graph Summary"]
        lines.append(f"Total active memories: {len(nodes)}")

        # Group by relationship type
        edges = list(self.store._edges.values())
        updates_count = len([e for e in edges if e.edge_type.value == "updates"])
        extends_count = len([e for e in edges if e.edge_type.value == "extends"])
        derives_count = len([e for e in edges if e.edge_type.value == "derives"])

        lines.append(f"Relationships: {updates_count} updates, "
                     f"{extends_count} extends, {derives_count} derives")

        # Latest memories preview
        lines.append("\n### Latest Memories")
        for node in nodes[:20]:
            event_info = f" [event: {node.event_date}]" if node.event_date else ""
            version_info = f" (v{node.version})" if node.version > 1 else ""
            lines.append(f"- {node.memory}{event_info}{version_info}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self) -> bool:
        """Process unprocessed history entries. Returns True if work was done."""
        from nanobot.agent.skills import BUILTIN_SKILLS_DIR

        last_cursor = self.store.get_last_dream_cursor()
        entries = self.store.read_unprocessed_history(since_cursor=last_cursor)
        if not entries:
            return False

        batch = entries[: self.max_batch_size]
        logger.info(
            "Supermemory Dream: processing {} entries (cursor {}→{}), batch={}",
            len(entries), last_cursor, batch[-1]["cursor"], len(batch),
        )

        # Build history text for LLM
        history_text = "\n".join(
            f"[{e['timestamp']}] {e['content']}" for e in batch
        )

        # Current file contents + per-line age annotations
        current_date = datetime.now().strftime("%Y-%m-%d")
        raw_memory = self.store.read_memory() or "(empty)"
        current_memory = (
            self._annotate_with_ages(raw_memory)
            if self.annotate_line_ages
            else raw_memory
        )
        current_soul = self.store.read_soul() or "(empty)"
        current_user = self.store.read_user() or "(empty)"

        # Memory graph context
        graph_context = self._build_graph_context()

        file_context = (
            f"## Current Date\n{current_date}\n\n"
            f"## Current MEMORY.md ({len(current_memory)} charts)\n{current_memory}\n\n"
            f"## Current SOUL.md ({len(current_soul)} charts)\n{current_soul}\n\n"
            f"## Current USER.md ({len(current_user)} charts)\n{current_user}\n\n"
            f"{graph_context}"
        )

        # Phase 1: Analyze with graph context and temporal grounding hints
        phase1_prompt = (
            f"## Conversation History\n{history_text}\n\n{file_context}"
        )

        try:
            phase1_response = await self.provider.chat_with_retry(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": render_template(
                            "agent/dream_phase1.md",
                            strip=True,
                            stale_threshold_days=_STALE_THRESHOLD_DAYS,
                        ),
                    },
                    {"role": "user", "content": phase1_prompt},
                ],
                tools=None,
                tool_choice=None,
            )
            analysis = phase1_response.content or ""
            logger.debug(
                "Supermemory Dream Phase 1 analysis ({} chars): {}",
                len(analysis), analysis[:500],
            )
        except Exception:
            logger.exception("Supermemory Dream Phase 1 failed")
            return False

        # Phase 2: Delegate to AgentRunner with read_file / edit_file
        existing_skills = self._list_existing_skills()
        skills_section = ""
        if existing_skills:
            skills_section = (
                "\n\n## Existing Skills\n"
                + "\n".join(f"- {s}" for s in existing_skills)
            )
        phase2_prompt = (
            f"## Analysis Result\n{analysis}\n\n"
            f"{file_context}"
            f"{skills_section}"
        )

        skill_creator_path = BUILTIN_SKILLS_DIR / "skill-creator" / "SKILL.md"
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": render_template(
                    "agent/dream_phase2.md",
                    strip=True,
                    skill_creator_path=str(skill_creator_path),
                    memory_rel_path=f"memory/{self._algo_name}/MEMORY.md",
                ),
            },
            {"role": "user", "content": phase2_prompt},
        ]

        try:
            result = await self._runner.run(AgentRunSpec(
                initial_messages=messages,
                tools=self._tools,
                model=self.model,
                max_iterations=self.max_iterations,
                max_tool_result_chars=self.max_tool_result_chars,
                fail_on_tool_error=False,
            ))
            logger.debug(
                "Supermemory Dream Phase 2 complete: stop_reason={}, tool_events={}",
                result.stop_reason, len(result.tool_events),
            )
            for ev in (result.tool_events or []):
                logger.info(
                    "Supermemory Dream tool_event: name={}, status={}, detail={}",
                    ev.get("name"), ev.get("status"),
                    ev.get("detail", "")[:200],
                )
        except Exception:
            logger.exception("Supermemory Dream Phase 2 failed")
            result = None

        # Build changelog from tool events
        changelog: list[str] = []
        if result and result.tool_events:
            for event in result.tool_events:
                if event["status"] == "ok":
                    changelog.append(f"{event['name']}: {event['detail']}")

        # Advance cursor
        new_cursor = batch[-1]["cursor"]
        self.store.set_last_dream_cursor(new_cursor)
        self.store.compact_history()

        # Auto-forget expired dynamic memories
        forgotten = self.store.auto_forget_expired()
        if forgotten > 0:
            logger.info("Supermemory Dream: auto-forgot {} expired dynamic memories", forgotten)

        if result and result.stop_reason == "completed":
            logger.info(
                "Supermemory Dream done: {} change(s), cursor advanced to {}",
                len(changelog), new_cursor,
            )
        else:
            reason = result.stop_reason if result else "exception"
            logger.warning(
                "Supermemory Dream incomplete ({}): cursor advanced to {}",
                reason, new_cursor,
            )

        # Git auto-commit
        if changelog and self.store.git.is_initialized():
            ts = batch[-1]["timestamp"]
            summary = f"dream: {ts}, {len(changelog)} change(s)"
            commit_msg = f"{summary}\n\n{analysis.strip()}"
            sha = self.store.git.auto_commit(commit_msg)
            if sha:
                logger.info("Supermemory Dream commit: {}", sha)

        return True
