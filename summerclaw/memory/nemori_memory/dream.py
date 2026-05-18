"""Nemori Dream — cron-scheduled deep memory processing with skill generation.

Adapts summerclaw's Dream interface to work with nemori's episode/semantic data.
Uses the nemori unified search to retrieve relevant memories, then delegates
to the shared Dream Phase 1/Phase 2 pattern for file editing and dreamed-* skill creation.
"""

from __future__ import annotations

import json
import re as _re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from summerclaw.agent.runner import AgentRunner, AgentRunSpec
from summerclaw.agent.tools.registry import ToolRegistry
from summerclaw.memory.nemori_memory.search import UnifiedSearch
from summerclaw.utils.prompt_templates import render_template

if TYPE_CHECKING:
    from summerclaw.providers.base import LLMProvider
    from summerclaw.memory.nemori_memory.store import NemoriStore

# Regex to pull description from SKILL.md frontmatter for skill listing
_DESC_RE = _re.compile(r"^description:\s*(.+)$", _re.MULTILINE | _re.IGNORECASE)

# Staleness threshold for per-line age annotation (shared with dream_phase1.md)
_STALE_THRESHOLD_DAYS = 14


class NemoriDream:
    """Dream processor for nemori memory: periodically consolidates
    collected episodes and semantic memories into the agent's MEMORY.md
    and creates ``dreamed-*`` skills from repeatable workflows.

    Two-phase:
      Phase 1: Analyze collected episodes + semantics → produce changeset plan
      Phase 2: Delegate to AgentRunner with read_file / edit_file / write_file
               (scoped to ``workspace/skills/`` via ``SkillPrefixWriteFileTool``)
    """

    _CURSOR_FILE = "dream_cursor.json"

    def __init__(
        self,
        store: "NemoriStore",
        search: UnifiedSearch,
        provider: "LLMProvider",
        model: str,
        workspace: Path,
        max_batch_size: int = 20,
        max_iterations: int = 10,
        max_tool_result_chars: int = 16_000,
        annotate_line_ages: bool = True,
        algo_name: str = "nemori_memory",
    ) -> None:
        self.store = store
        self._search = search
        self.provider = provider
        self.model = model
        self.workspace = workspace
        self.max_batch_size = max_batch_size
        self.max_iterations = max_iterations
        self.max_tool_result_chars = max_tool_result_chars
        self.annotate_line_ages = annotate_line_ages
        self._algo_name = algo_name

        self._runner = AgentRunner(provider)
        self._tools = self._build_tools()

    # -- tool registry -------------------------------------------------------

    def _build_tools(self) -> ToolRegistry:
        """Build tool registry for Dream Phase 2 agent.

        Includes:
        - ReadFileTool / EditFileTool for MEMORY.md / SOUL.md / USER.md edits
        - SkillPrefixWriteFileTool (dreamed-*) for skill creation under skills/
        """
        from summerclaw.agent.skills import BUILTIN_SKILLS_DIR
        from summerclaw.agent.tools.filesystem import (
            EditFileTool,
            ReadFileTool,
            SkillPrefixWriteFileTool,
        )

        tools = ToolRegistry()
        workspace = self.workspace
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

    # -- skill listing --------------------------------------------------------

    def _list_existing_skills(self) -> list[str]:
        """List existing skills as 'name — description' for dedup context."""
        from summerclaw.agent.skills import BUILTIN_SKILLS_DIR

        entries: dict[str, str] = {}
        for base in (self.workspace / "skills", BUILTIN_SKILLS_DIR):
            if not base.exists():
                continue
            for d in base.iterdir():
                if not d.is_dir():
                    continue
                skill_md = d / "SKILL.md"
                if not skill_md.exists():
                    continue
                # Prefer workspace skills over builtin (same name)
                if d.name in entries and base == BUILTIN_SKILLS_DIR:
                    continue
                content = skill_md.read_text(encoding="utf-8")[:500]
                m = _DESC_RE.search(content)
                desc = m.group(1).strip() if m else "(no description)"
                entries[d.name] = desc
        return [f"{name} — {desc}" for name, desc in sorted(entries.items())]

    # -- cursor tracking ------------------------------------------------------

    def _get_last_dream_cursor(self) -> str | None:
        """Read the last dream cursor (ISO timestamp) from disk."""
        cursor_path = self.workspace / self._CURSOR_FILE
        try:
            data = json.loads(cursor_path.read_text(encoding="utf-8"))
            return data.get("last_cursor")
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _set_last_dream_cursor(self, cursor: str) -> None:
        """Write the last dream cursor (ISO timestamp) to disk."""
        cursor_path = self.workspace / self._CURSOR_FILE
        cursor_path.parent.mkdir(parents=True, exist_ok=True)
        cursor_path.write_text(
            json.dumps({"last_cursor": cursor}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # -- main entry point ----------------------------------------------------

    async def run(self) -> bool:
        """Run dream processing — consolidates nemori episodes/semantics
        into MEMORY.md / SOUL.md / USER.md and creates ``dreamed-*`` skills.

        Called by the cron service as ``await agent.dream.run()`` with no args.
        Uses ``user_id="default"`` / ``agent_id="default"`` internally.

        Returns ``True`` if work was done.
        """
        from summerclaw.agent.skills import BUILTIN_SKILLS_DIR

        user_id = "default"
        agent_id = "default"

        # Collect recent episodes and semantic memories
        all_episodes = self.store.list_episodes(user_id, agent_id, limit=100)
        semantics = self.store.list_semantics(user_id, agent_id)

        # Filter episodes newer than last cursor for incremental processing
        last_cursor = self._get_last_dream_cursor()
        if last_cursor:
            episodes = [
                ep for ep in all_episodes
                if ep.created_at and ep.created_at.isoformat() > last_cursor
            ]
        else:
            episodes = all_episodes[: self.max_batch_size]

        if not episodes and not semantics:
            return False

        # Build context for Phase 1
        current_date = datetime.now().strftime("%Y-%m-%d")
        memory_file = self.workspace / "memory" / self._algo_name / "MEMORY.md"
        soul_file = self.workspace / "memory" / self._algo_name / "SOUL.md"
        user_file = self.workspace / "memory" / self._algo_name / "USER.md"

        raw_memory = self._read_file(memory_file)
        current_memory = (
            self._annotate_with_ages(raw_memory)
            if self.annotate_line_ages
            else raw_memory
        )
        current_soul = self._read_file(soul_file)
        current_user = self._read_file(user_file)

        # Format episodes and semantics
        episode_text = self._format_episodes(episodes)
        semantic_text = self._format_semantics(semantics)

        file_context = (
            f"## Current Date\n{current_date}\n\n"
            f"## Recent Episodes\n{episode_text}\n\n"
            f"## Semantic Knowledge\n{semantic_text}\n\n"
            f"## Current MEMORY.md ({len(current_memory)} chars)\n{current_memory}\n\n"
            f"## Current SOUL.md ({len(current_soul)} chars)\n{current_soul}\n\n"
            f"## Current USER.md ({len(current_user)} chars)\n{current_user}"
        )

        # -- Phase 1: Analyze ------------------------------------------------
        try:
            phase1_resp = await self.provider.chat_with_retry(
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
                    {"role": "user", "content": file_context},
                ],
                tools=None,
                tool_choice=None,
            )
            analysis = phase1_resp.content or ""
            logger.debug(
                "Nemori Dream Phase 1 analysis ({} chars): {}",
                len(analysis), analysis[:500],
            )
        except Exception:
            logger.exception("Nemori Dream Phase 1 failed")
            return False

        # -- Phase 2: Delegate to AgentRunner ---------------------------------
        existing_skills = self._list_existing_skills()
        skills_section = ""
        if existing_skills:
            skills_section = (
                "\n\n## Existing Skills\n"
                + "\n".join(f"- {s}" for s in existing_skills)
            )
        phase2_prompt = (
            f"## Analysis Result\n{analysis}\n\n{file_context}{skills_section}"
        )

        skill_creator_path = BUILTIN_SKILLS_DIR / "skill-creator" / "SKILL.md"
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": render_template(
                    "agent/dream_phase2.md",
                    strip=True,
                    skill_creator_path=str(skill_creator_path),
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
                "Nemori Dream Phase 2 complete: stop_reason={}, tool_events={}",
                result.stop_reason, len(result.tool_events or []),
            )
            for ev in (result.tool_events or []):
                logger.info(
                    "Nemori Dream tool_event: name={}, status={}, detail={}",
                    ev.get("name"), ev.get("status"),
                    ev.get("detail", "")[:200],
                )
        except Exception:
            logger.exception("Nemori Dream Phase 2 failed")
            result = None

        # Build changelog from tool events
        changelog: list[str] = []
        if result and result.tool_events:
            for event in result.tool_events:
                if event["status"] == "ok":
                    changelog.append(f"{event['name']}: {event['detail']}")

        # Advance cursor — always, to avoid re-processing Phase 1
        if episodes:
            new_cursor = episodes[0].created_at.isoformat()
        else:
            new_cursor = datetime.now().isoformat()
        self._set_last_dream_cursor(new_cursor)

        if result and result.stop_reason == "completed":
            logger.info(
                "Nemori Dream done: {} change(s), cursor advanced to {}",
                len(changelog), new_cursor,
            )
        else:
            reason = result.stop_reason if result else "exception"
            logger.warning(
                "Nemori Dream incomplete ({}): cursor advanced to {}",
                reason, new_cursor,
            )

        return True

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _read_file(path: Path) -> str:
        """Read a file, returning '(empty)' if it doesn't exist."""
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return "(empty)"

    def _format_episodes(self, episodes: list[Any]) -> str:
        """Format episode list for LLM context."""
        lines: list[str] = []
        for ep in episodes:
            ts = (
                ep.created_at.strftime("%Y-%m-%d %H:%M")
                if ep.created_at else "?"
            )
            lines.append(f"### {ep.title}  [{ts}]")
            lines.append(ep.content)
            lines.append("")
        return "\n".join(lines) if lines else "(no episodes)"

    def _format_semantics(self, semantics: list[Any]) -> str:
        """Format semantic memory list for LLM context."""
        lines: list[str] = []
        for sm in semantics:
            lines.append(f"- [{sm.memory_type}] {sm.content}")
        return "\n".join(lines) if lines else "(no semantic knowledge)"

    def _annotate_with_ages(self, content: str) -> str:
        """Append per-line age suffixes to MEMORY.md content.

        Only non-blank lines whose git age exceeds ``_STALE_THRESHOLD_DAYS``
        receive a suffix like ``← 30d``.  Falls back to raw content if git
        metadata is unavailable.
        """
        try:
            from summerclaw.utils.gitstore import GitStore

            git = GitStore(self.workspace)
            if not git.is_initialized():
                return content
            ages = git.line_ages(f"memory/{self._algo_name}/MEMORY.md")
        except Exception:
            logger.debug("line_ages unavailable for nemori dream; skipping annotation")
            return content

        if not ages:
            return content

        had_trailing = content.endswith("\n")
        lines = content.splitlines()
        if len(lines) != len(ages):
            logger.debug(
                "line_ages length mismatch (lines={}, ages={}); "
                "skipping annotation", len(lines), len(ages),
            )
            return content

        annotated: list[str] = []
        for line, age in zip(lines, ages):
            if not line.strip():
                annotated.append(line)
                continue
            if age.age_days > _STALE_THRESHOLD_DAYS:
                annotated.append(f"{line}  ← {age.age_days}d")
            else:
                annotated.append(line)
        result = "\n".join(annotated)
        if had_trailing:
            result += "\n"
        return result
