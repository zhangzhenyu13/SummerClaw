"""ReMe Dream — two-phase memory processor with ReMeLight summarisation and skill generation.

Two-phase architecture:
  Phase 1: Delegate to ReMeLight's ``summary_memory()`` for deep summarisation
           of conversation history.
  Phase 2: Delegate to AgentRunner with read_file / edit_file /
           SkillPrefixWriteFileTool to edit MEMORY.md and create ``dreamed-*``
           skills from repeatable workflows.
"""

from __future__ import annotations

import inspect
from datetime import datetime
from typing import TYPE_CHECKING, Any

from agentscope.message import Msg
from loguru import logger

from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.utils.prompt_templates import render_template

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider

from nanobot.memory.remem_memory.store import ReMeStore

_STALE_THRESHOLD_DAYS = 14


class ReMeDream:
    """Two-phase Dream processor backed by ReMeLight.

    Phase 1 leverages ReMeLight's ``summary_memory()`` for analysis.
    Phase 2 runs an AgentRunner with file-editing and skill-creation tools.
    """

    def __init__(
        self,
        store: ReMeStore,
        reme_light: Any,
        provider: LLMProvider,
        model: str,
        max_batch_size: int = 20,
        max_iterations: int = 10,
        max_tool_result_chars: int = 16_000,
        annotate_line_ages: bool = True,
    ):
        self.store = store
        self.reme_light = reme_light
        self.provider = provider
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_iterations = max_iterations
        self.max_tool_result_chars = max_tool_result_chars
        self.annotate_line_ages = annotate_line_ages

        self._runner = AgentRunner(provider)
        self._tools = self._build_tools()

    # -- tool registry -------------------------------------------------------

    def _build_tools(self) -> ToolRegistry:
        """Build tool registry for Dream Phase 2 agent.

        Includes:
        - ReadFileTool / EditFileTool for MEMORY.md / SOUL.md / USER.md edits
        - SkillPrefixWriteFileTool (dreamed-*) for skill creation under skills/
        """
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
            skill_prefix="dreamed",
            workspace=workspace,
            allowed_dir=skills_dir,
        ))
        return tools

    # -- skill listing --------------------------------------------------------

    def _list_existing_skills(self) -> list[str]:
        """List existing skills as 'name — description' for dedup context."""
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

    # -- age annotation -------------------------------------------------------

    def _annotate_with_ages(self, content: str) -> str:
        """Append per-line age suffixes to MEMORY.md content.

        Each non-blank line whose age exceeds ``_STALE_THRESHOLD_DAYS`` gets a
        suffix like ``← 30d`` indicating days since last modification.
        """
        file_path = "MEMORY.md"
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

    # -- main entry ----------------------------------------------------------

    async def run(self) -> bool:
        """Process unprocessed history entries via ReMeLight + AgentRunner.

        Phase 1: ReMeLight ``summary_memory()`` for deep analysis.
        Phase 2: AgentRunner with tools to edit MEMORY.md and create skills.

        Returns ``True`` if work was done.
        """
        from nanobot.agent.skills import BUILTIN_SKILLS_DIR

        last_cursor = self.store.get_last_dream_cursor()
        entries = self.store.read_unprocessed_history(since_cursor=last_cursor)
        if not entries:
            return False

        batch = entries[: self.max_batch_size]
        logger.info(
            "ReMe Dream: processing {} entries (cursor {}→{}), batch={}",
            len(entries),
            last_cursor,
            batch[-1]["cursor"],
            len(batch),
        )

        # Build history text for context
        history_text = "\n".join(
            f"[{e['timestamp']}] {e['content']}" for e in batch
        )

        # Current file contents
        current_date = datetime.now().strftime("%Y-%m-%d")
        raw_memory = self.store.read_memory() or "(empty)"
        current_memory = (
            self._annotate_with_ages(raw_memory)
            if self.annotate_line_ages
            else raw_memory
        )
        current_soul = self.store.read_soul() or "(empty)"
        current_user = self.store.read_user() or "(empty)"

        file_context = (
            f"## Current Date\n{current_date}\n\n"
            f"## Current MEMORY.md ({len(current_memory)} chars)\n{current_memory}\n\n"
            f"## Current SOUL.md ({len(current_soul)} chars)\n{current_soul}\n\n"
            f"## Current USER.md ({len(current_user)} chars)\n{current_user}"
        )

        # ------------------------------------------------------------------
        # Phase 1: ReMeLight summary_memory for deep analysis
        # ------------------------------------------------------------------
        analysis = ""
        try:
            # Convert history entries to AgentScope Msg for ReMeLight
            msgs = [
                Msg(name="user", role="user", content=e.get("content", ""))
                for e in batch
            ]
            result = self.reme_light.summary_memory(messages=msgs)
            if inspect.isawaitable(result):
                summary = await result
            else:
                summary = result

            if summary and isinstance(summary, str):
                analysis = summary
                logger.debug(
                    "ReMe Dream Phase 1 summary ({} chars): {}",
                    len(analysis), analysis[:500],
                )
            else:
                logger.info("ReMe Dream Phase 1: summary_memory returned no text")
        except Exception:
            logger.exception("ReMe Dream Phase 1 (summary_memory) failed")
            # Continue with empty analysis — Phase 2 can still work from
            # history_text and file_context alone.

        # ------------------------------------------------------------------
        # Phase 2: AgentRunner edits MEMORY.md + creates dreamed-* skills
        # ------------------------------------------------------------------
        existing_skills = self._list_existing_skills()
        skills_section = ""
        if existing_skills:
            skills_section = (
                "\n\n## Existing Skills\n"
                + "\n".join(f"- {s}" for s in existing_skills)
            )

        # Build Phase 2 prompt: analysis + file context + existing skills
        phase2_prompt = (
            f"## Conversation History\n{history_text}\n\n"
            f"## Analysis Result\n{analysis or '(no analysis available)'}\n\n"
            f"{file_context}{skills_section}"
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
            phase2_result = await self._runner.run(AgentRunSpec(
                initial_messages=messages,
                tools=self._tools,
                model=self.model,
                max_iterations=self.max_iterations,
                max_tool_result_chars=self.max_tool_result_chars,
                fail_on_tool_error=False,
            ))
            logger.debug(
                "ReMe Dream Phase 2 complete: stop_reason={}, tool_events={}",
                phase2_result.stop_reason, len(phase2_result.tool_events),
            )
        except Exception:
            logger.exception("ReMe Dream Phase 2 failed")
            phase2_result = None

        # Build changelog from tool events
        changelog: list[str] = []
        if phase2_result and phase2_result.tool_events:
            for event in phase2_result.tool_events:
                if event["status"] == "ok":
                    changelog.append(f"{event['name']}: {event['detail']}")

        # Advance cursor — always, to avoid re-processing
        new_cursor = batch[-1]["cursor"]
        self.store.set_last_dream_cursor(new_cursor)
        self.store.compact_history()

        if phase2_result and phase2_result.stop_reason == "completed":
            logger.info(
                "ReMe Dream done: {} change(s), cursor advanced to {}",
                len(changelog), new_cursor,
            )
        else:
            reason = phase2_result.stop_reason if phase2_result else "exception"
            logger.warning(
                "ReMe Dream incomplete ({}): cursor advanced to {}",
                reason, new_cursor,
            )

        # Git auto-commit
        if changelog and self.store.git.is_initialized():
            ts = batch[-1]["timestamp"]
            short_summary = f"dream: {ts}, {len(changelog)} change(s)"
            commit_msg = f"{short_summary}\n\n{analysis.strip()}"
            sha = self.store.git.auto_commit(commit_msg)
            if sha:
                logger.info("ReMe Dream commit: {}", sha)

        return True
