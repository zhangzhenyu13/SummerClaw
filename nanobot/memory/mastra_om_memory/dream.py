"""MastraOM Dream — cron-scheduled deep memory processing using Observer/Reflector.

Two-phase memory processor adapted from naive Dream:
Phase 1: Analyze history.jsonl + observations via Observer
Phase 2: Edit MEMORY.md, SOUL.md, USER.md via AgentRunner (with Reflector guidance)

The Dream also supports skill creation (dreamed-* skills) when it detects
repeated workflows in the conversation history.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.memory.mastra_om_memory.store import MastraOMStore
from nanobot.utils.prompt_templates import render_template

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider


_STALE_THRESHOLD_DAYS = 14


class MastraOMDream:
    """Two-phase memory processor: analyze + edit via AgentRunner.

    Uses MastraOM's Observer/Reflector pipeline for offline deep processing:

    Phase 1 (Analyze):
        Reads history.jsonl + current files, sends to an analysis LLM
        (using Observer-style prompt) to identify what should change.

    Phase 2 (Edit):
        Delegates to AgentRunner with read_file/edit_file tools to make
        targeted edits to MEMORY.md, SOUL.md, and USER.md.
        Optionally creates dreamed-* skills.
    """

    def __init__(
        self,
        store: MastraOMStore,
        provider: "LLMProvider",
        model: str,
        max_batch_size: int = 20,
        max_iterations: int = 10,
        max_tool_result_chars: int = 16_000,
        annotate_line_ages: bool = True,
    ):
        self.store = store
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

    # -- line age annotation --------------------------------------------------

    def _annotate_with_ages(self, content: str) -> str:
        """Append per-line age suffixes to MEMORY.md content."""
        file_path = "memory/MEMORY.md"
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
                "line_ages length mismatch for {} (lines={}, ages={}); skipping",
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
        """Process unprocessed history entries. Returns True if work was done."""
        from nanobot.agent.skills import BUILTIN_SKILLS_DIR

        last_cursor = self.store.get_last_dream_cursor()
        entries = self.store.read_unprocessed_history(since_cursor=last_cursor)
        if not entries:
            return False

        batch = entries[:self.max_batch_size]
        logger.info(
            "MastraOM Dream: processing {} entries (cursor {}→{}), batch={}",
            len(entries), last_cursor, batch[-1]["cursor"], len(batch),
        )

        # Build history text for LLM
        history_text = "\n".join(
            f"[{e['timestamp']}] {e['content']}" for e in batch
        )

        # Current file contents + line age annotations
        current_date = datetime.now().strftime("%Y-%m-%d")
        raw_memory = self.store.read_memory() or "(empty)"
        current_memory = (
            self._annotate_with_ages(raw_memory)
            if self.annotate_line_ages
            else raw_memory
        )
        current_soul = self.store.read_soul() or "(empty)"
        current_user = self.store.read_user() or "(empty)"
        current_observations = self.store.read_observations() or "(empty)"

        logger.info(
            "MastraOM Dream: loaded files — MEMORY.md={} chars, SOUL.md={} chars, "
            "USER.md={} chars, OBSERVATIONS.md={} chars",
            len(current_memory), len(current_soul),
            len(current_user), len(current_observations),
        )

        file_context = (
            f"## Current Date\n{current_date}\n\n"
            f"## Current OBSERVATIONS.md ({len(current_observations)} chars)\n{current_observations}\n\n"
            f"## Current MEMORY.md ({len(current_memory)} chars)\n{current_memory}\n\n"
            f"## Current SOUL.md ({len(current_soul)} chars)\n{current_soul}\n\n"
            f"## Current USER.md ({len(current_user)} chars)\n{current_user}"
        )

        # Phase 1: Analyze
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
            logger.info(
                "MastraOM Dream Phase 1 complete: {} chars analysis",
                len(analysis),
            )
        except Exception:
            logger.exception("MastraOM Dream Phase 1 failed")
            return False

        # Phase 2: Delegate to AgentRunner
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

        tools = self._tools
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
                tools=tools,
                model=self.model,
                max_iterations=self.max_iterations,
                max_tool_result_chars=self.max_tool_result_chars,
                fail_on_tool_error=False,
            ))
            logger.info(
                "MastraOM Dream Phase 2 done: stop_reason={}, tool_events={}",
                result.stop_reason, len(result.tool_events or []),
            )
            for ev in (result.tool_events or []):
                logger.info(
                    "Dream tool_event: name={}, status={}, detail={}",
                    ev.get("name"), ev.get("status"),
                    (ev.get("detail", "") or "")[:200],
                )
        except Exception:
            logger.exception("MastraOM Dream Phase 2 failed")
            result = None

        # Build changelog
        changelog: list[str] = []
        if result and result.tool_events:
            for event in result.tool_events:
                if event["status"] == "ok":
                    changelog.append(f"{event['name']}: {event['detail']}")

        # Advance cursor
        new_cursor = batch[-1]["cursor"]
        self.store.set_last_dream_cursor(new_cursor)
        self.store.compact_history()

        if result and result.stop_reason == "completed":
            logger.info(
                "MastraOM Dream done: {} change(s), cursor advanced to {}",
                len(changelog), new_cursor,
            )
        else:
            reason = result.stop_reason if result else "exception"
            logger.warning(
                "MastraOM Dream incomplete ({}): cursor advanced to {}",
                reason, new_cursor,
            )

        # Git auto-commit
        if changelog and self.store.git.is_initialized():
            ts = batch[-1]["timestamp"]
            summary = f"dream: {ts}, {len(changelog)} change(s)"
            commit_msg = f"{summary}\n\n{analysis.strip()}"
            sha = self.store.git.auto_commit(commit_msg)
            if sha:
                logger.info("Dream commit: {}", sha)

        return True
