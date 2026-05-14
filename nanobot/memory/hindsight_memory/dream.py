"""Hindsight Dream — offline cron-scheduled deep memory processing.

Two-phase processor: Phase 1 uses Hindsight recall for contextual retrieval,
Phase 2 uses Hindsight reflect for agentic reasoning, and optionally edits
MEMORY.md via AgentRunner.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.memory.naive_memory.store import MemoryStore
from nanobot.utils.prompt_templates import render_template

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider

_STALE_THRESHOLD_DAYS = 14


class HindsightDream:
    """Two-phase memory processor with Hindsight server integration.

    Phase 1: Queries the Hindsight server (recall + reflect) and the local
    history to build an analysis report.  Falls back to LLM-only analysis if
    the Hindsight server is unavailable.

    Phase 2: Delegates to AgentRunner with read_file / edit_file tools to
    make targeted, incremental edits to MEMORY.md and optionally generate
    dreamed-* skills.
    """

    def __init__(
        self,
        store: MemoryStore,
        provider: "LLMProvider",
        model: str,
        max_batch_size: int = 20,
        max_iterations: int = 10,
        max_tool_result_chars: int = 16_000,
        annotate_line_ages: bool = True,
        *,
        hindsight_store: Any = None,
        algo_name: str = "hindsight_memory",
    ):
        self.store = store
        self.provider = provider
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_iterations = max_iterations
        self.max_tool_result_chars = max_tool_result_chars
        self.annotate_line_ages = annotate_line_ages
        self._hindsight_store = hindsight_store
        self._algo_name = algo_name
        self._runner = AgentRunner(provider)
        self._tools = self._build_tools()

    @property
    def has_hindsight(self) -> bool:
        return self._hindsight_store is not None and self._hindsight_store.hindsight_enabled

    # -- tool registry -------------------------------------------------------

    def _build_tools(self) -> ToolRegistry:
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

    # -- skill listing -------------------------------------------------------

    def _list_existing_skills(self) -> list[str]:
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

    # -- line age annotation -------------------------------------------------

    def _annotate_with_ages(self, content: str) -> str:
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

    # -- main entry ----------------------------------------------------------

    async def run(self) -> bool:
        """Process unprocessed history entries. Returns True if work was done."""
        from nanobot.agent.skills import BUILTIN_SKILLS_DIR

        last_cursor = self.store.get_last_dream_cursor()
        entries = self.store.read_unprocessed_history(since_cursor=last_cursor)
        if not entries:
            return False

        batch = entries[: self.max_batch_size]
        logger.info(
            "HindsightDream: processing {} entries (cursor {}→{}), batch={}",
            len(entries), last_cursor, batch[-1]["cursor"], len(batch),
        )

        history_text = "\n".join(
            f"[{e['timestamp']}] {e['content']}" for e in batch
        )

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

        # Phase 1: Analyze — try Hindsight first, fall back to plain LLM
        phase1_prompt = (
            f"## Conversation History\n{history_text}\n\n{file_context}"
        )

        # Try Hindsight reflect for richer analysis
        hindsight_analysis = ""
        if self.has_hindsight:
            try:
                reflect_response = await self._hindsight_store.areflect(
                    query=(
                        "Analyze the following conversation history and current memory. "
                        "Identify: 1) new facts to add, 2) stale/outdated information, "
                        "3) potential contradictions, 4) skill-generation opportunities."
                    ),
                    context=phase1_prompt[:8000],
                    budget="mid",
                )
                if reflect_response and hasattr(reflect_response, "text"):
                    hindsight_analysis = reflect_response.text or ""
                    logger.info(
                        "HindsightDream Phase 1 (hindsight reflect): {} chars",
                        len(hindsight_analysis),
                    )
            except Exception:
                logger.exception("Hindsight reflect in Dream Phase 1 failed, falling back")

        # Fall back to plain LLM if Hindsight didn't give an answer
        analysis = hindsight_analysis
        if not analysis:
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
                    "HindsightDream Phase 1 (LLM fallback): {} chars",
                    len(analysis),
                )
            except Exception:
                logger.exception("HindsightDream Phase 1 failed")
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
                    memory_rel_path=f"memory/{self._algo_name}/MEMORY.md",
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
            logger.debug(
                "HindsightDream Phase 2 complete: stop_reason={}, tool_events={}",
                result.stop_reason, len(result.tool_events),
            )
            for ev in (result.tool_events or []):
                logger.info(
                    "HindsightDream tool_event: name={}, status={}, detail={}",
                    ev.get("name"), ev.get("status"),
                    ev.get("detail", "")[:200],
                )
        except Exception:
            logger.exception("HindsightDream Phase 2 failed")
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
                "HindsightDream done: {} change(s), cursor advanced to {}",
                len(changelog), new_cursor,
            )
        else:
            reason = result.stop_reason if result else "exception"
            logger.warning(
                "HindsightDream incomplete ({}): cursor advanced to {}",
                reason, new_cursor,
            )

        # Git auto-commit
        if changelog and self.store.git.is_initialized():
            ts = batch[-1]["timestamp"]
            summary = f"dream: {ts}, {len(changelog)} change(s)"
            commit_msg = f"{summary}\n\n{analysis.strip()}"
            sha = self.store.git.auto_commit(commit_msg)
            if sha:
                logger.info("HindsightDream commit: {}", sha)

        return True
