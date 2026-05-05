"""Skill auto-generation: background review that creates reusable SKILL.md files.

Distinct from Dream (memory consolidation):
- Dream   : cron-scheduled, reads history.jsonl, updates MEMORY/SOUL/USER.md
- SkillAutogen: threshold-triggered (cumulative tool calls), inspects the current
  turn's messages, creates skills/<name>/SKILL.md for non-trivial patterns

Trigger: after every ``nudge_interval`` cumulative tool calls across turns, the
loop schedules ``SkillAutogen.run(all_msgs)`` as a background task so it never
blocks the user.
"""

from __future__ import annotations

import re as _re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.utils.prompt_templates import render_template

if TYPE_CHECKING:
    from nanobot.memory.naive_memory.store import MemoryStore
    from nanobot.providers.base import LLMProvider


# Regex to pull description from SKILL.md frontmatter for listing
_DESC_RE = _re.compile(r"^description:\s*(.+)$", _re.MULTILINE | _re.IGNORECASE)


class SkillAutogen:
    """Background reviewer that generates SKILL.md files from recent conversations.

    Usage
    -----
    After each agent turn, call ``increment_tool_count(n)`` with the number of
    tool calls that turn made.  When ``should_trigger(interval)`` returns True,
    call ``reset_count()`` and schedule ``run(all_msgs)`` as a background task.

    The ``run`` method spins up a lightweight AgentRunner with a write_file tool
    (scoped to ``workspace/skills/``) and a read_file tool so the reviewer can
    reference the skill-creator guide.  It never touches memory files.
    """

    def __init__(
        self,
        store: MemoryStore,
        provider: LLMProvider,
        model: str,
        workspace: Path,
        max_iterations: int = 8,
        max_tool_result_chars: int = 16_000,
    ) -> None:
        self.store = store
        self.provider = provider
        self.model = model
        self.workspace = workspace
        self.max_iterations = max_iterations
        self.max_tool_result_chars = max_tool_result_chars

        # Cumulative tool-call counter (cross-turn)
        self._tool_call_count: int = 0

        self._runner = AgentRunner(provider)
        self._tools = self._build_tools()

    # ------------------------------------------------------------------
    # Tool registry
    # ------------------------------------------------------------------

    def _build_tools(self) -> ToolRegistry:
        """Minimal tool registry: read_file (workspace + builtin skills) + write_file (skills/).

        WriteFileTool is scoped to ``workspace/skills/``.  Since scripts/ is a
        subdirectory of that (``skills/<name>/scripts/<file>.py``), the reviewer
        agent can write both SKILL.md and scripts/ files with the same tool —
        no extra registration needed.
        """
        from nanobot.agent.skills import BUILTIN_SKILLS_DIR
        from nanobot.agent.tools.filesystem import ReadFileTool, SkillPrefixWriteFileTool

        tools = ToolRegistry()
        extra_read = [BUILTIN_SKILLS_DIR] if BUILTIN_SKILLS_DIR.exists() else None
        tools.register(ReadFileTool(
            workspace=self.workspace,
            allowed_dir=self.workspace,
            extra_allowed_dirs=extra_read,
        ))
        # write_file scoped to workspace/skills/ — covers both SKILL.md and scripts/ subdirs
        # SkillPrefixWriteFileTool hard-codes the "hermes-" prefix so that even if
        # the LLM ignores the prompt naming rule, the directory will always be correct.
        skills_dir = self.workspace / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        tools.register(SkillPrefixWriteFileTool(
            skill_prefix="hermes",
            workspace=self.workspace,
            allowed_dir=skills_dir,
        ))
        return tools

    # ------------------------------------------------------------------
    # Counter helpers
    # ------------------------------------------------------------------

    def increment_tool_count(self, n: int) -> None:
        """Add n tool calls to the cumulative counter."""
        self._tool_call_count += max(0, n)

    def should_trigger(self, nudge_interval: int) -> bool:
        """Return True if the counter has reached or exceeded the threshold."""
        return self._tool_call_count >= nudge_interval

    def reset_count(self) -> None:
        """Reset the counter after triggering a review pass."""
        self._tool_call_count = 0

    # ------------------------------------------------------------------
    # Skill listing (dedup)
    # ------------------------------------------------------------------

    def _list_existing_skills(self) -> str:
        """Return a formatted list of existing skills for dedup context.

        Each entry includes name, description, and whether a scripts/ subdir
        exists (helps the reviewer avoid duplicating script logic).
        """
        from nanobot.agent.skills import BUILTIN_SKILLS_DIR

        entries: dict[str, str] = {}  # name -> formatted line
        for base in (self.workspace / "skills", BUILTIN_SKILLS_DIR):
            if not base.exists():
                continue
            for d in base.iterdir():
                if not d.is_dir():
                    continue
                skill_md = d / "SKILL.md"
                if not skill_md.exists():
                    continue
                # workspace skills take priority over builtins with the same name
                if d.name in entries and base == BUILTIN_SKILLS_DIR:
                    continue
                content = skill_md.read_text(encoding="utf-8")[:500]
                m = _DESC_RE.search(content)
                desc = m.group(1).strip() if m else "(no description)"
                # Note if scripts/ subdir exists
                scripts_dir = d / "scripts"
                has_scripts = scripts_dir.is_dir() and any(scripts_dir.iterdir())
                suffix = " [has scripts/]" if has_scripts else ""
                entries[d.name] = f"- {d.name}: {desc}{suffix}"

        if not entries:
            return "(none yet)"
        return "\n".join(v for _, v in sorted(entries.items()))

    # ------------------------------------------------------------------
    # Conversation summarizer
    # ------------------------------------------------------------------

    @staticmethod
    def _summarize_messages(messages: list[dict[str, Any]]) -> str:
        """Build a compact, token-efficient summary of the recent conversation."""
        lines: list[str] = []
        tool_calls_seen: int = 0
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content")

            if role == "user":
                text = content if isinstance(content, str) else ""
                if text.strip():
                    lines.append(f"[User] {text[:400]}")

            elif role == "assistant":
                # Count / list tool calls
                tool_calls = msg.get("tool_calls") or []
                for tc in tool_calls:
                    name = tc.get("function", {}).get("name") or tc.get("name", "?")
                    # Truncate large arguments
                    args = str(tc.get("function", {}).get("arguments") or tc.get("arguments", ""))
                    args_preview = args[:120] + ("..." if len(args) > 120 else "")
                    lines.append(f"[Tool call] {name}({args_preview})")
                    tool_calls_seen += 1
                # Text response
                text = content if isinstance(content, str) else ""
                if text.strip():
                    lines.append(f"[Assistant] {text[:300]}")

            elif role == "tool":
                # Tool result — include first 200 chars
                text = content if isinstance(content, str) else ""
                if text.strip():
                    lines.append(f"[Tool result] {text[:200]}")

        summary = "\n".join(lines)
        header = f"[{tool_calls_seen} tool call(s) in this turn]\n\n"
        return header + summary

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self, recent_messages: list[dict[str, Any]]) -> bool:
        """Review recent_messages for skill-worthy patterns.

        Returns True if a skill was created.
        """
        from nanobot.agent.skills import BUILTIN_SKILLS_DIR

        tool_count = sum(1 for m in recent_messages if m.get("role") == "assistant"
                         and m.get("tool_calls"))
        if tool_count == 0:
            logger.debug("SkillAutogen: skipping review — no tool calls in recent messages")
            return False

        logger.info("SkillAutogen: reviewing {} messages ({} assistant turns with tools)",
                    len(recent_messages), tool_count)

        existing_skills = self._list_existing_skills()
        conversation_summary = self._summarize_messages(recent_messages)
        skill_creator_path = BUILTIN_SKILLS_DIR / "skill-creator" / "SKILL.md"

        system_prompt = render_template(
            "agent/skill_autogen_review.md",
            strip=True,
            existing_skills=existing_skills,
            skill_creator_path=str(skill_creator_path),
        )
        user_prompt = (
            "## Recent Conversation\n\n"
            + conversation_summary
            + "\n\n"
            "Review the conversation above and decide whether it contains a reusable workflow. "
            "Follow the instructions in the system prompt exactly."
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
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
                "SkillAutogen review complete: stop_reason={}, tool_events={}",
                result.stop_reason, len(result.tool_events or []),
            )
        except Exception:
            logger.exception("SkillAutogen review failed")
            return False

        # Check whether a write_file actually happened (skill / scripts created)
        skill_md_created = False
        scripts_created: list[str] = []
        if result and result.tool_events:
            for event in result.tool_events:
                if event.get("name") == "write_file" and event.get("status") == "ok":
                    detail = event.get("detail", "")
                    # Distinguish SKILL.md vs scripts/ writes by path pattern
                    if "SKILL.md" in detail:
                        logger.info("SkillAutogen: SKILL.md created \u2014 {}", detail)
                        skill_md_created = True
                    elif "/scripts/" in detail or "\\scripts\\" in detail:
                        logger.info("SkillAutogen: script created \u2014 {}", detail)
                        scripts_created.append(detail)
                    else:
                        # Generic write (could be either)
                        logger.info("SkillAutogen: file written \u2014 {}", detail)
                        skill_md_created = True
        
        if skill_md_created:
            extras = f" + {len(scripts_created)} script(s)" if scripts_created else ""
            logger.info("SkillAutogen: skill package created{}", extras)
        elif not scripts_created:
            logger.debug("SkillAutogen: no skill created this pass")
        created = skill_md_created or bool(scripts_created)
        return created
