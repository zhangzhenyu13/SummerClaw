"""Layerga Dream — multi-phase memory processor with L0-guided crystallization.

Extends the naive Dream with L0 decision-tree guided skill crystallization
and L1 index maintenance.

Three phases:
  Phase 1: Analyze L4 archives with L0 constitution context.
  Phase 2: Apply L0 decision tree → write to L1/L2/L3 via AgentRunner.
  Phase 3 (optional): L1 cleanup — evaluate ROI, compress, enforce ≤30 line limit.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.memory.naive_memory.dream import Dream, _STALE_THRESHOLD_DAYS
from nanobot.memory.layerga_memory.decision_tree import L0DecisionTree
from nanobot.memory.layerga_memory.store import LayergaStore
from nanobot.utils.prompt_templates import render_template

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider


class LayergaDream(Dream):
    """Multi-phase Dream processor with L0-guided crystallization.

    Phase 1 produces analysis with L0 constitution injected.
    Phase 2 delegates to AgentRunner with layered file editing tools.
    Phase 3 (optional) performs L1 cleanup to enforce the ≤30 line constraint.
    """

    def __init__(
        self,
        store: LayergaStore,
        decision_tree: L0DecisionTree,
        provider: LLMProvider,
        model: str,
        max_batch_size: int = 20,
        max_iterations: int = 10,
        max_tool_result_chars: int = 16_000,
        annotate_line_ages: bool = True,
        enable_l1_cleanup: bool = True,
        enable_auto_crystallize: bool = True,
        algo_name: str = "layerga_memory",
    ):
        super().__init__(
            store=store,
            provider=provider,
            model=model,
            max_batch_size=max_batch_size,
            max_iterations=max_iterations,
            max_tool_result_chars=max_tool_result_chars,
            annotate_line_ages=annotate_line_ages,
            algo_name=algo_name,
        )
        self.layered_store: LayergaStore = store  # typed alias
        self.decision_tree = decision_tree
        self.enable_l1_cleanup = enable_l1_cleanup
        self.enable_auto_crystallize = enable_auto_crystallize

    # ------------------------------------------------------------------
    # Override: main entry with layered context
    # ------------------------------------------------------------------

    async def run(self) -> bool:
        """Process unprocessed history with layered context enrichment.

        Enriches the Phase 1 prompt with:
        - L0 constitution summary (core axioms)
        - L1 insight index
        - L2 fact sections
        - L3 SOP listing

        Returns True if work was done.
        """
        from nanobot.agent.skills import BUILTIN_SKILLS_DIR

        last_cursor = self.layered_store.get_last_dream_cursor()
        entries = self.layered_store.read_unprocessed_history(since_cursor=last_cursor)
        if not entries:
            return False

        batch = entries[: self.max_batch_size]
        logger.info(
            "Layerga Dream: processing {} entries (cursor {}→{}), batch={}",
            len(entries), last_cursor, batch[-1]["cursor"], len(batch),
        )

        # Build history text
        history_text = "\n".join(
            f"[{e['timestamp']}] {e['content']}" for e in batch
        )

        # Current file contents
        current_date = datetime.now().strftime("%Y-%m-%d")
        raw_memory = self.layered_store.read_memory() or "(empty)"
        current_memory = (
            self._annotate_with_ages(raw_memory)
            if self.annotate_line_ages
            else raw_memory
        )
        current_soul = self.layered_store.read_soul() or "(empty)"
        current_user = self.layered_store.read_user() or "(empty)"

        # Layered memory context
        l1_insight = self.layered_store.read_insight() or "(empty)"
        l2_facts = self.layered_store.read_facts() or "(empty)"
        l2_sections = self.layered_store.get_fact_sections()
        l3_sops = self.layered_store.list_sops()
        l3_list = "\n".join(f"- {s.stem}" for s in l3_sops) if l3_sops else "(none)"

        # Constitution summary for Phase 1
        constitution_summary = self.layered_store.get_constitution_summary()

        layered_context = (
            f"## Current Date\n{current_date}\n\n"
            f"## L0 Constitution (Core Axioms)\n{constitution_summary[:1500]}\n\n"
            f"## L1 Insight Index\n```\n{l1_insight}\n```\n\n"
            f"## L2 Fact Sections\n{', '.join(l2_sections) if l2_sections else '(none)'}\n\n"
            f"## L3 Task SOPs\n{l3_list}\n\n"
            f"## Current MEMORY.md ({len(current_memory)} chars)\n{current_memory}\n\n"
            f"## Current SOUL.md ({len(current_soul)} chars)\n{current_soul}\n\n"
            f"## Current USER.md ({len(current_user)} chars)\n{current_user}\n\n"
            f"## L2 Facts ({len(l2_facts)} chars)\n{l2_facts}"
        )

        # ------------------------------------------------------------------
        # Phase 1: Analyze with layered context
        # ------------------------------------------------------------------
        phase1_prompt = (
            f"## Conversation History\n{history_text}\n\n{layered_context}"
        )

        analysis = ""
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
                        )
                        + self._layered_phase1_suffix(),
                    },
                    {"role": "user", "content": phase1_prompt},
                ],
                tools=None,
                tool_choice=None,
            )
            analysis = phase1_response.content or ""
            logger.debug(
                "Layerga Dream Phase 1 ({} chars): {}",
                len(analysis), analysis[:500],
            )
        except Exception:
            logger.exception("Layerga Dream Phase 1 failed")
            return False

        # ------------------------------------------------------------------
        # Phase 2: AgentRunner with layered file editing
        # ------------------------------------------------------------------
        existing_skills = self._list_existing_skills()
        skills_section = ""
        if existing_skills:
            skills_section = (
                "\n\n## Existing Skills\n"
                + "\n".join(f"- {s}" for s in existing_skills)
            )

        phase2_prompt = (
            f"## Analysis Result\n{analysis}\n\n"
            f"{layered_context}{skills_section}"
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
                )
                + self._layered_phase2_suffix(),
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
                "Layerga Dream Phase 2 complete: stop_reason={}, tool_events={}",
                result.stop_reason, len(result.tool_events),
            )
        except Exception:
            logger.exception("Layerga Dream Phase 2 failed")
            result = None

        # Build changelog
        changelog: list[str] = []
        if result and result.tool_events:
            for event in result.tool_events:
                if event["status"] == "ok":
                    changelog.append(f"{event['name']}: {event['detail']}")

        # ------------------------------------------------------------------
        # Phase 3: L1 cleanup (optional)
        # ------------------------------------------------------------------
        if self.enable_l1_cleanup and changelog:
            try:
                cleanup_stats = await self._cleanup_l1()
                if cleanup_stats:
                    logger.info(
                        "Layerga L1 cleanup: removed={}, compressed={}, remaining={}",
                        cleanup_stats.get("removed", 0),
                        cleanup_stats.get("compressed", 0),
                        cleanup_stats.get("remaining_lines", 0),
                    )
            except Exception:
                logger.exception("Layerga L1 cleanup failed")

        # Sync L1 index
        self.layered_store.sync_l1_index()

        # Advance cursor
        new_cursor = batch[-1]["cursor"]
        self.layered_store.set_last_dream_cursor(new_cursor)
        self.layered_store.compact_history()

        if result and result.stop_reason == "completed":
            logger.info(
                "Layerga Dream done: {} change(s), cursor advanced to {}",
                len(changelog), new_cursor,
            )
        else:
            reason = result.stop_reason if result else "exception"
            logger.warning(
                "Layerga Dream incomplete ({}): cursor advanced to {}",
                reason, new_cursor,
            )

        # Git auto-commit
        if changelog and self.layered_store.git.is_initialized():
            ts = batch[-1]["timestamp"]
            summary = f"dream: {ts}, {len(changelog)} change(s)"
            commit_msg = f"{summary}\n\n{analysis.strip()}"
            sha = self.layered_store.git.auto_commit(commit_msg)
            if sha:
                logger.info("Layerga Dream commit: {}", sha)

        return True

    # ------------------------------------------------------------------
    # Phase prompt suffixes (layered context)
    # ------------------------------------------------------------------

    def _layered_phase1_suffix(self) -> str:
        """Additional instructions for Phase 1 with L0 context."""
        return (
            "\n\n---\n"
            "## Layered Memory Guidelines\n\n"
            "You have access to the L0-L4 layered memory system:\n"
            "- **L1 Insight**: The minimal navigation index (≤30 lines). "
            "Check it first to discover existing knowledge.\n"
            "- **L2 Facts**: Environment-specific facts the LLM cannot infer.\n"
            "- **L3 SOPs**: Task-specific standard operating procedures.\n\n"
            "When analyzing, identify:\n"
            "1. **Environment facts** → suggest L2 entries "
            "(paths, credentials, configs, API endpoints)\n"
            "2. **Universal rules** → suggest L1 RULES "
            "(cross-task pitfalls, 1 sentence each)\n"
            "3. **Task techniques** → suggest L3 SOPs "
            "(only hidden preconditions + typical pitfalls)\n\n"
            "**CRITICAL**: Only include action-verified information. "
            "If you are unsure whether something was verified, do NOT suggest it.\n"
        )

    def _layered_phase2_suffix(self) -> str:
        """Additional instructions for Phase 2 with L0 constraints."""
        return (
            "\n\n---\n"
            "## Layered Memory Editing Rules\n\n"
            "When editing layered memory files, follow these strict rules:\n\n"
            "### L0 Axioms\n"
            "1. **No Execution, No Memory** — only write action-verified facts\n"
            "2. **Sanctity of Verified Data** — never discard verified configs/paths\n"
            "3. **No Volatile State** — never write timestamps, PIDs, session IDs\n"
            "4. **Minimum Sufficient** — use shortest content that preserves meaning\n\n"
            "### L1 Insight (memory/layer_insight.txt)\n"
            "- HARD LIMIT: ≤30 lines total\n"
            "- Use `edit_file` for minimal changes only\n"
            "- Only write keywords/names, NOT details\n"
            "- RULES entries: exactly 1 sentence each\n\n"
            "### L2 Facts (memory/layer_facts.txt)\n"
            "- Use `edit_file` for minimal patches\n"
            "- Add new facts under `## [SECTION]` headers\n"
            "- Never overwrite the entire file\n\n"
            "### L3 SOPs (memory/sop/)\n"
            "- Create via `write_file` to memory/sop/<name>_sop.md\n"
            "- Only record: hidden preconditions + typical pitfalls\n"
            "- Do NOT record: ordinary steps, inferrable paths\n\n"
            "**Golden Rule**: When in doubt, don't write. "
            "It's better to under-write than to pollute the memory.\n"
        )

    # ------------------------------------------------------------------
    # L1 cleanup (Phase 3)
    # ------------------------------------------------------------------

    async def _cleanup_l1(self) -> dict[str, int] | None:
        """Clean up L1 insight: evaluate ROI, remove low-value entries, compress.

        Uses the decision tree's ROI computation to determine which lines to keep.

        Returns cleanup stats dict or None if no work was done.
        """
        l1 = self.layered_store.read_insight()
        lines = l1.splitlines()
        non_empty = [
            (i, line)
            for i, line in enumerate(lines)
            if line.strip() and not line.strip().startswith("#")
            and line.strip() != "[RULES]"
        ]

        if len(non_empty) <= self.decision_tree.l1_max_lines:
            return None

        # Score each line by ROI
        scored: list[tuple[int, str, float]] = []
        for idx, line in non_empty:
            roi = self.decision_tree.compute_l1_roi(
                line,
                mistake_probability=0.1,
                mistake_cost_tokens=200,
            )
            scored.append((idx, line, roi))

        # Sort ascending by ROI — lowest first to remove
        scored.sort(key=lambda x: x[2])

        # Remove enough lines to get below limit
        target = self.decision_tree.l1_max_lines
        to_remove = len(non_empty) - target
        if to_remove <= 0:
            return None

        remove_indices = {s[0] for s in scored[:to_remove]}

        # Rebuild L1 without removed lines
        new_lines = [
            line for i, line in enumerate(lines)
            if i not in remove_indices
        ]
        new_l1 = "\n".join(new_lines)
        self.layered_store.write_insight(new_l1)

        stats = {
            "removed": len(remove_indices),
            "compressed": 0,
            "remaining_lines": len([l for l in new_lines if l.strip()]),
        }
        return stats
