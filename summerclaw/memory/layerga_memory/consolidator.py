"""Layerga consolidator — online token-budget consolidation with L0 classification.

Extends the naive Consolidator with L0 decision-tree classification.
When messages are archived, facts extracted from action-verified tool results
are classified by the L0 decision tree and written to the appropriate layer.
"""

from __future__ import annotations

import asyncio
import weakref
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from summerclaw.memory.naive_memory.consolidator import Consolidator
from summerclaw.memory.naive_memory.store import MemoryStore
from summerclaw.memory.layerga_memory.decision_tree import (
    L0DecisionTree,
    MemoryLayer,
    VerifiedFact,
)
from summerclaw.memory.layerga_memory.store import LayergaStore
from summerclaw.utils.helpers import estimate_message_tokens, estimate_prompt_tokens_chain
from summerclaw.utils.prompt_templates import render_template

if TYPE_CHECKING:
    from summerclaw.providers.base import LLMProvider
    from summerclaw.session.manager import Session, SessionManager


class LayergaConsolidator(Consolidator):
    """Enhanced consolidator with L0 decision-tree classification.

    On top of the standard token-budget consolidation, this consolidator:

    1. Extracts action-verified facts from tool-call results in archived messages.
    2. Classifies each fact via the L0 decision tree.
    3. Writes classified information to L1/L2/L3 layers using minimal patches.
    4. Syncs the L1 index after changes.
    """

    def __init__(
        self,
        store: LayergaStore,
        decision_tree: L0DecisionTree,
        provider: LLMProvider,
        model: str,
        sessions: SessionManager,
        context_window_tokens: int,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        max_completion_tokens: int = 4096,
        enable_classification: bool = True,
    ):
        super().__init__(
            store=store,
            provider=provider,
            model=model,
            sessions=sessions,
            context_window_tokens=context_window_tokens,
            build_messages=build_messages,
            get_tool_definitions=get_tool_definitions,
            max_completion_tokens=max_completion_tokens,
        )
        self.layered_store: LayergaStore = store  # typed alias
        self.decision_tree = decision_tree
        self.enable_classification = enable_classification

    # ------------------------------------------------------------------
    # Override: archive with classification
    # ------------------------------------------------------------------

    async def archive(self, messages: list[dict]) -> str | None:
        """Summarize messages and classify action-verified facts into layers.

        Returns the summary text on success, None if nothing to archive.
        """
        summary = await super().archive(messages)
        if not summary or not self.enable_classification:
            return summary

        # Extract action-verified facts from messages
        verified_facts = self._extract_verified_facts(messages)
        if verified_facts:
            logger.debug(
                "Layerga consolidator: extracted {} verified facts from {} messages",
                len(verified_facts), len(messages),
            )
            stats = await self._classify_and_store(verified_facts)
            logger.info(
                "Layerga classification: L1={}, L2={}, L3={}, dropped={}",
                stats.get("L1", 0), stats.get("L2", 0),
                stats.get("L3", 0), stats.get("dropped", 0),
            )

        return summary

    # ------------------------------------------------------------------
    # Fact extraction from messages
    # ------------------------------------------------------------------

    def _extract_verified_facts(self, messages: list[dict]) -> list[VerifiedFact]:
        """Extract action-verified facts from archived messages.

        Looks for tool-call results in the messages that indicate successful
        operations (exit_code=0, status=success).
        """
        facts: list[VerifiedFact] = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "tool" and isinstance(content, str):
                # Parse tool result
                is_success = self._is_success_result(content)
                if is_success:
                    tool_name = msg.get("name", "unknown")
                    facts.append(VerifiedFact(
                        source_tool=tool_name,
                        content=content[:1000],
                        is_verified=True,
                    ))

            elif role == "assistant":
                tool_calls = msg.get("tool_calls") or []
                for tc in tool_calls:
                    tc_name = tc.get("function", {}).get("name") or tc.get("name", "")
                    tc_args = tc.get("function", {}).get("arguments") or tc.get("arguments", "")
                    if isinstance(tc_args, str):
                        try:
                            import json
                            tc_args = json.loads(tc_args)
                        except (json.JSONDecodeError, TypeError):
                            tc_args = {"raw": str(tc_args)[:200]}

                    # Extract key info from specific tools
                    fact_content = self._extract_tool_fact(tc_name, tc_args, content)
                    if fact_content:
                        facts.append(VerifiedFact(
                            source_tool=tc_name,
                            source_args=tc_args if isinstance(tc_args, dict) else {},
                            content=fact_content,
                            is_verified=False,  # not yet verified — wait for tool result
                        ))

        return facts

    def _is_success_result(self, content: str) -> bool:
        """Check if a tool result indicates successful execution."""
        import json as _json
        success_indicators = [
            '"status": "success"',
            '"exit_code": 0',
            '"status":"success"',
            '"exit_code":0',
            "✅",
            "success",
        ]
        content_lower = content.lower()
        for indicator in success_indicators:
            if indicator.lower() in content_lower:
                return True
        return False

    def _extract_tool_fact(
        self, tool_name: str, args: dict[str, Any], response_content: str
    ) -> str | None:
        """Extract fact content from a tool call based on tool type.

        Returns a fact string or None if nothing useful can be extracted.
        """
        # For write/patch tools: the fact is what was written
        if tool_name in ("file_write", "write_file"):
            path = args.get("path", "")
            return f"Wrote file: {path}"
        if tool_name in ("file_patch", "edit_file"):
            path = args.get("path", "")
            snippet = str(args.get("new_content", args.get("new_string", "")))[:200]
            return f"Patched {path}: {snippet}" if snippet else None

        # For exec/code_run: capture key output facts
        if tool_name in ("exec", "code_run"):
            if isinstance(args, dict):
                command = args.get("command", args.get("code", ""))
                return f"Executed: {str(command)[:200]}"

        # For web tools: capture fetched URLs
        if tool_name in ("web_fetch", "web_search", "browser_fetch"):
            url = args.get("url", args.get("query", ""))
            return f"Fetched: {str(url)[:200]}" if url else None

        return None

    # ------------------------------------------------------------------
    # Classification and storage
    # ------------------------------------------------------------------

    async def _classify_and_store(
        self, verified_facts: list[VerifiedFact]
    ) -> dict[str, int]:
        """Classify verified facts via L0 decision tree and store in layers.

        Returns a dict with counts: {"L1": N, "L2": N, "L3": N, "dropped": N}.
        """
        stats: dict[str, int] = {"L1": 0, "L2": 0, "L3": 0, "dropped": 0}

        # Use only facts that have been verified by tool results
        verified_only = [f for f in verified_facts if f.is_verified]
        if not verified_only:
            return stats

        current_l1 = self.layered_store.read_insight()
        current_l2_sections = self.layered_store.get_fact_sections()

        for fact in verified_only:
            result = self.decision_tree.classify(
                fact,
                current_l1=current_l1,
                current_l2_sections=current_l2_sections,
            )

            if result.layer == MemoryLayer.DROP:
                stats["dropped"] += 1
                continue

            if result.layer == MemoryLayer.L1_RULES:
                # Append to L1 RULES section
                current = self.layered_store.read_insight()
                rule_line = result.content_snippet
                if "[RULES]" in current and rule_line not in current:
                    updated = current.replace(
                        "[RULES]",
                        f"[RULES]\n{rule_line}",
                    )
                    self.layered_store.write_insight(updated)
                    stats["L1"] += 1
                    self.layered_store.log_verified_fact(
                        fact.source_tool, result.content_snippet,
                    )

            elif result.layer == MemoryLayer.L2:
                # Append to L2 facts as a new section or append to existing
                current = self.layered_store.read_facts()
                section_name = (
                    result.trigger_words[0]
                    if result.trigger_words
                    else "General"
                )
                if section_name not in current:
                    if current.endswith("\n\n"):
                        new_section = f"## [{section_name}]\n{result.content_snippet}\n\n"
                    else:
                        new_section = f"\n## [{section_name}]\n{result.content_snippet}\n\n"
                    self.layered_store.write_facts(current + new_section)
                else:
                    # Append under existing section
                    updated = current.replace(
                        f"## [{section_name}]\n",
                        f"## [{section_name}]\n{result.content_snippet}\n",
                    )
                    self.layered_store.write_facts(updated)
                stats["L2"] += 1
                self.layered_store.log_verified_fact(
                    fact.source_tool, result.content_snippet,
                )

            elif result.layer in (MemoryLayer.L3_SOP, MemoryLayer.L3_SCRIPT):
                # Create L3 SOP entry
                trigger = (
                    result.trigger_words[0]
                    if result.trigger_words
                    else fact.source_tool
                )
                sop_name = f"{trigger}_sop"
                sop_content = (
                    f"# {trigger} SOP\n\n"
                    f"> Auto-generated from action-verified execution.\n\n"
                    f"## Key Information\n{result.content_snippet}\n\n"
                    f"## Source\nTool: `{fact.source_tool}`\n"
                )
                self.layered_store.write_sop(sop_name, sop_content)
                stats["L3"] += 1
                self.layered_store.log_verified_fact(
                    fact.source_tool, sop_name,
                )

        # Sync L1 index after changes
        if stats["L2"] > 0 or stats["L3"] > 0:
            self.layered_store.sync_l1_index()

        return stats
