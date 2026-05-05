"""Layerga memory algorithm — GenericAgent-style L0-L4 layered memory for nanobot.

Based on the GenericAgent multi-layer memory architecture:
  L0: Meta-rules constitution (layerga/constitution.md)
  L1: Minimal insight index (memory/layer_insight.txt, ≤30 lines)
  L2: Environment fact base (memory/layer_facts.txt, ## [SECTION] blocks)
  L3: Task SOP library (memory/sop/*.md + *.py)
  L4: Session archives (memory/archives/)

Core principles:
  - "No Execution, No Memory" — only action-verified facts are stored
  - "Minimum Sufficient Pointer" — upper layers only keep the shortest locator
  - "Self-Evolution" — Agent autonomously decides what, where, and how to remember
  - "LLM-driven management" — Agent is both executor and memory librarian

Usage::

    from nanobot.memory import MemoryRegistry
    from nanobot.memory.layerga_memory import LayergaMemoryAlgorithm

    registry = MemoryRegistry()
    registry.register(LayergaMemoryAlgorithm())
    algo = registry.get("layerga_memory")
    components = algo.build(...)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from nanobot.memory.base import MemoryAlgorithm, MemoryComponents
from nanobot.memory.layerga_memory.auto_compact import LayergaAutoCompact
from nanobot.memory.layerga_memory.consolidator import LayergaConsolidator
from nanobot.memory.layerga_memory.decision_tree import L0DecisionTree
from nanobot.memory.layerga_memory.dream import LayergaDream
from nanobot.memory.layerga_memory.store import LayergaStore

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import SessionManager


class LayergaMemoryAlgorithm(MemoryAlgorithm):
    """GenericAgent L0-L4 layered memory algorithm.

    Implements a hierarchical memory system where information is classified
    by the L0 decision tree and stored in the appropriate layer:

    - **L0**: Constitution (meta-rules) — the "memory law" governing all writes
    - **L1**: Insight index (≤30 lines) — minimal navigation index
    - **L2**: Fact base — environment-specific facts an LLM cannot infer
    - **L3**: Task records — SOPs and utility scripts
    - **L4**: Session archives — compressed conversation history

    Key features:
    - L0 decision-tree-driven information classification
    - Action-verification principle (No Execution, No Memory)
    - L1 ≤30 line hard constraint with ROI-based cleanup
    - Minimal patch-only file modifications
    - Multi-layer context injection into system prompt
    - Skill crystallization via Dream Phase 2
    - L4 session archive management

    Zero external dependencies — pure Python implementation using
    the standard nanobot file I/O stack.
    """

    name = "layerga_memory"

    def build(
        self,
        workspace: Path,
        provider: "LLMProvider",
        model: str,
        sessions: "SessionManager",
        context_window_tokens: int,
        build_messages: Any,
        get_tool_definitions: Any,
        max_completion_tokens: int,
        session_ttl_minutes: int,
        max_batch_size: int,
        max_iterations: int,
        max_tool_result_chars: int,
        annotate_line_ages: bool,
        embedding_config: Any = None,
    ) -> MemoryComponents:
        # ------------------------------------------------------------------
        # Store: L0-L4 file I/O layer
        # ------------------------------------------------------------------
        store = LayergaStore(
            workspace=workspace,
            l1_max_lines=30,
        )

        # ------------------------------------------------------------------
        # Decision tree: L0 classification engine
        # ------------------------------------------------------------------
        constitution = store.read_constitution()
        decision_tree = L0DecisionTree(
            constitution=constitution,
            l1_max_lines=30,
            confidence_threshold=0.5,
        )

        # ------------------------------------------------------------------
        # Consolidator: online token-budget consolidation with classification
        # ------------------------------------------------------------------
        consolidator = LayergaConsolidator(
            store=store,
            decision_tree=decision_tree,
            provider=provider,
            model=model,
            sessions=sessions,
            context_window_tokens=context_window_tokens,
            build_messages=build_messages,
            get_tool_definitions=get_tool_definitions,
            max_completion_tokens=max_completion_tokens,
            enable_classification=True,
        )

        # ------------------------------------------------------------------
        # Dream: offline deep crystallization (three-phase)
        # ------------------------------------------------------------------
        dream = LayergaDream(
            store=store,
            decision_tree=decision_tree,
            provider=provider,
            model=model,
            max_batch_size=max_batch_size,
            max_iterations=max_iterations,
            max_tool_result_chars=max_tool_result_chars,
            annotate_line_ages=annotate_line_ages,
            enable_l1_cleanup=True,
            enable_auto_crystallize=True,
        )

        # ------------------------------------------------------------------
        # AutoCompact: idle session compression + LTM triggering
        # ------------------------------------------------------------------
        auto_compact = LayergaAutoCompact(
            sessions=sessions,
            consolidator=consolidator,
            session_ttl_minutes=session_ttl_minutes,
            decision_tree=decision_tree,
            enable_l4_archive=True,
        )

        return MemoryComponents(
            store=store,
            consolidator=consolidator,
            dream=dream,
            auto_compact=auto_compact,
        )
