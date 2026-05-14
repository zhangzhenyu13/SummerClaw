"""Mem0V3 Dream — offline cron-scheduled memory deep processing.

Two-phase processor: Phase 1 analyzes MEMORY.md + vector memories; Phase 2
edits MEMORY.md via AgentRunner and optionally generates dreamed-* skills.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.memory.mem0v3_memory.store import Mem0V3Store
from nanobot.utils.prompt_templates import render_template

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider

_STALE_THRESHOLD_DAYS = 14

DREAM_PHASE1_TEMPLATE = """You are a memory analyst reviewing an agent's long-term memory.

## MEMORY.md (current state)
```
{{ memory_md | default("(empty)") }}
```

## Recent Vector Memories
{{ vector_memories | default("(none)") }}

## Task
Analyze the above memory content and produce a structured analysis:

1. **Duplicates**: Identify any duplicate or near-duplicate information.
2. **Contradictions**: Flag any conflicting or outdated facts.
3. **Gaps**: Note missing information that would be valuable to remember.
4. **Consolidation suggestions**: Recommend how to reorganize or merge information.
5. **Skill opportunities**: Identify patterns that could become reusable skills.

## Output Format
Return your analysis as a plain text report. Be concise but thorough.
For each finding, reference the specific memory text involved.

## Context
- Stale threshold: {{ stale_threshold_days }} days
- Total vector memories: {{ total_memories }}
- Total entities: {{ total_entities }}
"""

DREAM_PHASE2_SYSTEM = """You are a memory editor. Edit the agent's MEMORY.md file based on an analysis.

## Tools
- read_file: Read MEMORY.md or any file
- edit_file: Make targeted edits (use edit_file, prefer small patches)
- write_file: Create new skill files under skills/

## Guidelines
1. Be conservative — when in doubt, keep existing information.
2. Preserve specifics: names, dates, numbers, quotes.
3. Merge true duplicates; keep the richer version.
4. Resolve contradictions by noting the change explicitly.
5. Keep history: old facts stay for context; note when things changed.
6. Generate skills if analysis identifies a crystallizable pattern.

## Important
- Make minimal edits — small targeted changes
- MEMORY.md should remain human-readable
- Use ISO dates (YYYY-MM-DD) for temporal references
"""


class Mem0V3Dream:
    """Two-phase offline memory processor for mem0 v3."""

    def __init__(
        self,
        store: Mem0V3Store,
        provider: "LLMProvider",
        model: str,
        max_batch_size: int = 20,
        max_iterations: int = 15,
        max_tool_result_chars: int = 16_000,
        annotate_line_ages: bool = True,
        algo_name: str = "mem0v3_memory",
    ):
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

    def _build_tools(self) -> ToolRegistry:
        from nanobot.agent.skills import BUILTIN_SKILLS_DIR
        from nanobot.agent.tools.filesystem import EditFileTool, ReadFileTool, SkillPrefixWriteFileTool

        tools = ToolRegistry()
        workspace = self.store.workspace
        extra_read = [BUILTIN_SKILLS_DIR] if BUILTIN_SKILLS_DIR.exists() else None
        tools.register(ReadFileTool(workspace=workspace, allowed_dir=workspace, extra_allowed_dirs=extra_read))
        tools.register(EditFileTool(workspace=workspace, allowed_dir=workspace))
        skills_dir = workspace / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        tools.register(SkillPrefixWriteFileTool(skill_prefix=f"dreamed--{self._algo_name}", workspace=workspace, allowed_dir=skills_dir))
        return tools

    async def run(self) -> dict[str, Any]:
        logger.info("Mem0V3 Dream starting...")
        phase1_result = await self._phase1_analyze()
        analysis = phase1_result.get("analysis", "")
        logger.info("Dream Phase 1 complete: {} chars", len(analysis))
        if not analysis:
            return {"phase1_analysis": "", "phase2_edits": 0, "skills_generated": 0, "memories_processed": 0}
        phase2_result = await self._phase2_rewrite(analysis)
        logger.info("Dream Phase 2 complete: {} edits, {} skills",
                     phase2_result.get("edits", 0), phase2_result.get("skills", 0))
        return {
            "phase1_analysis": analysis,
            "phase2_edits": phase2_result.get("edits", 0),
            "skills_generated": phase2_result.get("skills", 0),
            "memories_processed": self.store.memory_count,
        }

    async def _phase1_analyze(self) -> dict[str, Any]:
        memory_md = self.store.read_memory_md()
        all_memories = self.store.get_all_memories()
        all_memories.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        recent = all_memories[:self.max_batch_size]
        vector_summary = "\n".join(
            f"- [{m.get('created_at', '?')[:10]}] {m.get('text', '')[:200]}"
            for m in recent
        )
        memory_md_display = memory_md
        if self.annotate_line_ages and memory_md:
            memory_md_display = self._annotate_with_ages(memory_md)
        prompt = render_template(
            DREAM_PHASE1_TEMPLATE,
            memory_md=memory_md_display,
            vector_memories=vector_summary,
            stale_threshold_days=_STALE_THRESHOLD_DAYS,
            total_memories=self.store.memory_count,
            total_entities=self.store.entity_count,
        )
        try:
            response = await self.provider.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=4096,
            )
            return {"analysis": response.content if response else ""}
        except Exception as e:
            logger.error(f"Dream Phase 1 failed: {e}")
            return {"analysis": ""}

    async def _phase2_rewrite(self, analysis: str) -> dict[str, int]:
        system_msg = DREAM_PHASE2_SYSTEM
        user_msg = f"""## Analysis Report
{analysis}

## Task
Based on the analysis above, edit MEMORY.md to improve the agent's memory.
Use read_file to check current state, then use edit_file for targeted changes.
If the analysis suggests a reusable skill, create it under skills/dreamed-<name>/SKILL.md.

## Key Files
- memory/{self._algo_name}/MEMORY.md — the main memory file to edit
- skills/dreamed-*/SKILL.md — skills to create (if applicable)
"""
        spec = AgentRunSpec(
            system_prompt=system_msg,
            user_message=user_msg,
            tools=self._tools,
            max_iterations=self.max_iterations,
            max_tool_result_chars=self.max_tool_result_chars,
        )
        try:
            summary = await self._runner.run(spec)
            edits = getattr(summary, "edit_count", 0) if summary else 0
            skills = getattr(summary, "write_count", 0) if summary else 0
            return {"edits": edits, "skills": skills}
        except Exception as e:
            logger.error(f"Dream Phase 2 failed: {e}")
            return {"edits": 0, "skills": 0}

    def _annotate_with_ages(self, memory_md: str) -> str:
        try:
            from nanobot.utils.gitstore import GitStore
            gs = GitStore(self.store.workspace)
            blame = gs.blame_file(self.store._memory_md_path)
            if not blame:
                return memory_md
        except Exception:
            return memory_md
        lines = memory_md.split("\n")
        annotated: list[str] = []
        now = datetime.now()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                annotated.append(line)
                continue
            line_blame = blame.get(i + 1) if hasattr(blame, "get") else None
            if line_blame and hasattr(line_blame, "date"):
                try:
                    age_days = (now - line_blame.date).days
                    annotated.append(f"{line}  ← {age_days}d")
                    continue
                except Exception:
                    pass
            annotated.append(line)
        return "\n".join(annotated)
