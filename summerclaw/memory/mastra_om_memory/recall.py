"""Auto-recall module: LLM-judged retrieval of original session history.

Parses history_cursor references from OBSERVATIONS.md cycle headers, calls
a lightweight LLM to judge which cycles are relevant to the current
conversation, fetches the corresponding raw history.jsonl entries, and
caches them for injection into the next context build.

This module is algorithm-internal — no tools or agent modules are modified.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from summerclaw.memory.mastra_om_memory.store import MastraOMStore
    from summerclaw.providers.base import LLMProvider


@dataclass
class RecallConfig:
    """Configuration for auto-recall behavior."""

    enabled: bool = True
    max_recall_bytes: int = 15_000
    max_cycles: int = 5
    session_tail_messages: int = 10
    model_override: str | None = None
    timeout_seconds: float = 10.0


_CYCLE_CURSOR_RE = re.compile(
    r"## Observation Cycle (\S+)\s*[—–-]\s*([^\n]+?)\s*history_cursor=\"(\d+):(\d+)\"",
)
_HISTORY_CURSOR_RE = re.compile(r'history_cursor="(\d+):(\d+)"')

_RECALL_PROMPT = """\
You are a memory recall judge. Given the recent conversation and available \
observation cycles, determine which cycles contain original session logs \
relevant to the current conversation.

## Recent Conversation
{session_tail}

## Available Observation Cycles
{cycle_summaries}

Return a JSON array of cursor ranges for cycles whose original history \
would provide useful additional context for the current conversation. \
Only include cycles that are clearly relevant.

Example: [{{"start": 42, "end": 47}}, {{"start": 55, "end": 60}}]
If no cycles are relevant, return: []

Response (JSON only):
"""


def parse_history_cursors(observations_text: str) -> list[dict[str, Any]]:
    """Parse history_cursor references from OBSERVATIONS.md cycle headers.

    Returns a list of dicts sorted by position in file (chronological):
    [{"cycle_id": "...", "timestamp": "...", "cursor_start": N, "cursor_end": M}, ...]

    Lines without history_cursor are skipped (e.g., post-reflection cycles).
    """
    if not observations_text:
        return []

    results: list[dict[str, Any]] = []
    for line in observations_text.split("\n"):
        if not line.startswith("## Observation Cycle"):
            continue
        m = _CYCLE_CURSOR_RE.search(line)
        if not m:
            # Try a more lenient match — header may have extra content
            hm = _HISTORY_CURSOR_RE.search(line)
            if not hm:
                continue
            # Extract cycle_id and timestamp from the line prefix
            prefix = line[: hm.start()]
            parts = prefix.split("—", 1) if "—" in prefix else prefix.split(" - ", 1)
            cycle_id = parts[0].replace("## Observation Cycle", "").strip() if parts else ""
            timestamp = parts[1].strip() if len(parts) > 1 else ""
            results.append({
                "cycle_id": cycle_id[:8],
                "timestamp": timestamp,
                "cursor_start": int(hm.group(1)),
                "cursor_end": int(hm.group(2)),
            })
        else:
            results.append({
                "cycle_id": m.group(1)[:8],
                "timestamp": m.group(2).strip(),
                "cursor_start": int(m.group(3)),
                "cursor_end": int(m.group(4)),
            })

    return results


def build_cycle_summaries(
    cycles: list[dict[str, Any]],
    observations_text: str,
    max_facts: int = 3,
) -> list[dict[str, Any]]:
    """Build summaries for each cycle, extracting the first N observation fact lines.

    Each summary contains: cycle_id, timestamp, cursor range, and up to
    max_facts observation lines from that cycle's section in OBSERVATIONS.md.
    """
    if not observations_text or not cycles:
        return []

    # Split observations into sections by cycle headers
    lines = observations_text.split("\n")
    sections: list[tuple[str, list[str]]] = []  # (cycle_id, content_lines)
    current_cycle_id: str | None = None
    current_lines: list[str] = []

    for line in lines:
        if line.startswith("## Observation Cycle"):
            if current_cycle_id is not None:
                sections.append((current_cycle_id, current_lines))
            # Extract cycle ID from header
            parts = line.split()
            current_cycle_id = parts[3][:8] if len(parts) > 3 else ""
            current_lines = []
        elif current_cycle_id is not None:
            current_lines.append(line)

    if current_cycle_id is not None:
        sections.append((current_cycle_id, current_lines))

    # Map cycle_id -> fact lines
    fact_map: dict[str, list[str]] = {}
    for cid, section_lines in sections:
        facts: list[str] = []
        for sl in section_lines:
            stripped = sl.strip()
            if stripped.startswith("*"):
                facts.append(stripped[1:].strip())
                if len(facts) >= max_facts:
                    break
        fact_map[cid] = facts

    # Build summaries
    summaries: list[dict[str, Any]] = []
    for c in cycles:
        cid = c["cycle_id"]
        facts = fact_map.get(cid, [])
        facts_text = "; ".join(facts) if facts else "(no facts extracted)"
        summaries.append({
            **c,
            "facts": facts,
            "summary_line": (
                f"- Cycle {cid} ({c['timestamp']}, "
                f"cursor {c['cursor_start']}:{c['cursor_end']}): {facts_text}"
            ),
        })

    return summaries


def build_recall_prompt(
    session_tail: list[dict[str, Any]],
    cycle_summaries: list[dict[str, Any]],
) -> str:
    """Build the LLM prompt for recall judgement."""
    # Format session tail
    tail_lines: list[str] = []
    for msg in session_tail:
        role = msg.get("role", "unknown").upper()
        content = str(msg.get("content", ""))[:200]  # truncate long messages
        ts = msg.get("timestamp", "")
        prefix = f"[{ts}] " if ts else ""
        tail_lines.append(f"{prefix}{role}: {content}")

    session_tail_text = "\n".join(tail_lines) if tail_lines else "(no recent messages)"

    # Format cycle summaries
    cycle_lines = [s["summary_line"] for s in cycle_summaries]
    cycle_text = "\n".join(cycle_lines) if cycle_lines else "(no cycles available)"

    return _RECALL_PROMPT.format(
        session_tail=session_tail_text,
        cycle_summaries=cycle_text,
    )


def parse_recall_response(response: str) -> list[tuple[int, int]]:
    """Parse LLM response into cursor ranges.

    Expected format: JSON array of {"start": N, "end": M} objects.
    Returns empty list on parse failure.
    """
    if not response:
        return []

    # Strip markdown code fences if present
    text = response.strip()
    if text.startswith("```"):
        # Remove opening fence (with optional language tag)
        text = re.sub(r"^```\w*\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("[OM:recall] failed to parse LLM response as JSON: {}", text[:200])
        return []

    if not isinstance(data, list):
        logger.warning("[OM:recall] LLM response is not a list: {}", type(data).__name__)
        return []

    ranges: list[tuple[int, int]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        start = item.get("start")
        end = item.get("end")
        if isinstance(start, int) and isinstance(end, int) and start <= end:
            ranges.append((start, end))

    return ranges


def fetch_recalled_entries(
    entries: list[dict[str, Any]],
    cursor_ranges: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    """Filter history.jsonl entries whose cursor falls within any of the ranges.

    Returns entries sorted by cursor ascending, without duplicates.
    """
    if not entries or not cursor_ranges:
        return []

    recalled: list[dict[str, Any]] = []
    for entry in entries:
        cursor = entry.get("cursor", 0)
        if any(start <= cursor <= end for start, end in cursor_ranges):
            recalled.append(entry)

    recalled.sort(key=lambda e: e.get("cursor", 0))
    return recalled


def format_recall_section(entries: list[dict[str, Any]], max_bytes: int) -> str:
    """Format recalled entries into injection text with byte budgeting.

    Groups entries by session (timestamp date), iterates newest-first,
    and stops when the byte budget is exhausted.
    """
    if not entries:
        return ""

    header = "## Recent Session Context\nRaw conversation logs recalled from relevant past sessions:\n"
    budget = max_bytes - len(header.encode("utf-8"))

    # Group by date (from timestamp)
    from datetime import datetime

    def _entry_date(entry: dict[str, Any]) -> str:
        ts = entry.get("timestamp", "")
        if len(ts) >= 10:
            return ts[:10]
        return "unknown"

    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        date_key = _entry_date(entry)
        groups.setdefault(date_key, []).append(entry)

    # Sort groups by date descending
    sorted_dates = sorted(groups.keys(), reverse=True)

    # Build sections newest-first, respecting byte budget
    accepted_sections: list[str] = []
    for date_key in sorted_dates:
        group_entries = groups[date_key]
        section_lines: list[str] = [f"\n[Session — {date_key}]"]
        for entry in group_entries:
            section_lines.append(entry.get("content", ""))
        section_text = "\n".join(section_lines)
        section_bytes = len(section_text.encode("utf-8"))

        if budget - section_bytes < 0:
            break
        accepted_sections.append(section_text)
        budget -= section_bytes

    if not accepted_sections:
        return ""

    # Reverse to chronological order
    accepted_sections.reverse()
    body = "\n".join(accepted_sections)
    return f"{header}{body}"


async def judge_and_recall(
    store: MastraOMStore,
    provider: LLMProvider,
    model: str,
    session: Any,
    config: RecallConfig,
) -> None:
    """Top-level recall orchestration.

    1. Parse cycle cursors from OBSERVATIONS.md
    2. Build cycle summaries
    3. Call LLM to judge relevance
    4. Fetch matching history entries
    5. Cache results in store._recalled_entries_cache
    """
    try:
        observations_text = store.read_observations()
        if not observations_text:
            store._recalled_entries_cache = []
            return

        # Parse cycles
        all_cycles = parse_history_cursors(observations_text)
        if not all_cycles:
            store._recalled_entries_cache = []
            return

        # Take the N most recent cycles
        recent_cycles = all_cycles[-config.max_cycles:]

        # Build summaries
        summaries = build_cycle_summaries(recent_cycles, observations_text)
        if not summaries:
            store._recalled_entries_cache = []
            return

        # Get session tail (recent messages)
        session_tail: list[dict[str, Any]] = []
        if hasattr(session, "messages") and session.messages:
            session_tail = session.messages[-config.session_tail_messages:]

        # Build prompt and call LLM
        prompt = build_recall_prompt(session_tail, summaries)
        recall_model = config.model_override or model

        try:
            response = await provider.chat_with_retry(
                messages=[
                    {"role": "system", "content": "You are a memory recall judge. Respond with JSON only."},
                    {"role": "user", "content": prompt},
                ],
                model=recall_model,
                max_tokens=512,
                temperature=0.0,
            )
            llm_text = response.content or ""
        except Exception as e:
            logger.warning("[OM:recall] LLM call failed: {}", e)
            store._recalled_entries_cache = []
            return

        # Parse response
        cursor_ranges = parse_recall_response(llm_text)
        if not cursor_ranges:
            store._recalled_entries_cache = []
            logger.debug("[OM:recall] no cycles selected for recall")
            return

        # Fetch entries
        all_entries = store._read_entries()
        recalled = fetch_recalled_entries(all_entries, cursor_ranges)

        store._recalled_entries_cache = recalled
        logger.info(
            "[OM:recall] recalled {} entries from {} cursor ranges",
            len(recalled), len(cursor_ranges),
        )

    except Exception:
        logger.warning("[OM:recall] judge_and_recall failed unexpectedly", exc_info=True)
        store._recalled_entries_cache = []
