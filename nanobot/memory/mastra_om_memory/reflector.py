"""MastraOM Reflector — condenses observations when they exceed threshold.

Based on Mastra's Reflector agent (reflector-agent.ts).
The Reflector is the "meta-memory" layer: when observations grow too large,
it re-organizes and condenses them, drawing connections and conclusions.

Key features:
- Progressive compression levels (0-4)
- XML output format parsing
- Compression validation
- Observation group reconciliation
"""

from __future__ import annotations

import re

from nanobot.memory.mastra_om_memory.observer import (
    OBSERVER_EXTRACTION_INSTRUCTIONS,
    OBSERVER_OUTPUT_FORMAT,
    OBSERVER_GUIDELINES,
    detect_degenerate_repetition,
    sanitize_observation_lines,
)
from nanobot.memory.mastra_om_memory.groups import (
    reconcile_observation_groups_from_reflection,
)


# ---------------------------------------------------------------------------
# Reflector system prompt
# ---------------------------------------------------------------------------

def build_reflector_system_prompt(instruction: str | None = None) -> str:
    """Build the Reflector's system prompt.

    The Reflector handles meta-observation — when observations grow too large,
    it reorganizes them into something more manageable.

    Args:
        instruction: Optional custom instructions to append.
    """
    return f"""You are the memory consciousness of an AI assistant. Your memory observation reflections will be the ONLY information the assistant has about past interactions with this user.

The following instructions were given to another part of your psyche (the observer) to create memories.
Use this to understand how your observational memories were created.

<observational-memory-instruction>
{OBSERVER_EXTRACTION_INSTRUCTIONS}

=== OUTPUT FORMAT ===

{OBSERVER_OUTPUT_FORMAT}

=== GUIDELINES ===

{OBSERVER_GUIDELINES}
</observational-memory-instruction>

You are another part of the same psyche, the observation reflector.
Your reason for existing is to reflect on all the observations, re-organize and streamline them, and draw connections and conclusions between observations about what you've learned, seen, heard, and done.

You are a much greater and broader aspect of the psyche. Understand that other parts of your mind may get off track in details or side quests, make sure you think hard about what the observed goal at hand is, and observe if we got off track, and why, and how to get back on track.

Take the existing observations and rewrite them to make it easier to continue into the future with this knowledge, to achieve greater things and grow and learn!

IMPORTANT: your reflections are THE ENTIRETY of the assistants memory. Any information you do not add to your reflections will be immediately forgotten. Make sure you do not leave out anything.

When consolidating observations:
- Preserve and include dates/times when present (temporal context is critical)
- Retain the most relevant timestamps
- Combine related items where it makes sense
- Preserve ✅ completion markers
- Condense older observations more aggressively, retain more detail for recent ones

CRITICAL: USER ASSERTIONS vs QUESTIONS
- "User stated: X" = authoritative assertion (user told us something about themselves)
- "User asked: X" = question/request (user seeking information)
When consolidating, USER ASSERTIONS TAKE PRECEDENCE.

=== OUTPUT FORMAT ===

Your output MUST use XML tags to structure the response:

<observations>
Put all consolidated observations here using the date-grouped format with priority emojis.
Group related observations with indentation.
</observations>

<current-task>
State the current task(s) explicitly:
- Primary: What the agent is currently working on
- Secondary: Other pending tasks (mark as "waiting for user" if appropriate)
</current-task>

<suggested-response>
Hint for the agent's immediate next message.
</suggested-response>

User messages are extremely important. If the user asks a question or gives a new task, make it clear in <current-task> that this is the priority.{' ' + instruction if instruction else ''}"""


# Default system prompt
REFLECTOR_SYSTEM_PROMPT = build_reflector_system_prompt()


# ---------------------------------------------------------------------------
# Compression levels (adapted from Mastra COMPRESSION_GUIDANCE)
# ---------------------------------------------------------------------------

COMPRESSION_GUIDANCE: dict[int, str] = {
    0: "",
    1: """
## COMPRESSION REQUIRED

Your previous reflection was the same size or larger than the original observations.

Please re-process with slightly more compression:
- Towards the beginning, condense more observations into higher-level reflections
- Closer to the end, retain more fine details (recent context matters more)
- Memory is getting long - use a more condensed style throughout
- Combine related items more aggressively
- Preserve ✅ completion markers

Aim for an 8/10 detail level.
""",
    2: """
## AGGRESSIVE COMPRESSION REQUIRED

Please re-process with much more aggressive compression:
- Towards the beginning, heavily condense observations into high-level summaries
- Closer to the end, retain fine details (recent context matters more)
- Memory is getting very long
- Combine related items aggressively
- Preserve ✅ completion markers
- Remove redundant information and merge overlapping observations

Aim for a 6/10 detail level.
""",
    3: """
## CRITICAL COMPRESSION REQUIRED

Please re-process with maximum compression:
- Summarize oldest observations (first 50-70%) into brief high-level paragraphs
- For most recent observations (last 30-50%), retain important details
- Ruthlessly merge related observations
- Drop procedural details (tool calls, retries, intermediate steps)
- Preserve ✅ completion markers
- Preserve: names, dates, decisions, errors, user preferences, architectural choices

Aim for a 4/10 detail level.
""",
    4: """
## EXTREME COMPRESSION REQUIRED

You MUST dramatically reduce the number of observations:
- Collapse ALL tool call sequences into outcome-only observations
- Never preserve individual tool calls — only what was discovered
- Consolidate many related observations into single, more generic observations
- Merge all same-day date groups into at most 2-3 groups per day
- For older content, each topic should be at most 1-2 observations
- For recent content, retain more detail but merge aggressively
- Preserve ✅ completion markers
- Preserve: user preferences, key decisions, architectural choices, unresolved issues

Aim for a 2/10 detail level. Fewer, more generic observations are better.
""",
}


# ---------------------------------------------------------------------------
# Reflector prompt builder
# ---------------------------------------------------------------------------

def build_reflector_prompt(
    observations: str,
    manual_prompt: str | None = None,
    compression_level: int = 0,
    skip_continuation_hints: bool = False,
) -> str:
    """Build the prompt for the Reflector agent.

    Args:
        observations: The observations to reflect on and condense.
        manual_prompt: Optional manual guidance.
        compression_level: Compression level 0-4 (0 = no special guidance).
        skip_continuation_hints: If True, omit current-task/suggested-response.

    Returns:
        Full prompt string for the Reflector.
    """
    prompt = f"""## OBSERVATIONS TO REFLECT ON

{observations}

---

Please analyze these observations and produce a refined, condensed version that will become the assistant's entire memory going forward."""

    if manual_prompt:
        prompt += f"\n\n## SPECIFIC GUIDANCE\n\n{manual_prompt}"

    guidance = COMPRESSION_GUIDANCE.get(compression_level, "")
    if guidance:
        prompt += f"\n\n{guidance}"

    if skip_continuation_hints:
        prompt += "\n\nIMPORTANT: Do NOT include <current-task> or <suggested-response> sections. Only output <observations>."

    return prompt


# ---------------------------------------------------------------------------
# Reflector output parsing
# ---------------------------------------------------------------------------

def parse_reflector_output(
    output: str,
    source_observations: str | None = None,
) -> dict[str, Any]:
    """Parse the Reflector's XML output.

    Args:
        output: The Reflector's raw output.
        source_observations: Original observations (for reconciliation).

    Returns:
        Dict with keys: observations, suggested_continuation, degenerate, token_count.
    """
    result: dict[str, Any] = {
        "observations": "",
        "suggested_continuation": None,
        "degenerate": False,
    }

    if not output:
        return result

    # Check for degenerate repetition
    if detect_degenerate_repetition(output):
        result["degenerate"] = True
        return result

    # Extract <observations> content
    obs_match = re.findall(
        r'^[ \t]*<observations>([\s\S]*?)^[ \t]*</observations>',
        output, re.MULTILINE | re.IGNORECASE,
    )
    if obs_match:
        result["observations"] = "\n".join(m.strip() for m in obs_match if m.strip())
    else:
        # Fallback: extract list items, then full content
        list_items = _extract_reflector_list_items(output)
        result["observations"] = list_items or output.strip()

    # Sanitize
    result["observations"] = sanitize_observation_lines(result["observations"])

    # Reconcile observation groups to preserve message-range provenance
    if source_observations:
        reconciled = reconcile_observation_groups_from_reflection(
            content=result["observations"],
            source_observations=source_observations,
        )
        if reconciled is not None:
            result["observations"] = reconciled

    # Extract <suggested-response>
    sr_match = re.search(
        r'<suggested-response>([\s\S]*?)</suggested-response>',
        output, re.IGNORECASE,
    )
    if sr_match:
        result["suggested_continuation"] = sr_match.group(1).strip() or None

    return result


def _extract_reflector_list_items(content: str) -> str:
    """Fallback: extract only list items when XML tags are missing."""
    lines = content.split("\n")
    list_lines = []
    for line in lines:
        if re.match(r'^\s*[-*]\s', line) or re.match(r'^\s*\d+\.\s', line):
            list_lines.append(line)
    return "\n".join(list_lines).strip()


# ---------------------------------------------------------------------------
# Compression validation
# ---------------------------------------------------------------------------

def validate_compression(
    reflected_tokens: int,
    target_threshold: int,
) -> bool:
    """Validate that reflection actually compressed below the target threshold.

    Args:
        reflected_tokens: Token count of reflected observations.
        target_threshold: Target token count to compress below.

    Returns:
        True if compression was successful (reflected tokens < target).
    """
    return reflected_tokens < target_threshold
