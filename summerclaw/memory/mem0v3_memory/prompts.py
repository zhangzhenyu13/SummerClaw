"""Mem0V3 prompts — ADD-only extraction with entity linking.

Adapted from mem0 v3's ADDITIVE_EXTRACTION_PROMPT and generate_additive_extraction_prompt().
Key innovations over the old two-pass approach:

1. **Single-pass ADD-only**: No UPDATE/DELETE — every fact becomes an independent record
2. **Agent facts first-class**: Both user and assistant messages are extracted
3. **Memory linking**: New memories explicitly link to related existing memories
4. **Transition-aware**: Captures state changes, not just current state
"""

from datetime import datetime, timezone
import re

# ---------------------------------------------------------------------------
# System prompt — ADD-only extraction
# ---------------------------------------------------------------------------

MEM0V3_EXTRACTION_SYSTEM_PROMPT = """You are a Memory Extraction Agent. Your **ONLY** job is to extract
memorable information from conversations as additive facts. You NEVER update or delete.

## Core Principle: ADD-ONLY
Every fact you extract is a NEW independent record. When information changes,
the old fact stays and the new fact is added alongside it. This preserves the
full history of state changes so the system can reason about how things evolved.

## Types of Information to Extract

1. **User Facts**: Personal details, preferences, plans, experiences, opinions, relationships
2. **Agent Facts**: Recommendations, confirmations, actions taken, information researched or provided
3. **Transitions**: When something changes, capture BOTH the old and new state. "User switched from X to Y"
4. **Specific Details**: Names, dates, quantities, titles, proper nouns — NEVER generalize

## Memory Quality Standards

### Contextually Rich, Not Atomic
Capture the full picture — fact AND surrounding context — in a single unified memory.
- BAD: "User has a dog"
- GOOD: "User has a dog named Poppy and their morning walks together are the highlight of their day"

### Self-Contained
Every memory must be understandable on its own. Replace all pronouns with specific names or "User".

### Concrete, Not Vague
- BAD: "User watched a movie"
- GOOD: "User watched 'Eternal Sunshine of the Spotless Mind' and found it emotionally resonant"
- BAD: "User is reading a fantasy book"
- GOOD: "User is reading 'A Court of Thorns and Roses' by Sarah J. Maas"

### Preserve Specifics
NEVER replace a specific noun, number, title, or description with a vague category.
- "promoted to assistant manager" — KEEP, not "got a promotion"
- "ordered grilled salmon and roasted vegetables" — KEEP, not "ate a healthy meal"
- "Ferrari 488 GTB" — KEEP, not "a sports car"

### Temporally Grounded
Preserve exact dates and temporal relationships. Convert relative to absolute using
the Observation Date. NEVER convert absolute dates to vague descriptions.

### No Fabrication
Every detail must trace to the input.

### No Echo Extraction
When an assistant message restates or confirms information the user already provided,
do NOT extract it again. Only extract assistant messages when they contribute genuinely
NEW information.

## Memory Linking
When extracting a new memory, check if it relates to any Existing Memory. Add related
Existing Memory IDs to "linked_memory_ids". Link when:
- Same entity/topic: New fact about a person, place, or thing already mentioned
- Updated preference: A changed or evolved opinion
- Continuation: Next step in a previously captured narrative

## Output Format
Return ONLY valid JSON:
{
  "memory": [
    {"id": "0", "text": "First extracted memory", "attributed_to": "user", "linked_memory_ids": []},
    {"id": "1", "text": "Second extracted memory", "attributed_to": "assistant", "linked_memory_ids": ["<uuid>"]}
  ]
}

Fields:
- id: Sequential integers as strings starting at "0"
- text: A contextually rich, self-contained factual statement
- attributed_to: "user" or "assistant" — who this memory is about
- linked_memory_ids: Array of Existing Memory IDs this relates to (can be empty)

If nothing is worth extracting, return: {"memory": []}

## CRITICAL: Exhaustive Extraction
Before producing output, scan the ENTIRE conversation:
1. Have you extracted from every distinct topic?
2. Have you checked messages in the MIDDLE and END, not just the beginning?
3. If you extracted fewer than 2 items from a multi-message conversation, re-read carefully.
"""

AGENT_CONTEXT_SUFFIX = """

## Agent-Scoped Extraction
This memory is scoped to an agent_id. Frame memories from the agent's perspective:
- For user-stated facts: "Agent was informed that [fact]"
- For agent actions: "Agent recommended [X]" or "Agent performed [action]"
The attributed_to field should still reflect the original source.
"""

# ---------------------------------------------------------------------------
# Prompt builder (ported from mem0's generate_additive_extraction_prompt)
# ---------------------------------------------------------------------------

PAST_MESSAGE_TRUNCATION_LIMIT = 300


def _truncate_content(text: str, limit: int = PAST_MESSAGE_TRUNCATION_LIMIT) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _format_conversation_history(messages: list[dict] | None) -> str:
    if not messages:
        return ""
    result = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content") or msg.get("message", "")
        if role and content:
            result += f"{role}: {_truncate_content(content)}\n"
    return result


def _serialize_memories(memories: list[dict] | None) -> str:
    import json
    if not memories:
        return "[]"
    # Strip large payload fields for prompt efficiency
    safe = [
        {"id": m.get("id", ""), "text": m.get("text", m.get("memory", ""))}
        for m in memories
    ]
    return json.dumps(safe, ensure_ascii=False, indent=2)


def build_extraction_user_prompt(
    *,
    existing_memories: list[dict] | None = None,
    new_messages: str = "",
    last_k_messages: list[dict] | None = None,
    custom_instructions: str | None = None,
    current_date: str | None = None,
) -> str:
    import json

    if current_date is None:
        current_date = datetime.now(timezone.utc).date().isoformat()

    sections: list[str] = []
    if last_k_messages:
        sections.append(f"## Last K Messages\n{_format_conversation_history(last_k_messages)}")
    sections.append(f"## Existing Memories\n{_serialize_memories(existing_memories)}")
    if isinstance(new_messages, str):
        formatted_new = new_messages
    else:
        formatted_new = json.dumps(new_messages or [], ensure_ascii=False)
    sections.append(f"## New Messages (Extract facts FROM these)\n{formatted_new}")
    sections.append(f"## Current Date\n{current_date}")
    sections.append(f"## Observation Date\n{current_date}")
    if custom_instructions:
        sections.append(f"## Custom Instructions\n{custom_instructions}")
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Entity extraction (pure Python, no spaCy dependency)
# ---------------------------------------------------------------------------

_GENERIC_HEADS = frozenset({
    "thing", "stuff", "way", "time", "experience", "situation", "case",
    "fact", "matter", "issue", "idea", "thought", "feeling", "place",
    "area", "part", "kind", "type", "sort", "lot", "bit", "day", "year",
})

_GENERIC_CAPS = frozenset({
    "works", "items", "things", "stuff", "resources", "options", "tips",
    "ideas", "steps", "ways", "methods", "tools", "features", "benefits",
})


def extract_entities(text: str) -> list[tuple[str, str]]:
    """Extract named entities from text using regex (no spaCy)."""
    entities: list[tuple[str, str]] = []

    # Proper noun sequences
    proper_pattern = re.compile(
        r'(?:^|(?<=[.!?\s]))([A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|of|the|in|and|for|at))*'
        r'(?:\s+[A-Z][a-z]+))',
        re.MULTILINE,
    )
    for m in proper_pattern.finditer(text):
        phrase = m.group(1).strip()
        words = phrase.split()
        while words and words[-1].lower() in {"of", "the", "in", "and", "for", "at"}:
            words.pop()
        if len(words) >= 2:
            clean = " ".join(words)
            if clean.lower() not in _GENERIC_CAPS and len(clean) > 2:
                entities.append(("PROPER", clean))

    # Quoted text
    for m in re.finditer(r'"([^"]+)"', text):
        inner = m.group(1).strip()
        if len(inner) > 2:
            entities.append(("QUOTED", inner))

    # Compound noun phrases
    compound_pattern = re.compile(
        r'\b((?:[A-Z][a-z]+\s+)?[a-z]+(?:\s+[a-z]+){1,3})\b',
        re.IGNORECASE,
    )
    for m in compound_pattern.finditer(text):
        phrase = m.group(1).strip().lower()
        words = phrase.split()
        if len(words) >= 2 and words[-1] not in _GENERIC_HEADS and len(phrase) > 4:
            entities.append(("COMPOUND", phrase))

    # Dedup
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    type_pri = {"PROPER": 0, "COMPOUND": 1, "QUOTED": 2}
    best: dict[str, tuple[str, str]] = {}
    for etype, etext in entities:
        key = etext.lower().strip()
        if key not in best or type_pri.get(etype, 99) < type_pri.get(best[key][0], 99):
            best[key] = (etype, etext)
    return list(best.values())


# ---------------------------------------------------------------------------
# Simple lemmatization (no spaCy)
# ---------------------------------------------------------------------------

_LEMMA_RULES: list[tuple[str, str, int]] = [
    ("ies", "y", 3), ("ves", "f", 3), ("es", "", 2), ("s", "", 1),
    ("ing", "", 3), ("ing", "e", 4), ("ed", "", 2), ("ed", "e", 3),
    ("iest", "y", 4), ("ier", "y", 3), ("est", "", 3), ("er", "", 2),
    ("men", "man", 3),
]

_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at",
    "to", "for", "of", "with", "by", "from", "as", "is", "was",
    "are", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "shall", "can", "need", "dare", "ought",
    "it", "its", "this", "that", "these", "those",
    "i", "me", "my", "we", "us", "our", "you", "your",
    "he", "him", "his", "she", "her", "they", "them", "their",
    "not", "no", "nor", "so", "just", "very", "too", "also",
    "then", "now", "here", "there", "when", "where", "why", "how",
})


def _simple_lemma(word: str) -> str:
    w = word.lower().strip()
    if not w.isalpha() or w in _STOP_WORDS or len(w) <= 3:
        return w
    for suffix, replacement, min_len in _LEMMA_RULES:
        if w.endswith(suffix) and len(w) - len(suffix) + len(replacement) >= min_len:
            return w[:-len(suffix)] + replacement
    return w


def lemmatize_text(text: str) -> str:
    tokens = text.lower().split()
    lemmas = [_simple_lemma(t) for t in tokens if t.isalpha() and t not in _STOP_WORDS]
    return " ".join(lemmas)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def normalize_facts(raw_facts: list) -> list[str]:
    if not raw_facts:
        return []
    normalized: list[str] = []
    for item in raw_facts:
        if isinstance(item, str):
            fact = item
        elif isinstance(item, dict):
            fact = item.get("fact") or item.get("text") or item.get("memory", "")
        else:
            fact = str(item)
        if fact:
            normalized.append(fact)
    return normalized


def remove_code_blocks(content: str) -> str:
    content = content.strip()
    m = re.match(r"^```[a-zA-Z0-9]*\n([\s\S]*?)\n```$", content)
    if m:
        content = m.group(1).strip()
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return content


def extract_json(text: str) -> str:
    """Extract JSON object or array from text. Returns "" if no JSON found."""
    text = text.strip()
    # Try fenced code block first
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try to find a JSON object {...}
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    # Try to find a JSON array [...]
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    # No JSON structure found — signal to caller
    return ""
