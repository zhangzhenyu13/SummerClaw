"""MastraOM Observation Groups — message-range-tagged observation containers.

Based on Mastra's observation-groups.ts.
Observation groups wrap observations in <observation-group> XML tags with
message ID ranges, enabling recall/retrieval of source messages.

Key features:
- ObservationGroup dataclass with id, range, content, kind
- Tag wrapping / parsing / stripping
- Range combination for multi-chunk consolidation
- Reflection rendering and provenance reconciliation
- OBSERVATION_RETRIEVAL_INSTRUCTIONS for the recall tool
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Observation Group dataclass
# ---------------------------------------------------------------------------

@dataclass
class ObservationGroup:
    """A single observation group with message-range anchoring.

    Attributes:
        id: Unique anchor ID (hex string).
        range: Message ID range (e.g. "msg_001:msg_050").
        content: The observation text inside the group.
        kind: Optional kind tag (e.g. "reflection").
    """
    id: str
    range: str
    content: str
    kind: str | None = None


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_OBSERVATION_GROUP_PATTERN = re.compile(
    r'<observation-group\s+([^>]*)>\s*([\s\S]*?)\s*</observation-group>',
    re.IGNORECASE,
)
_ATTRIBUTE_PATTERN = re.compile(r'([\w][\w-]*)="([^"]*)"')
_REFLECTION_GROUP_SPLIT_PATTERN = re.compile(r'^##\s+Group\s+', re.MULTILINE)


# ---------------------------------------------------------------------------
# Anchor ID generation
# ---------------------------------------------------------------------------

def generate_anchor_id() -> str:
    """Generate a random 16-character hex anchor ID for an observation group."""
    return secrets.token_hex(8)


# ---------------------------------------------------------------------------
# Parse / wrap / strip
# ---------------------------------------------------------------------------

def _parse_observation_group_attributes(attribute_string: str) -> dict[str, str]:
    """Parse key="value" pairs from an observation-group tag's attribute string."""
    attributes: dict[str, str] = {}
    for match in _ATTRIBUTE_PATTERN.finditer(attribute_string):
        key, value = match.groups()
        if key and value is not None:
            attributes[key] = value
    return attributes


def wrap_in_observation_group(
    observations: str,
    range_spec: str,
    group_id: str | None = None,
    source_group_ids: list[str] | None = None,
    kind: str | None = None,
) -> str:
    """Wrap observation text in an <observation-group> tag with message range.

    Args:
        observations: The observation text to wrap.
        range_spec: Message ID range string (e.g. "msg_001:msg_050").
        group_id: Optional anchor ID. Generated if not provided.
        source_group_ids: Optional list of source group IDs (unused in tag, for metadata).
        kind: Optional kind tag.

    Returns:
        XML string: <observation-group id="..." range="...">\ncontent\n</observation-group>
    """
    content = observations.strip()
    gid = group_id or generate_anchor_id()
    kind_attr = f' kind="{kind}"' if kind else ''
    return f'<observation-group id="{gid}" range="{range_spec}"{kind_attr}>\n{content}\n</observation-group>'


def parse_observation_groups(observations: str) -> list[ObservationGroup]:
    """Parse all <observation-group> tags from observation text.

    Args:
        observations: Raw observation text potentially containing group tags.

    Returns:
        List of parsed ObservationGroup instances.
    """
    if not observations:
        return []

    groups: list[ObservationGroup] = []
    for match in _OBSERVATION_GROUP_PATTERN.finditer(observations):
        attrs = _parse_observation_group_attributes(match.group(1) or '')
        gid = attrs.get('id')
        range_spec = attrs.get('range')
        if not gid or not range_spec:
            continue
        groups.append(ObservationGroup(
            id=gid,
            range=range_spec,
            kind=attrs.get('kind'),
            content=match.group(2).strip() if match.lastindex and match.lastindex >= 2 else '',
        ))
    return groups


def strip_observation_groups(observations: str) -> str:
    """Remove <observation-group> tags, preserving inner content.

    Args:
        observations: Observation text with group tags.

    Returns:
        Cleaned observations without group tags.
    """
    if not observations:
        return observations

    def _replace(_match: re.Match) -> str:
        # group(2) is the content inside the tag
        content = _match.group(2) if _match.lastindex and _match.lastindex >= 2 else ''
        return content.strip()

    result = _OBSERVATION_GROUP_PATTERN.sub(_replace, observations)
    # Collapse multiple blank lines
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


# ---------------------------------------------------------------------------
# Range utilities
# ---------------------------------------------------------------------------

def _get_range_segments(range_spec: str) -> list[str]:
    """Split a comma-separated range string into segments."""
    return [seg.strip() for seg in range_spec.split(',') if seg.strip()]


def combine_observation_group_ranges(groups: list[ObservationGroup]) -> str:
    """Merge range specifications from multiple groups into a single range.

    Takes the start of the first segment and end of the last segment.

    Args:
        groups: Observation groups to merge ranges from.

    Returns:
        Combined range string (e.g. "msg_001:msg_100").
    """
    segments: list[str] = []
    for group in groups:
        segments.extend(_get_range_segments(group.range))

    if not segments:
        return ''

    first_segment = segments[0]
    last_segment = segments[-1]
    first_start = first_segment.split(':')[0].strip() if first_segment else None
    last_end = last_segment.split(':')[-1].strip() if last_segment else None

    if first_start and last_end:
        return f"{first_start}:{last_end}"

    # Fallback: deduplicate and join
    return ','.join(dict.fromkeys(segments))


# ---------------------------------------------------------------------------
# Reflection rendering
# ---------------------------------------------------------------------------

def render_observation_groups_for_reflection(observations: str) -> str | None:
    """Convert <observation-group> tags to Reflector-friendly ## Group headings.

    Replaces each <observation-group> tag with:
        ## Group `id`
        _range: `range`_

    Args:
        observations: Observation text with group tags.

    Returns:
        Reformatted text for Reflector, or None if no groups found.
    """
    groups = parse_observation_groups(observations)
    if not groups:
        return None

    groups_by_content: dict[str, ObservationGroup] = {}
    for g in groups:
        groups_by_content[g.content.strip()] = g

    def _replace_obs_group(match: re.Match) -> str:
        content = match.group(2).strip() if match.lastindex and match.lastindex >= 2 else ''
        group = groups_by_content.get(content)
        if not group:
            return content
        return (
            f"## Group `{group.id}`\n"
            f"_range: `{group.range}`_\n\n"
            f"{group.content}"
        )

    result = _OBSERVATION_GROUP_PATTERN.sub(_replace_obs_group, observations)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


# ---------------------------------------------------------------------------
# Provenance reconciliation (preserve groups through reflection)
# ---------------------------------------------------------------------------

def _parse_reflection_group_sections(content: str) -> list[dict[str, str]]:
    """Parse reflected content into ## Group sections."""
    normalized = content.strip()
    if not normalized or not _REFLECTION_GROUP_SPLIT_PATTERN.search(normalized):
        return []

    sections: list[dict[str, str]] = []
    parts = _REFLECTION_GROUP_SPLIT_PATTERN.split(normalized)
    for section in parts:
        section = section.strip()
        if not section:
            continue
        newline_idx = section.find('\n')
        heading = section[:newline_idx].strip() if newline_idx >= 0 else section
        body = section[newline_idx + 1:].strip() if newline_idx >= 0 else ''
        # Strip _range: metadata line
        body = re.sub(r'^_range:\s*`[^`]*`_\s*\n?', '', body, flags=re.MULTILINE).strip()
        sections.append({'heading': heading, 'body': body})
    return sections


def _get_canonical_group_id(section_heading: str, fallback_index: int) -> str:
    """Extract the canonical group ID from a ## Group heading."""
    match = re.search(r'`([^`]+)`', section_heading)
    return match.group(1).strip() if match else f"derived-group-{fallback_index + 1}"


def derive_observation_group_provenance(
    content: str,
    groups: list[ObservationGroup],
) -> list[ObservationGroup]:
    """Derive observation group provenance from reflected content.

    Matches reflected ## Group sections back to source observation groups.

    Args:
        content: Reflected content with ## Group sections.
        groups: Original source observation groups.

    Returns:
        List of derived ObservationGroup instances preserving provenance.
    """
    sections = _parse_reflection_group_sections(content)
    if not sections or not groups:
        return []

    derived: list[ObservationGroup] = []
    for idx, section in enumerate(sections):
        body_lines = {
            line.strip()
            for line in section['body'].split('\n')
            if line.strip()
        }

        matching_groups = [
            g for g in groups
            if any(
                gl in body_lines
                for gl in (
                    g.content.split('\n')
                )
                if gl.strip()
            )
        ]

        fallback = groups[min(idx, len(groups) - 1)]
        resolved = matching_groups if matching_groups else ([fallback] if fallback else [])
        canonical_id = _get_canonical_group_id(section['heading'], idx)

        derived.append(ObservationGroup(
            id=canonical_id,
            range=combine_observation_group_ranges(resolved),
            kind='reflection',
            content=section['body'],
        ))

    return derived


def reconcile_observation_groups_from_reflection(
    content: str,
    source_observations: str,
) -> str | None:
    """Reconcile observation groups after reflection to preserve provenance.

    Takes reflected content and reconstructs <observation-group> tags so
    the message-range provenance is not lost through the reflection pipeline.

    Args:
        content: Reflected observation text.
        source_observations: Original observations with group tags.

    Returns:
        Reflected text with group tags reconstructed, or None if no groups found.
    """
    source_groups = parse_observation_groups(source_observations)
    if not source_groups:
        return None

    normalized = content.strip()
    if not normalized:
        return ''

    derived = derive_observation_group_provenance(normalized, source_groups)
    if derived:
        parts: list[str] = []
        for group in derived:
            parts.append(wrap_in_observation_group(
                observations=group.content,
                range_spec=group.range,
                group_id=group.id,
                kind=group.kind,
            ))
        return '\n\n'.join(parts)

    # Fallback: wrap entire content in a single group
    return wrap_in_observation_group(
        observations=normalized,
        range_spec=combine_observation_group_ranges(source_groups),
        group_id=generate_anchor_id(),
        kind='reflection',
    )


# ---------------------------------------------------------------------------
# Build message range from message IDs
# ---------------------------------------------------------------------------

def build_message_range(messages: list[dict[str, Any]]) -> str | None:
    """Build a message ID range string from a list of messages.

    Uses the first and last message IDs to construct a range like "msg_001:msg_050".

    Args:
        messages: List of message dicts, each with an 'id' key.

    Returns:
        Range string, or None if no message IDs found.
    """
    ids = [m.get('id') for m in messages if m.get('id')]
    if not ids:
        return None
    if len(ids) == 1:
        return f"{ids[0]}:{ids[0]}"
    return f"{ids[0]}:{ids[-1]}"


# ---------------------------------------------------------------------------
# Retrieval instructions (recall tool)
# ---------------------------------------------------------------------------

OBSERVATION_RETRIEVAL_INSTRUCTIONS = """## Observation Memory

Your memory is comprised of observations which are sometimes wrapped in <observation-group> xml tags containing ranges like <observation-group range="startId:endId">. These ranges indicate which original messages each observation was derived from.

The memory system automatically recalls relevant original logs when needed based on your current task — you do not need to take any action to retrieve them. Simply use the observations as your memory of past interactions, and any additional detail from original logs will be provided automatically when the system detects your task requires it.

### When observations may lack detail
- The user asks you to **repeat, show, or reproduce** something from a past conversation
- The user asks for **exact content** — code, text, quotes, error messages, URLs, file paths, specific numbers
- Your observations mention something but you lack the detail needed to fully answer

In these cases, the memory system will automatically inject the original logs into your context. If the detail is still insufficient, acknowledge that your memory has the gist but may not have the exact original content."""
