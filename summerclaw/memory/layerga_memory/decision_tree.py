"""L0 decision tree — programmable classification engine for memory information.

Translates the L0 constitution's decision tree into executable Python rules.
Used by both Consolidator (online) and Dream (offline) to classify information
into the correct memory layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MemoryLayer(Enum):
    """Target memory layer for classified information."""

    L1_RULES = "l1_rules"       # Cross-task pitfall rule (≤1 compressed sentence)
    L1_HIGH = "l1_high"         # L1 Tier 1: high-frequency scenario key→value
    L1_LOW = "l1_low"           # L1 Tier 2: low-frequency scenario keyword only
    L2 = "l2"                   # Environment-specific fact
    L3_SOP = "l3_sop"           # Task SOP (.md file)
    L3_SCRIPT = "l3_script"     # Utility script (.py file)
    DROP = "drop"               # Discard: common knowledge or volatile state


@dataclass
class ClassificationResult:
    """Result of classifying a piece of information through the L0 decision tree."""

    layer: MemoryLayer
    confidence: float          # 0.0–1.0
    reason: str                # Human-readable classification rationale
    trigger_words: list[str] = field(default_factory=list)  # L1 index trigger words
    content_snippet: str = ""  # Minimum-sufficient content to write


@dataclass
class VerifiedFact:
    """An action-verified fact extracted from tool-call results."""

    source_tool: str           # Tool name that produced this fact
    source_args: dict[str, Any] = field(default_factory=dict)
    content: str = ""          # The fact content
    is_verified: bool = True   # Whether the source tool returned success


# Patterns that indicate volatile/temporary information
_VOLATILE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}\b"),  # ISO timestamps
    re.compile(r"\b(?:pid|process\s*id)\s*[:=]?\s*\d+\b", re.IGNORECASE),
    re.compile(r"\b(?:session|token|jwt)\s*[:=]?\s*[a-zA-Z0-9\-_.]+\b", re.IGNORECASE),
    re.compile(r"\b(?:temp|tmp)[\\/]", re.IGNORECASE),
    re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+\b"),  # ephemeral IP:port
]

# Patterns indicating environment-specific facts (LLM cannot infer)
_ENV_FACT_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(?:api[_\s]?key|api[_\s]?secret|access[_\s]?token)\b", re.IGNORECASE),
    re.compile(r"\b(?:proxy[_\s]?port|proxy[_\s]?url|proxy[_\s]?host)\b", re.IGNORECASE),
    re.compile(r"\b(?:endpoint|base[_\s]?url|hostname)\s*[:=]", re.IGNORECASE),
    re.compile(r"\b(?:directory|folder|path)\s*[:=]\s*[/\\]", re.IGNORECASE),
    re.compile(r"\b(?:config|configuration)\s*file", re.IGNORECASE),
    re.compile(r"\b(?:user[_\s]?id|account[_\s]?id|client[_\s]?id)\s*[:=]", re.IGNORECASE),
    re.compile(r"\b(?:port|listen)\s*[:=]?\s*\d{2,5}\b", re.IGNORECASE),
]

# Patterns indicating universal pitfalls/rules
_RULE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(?:never|always|must|must\s*not|禁止|必须|绝不)\b", re.IGNORECASE),
    re.compile(r"\b(?:warning|caution|注意|小心|⚠)\b", re.IGNORECASE),
    re.compile(r"\b(?:会?导致.*崩溃|会?导致.*失败|会?导致.*错误)\b"),
    re.compile(r"\b(?:不要|不能|严禁|avoid)\s+\w", re.IGNORECASE),
]

# Patterns indicating task-specific hard-won experience
_TASK_TECH_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(?:trick|hack|workaround|特殊|特殊处理)\b", re.IGNORECASE),
    re.compile(r"\b(?:hidden|隐蔽|不容易发现|不容易找到)\b", re.IGNORECASE),
    re.compile(r"\b(?:retry|重试|多次尝试|花.*时间)\b", re.IGNORECASE),
    re.compile(r"\b(?:specific|特定|专门|针对)\b", re.IGNORECASE),
]

# Patterns indicating common knowledge (should be dropped)
_COMMON_KNOWLEDGE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(?:hello|hi|thanks|thank you|好的|谢谢)\b", re.IGNORECASE),
    re.compile(r"\b(?:ok|okay|fine|没问题)\b", re.IGNORECASE),
    re.compile(r"\b(?:standard|标准|常规|普通)\s+(?:way|method|方法|步骤)\b", re.IGNORECASE),
]


class L0DecisionTree:
    """Programmable implementation of the L0 classification decision tree.

    Follows the decision tree from ``constitution.md``:

    1. Is it an environment-specific fact? → L2 (+ L1 frequency sync)
    2. Is it a universal operating rule? → L1 [RULES]
    3. Is it a task-specific technique? → L3 (SOP or script)
    4. Otherwise → Discard
    """

    # Token cost per line for ROI calculation
    _AVG_TOKENS_PER_LINE = 20

    def __init__(
        self,
        constitution: str = "",
        l1_max_lines: int = 30,
        confidence_threshold: float = 0.5,
    ):
        self.constitution = constitution
        self.l1_max_lines = l1_max_lines
        self.confidence_threshold = confidence_threshold

    # ------------------------------------------------------------------
    # Main classification entry point
    # ------------------------------------------------------------------

    def classify(
        self,
        fact: VerifiedFact,
        current_l1: str = "",
        current_l2_sections: list[str] | None = None,
    ) -> ClassificationResult:
        """Classify a verified fact into the correct memory layer.

        Args:
            fact: The action-verified fact to classify.
            current_l1: Current L1 insight content (for dedup / frequency check).
            current_l2_sections: Current L2 section names (for dedup).

        Returns:
            A ClassificationResult indicating the target layer and rationale.
        """
        if current_l2_sections is None:
            current_l2_sections = []

        # Step 0: Axiom checks — fail early if any axiom is violated
        passed, violations = self.check_axioms(fact)
        if not passed:
            return ClassificationResult(
                layer=MemoryLayer.DROP,
                confidence=1.0,
                reason=f"Axiom violation: {'; '.join(violations)}",
            )

        content = fact.content

        # Step 1: Environment-specific fact?
        if self._is_env_fact(content):
            freq = self._assess_frequency(content, current_l1)
            layer = MemoryLayer.L1_HIGH if freq == "high" else MemoryLayer.L2
            trigger = self._extract_trigger_words(content, max_words=3)
            # For L1_HIGH, return L1_HIGH layer; the caller will:
            #   1. Write to L2 first
            #   2. Sync trigger words to L1 Tier 1
            return ClassificationResult(
                layer=MemoryLayer.L2,  # Primary target is L2
                confidence=0.85,
                reason="Environment-specific fact that LLM cannot infer zero-shot",
                trigger_words=trigger,
                content_snippet=self._minimize_content(content),
            )

        # Step 2: Universal operating rule?
        if self._is_universal_rule(content):
            compressed = self._compress_to_one_sentence(content)
            return ClassificationResult(
                layer=MemoryLayer.L1_RULES,
                confidence=0.8,
                reason="Cross-task pitfall rule or universal operating guideline",
                content_snippet=compressed,
            )

        # Step 3: Task-specific technique?
        if self._is_task_technique(content):
            is_script = self._is_script_worthy(content)
            return ClassificationResult(
                layer=MemoryLayer.L3_SCRIPT if is_script else MemoryLayer.L3_SOP,
                confidence=0.7,
                reason="Task-specific technique learned through hard-won experience",
                trigger_words=self._extract_trigger_words(content, max_words=2),
                content_snippet=self._minimize_content(content),
            )

        # Step 4: Discard (common knowledge, volatile, redundant)
        return ClassificationResult(
            layer=MemoryLayer.DROP,
            confidence=0.9,
            reason="Common knowledge, volatile state, or redundant information",
        )

    # ------------------------------------------------------------------
    # Axiom checks
    # ------------------------------------------------------------------

    def check_axioms(self, fact: VerifiedFact) -> tuple[bool, list[str]]:
        """Check all four L0 axioms. Returns (passed, violations_list)."""
        violations: list[str] = []

        # Axiom 1: Action-Verified
        if not fact.is_verified:
            violations.append(
                "Axiom 1 (No Execution, No Memory): "
                "fact is not action-verified"
            )

        # Axiom 3: No Volatile State
        if self._contains_volatile_state(fact.content):
            violations.append(
                "Axiom 3 (No Volatile State): "
                "content contains ephemeral data (timestamp, PID, session ID, etc.)"
            )

        # Axiom 4: Minimum Sufficient — warn if too verbose (>500 chars)
        if len(fact.content) > 500:
            violations.append(
                "Axiom 4 (Minimum Sufficient): "
                "content exceeds 500 chars — consider compressing"
            )

        return len(violations) == 0, violations

    # ------------------------------------------------------------------
    # Classification heuristics
    # ------------------------------------------------------------------

    def _is_env_fact(self, content: str) -> bool:
        """Check if content describes an environment-specific fact."""
        if self._contains_common_knowledge(content):
            return False
        for pattern in _ENV_FACT_PATTERNS:
            if pattern.search(content):
                return True
        return False

    def _is_universal_rule(self, content: str) -> bool:
        """Check if content describes a universal operating rule."""
        score = 0
        for pattern in _RULE_PATTERNS:
            if pattern.search(content):
                score += 1
        return score >= 2

    def _is_task_technique(self, content: str) -> bool:
        """Check if content describes a task-specific technique."""
        score = 0
        for pattern in _TASK_TECH_PATTERNS:
            if pattern.search(content):
                score += 1
        return score >= 1

    def _is_script_worthy(self, content: str) -> bool:
        """Check if the task technique warrants a .py script instead of a .md SOP."""
        script_indicators = [
            r"\b(?:code|script|function|class|import|def\s)\b",
            r"\b(?:algorithm|logic|loop|iterate)\b",
            r"```(?:python|py|javascript|js|bash|sh)",
        ]
        for pattern in script_indicators:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        return False

    def _contains_volatile_state(self, content: str) -> bool:
        """Check if content contains volatile/temporary information."""
        for pattern in _VOLATILE_PATTERNS:
            if pattern.search(content):
                return True
        return False

    def _contains_common_knowledge(self, content: str) -> bool:
        """Check if content is common knowledge (should not be stored)."""
        for pattern in _COMMON_KNOWLEDGE_PATTERNS:
            if pattern.search(content):
                return True
        return False

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _assess_frequency(self, content: str, current_l1: str) -> str:
        """Assess whether a fact is likely high-frequency or low-frequency.

        Returns 'high' or 'low'.
        """
        # If the fact's trigger words already appear in L1, it's high-frequency
        triggers = self._extract_trigger_words(content, max_words=2)
        for tw in triggers:
            if tw.lower() in current_l1.lower():
                return "high"
        # Heuristic: short, domain-specific facts are often high-frequency
        if len(content) < 100 and len(_ENV_FACT_PATTERNS) > 0:
            return "high"
        return "low"

    def _extract_trigger_words(
        self, content: str, max_words: int = 3
    ) -> list[str]:
        """Extract minimal trigger words from content for L1 index.

        Prioritizes nouns, identifiers, and domain terms.
        """
        # Simple extraction: find capitalized words, identifiers, names
        words = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", content)
        if not words:
            # Fall back to key noun phrases
            words = re.findall(
                r"\b(?:config|path|port|key|token|url|api|endpoint|"
                r"proxy|database|server|client|file|dir|user|account)\b",
                content,
                re.IGNORECASE,
            )
        # Deduplicate while preserving order
        seen: set[str] = set()
        result: list[str] = []
        for w in words:
            w_lower = w.lower()
            if w_lower not in seen:
                seen.add(w_lower)
                result.append(w)
        return result[:max_words]

    def _compress_to_one_sentence(self, content: str) -> str:
        """Compress a universal rule into one sentence for L1 RULES."""
        # Take first sentence or first 120 chars
        sentences = re.split(r"[.。!！?\n]", content)
        for s in sentences:
            s = s.strip()
            if len(s) > 10:
                return s[:120]
        return content.strip()[:120]

    def _minimize_content(self, content: str) -> str:
        """Produce the minimum-sufficient version of content for storage."""
        # Strip leading/trailing whitespace and excessive newlines
        minimized = re.sub(r"\n{3,}", "\n\n", content.strip())
        return minimized

    # ------------------------------------------------------------------
    # L1 maintenance helpers
    # ------------------------------------------------------------------

    def compute_l1_roi(
        self,
        trigger_line: str,
        mistake_probability: float,
        mistake_cost_tokens: int,
    ) -> float:
        """Compute ROI for a single L1 line.

        ROI = (mistake_probability × mistake_cost_tokens) / avg_tokens_per_line

        Args:
            trigger_line: The L1 line content.
            mistake_probability: Estimated probability of mistake without this line.
            mistake_cost_tokens: Estimated token cost of the resulting mistake.

        Returns:
            ROI value — higher means more worth keeping.
        """
        token_cost = max(len(trigger_line) // 4, self._AVG_TOKENS_PER_LINE)
        if token_cost <= 0:
            return 0.0
        return (mistake_probability * mistake_cost_tokens) / token_cost

    def should_keep_l1_line(
        self,
        line: str,
        mistake_probability: float = 0.1,
        mistake_cost_tokens: int = 200,
        min_roi: float = 0.5,
    ) -> bool:
        """Decide whether an L1 line is worth keeping.

        Args:
            line: The L1 line to evaluate.
            mistake_probability: Estimated probability of mistake (default 0.1).
            mistake_cost_tokens: Estimated token cost of mistake (default 200).
            min_roi: Minimum ROI threshold (default 0.5).

        Returns:
            True if the line should be kept.
        """
        roi = self.compute_l1_roi(line, mistake_probability, mistake_cost_tokens)
        return roi >= min_roi
