"""Custom scorer for multiple-choice answer extraction and matching.

Extracts <answer>...</answer> from the agent's predicted output and
compares against the candidate answers in the data item.

Returns 1.0 if the extracted answer matches any candidate, 0.0 otherwise.

Usage:
    Place this file as custom-scorer.py in the training output directory,
    and set "scorer": "custom" in the data items.
"""

import re

# Pre-compiled regex: match <answer>...</answer> (non-greedy, last occurrence)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _extract_answer(text: str) -> str:
    """Extract the content inside the last <answer>...</answer> tag.

    Returns the stripped inner text, or empty string if not found.
    """
    matches = _ANSWER_RE.findall(text)
    if not matches:
        return ""
    # Use the last match (final answer in case of multiple tags)
    return matches[-1].strip()


def score(sample: dict, predicted: str) -> float:
    """Score a prediction against the sample's candidate answers.

    Args:
        sample: The full data item dict containing 'answers' list.
                Each answer is expected to be in <answer>X</answer> format.
        predicted: The agent's full predicted text (assistant content).

    Returns:
        1.0 if the extracted predicted answer matches any candidate answer,
        0.0 otherwise.
    """
    predicted_answer = _extract_answer(predicted)
    if not predicted_answer:
        return 0.0

    answers = sample.get("answers", [])
    if not answers:
        return 0.0

    # Compare the extracted answer against each candidate
    for ans in answers:
        candidate = _extract_answer(str(ans))
        if not candidate:
            # Fallback: treat the whole answer string as the candidate
            candidate = str(ans).strip()
        if predicted_answer == candidate:
            return 1.0

    return 0.0
