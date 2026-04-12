"""Salience scoring heuristics used by the hook to decide when
Reflexion-style self-critique is worth firing.

Reflexion is valuable when the agent just did something non-trivial:
recovered from an error, burned several tool calls on a dead-end, or got
corrected by the user. It is NOT valuable on ordinary turns — nudging the
agent to reflect on "what is 2+2" wastes tokens and trains it to ignore
us.

The score is deliberately conservative. False negatives (missed reflection
opportunities) cost nothing. False positives (spurious nudges) degrade
prompt quality.

Default weights — tuned so one error OR one tool-burst OR three user
corrections is enough to cross the default threshold of 5.0 in the config:

- 3.0 per iteration with an error
- 2.0 per iteration with a tool-call burst (>= 5 calls)
- 1.0 per iteration where the latest user message reads like a correction
"""
from __future__ import annotations

from typing import Any

_TOOL_BURST_MIN = 5
_TOOL_BURST_SCORE = 2.0
_ERROR_SCORE = 3.0
_CORRECTION_SCORE = 1.0

# Keyword matches for "the user is pushing back." Cheap substring check,
# no embedding. Intentionally lenient — better to catch more corrections
# than to be precise.
_CORRECTION_PHRASES: tuple[str, ...] = (
    "no,", "no that", "wrong", "that's not", "that is not",
    "actually", "incorrect", "mistake", "try again",
    "doesn't work", "didn't work", "not quite", "not right",
    "that was wrong",
)


def tool_burst_score(tool_call_count: int) -> float:
    return _TOOL_BURST_SCORE if tool_call_count >= _TOOL_BURST_MIN else 0.0


def error_score(had_error: bool) -> float:
    return _ERROR_SCORE if had_error else 0.0


def correction_score(user_text: str | None) -> float:
    if not user_text:
        return 0.0
    low = user_text.lower()
    return _CORRECTION_SCORE if any(p in low for p in _CORRECTION_PHRASES) else 0.0


def last_user_text(messages: list[dict[str, Any]]) -> str | None:
    """Return the most recent user message's text, flattened from content
    blocks if necessary. Returns ``None`` if no user message is found."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    block_text = block.get("text")
                    if block_text:
                        parts.append(str(block_text))
            return " ".join(parts) if parts else None
    return None
