"""Fuzzy find-and-replace for propose_skill patch action.

Matching tiers (tried in order):
1. Exact — unchanged behaviour; confidence 1.0.
2. Indentation-tolerant — strips leading whitespace per line; catches the
   common case where the LLM copied old_string from a context window with
   wrong indentation. Confidence 0.95.
3. SequenceMatcher — line-window similarity for minor typos and extra spaces.
   Refused when best ratio < min_confidence (default 0.85).

All tiers refuse ambiguous matches (>1 location) unless replace_all is True,
in which case only tier 1 (exact) is used.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass


@dataclass
class PatchMatch:
    start: int       # character offset into content (inclusive)
    end: int         # character offset into content (exclusive)
    confidence: float


def _strip_leading_per_line(text: str) -> str:
    return "\n".join(line.lstrip() for line in text.splitlines())


def find_match(
    content: str,
    old_string: str,
    *,
    min_confidence: float = 0.85,
) -> tuple[PatchMatch | None, str | None]:
    """Locate old_string in content using the three-tier strategy.

    Returns ``(PatchMatch, None)`` on success or ``(None, error_message)``
    when no unambiguous match is found above *min_confidence*.
    """
    if not old_string:
        return None, "old_string must not be empty."

    # --- Tier 1: exact ---
    count = content.count(old_string)
    if count == 1:
        idx = content.index(old_string)
        return PatchMatch(idx, idx + len(old_string), 1.0), None
    if count > 1:
        return None, (
            f"old_string matches {count} times (exact). "
            "Provide more surrounding context to make it unique, "
            "or set replace_all=true."
        )

    # --- Tier 2: indentation-tolerant ---
    norm_old = _strip_leading_per_line(old_string)
    if norm_old:
        norm_content = _strip_leading_per_line(content)
        norm_count = norm_content.count(norm_old)
        if norm_count == 1:
            # Map normalized position back to original content via line count.
            norm_start = norm_content.index(norm_old)
            line_offset = norm_content[:norm_start].count("\n")
            n_lines = norm_old.count("\n") + 1

            orig_lines = content.split("\n")
            char_start = sum(len(ln) + 1 for ln in orig_lines[:line_offset])
            char_end = char_start + sum(
                len(ln) + 1 for ln in orig_lines[line_offset : line_offset + n_lines]
            )
            # Don't include the trailing newline if old_string didn't end with one.
            if not old_string.endswith("\n") and char_end > char_start:
                char_end -= 1
            char_end = min(char_end, len(content))
            return PatchMatch(char_start, char_end, 0.95), None
        if norm_count > 1:
            return None, (
                f"old_string matches {norm_count} times (whitespace-normalized). "
                "Provide more surrounding context, or set replace_all=true."
            )

    # --- Tier 3: SequenceMatcher over same-length line windows ---
    old_lines = old_string.splitlines()
    content_lines = content.splitlines(keepends=True)
    n_old = len(old_lines)
    n_content = len(content_lines)

    if n_old == 0 or n_old > n_content:
        return None, (
            "old_string not found in file. "
            "Check for typos or copy the exact text from the file."
        )

    old_joined = "\n".join(old_lines)
    best_ratio = 0.0
    best_idx = 0

    for i in range(n_content - n_old + 1):
        window_joined = "\n".join(
            ln.rstrip("\n") for ln in content_lines[i : i + n_old]
        )
        ratio = difflib.SequenceMatcher(
            None, old_joined, window_joined, autojunk=False
        ).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = i

    if best_ratio < min_confidence:
        return None, (
            f"old_string not found in file "
            f"(best similarity: {best_ratio:.0%}, threshold: {min_confidence:.0%}). "
            "Check for typos or copy the exact text from the file."
        )

    char_start = sum(len(ln) for ln in content_lines[:best_idx])
    char_end = char_start + sum(len(ln) for ln in content_lines[best_idx : best_idx + n_old])
    if not old_string.endswith("\n") and char_end > char_start and content[char_end - 1] == "\n":
        char_end -= 1
    return PatchMatch(char_start, char_end, best_ratio), None


def apply_patch(
    content: str,
    old_string: str,
    new_string: str,
    *,
    replace_all: bool = False,
    min_confidence: float = 0.85,
) -> tuple[str | None, str | None, float]:
    """Apply old_string → new_string with fuzzy matching.

    Returns ``(new_content, None, confidence)`` on success or
    ``(None, error_message, 0.0)`` on failure.
    """
    if replace_all:
        if old_string not in content:
            return None, "old_string not found in file.", 0.0
        return content.replace(old_string, new_string), None, 1.0

    match, err = find_match(content, old_string, min_confidence=min_confidence)
    if err or match is None:
        return None, err or "no match found", 0.0

    new_content = content[: match.start] + new_string + content[match.end :]
    return new_content, None, match.confidence
