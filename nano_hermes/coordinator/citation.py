"""Citation-overlap heuristic for RMM (Reflective Memory Management).

Given a candidate injection's content and an assistant response, decide
whether the response "cited" the injection. There is no explicit citation
marker, so we approximate with a token-overlap score on tokens long
enough (>=4 chars) to be meaningful.

The threshold is intentionally conservative — false positives waste a
"good retrieval" signal, false negatives just slow learning down. We
prefer the latter.
"""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]{4,}")
_DEFAULT_THRESHOLD = 0.30


def _significant_tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def cite_score(injection_content: str, response_text: str) -> float:
    """Return fraction of injection tokens that appear in the response (0-1).

    Tokens are ASCII-lower alphanumerics of length ≥ 4. Returns 0 when
    either side has no significant tokens.
    """
    inj_tokens = _significant_tokens(injection_content)
    if not inj_tokens:
        return 0.0
    resp_tokens = _significant_tokens(response_text)
    if not resp_tokens:
        return 0.0
    overlap = inj_tokens & resp_tokens
    return len(overlap) / len(inj_tokens)


def is_cited(
    injection_content: str,
    response_text: str,
    threshold: float = _DEFAULT_THRESHOLD,
) -> bool:
    return cite_score(injection_content, response_text) >= threshold
