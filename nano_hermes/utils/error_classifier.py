"""Typed error classification for evolution LLM calls.

Provides a small taxonomy of failure reasons relevant to nano-hermes's
embedding and evolution callers (GEPA, rewriter). Used to distinguish
billing exhaustion (abort cycle) from transient failures (skip skill).
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class FailoverReason(enum.Enum):
    auth = "auth"                         # 401/403 — bad or expired credentials
    billing = "billing"                   # 402 / quota exhausted — abort callers
    rate_limit = "rate_limit"             # 429 — backoff, don't abort
    overloaded = "overloaded"             # 503/529 — provider overloaded
    server_error = "server_error"         # 500/502 — internal server error
    timeout = "timeout"                   # network timeout
    context_overflow = "context_overflow" # context too large — skip this skill
    unknown = "unknown"                   # unclassified transient


@dataclass
class ClassifiedError:
    reason: FailoverReason
    status_code: int | None = None
    message: str = ""
    error_context: dict[str, Any] = field(default_factory=dict)

    @property
    def should_abort(self) -> bool:
        """True when retrying makes no sense and the caller should stop entirely."""
        return self.reason in (FailoverReason.billing, FailoverReason.auth)

    @property
    def should_backoff(self) -> bool:
        return self.reason in (FailoverReason.rate_limit, FailoverReason.overloaded)

    @property
    def should_skip_skill(self) -> bool:
        return self.reason == FailoverReason.context_overflow


class EvolutionAbortError(Exception):
    """Raised by GEPA/rewriter when an abort-class error is detected.

    Propagates through per-stage exception handlers in _run_evolution_cycle
    so the entire cycle can be cut short (e.g. billing exhausted).
    """
    def __init__(self, classified: ClassifiedError) -> None:
        self.classified = classified
        super().__init__(f"evolution aborted: {classified.reason.value} — {classified.message}")


# ---- LLMResponse classifier ------------------------------------------------

_BILLING_TOKENS = frozenset({
    "billing", "payment_required", "insufficient_quota",
    "billing_hard_limit", "billing_not_active", "credit",
    "insufficient_credits", "out of credits",
})

_CONTEXT_TOKENS = frozenset({
    "context_length_exceeded", "context window", "too many tokens",
    "maximum context", "context limit", "input too long",
})


def classify_llm_response(resp: Any) -> ClassifiedError | None:
    """Inspect a nanobot LLMResponse. Returns None on success.

    Checks ``finish_reason``, ``error_status_code``, ``error_type``,
    ``error_code``, and ``content`` in priority order.
    """
    if resp is None:
        return ClassifiedError(reason=FailoverReason.unknown, message="null response")

    finish = getattr(resp, "finish_reason", "stop") or "stop"
    if finish != "error":
        return None  # success

    status = getattr(resp, "error_status_code", None)
    error_type = (getattr(resp, "error_type", None) or "").lower()
    error_code = (getattr(resp, "error_code", None) or "").lower()
    content = (getattr(resp, "content", None) or "").lower()
    combined = f"{error_type} {error_code} {content}"

    # Status-code takes priority over text signals.
    if status == 401 or status == 403:
        return ClassifiedError(reason=FailoverReason.auth, status_code=status,
                               message=content[:200])
    if status == 402 or any(t in combined for t in _BILLING_TOKENS):
        return ClassifiedError(reason=FailoverReason.billing, status_code=status,
                               message=content[:200])
    if status == 429 or "rate_limit" in combined or "too many requests" in content:
        return ClassifiedError(reason=FailoverReason.rate_limit, status_code=status,
                               message=content[:200])
    if status in (503, 529) or "overload" in content or "overloaded" in content:
        return ClassifiedError(reason=FailoverReason.overloaded, status_code=status,
                               message=content[:200])
    if status in (500, 502):
        return ClassifiedError(reason=FailoverReason.server_error, status_code=status,
                               message=content[:200])
    if any(t in combined for t in _CONTEXT_TOKENS):
        return ClassifiedError(reason=FailoverReason.context_overflow,
                               message=content[:200])

    # error kind == timeout
    error_kind = (getattr(resp, "error_kind", None) or "").lower()
    if "timeout" in error_kind or "timeout" in content:
        return ClassifiedError(reason=FailoverReason.timeout, message=content[:200])

    return ClassifiedError(reason=FailoverReason.unknown, status_code=status,
                           message=content[:200])


def classify_http_status(status: int, message: str = "") -> ClassifiedError:
    """Classify a bare HTTP error status code (for embedding chain use)."""
    if status in (401, 403):
        return ClassifiedError(reason=FailoverReason.auth, status_code=status,
                               message=message)
    if status == 402:
        return ClassifiedError(reason=FailoverReason.billing, status_code=status,
                               message=message)
    if status == 429:
        return ClassifiedError(reason=FailoverReason.rate_limit, status_code=status,
                               message=message)
    if status in (503, 529):
        return ClassifiedError(reason=FailoverReason.overloaded, status_code=status,
                               message=message)
    if status >= 500:
        return ClassifiedError(reason=FailoverReason.server_error, status_code=status,
                               message=message)
    return ClassifiedError(reason=FailoverReason.unknown, status_code=status,
                           message=message)
