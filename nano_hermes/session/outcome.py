"""Session outcome labels: ``ok | partial | fail | unknown``.

Precision first (see ROADMAP "outcome labels"): every learning loop — lesson
trust, MemRL utility, principle attribution, ExpeL, the skill designer — reads
this label, and self-graded or silence-as-success labels get amplified by
them. So a session is ``ok`` only on positive evidence (the user's reaction,
or a completed goal), ``fail``/``partial`` on hard run failures or user
corrections, and ``unknown`` otherwise. Tool errors alone never decide it:
a missed ``read_file`` during exploration is routine.

Signals come from a recent window — the last few user reactions and the last
run — not session totals, because a chat session can span weeks of tasks.
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from ..lru import touch
from .archiver import _MESSAGE_TIME_PREFIX, _RUNTIME_SENTINEL, _extract_text

OUTCOMES = ("ok", "partial", "fail", "unknown")

_WINDOW = 3
# Stop reasons that mean the run itself failed (nanobot runner).
_FAIL_STOPS = frozenset({"error", "tool_error", "empty_final_response"})
# nanobot's goal-continuation turn (loop.py; same prefix on 0.2.2..HEAD).
_GOAL_CONTINUE = "You have an active sustained goal:"

# Strict lists for labelling — the lenient salience list (reflect/salience.py)
# stays for Reflexion nudges. Matched at the start of the user's message, on
# word boundaries. A leading "no"/"nope" is allowed but never decides: it is
# as often the answer to the assistant's own question ("no, just commit it",
# "no, looks good") as pushback, so what follows it does.
_LEAD = r"^\W*(?:(?:no|nope)\b[,.!]?\s*)?"
_CORRECTION_RE = re.compile(
    _LEAD
    + r"(?:wrong\b|that'?s\s+(?:not|wrong)\b|that\s+is\s+(?:not|wrong)\b|"
    r"not\s+(?:quite|right|what)\b|"
    r"(?:it|this|that)\s+(?:still\s+)?(?:doesn'?t|didn'?t|does\s+not|did\s+not)\s+work\b|"
    r"(?:still|it'?s\s+still)\s+(?:broken|failing|wrong|not\s+working)\b|try\s+again\b)",
    re.IGNORECASE,
)
# Pushback anywhere in a message that opened positively ("great, but …").
_BUT_RE = re.compile(r"\b(?:but|however|except)\b", re.IGNORECASE)
_BROKEN_RE = re.compile(
    r"\b(?:broken|fails?|failing|failed|wrong|doesn'?t\s+work|didn'?t\s+work|"
    r"not\s+working|not\s+right)\b",
    re.IGNORECASE,
)
_POSITIVE_RE = re.compile(
    _LEAD
    + r"(?:(?:that'?s|it'?s)\s+(?:perfect|great|exactly\s+it|right)|"
    r"(?<=no\s)problem\b|(?<=no\s)worries\b|thanks?\b|thank\s+you\b|thx\b|ty\b|perfect\b|"
    r"great\b|awesome\b|nice\b|all\s+good\b|"
    r"(?:that|it|this)\s+(?:works|worked(?!\s+before)|did\s+it)\b|works\s+now\b|"
    r"looks\s+good\b|lgtm\b)",
    re.IGNORECASE,
)


def trailing_user_texts(messages: list[dict[str, Any]]) -> list[str]:
    """Texts of the window's trailing run of user messages (this turn's new
    ones), skipping this hook's own system injections at the end."""
    out: list[str] = []
    for m in reversed(messages):
        role = m.get("role")
        if role == "system":
            continue
        if role != "user":
            break
        out.append(_extract_text(m))
    return out[::-1]


def carries_history(messages: list[dict[str, Any]]) -> bool:
    """Whether the window holds earlier conversation, not just this turn.

    After compaction the previous exchange survives (as history or merged
    onto this turn's user message under a ``[Message Time:`` stamp); after
    ``/new`` the window is the system prompt and the new task alone, so its
    opener is no reaction to the ended session.
    """
    convo = [m for m in messages if m.get("role") != "system"]
    if len(convo) > len(trailing_user_texts(messages)):
        return True
    return bool(convo) and _extract_text(convo[0]).startswith(_MESSAGE_TIME_PREFIX)


def _current_text(text: str) -> str:
    """The user's own words: no runtime-context block, no merged history.

    A message nanobot merged onto the previous user entry (post-compaction
    summary, interrupted turn) starts with the ``[Message Time:`` stamp and
    ends with the current message after a blank line.
    """
    cut = text.find("\n" + _RUNTIME_SENTINEL)
    if cut >= 0:
        text = text[:cut]
    # ponytail: the history half (often a compaction summary) has blank lines
    # too, so the split point is unknowable; the last paragraph is the best
    # guess and loses a multi-paragraph reply's opening. Fine for reactions,
    # which are short.
    if text.startswith(_MESSAGE_TIME_PREFIX) and "\n\n" in text:
        text = text.rsplit("\n\n", 1)[1]
    return text.strip()


def classify_user_text(text: str | None) -> str | None:
    """``"correction"``, ``"positive"``, ``"neutral"``, or None (not a reaction)."""
    if not text:
        return None
    text = _current_text(text)
    if not text or text.startswith(_GOAL_CONTINUE):
        return None
    if _CORRECTION_RE.search(text):
        return "correction"
    if _POSITIVE_RE.search(text):
        # "thanks, but it's still broken" is pushback; "great, but add X" is
        # a new request, not a verdict.
        if _BROKEN_RE.search(text):
            return "correction"
        return "neutral" if _BUT_RE.search(text) else "positive"
    return "neutral"


@dataclass
class OutcomeSignals:
    reactions: deque = field(default_factory=lambda: deque(maxlen=_WINDOW))
    last_stop: str | None = None
    goal: str | None = None          # last update_goal action: complete|cancel|block|replace
    goal_active: bool = False
    llm_error: bool = False
    rated_failure: bool = False
    tool_calls: int = 0
    tool_errors: int = 0
    user_messages: int = 0           # real user messages seen this session

    def as_dict(self) -> dict[str, Any]:
        return {
            "reactions": list(self.reactions),
            "last_stop": self.last_stop,
            "goal": self.goal,
            "llm_error": self.llm_error,
            "rated_failure": self.rated_failure,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "user_messages": self.user_messages,
        }


def is_hard_failure(sig: OutcomeSignals) -> bool:
    """A failure no judge should overrule."""
    stop_fail = sig.last_stop in _FAIL_STOPS or (
        sig.last_stop == "max_iterations" and not sig.goal_active
    )
    return stop_fail or sig.llm_error or sig.goal == "block"


def label(sig: OutcomeSignals) -> str:
    """Rule label for one session (see module docstring)."""
    if is_hard_failure(sig):
        return "fail"
    window = list(sig.reactions)
    last_positive = max((i for i, r in enumerate(window) if r == "positive"), default=-1)
    corrections = sum(1 for r in window[last_positive + 1:] if r == "correction")
    if corrections >= 2:
        return "fail"
    if corrections == 1 or (sig.rated_failure and last_positive < 0):
        return "partial"
    if window and window[-1] == "positive":
        return "ok"
    if sig.goal == "complete" and "correction" not in window:
        return "ok"  # the agent's own claim, accepted only without pushback
    return "unknown"


class OutcomeTracker:
    """Per-conversation :class:`OutcomeSignals`, taken at the session boundary."""

    def __init__(self) -> None:
        self._states: dict[str, OutcomeSignals] = {}

    def _state(self, key: str) -> OutcomeSignals:
        touch(self._states, key, OutcomeSignals)
        return self._states[key]

    @staticmethod
    def labelled(key: str) -> bool:
        """Cron and heartbeat sessions have no user to react."""
        return not (key == "heartbeat" or key.startswith("cron:"))

    def record_user_messages(self, key: str, texts: list[str]) -> None:
        """Score new real user messages; the session's first one reacts to nothing."""
        if not self.labelled(key):
            return
        st = self._state(key)
        for text in texts:
            verdict = classify_user_text(text)
            if verdict is None:
                continue
            st.user_messages += 1
            if st.user_messages > 1:
                st.reactions.append(verdict)

    def credit_followup(self, key: str, texts: list[str]) -> None:
        """Credit the next session's opening message to the one just ended.

        The user's reaction to a session's last reply usually opens the next
        session (compaction splits them); only a correction or a positive
        counts, so a new task's neutral opener changes nothing.
        """
        if not self.labelled(key):
            return
        st = self._state(key)
        for text in texts:
            verdict = classify_user_text(text)
            if verdict in ("correction", "positive"):
                st.reactions.append(verdict)

    def record_run(self, key: str, stop_reason: str | None, tool_events: list[dict]) -> None:
        st = self._state(key)
        if stop_reason:
            st.last_stop = stop_reason
            if stop_reason == "completed":
                st.llm_error = False  # a later run recovered: label the last run
        st.tool_calls += len(tool_events)
        st.tool_errors += sum(1 for ev in tool_events if ev.get("status") == "error")

    def record_goal(self, key: str, action: str) -> None:
        st = self._state(key)
        st.goal = action
        st.goal_active = action not in ("complete", "cancel", "block")

    def set_goal_active(self, key: str, active: bool) -> None:
        self._state(key).goal_active = active

    def record_llm_error(self, key: str) -> None:
        self._state(key).llm_error = True

    def record_rated_failure(self, key: str) -> None:
        self._state(key).rated_failure = True

    def take(self, key: str) -> OutcomeSignals:
        """Return *key*'s signals and reset them for the next session.

        An active goal carries over: it spans sessions until updated.
        """
        st = self._state(key)
        self._states[key] = OutcomeSignals(goal_active=st.goal_active)
        return st
