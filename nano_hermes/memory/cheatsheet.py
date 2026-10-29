"""Dynamic Cheatsheet — per-session transferable lesson extraction (arXiv:2504.07952).

After each session completion:
  1. One LLM call extracts ONE transferable lesson from the session.
  2. Stored in ``semantic_facts`` with ``fact_type='cheatsheet'`` and
     ``task_category`` = the first 120 chars of the first user message.

At inference (``before_iteration``, iteration 0 only):
  1. KNN search finds the top-k most relevant cheatsheet lessons.
  2. A synthetic system message prefixes them into context.

Default off (``cheatsheet_enabled=False`` in ``SkillStatsConfig``).
One LLM call per completed session, zero per-turn overhead when retrieval
produces no results.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from typing import TYPE_CHECKING, Any, Callable

from ..utils.error_classifier import classify_llm_response

if TYPE_CHECKING:
    from ..hook import NanoHermesHook

log = logging.getLogger(__name__)

_EXTRACT_PROMPT = """\
Given this completed session, extract ONE concise transferable lesson \
(max 2 sentences) that would help on similar future tasks, AND a one-line \
condition describing WHEN the lesson applies. Focus on what worked or what to avoid.
A useful lesson states the mechanism (why it went wrong or right), names a \
concrete action (not "be careful"), and names any action to never take.

Task: {task}
Outcome: {outcome}
Key events: {events}

Output exactly three lines:
LESSON: <the lesson>
WHEN: <one-line situation where this lesson applies>
EVIDENCE: <the specific event above that the lesson rests on>
If you cannot name such an event, the lesson is speculation: output exactly SKIP."""


_NO_EVIDENCE = {"", "none", "n/a", "na", "unknown", "nothing", "no evidence"}


def _parse_marked(raw: str, head: str) -> tuple[str, str, str]:
    """Parse a ``HEAD: …`` / ``WHEN: …`` / ``EVIDENCE: …`` response into
    (body, condition, evidence).

    Each section runs until the next marker, so multi-line values survive.
    When the model omits the HEAD marker, the unmarked text before the first
    marker is the body — never the WHEN/EVIDENCE lines. Placeholder evidence
    ("none", "N/A", …) comes back as "".
    """
    parts: dict[str, list[str]] = {"pre": [], "body": [], "when": [], "evidence": []}
    section = "pre"
    markers = ((head + ":", "body"), ("WHEN:", "when"), ("EVIDENCE:", "evidence"))
    for line in raw.splitlines():
        s = line.strip()
        up = s.upper()
        for marker, name in markers:
            if up.startswith(marker):
                section = name
                s = s[len(marker):].strip()
                break
        parts[section].append(s)

    def join(name: str) -> str:
        return " ".join(p for p in parts[name] if p).strip()

    body = join("body") or join("pre")
    evidence = join("evidence")
    if evidence.strip(" .-").lower() in _NO_EVIDENCE:
        evidence = ""
    return body, join("when"), evidence


def _parse_lesson(raw: str) -> tuple[str, str, str]:
    """Parse the LESSON/WHEN/EVIDENCE response. Tolerates a missing LESSON marker."""
    return _parse_marked(raw, "LESSON")


def _first_user_text(messages: list[dict[str, Any]]) -> str:
    for m in messages:
        if m.get("role") == "user":
            text = m.get("content", "")
            if isinstance(text, list):
                text = " ".join(
                    b.get("text", "") for b in text if isinstance(b, dict)
                )
            return str(text)[:400]
    return ""


def _last_assistant_text(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "assistant":
            text = m.get("content", "")
            if isinstance(text, list):
                text = " ".join(
                    b.get("text", "") for b in text if isinstance(b, dict)
                )
            return str(text)[:400]
    return ""


def _task_category(first_user: str) -> str:
    return first_user[:120].strip()


async def extract_cheatsheet_lesson(
    hook: "NanoHermesHook",
    messages: list[dict[str, Any]],
    outcome: str,
) -> None:
    """Extract and store one transferable lesson for the completed session.

    Silently skips on LLM error, SKIP responses, or uninformative sessions.
    Must be called with a running event loop (called from ``after_iteration``).
    """
    if outcome == "unknown":
        return

    first_user = _first_user_text(messages)
    if not first_user:
        return
    last_assistant = _last_assistant_text(messages)

    prompt = _EXTRACT_PROMPT.format(
        task=first_user,
        outcome=outcome,
        events=last_assistant,
    )
    try:
        resp = await hook._loop.provider.chat_with_retry(
            messages=[{"role": "user", "content": prompt}],
            model=hook.background_model,
            max_tokens=300,  # three lines; a cut-off EVIDENCE line drops the lesson
        )
    except Exception:
        log.debug("cheatsheet: LLM call failed — skipping", exc_info=True)
        return

    err = classify_llm_response(resp)
    if err is not None:
        log.debug("cheatsheet: LLM error %s — skipping", err.reason.value)
        return

    raw = (resp.content or "").strip()
    if not raw or raw.upper().startswith("SKIP"):
        return
    lesson, condition, evidence = _parse_lesson(raw)
    if not lesson or len(lesson) < 20:
        return
    if not evidence:
        log.debug("cheatsheet: lesson cites no evidence — skipping")
        return

    category = _task_category(first_user)
    # The applicability condition is what retrieval matches the current task
    # against; fall back to the task category when the model omits it.
    applies_when = condition or category
    await store_if_novel(
        hook, applies_when, lambda: _store_lesson(hook.db, lesson, category, applies_when)
    )


async def _embed_bytes(hook: "NanoHermesHook", text: str) -> bytes | None:
    try:
        import numpy as np  # noqa: PLC0415

        async with hook.embedder() as chain:
            [vec] = await chain.embed([text])
        return None if vec is None else np.asarray(vec, dtype=np.float32).tobytes()
    except Exception:
        log.debug("lesson: embedding failed", exc_info=True)
        return None


async def store_if_novel(
    hook: "NanoHermesHook", applies_when: str, insert: Callable[[], int]
) -> int | None:
    """Store a lesson only if it is indexable and new; return its id or None.

    Selective addition beats storing every experience (arXiv:2505.16067), so a
    lesson whose condition nearly matches a live lesson's is dropped. With no
    vector the lesson is dropped too: retrieval is KNN-only, so it could never
    be injected.
    """
    vec_bytes = await _embed_bytes(hook, applies_when)
    if vec_bytes is None:
        return None
    db = hook.db
    dup = db.execute(
        "SELECT 1 FROM semantic_facts_vec sfv "
        "JOIN semantic_facts sf ON sf.id = sfv.fact_id "
        "WHERE sfv.embedding MATCH ? AND k = 64 "
        "  AND sf.fact_type IN ('cheatsheet', 'expel') AND sf.invalid_at IS NULL "
        # A distrusted lesson is hidden from retrieval; it must not also block
        # the corrective lesson that would replace it.
        "  AND sf.trust_score >= ? AND 1.0 - sfv.distance >= ? LIMIT 1",
        (vec_bytes, hook.config.decay.fact_trust_min, hook.config.lesson_dedup_similarity),
    ).fetchone()
    if dup is not None:
        log.debug("lesson: near-duplicate of a live lesson — skipping")
        return None
    fact_id = insert()
    db.execute(
        "INSERT OR REPLACE INTO semantic_facts_vec (fact_id, embedding) VALUES (?, ?)",
        (fact_id, vec_bytes),
    )
    db.commit()
    return fact_id


def _store_lesson(
    db: sqlite3.Connection, lesson: str, category: str, condition: str = ""
) -> int:
    """Insert lesson into semantic_facts. ``context`` holds the applicability
    condition (what retrieval matches against). Returns new row id."""
    now = time.time()
    cur = db.execute(
        "INSERT INTO semantic_facts "
        "(content, source_chunk_ids, keywords, tags, context, importance, "
        " fact_type, task_category, created_at) "
        "VALUES (?, '[]', '[]', '[]', ?, 5, 'cheatsheet', ?, ?)",
        (lesson, condition or category, category, now),
    )
    db.commit()
    return int(cur.lastrowid)


async def retrieve_lessons(
    hook: "NanoHermesHook",
    task_text: str,
    top_k: int = 3,
) -> list[tuple[int, str, float]]:
    """Return up to *top_k* relevant ``(fact_id, content, relevance)`` lessons.

    Ranks by ``distance / trust`` and hides facts below the trust floor, so
    lessons that correlated with past failures sink and eventually drop out.
    ``relevance`` = ``max(0, 1 − distance)``, used to weight outcome-based
    trust attribution. KNN only: with embeddings down nothing is matched to the
    task, and injecting unmatched lessons hurts (arXiv:2505.16067) while
    crediting them with outcomes they had no part in.
    """
    db = hook.db
    min_trust = hook.config.decay.fact_trust_min
    lessons: list[tuple[int, str, float]] = []

    vec_bytes = await _embed_bytes(hook, task_text[:400])
    if vec_bytes is None:
        return lessons  # nothing can be matched to the task
    # semantic_facts_vec also holds distilled 'fact' embeddings, and
    # fact_type/trust are post-KNN join filters — a small k slice can be
    # crowded out by 'fact' rows before any lesson survives the filter.
    # ponytail: fetch a generous neighbourhood; if heavy 'fact' volumes
    # still starve lessons, give lessons a dedicated vec0 table.
    knn = max(64, top_k * 8)
    try:
        rows = db.execute(
            "SELECT sf.id, sf.content, sfv.distance FROM semantic_facts_vec sfv "
            "JOIN semantic_facts sf ON sf.id = sfv.fact_id "
            "WHERE sfv.embedding MATCH ? AND sf.fact_type IN ('cheatsheet', 'expel') "
            "  AND sf.invalid_at IS NULL AND sf.trust_score >= ? AND k = ? "
            "ORDER BY sfv.distance / MAX(sf.trust_score, 0.01) "
            "LIMIT ?",
            (vec_bytes, min_trust, knn, top_k),
        ).fetchall()
        lessons = [(r[0], r[1], max(0.0, 1.0 - r[2])) for r in rows]
    except Exception:
        log.debug("cheatsheet: KNN retrieval failed — injecting none", exc_info=True)
    return lessons


def build_injection_message(lessons: list[str]) -> dict[str, str] | None:
    """Build a system injection message from retrieved lesson texts, or None."""
    if not lessons:
        return None
    bullet_list = "\n".join(f"- {lesson}" for lesson in lessons)
    return {
        "role": "system",
        "content": f"Relevant lessons from past sessions:\n{bullet_list}",
    }
