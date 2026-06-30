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
from typing import TYPE_CHECKING, Any

from ..utils.error_classifier import classify_llm_response

if TYPE_CHECKING:
    from ..hook import NanoHermesHook

log = logging.getLogger(__name__)

_EXTRACT_PROMPT = """\
Given this completed session, extract ONE concise transferable lesson \
(max 2 sentences) that would help on similar future tasks, AND a one-line \
condition describing WHEN the lesson applies. Focus on what worked or what to avoid.

Task: {task}
Outcome: {outcome}
Key events: {events}

Output exactly two lines:
LESSON: <the lesson>
WHEN: <one-line situation where this lesson applies>
If no meaningful lesson can be extracted, output exactly: SKIP"""


def _parse_marked(raw: str, head: str) -> tuple[str, str]:
    """Parse a ``HEAD: …`` / ``WHEN: …`` response into (body, condition).

    Captures multi-line bodies (everything up to the WHEN marker). Falls back
    to treating the whole text as the body when no HEAD marker is present.
    """
    body_lines: list[str] = []
    when = ""
    section = None  # None | "body" | "when"
    for line in raw.splitlines():
        s = line.strip()
        up = s.upper()
        if up.startswith(head + ":"):
            section = "body"
            body_lines.append(s[len(head) + 1:].strip())
        elif up.startswith("WHEN:"):
            section = "when"
            when = s[len("WHEN:"):].strip()
        elif section == "body":
            body_lines.append(s)
        elif section == "when" and s:
            when = (when + " " + s).strip()
    body = " ".join(p for p in body_lines if p).strip()
    if not body:
        body = raw.strip()  # old-style: whole text is the body
    return body, when


def _parse_lesson(raw: str) -> tuple[str, str]:
    """Parse the LESSON/WHEN response. Tolerates models that ignore the format."""
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
            model=getattr(hook._loop, "model", None),
            max_tokens=150,
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
    lesson, condition = _parse_lesson(raw)
    if not lesson or len(lesson) < 20:
        return

    category = _task_category(first_user)
    # The applicability condition is what retrieval matches the current task
    # against; fall back to the task category when the model omits it.
    applies_when = condition or category
    fact_id = _store_lesson(hook.db, lesson, category, applies_when)

    # Embed asynchronously — fire-and-forget; embedding failure is non-fatal.
    import asyncio  # noqa: PLC0415
    asyncio.create_task(_embed_lesson(hook, fact_id, applies_when))


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


async def _embed_lesson(
    hook: "NanoHermesHook", fact_id: int, embed_text: str
) -> None:
    """Embed the applicability condition into semantic_facts_vec (best-effort)."""
    if fact_id <= 0 or not embed_text:
        return
    try:
        import numpy as np  # noqa: PLC0415

        async with hook.embedder() as chain:
            [vec] = await chain.embed([embed_text])
        if vec is None:
            return
        vec_bytes = np.asarray(vec, dtype=np.float32).tobytes()
        hook.db.execute(
            "INSERT OR REPLACE INTO semantic_facts_vec (fact_id, embedding) VALUES (?, ?)",
            (fact_id, vec_bytes),
        )
        hook.db.commit()
    except Exception:
        log.debug("cheatsheet: embedding failed for fact %d", fact_id, exc_info=True)


async def retrieve_lessons(
    hook: "NanoHermesHook",
    task_text: str,
    top_k: int = 3,
) -> list[tuple[int, str, float]]:
    """Return up to *top_k* relevant ``(fact_id, content, relevance)`` lessons.

    Ranks by ``distance / trust`` and hides facts below the trust floor, so
    lessons that correlated with past failures sink and eventually drop out.
    ``relevance`` = ``max(0, 1 − distance)`` (1.0 on the FTS fallback), used to
    weight outcome-based trust attribution. Tries KNN over ``semantic_facts_vec``
    first; falls back to trust/recency-ordered scan when embeddings are down.
    """
    db = hook.db
    min_trust = hook.config.decay.fact_trust_min
    lessons: list[tuple[int, str, float]] = []

    # Try KNN embedding search.
    try:
        import numpy as np  # noqa: PLC0415

        async with hook.embedder() as chain:
            [vec] = await chain.embed([task_text[:400]])
        if vec is not None:
            vec_bytes = np.asarray(vec, dtype=np.float32).tobytes()
            # semantic_facts_vec also holds distilled 'fact' embeddings, and
            # fact_type/trust are post-KNN join filters — a small k slice can be
            # crowded out by 'fact' rows before any lesson survives the filter.
            # ponytail: fetch a generous neighbourhood; if heavy 'fact' volumes
            # still starve lessons, give lessons a dedicated vec0 table.
            knn = max(64, top_k * 8)
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
        log.debug("cheatsheet: KNN retrieval failed — using fallback", exc_info=True)

    if not lessons:
        try:
            rows = db.execute(
                "SELECT id, content FROM semantic_facts "
                "WHERE fact_type IN ('cheatsheet', 'expel') AND invalid_at IS NULL "
                "  AND trust_score >= ? "
                "ORDER BY trust_score DESC, created_at DESC LIMIT ?",
                (min_trust, top_k),
            ).fetchall()
            lessons = [(r[0], r[1], 1.0) for r in rows]
        except Exception:
            log.debug("cheatsheet: fallback retrieval failed", exc_info=True)

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
