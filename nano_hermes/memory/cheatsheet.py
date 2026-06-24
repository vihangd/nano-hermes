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
(max 2 sentences) that would help on similar future tasks. \
Focus on what worked or what to avoid.

Task: {task}
Outcome: {outcome}
Key events: {events}

Output ONLY the lesson text. \
If no meaningful lesson can be extracted, output exactly: SKIP"""


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

    lesson = (resp.content or "").strip()
    if not lesson or lesson.upper().startswith("SKIP") or len(lesson) < 20:
        return

    category = _task_category(first_user)
    fact_id = _store_lesson(hook.db, lesson, category)

    # Embed asynchronously — fire-and-forget; embedding failure is non-fatal.
    import asyncio  # noqa: PLC0415
    asyncio.create_task(_embed_lesson(hook, fact_id, lesson, category))


def _store_lesson(db: sqlite3.Connection, lesson: str, category: str) -> int:
    """Insert lesson into semantic_facts. Returns new row id."""
    now = time.time()
    cur = db.execute(
        "INSERT INTO semantic_facts "
        "(content, source_chunk_ids, keywords, tags, context, importance, "
        " fact_type, task_category, created_at) "
        "VALUES (?, '[]', '[]', '[]', ?, 5, 'cheatsheet', ?, ?)",
        (lesson, category, category, now),
    )
    db.commit()
    return int(cur.lastrowid)


async def _embed_lesson(
    hook: "NanoHermesHook", fact_id: int, lesson: str, category: str
) -> None:
    """Embed the lesson and store in semantic_facts_vec (best-effort)."""
    if fact_id <= 0:
        return
    try:
        import numpy as np  # noqa: PLC0415

        async with hook.embedder() as chain:
            [vec] = await chain.embed([lesson])
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
) -> list[str]:
    """Return up to *top_k* relevant cheatsheet lessons for the current task.

    Tries KNN over ``semantic_facts_vec`` first; falls back to recency-ordered
    FTS5 when embeddings are unavailable.
    """
    db = hook.db
    lessons: list[str] = []

    # Try KNN embedding search.
    try:
        import numpy as np  # noqa: PLC0415

        async with hook.embedder() as chain:
            [vec] = await chain.embed([task_text[:400]])
        if vec is not None:
            vec_bytes = np.asarray(vec, dtype=np.float32).tobytes()
            rows = db.execute(
                "SELECT sf.content FROM semantic_facts_vec sfv "
                "JOIN semantic_facts sf ON sf.id = sfv.fact_id "
                "WHERE sfv.embedding MATCH ? AND sf.fact_type IN ('cheatsheet', 'expel') "
                "  AND sf.invalid_at IS NULL AND k = ? "
                "ORDER BY sfv.distance",
                (vec_bytes, top_k),
            ).fetchall()
            lessons = [r[0] for r in rows]
    except Exception:
        log.debug("cheatsheet: KNN retrieval failed — using FTS fallback", exc_info=True)

    if not lessons:
        try:
            rows = db.execute(
                "SELECT content FROM semantic_facts "
                "WHERE fact_type IN ('cheatsheet', 'expel') AND invalid_at IS NULL "
                "ORDER BY created_at DESC LIMIT ?",
                (top_k,),
            ).fetchall()
            lessons = [r[0] for r in rows]
        except Exception:
            log.debug("cheatsheet: FTS fallback failed", exc_info=True)

    return lessons


def build_injection_message(lessons: list[str]) -> dict[str, str] | None:
    """Build a system injection message from retrieved lessons, or None."""
    if not lessons:
        return None
    bullet_list = "\n".join(f"- {lesson}" for lesson in lessons)
    return {
        "role": "system",
        "content": f"Relevant lessons from past sessions:\n{bullet_list}",
    }
