"""The ``session_search`` agent-facing Tool.

Hybrid FTS5 + sqlite-vec retrieval fused with Reciprocal Rank Fusion
(Cormack et al., k=60 default). On any embedding-provider failure we
degrade to FTS5-only so the tool still returns something useful.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from nanobot.agent.tools.base import Tool, tool_parameters

from ..config import RetrievalConfig
from .mmr import MMRHit, mmr_rerank

if TYPE_CHECKING:
    from ..hook import NanoHermesHook

# CJK Unified, Extensions A–F, Compatibility Ideographs, Symbols,
# Hiragana, Katakana, Hangul — ranges where FTS5's unicode61 tokenizer
# produces whole-string tokens rather than per-character tokens, causing
# multi-character CJK queries to return 0 FTS results.
# Extensions B–F live in the supplementary plane (U+20000–U+2FFFF).
_CJK_RE = re.compile(
    r"[\u2E80-\u9FFF\uF900-\uFAFF\uFE30-\uFE4F"
    r"\u3000-\u303F\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF"
    r"\U00020000-\U0002FFFF]"
)


def _contains_cjk(s: str) -> bool:
    return bool(_CJK_RE.search(s))


def _like_escape(s: str) -> str:
    """Escape SQLite LIKE wildcards so the pattern matches the literal string."""
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


_FTS_STOPWORDS = frozenset(
    "a an and are as at be but by for from in is it of on or that the this to "
    "was were will with what how why when who".split()
)


def sanitize_fts_query(text: str) -> str:
    """Free prose → a safe FTS5 OR-query of quoted terms.

    FTS5 AND-joins bare tokens (zeroing recall on prose) and raises a syntax
    error on punctuation; quoting each term and OR-joining restores recall and
    makes special characters literal. Common stopwords are dropped so they
    don't widen the BM25 candidate scan. Natural-language only — FTS5 operators
    / phrases / prefixes are quoted literal, not interpreted. Returns '' when
    no usable term remains.
    """
    terms: list[str] = []
    stop: list[str] = []
    for raw in text.split():
        t = raw.replace('"', "")
        if not any(c.isalnum() for c in t):
            continue
        (stop if t.lower() in _FTS_STOPWORDS else terms).append(f'"{t}"')
    return " OR ".join(terms or stop)  # keep stopwords only if nothing else left


def _fts_rows(executor, sql: str, query_text: str, *params) -> list:
    """Run one FTS5 text query safely: sanitize prose → OR-query, skip on an
    empty query, and drop the lexical channel on a malformed query instead of
    raising. *sql* holds one ``MATCH ?`` placeholder (bound to the sanitized
    query) followed by placeholders for *params*; ``ORDER BY rank`` belongs in
    *sql* so results come back BM25-ranked, not rowid-ordered.
    """
    fts_q = sanitize_fts_query(query_text)
    if not fts_q:
        return []
    try:
        return executor.execute(sql, (fts_q, *params)).fetchall()
    except sqlite3.OperationalError:
        return []


def _match_centered_snippet(text: str, query: str, max_chars: int = 240) -> str:
    """Return up to *max_chars* of *text*, centered on where *query* matches.

    Three-tier fallback (mirrors hermes-agent a5bc698b):
    1. Full-phrase match — center window on the literal query string.
    2. Proximity co-occurrence — find the densest span where all query
       terms appear within 200 chars of each other.
    3. Individual terms — first occurrence of any single term.
    Within the chosen span a 25%-before / 75%-after window bias is applied
    so the context flows forward from the match.
    """
    if len(text) <= max_chars:
        return text

    text_lower = text.lower()
    query_lower = query.lower().strip()
    positions: list[int] = []

    # 1. Full-phrase
    for m in re.finditer(re.escape(query_lower), text_lower):
        positions.append(m.start())

    # 2. Proximity co-occurrence of all terms within 200 chars
    if not positions:
        terms = query_lower.split()
        if len(terms) > 1:
            term_pos: dict[str, list[int]] = {
                t: [m.start() for m in re.finditer(re.escape(t), text_lower)]
                for t in terms
            }
            rarest = min(terms, key=lambda t: len(term_pos.get(t, [])))
            for pos in term_pos.get(rarest, []):
                if all(
                    any(abs(p - pos) < 200 for p in term_pos.get(t, []))
                    for t in terms
                    if t != rarest
                ):
                    positions.append(pos)

    # 3. Individual terms (last resort)
    if not positions:
        for t in query_lower.split():
            for m in re.finditer(re.escape(t), text_lower):
                positions.append(m.start())

    if not positions:
        return text[:max_chars] + ("…" if len(text) > max_chars else "")

    # Pick the window that covers the most match positions.
    positions.sort()
    best_start, best_count = 0, 0
    for cand in positions:
        ws = max(0, cand - max_chars // 4)  # 25% before, 75% after
        if ws + max_chars > len(text):
            ws = max(0, len(text) - max_chars)
        count = sum(1 for p in positions if ws <= p < ws + max_chars)
        if count > best_count:
            best_count, best_start = count, ws

    start = best_start
    end = min(len(text), start + max_chars)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return prefix + text[start:end] + suffix


@dataclass
class Hit:
    chunk_id: int
    session_id: int
    content: str
    score: float


def reciprocal_rank_fusion(
    fts_hits: list[int], vec_hits: list[int], k: int
) -> dict[int, float]:
    scores: dict[int, float] = {}
    for rank, cid in enumerate(fts_hits):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    for rank, cid in enumerate(vec_hits):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return scores


def hybrid_search(
    conn: sqlite3.Connection,
    query_text: str,
    query_vec: np.ndarray,
    cfg: RetrievalConfig,
) -> list[Hit]:
    fts_rows = _fts_rows(
        conn,
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
        query_text,
        cfg.fts_k,
    )
    fts_ids = [r[0] for r in fts_rows]

    # FTS5's unicode61 tokenizer doesn't segment CJK text, so queries in
    # Chinese/Japanese/Korean return 0 FTS hits despite matching content.
    # Fall back to a LIKE scan when FTS5 comes up empty on a CJK query.
    if not fts_ids and _contains_cjk(query_text):
        like_rows = conn.execute(
            "SELECT id FROM chunks WHERE content LIKE ? ESCAPE '\\' LIMIT ?",
            (f"%{_like_escape(query_text)}%", cfg.fts_k),
        ).fetchall()
        fts_ids = [r[0] for r in like_rows]

    vec_blob = query_vec.astype(np.float32).tobytes()
    vec_rows = conn.execute(
        "SELECT chunk_id FROM chunks_vec "
        "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (vec_blob, cfg.vec_k),
    ).fetchall()
    vec_ids = [r[0] for r in vec_rows]

    fused = reciprocal_rank_fusion(fts_ids, vec_ids, cfg.rrf_k)
    # Sort by RRF score descending; keep a wider pool for MMR to re-rank.
    candidates = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    if not candidates:
        return []

    placeholders = ",".join("?" * len(candidates))
    rows = conn.execute(
        f"SELECT id, session_id, content FROM chunks WHERE id IN ({placeholders})",
        [cid for cid, _ in candidates],
    ).fetchall()
    by_id = {r[0]: r for r in rows}

    mmr_hits = [
        MMRHit(
            chunk_id=cid,
            session_id=by_id[cid][1],
            content=by_id[cid][2],
            score=score,
        )
        for cid, score in candidates
        if cid in by_id
    ]

    lam = getattr(cfg, "mmr_lambda", 1.0)
    reranked = mmr_rerank(mmr_hits, lam=lam, k=cfg.final_k)
    return [
        Hit(chunk_id=h.chunk_id, session_id=h.session_id, content=h.content, score=h.score)
        for h in reranked
    ]


_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "Free-text query. Used both for FTS5 keyword match and "
                "embedding similarity search — write it like you would to "
                "a human, not like a keyword search."
            ),
        },
        "k": {
            "type": "integer",
            "minimum": 1,
            "maximum": 32,
            "description": "How many hits to return. Defaults to 8.",
        },
    },
    "required": ["query"],
}


@tool_parameters(_SCHEMA)
class SessionSearchTool(Tool):
    """Search past sessions via hybrid FTS5 + embedding retrieval (RRF).

    Use to recall prior decisions, facts, or workflows that aren't in hot
    memory (MEMORY.md / USER.md / SOUL.md). Degrades to keyword-only if
    every embedding provider is unreachable — the tool still returns
    something in that case, just with lower recall on rephrased queries.
    """

    def __init__(self, *, hook: "NanoHermesHook") -> None:
        self._hook = hook

    @property
    def name(self) -> str:
        return "session_search"

    @property
    def description(self) -> str:
        return (type(self).__doc__ or "").strip()

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, **kwargs: Any) -> str:
        query: str = kwargs["query"]
        if not query.strip():
            return "Error: query must not be empty."
        k = int(kwargs.get("k") or 8)
        cfg = self._hook.config.retrieval.model_copy(update={"final_k": k})

        try:
            async with self._hook.embedder() as chain:
                [qv] = await chain.embed([query])
        except Exception as e:
            return self._fts_only_fallback(query, cfg, reason=str(e))

        try:
            hits = hybrid_search(self._hook.db, query, qv, cfg)
        except Exception as e:
            return self._fts_only_fallback(query, cfg, reason=str(e))
        if not hits:
            return "no matches"
        return "\n".join(
            f"[{h.session_id}#{h.chunk_id} score={h.score:.3f}] "
            f"{_match_centered_snippet(h.content, query)}"
            for h in hits
        )

    def _fts_only_fallback(
        self, query: str, cfg: RetrievalConfig, *, reason: str
    ) -> str:
        # FTS5's MATCH operator requires the real table name, not an alias —
        # `f MATCH ?` parses as referencing a column called `f`. Join from
        # chunks_fts and order by its hidden BM25 `rank` column.
        rows = _fts_rows(
            self._hook.db,
            "SELECT chunks.id, chunks.session_id, chunks.content "
            "FROM chunks_fts "
            "JOIN chunks ON chunks.id = chunks_fts.rowid "
            "WHERE chunks_fts MATCH ? "
            "ORDER BY chunks_fts.rank "
            "LIMIT ?",
            query,
            cfg.final_k,
        )
        if not rows and _contains_cjk(query):
            rows = self._hook.db.execute(
                "SELECT id, session_id, content FROM chunks "
                "WHERE content LIKE ? ESCAPE '\\' LIMIT ?",
                (f"%{_like_escape(query)}%", cfg.final_k),
            ).fetchall()
        if not rows:
            return f"no matches (embedding unavailable: {reason})"
        return "\n".join(
            f"[{r[1]}#{r[0]}] {_match_centered_snippet(r[2], query)}"
            for r in rows
        )
