"""The ``session_search`` agent-facing Tool.

Hybrid FTS5 + sqlite-vec retrieval fused with Reciprocal Rank Fusion
(Cormack et al., k=60 default). On any embedding-provider failure we
degrade to FTS5-only so the tool still returns something useful.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from nanobot.agent.tools.base import Tool, tool_parameters

from ..config import RetrievalConfig

if TYPE_CHECKING:
    from ..hook import NanoHermesHook


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
    fts_rows = conn.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT ?",
        (query_text, cfg.fts_k),
    ).fetchall()
    fts_ids = [r[0] for r in fts_rows]

    vec_blob = query_vec.astype(np.float32).tobytes()
    vec_rows = conn.execute(
        "SELECT chunk_id FROM chunks_vec "
        "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (vec_blob, cfg.vec_k),
    ).fetchall()
    vec_ids = [r[0] for r in vec_rows]

    fused = reciprocal_rank_fusion(fts_ids, vec_ids, cfg.rrf_k)
    top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[: cfg.final_k]
    if not top:
        return []

    placeholders = ",".join("?" * len(top))
    rows = conn.execute(
        f"SELECT id, session_id, content FROM chunks WHERE id IN ({placeholders})",
        [cid for cid, _ in top],
    ).fetchall()
    by_id = {r[0]: r for r in rows}
    return [
        Hit(chunk_id=cid, session_id=by_id[cid][1], content=by_id[cid][2], score=score)
        for cid, score in top
        if cid in by_id
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
        k = int(kwargs.get("k") or 8)
        cfg = self._hook.config.retrieval.model_copy(update={"final_k": k})

        try:
            async with self._hook.embedder() as chain:
                [qv] = await chain.embed([query])
        except Exception as e:
            return self._fts_only_fallback(query, cfg, reason=str(e))

        hits = hybrid_search(self._hook.db, query, qv, cfg)
        if not hits:
            return "no matches"
        return "\n".join(
            f"[{h.session_id}#{h.chunk_id} score={h.score:.3f}] {h.content[:240]}"
            for h in hits
        )

    def _fts_only_fallback(
        self, query: str, cfg: RetrievalConfig, *, reason: str
    ) -> str:
        # FTS5's MATCH operator requires the real table name, not an alias —
        # `f MATCH ?` parses as referencing a column called `f`. Join from
        # chunks_fts and order by its hidden BM25 `rank` column.
        rows = self._hook.db.execute(
            "SELECT chunks.id, chunks.session_id, chunks.content "
            "FROM chunks_fts "
            "JOIN chunks ON chunks.id = chunks_fts.rowid "
            "WHERE chunks_fts MATCH ? "
            "ORDER BY chunks_fts.rank "
            "LIMIT ?",
            (query, cfg.final_k),
        ).fetchall()
        if not rows:
            return f"no matches (embedding unavailable: {reason})"
        return "\n".join(f"[{r[1]}#{r[0]}] {r[2][:240]}" for r in rows)
