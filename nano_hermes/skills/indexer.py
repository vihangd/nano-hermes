"""Embedding-indexed skill retrieval (Voyager-style).

Complements nanobot's static ``SkillsLoader.build_skills_summary()`` —
which dumps every skill's name + description into the system prompt in
alphabetical order — with dynamic top-K semantic ranking keyed on the
agent's current query.

Index bookkeeping:

- ``skill_stats`` stores one row per skill name with a ``content_hash``
  and ``indexed_at`` timestamp. The stable integer PK ``id`` doubles as
  the rowid in ``skill_vec``.
- Unchanged skills (same content hash) are NOT re-embedded on refresh.
- Deleted skills (no longer found on disk) are removed from both tables
  on the next refresh.
- ``search()`` runs an implicit refresh before embedding the query —
  cheap when nothing changed, at most one batched embed when things did.

We embed name + description only, NOT the full body. Two reasons:
1. Full bodies drift in meaning ("weather" mentioned in passing would
   pollute the vector for "forecast retrieval"). Descriptions are
   precise by design.
2. Embedding cost scales with skill count, not body length.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from nanobot.agent.skills import SkillsLoader as NanobotSkillsLoader

from ..embedding.chain import AllProvidersFailed, EmbeddingChain

log = logging.getLogger(__name__)


@dataclass
class SkillHit:
    name: str
    description: str
    location: str
    distance: float


def _content_text(name: str, description: str) -> str:
    return f"{name}: {description}" if description else name


def _hash_content(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class SkillIndexer:
    def __init__(
        self,
        *,
        db: sqlite3.Connection,
        skills_loader: NanobotSkillsLoader,
        embedder_factory: Callable[[], EmbeddingChain],
    ) -> None:
        self._db = db
        self._skills_loader = skills_loader
        self._embedder_factory = embedder_factory

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(self, query: str, k: int = 5) -> list[SkillHit]:
        """Refresh the index, then return top-k skills by similarity to *query*.

        Raises ``AllProvidersFailed`` if every embedding provider is
        unreachable — the wrapping Tool decides how to surface the error.
        """
        entries = self._list_entries()
        description_by_name = self._descriptions(entries)
        location_by_name = {e["name"]: e["path"] for e in entries}

        async with self._embedder_factory() as chain:
            await self._refresh_with_chain(entries, description_by_name, chain)
            [query_vec] = await chain.embed([query])

        return self._vec_query(query_vec, k, description_by_name, location_by_name)

    async def refresh(self) -> dict[str, int]:
        """Explicit refresh — same work that ``search`` does automatically."""
        entries = self._list_entries()
        description_by_name = self._descriptions(entries)
        async with self._embedder_factory() as chain:
            return await self._refresh_with_chain(
                entries, description_by_name, chain
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _list_entries(self) -> list[dict[str, Any]]:
        return self._skills_loader.list_skills(filter_unavailable=False)

    def _descriptions(self, entries: list[dict[str, Any]]) -> dict[str, str]:
        out: dict[str, str] = {}
        for entry in entries:
            meta = self._skills_loader.get_skill_metadata(entry["name"]) or {}
            out[entry["name"]] = meta.get("description", "")
        return out

    async def _refresh_with_chain(
        self,
        entries: list[dict[str, Any]],
        description_by_name: dict[str, str],
        chain: EmbeddingChain,
    ) -> dict[str, int]:
        stale: list[tuple[str, str, str]] = []  # (name, text, digest)
        for entry in entries:
            name = entry["name"]
            text = _content_text(name, description_by_name.get(name, ""))
            digest = _hash_content(text)
            row = self._db.execute(
                "SELECT content_hash FROM skill_stats WHERE name = ?", (name,)
            ).fetchone()
            if row and row[0] == digest:
                continue
            stale.append((name, text, digest))

        if stale:
            vecs = await chain.embed([t[1] for t in stale])
            self._write_vectors(stale, vecs)

        removed = self._cleanup_removed({e["name"] for e in entries})
        return {
            "total": len(entries),
            "reindexed": len(stale),
            "removed": removed,
        }

    def _write_vectors(
        self,
        stale: list[tuple[str, str, str]],
        vecs: list[np.ndarray],
    ) -> None:
        now = time.time()
        with self._db:
            for (name, _text, digest), vec in zip(stale, vecs):
                self._db.execute(
                    "INSERT INTO skill_stats (name, content_hash, indexed_at) "
                    "VALUES (?, ?, ?) "
                    "ON CONFLICT(name) DO UPDATE SET "
                    "content_hash = excluded.content_hash, "
                    "indexed_at = excluded.indexed_at",
                    (name, digest, now),
                )
                skill_id = self._db.execute(
                    "SELECT id FROM skill_stats WHERE name = ?", (name,)
                ).fetchone()[0]
                # vec0 doesn't support ON CONFLICT — delete then insert.
                self._db.execute(
                    "DELETE FROM skill_vec WHERE skill_id = ?", (skill_id,)
                )
                self._db.execute(
                    "INSERT INTO skill_vec (skill_id, embedding) VALUES (?, ?)",
                    (skill_id, vec.astype(np.float32).tobytes()),
                )

    def _cleanup_removed(self, current_names: set[str]) -> int:
        existing = self._db.execute("SELECT id, name FROM skill_stats").fetchall()
        removed = 0
        with self._db:
            for row_id, name in existing:
                if name in current_names:
                    continue
                self._db.execute(
                    "DELETE FROM skill_vec WHERE skill_id = ?", (row_id,)
                )
                self._db.execute(
                    "DELETE FROM skill_stats WHERE id = ?", (row_id,)
                )
                removed += 1
        return removed

    def _vec_query(
        self,
        query_vec: np.ndarray,
        k: int,
        description_by_name: dict[str, str],
        location_by_name: dict[str, str],
    ) -> list[SkillHit]:
        vec_blob = query_vec.astype(np.float32).tobytes()
        rows = self._db.execute(
            "SELECT skill_id, distance FROM skill_vec "
            "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (vec_blob, k),
        ).fetchall()
        if not rows:
            return []

        placeholders = ",".join("?" * len(rows))
        name_rows = self._db.execute(
            f"SELECT id, name FROM skill_stats WHERE id IN ({placeholders})",
            [r[0] for r in rows],
        ).fetchall()
        id_to_name = {r[0]: r[1] for r in name_rows}

        hits: list[SkillHit] = []
        for skill_id, distance in rows:
            name = id_to_name.get(skill_id)
            if not name:
                continue
            hits.append(
                SkillHit(
                    name=name,
                    description=description_by_name.get(name, ""),
                    location=location_by_name.get(name, ""),
                    distance=float(distance),
                )
            )
        return hits
