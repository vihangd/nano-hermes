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

from ..config import SkillStatsConfig
from ..embedding.chain import EmbeddingChain

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
        stats_config: SkillStatsConfig | None = None,
    ) -> None:
        self._db = db
        self._skills_loader = skills_loader
        self._embedder_factory = embedder_factory
        self._stats_config = stats_config

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
        source_by_name = {e["name"]: e.get("source", "workspace") for e in entries}
        stale: list[tuple[str, str, str, str]] = []  # (name, text, digest, source)
        for entry in entries:
            name = entry["name"]
            text = _content_text(name, description_by_name.get(name, ""))
            digest = _hash_content(text)
            row = self._db.execute(
                "SELECT content_hash FROM skill_stats WHERE name = ?", (name,)
            ).fetchone()
            if row and row[0] == digest:
                continue
            stale.append((name, text, digest, source_by_name[name]))

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
        stale: list[tuple[str, str, str, str]],
        vecs: list[np.ndarray],
    ) -> None:
        now = time.time()
        with self._db:
            for (name, _text, digest, source), vec in zip(stale, vecs):
                # Builtin skills are trusted and start active. Workspace skills
                # (including those created via shell scripts outside propose_skill)
                # start as draft so they must pass skill_rate to promote.
                # ON CONFLICT preserves the existing status — only set it on INSERT.
                status = "active" if source == "builtin" else "draft"
                self._db.execute(
                    "INSERT INTO skill_stats (name, status, content_hash, indexed_at) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(name) DO UPDATE SET "
                    "content_hash = excluded.content_hash, "
                    "indexed_at = excluded.indexed_at",
                    (name, status, digest, now),
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
        # Fetch a wider pool when stat weighting is on so re-ranking has
        # enough candidates — top-k might not survive after boost.
        fetch_k = k * 3 if (self._stats_config and self._stats_config.use_stat_weighting) else k
        vec_blob = query_vec.astype(np.float32).tobytes()
        rows = self._db.execute(
            "SELECT skill_id, distance FROM skill_vec "
            "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (vec_blob, fetch_k),
        ).fetchall()
        if not rows:
            return []

        placeholders = ",".join("?" * len(rows))
        stat_rows = self._db.execute(
            f"SELECT id, name, use_count, success_count FROM skill_stats "
            f"WHERE id IN ({placeholders}) AND status != 'deprecated'",
            [r[0] for r in rows],
        ).fetchall()
        id_to_name = {r[0]: r[1] for r in stat_rows}
        id_to_stats = {r[0]: (r[2], r[3]) for r in stat_rows}  # id → (use_count, success_count)

        cfg = self._stats_config
        min_uses = cfg.min_uses_for_success_rate if cfg else 3
        boost = cfg.success_rate_boost if cfg else 0.0
        use_weighting = cfg.use_stat_weighting if cfg else False

        hits: list[SkillHit] = []
        for skill_id, distance in rows:
            name = id_to_name.get(skill_id)
            if not name:
                continue
            effective_distance = float(distance)
            if use_weighting:
                use_count, success_count = id_to_stats.get(skill_id, (0, 0))
                if use_count >= min_uses:
                    success_rate = success_count / use_count
                    # Subtract a small success-rate bonus from distance so
                    # highly reliable skills win ties and narrow races.
                    # Using additive adjustment (not multiplicative) so the
                    # bonus works even when raw distance is 0.
                    # Default boost=0.3 → max adjustment of 0.003 at 100%
                    # success rate — tiny vs typical L2 gaps (~0.1–1.4).
                    effective_distance -= boost * success_rate * 0.01
            hits.append(
                SkillHit(
                    name=name,
                    description=description_by_name.get(name, ""),
                    location=location_by_name.get(name, ""),
                    distance=effective_distance,
                )
            )

        hits.sort(key=lambda h: h.distance)
        return hits[:k]
