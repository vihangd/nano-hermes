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
import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from nanobot.agent.skills import SkillsLoader as NanobotSkillsLoader

from ..config import SkillStatsConfig
from ..embedding.chain import EmbeddingChain
from .external import discover_external_skills, expand_external_dirs

log = logging.getLogger(__name__)


@dataclass
class SkillHit:
    name: str
    description: str
    location: str
    distance: float
    siblings: list[str] = field(default_factory=list)


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
        external_dirs: list[str] | None = None,
    ) -> None:
        self._db = db
        self._skills_loader = skills_loader
        self._embedder_factory = embedder_factory
        self._stats_config = stats_config
        self._external_dirs = expand_external_dirs(external_dirs or [])

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
        entries = self._skills_loader.list_skills(filter_unavailable=False)
        if not self._external_dirs:
            return entries
        seen_names = {e["name"] for e in entries}
        # Workspace > builtin > external precedence: skip externals shadowed
        # by a same-named skill from SkillsLoader.
        for ext in discover_external_skills(self._external_dirs):
            if ext["name"] in seen_names:
                continue
            entries.append(ext)
        return entries

    def _descriptions(self, entries: list[dict[str, Any]]) -> dict[str, str]:
        out: dict[str, str] = {}
        for entry in entries:
            if entry.get("source") == "external":
                # SkillsLoader.get_skill_metadata returns None for paths
                # outside its known dirs; the external entry carries its
                # description inline.
                out[entry["name"]] = entry.get("description", "")
            else:
                meta = self._skills_loader.get_skill_metadata(entry["name"]) or {}
                out[entry["name"]] = meta.get("description", "")
        return out

    def find_external_skill(self, name: str) -> Path | None:
        """Return the directory containing an external SKILL.md by name,
        or None when no external dir contains a skill with that name.
        """
        for ext in discover_external_skills(self._external_dirs):
            if ext["name"] == name:
                return Path(ext["path"]).parent
        return None

    async def _refresh_with_chain(
        self,
        entries: list[dict[str, Any]],
        description_by_name: dict[str, str],
        chain: EmbeddingChain,
    ) -> dict[str, int]:
        source_by_name = {e["name"]: e.get("source", "workspace") for e in entries}
        # One scan of skill_stats serves both stale-detection and cleanup,
        # instead of a per-entry SELECT (N queries) plus a second full scan.
        existing = self._db.execute(
            "SELECT id, name, content_hash FROM skill_stats"
        ).fetchall()
        hash_by_name = {name: content_hash for _id, name, content_hash in existing}

        stale: list[tuple[str, str, str, str]] = []  # (name, text, digest, source)
        for entry in entries:
            name = entry["name"]
            text = _content_text(name, description_by_name.get(name, ""))
            digest = _hash_content(text)
            if hash_by_name.get(name) == digest:
                continue
            stale.append((name, text, digest, source_by_name[name]))

        if stale:
            vecs = await chain.embed([t[1] for t in stale])
            self._write_vectors(stale, vecs)

        removed = self._cleanup_removed({e["name"] for e in entries}, existing=existing)
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
                # Builtin and external skills are trusted (config-curated)
                # and start active. Workspace skills (including those created
                # via shell scripts outside propose_skill) start as draft so
                # they must pass skill_rate to promote.
                # ON CONFLICT preserves the existing status — only set it on INSERT.
                status = "active" if source in ("builtin", "external") else "draft"
                self._db.execute(
                    "INSERT INTO skill_stats (name, status, content_hash, indexed_at) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(name) DO UPDATE SET "
                    "content_hash = excluded.content_hash, "
                    "indexed_at = excluded.indexed_at",
                    (name, status, digest, now),
                )
                # cursor.lastrowid is NOT the conflicting row's id on the
                # ON CONFLICT DO UPDATE branch (it keeps the last real INSERT's
                # rowid), which would write the vector under the wrong skill and
                # clobber another skill's embedding. Read the canonical id by
                # name instead (UNIQUE-indexed, works on all SQLite versions).
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

    def _cleanup_removed(
        self,
        current_names: set[str],
        existing: list[tuple] | None = None,
    ) -> int:
        # Reuse the caller's scan when provided (rows may carry extra columns —
        # only id and name are read); otherwise do our own minimal fetch.
        if existing is None:
            existing = self._db.execute("SELECT id, name FROM skill_stats").fetchall()
        removed = 0
        with self._db:
            for row in existing:
                row_id, name = row[0], row[1]
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
        cfg = self._stats_config
        ranking_mode = cfg.ranking_mode if cfg else "off"

        # Widen the candidate pool when re-ranking so the top-k after
        # adjustment isn't constrained by the pre-rerank ordering.
        fetch_k = k * 3 if ranking_mode != "off" else k
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

        # UCB1: fetch total pull count across all active skills once.
        total_uses: int = 0
        if ranking_mode == "ucb1":
            row = self._db.execute(
                "SELECT COALESCE(SUM(use_count), 0) FROM skill_stats "
                "WHERE status != 'deprecated'"
            ).fetchone()
            total_uses = int(row[0]) if row else 0

        hits: list[SkillHit] = []
        for skill_id, distance in rows:
            name = id_to_name.get(skill_id)
            if not name:
                continue
            effective_distance = float(distance)
            use_count, success_count = id_to_stats.get(skill_id, (0, 0))

            if ranking_mode == "ucb1":
                success_rate = success_count / use_count if use_count > 0 else 0.0
                # Exploration bonus: large for cold-start skills, decays as
                # skill_uses grows relative to total_uses.
                exploration = math.sqrt(
                    2.0 * math.log(total_uses + 1) / max(use_count, 1)
                )
                ucb1_score = success_rate + exploration
                effective_distance -= cfg.ucb1_coefficient * ucb1_score  # type: ignore[union-attr]
            elif ranking_mode == "stat_weighted":
                min_uses = cfg.min_uses_for_success_rate if cfg else 3  # type: ignore[union-attr]
                boost = cfg.success_rate_boost if cfg else 0.0  # type: ignore[union-attr]
                if use_count >= min_uses:
                    success_rate = success_count / use_count
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

        # Greedy diversity dedup: suppress near-duplicate skills from lower-ranked
        # positions so the agent sees a more diverse result set.
        dedup_threshold = cfg.skill_search_dedup_threshold if cfg else 1.0
        if dedup_threshold < 1.0 and len(hits) > 1:
            name_to_id = {v: k for k, v in id_to_name.items()}
            hit_ids = [name_to_id[h.name] for h in hits if h.name in name_to_id]
            if hit_ids:
                placeholders = ",".join("?" * len(hit_ids))
                emb_rows = self._db.execute(
                    f"SELECT skill_id, embedding FROM skill_vec "
                    f"WHERE skill_id IN ({placeholders})",
                    hit_ids,
                ).fetchall()
                emb_by_id = {
                    r[0]: np.frombuffer(r[1], dtype=np.float32) for r in emb_rows
                }
                kept: list[SkillHit] = []
                kept_embs: list[np.ndarray | None] = []
                for h in hits:
                    sid = name_to_id.get(h.name)
                    emb = emb_by_id.get(sid) if sid is not None else None  # type: ignore[arg-type]
                    dominated = False
                    if emb is not None:
                        for i, ke in enumerate(kept_embs):
                            if ke is not None and float(np.dot(emb, ke)) >= dedup_threshold:
                                kept[i].siblings.append(h.name)
                                dominated = True
                                break
                    if not dominated:
                        kept.append(h)
                        kept_embs.append(emb)
                hits = kept

        return hits[:k]
