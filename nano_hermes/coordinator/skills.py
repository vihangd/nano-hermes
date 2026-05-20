"""SkillUsageTracker — extracted from NanoHermesHook.

Manages per-iteration and per-session skill state: which skills the agent
searched for (candidates), which it actually loaded (observed reads), and
which were explicitly rated. Also owns the promotion/deprecation lifecycle.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..config import SkillStatsConfig

log = logging.getLogger(__name__)


class SkillUsageTracker:
    def __init__(
        self,
        *,
        db: sqlite3.Connection,
        config: "SkillStatsConfig | None",
    ) -> None:
        self._db = db
        self._config = config
        # Per-iteration state (reset each iteration)
        self._candidate_skills: list[str] = []
        self._loaded_skills: dict[str, int] = {}
        # Per-session accumulators (reset at session boundary)
        self._session_skills_used: set[str] = set()
        # Narrow signal: only skills the agent actually loaded (read_file on SKILL.md)
        # or explicitly rated. Used for composition tracking — candidates (search hits)
        # are excluded to avoid false co-occurrence pairs.
        self._session_skills_loaded: set[str] = set()
        self._session_had_errors: bool = False

    # ------------------------------------------------------------------
    # Per-iteration resets
    # ------------------------------------------------------------------

    def reset_iteration(self) -> None:
        """Clear per-iteration state. Call at the top of before_iteration."""
        self._candidate_skills = []
        self._loaded_skills = {}

    # ------------------------------------------------------------------
    # Recording (called from hook lifecycle methods)
    # ------------------------------------------------------------------

    def record_candidates(self, names: list[str]) -> None:
        """Extend candidate skill list (called by SkillSearchTool)."""
        self._candidate_skills.extend(names)

    def record_read(self, skill_name: str, tool_index: int) -> None:
        """Record a directly-observed read_file on a SKILL.md path."""
        self._loaded_skills[skill_name] = tool_index

    def record_rating(self, name: str) -> None:
        """Register an explicitly-rated skill for trajectory tracking."""
        self._session_skills_used.add(name)
        self._session_skills_loaded.add(name)

    def record_error(self) -> None:
        """Signal that an error occurred this iteration."""
        self._session_had_errors = True

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def update_accumulators(self) -> None:
        """Merge iteration-level observations into session-level accumulators.

        Trajectory tracking only — no DB writes. Stat crediting
        (use_count, success_count) is exclusively handled by SkillRateTool.
        """
        self._session_skills_used.update(self._candidate_skills)
        self._session_skills_used.update(self._loaded_skills.keys())
        self._session_skills_loaded.update(self._loaded_skills.keys())

    def reset_session(self) -> tuple[set[str], set[str], bool]:
        """Return accumulated session data and reset state.

        Returns (skills_used_broad, skills_loaded_narrow, had_errors).
        skills_loaded_narrow contains only skills the agent actually read
        or explicitly rated — suitable for composition co-occurrence tracking.
        skills_used_broad includes search candidates — suitable for trajectory.
        """
        skills = self._session_skills_used
        loaded = self._session_skills_loaded
        errors = self._session_had_errors
        self._session_skills_used = set()
        self._session_skills_loaded = set()
        self._session_had_errors = False
        return skills, loaded, errors

    # ------------------------------------------------------------------
    # Properties for read-only access
    # ------------------------------------------------------------------

    @property
    def session_skills_used(self) -> set[str]:
        return self._session_skills_used

    @property
    def session_had_errors(self) -> bool:
        return self._session_had_errors

    # ------------------------------------------------------------------
    # Path helper
    # ------------------------------------------------------------------

    @staticmethod
    def extract_skill_name_from_path(path: str) -> str | None:
        """Extract skill name from a path ending in skills/<name>/SKILL.md.

        Handles absolute paths, relative paths, and paths with './' prefixes.
        Returns None for any path that doesn't match the expected pattern.
        """
        try:
            parts = Path(path).parts
        except (TypeError, ValueError):
            return None
        for i, part in enumerate(parts):
            if part == "skills" and i + 2 < len(parts) and parts[i + 2] == "SKILL.md":
                return parts[i + 1]
        return None

    # ------------------------------------------------------------------
    # Promotion / deprecation
    # ------------------------------------------------------------------

    def _is_too_similar_to_active(
        self,
        name: str,
        threshold: float,
        active_embeddings: list[np.ndarray],
    ) -> bool:
        """Return True if draft skill embedding is too similar to any active skill.

        *active_embeddings* is pre-fetched by the caller to avoid repeated DB scans.
        """
        row = self._db.execute(
            "SELECT sv.embedding FROM skill_vec sv "
            "JOIN skill_stats ss ON ss.id = sv.skill_id "
            "WHERE ss.name = ?",
            (name,),
        ).fetchone()
        if row is None:
            return False  # not yet indexed — allow promotion, indexer will catch up

        candidate = np.frombuffer(row[0], dtype=np.float32)
        return any(
            float(np.dot(candidate, ae)) >= threshold for ae in active_embeddings
        )

    def check_promotions(self, names: list[str]) -> None:
        """Promote draft skills to active, or deprecate skills with low success."""
        cfg = self._config
        if cfg is None:
            return
        try:
            with self._db:
                # Fetch all active skill embeddings once — reused per candidate.
                active_emb_rows = self._db.execute(
                    "SELECT sv.embedding FROM skill_vec sv "
                    "JOIN skill_stats ss ON ss.id = sv.skill_id "
                    "WHERE ss.status = 'active'",
                ).fetchall()
                active_embeddings = [
                    np.frombuffer(r[0], dtype=np.float32) for r in active_emb_rows
                ]
                for name in names:
                    row = self._db.execute(
                        "SELECT status, use_count, success_count FROM skill_stats WHERE name = ?",
                        (name,),
                    ).fetchone()
                    if not row:
                        continue
                    status, use_count, success_count = row

                    # Promotion: draft -> active after enough successes
                    if status == "draft" and success_count >= cfg.promotion_threshold:
                        threshold = getattr(
                            cfg, "diversity_similarity_threshold", 0.88
                        )
                        if self._is_too_similar_to_active(name, threshold, active_embeddings):
                            log.info(
                                "skill '%s' promotion blocked: too similar to an "
                                "existing active skill (threshold=%.2f)",
                                name,
                                threshold,
                            )
                        else:
                            self._db.execute(
                                "UPDATE skill_stats SET status = 'active' WHERE name = ?",
                                (name,),
                            )
                            log.info(
                                "skill '%s' promoted draft -> active (success_count=%d)",
                                name,
                                success_count,
                            )
                            status = "active"

                    # Deprecation: any non-deprecated skill with chronic low success
                    if (
                        status != "deprecated"
                        and use_count >= cfg.deprecation_min_uses
                        and use_count > 0
                        and success_count / use_count < cfg.deprecation_max_success_rate
                    ):
                        self._db.execute(
                            "UPDATE skill_stats SET status = 'deprecated' WHERE name = ?",
                            (name,),
                        )
                        log.info(
                            "skill '%s' deprecated (success_rate=%.2f after %d uses)",
                            name,
                            success_count / use_count,
                            use_count,
                        )
        except Exception:
            log.exception("skill promotion check failed")
