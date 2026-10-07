"""SkillUsageTracker — extracted from NanoHermesHook.

Manages per-iteration and per-session skill state: which skills the agent
searched for (candidates), which it actually loaded (observed reads), and
which were explicitly rated. Also owns the promotion/deprecation lifecycle.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from ..config import SkillStatsConfig

from ..lru import touch
from .session import conversation_key

log = logging.getLogger(__name__)


@dataclass
class _SkillState:
    """One conversation's skill observations.

    Per conversation because the session boundary that drains them is: with
    shared fields, one chat's /new reset another's tracking and wrote that
    chat's skills into its own trajectory and composition record.
    """

    # Per-iteration (reset each iteration)
    candidate_skills: list[str] = field(default_factory=list)
    loaded_skills: dict[str, int] = field(default_factory=dict)
    # Per-session accumulators (reset at the session boundary)
    session_skills_used: set[str] = field(default_factory=set)
    # Narrow signal: only skills the agent actually loaded (read_file on
    # SKILL.md) or explicitly rated. Used for composition tracking — search
    # hits are excluded to avoid false co-occurrence pairs.
    session_skills_loaded: set[str] = field(default_factory=set)
    session_had_errors: bool = False


class SkillUsageTracker:
    def __init__(
        self,
        *,
        db: sqlite3.Connection,
        config: "SkillStatsConfig | None",
    ) -> None:
        self._db = db
        self._config = config
        self._states: dict[str, _SkillState] = {}

    def _state(self, key: str | None = None) -> _SkillState:
        """*key*'s state; ``None`` resolves the calling turn's conversation
        (a tool) via nanobot's request contextvar."""
        key = conversation_key(key)
        touch(self._states, key, _SkillState)
        return self._states[key]

    # ------------------------------------------------------------------
    # Per-iteration resets
    # ------------------------------------------------------------------

    def reset_iteration(self, key: str | None = None) -> None:
        """Clear per-iteration state. Call at the top of before_iteration."""
        st = self._state(key)
        st.candidate_skills = []
        st.loaded_skills = {}

    # ------------------------------------------------------------------
    # Recording (called from hook lifecycle methods)
    # ------------------------------------------------------------------

    def record_candidates(self, names: list[str], key: str | None = None) -> None:
        """Extend candidate skill list (called by SkillSearchTool)."""
        self._state(key).candidate_skills.extend(names)

    def record_read(self, skill_name: str, tool_index: int, key: str | None = None) -> None:
        """Record a directly-observed read_file on a SKILL.md path."""
        self._state(key).loaded_skills[skill_name] = tool_index

    def record_rating(self, name: str, key: str | None = None) -> None:
        """Register an explicitly-rated skill for trajectory tracking."""
        st = self._state(key)
        st.session_skills_used.add(name)
        st.session_skills_loaded.add(name)

    def record_error(self, key: str | None = None) -> None:
        """Signal that an error occurred this iteration."""
        self._state(key).session_had_errors = True

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def update_accumulators(self, key: str | None = None) -> None:
        """Merge iteration-level observations into session-level accumulators.

        Trajectory tracking only — no DB writes. Stat crediting
        (use_count, success_count) is exclusively handled by SkillRateTool.
        """
        st = self._state(key)
        st.session_skills_used.update(st.candidate_skills)
        st.session_skills_used.update(st.loaded_skills.keys())
        st.session_skills_loaded.update(st.loaded_skills.keys())

    def reset_session(self, key: str | None = None) -> tuple[set[str], set[str], bool]:
        """Return *key*'s accumulated session data and reset it.

        Returns (skills_used_broad, skills_loaded_narrow, had_errors).
        skills_loaded_narrow contains only skills the agent actually read
        or explicitly rated — suitable for composition co-occurrence tracking.
        skills_used_broad includes search candidates — suitable for trajectory.
        """
        st = self._state(key)
        out = (st.session_skills_used, st.session_skills_loaded, st.session_had_errors)
        st.session_skills_used = set()
        st.session_skills_loaded = set()
        st.session_had_errors = False
        return out

    # ------------------------------------------------------------------
    # The calling conversation's fields under their old names
    # ------------------------------------------------------------------

    @property
    def session_skills_used(self) -> set[str]:
        return self._state().session_skills_used

    @property
    def session_had_errors(self) -> bool:
        return self._state().session_had_errors

    @property
    def _candidate_skills(self):
        return self._state().candidate_skills

    @_candidate_skills.setter
    def _candidate_skills(self, value) -> None:
        self._state().candidate_skills = value

    @property
    def _loaded_skills(self):
        return self._state().loaded_skills

    @_loaded_skills.setter
    def _loaded_skills(self, value) -> None:
        self._state().loaded_skills = value

    @property
    def _session_skills_used(self):
        return self._state().session_skills_used

    @_session_skills_used.setter
    def _session_skills_used(self, value) -> None:
        self._state().session_skills_used = value

    @property
    def _session_skills_loaded(self):
        return self._state().session_skills_loaded

    @_session_skills_loaded.setter
    def _session_skills_loaded(self, value) -> None:
        self._state().session_skills_loaded = value

    @property
    def _session_had_errors(self):
        return self._state().session_had_errors

    @_session_had_errors.setter
    def _session_had_errors(self, value) -> None:
        self._state().session_had_errors = value

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
        active_embeddings: "list[np.ndarray]",
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

        import numpy as np  # noqa: PLC0415
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
                import numpy as np  # noqa: PLC0415
                active_embeddings = [
                    np.frombuffer(r[0], dtype=np.float32) for r in active_emb_rows
                ]
                for name in names:
                    row = self._db.execute(
                        "SELECT status, use_count, success_count, origin, pinned "
                        "FROM skill_stats WHERE name = ?",
                        (name,),
                    ).fetchone()
                    if not row:
                        continue
                    status, use_count, success_count, origin, pinned = row

                    # Reactivation: a stale skill used again returns to active.
                    # The curator can re-stale it later if it lapses once more.
                    if status == "stale":
                        self._db.execute(
                            "UPDATE skill_stats SET status = 'active' WHERE name = ?",
                            (name,),
                        )
                        log.info("skill '%s' reactivated (stale -> active)", name)
                        status = "active"

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

                    # Deprecation: agent-authored, unpinned skills with chronic
                    # low success. Builtin/external/hand-authored ('user') and
                    # pinned skills are exempt from automatic deprecation.
                    if (
                        status != "deprecated"
                        and origin == "agent"
                        and not pinned
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
