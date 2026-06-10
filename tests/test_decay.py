"""Memory decay: recency scoring + low-value fact eviction."""
from __future__ import annotations

import time

import nano_hermes
from nano_hermes.decay import recency_decay
from nano_hermes.session.db import evict_low_value_facts

DAY = 86400.0


def _insert_fact(
    db,
    content: str,
    *,
    age_days: float = 0.0,
    importance: int = 5,
    invalid_age_days: float | None = None,
    now: float | None = None,
) -> int:
    now = now if now is not None else time.time()
    invalid_at = None if invalid_age_days is None else now - invalid_age_days * DAY
    cur = db.execute(
        "INSERT INTO semantic_facts "
        "(content, source_chunk_ids, created_at, keywords, tags, context, importance, invalid_at) "
        "VALUES (?, '[]', ?, '[]', '[]', '', ?, ?)",
        (content, now - age_days * DAY, importance, invalid_at),
    )
    db.commit()
    return int(cur.lastrowid)


def _names(db) -> set[str]:
    return {r[0] for r in db.execute("SELECT content FROM semantic_facts").fetchall()}


class TestRecencyDecay:
    def test_fresh_is_one(self):
        assert recency_decay(0.0, 30.0) == 1.0

    def test_half_life(self):
        assert recency_decay(30.0, 30.0) == 0.5
        assert recency_decay(60.0, 30.0) == 0.25

    def test_negative_age_clamps_to_one(self):
        assert recency_decay(-5.0, 30.0) == 1.0

    def test_nonpositive_half_life_disables(self):
        assert recency_decay(100.0, 0.0) == 1.0
        assert recency_decay(100.0, -1.0) == 1.0


class TestEviction:
    _kw = dict(
        retention_days=90,
        importance_floor=4,
        superseded_grace_days=14,
        max_per_run=500,
    )

    def test_old_low_importance_evicted(self, loop):
        hook = nano_hermes.install(loop)
        _insert_fact(hook.db, "old-trivial", age_days=120, importance=2)
        n = evict_low_value_facts(hook.db, **self._kw)
        assert n == 1
        assert "old-trivial" not in _names(hook.db)

    def test_high_importance_never_evicted(self, loop):
        hook = nano_hermes.install(loop)
        _insert_fact(hook.db, "old-important", age_days=400, importance=9)
        n = evict_low_value_facts(hook.db, **self._kw)
        assert n == 0
        assert "old-important" in _names(hook.db)

    def test_recent_low_importance_kept(self, loop):
        hook = nano_hermes.install(loop)
        _insert_fact(hook.db, "fresh-trivial", age_days=3, importance=1)
        n = evict_low_value_facts(hook.db, **self._kw)
        assert n == 0
        assert "fresh-trivial" in _names(hook.db)

    def test_superseded_past_grace_evicted_within_grace_kept(self, loop):
        hook = nano_hermes.install(loop)
        # invalidated long ago — dead weight, even though high importance.
        _insert_fact(hook.db, "stale-invalid", age_days=30, importance=9, invalid_age_days=30)
        # invalidated just now — still inside the grace window.
        _insert_fact(hook.db, "fresh-invalid", age_days=30, importance=9, invalid_age_days=1)
        n = evict_low_value_facts(hook.db, **self._kw)
        assert n == 1
        names = _names(hook.db)
        assert "stale-invalid" not in names
        assert "fresh-invalid" in names

    def test_cap_limits_deletions(self, loop):
        hook = nano_hermes.install(loop)
        for i in range(5):
            _insert_fact(hook.db, f"junk-{i}", age_days=120, importance=1)
        n = evict_low_value_facts(hook.db, **{**self._kw, "max_per_run": 2})
        assert n == 2
        assert len(_names(hook.db)) == 3

    def test_disabled_retention_skips_valid_facts(self, loop):
        hook = nano_hermes.install(loop)
        _insert_fact(hook.db, "old-trivial", age_days=120, importance=1)
        n = evict_low_value_facts(hook.db, **{**self._kw, "retention_days": 0})
        assert n == 0
        assert "old-trivial" in _names(hook.db)

    def test_vec_row_removed_by_trigger(self, loop):
        """Deleting a fact must cascade to semantic_facts_vec via the AD trigger."""
        import numpy as np
        hook = nano_hermes.install(loop)
        fid = _insert_fact(hook.db, "old-trivial", age_days=120, importance=1)
        hook.db.execute(
            "INSERT INTO semantic_facts_vec (fact_id, embedding) VALUES (?, ?)",
            (fid, np.ones(512, dtype=np.float32).tobytes()),
        )
        hook.db.commit()
        evict_low_value_facts(hook.db, **self._kw)
        remaining = hook.db.execute(
            "SELECT COUNT(*) FROM semantic_facts_vec WHERE fact_id = ?", (fid,)
        ).fetchone()[0]
        assert remaining == 0
