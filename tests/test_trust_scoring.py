"""Tests for auto trust-scoring of injected cheatsheet/expel lessons."""
from __future__ import annotations

from unittest.mock import patch

import nano_hermes
import pytest
from conftest import _make_loop
from nano_hermes.memory.cheatsheet import _store_lesson, retrieve_lessons


def _hook(tmp_path):
    return nano_hermes.install(_make_loop(tmp_path), config={})


def _trust(hook, fact_id):
    return hook.db.execute(
        "SELECT trust_score FROM semantic_facts WHERE id = ?", (fact_id,)
    ).fetchone()[0]


def _set_trust(hook, fact_id, value):
    hook.db.execute(
        "UPDATE semantic_facts SET trust_score = ? WHERE id = ?", (value, fact_id)
    )
    hook.db.commit()


class TestDefaultTrust:
    def test_new_lesson_is_neutral(self, tmp_path):
        hook = _hook(tmp_path)
        fid = _store_lesson(hook.db, "Lesson about caching.", "cache task")
        assert _trust(hook, fid) == 1.0


class TestApplyLessonTrust:
    def test_success_raises_trust(self, tmp_path):
        hook = _hook(tmp_path)
        fid = _store_lesson(hook.db, "A useful lesson here.", "t")
        hook._injected_lesson_ids = {fid: 1.0}
        hook._apply_lesson_trust("success")
        assert _trust(hook, fid) == pytest.approx(1.05)
        assert hook._injected_lesson_ids == {}  # cleared

    def test_fail_lowers_trust(self, tmp_path):
        hook = _hook(tmp_path)
        fid = _store_lesson(hook.db, "A misleading lesson here.", "t")
        hook._injected_lesson_ids = {fid: 1.0}
        hook._apply_lesson_trust("fail")
        assert _trust(hook, fid) == pytest.approx(0.90)

    def test_clamps_low(self, tmp_path):
        hook = _hook(tmp_path)
        fid = _store_lesson(hook.db, "Repeatedly bad lesson.", "t")
        _set_trust(hook, fid, 0.05)
        hook._injected_lesson_ids = {fid: 1.0}
        hook._apply_lesson_trust("fail")  # 0.05 - 0.10 floored at 0.0
        assert _trust(hook, fid) == 0.0

    def test_clamps_high(self, tmp_path):
        hook = _hook(tmp_path)
        fid = _store_lesson(hook.db, "Repeatedly great lesson.", "t")
        _set_trust(hook, fid, 1.98)
        hook._injected_lesson_ids = {fid: 1.0}
        hook._apply_lesson_trust("success")  # 1.98 + 0.05 capped at 2.0
        assert _trust(hook, fid) == 2.0

    def test_empty_set_noop(self, tmp_path):
        hook = _hook(tmp_path)
        hook._injected_lesson_ids = {}
        hook._apply_lesson_trust("success")  # must not raise

    def test_delta_scaled_by_relevance(self, tmp_path):
        # On a fail, a bullseye lesson (rel 1.0) drops ~5× more than a
        # marginal one (rel 0.2): 1.0*0.10 vs 0.2*0.10.
        hook = _hook(tmp_path)
        near = _store_lesson(hook.db, "Highly relevant lesson text.", "t")
        far = _store_lesson(hook.db, "Barely relevant lesson text.", "t")
        hook._injected_lesson_ids = {near: 1.0, far: 0.2}
        hook._apply_lesson_trust("fail")
        assert _trust(hook, near) == pytest.approx(1.0 - 0.10)
        assert _trust(hook, far) == pytest.approx(1.0 - 0.02)

    def test_relevance_one_matches_uniform(self, tmp_path):
        # Mutation guard: rel=1.0 collapses to the old uniform delta.
        hook = _hook(tmp_path)
        fid = _store_lesson(hook.db, "A lesson at full relevance.", "t")
        hook._injected_lesson_ids = {fid: 1.0}
        hook._apply_lesson_trust("success")
        assert _trust(hook, fid) == pytest.approx(1.05)


class TestTrustFiltersRetrieval:
    async def test_below_floor_hidden(self, tmp_path):
        hook = _hook(tmp_path)
        good = _store_lesson(hook.db, "Trusted lesson stays.", "task")
        bad = _store_lesson(hook.db, "Distrusted lesson hidden.", "task")
        _set_trust(hook, bad, 0.1)  # below default floor 0.3
        with patch.object(hook, "embedder", side_effect=Exception("no provider")):
            lessons = await retrieve_lessons(hook, "task", top_k=5)
        ids = [fid for fid, _, _ in lessons]
        assert good in ids
        assert bad not in ids

    async def test_distrusted_zero_not_ranked_first(self, tmp_path):
        # With floor=0, a trust=0 fact is admitted; it must NOT sort first
        # (regression for NULLIF making distance/0 → NULL → sorts first).
        import numpy as np
        from nano_hermes.memory.cheatsheet import retrieve_lessons as _rl

        hook = nano_hermes.install(
            _make_loop(tmp_path), config={"decay": {"fact_trust_min": 0.0}}
        )
        def _vec(i):
            v = np.zeros(512, dtype=np.float32)
            v[i] = 1.0
            return v
        good = _store_lesson(hook.db, "Good trusted lesson here.", "task")
        bad = _store_lesson(hook.db, "Bad distrusted lesson here.", "task")
        _set_trust(hook, bad, 0.0)
        # good is the query's nearest; bad is far. Under the old NULLIF ordering
        # bad (trust 0 → NULL) would sort FIRST regardless of its distance.
        hook.db.execute("INSERT INTO semantic_facts_vec (fact_id, embedding) VALUES (?, ?)",
                        (good, _vec(0).tobytes()))
        hook.db.execute("INSERT INTO semantic_facts_vec (fact_id, embedding) VALUES (?, ?)",
                        (bad, _vec(1).tobytes()))
        hook.db.commit()

        class _Chain:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def embed(self, texts): return [_vec(0)]
        with patch.object(hook, "embedder", lambda: _Chain()):
            lessons = await _rl(hook, "task", top_k=2)
        assert lessons[0][0] == good  # nearest+trusted first, trust=0 not floated up

    async def test_old_distrusted_fact_evicted(self, tmp_path):
        from nano_hermes.session.db import evict_low_value_facts

        hook = _hook(tmp_path)
        old = 1.0  # ancient created_at
        fid = hook.db.execute(
            "INSERT INTO semantic_facts (content, source_chunk_ids, keywords, tags, "
            "context, importance, fact_type, task_category, created_at, trust_score) "
            "VALUES ('hidden lesson', '[]','[]','[]','c', 6, 'expel', 'cat', ?, 0.1)",
            (old,),
        ).lastrowid
        hook.db.commit()
        # importance 6 > floor 4, so only the trust_floor clause can evict it.
        evict_low_value_facts(
            hook.db, retention_days=30, importance_floor=4,
            superseded_grace_days=14, max_per_run=100, trust_floor=0.3,
        )
        gone = hook.db.execute(
            "SELECT COUNT(*) FROM semantic_facts WHERE id = ?", (fid,)
        ).fetchone()[0]
        assert gone == 0

    async def test_higher_trust_ranks_first_fallback(self, tmp_path):
        hook = _hook(tmp_path)
        lo = _store_lesson(hook.db, "Lower trust lesson.", "task")
        hi = _store_lesson(hook.db, "Higher trust lesson.", "task")
        _set_trust(hook, lo, 0.5)
        _set_trust(hook, hi, 1.8)
        with patch.object(hook, "embedder", side_effect=Exception("no provider")):
            lessons = await retrieve_lessons(hook, "task", top_k=5)
        ids = [fid for fid, _, _ in lessons]
        assert ids[0] == hi  # trust_score DESC in fallback
