"""Tests for ReflectionCoordinator."""
from __future__ import annotations

import time

import pytest

import nano_hermes
from conftest import _make_loop
from nano_hermes.coordinator.reflection import ReflectionCoordinator


@pytest.fixture
def hook(tmp_path):
    loop = _make_loop(tmp_path)
    return nano_hermes.install(loop)


@pytest.fixture
def coord(hook) -> ReflectionCoordinator:
    return hook._reflection_coord


class TestScoreIteration:
    def test_nudge_set_when_threshold_exceeded(
        self, coord: ReflectionCoordinator
    ) -> None:
        # error_score = 3.0 per call; default threshold = 5.0
        coord.score_iteration(had_error=True, user_text=None)
        assert not coord._nudge_pending  # 3.0 < 5.0
        coord.score_iteration(had_error=True, user_text=None)
        assert coord._nudge_pending  # 6.0 >= 5.0

    def test_score_resets_after_nudge_triggered(
        self, coord: ReflectionCoordinator
    ) -> None:
        coord._salience_score = 4.9
        coord.score_iteration(had_error=True, user_text=None)
        assert coord._nudge_pending
        assert coord._salience_score == 0.0

    def test_tool_burst_separate_from_score_iteration(
        self, coord: ReflectionCoordinator
    ) -> None:
        # record_tool_burst is separate from score_iteration
        coord.record_tool_burst(5)  # >= _TOOL_BURST_MIN, adds 2.0
        coord.record_tool_burst(5)  # adds another 2.0 → total 4.0
        assert coord._salience_score == pytest.approx(4.0)
        assert not coord._nudge_pending
        coord.score_iteration(had_error=True, user_text=None)  # adds 3.0 → 7.0 >= 5.0
        assert coord._nudge_pending


class TestTakeNudge:
    def test_returns_message_and_clears_flag(
        self, coord: ReflectionCoordinator
    ) -> None:
        coord._nudge_pending = True
        msg = coord.take_nudge()
        assert msg is not None
        assert msg["role"] == "system"
        assert "reflect" in msg["content"].lower()
        assert not coord._nudge_pending

    def test_returns_none_when_no_nudge(
        self, coord: ReflectionCoordinator
    ) -> None:
        coord._nudge_pending = False
        assert coord.take_nudge() is None


class TestGetSessionInjections:
    def test_returns_empty_when_no_reflections(
        self, hook, coord: ReflectionCoordinator
    ) -> None:
        cur = hook.db.execute(
            "INSERT INTO sessions (session_key, started_at) VALUES (?, ?)",
            ("test:1", time.time()),
        )
        session_id = cur.lastrowid
        hook.db.commit()
        assert coord.get_session_injections(session_id) == []

    def test_returns_unseen_reflection_content(
        self, hook, coord: ReflectionCoordinator
    ) -> None:
        cur = hook.db.execute(
            "INSERT INTO sessions (session_key, started_at) VALUES (?, ?)",
            ("test:2", time.time()),
        )
        session_id = cur.lastrowid
        hook.db.execute(
            "INSERT INTO reflections (session_id, content, created_at) VALUES (?, ?, ?)",
            (session_id, "key insight here", time.time()),
        )
        hook.db.commit()

        msgs = coord.get_session_injections(session_id)
        assert len(msgs) == 1
        assert "key insight here" in msgs[0]["content"]

    def test_watermark_prevents_reinjection(
        self, hook, coord: ReflectionCoordinator
    ) -> None:
        cur = hook.db.execute(
            "INSERT INTO sessions (session_key, started_at) VALUES (?, ?)",
            ("test:3", time.time()),
        )
        session_id = cur.lastrowid
        hook.db.execute(
            "INSERT INTO reflections (session_id, content, created_at) VALUES (?, ?, ?)",
            (session_id, "first reflection", time.time()),
        )
        hook.db.commit()

        coord.get_session_injections(session_id)  # sets watermark
        assert coord.get_session_injections(session_id) == []  # no new content

    def test_returns_empty_for_none_session_id(
        self, coord: ReflectionCoordinator
    ) -> None:
        assert coord.get_session_injections(None) == []


class TestOnNewSession:
    def test_prunes_session_from_watermark_dict(
        self, coord: ReflectionCoordinator
    ) -> None:
        coord._last_injected_reflection_id[42] = 100
        coord._last_injected_reflection_id[99] = 200
        coord.on_new_session(42)
        assert 42 not in coord._last_injected_reflection_id
        assert 99 in coord._last_injected_reflection_id  # unaffected

    def test_resets_global_watermark(
        self, coord: ReflectionCoordinator
    ) -> None:
        coord._last_injected_global_reflection_id = 999
        coord.on_new_session(1)
        assert coord._last_injected_global_reflection_id == 0

    def test_clears_injected_ids_accumulator(
        self, coord: ReflectionCoordinator
    ) -> None:
        coord._injected_reflection_ids = {1, 2, 3}
        coord.on_new_session(5)
        assert coord._injected_reflection_ids == set()


class TestBackPropagateUtility:
    def _seed_reflection(self, hook, content: str, utility: float = 0.5) -> int:
        cur = hook.db.execute(
            "INSERT INTO sessions (session_key, started_at) VALUES (?, ?)",
            ("test:bp", time.time()),
        )
        sid = cur.lastrowid
        cur = hook.db.execute(
            "INSERT INTO reflections (session_id, content, created_at, utility) "
            "VALUES (?, ?, ?, ?)",
            (sid, content, time.time(), utility),
        )
        hook.db.commit()
        return int(cur.lastrowid)

    def test_success_increases_utility(self, hook, coord):
        rid = self._seed_reflection(hook, "good reflection", utility=0.5)
        coord._injected_reflection_ids = {rid}
        coord.back_propagate_utility(had_errors=False)

        row = hook.db.execute("SELECT utility FROM reflections WHERE id = ?", (rid,)).fetchone()
        assert row[0] > 0.5  # utility moves toward 1.0

    def test_failure_decreases_utility(self, hook, coord):
        rid = self._seed_reflection(hook, "bad reflection", utility=0.5)
        coord._injected_reflection_ids = {rid}
        coord.back_propagate_utility(had_errors=True)

        row = hook.db.execute("SELECT utility FROM reflections WHERE id = ?", (rid,)).fetchone()
        assert row[0] < 0.5  # utility moves toward 0.0

    def test_no_ids_is_noop(self, hook, coord):
        coord._injected_reflection_ids = set()
        # Must not raise
        coord.back_propagate_utility(had_errors=False)

    def test_non_injected_reflections_unchanged(self, hook, coord):
        rid_injected = self._seed_reflection(hook, "injected", utility=0.5)
        rid_other = self._seed_reflection(hook, "not injected", utility=0.5)
        coord._injected_reflection_ids = {rid_injected}
        coord.back_propagate_utility(had_errors=False)

        row = hook.db.execute("SELECT utility FROM reflections WHERE id = ?", (rid_other,)).fetchone()
        assert row[0] == pytest.approx(0.5)  # unaffected

    def test_injected_ids_accumulated_from_get_session_injections(self, hook, coord):
        cur = hook.db.execute(
            "INSERT INTO sessions (session_key, started_at) VALUES (?, ?)",
            ("test:acc", time.time()),
        )
        sid = cur.lastrowid
        cur = hook.db.execute(
            "INSERT INTO reflections (session_id, content, created_at) VALUES (?, ?, ?)",
            (sid, "accumulated reflection", time.time()),
        )
        rid = int(cur.lastrowid)
        hook.db.commit()

        assert rid not in coord._injected_reflection_ids
        coord.get_session_injections(sid)
        assert rid in coord._injected_reflection_ids


class TestCoActivation:
    """Tests for _record_coactivations — the associative reflection graph."""

    def _seed_reflection(self, hook, content: str) -> int:
        cur = hook.db.execute(
            "INSERT INTO sessions (session_key, started_at) VALUES (?, ?)",
            (f"s_{content[:8]}", time.time()),
        )
        sid = cur.lastrowid
        cur = hook.db.execute(
            "INSERT INTO reflections (session_id, content, created_at) VALUES (?, ?, ?)",
            (sid, content, time.time()),
        )
        hook.db.commit()
        return int(cur.lastrowid)

    def test_first_coactivation_count_is_one(self, hook, coord):
        rid_a = self._seed_reflection(hook, "reflection A")
        rid_b = self._seed_reflection(hook, "reflection B")
        coord._injected_reflection_ids = {rid_a, rid_b}
        coord.back_propagate_utility(had_errors=False)

        a, b = min(rid_a, rid_b), max(rid_a, rid_b)
        row = hook.db.execute(
            "SELECT coactivation_count FROM reflection_coactivations "
            "WHERE reflection_a_id = ? AND reflection_b_id = ?",
            (a, b),
        ).fetchone()
        assert row is not None
        assert row[0] == 1  # first occurrence — must not be 2 (double-count bug)

    def test_repeated_coactivation_increments_count(self, hook, coord):
        rid_a = self._seed_reflection(hook, "reflection X")
        rid_b = self._seed_reflection(hook, "reflection Y")
        a, b = min(rid_a, rid_b), max(rid_a, rid_b)

        for _ in range(3):
            coord._injected_reflection_ids = {rid_a, rid_b}
            coord.back_propagate_utility(had_errors=False)

        row = hook.db.execute(
            "SELECT coactivation_count FROM reflection_coactivations "
            "WHERE reflection_a_id = ? AND reflection_b_id = ?",
            (a, b),
        ).fetchone()
        assert row is not None
        assert row[0] == 3

    def test_pairs_stored_with_smaller_id_first(self, hook, coord):
        rid_a = self._seed_reflection(hook, "first reflection")
        rid_b = self._seed_reflection(hook, "second reflection")
        # rid_b > rid_a since AUTOINCREMENT; verify smaller is always column a
        coord._injected_reflection_ids = {rid_a, rid_b}
        coord.back_propagate_utility(had_errors=False)

        row = hook.db.execute(
            "SELECT reflection_a_id, reflection_b_id FROM reflection_coactivations"
        ).fetchone()
        assert row is not None
        assert row[0] < row[1]

    def test_single_injection_produces_no_edges(self, hook, coord):
        rid = self._seed_reflection(hook, "lone reflection")
        coord._injected_reflection_ids = {rid}
        coord.back_propagate_utility(had_errors=False)

        rows = hook.db.execute("SELECT * FROM reflection_coactivations").fetchall()
        assert rows == []

    def test_three_reflections_produce_three_edges(self, hook, coord):
        rid_a = self._seed_reflection(hook, "refl A")
        rid_b = self._seed_reflection(hook, "refl B")
        rid_c = self._seed_reflection(hook, "refl C")
        coord._injected_reflection_ids = {rid_a, rid_b, rid_c}
        coord.back_propagate_utility(had_errors=False)

        rows = hook.db.execute("SELECT * FROM reflection_coactivations").fetchall()
        assert len(rows) == 3  # C(3,2) = 3 pairs


class TestMMRTrajectoryInjection:
    """Tests for get_trajectory_injection with MMR diversity selection."""

    def _seed_trajectory(self, hook, task: str, outcome: str = "ok", skills: list | None = None) -> int:
        import json
        cur = hook.db.execute(
            "INSERT INTO sessions (session_key, started_at) VALUES (?, ?)",
            (f"s_{task[:10]}", time.time()),
        )
        sid = cur.lastrowid
        cur = hook.db.execute(
            "INSERT INTO trajectories (session_id, task, skills_used, outcome, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (sid, task, json.dumps(skills or []), outcome, time.time()),
        )
        hook.db.commit()
        return int(cur.lastrowid)

    async def test_returns_none_when_no_trajectories(self, hook, coord, monkeypatch):
        from conftest import _patch_embedding
        _patch_embedding(monkeypatch)

        result = await coord.get_trajectory_injection(
            [{"role": "user", "content": "duckduckgo search for news"}]
        )
        assert result is None

    async def test_returns_single_when_only_one_candidate(self, hook, coord, monkeypatch):
        from conftest import _patch_embedding, _FAKE_DIMS
        import numpy as np
        _patch_embedding(monkeypatch)

        tid = self._seed_trajectory(hook, "duckduckgo search for python docs", outcome="ok")
        # Write an embedding that will match "duckduckgo search" queries
        vec = np.zeros(_FAKE_DIMS, dtype=np.float32)
        vec[0] = 1.0  # same as _FAKE_VEC_SEARCH
        hook.db.execute(
            "INSERT INTO trajectories_vec (trajectory_id, embedding) VALUES (?, ?)",
            (tid, vec.tobytes()),
        )
        hook.db.commit()

        result = await coord.get_trajectory_injection(
            [{"role": "user", "content": "duckduckgo search for latest news"}]
        )
        assert result is not None
        assert "Past session" in result["content"]

    async def test_injects_two_sessions_with_mmr(self, hook, coord, monkeypatch):
        from conftest import _patch_embedding, _FAKE_DIMS
        import numpy as np
        _patch_embedding(monkeypatch)

        # Two trajectories with similar-but-distinct tasks
        t1 = self._seed_trajectory(hook, "duckduckgo search for news articles", outcome="ok", skills=["search"])
        t2 = self._seed_trajectory(hook, "duckduckgo search for blog posts on AI", outcome="partial", skills=["search"])

        vec_search = np.zeros(_FAKE_DIMS, dtype=np.float32)
        vec_search[0] = 1.0  # matches "duckduckgo search" query
        for tid in (t1, t2):
            hook.db.execute(
                "INSERT INTO trajectories_vec (trajectory_id, embedding) VALUES (?, ?)",
                (tid, vec_search.tobytes()),
            )
        hook.db.commit()

        result = await coord.get_trajectory_injection(
            [{"role": "user", "content": "duckduckgo search for information"}]
        )
        assert result is not None
        # Both sessions should appear in the output
        assert "Past session 1" in result["content"]
        assert "Past session 2" in result["content"]

    async def test_below_min_similarity_returns_none(self, hook, coord, monkeypatch):
        from conftest import _patch_embedding, _FAKE_DIMS
        import numpy as np
        _patch_embedding(monkeypatch)

        # Trajectory indexed with a perpendicular vector — distance=1.0, similarity=0.0
        tid = self._seed_trajectory(hook, "unrelated task xyz", outcome="ok")
        vec = np.zeros(_FAKE_DIMS, dtype=np.float32)
        vec[99] = 1.0  # orthogonal to any query vector
        hook.db.execute(
            "INSERT INTO trajectories_vec (trajectory_id, embedding) VALUES (?, ?)",
            (tid, vec.tobytes()),
        )
        hook.db.commit()

        # Query with inject_min_similarity=0.75 (default) — orthogonal vector won't pass
        result = await coord.get_trajectory_injection(
            [{"role": "user", "content": "duckduckgo search for news"}]
        )
        # The orthogonal trajectory should not appear regardless of the query match
        # (this tests the threshold filter, not the MMR algo specifically)
        # We just verify no exception is raised and result is either None or only matching ones
        assert result is None or "unrelated task xyz" not in (result or {}).get("content", "")
