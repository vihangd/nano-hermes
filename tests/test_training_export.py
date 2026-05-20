"""Tests for Phase 4.4: offline training data export (skill_export tool)."""
from __future__ import annotations

import json
import time
from pathlib import Path
import nano_hermes
from conftest import _make_loop
from nano_hermes.skills.training_export import (
    count_skill_sessions,
    export_mature_skills,
    export_skill_training_data,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_session(db, session_key: str) -> int:
    cur = db.execute(
        "INSERT INTO sessions (session_key, started_at) VALUES (?, ?)",
        (session_key, time.time()),
    )
    db.commit()
    return int(cur.lastrowid)


def _insert_trajectory(db, session_id: int, outcome: str = "ok", skills: list[str] | None = None) -> int:
    skills_used = json.dumps(skills or [])
    cur = db.execute(
        "INSERT INTO trajectories (session_id, task, skills_used, outcome, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (session_id, f"task for session {session_id}", skills_used, outcome, time.time()),
    )
    db.commit()
    return int(cur.lastrowid)


def _insert_chunk(db, session_id: int, content: str) -> int:
    cur = db.execute(
        "INSERT INTO chunks (session_id, turn_index, role, content, created_at) "
        "VALUES (?, 0, 'user', ?, ?)",
        (session_id, content, time.time()),
    )
    db.commit()
    return int(cur.lastrowid)


def _insert_active_skill(db, name: str) -> None:
    db.execute(
        "INSERT OR REPLACE INTO skill_stats (name, status, content_hash, indexed_at) "
        "VALUES (?, 'active', 'abc123', ?)",
        (name, time.time()),
    )
    db.commit()


def _make_hook(tmp_path, config_overrides=None):
    loop = _make_loop(tmp_path)
    hook = nano_hermes.install(loop, config=config_overrides or {})
    return hook


# ---------------------------------------------------------------------------
# Unit tests: count_skill_sessions
# ---------------------------------------------------------------------------

class TestCountSkillSessions:
    def _make_db(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        return hook.db

    def test_empty_corpus_returns_zero(self, tmp_path):
        db = self._make_db(tmp_path)
        assert count_skill_sessions(db, "my_skill") == 0

    def test_counts_distinct_sessions(self, tmp_path):
        db = self._make_db(tmp_path)
        for i in range(3):
            sid = _insert_session(db, f"s:{i}")
            _insert_trajectory(db, sid, skills=["my_skill"])
        assert count_skill_sessions(db, "my_skill") == 3

    def test_multiple_trajectories_per_session_count_once(self, tmp_path):
        db = self._make_db(tmp_path)
        sid = _insert_session(db, "s:1")
        _insert_trajectory(db, sid, skills=["my_skill"])
        _insert_trajectory(db, sid, skills=["my_skill"])
        assert count_skill_sessions(db, "my_skill") == 1

    def test_other_skill_not_counted(self, tmp_path):
        db = self._make_db(tmp_path)
        sid = _insert_session(db, "s:1")
        _insert_trajectory(db, sid, skills=["other_skill"])
        assert count_skill_sessions(db, "my_skill") == 0


# ---------------------------------------------------------------------------
# Unit tests: export_skill_training_data
# ---------------------------------------------------------------------------

class TestExportSkillTrainingData:
    def _setup(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        return hook.db, tmp_path

    def test_returns_one_row_per_trajectory(self, tmp_path):
        db, ws = self._setup(tmp_path)
        for i in range(3):
            sid = _insert_session(db, f"s:{i}")
            _insert_trajectory(db, sid, skills=["fetch_data"])
            _insert_chunk(db, sid, f"chunk content {i}")

        rows = export_skill_training_data(db, Path(ws), "fetch_data")
        assert len(rows) == 3

    def test_row_structure(self, tmp_path):
        db, ws = self._setup(tmp_path)
        sid = _insert_session(db, "s:1")
        _insert_trajectory(db, sid, outcome="ok", skills=["fetch_data"])
        _insert_chunk(db, sid, "some content")

        rows = export_skill_training_data(db, Path(ws), "fetch_data")
        assert len(rows) == 1
        row = rows[0]
        assert row["skill_name"] == "fetch_data"
        assert row["session_id"] == sid
        assert row["outcome"] == "ok"
        assert isinstance(row["chunks"], list)
        assert isinstance(row["skill_text"], str)

    def test_chunks_capped_at_limit(self, tmp_path):
        db, ws = self._setup(tmp_path)
        sid = _insert_session(db, "s:1")
        _insert_trajectory(db, sid, skills=["fetch_data"])
        for i in range(10):
            _insert_chunk(db, sid, f"chunk {i}")

        rows = export_skill_training_data(db, Path(ws), "fetch_data", chunk_limit=3)
        assert len(rows[0]["chunks"]) <= 3

    def test_skill_text_from_workspace_file(self, tmp_path):
        db, ws = self._setup(tmp_path)
        skill_dir = Path(ws) / "skills" / "my_skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# my_skill\nDoes something.", encoding="utf-8")

        sid = _insert_session(db, "s:1")
        _insert_trajectory(db, sid, skills=["my_skill"])

        rows = export_skill_training_data(db, Path(ws), "my_skill")
        assert "Does something" in rows[0]["skill_text"]

    def test_missing_workspace_skill_uses_sentinel(self, tmp_path):
        db, ws = self._setup(tmp_path)
        sid = _insert_session(db, "s:1")
        _insert_trajectory(db, sid, skills=["ghost_skill"])

        rows = export_skill_training_data(db, Path(ws), "ghost_skill")
        assert "ghost_skill" in rows[0]["skill_text"]

    def test_no_trajectories_returns_empty(self, tmp_path):
        db, ws = self._setup(tmp_path)
        rows = export_skill_training_data(db, Path(ws), "unused_skill")
        assert rows == []


# ---------------------------------------------------------------------------
# Unit tests: export_mature_skills
# ---------------------------------------------------------------------------

class TestExportMatureSkills:
    def _setup(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        return hook.db, Path(tmp_path)

    def test_below_threshold_not_exported(self, tmp_path):
        db, ws = self._setup(tmp_path)
        _insert_active_skill(db, "small_skill")
        for i in range(3):
            sid = _insert_session(db, f"s:{i}")
            _insert_trajectory(db, sid, skills=["small_skill"])

        exports_dir = ws / "nano_hermes" / "exports"
        written = export_mature_skills(db, ws, exports_dir, min_sessions=50)
        assert written == []

    def test_mature_skill_exported(self, tmp_path):
        db, ws = self._setup(tmp_path)
        _insert_active_skill(db, "big_skill")
        for i in range(5):
            sid = _insert_session(db, f"s:{i}")
            _insert_trajectory(db, sid, skills=["big_skill"])
            _insert_chunk(db, sid, "chunk content")

        exports_dir = ws / "nano_hermes" / "exports"
        written = export_mature_skills(db, ws, exports_dir, min_sessions=3)
        assert len(written) == 1
        assert "big_skill" in written[0]

    def test_written_file_is_valid_jsonl(self, tmp_path):
        db, ws = self._setup(tmp_path)
        _insert_active_skill(db, "my_skill")
        for i in range(3):
            sid = _insert_session(db, f"s:{i}")
            _insert_trajectory(db, sid, skills=["my_skill"])
            _insert_chunk(db, sid, f"content {i}")

        exports_dir = ws / "nano_hermes" / "exports"
        written = export_mature_skills(db, ws, exports_dir, min_sessions=2)
        assert len(written) == 1

        lines = Path(written[0]).read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        for line in lines:
            obj = json.loads(line)
            assert "skill_name" in obj
            assert "outcome" in obj
            assert "chunks" in obj

    def test_exports_dir_created_if_missing(self, tmp_path):
        db, ws = self._setup(tmp_path)
        _insert_active_skill(db, "s")
        for i in range(3):
            sid = _insert_session(db, f"s:{i}")
            _insert_trajectory(db, sid, skills=["s"])

        exports_dir = ws / "nano_hermes" / "exports" / "deep" / "nested"
        export_mature_skills(db, ws, exports_dir, min_sessions=2)
        assert exports_dir.exists()


# ---------------------------------------------------------------------------
# Integration tests: skill_export tool
# ---------------------------------------------------------------------------

class TestSkillExportTool:
    def _make_hook(self, tmp_path, config_overrides=None):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop, config=config_overrides or {})
        return hook

    async def test_below_threshold_returns_guidance(self, tmp_path):
        hook = self._make_hook(tmp_path)
        _insert_active_skill(hook.db, "sparse_skill")
        sid = _insert_session(hook.db, "s:1")
        _insert_trajectory(hook.db, sid, skills=["sparse_skill"])

        tool = hook._loop.tools.get("skill_export")
        result = await tool.execute(skill_name="sparse_skill", min_sessions=50)

        assert "only 1" in result or "1 distinct session" in result or "1 session" in result
        assert "50" in result

    async def test_mature_skill_writes_file(self, tmp_path):
        hook = self._make_hook(tmp_path)
        _insert_active_skill(hook.db, "good_skill")
        for i in range(3):
            sid = _insert_session(hook.db, f"s:{i}")
            _insert_trajectory(hook.db, sid, outcome="ok", skills=["good_skill"])
            _insert_chunk(hook.db, sid, f"content {i}")

        tool = hook._loop.tools.get("skill_export")
        result = await tool.execute(skill_name="good_skill", min_sessions=2)

        assert "good_skill" in result
        assert "3 training row" in result

        exports_dir = Path(hook._loop.workspace) / "nano_hermes" / "exports"
        files = list(exports_dir.glob("good_skill.*.jsonl"))
        assert len(files) == 1

    async def test_all_no_mature_skills(self, tmp_path):
        hook = self._make_hook(tmp_path)
        tool = hook._loop.tools.get("skill_export")
        result = await tool.execute(skill_name="all")
        assert "no active skills" in result.lower() or "0" in result

    async def test_all_exports_only_mature(self, tmp_path):
        hook = self._make_hook(tmp_path)
        _insert_active_skill(hook.db, "mature")
        _insert_active_skill(hook.db, "sparse")

        for i in range(5):
            sid = _insert_session(hook.db, f"m:{i}")
            _insert_trajectory(hook.db, sid, skills=["mature"])
            _insert_chunk(hook.db, sid, "data")

        sid = _insert_session(hook.db, "sp:1")
        _insert_trajectory(hook.db, sid, skills=["sparse"])

        tool = hook._loop.tools.get("skill_export")
        result = await tool.execute(skill_name="all", min_sessions=3)

        assert "1 skill" in result
        exports_dir = Path(hook._loop.workspace) / "nano_hermes" / "exports"
        files = list(exports_dir.glob("*.jsonl"))
        assert len(files) == 1
        assert "mature" in files[0].name

    async def test_multi_outcome_rows(self, tmp_path):
        hook = self._make_hook(tmp_path)
        for i, outcome in enumerate(["ok", "fail", "partial"]):
            sid = _insert_session(hook.db, f"s:{i}")
            _insert_trajectory(hook.db, sid, outcome=outcome, skills=["mix_skill"])

        tool = hook._loop.tools.get("skill_export")
        result = await tool.execute(skill_name="mix_skill", min_sessions=2)

        # All three outcomes present in result
        assert "ok" in result or "fail" in result or "partial" in result

        exports_dir = Path(hook._loop.workspace) / "nano_hermes" / "exports"
        files = list(exports_dir.glob("mix_skill.*.jsonl"))
        assert len(files) == 1
        lines = Path(files[0]).read_text(encoding="utf-8").splitlines()
        outcomes = {json.loads(line)["outcome"] for line in lines}
        assert outcomes == {"ok", "fail", "partial"}
