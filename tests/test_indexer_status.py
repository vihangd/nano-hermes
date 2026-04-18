"""Tests for SkillIndexer insert status: builtin → active, workspace → draft."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from conftest import _make_loop

import nano_hermes
from nano_hermes.skills.indexer import SkillIndexer


def _fake_entry(name: str, source: str, tmp_path) -> dict:
    skill_dir = tmp_path / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: A {source} skill\n---\n\nBody.\n",
        encoding="utf-8",
    )
    return {"name": name, "path": str(skill_dir / "SKILL.md"), "source": source}


class TestIndexerInsertStatus:
    def _make_indexer(self, loop, entries, descriptions):
        hook = nano_hermes.install(loop)

        skills_loader = MagicMock()
        skills_loader.list_skills.return_value = entries
        skills_loader.get_skill_metadata.side_effect = lambda name: {
            "description": descriptions.get(name, "")
        }

        from nano_hermes.embedding.chain import EmbeddingChain

        vec = np.zeros(512, dtype=np.float32)
        vec[0] = 1.0

        async def fake_embed(self_inner, texts):
            return [vec.copy() for _ in texts]

        def embedder_factory():
            chain = MagicMock(spec=EmbeddingChain)
            chain.__aenter__ = AsyncMock(return_value=chain)
            chain.__aexit__ = AsyncMock(return_value=False)
            chain.embed = AsyncMock(side_effect=lambda texts: [vec.copy() for _ in texts])
            return chain

        return SkillIndexer(
            db=hook.db,
            skills_loader=skills_loader,
            embedder_factory=embedder_factory,
            stats_config=hook.config.skill_stats if hook.config else None,
        ), hook

    @pytest.mark.asyncio
    async def test_builtin_skill_inserted_as_active(self, tmp_path):
        loop = _make_loop(tmp_path)
        entries = [_fake_entry("my-builtin", "builtin", tmp_path)]
        indexer, hook = self._make_indexer(loop, entries, {"my-builtin": "A builtin skill"})

        await indexer.refresh()

        row = hook.db.execute(
            "SELECT status FROM skill_stats WHERE name = ?", ("my-builtin",)
        ).fetchone()
        assert row is not None
        assert row[0] == "active"

    @pytest.mark.asyncio
    async def test_workspace_skill_inserted_as_draft(self, tmp_path):
        loop = _make_loop(tmp_path)
        entries = [_fake_entry("my-workspace", "workspace", tmp_path)]
        indexer, hook = self._make_indexer(loop, entries, {"my-workspace": "A workspace skill"})

        await indexer.refresh()

        row = hook.db.execute(
            "SELECT status FROM skill_stats WHERE name = ?", ("my-workspace",)
        ).fetchone()
        assert row is not None
        assert row[0] == "draft"

    @pytest.mark.asyncio
    async def test_update_preserves_existing_status(self, tmp_path):
        """Re-indexing a skill with changed description must not reset its status."""
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)

        # Pre-seed the DB row as 'active' (simulates a promoted skill).
        with hook.db:
            hook.db.execute(
                "INSERT INTO skill_stats (name, status, use_count, success_count) "
                "VALUES (?, 'active', 5, 5)",
                ("promoted",),
            )

        # Now index it as a workspace skill with a *different* description
        # so the content hash is stale and a re-index fires.
        entries = [_fake_entry("promoted", "workspace", tmp_path)]
        descriptions = {"promoted": "Updated description triggers reindex"}

        skills_loader = MagicMock()
        skills_loader.list_skills.return_value = entries
        skills_loader.get_skill_metadata.side_effect = lambda name: {
            "description": descriptions.get(name, "")
        }

        vec = np.zeros(512, dtype=np.float32)
        vec[0] = 1.0

        def embedder_factory():
            chain = MagicMock()
            chain.__aenter__ = AsyncMock(return_value=chain)
            chain.__aexit__ = AsyncMock(return_value=False)
            chain.embed = AsyncMock(return_value=[vec])
            return chain

        indexer = SkillIndexer(
            db=hook.db,
            skills_loader=skills_loader,
            embedder_factory=embedder_factory,
        )
        await indexer.refresh()

        row = hook.db.execute(
            "SELECT status FROM skill_stats WHERE name = ?", ("promoted",)
        ).fetchone()
        assert row is not None
        # Must remain active — not reset to draft by the re-index.
        assert row[0] == "active"

    @pytest.mark.asyncio
    async def test_disappearing_skill_cleaned_from_skill_stats(self, tmp_path):
        """When a skill present at refresh #1 is gone at refresh #2, its
        skill_stats and skill_vec rows must be removed by ``_cleanup_removed``.
        Without this, deprecated/deleted skills leave dangling rows that
        ``skill_stats`` queries would surface as ghosts.
        """
        loop = _make_loop(tmp_path)
        entries_v1 = [
            _fake_entry("alive", "workspace", tmp_path),
            _fake_entry("doomed", "workspace", tmp_path),
        ]
        descriptions = {"alive": "stays", "doomed": "removed next refresh"}
        indexer, hook = self._make_indexer(loop, entries_v1, descriptions)
        await indexer.refresh()

        # Sanity: both rows exist after first refresh.
        names = {
            r[0] for r in hook.db.execute(
                "SELECT name FROM skill_stats"
            ).fetchall()
        }
        assert {"alive", "doomed"} <= names

        # Now drop "doomed" from the entries the loader returns and refresh.
        indexer._skills_loader.list_skills.return_value = [entries_v1[0]]
        await indexer.refresh()

        names = {
            r[0] for r in hook.db.execute(
                "SELECT name FROM skill_stats"
            ).fetchall()
        }
        assert "alive" in names
        assert "doomed" not in names, "skill_stats row leaked after disappearance"

        # And its vector row should be gone too (no orphan in skill_vec).
        # Get the 'alive' id; any other id-having vec row would be an orphan.
        alive_id = hook.db.execute(
            "SELECT id FROM skill_stats WHERE name = ?", ("alive",)
        ).fetchone()[0]
        vec_ids = {
            r[0] for r in hook.db.execute(
                "SELECT skill_id FROM skill_vec"
            ).fetchall()
        }
        assert vec_ids == {alive_id}, f"orphaned vec rows: {vec_ids}"

    @pytest.mark.asyncio
    async def test_unknown_source_defaults_to_draft(self, tmp_path):
        """A missing 'source' key should not crash — default to draft."""
        loop = _make_loop(tmp_path)

        entry = _fake_entry("no-source", "workspace", tmp_path)
        del entry["source"]  # simulate a loader that omits the key

        skills_loader = MagicMock()
        skills_loader.list_skills.return_value = [entry]
        skills_loader.get_skill_metadata.return_value = {"description": "test"}

        vec = np.zeros(512, dtype=np.float32)
        vec[0] = 1.0

        def embedder_factory():
            chain = MagicMock()
            chain.__aenter__ = AsyncMock(return_value=chain)
            chain.__aexit__ = AsyncMock(return_value=False)
            chain.embed = AsyncMock(return_value=[vec])
            return chain

        hook = nano_hermes.install(loop)
        indexer = SkillIndexer(
            db=hook.db,
            skills_loader=skills_loader,
            embedder_factory=embedder_factory,
        )
        await indexer.refresh()

        row = hook.db.execute(
            "SELECT status FROM skill_stats WHERE name = ?", ("no-source",)
        ).fetchone()
        assert row is not None
        assert row[0] == "draft"
