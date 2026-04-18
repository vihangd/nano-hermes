"""External skill dir tests — discovery, indexing overlay, mutation refusal."""
from __future__ import annotations

from pathlib import Path

import pytest

import nano_hermes
from conftest import _make_loop, _patch_embedding, _unset_embedding_keys
from nano_hermes.skills.external import (
    discover_external_skills,
    expand_external_dirs,
)
from nano_hermes.skills.propose_tool import ProposeSkillTool


def _write_skill(dir_: Path, name: str, description: str = "ext skill") -> None:
    (dir_ / name).mkdir(parents=True)
    (dir_ / name / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\nBody.\n"
    )


class TestDiscovery:
    def test_expand_handles_tilde_and_envvar(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EXT_BASE", str(tmp_path))
        result = expand_external_dirs(["${EXT_BASE}"])
        assert result == [tmp_path.resolve()]

    def test_nonexistent_dir_skipped(self, tmp_path):
        bogus = tmp_path / "does-not-exist"
        result = expand_external_dirs([str(bogus)])
        assert result == []

    def test_dedup(self, tmp_path):
        result = expand_external_dirs([str(tmp_path), str(tmp_path)])
        assert result == [tmp_path.resolve()]

    def test_empty_entries_skipped(self, tmp_path):
        result = expand_external_dirs(["", "   ", str(tmp_path)])
        assert result == [tmp_path.resolve()]

    def test_discover_finds_skills(self, tmp_path):
        _write_skill(tmp_path, "alpha", "alpha skill")
        _write_skill(tmp_path, "beta", "beta skill")
        results = discover_external_skills([tmp_path])
        names = {r["name"] for r in results}
        assert names == {"alpha", "beta"}
        for r in results:
            assert r["source"] == "external"
            assert r["path"].endswith("/SKILL.md")
            assert r["description"]

    def test_excluded_dirs_skipped(self, tmp_path):
        _write_skill(tmp_path, "good")
        # Build a SKILL.md inside .git — should be ignored.
        (tmp_path / ".git" / "shadow").mkdir(parents=True)
        (tmp_path / ".git" / "shadow" / "SKILL.md").write_text(
            "---\nname: shadow\ndescription: x\n---\n\nB\n"
        )
        names = {r["name"] for r in discover_external_skills([tmp_path])}
        assert names == {"good"}

    def test_unparseable_frontmatter_falls_back_to_dir_name(self, tmp_path):
        # No frontmatter at all — name should fall back to the directory name.
        (tmp_path / "barebones").mkdir()
        (tmp_path / "barebones" / "SKILL.md").write_text("Just a body.\n")
        results = discover_external_skills([tmp_path])
        names = {r["name"] for r in results}
        assert "barebones" in names

    def test_empty_description_handled(self, tmp_path):
        """SKILL.md with `description:` (empty value) parses without crashing
        and produces an entry with an empty description string. The indexer
        embeds 'name: ' alone via _content_text — still valid, just lower
        signal for retrieval.
        """
        (tmp_path / "no-desc").mkdir()
        (tmp_path / "no-desc" / "SKILL.md").write_text(
            "---\nname: no-desc\ndescription:\n---\n\nBody.\n"
        )
        results = discover_external_skills([tmp_path])
        entry = next(r for r in results if r["name"] == "no-desc")
        assert entry["description"] == ""

    def test_crlf_line_endings_parse(self, tmp_path):
        """External SKILL.md authored on Windows (\\r\\n line endings) must
        parse. Python's text-mode read normalizes \\r\\n → \\n before our
        regex sees the content; this test pins that behavior so swapping to
        binary read (or a regex that requires literal \\n) would fail loudly.
        """
        (tmp_path / "crlf").mkdir()
        (tmp_path / "crlf" / "SKILL.md").write_bytes(
            b"---\r\nname: crlf\r\ndescription: from windows\r\n---\r\n\r\nBody.\r\n"
        )
        results = discover_external_skills([tmp_path])
        entry = next(r for r in results if r["name"] == "crlf")
        assert entry["description"] == "from windows"

    def test_symlink_loop_does_not_crash_discovery(self, tmp_path):
        """A symlink that loops within the external dir must not crash rglob.
        Path.rglob doesn't follow symlinks by default; this test pins that
        assumption — a future change to follow_symlinks=True would break it.
        """
        _write_skill(tmp_path, "real", "real skill")
        # Create a self-referential symlink. Without follow_symlinks the
        # walker doesn't enter it, so discovery still completes.
        (tmp_path / "loop").symlink_to(tmp_path)
        results = discover_external_skills([tmp_path])
        names = {r["name"] for r in results}
        assert "real" in names


class TestIndexerOverlay:
    @pytest.mark.asyncio
    async def test_external_skill_appears_in_list(self, tmp_path, monkeypatch):
        _unset_embedding_keys(monkeypatch)
        ext_dir = tmp_path / "ext"
        ext_dir.mkdir()
        _write_skill(ext_dir, "from-ext", "ext skill")

        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(
            loop, config={"skills": {"external_dirs": [str(ext_dir)]}}
        )
        entries = hook.skill_indexer._list_entries()
        ext_entry = next(e for e in entries if e["name"] == "from-ext")
        assert ext_entry["source"] == "external"

    @pytest.mark.asyncio
    async def test_workspace_overrides_external_with_same_name(
        self, tmp_path, monkeypatch
    ):
        _unset_embedding_keys(monkeypatch)
        ext_dir = tmp_path / "ext"
        ext_dir.mkdir()
        _write_skill(ext_dir, "shared", "ext version")

        ws_skill = tmp_path / "skills" / "shared"
        ws_skill.mkdir(parents=True)
        (ws_skill / "SKILL.md").write_text(
            "---\nname: shared\ndescription: ws version\n---\n\nB\n"
        )

        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(
            loop, config={"skills": {"external_dirs": [str(ext_dir)]}}
        )
        entries = hook.skill_indexer._list_entries()
        shared = next(e for e in entries if e["name"] == "shared")
        assert shared["source"] == "workspace"


class TestStatus:
    @pytest.mark.asyncio
    async def test_external_indexed_as_active(self, tmp_path, monkeypatch):
        _unset_embedding_keys(monkeypatch)
        _patch_embedding(monkeypatch)
        ext_dir = tmp_path / "ext"
        ext_dir.mkdir()
        _write_skill(ext_dir, "trusted-ext", "trusted")

        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(
            loop, config={"skills": {"external_dirs": [str(ext_dir)]}}
        )
        await hook.skill_indexer.refresh()
        row = hook.db.execute(
            "SELECT status FROM skill_stats WHERE name=?", ("trusted-ext",)
        ).fetchone()
        assert row is not None, "external skill was not indexed"
        assert row[0] == "active"


class TestProposeSkillRefusal:
    @pytest.mark.asyncio
    async def test_create_collides_with_external(self, tmp_path, monkeypatch):
        _unset_embedding_keys(monkeypatch)
        _patch_embedding(monkeypatch)
        ext_dir = tmp_path / "ext"
        ext_dir.mkdir()
        _write_skill(ext_dir, "occupied", "ext skill")

        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(
            loop, config={"skills": {"external_dirs": [str(ext_dir)]}}
        )
        # Index so the conflict check sees the external skill.
        await hook.skill_indexer.refresh()

        tool = ProposeSkillTool(hook=hook)
        out = await tool.execute(
            name="occupied",
            description="Try to overwrite ext skill.",
            body="Body.",
        )
        assert "Error" in out
        assert "already exists" in out

    @pytest.mark.asyncio
    async def test_edit_external_refused(self, tmp_path, monkeypatch):
        _unset_embedding_keys(monkeypatch)
        _patch_embedding(monkeypatch)
        ext_dir = tmp_path / "ext"
        ext_dir.mkdir()
        _write_skill(ext_dir, "ext-edit", "ext skill")

        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(
            loop, config={"skills": {"external_dirs": [str(ext_dir)]}}
        )
        await hook.skill_indexer.refresh()

        tool = ProposeSkillTool(hook=hook)
        out = await tool.execute(
            action="edit",
            name="ext-edit",
            description="Updated.",
            body="v2",
        )
        assert "Error" in out
        assert "external" in out
        assert "read-only" in out

    @pytest.mark.asyncio
    async def test_patch_external_refused(self, tmp_path, monkeypatch):
        _unset_embedding_keys(monkeypatch)
        _patch_embedding(monkeypatch)
        ext_dir = tmp_path / "ext"
        ext_dir.mkdir()
        _write_skill(ext_dir, "ext-patch", "ext skill")

        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(
            loop, config={"skills": {"external_dirs": [str(ext_dir)]}}
        )
        await hook.skill_indexer.refresh()

        tool = ProposeSkillTool(hook=hook)
        out = await tool.execute(
            action="patch",
            name="ext-patch",
            old_string="Body",
            new_string="NewBody",
        )
        assert "Error" in out
        assert "external" in out
        assert "read-only" in out

    @pytest.mark.asyncio
    async def test_workspace_shadow_allows_edit(self, tmp_path, monkeypatch):
        """Workspace > external precedence: if a workspace copy exists,
        edit operates on it (the external is shadowed and not refused).
        """
        _unset_embedding_keys(monkeypatch)
        _patch_embedding(monkeypatch)

        ext_dir = tmp_path / "ext"
        ext_dir.mkdir()
        _write_skill(ext_dir, "doppelganger", "ext version")

        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(
            loop, config={"skills": {"external_dirs": [str(ext_dir)]}}
        )
        tool = ProposeSkillTool(hook=hook)

        # Create a workspace copy with the same name (this is the path
        # the user is told to take by the read-only error message).
        out = await tool.execute(
            name="doppelganger",
            description="ws version.",
            body="ws body.",
        )
        assert out.startswith("ok:"), out

        # Editing the workspace copy must succeed — the external is shadowed.
        out = await tool.execute(
            action="edit",
            name="doppelganger",
            description="ws v2.",
            body="ws body v2.",
        )
        assert out.startswith("ok:"), out
        # And the workspace SKILL.md has v2.
        text = (tmp_path / "skills" / "doppelganger" / "SKILL.md").read_text()
        assert "ws body v2." in text


class TestSkillSearchEndToEnd:
    @pytest.mark.asyncio
    async def test_external_skill_appears_in_skill_search_results(
        self, tmp_path, monkeypatch
    ):
        """End-to-end: external dir → indexer overlay → vec0 query →
        SkillHit serialized into the agent-visible result lines.

        Closes the structural gap between "indexer knows about external"
        (covered by other tests) and "agent-facing tool surfaces them".
        """
        _unset_embedding_keys(monkeypatch)
        _patch_embedding(monkeypatch)

        ext_dir = tmp_path / "ext"
        ext_dir.mkdir()
        # Description maps to _FAKE_VEC_ACADEMIC via 'arxiv' / 'academic'
        # / 'papers' keywords in conftest._FAKE_KEYWORDS.
        _write_skill(ext_dir, "papers-finder", "Find arxiv academic papers")

        loop = _make_loop(tmp_path)
        nano_hermes.install(
            loop, config={"skills": {"external_dirs": [str(ext_dir)]}}
        )

        tool = loop.tools.get("skill_search")
        assert tool is not None
        # Query embeds to the same fake vector as the description.
        out = await tool.execute(query="academic papers")

        assert "papers-finder" in out, (
            f"external skill should appear in skill_search results:\n{out}"
        )
        # Description surfaces alongside the name (parallel to test_skills.py).
        assert "arxiv" in out.lower()
