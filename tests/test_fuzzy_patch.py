"""Tests for nano_hermes/utils/fuzzy_patch.py."""
from __future__ import annotations

import pytest

from nano_hermes.utils.fuzzy_patch import apply_patch, find_match, PatchMatch


SKILL_MD = """\
---
name: my-skill
description: Does something useful
---

## Overview

This skill helps you do something.

## Procedure

1. First step
2. Second step
   - sub-item
3. Third step
"""


class TestExactMatch:
    def test_single_exact_match(self):
        new, err, conf = apply_patch(SKILL_MD, "First step", "Zeroth step")
        assert err is None
        assert conf == 1.0
        assert "Zeroth step" in new
        assert "First step" not in new

    def test_no_match_returns_error(self):
        new, err, conf = apply_patch(SKILL_MD, "nonexistent text", "replacement")
        assert new is None
        assert err is not None
        assert conf == 0.0

    def test_multi_match_without_replace_all_returns_error(self):
        content = "foo bar foo"
        new, err, conf = apply_patch(content, "foo", "baz")
        assert new is None
        assert "2" in err  # mentions count

    def test_replace_all_exact(self):
        content = "foo bar foo"
        new, err, conf = apply_patch(content, "foo", "baz", replace_all=True)
        assert err is None
        assert new == "baz bar baz"
        assert conf == 1.0

    def test_replace_all_no_match_returns_error(self):
        new, err, conf = apply_patch("abc", "xyz", "zzz", replace_all=True)
        assert new is None
        assert err is not None

    def test_empty_old_string_returns_error(self):
        new, err, conf = apply_patch("content", "", "new")
        assert new is None
        assert err is not None


class TestIndentationTolerance:
    def test_indented_multiline_matched(self):
        content = "class Foo:\n    def bar(self):\n        return 42\n"
        # LLM provides old_string with 0 indent; actual content has 4+8 spaces
        old = "def bar(self):\n    return 42"
        new, err, conf = apply_patch(content, old, "def bar(self):\n    return 0")
        assert err is None
        assert conf == pytest.approx(0.95)
        assert "return 0" in new
        assert "return 42" not in new

    def test_different_indent_width_matched(self):
        content = "  step one\n  step two\n"
        # LLM provided old_string with 4-space indent instead of 2
        old = "    step one\n    step two"
        new, err, conf = apply_patch(content, old, "  replaced\n  content")
        assert err is None
        assert conf == pytest.approx(0.95)
        assert "replaced" in new

    def test_multi_match_indentation_returns_error(self):
        # Two identical stripped lines → ambiguous
        content = "  foo\n  foo\n"
        new, err, conf = apply_patch(content, "    foo", "bar")
        assert new is None
        assert err is not None
        assert "2" in err


class TestSequenceMatcherFallback:
    def test_minor_typo_matched_above_threshold(self):
        content = "## Procedure\n\n1. First step\n2. Second step\n"
        # Old string has a minor typo ("Fist" vs "First")
        old = "1. Fist step\n2. Second step"
        new, err, conf = apply_patch(content, old, "1. Initial step\n2. Second step",
                                     min_confidence=0.7)
        assert err is None
        assert conf >= 0.7
        assert "Initial step" in new

    def test_low_similarity_refused(self):
        content = "completely different content here"
        old = "nothing like this at all whatsoever"
        new, err, conf = apply_patch(content, old, "replacement", min_confidence=0.85)
        assert new is None
        assert err is not None
        assert "similarity" in err.lower() or "not found" in err.lower()

    def test_confidence_below_threshold_refused(self):
        content = "alpha beta gamma delta"
        old = "TOTALLY UNRELATED GARBAGE TEXT"
        new, err, _ = apply_patch(content, old, "x")
        assert new is None


class TestFindMatchDirect:
    def test_returns_patch_match_on_exact(self):
        content = "abc def ghi"
        m, err = find_match(content, "def")
        assert err is None
        assert isinstance(m, PatchMatch)
        assert m.confidence == 1.0
        assert content[m.start:m.end] == "def"

    def test_returns_none_on_no_match(self):
        m, err = find_match("abc", "xyz")
        assert m is None
        assert err is not None

    def test_indented_match_positions_correct(self):
        content = "line one\n    indented line\nline three\n"
        m, err = find_match(content, "indented line")
        assert err is None
        assert m is not None
        # The replacement should produce valid content
        patched = content[: m.start] + "replaced" + content[m.end :]
        assert "replaced" in patched
        assert "line one" in patched
        assert "line three" in patched


class TestProposeSkillPatchIntegration:
    """Integration: fuzzy patch wired into ProposeSkillTool._patch."""

    async def test_patch_with_indented_old_string(self, tmp_path, monkeypatch):
        import nano_hermes
        from conftest import _make_loop

        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)

        # Create a skill
        skill_dir = tmp_path / "skills" / "test-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: test-skill\ndescription: A test skill\n---\n\n"
            "## Steps\n\n1. do something\n2. do another thing\n"
        )
        hook.db.execute(
            "INSERT INTO skill_stats (name, status) VALUES ('test-skill', 'active')"
        )
        hook.db.commit()

        from nano_hermes.skills.propose_tool import ProposeSkillTool

        tool = ProposeSkillTool(hook=hook)
        # old_string with extra leading spaces (indentation mismatch)
        result = await tool.execute(
            action="patch",
            name="test-skill",
            old_string="   do something",  # 3-space indent vs 0
            new_string="do the thing",
        )
        assert "error" not in result.lower(), f"unexpected error: {result}"
        assert "patched" in result.lower()
        patched = (skill_dir / "SKILL.md").read_text()
        assert "do the thing" in patched
