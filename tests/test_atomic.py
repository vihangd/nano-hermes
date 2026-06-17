"""Unit tests for nano_hermes._atomic.atomic_write_text.

The helper is used by both propose_skill (skills) and BudgetedMemory
(memory). These tests exercise it in isolation; the integration with
each caller is covered by test_propose_skill_rollback and
test_memory_atomic respectively.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

import nano_hermes._atomic as mod
from nano_hermes._atomic import atomic_write_text


def test_writes_content(tmp_path):
    target = tmp_path / "out.txt"
    atomic_write_text(target, "hello")
    assert target.read_text() == "hello"


def test_creates_parent_dirs(tmp_path):
    target = tmp_path / "deep" / "nested" / "out.txt"
    atomic_write_text(target, "hi")
    assert target.read_text() == "hi"


def test_no_tmp_file_leftover_on_success(tmp_path):
    target = tmp_path / "out.txt"
    atomic_write_text(target, "hello")
    leftover = list(tmp_path.glob(".out.txt.tmp.*"))
    assert leftover == []


def test_overwrites_existing(tmp_path):
    target = tmp_path / "out.txt"
    target.write_text("v1")
    atomic_write_text(target, "v2")
    assert target.read_text() == "v2"


def test_crash_mid_write_preserves_original_and_cleans_tmp(tmp_path):
    """If the write to the tempfile raises, the target file is untouched
    and no tempfile remains in the parent dir.
    """
    target = tmp_path / "out.txt"
    target.write_text("original")

    real_fdopen = mod.os.fdopen

    def failing_fdopen(fd, mode, **kwargs):
        f = real_fdopen(fd, mode, **kwargs)

        def _raise(_):
            raise OSError("simulated disk error")

        f.write = _raise  # type: ignore[method-assign]
        return f

    with patch.object(mod.os, "fdopen", failing_fdopen):
        with pytest.raises(OSError, match="simulated disk error"):
            atomic_write_text(target, "v2")

    assert target.read_text() == "original"
    assert list(tmp_path.glob(".out.txt.tmp.*")) == []


def test_unicode_content(tmp_path):
    target = tmp_path / "out.txt"
    atomic_write_text(target, "héllo 🚀 世界")
    assert target.read_text() == "héllo 🚀 世界"


def test_explicit_encoding_passes_through(tmp_path):
    target = tmp_path / "out.txt"
    # Plain ASCII so the test is encoding-agnostic but we exercise the
    # parameter path.
    atomic_write_text(target, "hello", encoding="ascii")
    assert target.read_text(encoding="ascii") == "hello"


def test_preserves_target_mode_bits_when_target_exists(tmp_path):
    """tempfile.mkstemp creates files with 0o600 by default. If the target
    already exists we must copy its mode over the tempfile before os.replace,
    otherwise atomic rewrites silently demote permissions on every edit
    (e.g. a 0o644 SKILL.md becomes 0o600).
    """
    import os
    import stat

    target = tmp_path / "out.txt"
    target.write_text("v1")
    target.chmod(0o644)

    atomic_write_text(target, "v2")

    mode = stat.S_IMODE(os.stat(target).st_mode)
    assert mode == 0o644, f"expected 0o644, got 0o{mode:o}"
    assert target.read_text() == "v2"


def test_copymode_oserror_silently_ignored(tmp_path):
    """OSError from shutil.copymode is swallowed — permissions are best-effort."""
    target = tmp_path / "out.txt"
    target.write_text("v1")
    with patch.object(mod.shutil, "copymode", side_effect=OSError("no perms")):
        atomic_write_text(target, "v2")  # must not raise
    assert target.read_text() == "v2"


def test_unlink_oserror_during_cleanup_propagates_write_error(tmp_path):
    """If the write fails AND the tmp cleanup also fails, the original write
    error still propagates (the cleanup OSError is swallowed).
    """
    target = tmp_path / "out.txt"
    target.write_text("original")

    real_fdopen = mod.os.fdopen

    def failing_fdopen(fd, mode, **kwargs):
        f = real_fdopen(fd, mode, **kwargs)

        def _raise(_):
            raise OSError("disk full")

        f.write = _raise  # type: ignore[method-assign]
        return f

    with patch.object(mod.os, "fdopen", failing_fdopen), \
         patch("nano_hermes._atomic.Path.unlink", side_effect=OSError("perm denied")):
        with pytest.raises(OSError, match="disk full"):
            atomic_write_text(target, "v2")

    assert target.read_text() == "original"
