"""Tests for nano_hermes/governance/pending.py — the offline CLI."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import nano_hermes
from conftest import _make_loop
from nano_hermes.governance import write_approval as wa
from nano_hermes.governance.pending import _load_config, _run


def _make_hook(tmp_path, config=None):
    return nano_hermes.install(_make_loop(tmp_path), config=config or {})


def _seed_skill(hook, name="my_skill"):
    d = hook.workspace / "skills" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: does {name}\n---\n\nbody\n"
    )
    hook.db.execute(
        "INSERT OR IGNORE INTO skill_stats (name, status, origin) VALUES (?, 'active', 'agent')",
        (name,),
    )
    hook.db.commit()


class TestLoadConfig:
    def test_returns_valid_config_for_real_workspace(self, tmp_path):
        cfg = _load_config(tmp_path)
        # Default embedding dims (512) should come back
        assert cfg.embedding.target_dims > 0

    def test_falls_back_on_bad_config_file(self, tmp_path):
        (tmp_path / "nano-hermes.yaml").write_text("not: {valid: yaml: !!binary X")
        cfg = _load_config(tmp_path)
        # Should not raise; returns default config
        assert cfg.embedding.target_dims > 0

    def test_falls_back_when_load_config_files_raises(self, tmp_path):
        with patch("nano_hermes._load_config_files", side_effect=RuntimeError("boom")):
            cfg = _load_config(tmp_path)
        assert cfg.embedding.target_dims > 0


class TestRunNoArgs:
    def test_no_args_prints_docstring_and_returns_2(self, tmp_path, capsys):
        rc = _run([])
        assert rc == 2
        out = capsys.readouterr().out
        assert "nano-hermes pending" in out

    def test_one_arg_only_prints_docstring_and_returns_2(self, tmp_path, capsys):
        rc = _run([str(tmp_path)])
        assert rc == 2


class TestRunList:
    def test_list_empty(self, tmp_path, capsys):
        hook = _make_hook(tmp_path)
        rc = _run([str(hook.workspace), "list"])
        assert rc == 0
        assert "No pending" in capsys.readouterr().out

    def test_list_with_pending_rows(self, tmp_path, capsys):
        hook = _make_hook(tmp_path)
        _seed_skill(hook)
        wa.stage_skill_write(
            hook, skill_name="my_skill", description="desc", body="new body",
            reason="test reason", origin="rewriter",
        )
        rc = _run([str(hook.workspace), "list"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "my_skill" in out
        assert "rewriter" in out


class TestRunMissingId:
    def test_diff_without_id(self, tmp_path, capsys):
        hook = _make_hook(tmp_path)
        rc = _run([str(hook.workspace), "diff"])
        assert rc == 2
        assert "needs an id" in capsys.readouterr().out

    def test_reject_without_id(self, tmp_path, capsys):
        hook = _make_hook(tmp_path)
        rc = _run([str(hook.workspace), "reject"])
        assert rc == 2

    def test_approve_without_id(self, tmp_path, capsys):
        hook = _make_hook(tmp_path)
        rc = _run([str(hook.workspace), "approve"])
        assert rc == 2


class TestRunDiff:
    def test_diff_existing(self, tmp_path, capsys):
        hook = _make_hook(tmp_path)
        _seed_skill(hook)
        pid = wa.stage_skill_write(
            hook, skill_name="my_skill", description="desc", body="proposed body",
            reason="r", origin="gepa",
        )
        rc = _run([str(hook.workspace), "diff", str(pid)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "proposed body" in out

    def test_diff_nonexistent(self, tmp_path, capsys):
        hook = _make_hook(tmp_path)
        rc = _run([str(hook.workspace), "diff", "999"])
        assert rc == 0  # diff_pending returns a string message


class TestRunReject:
    def test_reject_existing(self, tmp_path, capsys):
        hook = _make_hook(tmp_path)
        _seed_skill(hook)
        pid = wa.stage_skill_write(
            hook, skill_name="my_skill", description="desc", body="body",
            reason="r", origin="rewriter",
        )
        rc = _run([str(hook.workspace), "reject", str(pid)])
        assert rc == 0
        assert "rejected" in capsys.readouterr().out


class TestRunApprove:
    def test_approve_nonexistent(self, tmp_path, capsys):
        hook = _make_hook(tmp_path)
        rc = _run([str(hook.workspace), "approve", "999"])
        assert rc == 1
        assert "no open pending write" in capsys.readouterr().out

    def test_approve_principle_refused_offline(self, tmp_path, capsys):
        hook = _make_hook(tmp_path, config={"principles": {"enabled": True}})
        ops = [{"op": "add", "condition": "when X", "action": "do Y"}]
        pid = wa.stage_principle_ops(hook, ops=ops, reason="test")
        rc = _run([str(hook.workspace), "approve", str(pid)])
        assert rc == 1
        out = capsys.readouterr().out
        assert "principle" in out.lower()

    def test_approve_skill_applies(self, tmp_path, capsys):
        hook = _make_hook(
            tmp_path,
            config={"skill_stats": {"write_approval": "approve"}},
        )
        _seed_skill(hook)
        pid = wa.stage_skill_write(
            hook, skill_name="my_skill", description="desc", body="approved body",
            reason="r", origin="rewriter",
        )
        rc = _run([str(hook.workspace), "approve", str(pid)])
        assert rc == 0
        content = (hook.workspace / "skills" / "my_skill" / "SKILL.md").read_text()
        assert "approved body" in content

    def test_approve_already_rejected(self, tmp_path, capsys):
        hook = _make_hook(tmp_path)
        _seed_skill(hook)
        pid = wa.stage_skill_write(
            hook, skill_name="my_skill", description="", body="b",
            reason="r", origin="gepa",
        )
        # reject first, then try approve
        wa.reject(hook.db, pid)
        rc = _run([str(hook.workspace), "approve", str(pid)])
        assert rc == 1


class TestRunUnknownAction:
    def test_unknown_action_returns_2(self, tmp_path, capsys):
        hook = _make_hook(tmp_path)
        rc = _run([str(hook.workspace), "frobnicate", "1"])
        assert rc == 2
        assert "unknown action" in capsys.readouterr().out
