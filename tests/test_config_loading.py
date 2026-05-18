"""Tests for auto-discovery of nano_hermes.json config files."""
from __future__ import annotations

import json
from unittest.mock import patch

import nano_hermes
from nano_hermes import _deep_merge, _load_config_files


class TestDeepMerge:
    def test_flat_override(self):
        result = _deep_merge({"a": 1, "b": 2}, {"b": 3, "c": 4})
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"skill_stats": {"threshold": 0.6, "min_uses": 5}}
        override = {"skill_stats": {"min_uses": 10}}
        result = _deep_merge(base, override)
        assert result["skill_stats"]["threshold"] == 0.6  # preserved
        assert result["skill_stats"]["min_uses"] == 10    # overridden

    def test_nested_override_with_scalar(self):
        result = _deep_merge({"a": {"x": 1}}, {"a": 99})
        assert result["a"] == 99  # scalar replaces dict

    def test_empty_override(self):
        base = {"a": 1}
        assert _deep_merge(base, {}) == {"a": 1}

    def test_empty_base(self):
        assert _deep_merge({}, {"a": 1}) == {"a": 1}

    def test_does_not_mutate_base(self):
        base = {"a": {"x": 1}}
        _deep_merge(base, {"a": {"y": 2}})
        assert "y" not in base["a"]


class TestLoadConfigFiles:
    def test_no_files_returns_empty(self, tmp_path):
        with patch("pathlib.Path.home", return_value=tmp_path / "home"):
            result = _load_config_files(tmp_path / "workspace")
        assert result == {}

    def test_user_file_loaded(self, tmp_path):
        home = tmp_path / "home"
        nanobot_dir = home / ".nanobot"
        nanobot_dir.mkdir(parents=True)
        (nanobot_dir / "nano_hermes.json").write_text(
            json.dumps({"skill_stats": {"rewrite_session_interval": 7}})
        )
        with patch("pathlib.Path.home", return_value=home):
            result = _load_config_files(tmp_path / "workspace")
        assert result["skill_stats"]["rewrite_session_interval"] == 7

    def test_workspace_file_loaded(self, tmp_path):
        ws = tmp_path / "workspace"
        cfg_dir = ws / "nano_hermes"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "config.json").write_text(
            json.dumps({"reflection_scope": "global"})
        )
        with patch("pathlib.Path.home", return_value=tmp_path / "home"):
            result = _load_config_files(ws)
        assert result["reflection_scope"] == "global"

    def test_workspace_overrides_user(self, tmp_path):
        home = tmp_path / "home"
        (home / ".nanobot").mkdir(parents=True)
        (home / ".nanobot" / "nano_hermes.json").write_text(
            json.dumps({"skill_stats": {"rewrite_session_interval": 5, "min_uses": 3}})
        )
        ws = tmp_path / "workspace"
        (ws / "nano_hermes").mkdir(parents=True)
        (ws / "nano_hermes" / "config.json").write_text(
            json.dumps({"skill_stats": {"rewrite_session_interval": 10}})
        )
        with patch("pathlib.Path.home", return_value=home):
            result = _load_config_files(ws)
        # workspace wins on the key it specifies
        assert result["skill_stats"]["rewrite_session_interval"] == 10
        # user value preserved for keys not in workspace file
        assert result["skill_stats"]["min_uses"] == 3

    def test_invalid_json_skipped(self, tmp_path):
        home = tmp_path / "home"
        (home / ".nanobot").mkdir(parents=True)
        (home / ".nanobot" / "nano_hermes.json").write_text("not json {{{")
        with patch("pathlib.Path.home", return_value=home):
            result = _load_config_files(tmp_path / "workspace")
        assert result == {}

    def test_non_object_json_skipped(self, tmp_path):
        home = tmp_path / "home"
        (home / ".nanobot").mkdir(parents=True)
        (home / ".nanobot" / "nano_hermes.json").write_text("[1, 2, 3]")
        with patch("pathlib.Path.home", return_value=home):
            result = _load_config_files(tmp_path / "workspace")
        assert result == {}


class TestInstallPicksUpConfigFiles:
    def test_install_no_config_reads_files(self, tmp_path):
        from conftest import _make_loop

        home = tmp_path / "home"
        (home / ".nanobot").mkdir(parents=True)
        (home / ".nanobot" / "nano_hermes.json").write_text(
            json.dumps({"skill_stats": {"rewrite_session_interval": 9}})
        )
        loop = _make_loop(tmp_path)
        with patch("pathlib.Path.home", return_value=home):
            hook = nano_hermes.install(loop)
        assert hook.config.skill_stats.rewrite_session_interval == 9

    def test_explicit_config_dict_bypasses_files(self, tmp_path):
        from conftest import _make_loop

        home = tmp_path / "home"
        (home / ".nanobot").mkdir(parents=True)
        (home / ".nanobot" / "nano_hermes.json").write_text(
            json.dumps({"skill_stats": {"rewrite_session_interval": 9}})
        )
        loop = _make_loop(tmp_path)
        with patch("pathlib.Path.home", return_value=home):
            hook = nano_hermes.install(loop, config={"skill_stats": {"rewrite_session_interval": 3}})
        # explicit config wins; file is not consulted
        assert hook.config.skill_stats.rewrite_session_interval == 3
