"""Tests for nano_hermes.cli auto-install monkey-patch.

The cli module patches global classes (``AgentLoop.__init__`` and
``AgentRunner._request_model``) at module level. Tests must save and
restore those references plus the ``_PATCHED`` flag, otherwise the
side-effects leak to every subsequent test in the suite.
"""
from __future__ import annotations

import inspect
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.hook import AgentHook
from nanobot.agent.runner import AgentRunner

import nano_hermes.cli as cli_mod
from nano_hermes.hook import NanoHermesHook

from conftest import _make_loop


class TestFindHermesHook:
    def test_returns_none_when_no_hook(self):
        bare = AgentHook()
        assert cli_mod._find_hermes_hook(bare) is None

    def test_finds_direct_nano_hermes_hook(self, tmp_path):
        loop = _make_loop(tmp_path)
        from nano_hermes import install
        hook = install(loop)
        assert cli_mod._find_hermes_hook(hook) is hook

    def test_finds_hook_nested_in_composite(self, tmp_path):
        loop = _make_loop(tmp_path)
        from nano_hermes import install
        hermes_hook = install(loop)

        class Composite:
            def __init__(self, *hooks):
                self._hooks = list(hooks)

        # NanoHermesHook nested inside something with `_hooks` attr.
        composite = Composite(AgentHook(), hermes_hook)
        assert cli_mod._find_hermes_hook(composite) is hermes_hook


class TestPatchAgentLoop:
    def test_patch_idempotent_double_call_doesnt_double_install(
        self, restore_cli_patches, tmp_path
    ):
        """Calling _patch_agent_loop twice must not double-wrap __init__.
        If it did, a single AgentLoop construction would call install()
        twice, which would register tools twice and crash on duplicate
        tool registration.
        """
        cli_mod._patch_agent_loop()
        cli_mod._patch_agent_loop()  # second call — should be a no-op

        loop = _make_loop(tmp_path)
        hermes_hooks = [
            h for h in loop._extra_hooks if isinstance(h, NanoHermesHook)
        ]
        assert len(hermes_hooks) == 1, (
            f"expected 1 nano-hermes hook after construction, got {len(hermes_hooks)}"
        )

    def test_first_patch_installs_hook_on_loop_construction(
        self, restore_cli_patches, tmp_path
    ):
        """End-to-end: after _patch_agent_loop(), constructing an
        AgentLoop auto-installs nano-hermes via the patched __init__.
        """
        cli_mod._patch_agent_loop()
        loop = _make_loop(tmp_path)
        hermes_hooks = [
            h for h in loop._extra_hooks if isinstance(h, NanoHermesHook)
        ]
        assert len(hermes_hooks) == 1

    def test_install_failure_is_swallowed_and_loop_still_constructs(
        self, restore_cli_patches, tmp_path
    ):
        """If install() raises, the patched __init__ must not propagate it —
        the host CLI must keep working without nano-hermes.
        """
        cli_mod._patch_agent_loop()
        with patch("nano_hermes.install", side_effect=RuntimeError("boom")):
            # _make_loop calls AgentLoop.__init__ which is now patched
            loop = _make_loop(tmp_path)
        # No NanoHermesHook — install failed and was swallowed
        hermes_hooks = [h for h in loop._extra_hooks if isinstance(h, NanoHermesHook)]
        assert hermes_hooks == []


class TestPatchedRequestModelDrain:
    async def test_pending_injections_appended_to_messages(self, restore_cli_patches):
        """patched_request_model must append pending injections to the message list."""
        original_mock = AsyncMock(return_value="sentinel")
        AgentRunner._request_model = original_mock  # type: ignore[method-assign]

        cli_mod._patch_agent_loop()  # closure captures original_mock

        injection = {"role": "user", "content": "injected context"}
        mock_hermes = MagicMock()
        mock_hermes.drain_injections.return_value = [injection]

        messages = [{"role": "user", "content": "original"}]
        runner = MagicMock()

        # Patch _find_hermes_hook so it returns our mock regardless of isinstance check.
        with patch("nano_hermes.cli._find_hermes_hook", return_value=mock_hermes):
            await AgentRunner._request_model(runner, None, messages, MagicMock(), None)

        sent_messages = original_mock.call_args[0][2]
        assert injection in sent_messages
        assert len(sent_messages) == 2

    async def test_no_injections_messages_unchanged(self, restore_cli_patches):
        original_mock = AsyncMock(return_value="sentinel")
        AgentRunner._request_model = original_mock  # type: ignore[method-assign]

        cli_mod._patch_agent_loop()

        mock_hermes = MagicMock()
        mock_hermes.drain_injections.return_value = []

        messages = [{"role": "user", "content": "only message"}]
        runner = MagicMock()

        with patch("nano_hermes.cli._find_hermes_hook", return_value=mock_hermes):
            await AgentRunner._request_model(runner, None, messages, MagicMock(), None)

        sent_messages = original_mock.call_args[0][2]
        assert sent_messages == messages


    async def test_injection_already_in_request_not_appended_again(self, restore_cli_patches):
        """When governance changed nothing, messages_for_model *is* the hook's
        list and already holds the injection; appending it would send it twice."""
        original_mock = AsyncMock(return_value="sentinel")
        AgentRunner._request_model = original_mock  # type: ignore[method-assign]
        cli_mod._patch_agent_loop()

        injection = {"role": "user", "content": "injected context"}
        late = {"role": "user", "content": "queued after the copy"}
        mock_hermes = MagicMock()
        mock_hermes.drain_injections.return_value = [injection, late]
        messages = [{"role": "user", "content": "original"}, injection]

        with patch("nano_hermes.cli._find_hermes_hook", return_value=mock_hermes):
            await AgentRunner._request_model(MagicMock(), None, messages, MagicMock(), None)

        sent = original_mock.call_args[0][2]
        assert sent == [messages[0], injection, late]

    async def test_keyword_args_forwarded(self, restore_cli_patches):
        """nanobot 0.3.0 calls _request_model with keyword-only args
        (malformed_retry=); the wrapper must accept and forward them or every
        LLM call raises TypeError."""
        original_mock = AsyncMock(return_value="sentinel")
        AgentRunner._request_model = original_mock  # type: ignore[method-assign]
        cli_mod._patch_agent_loop()

        with patch("nano_hermes.cli._find_hermes_hook", return_value=None):
            out = await AgentRunner._request_model(
                MagicMock(), None, [], MagicMock(), None, malformed_retry=True,
            )
        assert out == "sentinel"
        assert original_mock.call_args.kwargs == {"malformed_retry": True}

    def test_not_patched_when_request_built_after_before_iteration(self, restore_cli_patches):
        """nanobot >=0.3.5 (``request_state`` parameter) builds the request
        from context.messages after before_iteration, so the hook's own
        append already reaches the model; patching would double-inject."""
        async def new_shape(self, spec, messages, hook, context, *, request_state, transcript):
            return None

        AgentRunner._request_model = new_shape  # type: ignore[method-assign]
        cli_mod._patch_agent_loop()
        assert AgentRunner._request_model is new_shape

    def test_installed_nanobot_matches_the_patch_decision(self, restore_cli_patches):
        """Whatever nanobot is installed: patched iff it lacks request_state."""
        real = AgentRunner._request_model
        cli_mod._patch_agent_loop()
        builds_late = "request_state" in inspect.signature(real).parameters
        assert (AgentRunner._request_model is real) == builds_late


class TestMainPendingSubcommand:
    def test_pending_subcommand_dispatches_and_exits(self, restore_cli_patches, tmp_path):
        with patch.object(sys, "argv", ["nano-hermes", "pending", str(tmp_path), "list"]), \
             patch("nano_hermes.governance.pending._run", return_value=0) as mock_run:
            with pytest.raises(SystemExit) as exc_info:
                cli_mod.main()
        mock_run.assert_called_once_with([str(tmp_path), "list"])
        assert exc_info.value.code == 0

    def test_non_pending_subcommand_delegates_to_nanobot(self, restore_cli_patches):
        mock_app = MagicMock()
        with patch.object(sys, "argv", ["nano-hermes", "agent"]), \
             patch("nanobot.cli.commands.app", mock_app):
            cli_mod.main()
        mock_app.assert_called_once()
