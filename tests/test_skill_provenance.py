"""Skill write-origin provenance + pinning.

Only ``origin='agent'`` skills (created via propose_skill) and not user-pinned
are eligible for automatic evolution (rewrite/deprecation). Everything else —
builtin, external, and hand-authored workspace skills — defaults to ``'user'``
and is protected. ``pin``/``unpin`` let a user exempt any skill explicitly.
"""
from __future__ import annotations

import time

import pytest
from nanobot.agent.hook import AgentHookContext

import nano_hermes
from conftest import _unset_embedding_keys
from nano_hermes.skills.curator import find_stale_skills
from nano_hermes.skills.rewriter import get_rewrite_candidates


def _seed(hook, name, *, use_count, success_count, origin, pinned=0, status="active"):
    hook.db.execute(
        "INSERT OR REPLACE INTO skill_stats "
        "(name, status, use_count, success_count, origin, pinned) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (name, status, use_count, success_count, origin, pinned),
    )
    hook.db.commit()


class TestOrigin:
    async def test_proposed_skill_is_agent_origin(self, loop):
        hook = nano_hermes.install(loop)
        tool = loop.tools.get("propose_skill")

        out = await tool.execute(
            name="my-helper",
            description="Does something useful",
            body="# My Helper\n\nUse this to do something useful.\n",
        )
        assert out.startswith("ok"), out

        row = hook.db.execute(
            "SELECT origin, pinned FROM skill_stats WHERE name = ?", ("my-helper",)
        ).fetchone()
        assert row == ("agent", 0)

    def test_default_origin_is_user(self, loop):
        """A row inserted without an explicit origin (as the indexer does for
        disk-discovered skills) defaults to 'user' and is thus protected."""
        hook = nano_hermes.install(loop)
        hook.db.execute(
            "INSERT INTO skill_stats (name, status, content_hash, indexed_at) "
            "VALUES ('hand-authored', 'active', 'h', 0)"
        )
        hook.db.commit()
        row = hook.db.execute(
            "SELECT origin, pinned FROM skill_stats WHERE name = 'hand-authored'"
        ).fetchone()
        assert row == ("user", 0)


class TestCandidateFilter:
    def test_user_origin_skill_is_never_a_candidate(self, loop):
        hook = nano_hermes.install(loop)
        # Identical failing stats; only origin differs.
        _seed(hook, "agent-bad", use_count=10, success_count=2, origin="agent")
        _seed(hook, "user-bad", use_count=10, success_count=2, origin="user")

        names = [
            c.skill_name
            for c in get_rewrite_candidates(hook.db, failure_threshold=0.6, min_uses=5)
        ]
        assert "agent-bad" in names
        assert "user-bad" not in names

    def test_pinned_agent_skill_is_excluded(self, loop):
        hook = nano_hermes.install(loop)
        _seed(hook, "pinned-bad", use_count=10, success_count=1, origin="agent", pinned=1)

        names = [
            c.skill_name
            for c in get_rewrite_candidates(hook.db, failure_threshold=0.6, min_uses=5)
        ]
        assert "pinned-bad" not in names


class TestCuratorArchivalRespectsProvenance:
    """The curator's time-based stale-skill archival must also honor origin/pin."""

    def _seed_stale(self, hook, name, *, origin, pinned=0):
        hook.db.execute(
            "INSERT INTO skill_stats (name, status, use_count, last_used_at, origin, pinned) "
            "VALUES (?, 'active', 10, ?, ?, ?)",
            (name, time.time() - 40 * 86400, origin, pinned),
        )
        hook.db.commit()

    def test_user_and_pinned_skills_are_not_stale_candidates(self, loop):
        hook = nano_hermes.install(loop)
        self._seed_stale(hook, "agent-stale", origin="agent")
        self._seed_stale(hook, "user-stale", origin="user")
        self._seed_stale(hook, "pinned-stale", origin="agent", pinned=1)

        names = [s.name for s in find_stale_skills(hook.db, stale_after_days=30, min_uses=3)]
        assert "agent-stale" in names
        assert "user-stale" not in names
        assert "pinned-stale" not in names


class TestFailureRateDeprecationRespectsProvenance:
    """check_promotions must not auto-deprecate user-origin or pinned skills."""

    async def test_user_origin_skill_is_not_deprecated_on_failure(
        self, loop, monkeypatch: pytest.MonkeyPatch
    ):
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(
            loop,
            config={
                "skill_stats": {
                    "deprecation_min_uses": 5,
                    "deprecation_max_success_rate": 0.2,
                }
            },
        )
        # 4 uses / 0 success, origin='user' — a failing hand-authored skill.
        hook.db.execute(
            "INSERT INTO skill_stats (name, status, use_count, success_count, origin) "
            "VALUES ('user-bad', 'active', 4, 0, 'user')"
        )
        hook.db.commit()

        await hook.before_iteration(
            AgentHookContext(iteration=0, messages=[{"role": "user", "content": "task"}])
        )
        # 5th failure crosses the threshold — but origin='user' must shield it.
        await loop.tools.get("skill_rate").execute(name="user-bad", outcome="failure")

        status = hook.db.execute(
            "SELECT status FROM skill_stats WHERE name = 'user-bad'"
        ).fetchone()[0]
        assert status == "active", f"user-origin skill should not be deprecated, got {status}"

    async def test_pinned_agent_skill_is_not_deprecated_on_failure(
        self, loop, monkeypatch: pytest.MonkeyPatch
    ):
        """Exercises the `not pinned` arm specifically (origin is 'agent', so the
        origin check passes and the pin is the only thing shielding it)."""
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(
            loop,
            config={
                "skill_stats": {
                    "deprecation_min_uses": 5,
                    "deprecation_max_success_rate": 0.2,
                }
            },
        )
        hook.db.execute(
            "INSERT INTO skill_stats (name, status, use_count, success_count, origin, pinned) "
            "VALUES ('pinned-bad', 'active', 4, 0, 'agent', 1)"
        )
        hook.db.commit()

        await hook.before_iteration(
            AgentHookContext(iteration=0, messages=[{"role": "user", "content": "task"}])
        )
        await loop.tools.get("skill_rate").execute(name="pinned-bad", outcome="failure")

        status = hook.db.execute(
            "SELECT status FROM skill_stats WHERE name = 'pinned-bad'"
        ).fetchone()[0]
        assert status == "active", f"pinned agent skill should not be deprecated, got {status}"


class TestPinAction:
    async def test_pin_then_unpin_toggles_flag(self, loop):
        hook = nano_hermes.install(loop)
        tool = loop.tools.get("propose_skill")
        _seed(hook, "keepme", use_count=0, success_count=0, origin="agent")

        out = await tool.execute(action="pin", name="keepme")
        assert out.startswith("ok"), out
        assert hook.db.execute(
            "SELECT pinned FROM skill_stats WHERE name = 'keepme'"
        ).fetchone()[0] == 1

        out = await tool.execute(action="unpin", name="keepme")
        assert out.startswith("ok"), out
        assert hook.db.execute(
            "SELECT pinned FROM skill_stats WHERE name = 'keepme'"
        ).fetchone()[0] == 0

    async def test_pin_unknown_skill_errors(self, loop):
        nano_hermes.install(loop)
        tool = loop.tools.get("propose_skill")
        out = await tool.execute(action="pin", name="ghost")
        assert out.startswith("Error"), out

    async def test_pin_survives_reproposal_and_edit(self, loop):
        """A user pin must persist across the agent re-editing the skill."""
        hook = nano_hermes.install(loop)
        tool = loop.tools.get("propose_skill")
        await tool.execute(
            name="durable",
            description="d",
            body="# Durable\n\nbody one.\n",
        )
        await tool.execute(action="pin", name="durable")

        out = await tool.execute(
            action="edit",
            name="durable",
            description="d2",
            body="# Durable\n\nbody two.\n",
        )
        assert out.startswith("ok"), out
        assert hook.db.execute(
            "SELECT pinned FROM skill_stats WHERE name = 'durable'"
        ).fetchone()[0] == 1
