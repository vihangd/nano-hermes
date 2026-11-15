"""Skill++ description differentiation (skills/differentiate.py)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

import nano_hermes
from conftest import _make_loop
from nano_hermes.session.db import _apply_migrations
from nano_hermes.skills.differentiate import run_differentiation
from nano_hermes.skills.indexer import SkillIndexer

_TOPICS = ("zorblax", "quuxify")


def _vec(text: str) -> np.ndarray:
    v = np.array([float(t in text.lower()) for t in _TOPICS] + [0.01] * 510, dtype=np.float32)
    return v / np.linalg.norm(v)


class _Chain:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def embed(self, texts):
        return [_vec(t) for t in texts]


def _setup(tmp_path, skills: dict[str, tuple[str, str]], **cfg):
    """*skills*: name → (description, origin). Indexed with a topic embedder."""
    hook = nano_hermes.install(
        _make_loop(tmp_path),
        config={"skill_stats": {"differentiation_enabled": True, **cfg}},
    )
    entries = []
    for name, (desc, _origin) in skills.items():
        p = hook.workspace / "skills" / name / "SKILL.md"
        p.parent.mkdir(parents=True)
        p.write_text(f"---\nname: {name}\ndescription: {desc}\n---\n\n## Steps\n1. go\n")
        entries.append({"name": name, "path": str(p), "source": "workspace"})
    loader = MagicMock()
    loader.list_skills.return_value = entries
    hook.skill_indexer = SkillIndexer(
        db=hook.db, skills_loader=loader, embedder_factory=lambda: _Chain()
    )
    return hook, loader


async def _index(hook, skills):
    await hook.skill_indexer.refresh()
    for name, (_desc, origin) in skills.items():
        hook.db.execute("UPDATE skill_stats SET origin = ? WHERE name = ?", (origin, name))
    hook.db.commit()


def _reply(hook, text: str, finish: str = "stop") -> AsyncMock:
    r = MagicMock(content=text, finish_reason=finish, error_status_code=None)
    hook._loop.provider.chat_with_retry = AsyncMock(return_value=r)
    return hook._loop.provider.chat_with_retry


_PAIR = {
    "zorb-report": ("Build zorblax reports", "agent"),
    "zorb-chart": ("Build zorblax charts", "user"),
}
_NEW = "Builds tabular zorblax reports from raw rows. Not for zorblax charts or plots."


def _desc(hook, name):
    from nanobot.agent.skills import SkillsLoader

    return SkillsLoader(hook.workspace).get_skill_metadata(name)["description"]


async def test_close_neighbour_gets_a_distinct_description(tmp_path):
    hook, _ = _setup(tmp_path, _PAIR)
    await _index(hook, _PAIR)
    chat = _reply(hook, _NEW)
    assert await run_differentiation(hook) == ["zorb-report"]
    assert chat.call_count == 1  # zorb-chart is user-authored: never rewritten
    prompt = chat.call_args.kwargs["messages"][0]["content"]
    assert "zorb-chart" in prompt and "Build zorblax charts" in prompt
    assert _desc(hook, "zorb-report") == _NEW
    assert _desc(hook, "zorb-chart") == "Build zorblax charts"


@pytest.mark.parametrize("refresh", [False, True])
async def test_a_differentiated_skill_is_not_redone(tmp_path, refresh):
    """Before re-indexing the edited skill's hash is NULL; after, it equals
    the stored digest. Either way the next cycle makes no call."""
    hook, _ = _setup(tmp_path, _PAIR)
    await _index(hook, _PAIR)
    chat = _reply(hook, _NEW)
    await run_differentiation(hook)
    if refresh:
        await hook.skill_indexer.refresh()
        row = hook.db.execute(
            "SELECT content_hash, differentiated_hash FROM skill_stats WHERE name='zorb-report'"
        ).fetchone()
        assert row[0] is not None and row[0] == row[1]
    assert await run_differentiation(hook) == []
    assert chat.call_count == 1


async def test_far_neighbour_needs_no_call(tmp_path):
    skills = {"zorb": ("Build zorblax reports", "agent"), "quux": ("Run quuxify jobs", "user")}
    hook, _ = _setup(tmp_path, skills)
    await _index(hook, skills)
    chat = _reply(hook, _NEW)
    assert await run_differentiation(hook) == []
    assert chat.call_count == 0
    assert await run_differentiation(hook) == [] and chat.call_count == 0  # marked done


@pytest.mark.parametrize("reply, finish", [
    ("Builds zorblax reports. Not for charts: plots.", "stop"),  # ": " breaks YAML
    ("Builds zorblax reports\nNot for charts.", "stop"),
    ("[beta] Builds zorblax reports. Not for charts.", "stop"),  # YAML flow sequence
    ("Builds zorblax reports and nothing else.", "stop"),  # no exclusion
    ("I can't help with that. Not for this.", "refusal"),
])
async def test_a_bad_reply_changes_nothing(tmp_path, reply, finish):
    hook, _ = _setup(tmp_path, _PAIR)
    await _index(hook, _PAIR)
    before = (hook.workspace / "skills" / "zorb-report" / "SKILL.md").read_text()
    _reply(hook, reply, finish)
    assert await run_differentiation(hook) == []
    assert (hook.workspace / "skills" / "zorb-report" / "SKILL.md").read_text() == before


async def test_approval_gate_stages_once(tmp_path):
    hook, _ = _setup(tmp_path, _PAIR, write_approval="approve")
    await _index(hook, _PAIR)
    chat = _reply(hook, _NEW)
    assert await run_differentiation(hook) == ["zorb-report"]
    assert _desc(hook, "zorb-report") == "Build zorblax reports"
    assert hook.db.execute(
        "SELECT skill_name FROM pending_writes WHERE origin='differentiate'"
    ).fetchall() == [("zorb-report",)]
    assert await run_differentiation(hook) == [] and chat.call_count == 1


async def test_cap_per_cycle(tmp_path):
    skills = {f"zorb-{i}": (f"Build zorblax thing {i}", "agent") for i in range(4)}
    hook, _ = _setup(tmp_path, skills, differentiation_max_per_cycle=2)
    await _index(hook, skills)
    chat = _reply(hook, _NEW)
    await run_differentiation(hook)
    assert chat.call_count == 2


async def test_disabled_by_default(tmp_path):
    hook = nano_hermes.install(_make_loop(tmp_path))
    assert await run_differentiation(hook) == []


def test_migration_is_idempotent(tmp_path):
    hook = nano_hermes.install(_make_loop(tmp_path))
    hook.db.execute("PRAGMA user_version = 0")
    _apply_migrations(hook.db)
    _apply_migrations(hook.db)
    cols = {r[1] for r in hook.db.execute("PRAGMA table_info(skill_stats)")}
    assert "differentiated_hash" in cols
