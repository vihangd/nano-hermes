"""skill_search hides the skills nanobot keeps out of this turn (nanobot > 0.4.1)."""
from __future__ import annotations

from unittest.mock import AsyncMock

import nano_hermes
from conftest import _make_loop
from nano_hermes.skills.indexer import SkillHit
from nano_hermes.skills.tool import SkillSearchTool


class _Context:
    @classmethod
    def _excluded_skills(cls, include_memory: bool, has_cron: bool) -> set[str]:
        # nanobot 0.4.2 ContextBuilder._excluded_skills
        out = set() if include_memory else {"memory"}
        if not has_cron:
            out.add("cron")
        return out


def _hits(*names):
    return [SkillHit(name=n, description=f"{n} skill", location="x", distance=0.1) for n in names]


async def _search(tmp_path, *, has_cron: bool, persists: bool, monkeypatch, context=None):
    loop = _make_loop(tmp_path)
    hook = nano_hermes.install(loop)
    hook.skill_indexer.search = AsyncMock(return_value=_hits("cron", "memory", "weather"))
    monkeypatch.setattr(loop, "context", context or _Context(), raising=False)
    monkeypatch.setattr(loop.tools, "has", lambda name: has_cron and name == "cron", raising=False)
    monkeypatch.setattr("nano_hermes.skills.tool.request_persists", lambda: persists)
    tool = SkillSearchTool(hook=hook)
    return await tool.execute(query="anything")


async def test_cron_hidden_without_the_cron_tool(tmp_path, monkeypatch):
    out = await _search(tmp_path, has_cron=False, persists=True, monkeypatch=monkeypatch)
    assert "cron" not in out and "memory" in out and "weather" in out


async def test_memory_hidden_in_a_session_nanobot_wont_keep(tmp_path, monkeypatch):
    out = await _search(tmp_path, has_cron=True, persists=False, monkeypatch=monkeypatch)
    assert "memory" not in out and "cron" in out


async def test_hidden_skills_do_not_eat_the_top_k(tmp_path, monkeypatch):
    loop = _make_loop(tmp_path)
    hook = nano_hermes.install(loop)
    search = AsyncMock(return_value=_hits("cron", "memory", "weather", "maps"))
    hook.skill_indexer.search = search
    monkeypatch.setattr(loop, "context", _Context(), raising=False)
    monkeypatch.setattr(loop.tools, "has", lambda name: False, raising=False)
    monkeypatch.setattr("nano_hermes.skills.tool.request_persists", lambda: False)
    out = await SkillSearchTool(hook=hook).execute(query="anything", k=2)
    assert search.call_args.kwargs["k"] == 4
    assert "weather" in out and "maps" in out and "cron" not in out


async def test_all_hidden_says_no_match(tmp_path, monkeypatch):
    loop = _make_loop(tmp_path)
    hook = nano_hermes.install(loop)
    hook.skill_indexer.search = AsyncMock(return_value=_hits("cron"))
    monkeypatch.setattr(loop, "context", _Context(), raising=False)
    monkeypatch.setattr(loop.tools, "has", lambda name: False, raising=False)
    out = await SkillSearchTool(hook=hook).execute(query="anything")
    assert out == "no matching skills"


async def test_older_nanobot_hides_nothing(tmp_path, monkeypatch):
    out = await _search(tmp_path, has_cron=False, persists=False, monkeypatch=monkeypatch,
                        context=object())
    assert "cron" in out and "memory" in out
