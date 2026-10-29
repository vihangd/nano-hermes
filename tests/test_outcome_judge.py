"""Background outcome judge (memory/outcome_judge.py): Jev primary, LLM fallback."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from nanobot.agent.hook import AgentHookContext

import nano_hermes
from conftest import _make_loop
from nano_hermes.memory import outcome_judge as oj

_SECRET = "sk-ant-api03-" + "a" * 40


def _hook(tmp_path, **judge):
    cfg = {"outcome_judge": {"enabled": True, **judge}}
    return nano_hermes.install(_make_loop(tmp_path), config=cfg)


def _jev_transport(responses: list, seen: list):
    """httpx transport answering with *responses* in order, recording requests."""
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        status, body = responses.pop(0)
        return httpx.Response(status, json=body)
    return httpx.MockTransport(handler)


@pytest.fixture
def jev(monkeypatch):
    """Route the judge's httpx client to a mock; returns (responses, seen)."""
    responses: list = []
    seen: list = []
    real = httpx.AsyncClient
    monkeypatch.setattr(
        oj.httpx, "AsyncClient",
        lambda **kw: real(transport=_jev_transport(responses, seen), **kw),
    )
    monkeypatch.setattr(oj.asyncio, "sleep", AsyncMock())
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    return responses, seen


def _answers(p_ok, p_fail):
    return {"model": "jev-1.14.0", "answers": {
        "accomplished": {"type": "noul", "noul": p_ok},
        "failed": {"type": "noul", "noul": p_fail},
    }}


@pytest.mark.parametrize(
    "p_ok, p_fail, verdict",
    [
        (0.97, 0.04, "ok"),
        (0.03, 0.95, "fail"),
        (0.80, 0.20, None),   # below min_confidence
        (0.95, 0.40, None),   # pair disagrees (sums to 1.35): noise
    ],
)
def test_jev_verdict_thresholds(p_ok, p_fail, verdict):
    assert oj._jev_verdict(_answers(p_ok, p_fail)["answers"], 0.9)[0] == verdict


def test_jev_verdict_missing_answer_is_no_verdict():
    assert oj._jev_verdict({"accomplished": {"noul": 0.99}}, 0.9) == (None, {})


async def test_jev_request_shape_and_redaction(tmp_path, jev):
    responses, seen = jev
    responses.append((200, _answers(0.97, 0.02)))
    hook = _hook(tmp_path)
    msgs = [
        {"role": "user", "content": f"deploy with key {_SECRET}"},
        {"role": "assistant", "content": "deployed"},
    ]
    result = await oj.judge(hook, msgs)
    assert result["verdict"] == "ok" and result["backend"] == "jev"
    req = seen[0]
    assert str(req.url) == "https://api.typesafe.ai/v1/systemone"
    assert req.headers["Authorization"] == "Bearer test-key"
    body = json.loads(req.content)
    assert body["model"] == "jev-1.14.0"
    assert set(body["questions"]) == {"accomplished", "failed"}
    assert all(q["type"] == "noul" for q in body["questions"].values())
    assert _SECRET not in json.dumps(body["state"])


async def test_jev_retries_overload_then_answers(tmp_path, jev):
    responses, seen = jev
    responses.extend([(529, {}), (429, {}), (200, _answers(0.02, 0.97))])
    result = await oj.judge(_hook(tmp_path), [{"role": "user", "content": "x"}])
    assert result["verdict"] == "fail" and len(seen) == 3


async def test_missing_key_falls_back_to_llm(tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    hook = _hook(tmp_path)
    hook._loop.provider = SimpleNamespace(
        chat_with_retry=AsyncMock(return_value=SimpleNamespace(
            content="SUCCESS", finish_reason="stop", error=None
        ))
    )
    result = await oj.judge(hook, [{"role": "user", "content": "x"}])
    assert result == {"backend": "llm", "model": hook.background_model, "verdict": "ok"}


async def test_jev_failure_without_fallback_gives_nothing(tmp_path, jev):
    responses, _ = jev
    responses.append((401, {"error": "bad key"}))
    hook = _hook(tmp_path, fallback="none")
    assert await oj.judge(hook, [{"role": "user", "content": "x"}]) is None


def test_fit_transcript_keeps_the_end():
    msgs = [{"role": "user", "content": f"msg {i} " + "word " * 200} for i in range(50)]
    task, transcript = oj._fit_transcript(msgs, 2000)
    assert task.startswith("msg 0")
    assert "msg 49" in transcript and "msg 0 " not in transcript


# --- boundary integration ---------------------------------------------------

async def _session(hook, first: str, key="k", error=None):
    msgs = [{"role": "user", "content": first}]
    await hook.before_iteration(AgentHookContext(iteration=0, messages=msgs, session_key=key))
    msgs.append({"role": "assistant", "content": "done"})
    await hook.after_iteration(
        AgentHookContext(iteration=0, messages=msgs, session_key=key, error=error)
    )


async def _end(hook):
    await _session(hook, "an unrelated new task")
    await asyncio.gather(*hook._lesson_tasks)
    return hook.db.execute(
        "SELECT outcome, outcome_source, outcome_signals FROM trajectories ORDER BY id"
    ).fetchone()


async def test_shadow_records_the_verdict_but_keeps_the_rule_label(tmp_path, jev):
    jev[0].append((200, _answers(0.98, 0.01)))
    hook = _hook(tmp_path)
    await _session(hook, "write a haiku")
    outcome, source, signals = await _end(hook)
    assert (outcome, source) == ("unknown", "rules")
    assert json.loads(signals)["judge"]["verdict"] == "ok"


async def test_apply_sets_outcome_then_extracts_lessons_with_it(tmp_path, jev):
    jev[0].append((200, _answers(0.98, 0.01)))
    hook = _hook(tmp_path, mode="apply")
    hook.config.skill_stats.cheatsheet_enabled = True
    seen: list = []

    async def capture(messages, outcome):
        seen.append(outcome)

    hook._extract_cheatsheet_lesson = capture
    await _session(hook, "write a haiku")
    outcome, source, _ = await _end(hook)
    await asyncio.gather(*hook._lesson_tasks)
    assert (outcome, source) == ("ok", "judge:jev")
    assert seen == ["ok"]


async def test_hard_failure_never_reaches_the_judge(tmp_path, jev):
    hook = _hook(tmp_path, mode="apply")
    await _session(hook, "write a haiku", error="boom")
    outcome, source, signals = await _end(hook)
    assert (outcome, source) == ("fail", "rules")
    assert jev[1] == [] and "judge" not in json.loads(signals)
