"""LongMemEval adapter: loader, session scoring, and an end-to-end fake run."""
from __future__ import annotations

import json

import pytest

from evals.longmemeval import format_table, load, run, session_scores


def _item(qid, question, gold_session, evidence):
    other = [{"role": "user", "content": "what should I cook tonight"},
             {"role": "assistant", "content": "try a lentil curry"}]
    hit = [{"role": "user", "content": evidence},
           {"role": "assistant", "content": "noted"}]
    return {
        "question_id": qid, "question_type": "single-session-user",
        "question": question, "answer": "x",
        "haystack_session_ids": ["s_other", gold_session],
        "haystack_sessions": [other, hit],
        "answer_session_ids": [gold_session],
    }


def test_session_scores_hand_case():
    m = session_scores(["a", "b", "g1", "c", "d", "e", "g2"], {"g1", "g2"})
    assert m["any@5"] == 1.0 and m["all@5"] == 0.0
    assert m["all@10"] == 1.0
    ideal = 1 + 1 / __import__("math").log2(3)
    assert m["ndcg@10"] == pytest.approx((1 / 2 + 1 / 3) / ideal)


def test_abstention_skipped_and_limit(tmp_path):
    data = [_item("q1", "a", "g", "b"), _item("q2_abs", "a", "g", "b"), _item("q3", "a", "g", "b")]
    p = tmp_path / "d.json"
    p.write_text(json.dumps(data))
    assert [q["question_id"] for q in load(p, None)] == ["q1", "q3"]
    assert [q["question_id"] for q in load(p, 1)] == ["q1"]


def test_fake_run_end_to_end_lexical_finds_the_evidence_session(tmp_path):
    data = [
        _item("q1", "which kubernetes namespace did I deploy to", "g1",
              "I deployed it to the kubernetes namespace payments-prod"),
        _item("q2", "what is my dentist called", "g2",
              "my dentist is called Dr. Okafor"),
    ]
    p = tmp_path / "d.json"
    p.write_text(json.dumps(data))
    import asyncio

    res = asyncio.run(run(p, None, "fake"))
    assert res["n"] == 2
    assert res["arms"]["fts_only"]["ndcg@10"] == 1.0  # evidence session ranked first
    assert "fts_only" in format_table(res)
