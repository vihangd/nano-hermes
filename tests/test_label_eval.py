"""evals/label_eval.py: export → hand label → import → report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

import label_eval  # noqa: E402

from nano_hermes.session.db import open_db  # noqa: E402


def _seed(db, outcome, source, signals, user_text):
    sid = db.execute(
        "INSERT INTO sessions (session_key, started_at, ended_at) VALUES ('k', 0, 1)"
    ).lastrowid
    db.execute(
        "INSERT INTO chunks (session_id, turn_index, role, content, created_at) "
        "VALUES (?, 0, 'user', ?, 0)",
        (sid, user_text),
    )
    db.execute(
        "INSERT INTO trajectories (session_id, task, outcome, outcome_source, outcome_signals, "
        "created_at) VALUES (?, 't', ?, ?, ?, 0)",
        (sid, outcome, source, json.dumps(signals)),
    )
    db.commit()


def test_round_trip(tmp_path, capsys):
    db = open_db(tmp_path, 4)
    judge = {"backend": "jev", "verdict": None, "p_accomplished": 0.85, "p_failed": 0.1}
    _seed(db, "ok", "rules", {}, "do a\n[Runtime Context]\nchannel: x")
    _seed(db, "unknown", "rules", {"judge": judge}, "do b")
    db.close()

    label_eval.main([str(tmp_path), "export", "--n", "10"])
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(records) == 2
    assert all("[Runtime Context]" not in r["transcript_tail"] for r in records)
    for r in records:
        r["human"] = "ok"
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join(json.dumps(r) for r in records))
    label_eval.main([str(tmp_path), "import", str(labels)])

    label_eval.main([str(tmp_path), "report"])
    out = capsys.readouterr().out
    assert "2 labelled sessions" in out
    assert "ok       precision  1.00 (n=  1)  recall  0.50 (n=  2)" in out
    # Jev's 0.85 counts as ok at 0.8 but not at 0.9.
    at_08 = out.split("min_confidence=0.8:")[1].split("jev verdict")[0]
    at_09 = out.split("min_confidence=0.9:")[1].split("jev verdict")[0]
    assert "ok       precision  1.00 (n=  1)" in at_08
    assert "ok       precision   nan (n=  0)" in at_09
