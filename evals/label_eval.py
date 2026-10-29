"""Measure session outcome-label precision against hand labels.

    python evals/label_eval.py <workspace> export [--n 50] > labels.jsonl
    # fill each line's "human" with ok | partial | fail | unknown
    python evals/label_eval.py <workspace> import labels.jsonl
    python evals/label_eval.py <workspace> report

``report`` prints precision/recall of the stored label per value and source
(rules / judge:…), and for Jev verdicts recorded in shadow or apply mode, how
the verdict would change at other ``outcome_judge.min_confidence`` values.
Learned ranking on outcomes (ROADMAP) waits for ok/fail precision ≥ ~0.85.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from nano_hermes.governance.pending import _load_config
from nano_hermes.memory.outcome_judge import _jev_verdict
from nano_hermes.session.db import open_db
from nano_hermes.session.outcome import OUTCOMES, _current_text

_TAIL_CHARS = 3000
_THRESHOLDS = (0.6, 0.7, 0.8, 0.9, 0.95)


def _transcript_tail(db, session_id: int) -> str:
    rows = db.execute(
        "SELECT role, content FROM chunks WHERE session_id = ? AND role IN ('user', 'assistant') "
        "ORDER BY id",
        (session_id,),
    ).fetchall()
    text = "\n\n".join(f"{r.upper()}: {_current_text(c) if r == 'user' else c}" for r, c in rows)
    return text[-_TAIL_CHARS:]


def export(db, n: int) -> None:
    rows = db.execute(
        "SELECT id, session_id, task, outcome, outcome_source, outcome_signals FROM trajectories "
        "WHERE session_id IS NOT NULL AND outcome_source IS NOT NULL ORDER BY id DESC LIMIT ?",
        (n,),
    ).fetchall()
    for tid, sid, task, outcome, source, signals in rows:
        print(json.dumps({
            "id": tid, "task": task, "outcome": outcome, "source": source,
            "signals": json.loads(signals) if signals else None,
            "transcript_tail": _transcript_tail(db, sid), "human": "",
        }))


def import_labels(db, path: Path) -> int:
    n = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        human = (rec.get("human") or "").strip().lower()
        if human not in OUTCOMES:
            continue
        db.execute("UPDATE trajectories SET human_outcome = ? WHERE id = ?", (human, rec["id"]))
        n += 1
    db.commit()
    return n


def _pr(pairs: list[tuple[str, str]]) -> list[str]:
    """Per-label precision/recall lines for (predicted, human) pairs."""
    out = []
    for value in OUTCOMES:
        pred = [h for p, h in pairs if p == value]
        actual = [p for p, h in pairs if h == value]
        if not pred and not actual:
            continue
        prec = sum(h == value for h in pred) / len(pred) if pred else float("nan")
        rec = sum(p == value for p in actual) / len(actual) if actual else float("nan")
        out.append(f"  {value:8} precision {prec:5.2f} (n={len(pred):3})  "
                   f"recall {rec:5.2f} (n={len(actual):3})")
    return out


def report(db) -> list[str]:
    rows = db.execute(
        "SELECT outcome, outcome_source, outcome_signals, human_outcome FROM trajectories "
        "WHERE human_outcome IS NOT NULL"
    ).fetchall()
    if not rows:
        return ["No hand-labelled sessions yet (export, fill 'human', import)."]
    lines = [f"{len(rows)} labelled sessions; human labels: "
             + ", ".join(f"{k}={v}" for k, v in Counter(r[3] for r in rows).items())]
    for source in sorted({r[1] or "legacy" for r in rows}):
        lines.append(f"stored label, source={source}:")
        lines += _pr([(r[0], r[3]) for r in rows if (r[1] or "legacy") == source])
    jev = []
    for outcome, _, signals, human in rows:
        judge = (json.loads(signals) if signals else {}).get("judge") or {}
        if judge.get("backend") == "jev" and "p_accomplished" in judge:
            jev.append((judge["p_accomplished"], judge["p_failed"], human))
    for t in _THRESHOLDS if jev else ():
        pairs = [
            (_jev_verdict({"accomplished": {"noul": p_ok}, "failed": {"noul": p_fail}}, t)[0]
             or "unknown", human)
            for p_ok, p_fail, human in jev
        ]
        lines.append(f"jev verdict at min_confidence={t}:")
        lines += _pr(pairs)
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("workspace", type=Path)
    sub = ap.add_subparsers(dest="action", required=True)
    sub.add_parser("export").add_argument("--n", type=int, default=50)
    sub.add_parser("import").add_argument("file", type=Path)
    sub.add_parser("report")
    args = ap.parse_args(argv)
    workspace = args.workspace.expanduser().resolve()
    db = open_db(workspace, _load_config(workspace).embedding.target_dims)
    try:
        if args.action == "export":
            export(db, args.n)
        elif args.action == "import":
            print(f"imported {import_labels(db, args.file)} labels", file=sys.stderr)
        else:
            print("\n".join(report(db)))
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
