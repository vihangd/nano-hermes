"""Offline CLI for the write-approval pending store.

Review/resolve staged autonomous writes without a running agent::

    nano-hermes pending <workspace> list
    nano-hermes pending <workspace> diff <id>
    nano-hermes pending <workspace> approve <id>
    nano-hermes pending <workspace> reject <id>

Skill approvals are fully offline (write SKILL.md + snapshot, no network).
Principle approvals re-run the curator ops, which need embeddings — those must
be approved from the running agent via the ``pending_review`` tool.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from ..config import NanoHermesConfig
from ..session.db import open_db
from . import write_approval as wa

log = logging.getLogger(__name__)


def _load_config(workspace: Path) -> NanoHermesConfig:
    """Resolve the same nano-hermes config the running hook would use, so the
    CLI opens the DB with the *correct* embedding dims (a wrong value would make
    open_db rebuild the vec0 tables at the wrong dimension) and honours the
    configured snapshot_retain."""
    from .. import _load_config_files  # noqa: PLC0415

    try:
        return NanoHermesConfig.model_validate(_load_config_files(workspace))
    except Exception:
        log.warning(
            "pending: config load failed for %s — falling back to DEFAULTS "
            "(write-approval governance posture reset to off)", workspace, exc_info=True
        )
        return NanoHermesConfig.model_validate({})


def _run(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    workspace = Path(argv[0]).expanduser().resolve()
    action = argv[1]
    cfg = _load_config(workspace)
    db = open_db(workspace, cfg.embedding.target_dims)

    if action == "list":
        rows = wa.list_pending(db)
        if not rows:
            print("No pending writes.")
            return 0
        for r in rows:
            print(
                f"#{r['id']} | {r['subsystem']} | {r['skill_name'] or '-'} | "
                f"{r['origin']} | {r['reason']}"
            )
        return 0

    if len(argv) < 3:
        print(f"error: '{action}' needs an id")
        return 2
    pid = int(argv[2])

    if action == "diff":
        print(wa.diff_pending(db, workspace, pid))
        return 0
    if action == "reject":
        print(wa.reject(db, pid))
        return 0
    if action == "approve":
        rec = wa.get_pending(db, pid)
        if not rec or rec["status"] != "pending":
            print(f"no open pending write #{pid}")
            return 1
        if rec["subsystem"] == "principles":
            print(
                f"#{pid} is a principle write — approve it from the running agent "
                "(pending_review tool); offline approval can't compute embeddings."
            )
            return 1
        print(wa.approve_skill(
            db, workspace, pid, snapshot_retain=cfg.skill_stats.snapshot_retain
        ))
        return 0

    print(f"unknown action {action!r}")
    return 2


def main() -> None:  # pragma: no cover - thin argv wrapper
    raise SystemExit(_run(sys.argv[1:]))


if __name__ == "__main__":  # pragma: no cover
    main()
