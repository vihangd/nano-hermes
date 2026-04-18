"""Atomic file writes via tempfile + os.replace.

A same-directory tempfile guarantees the rename is atomic on the same
filesystem. If the write crashes mid-stream the target file is untouched;
on success the file is fully written. The tempfile is cleaned up on any
error before the exception propagates.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


def atomic_write_text(file_path: Path, content: str, encoding: str = "utf-8") -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(
        dir=str(file_path.parent),
        prefix=f".{file_path.name}.tmp.",
        suffix="",
    )
    tmp_path = Path(tmp_str)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
        # mkstemp creates with 0o600. If the target already exists, preserve
        # its mode bits so atomic rewrites don't silently demote permissions
        # (e.g. a 0o644 file becoming 0o600 after one edit).
        if file_path.exists():
            try:
                shutil.copymode(file_path, tmp_path)
            except OSError:
                pass
        os.replace(tmp_path, file_path)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise
