"""The ``propose_skill`` agent-facing Tool — writes a new SKILL.md to disk.

Skills start in ``draft`` status and are promoted to ``active`` after N
successful uses (tracked by the skill_stats table). Drafts behave exactly
like active skills in search — only ``deprecated`` skills are filtered out.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool, tool_parameters

from .._atomic import atomic_write_text
from ..redact import RedactionResult, format_redaction_note, redact
from .guard import scan_skill_content, scan_skill_file

if TYPE_CHECKING:
    from ..hook import NanoHermesHook

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

# Companion files must live under scripts/, references/, or assets/ with a
# simple filename (letters, digits, dots, hyphens, underscores, max 128 chars).
_FILE_PATH_RE = re.compile(
    r"^(scripts|references|assets)/[A-Za-z0-9._-]{1,128}$"
)

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["create", "edit", "patch"],
            "description": (
                "'create' (default) writes a new draft skill — fails if an "
                "active or draft skill with that name already exists. "
                "'edit' rewrites the SKILL.md of an existing active or draft "
                "skill (full body replacement, may add/remove companion files). "
                "'patch' is a surgical find-and-replace inside SKILL.md (or one "
                "companion file via file_path) — preferred for typos and "
                "single-line fixes since it avoids re-emitting the whole body."
            ),
        },
        "name": {
            "type": "string",
            "description": (
                "Skill identifier — lowercase letters, digits, hyphens, and "
                "underscores only (e.g. 'fetch-webpage'). Used as the "
                "directory name under workspace/skills/."
            ),
        },
        "description": {
            "type": "string",
            "description": (
                "One-line summary of what the skill does. Required for create "
                "and edit. This is embedded for semantic search, so precision "
                "matters. Ignored for patch — to change the description, use "
                "patch with old_string/new_string targeting the frontmatter line."
            ),
        },
        "body": {
            "type": "string",
            "description": (
                "Full Markdown body of the skill. Required for create and edit. "
                "Follow the structure template in the skill-creator skill's "
                "references (overview, when-to-use, procedure, examples, edge "
                "cases, guidelines). The body is NOT embedded for search — "
                "only name+description is — so don't rely on body prose for "
                "discoverability. Ignored for patch."
            ),
        },
        "files": {
            "type": "array",
            "description": (
                "Optional companion files to write alongside SKILL.md (create "
                "and edit only). Each path must be relative to the skill root "
                "and start with 'scripts/', 'references/', or 'assets/'. "
                "Scripts may be Python (.py), Node (.js/.mjs/.ts), or shell (.sh/.bash). "
                "In edit mode, listed files are created or overwritten."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path, e.g. 'scripts/run.py' or 'references/api.md'.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Text content of the file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
        "delete_files": {
            "type": "array",
            "description": (
                "Relative paths of companion files to delete (edit mode only). "
                "Paths must follow the same rules as 'files'. "
                "Silently skipped if a path does not exist."
            ),
            "items": {"type": "string"},
        },
        "old_string": {
            "type": "string",
            "description": (
                "Text to find for action='patch'. Must match exactly. By "
                "default the match must be unique — provide enough surrounding "
                "context to disambiguate, or set replace_all=true."
            ),
        },
        "new_string": {
            "type": "string",
            "description": (
                "Replacement text for action='patch'. Use an empty string to "
                "delete the matched text."
            ),
        },
        "file_path": {
            "type": "string",
            "description": (
                "For action='patch' only. Companion file to patch instead of "
                "SKILL.md — must be 'scripts/', 'references/', or 'assets/' "
                "followed by a filename. Omit to patch SKILL.md."
            ),
        },
        "replace_all": {
            "type": "boolean",
            "description": (
                "For action='patch'. When true, replace every occurrence of "
                "old_string instead of requiring a unique match. Default false."
            ),
        },
    },
    "required": ["name"],
}


@tool_parameters(_SCHEMA)
class ProposeSkillTool(Tool):
    """Propose a new skill and save it as a draft SKILL.md on disk.

    Use this when you've figured out a reusable procedure that would be
    valuable in future sessions — e.g. a reliable way to call an API,
    a multi-step data-processing recipe, or a debugging checklist.

    Before authoring a skill, read the ``skill-creator`` skill (in the
    ``<skills>`` list) for description craft, body templates, progressive-
    disclosure guidance, and the pre-submission checklist — skipping it
    produces thin skills that fail the promotion gate.

    The skill starts as a *draft*. Use ``skill_rate`` after applying the skill
    to record whether it helped. After enough successful ratings (default 3),
    it promotes to *active*. Skills that chronically fail get *deprecated* and
    stop appearing in search results.

    Rules:
    - Name must be lowercase, use hyphens/underscores, no spaces or slashes.
    - Cannot overwrite an existing active or draft skill (use a new name,
      or wait for deprecation then re-propose).
    - Deprecated skills CAN be re-proposed with fresh content.
    - Companion files (scripts/, references/, assets/) are optional.
      Script files may be Python, Node, or shell — security scanning is
      adjusted per subdirectory so legitimate code constructs are allowed.

    Three actions:
    - ``create`` (default): write a new draft skill.
    - ``edit``: full SKILL.md rewrite of an existing skill, plus optional
      companion file add/remove via ``files`` and ``delete_files``.
    - ``patch``: surgical find-and-replace inside SKILL.md or one companion
      file via ``file_path``. Prefer this for typos and one-line fixes — it
      avoids re-emitting the whole body.
    """

    def __init__(self, *, hook: "NanoHermesHook") -> None:
        self._hook = hook

    @property
    def name(self) -> str:
        return "propose_skill"

    @property
    def description(self) -> str:
        return (type(self).__doc__ or "").strip()

    async def execute(self, **kwargs: Any) -> str:
        action: str = kwargs.get("action") or "create"
        skill_name: str = kwargs.get("name", "").strip()

        # --- Name validation (required for every action) ---
        if not _NAME_RE.match(skill_name):
            return (
                "Error: invalid skill name — must be lowercase letters, "
                "digits, hyphens, or underscores, starting with a letter or "
                f"digit (got '{skill_name}')."
            )

        if action == "patch":
            return await self._patch(skill_name, kwargs)

        if action not in ("create", "edit"):
            return (
                f"Error: unknown action {action!r} — must be 'create', 'edit', "
                "or 'patch'."
            )

        # --- create/edit-only validation ---
        description: str = kwargs.get("description", "").strip()
        body: str = kwargs.get("body", "").strip()
        files: list[dict[str, str]] = kwargs.get("files") or []
        delete_files: list[str] = kwargs.get("delete_files") or []

        if not description:
            return "Error: description must not be empty."
        if not body:
            return "Error: body must not be empty."

        # --- Validate companion file paths ---
        err = self._validate_file_list(files)
        if err:
            return f"Error: {err}"
        if delete_files:
            err = self._validate_file_list(
                [{"path": p, "content": ""} for p in delete_files]
            )
            if err:
                return f"Error: {err}"

        # --- Apply secret redaction (before security scan + size cap so a
        # body that's clean except for embedded secrets still passes scan
        # and so the redacted size is what counts against the cap). ---
        description, body, files, redaction_note = self._apply_redaction(
            description, body, files
        )

        # --- Size cap ---
        max_bytes = self._hook.config.skill_stats.max_skill_bytes
        total = len(body.encode("utf-8")) + sum(
            len(f["content"].encode("utf-8")) for f in files
        )
        if total > max_bytes:
            return (
                f"Error: skill content too large — {total:,} bytes exceeds the "
                f"{max_bytes:,}-byte limit. Reduce body or companion file size."
            )

        if action == "edit":
            return await self._edit(
                skill_name, description, body, files, delete_files, redaction_note
            )
        return await self._create(
            skill_name, description, body, files, redaction_note
        )

    def _validate_file_list(self, files: list[dict[str, str]]) -> str | None:
        seen: set[str] = set()
        for f in files:
            path = f.get("path", "")
            if not _FILE_PATH_RE.match(path):
                return (
                    f"invalid file path {path!r} — must be 'scripts/', "
                    "'references/', or 'assets/' followed by a filename "
                    "(letters, digits, dots, hyphens, underscores, max 128 chars)"
                )
            if path in seen:
                return f"duplicate file path {path!r}"
            seen.add(path)
        return None

    def _apply_redaction(
        self,
        description: str,
        body: str,
        files: list[dict[str, str]],
    ) -> tuple[str, str, list[dict[str, str]], str]:
        """Redact secrets from agent-supplied content.

        Returns ``(description, body, files, note)``. ``note`` is the
        human-readable suffix to append to a success message — empty string
        when nothing was redacted (or when redaction is disabled).
        """
        if not self._hook.config.redact_secrets:
            return description, body, files, ""

        desc_r = redact(description)
        body_r = redact(body)
        new_files: list[dict[str, str]] = []
        file_results: list[RedactionResult] = []
        for f in files:
            cr = redact(f["content"])
            file_results.append(cr)
            new_files.append({"path": f["path"], "content": cr.text})

        total = desc_r.count + body_r.count + sum(r.count for r in file_results)
        if total == 0:
            return desc_r.text, body_r.text, new_files, ""

        kinds: set[str] = set(desc_r.kinds) | set(body_r.kinds)
        for r in file_results:
            kinds.update(r.kinds)
        aggregate = RedactionResult(
            text="", count=total, kinds=tuple(sorted(kinds))
        )
        return desc_r.text, body_r.text, new_files, format_redaction_note(aggregate)

    async def _create(
        self,
        skill_name: str,
        description: str,
        body: str,
        files: list[dict[str, str]],
        redaction_note: str = "",
    ) -> str:
        # --- Conflict check ---
        row = self._hook.db.execute(
            "SELECT status FROM skill_stats WHERE name = ?", (skill_name,)
        ).fetchone()
        if row:
            existing_status = row[0]
            if existing_status in ("active", "draft"):
                return (
                    f"Error: skill '{skill_name}' already exists with status "
                    f"'{existing_status}'. Choose a different name, or wait "
                    f"for the skill to be deprecated before re-proposing."
                )
            # deprecated -> fall through, allow overwrite

        return await self._write_skill(
            skill_name, description, body, files, delete_files=[],
            upsert_sql=(
                "INSERT INTO skill_stats "
                "(name, status, use_count, success_count, last_used_at, provenance, content_hash) "
                "VALUES (?, 'draft', 0, 0, NULL, NULL, NULL) "
                "ON CONFLICT(name) DO UPDATE SET "
                "status = 'draft', "
                "use_count = 0, "
                "success_count = 0, "
                "last_used_at = NULL, "
                "provenance = NULL, "
                "content_hash = NULL, "
                "indexed_at = NULL"
            ),
            success_msg=(
                f"ok: created draft skill '{skill_name}' at "
                f"workspace/skills/{skill_name}/SKILL.md. "
                "It will appear in skill_search after the next search call triggers re-indexing."
                + redaction_note
            ),
        )

    async def _edit(
        self,
        skill_name: str,
        description: str,
        body: str,
        files: list[dict[str, str]],
        delete_files: list[str],
        redaction_note: str = "",
    ) -> str:
        # --- Must exist and not be deprecated ---
        row = self._hook.db.execute(
            "SELECT status FROM skill_stats WHERE name = ?", (skill_name,)
        ).fetchone()
        if not row:
            return (
                f"Error: skill '{skill_name}' not found. "
                "Use action='create' to propose a new skill."
            )
        existing_status = row[0]
        if existing_status == "deprecated":
            return (
                f"Error: skill '{skill_name}' is deprecated. "
                "Use action='create' to re-propose it with fresh counters."
            )

        err = self._refuse_if_external(skill_name)
        if err:
            return err

        return await self._write_skill(
            skill_name, description, body, files, delete_files,
            upsert_sql=(
                "UPDATE skill_stats SET content_hash = NULL, indexed_at = NULL WHERE name = ?"
            ),
            success_msg=(
                f"ok: updated skill '{skill_name}' at "
                f"workspace/skills/{skill_name}/SKILL.md. "
                "Usage stats preserved. Re-indexing will happen on next skill_search call."
                + redaction_note
            ),
            edit_mode=True,
        )

    async def _patch(self, skill_name: str, kwargs: dict[str, Any]) -> str:
        old_string = kwargs.get("old_string")
        new_string = kwargs.get("new_string")
        file_path = kwargs.get("file_path")  # None → SKILL.md
        replace_all = bool(kwargs.get("replace_all", False))

        if not old_string:
            return "Error: old_string is required for action='patch'."
        if new_string is None:
            return (
                "Error: new_string is required for action='patch'. "
                "Use an empty string to delete the matched text."
            )

        # --- Redact secrets in new_string (not old_string — locator must
        # match what's already on disk). Aggregated note is appended to
        # the success message. ---
        redaction_note = ""
        if self._hook.config.redact_secrets:
            r = redact(new_string)
            new_string = r.text
            redaction_note = format_redaction_note(r)

        # --- Must exist and not be deprecated ---
        row = self._hook.db.execute(
            "SELECT status FROM skill_stats WHERE name = ?", (skill_name,)
        ).fetchone()
        if not row:
            return (
                f"Error: skill '{skill_name}' not found. "
                "Use action='create' to propose a new skill."
            )
        if row[0] == "deprecated":
            return (
                f"Error: skill '{skill_name}' is deprecated. "
                "Use action='create' to re-propose it with fresh counters."
            )

        err = self._refuse_if_external(skill_name)
        if err:
            return err

        skill_dir = self._hook.workspace / "skills" / skill_name

        # --- Resolve target ---
        if file_path is None:
            target = skill_dir / "SKILL.md"
            target_label = "SKILL.md"
        else:
            if not _FILE_PATH_RE.match(file_path):
                return (
                    f"Error: invalid file_path {file_path!r} — must be 'scripts/', "
                    "'references/', or 'assets/' followed by a filename "
                    "(letters, digits, dots, hyphens, underscores, max 128 chars)."
                )
            resolved = self._resolve_companion(skill_dir, file_path)
            if resolved is None:
                return f"Error: file_path {file_path!r} escapes skill directory."
            target = resolved
            target_label = file_path

        if not target.exists():
            return f"Error: {target_label} does not exist in skill '{skill_name}'."

        original = target.read_text(encoding="utf-8")

        # --- Find/replace (exact match) ---
        count = original.count(old_string)
        if count == 0:
            return f"Error: old_string not found in {target_label}."
        if count > 1 and not replace_all:
            return (
                f"Error: old_string matches {count} times in {target_label}. "
                "Provide more surrounding context to make the match unique, or "
                "set replace_all=true."
            )

        if replace_all:
            new_content = original.replace(old_string, new_string)
            match_count = count
        else:
            new_content = original.replace(old_string, new_string, 1)
            match_count = 1

        if new_content == original:
            return (
                f"Error: patch is a no-op — old_string and new_string produce "
                f"identical content for {target_label}."
            )

        # --- If patching SKILL.md, frontmatter must still be intact ---
        if file_path is None:
            err = self._validate_frontmatter_intact(new_content, skill_name)
            if err:
                return f"Error: patch would break SKILL.md: {err}"

        # --- Security scan the result ---
        if file_path is None:
            scan_err = scan_skill_content(self._extract_body(new_content))
        else:
            scan_err = scan_skill_file(file_path, new_content)
        if scan_err:
            return f"Error: patch result failed security scan: {scan_err}"

        # --- Size cap (combined size of all files in the skill dir, with
        # the patched target replaced) ---
        err = self._check_size_after_patch(skill_dir, target, new_content)
        if err:
            return f"Error: {err}"

        # --- Apply with rollback ---
        try:
            atomic_write_text(target, new_content)
            with self._hook.db:
                self._hook.db.execute(
                    "UPDATE skill_stats SET content_hash = NULL, indexed_at = NULL "
                    "WHERE name = ?",
                    (skill_name,),
                )
        except Exception as e:
            try:
                atomic_write_text(target, original)
            except Exception:
                pass
            return (
                f"Error: failed to patch {target_label} in '{skill_name}': {e}. "
                "Rolled back — original content restored."
            )

        plural = "s" if match_count > 1 else ""
        return (
            f"ok: patched {target_label} in '{skill_name}' "
            f"({match_count} replacement{plural}). "
            "Re-indexing on next skill_search call."
            + redaction_note
        )

    @staticmethod
    def _validate_frontmatter_intact(content: str, skill_name: str) -> str | None:
        """Return ``None`` if *content* still has a parseable name+description
        frontmatter block whose name matches *skill_name*; otherwise an error.
        """
        if not content.startswith("---"):
            return "frontmatter opener '---' is missing"
        end = content.find("\n---", 3)
        if end == -1:
            return "frontmatter is not closed (no '---' after the opener)"
        raw = content[4:end]
        fm: dict[str, str] = {}
        for line in raw.splitlines():
            if ":" not in line or line.startswith("#"):
                continue
            key, _, value = line.partition(":")
            fm[key.strip()] = value.strip()
        if "name" not in fm:
            return "frontmatter no longer contains 'name'"
        if "description" not in fm or not fm["description"]:
            return "frontmatter no longer contains a non-empty 'description'"
        if fm["name"] != skill_name:
            return (
                f"frontmatter name {fm['name']!r} no longer matches the skill "
                f"directory ({skill_name!r})"
            )
        return None

    @staticmethod
    def _extract_body(content: str) -> str:
        """Return the body of a SKILL.md (everything after the closing '---')."""
        if not content.startswith("---"):
            return content
        end = content.find("\n---", 3)
        if end == -1:
            return content
        # Skip the closing '---' line and any leading newline.
        body_start = end + len("\n---")
        return content[body_start:].lstrip("\n")

    def _check_size_after_patch(
        self, skill_dir: Path, target: Path, new_content: str
    ) -> str | None:
        """Verify the skill's total size still fits the cap after the patch."""
        max_bytes = self._hook.config.skill_stats.max_skill_bytes
        total = 0
        for p in skill_dir.rglob("*"):
            if not p.is_file():
                continue
            if p == target:
                total += len(new_content.encode("utf-8"))
            else:
                try:
                    total += p.stat().st_size
                except OSError:
                    continue
        if total > max_bytes:
            return (
                f"skill content too large after patch — {total:,} bytes exceeds "
                f"the {max_bytes:,}-byte limit."
            )
        return None

    async def _write_skill(
        self,
        skill_name: str,
        description: str,
        body: str,
        files: list[dict[str, str]],
        delete_files: list[str],
        *,
        upsert_sql: str,
        success_msg: str,
        edit_mode: bool = False,
    ) -> str:
        # Security scan the body before touching the filesystem.
        err = scan_skill_content(body)
        if err:
            return f"Error: {err}"

        # Security scan each companion file with per-subdir rules.
        for f in files:
            err = scan_skill_file(f["path"], f["content"])
            if err:
                return f"Error: {err}"

        skill_dir = self._hook.workspace / "skills" / skill_name

        # Guard: if directory exists on disk but no DB row present, refuse.
        # This prevents silently adopting a shell-script-created skill without
        # the user explicitly re-proposing it.
        if skill_dir.exists() and not edit_mode:
            db_row = self._hook.db.execute(
                "SELECT status FROM skill_stats WHERE name = ?", (skill_name,)
            ).fetchone()
            if not db_row:
                return (
                    f"Error: skill directory '{skill_name}' already exists on "
                    "disk but has no skill_stats entry. Delete the directory "
                    "manually and re-propose, or use action='edit' if you "
                    "intended to update an existing skill."
                )

        # Track whether the directory existed BEFORE this call so a fresh-dir
        # create can safely rmtree on rollback (the cleanest cleanup). In all
        # other cases we rely on snapshot/restore to undo individual file changes.
        dir_preexisted = skill_dir.exists()
        skill_dir.mkdir(parents=True, exist_ok=True)

        # Resolve every companion path upfront so a path-traversal attempt
        # fails before any I/O, and so we know exactly which files to snapshot.
        skill_md = skill_dir / "SKILL.md"
        skill_md_content = (
            f"---\nname: {skill_name}\ndescription: {description}\n---\n\n{body}\n"
        )
        write_targets: list[tuple[Path, str]] = [(skill_md, skill_md_content)]

        for f in files:
            resolved = self._resolve_companion(skill_dir, f["path"])
            if resolved is None:
                if not edit_mode and not dir_preexisted:
                    shutil.rmtree(skill_dir, ignore_errors=True)
                return (
                    f"Error: companion file path escapes skill directory: {f['path']!r}"
                )
            write_targets.append((resolved, f["content"]))

        delete_targets: list[Path] = []
        for rel in delete_files:
            resolved = self._resolve_companion(skill_dir, rel)
            if resolved is None:
                if not edit_mode and not dir_preexisted:
                    shutil.rmtree(skill_dir, ignore_errors=True)
                return f"Error: delete_files path escapes skill directory: {rel!r}"
            delete_targets.append(resolved)

        # Snapshot every file we'll touch so we can roll back on failure.
        # `None` means the file did not exist prior — restore == unlink.
        snapshots: list[tuple[Path, str | None]] = []
        for path, _ in write_targets:
            snapshots.append(
                (path, path.read_text(encoding="utf-8") if path.exists() else None)
            )
        for path in delete_targets:
            snapshots.append(
                (path, path.read_text(encoding="utf-8") if path.exists() else None)
            )

        try:
            for path, payload in write_targets:
                atomic_write_text(path, payload)
            for path in delete_targets:
                if path.exists():
                    path.unlink()

            with self._hook.db:
                self._hook.db.execute(upsert_sql, (skill_name,))
        except Exception as e:
            # Restore snapshots in reverse so any newly-created files are
            # cleaned up before any restored writes happen.
            for path, original in reversed(snapshots):
                try:
                    if original is None:
                        if path.exists():
                            path.unlink()
                    else:
                        atomic_write_text(path, original)
                except Exception:
                    # Best-effort restore — surface the original error regardless.
                    pass

            if not edit_mode and not dir_preexisted:
                # Fresh dir we created: rmtree to also clean up any empty
                # subdirs (scripts/, references/, assets/) we made.
                shutil.rmtree(skill_dir, ignore_errors=True)
                return (
                    f"Error: failed to write skill '{skill_name}': {e}. "
                    "Rolled back — the skill directory was removed. "
                    "Fix the underlying issue and retry."
                )
            return (
                f"Error: failed to write skill '{skill_name}': {e}. "
                "Rolled back — original content restored. "
                "Fix the underlying issue and retry."
            )

        return success_msg

    @staticmethod
    def _resolve_companion(skill_dir: Path, rel_path: str) -> Path | None:
        """Resolve a companion file path and verify it stays inside skill_dir."""
        resolved = (skill_dir / rel_path).resolve()
        try:
            resolved.relative_to(skill_dir.resolve())
        except ValueError:
            return None
        return resolved

    def _refuse_if_external(self, skill_name: str) -> str | None:
        """Return an error string if *skill_name* lives ONLY in an external dir.

        External dirs are configured read-only mirrors. Editing or patching
        an external skill in place would require writing outside the
        workspace — instead the agent must copy the skill into
        workspace/skills/ and modify the copy.

        Honors workspace > external precedence: if a workspace copy exists
        (which is what propose_skill would write to), the external one is
        shadowed and the edit/patch proceeds against the workspace copy.
        """
        workspace_skill = self._hook.workspace / "skills" / skill_name / "SKILL.md"
        if workspace_skill.exists():
            return None
        ext_dir = self._hook.skill_indexer.find_external_skill(skill_name)
        if ext_dir is None:
            return None
        return (
            f"Error: skill '{skill_name}' lives in an external directory "
            f"({ext_dir}) and is read-only. Copy it to "
            f"workspace/skills/{skill_name}/ first if you want to modify it."
        )
