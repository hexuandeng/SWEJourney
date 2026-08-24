from __future__ import annotations

import datetime
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

# event_kind (synthesis stage) -> skills filename dedicated to that stage.
# These three stages are the ones that go through the call_json_with_docker_cli_messages
# tool loop, and therefore can access read_skills / patch_skills; user_query / verify are not.
_STAGE_FILE_BY_EVENT_KIND = {
    "PLAN": "planning.md",
    "ATTEMPT": "test_f2p.md",
    "AUGMENT_TESTS": "augment_p2p.md",
}

# Official experience files the model may patch (fixed order, for easy comparison).
SKILL_DOC_FILES = ("planning.md", "test_f2p.md", "augment_p2p.md")
# All files copied during a snapshot.
ALL_SKILL_FILES = SKILL_DOC_FILES


def stage_file_for_event_kind(event_kind: str) -> str | None:
    """Map a synthesis stage's event_kind to its dedicated skills filename; return None for non-skills stages."""
    return _STAGE_FILE_BY_EVENT_KIND.get(str(event_kind or "").strip())


class SkillsManager:
    """host-side manager for the skills experience library.

    Skills always live outside docker (host filesystem), so they persist permanently
    across stages and across containers.
    Responsibilities:
    - read_skills: return the current stage's dedicated file (only this one, no other stages' files).
    - patch_skills: apply the model's unified diff to the skills directory via git apply;
      on success, snapshot the whole skills directory to history_dir/<instance>/NNN_<stage>/.
    - On apply failure, return git's error verbatim to the model so it can rewrite the patch and retry.
    """

    def __init__(self, *, skills_dir: str | Path, history_dir: str | Path, instance: str):
        self.skills_dir = Path(skills_dir)
        self.history_dir = Path(history_dir)
        self.instance = str(instance or "unknown")
        self._snapshot_seq = 0
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        # Ensure all files exist (create empty skeletons if missing, to avoid read / patch missing-file errors).
        for name in ALL_SKILL_FILES:
            fp = self.skills_dir / name
            if not fp.exists():
                fp.write_text("", encoding="utf-8")
        self._init_snapshot_seq()

    # ---- history snapshot numbering ----
    def _instance_history_dir(self) -> Path:
        return self.history_dir / self.instance

    def _init_snapshot_seq(self) -> None:
        """Restore numbering from existing history dir, so resumed runs keep incrementing the sequence."""
        base = self._instance_history_dir()
        if not base.exists():
            self._snapshot_seq = 0
            return
        max_seq = 0
        for child in base.iterdir():
            if not child.is_dir():
                continue
            head = child.name.split("_", 1)[0]
            if head.isdigit():
                max_seq = max(max_seq, int(head))
        self._snapshot_seq = max_seq

    # ---- read ----
    def _read_file(self, name: str) -> str:
        fp = self.skills_dir / name
        try:
            return fp.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def read(self, event_kind: str) -> dict[str, Any]:
        """Return the current stage's dedicated file content."""
        stage_file = stage_file_for_event_kind(event_kind)
        if stage_file is None:
            return {
                "tool": "read_skills",
                "error": (
                    f"read_skills is not available for stage {event_kind!r}; "
                    "only PLAN / ATTEMPT / AUGMENT_TESTS expose the skills library"
                ),
            }
        return {
            "tool": "read_skills",
            "stage": event_kind,
            "files": {
                stage_file: self._read_file(stage_file),
            },
            "note": (
                "These are the persistent engineering-experience notes. "
                f"Use patch_skills to inherit and enrich them. You may only patch "
                f"'{stage_file}' (this stage). "
                "If you spot entries that are wrong or outdated, correct them; "
                "for entries that have been stable and are clearly correct, avoid "
                "unnecessary changes."
            ),
        }

    # ---- patch ----
    def patch(self, event_kind: str, patch_text: str) -> dict[str, Any]:
        """Apply the unified diff to the skills directory, then snapshot the whole directory on success.

        - The patch may only touch the current stage's dedicated file.
        - On apply failure, return git apply's stderr verbatim to the model to retry.
        """
        stage_file = stage_file_for_event_kind(event_kind)
        if stage_file is None:
            return {
                "tool": "patch_skills",
                "error": (
                    f"patch_skills is not available for stage {event_kind!r}; "
                    "only PLAN / ATTEMPT / AUGMENT_TESTS expose the skills library"
                ),
            }
        if not isinstance(patch_text, str) or not patch_text.strip():
            return {
                "tool": "patch_skills",
                "error": "patch_skills requires a non-empty unified-diff string argument: patch",
            }

        allowed = {stage_file}
        touched = self._touched_paths(patch_text)
        illegal = sorted(p for p in touched if p not in allowed)
        if illegal:
            return {
                "tool": "patch_skills",
                "error": (
                    f"patch touches files outside this stage's allowance {sorted(allowed)}: "
                    f"{illegal}. You may only edit {stage_file}."
                ),
            }

        # Write to a temp file then git apply, to avoid command-line escaping issues.
        with tempfile.NamedTemporaryFile(
            "w", suffix=".patch", delete=False, encoding="utf-8"
        ) as tf:
            tf.write(patch_text if patch_text.endswith("\n") else patch_text + "\n")
            patch_path = tf.name

        try:
            check = subprocess.run(
                ["git", "apply", "--check", "--unsafe-paths", patch_path],
                cwd=str(self.skills_dir),
                capture_output=True,
                text=True,
            )
            if check.returncode != 0:
                return {
                    "tool": "patch_skills",
                    "status": "apply_failed",
                    "error": (check.stderr or check.stdout or "git apply --check failed").strip(),
                    "hint": (
                        "The patch does not apply cleanly to the current skills files. "
                        "Re-read with read_skills and regenerate the diff against the exact "
                        "current content. Do not hand-edit hunk headers."
                    ),
                }
            applied = subprocess.run(
                ["git", "apply", "--unsafe-paths", patch_path],
                cwd=str(self.skills_dir),
                capture_output=True,
                text=True,
            )
            if applied.returncode != 0:
                return {
                    "tool": "patch_skills",
                    "status": "apply_failed",
                    "error": (applied.stderr or applied.stdout or "git apply failed").strip(),
                }
        finally:
            try:
                Path(patch_path).unlink()
            except OSError:
                pass

        snapshot_dir = self._snapshot(event_kind)
        return {
            "tool": "patch_skills",
            "status": "ok",
            "stage": event_kind,
            "patched_files": sorted(touched),
            "snapshot": str(snapshot_dir),
            "message": (
                "Skills updated and snapshotted. The new content is now the latest "
                "experience library and will be read on subsequent steps."
            ),
        }

    @staticmethod
    def _touched_paths(patch_text: str) -> set[str]:
        """Parse the changed filenames from the unified diff (stripping a/ b/ prefixes)."""
        touched: set[str] = set()
        for line in patch_text.splitlines():
            if line.startswith("+++ ") or line.startswith("--- "):
                raw = line[4:].strip()
                if raw in ("/dev/null", ""):
                    continue
                if raw.startswith("a/") or raw.startswith("b/"):
                    raw = raw[2:]
                # Strip a possible tab + timestamp
                raw = raw.split("\t", 1)[0].strip()
                if raw:
                    touched.add(raw)
            elif line.startswith("diff --git "):
                parts = line.split()
                for tok in parts[2:]:
                    if tok.startswith("a/") or tok.startswith("b/"):
                        touched.add(tok[2:])
        return touched

    def _snapshot(self, event_kind: str) -> Path:
        """Copy the whole skills directory to history_dir/<instance>/NNN_<stage>/."""
        self._snapshot_seq += 1
        stage = str(event_kind or "stage").strip().lower()
        name = f"{self._snapshot_seq:03d}_{stage}"
        dest = self._instance_history_dir() / name
        dest.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        for fname in ALL_SKILL_FILES:
            src = self.skills_dir / fname
            if src.exists():
                shutil.copy2(src, dest / fname)
        # Record a line of metadata for traceability.
        (dest / "_meta.txt").write_text(
            f"instance={self.instance}\nstage={event_kind}\nseq={self._snapshot_seq}\ntime={ts}\n",
            encoding="utf-8",
        )
        return dest
