"""Sidecar log for tool-call details.

Design goals:
- The main log (stdout / subprocess log) keeps only a **single line** tool-event
  summary, e.g.:
    [synthesize] tool result event=ATTEMPT index=16 tool=docker_cli id=toolu_...
- The full tool-call arguments, return results, docker_cli stdout/stderr and other
  "heavy content" are all written to a separate sidecar file, consulted only when
  investigating errors, so the main log stays clean in normal operation.

Each synthesize subprocess binds its own sidecar file via configure() (derived from
output_path), so when run_all_instances.py runs multiple subprocesses in parallel
each writes to its own file without interfering with the others.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from synthesize_flow_utils import printable_text, sanitize_json_strings

_lock = threading.Lock()
_detail_path: Path | None = None


def configure(path: Any, *, append: bool = False) -> None:
    """Bind the sidecar file path; pass None to disable detail persistence (degrade to only a one-line summary).

    When append=True (resume), append rather than truncate, preserving details from previous rounds.
    """
    global _detail_path
    if not path:
        _detail_path = None
        return
    _detail_path = Path(path)
    try:
        _detail_path.parent.mkdir(parents=True, exist_ok=True)
        # append=False: truncate old content, ensuring a fresh run corresponds to one clean detail file;
        # append=True: resume, append and write a separator header.
        open_mode = "a" if append else "w"
        with _detail_path.open(open_mode, encoding="utf-8") as f:
            if append:
                f.write(f"\n\n# ===== RESUME @ {datetime.now().isoformat()} =====\n")
            f.write(f"# tool detail log\n# started_at: {datetime.now().isoformat()}\n\n")
    except Exception:
        _detail_path = None


def _write_block(text: str) -> None:
    if _detail_path is None:
        return
    try:
        with _lock:
            with _detail_path.open("a", encoding="utf-8") as f:
                f.write(text)
                if not text.endswith("\n"):
                    f.write("\n")
    except Exception:
        pass


def log_payload(label: str, payload: Any) -> None:
    """Full payload written to the sidecar file; tool call/result summary no longer pollutes the main log."""
    if _detail_path is None:
        return
    try:
        body = json.dumps(sanitize_json_strings(payload), ensure_ascii=False, indent=2)
    except Exception:
        body = printable_text(payload)
    _write_block(
        f"{'=' * 80}\n[{datetime.now().isoformat()}] {label}\n{'-' * 80}\n{body}\n"
    )


def log_text(label: str, text: str) -> None:
    """Plain-text block (e.g. docker_cli stdout/stderr) written only to the sidecar file; nothing goes to the main log."""
    if _detail_path is None:
        return
    _write_block(
        f"{'=' * 80}\n[{datetime.now().isoformat()}] {label}\n{'-' * 80}\n"
        f"{printable_text(text)}\n"
    )
