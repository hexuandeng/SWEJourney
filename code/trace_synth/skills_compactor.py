"""After each data item finishes generating, run one "compact + dedup" pass over the host-side skills experience library.

Purpose: skills files get repeatedly appended via patch_skills during the first round of
PLAN/ATTEMPT/AUGMENT, so they grow without bound and fill up with near-duplicate entries.
Here, after an instance runs fully to completion (DONE), the same model used for the rubric
(azure gpt-5.4) compacts and rewrites each of the three skills files separately, reducing the
entry count without losing coverage.
"""
from __future__ import annotations

from pathlib import Path

from skills_manager import SKILL_DOC_FILES, stage_file_for_event_kind

# filename -> event_kind of the stage it belongs to (used for logging and prompt placeholders).
_FILE_TO_STAGE = {
    "planning.md": "PLAN",
    "test_f2p.md": "ATTEMPT",
    "augment_p2p.md": "AUGMENT_TESTS",
}

_COMPACT_PROMPT_NAME = "compact_skills.txt"


def _log(msg: str) -> None:
    print(f"[skills-compact] {msg}", flush=True)


async def _compact_one_file(*, client, prompt_template: str, filename: str, content: str) -> str | None:
    """Call gpt-5.4 to compact a single file, returning the full compacted text; return None on failure."""
    stage = _FILE_TO_STAGE.get(filename, "UNKNOWN")
    prompt = (
        prompt_template
        .replace("{{STAGE}}", stage)
        .replace("{{FILENAME}}", filename)
        .replace("{{CONTENT}}", content)
    )
    messages = [{"role": "user", "content": prompt}]
    body = await client.chat_once(messages, usage_stage="SKILLS_COMPACT")
    new_text = client.client_response_text(body)
    if not isinstance(new_text, str):
        return None
    new_text = new_text.strip()
    # Strip an overall ``` code fence the model may have mistakenly added.
    if new_text.startswith("```"):
        lines = new_text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        new_text = "\n".join(lines).strip()
    if not new_text:
        return None
    return new_text


async def compact_skills_after_instance(
    *,
    client,
    skills_dir: str | Path | None,
    prompt_dir: str | Path,
    label: str = "",
) -> None:
    """Compact the three skills files after an instance is DONE. All exceptions are swallowed (compaction is optional maintenance, never affects the main artifact).

    - client: the same gpt-5.4 client used for rubric / json-repair.
    - skills_dir: the skills directory; skip directly if empty / nonexistent.
    """
    try:
        if not skills_dir:
            return
        sdir = Path(skills_dir)
        if not sdir.is_dir():
            return
        prompt_path = Path(prompt_dir) / _COMPACT_PROMPT_NAME
        if not prompt_path.is_file():
            _log(f"prompt missing: {prompt_path}; skip")
            return
        prompt_template = prompt_path.read_text(encoding="utf-8")

        _log(f"compacting skills label={label}")
        for filename in SKILL_DOC_FILES:
            fp = sdir / filename
            try:
                if not fp.is_file():
                    continue
                old = fp.read_text(encoding="utf-8")
                if not old.strip():
                    continue
                old_lines = old.count("\n") + 1
                new_text = await _compact_one_file(
                    client=client,
                    prompt_template=prompt_template,
                    filename=filename,
                    content=old,
                )
                if not new_text:
                    _log(f"{filename}: empty/failed compaction; keep original")
                    continue
                # Safety valve: the compacted result should not end up significantly longer (>1.1x is treated as the model not compacting, skip overwrite).
                if len(new_text) > int(len(old) * 1.1) + 200:
                    _log(f"{filename}: compaction longer than original ({len(old)}->{len(new_text)} chars); keep original")
                    continue
                fp.write_text(new_text if new_text.endswith("\n") else new_text + "\n", encoding="utf-8")
                new_lines = new_text.count("\n") + 1
                _log(f"{filename}: {old_lines}->{new_lines} lines, {len(old)}->{len(new_text)} chars")
            except Exception as e:  # noqa: BLE001 - a single-file failure does not affect other files
                _log(f"{filename}: error {type(e).__name__}: {str(e)[:200]}; keep original")
        _log("done")
    except Exception as e:  # noqa: BLE001 - compaction never affects the main flow
        _log(f"skipped due to error {type(e).__name__}: {str(e)[:200]}")
