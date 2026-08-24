from __future__ import annotations

import ast
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from swebench_dockerlib import DockerSession, PatchApplyError
from swebench.harness.utils import get_modified_files, get_new_files


HERE = Path(__file__).resolve().parent
OFFICIAL_ROOT = HERE / "swe_bench_pro_official"
RUN_SCRIPTS_DIR = OFFICIAL_ROOT / "run_scripts"


def is_pro_instance(
    instance: dict | None,
    *,
    image: str | None = None,
    runner_kind: str | None = None,
) -> bool:
    instance = instance or {}
    if str(runner_kind or "").strip():
        return str(runner_kind).strip() == "pro_official"
    image = str(image or "").strip()
    return (
        bool((instance.get("dockerhub_tag") or "").strip())
        or image.startswith("jefzda/sweap-images:")
        or image.startswith("docker.io/jefzda/sweap-images:")
    )


def parse_list_field(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                value = parser(text)
            except Exception:
                continue
            if isinstance(value, list):
                return [str(x) for x in value]
        return [text]
    return []


def selected_tests_for_instance(instance: dict, requested_nodes: list[str], *, official_test_patch: bool) -> list[str]:
    if official_test_patch:
        selected = parse_list_field(instance.get("selected_test_files_to_run"))
        if selected:
            return selected
    selected: list[str] = []
    for node in requested_nodes:
        text = str(node or "").strip()
        if not text:
            continue
        # JS/TS Pro run_scripts generally accept test files, while parser.py
        # emits full names like "file | suite | test".
        selected.append(text.split(" | ", 1)[0].strip() if " | " in text else text)
    return list(dict.fromkeys(selected))


def official_scripts_available(instance_id: str) -> bool:
    script_dir = RUN_SCRIPTS_DIR / instance_id
    return (script_dir / "run_script.sh").exists() and (script_dir / "parser.py").exists()


def _load_official_script(instance_id: str, name: str) -> str:
    path = RUN_SCRIPTS_DIR / instance_id / name
    if not path.exists():
        raise FileNotFoundError(
            f"SWE-bench Pro official script missing: {path}. "
            "Copy scaleapi/SWE-bench_Pro-os run_scripts under swe_bench_pro_official/."
        )
    return path.read_text(encoding="utf-8", errors="replace")


def _same_patch(a: Any, b: Any) -> bool:
    a_text = "\n".join(_patch_parts(a)).strip()
    b_text = "\n".join(_patch_parts(b)).strip()
    return bool(a_text and b_text and a_text == b_text)


def _patch_parts(raw: Any) -> list[str]:
    """Normalize storage shapes while preserving sequential patch semantics."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(part) for part in raw if str(part).strip()]
    if isinstance(raw, str) and raw.strip():
        return [raw]
    return []


def _inplace_checkout_files(
    session: DockerSession, ref: str, files: list[str], *, cwd: str
) -> None:
    """Materialize worktree files from a git ref using in-place overwrite (O_TRUNC).

    Unlike ``git checkout <ref> -- <files>``, the shell redirection
    ``git show <ref>:<file> > <file>`` never calls ``open(O_CREAT|O_EXCL)``.
    This makes the operation immune to the fuse-overlayfs delete-incoherence
    race that causes ``error: unable to create file X: File exists`` under
    concurrent container IO (30-way parallel seed precheck on sweap-images).

    After overwriting the file content we also restore the git mode and update
    the index via ``git update-index --cacheinfo`` so subsequent git operations
    see a coherent worktree state.
    """
    for f in files:
        qf = shlex.quote(f)
        qref = shlex.quote(ref)
        # Ensure parent directory exists (git checkout does this automatically).
        p = session.exec(
            f"cd {shlex.quote(cwd)} && "
            f"mkdir -p $(dirname {qf}) && "
            f"git show {qref}:{qf} > {qf}"
        )
        if p.returncode != 0:
            raise PatchApplyError(
                f"[pro-inplace-checkout-fail] {f}: "
                f"{(p.stderr or p.stdout).strip()}"
            )
        # Restore file mode and keep index coherent.
        ls = session.exec(
            f"cd {shlex.quote(cwd)} && git ls-tree {qref} -- {qf}"
        )
        ls_out = ls.stdout.strip()
        if ls_out:
            parts_ls = ls_out.split()
            if len(parts_ls) >= 3:
                mode, sha = parts_ls[0], parts_ls[2]
                session.exec(
                    f"cd {shlex.quote(cwd)} && "
                    f"chmod {shlex.quote(mode[-3:])} {qf} && "
                    f"git update-index --cacheinfo "
                    f"{shlex.quote(mode)},{shlex.quote(sha)},{qf}"
                )


def _pro_reset_test_files(
    session: DockerSession, *, test_patch: Any, base_commit: str, cwd: str,
) -> None:
    """Pro-specific variant of reset_eval_test_files.

    Uses in-place overwrite (``git show <ref>:<file> > <file>``) instead of
    ``git checkout <ref> -- <files>`` to avoid the fuse-overlayfs EEXIST race
    that triggers ``error: unable to create file X: File exists`` on sweap-images
    under concurrent container IO.  New files added by test_patch are still
    removed with ``rm -f``.
    """
    parts = _patch_parts(test_patch)
    modified: list[str] = []
    new: list[str] = []
    for part in parts:
        try:
            modified.extend(get_modified_files(part))
            new.extend(get_new_files(part))
        except Exception as exc:
            # A part we cannot parse means we cannot enumerate the files it
            # touched, so file-list-based reset will silently MISS them and
            # leak worktree pollution into later git-apply. The worktree
            # snapshot/restore guard in run_tests_under_test_patch is the real
            # safety net now, but a parse failure here is still a quality
            # signal worth surfacing rather than swallowing.
            print(
                f"[_pro_reset_test_files] WARN: cannot parse test_patch part "
                f"for file enumeration ({type(exc).__name__}: {exc}); "
                f"reset may miss files from this part",
                file=sys.stderr,
            )
    new_set = set(new)
    modified = [f for f in modified if f not in new_set]
    if modified:
        ref = base_commit or "HEAD"
        _inplace_checkout_files(session, ref, modified, cwd=cwd)
    if new:
        quoted_new = " ".join(shlex.quote(f) for f in new)
        session.exec(f"cd {shlex.quote(cwd)} && rm -f -- {quoted_new}")


def _apply_official_test_state(session: DockerSession, instance: dict, *, cwd: str) -> None:
    before_cmd = str(instance.get("before_repo_set_cmd") or "").strip()
    checkout_cmd = before_cmd.splitlines()[-1].strip() if before_cmd else ""
    if not checkout_cmd:
        raise PatchApplyError("Pro official test setup missing before_repo_set_cmd")
    # Parse "git checkout <ref> -- <file1> <file2> ..." and use in-place
    # overwrite instead of git checkout's O_CREAT|O_EXCL to avoid EEXIST on
    # fuse-overlayfs under concurrent IO.
    try:
        cmd_parts = shlex.split(checkout_cmd)
    except ValueError as exc:
        raise PatchApplyError(
            f"[pro-before-repo-set-fail] cannot parse checkout_cmd: {exc}"
        )
    if (
        len(cmd_parts) >= 5
        and cmd_parts[0] == "git"
        and cmd_parts[1] == "checkout"
        and cmd_parts[3] == "--"
    ):
        ref = cmd_parts[2]
        files = cmd_parts[4:]
        _inplace_checkout_files(session, ref, files, cwd=cwd)
    else:
        # Unexpected format — fall back to direct exec.
        p = session.exec(f"cd {shlex.quote(cwd)} && {checkout_cmd}", timeout=600)
        if p.returncode != 0:
            raise PatchApplyError(
                "[pro-before-repo-set-fail]\n" + ((p.stderr or p.stdout).strip())
            )


_JEST_PREFIX = "npx jest --verbose --silent "
_JEST_STABLE = "npx jest --verbose --silent --runInBand --forceExit "


def _stabilize_jest(script_text: str) -> str:
    """Force jest to run single-process and exit cleanly.

    75/1000 Pro instances (element-web / matrix-react-sdk family) run tests with
    `npx jest --verbose --silent ...` across multiple workers. These suites crash
    during worker startup (skin `Cannot find module './component-index'` /
    `localStorage is not defined`) -> the worker becomes a <defunct> zombie, the
    jest main process blocks in epoll forever waiting for IPC from the dead
    worker, and the whole `docker exec` hangs permanently (observed a single case
    stuck 6h+).

    Fix: add `--runInBand` to jest (single process, no worker IPC, crash exits
    directly) plus `--forceExit` (force exit after the run to avoid hanging on
    leaked handles). Only the copy written into the container is rewritten; the
    official data files under swe_bench_pro_official/run_scripts are left
    untouched. No-op for non-jest repos; idempotent and does not re-insert when
    these two flags are already present.
    """
    if _JEST_PREFIX not in script_text or "--runInBand" in script_text:
        return script_text
    return script_text.replace(_JEST_PREFIX, _JEST_STABLE)


def _write_official_workspace(session: DockerSession, instance_id: str) -> None:
    session.exec("mkdir -p /workspace")
    run_script = _stabilize_jest(_load_official_script(instance_id, "run_script.sh"))
    session.write("/workspace/run_script.sh", run_script)
    session.write("/workspace/parser.py", _load_official_script(instance_id, "parser.py"))
    session.exec("chmod +x /workspace/run_script.sh /workspace/parser.py")


def _read_output_json(session: DockerSession) -> dict:
    try:
        return json.loads(session.read("/workspace/output.json"))
    except Exception:
        return {"tests": []}


def _read_workspace_file(session: DockerSession, path: str) -> str:
    try:
        return session.read(path)
    except Exception:
        return ""


def _status_from_output_json(output: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in output.get("tests") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        status = str(item.get("status") or "").strip().upper()
        if name:
            out[name] = status or "ERROR"
    return out


def _iter_brace_matched_objects(text: str, anchor: str):
    """Yield substrings that are top-level ``{...}`` JSON objects containing anchor.

    The official mocha parser.py extracts the reporter JSON with the regex
    ``({\\s*"stats".*?"tests":\\s*\\[.*?\\]\\s*})``.  The ``\\[.*?\\]`` is
    *non-greedy*, so it stops at the FIRST nested empty array inside the
    payload (e.g. ``"cookies": []``), truncating the object mid-stream and
    making ``json.loads`` fail -> zero parsed tests -> every requested node is
    reported NOT_FOUND even though the suite ran to completion.

    We instead locate each object by the anchor key (``"stats"``) and walk
    forward with a brace counter (string-literal aware) to find the true
    matching ``}``.  This recovers the entire reporter object regardless of how
    many nested arrays/objects it contains.
    """
    idx = 0
    while True:
        a = text.find(anchor, idx)
        if a < 0:
            return
        start = text.rfind("{", 0, a)
        if start < 0:
            idx = a + len(anchor)
            continue
        depth = 0
        in_str = False
        esc = False
        end = None
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            return
        yield text[start:end]
        idx = end


# Mirror the official mocha parser.py name-construction so reconstructed names
# match the dataset's stored node ids exactly.
_MOCHA_FILE_PAT = re.compile(r"^(?:/app/)?(.*)$")
_MOCHA_TESTFILE_PAT = re.compile(r"(\S+)::")


def _mocha_name(test: dict) -> str:
    file = str(test.get("file") or "")
    m = _MOCHA_FILE_PAT.match(file)
    if m:
        file = m.group(1)
    full_title = str(test.get("fullTitle") or "")
    tf = _MOCHA_TESTFILE_PAT.search(full_title)
    if tf:
        file = tf.group(1)
        full_title = full_title.replace(f"{file}::", "")
    return f"{file} | {full_title}"


def _mocha_status_from_stdout(combined: str) -> dict[str, str]:
    """Robustly recover name->status from a mocha ``--reporter=json`` dump.

    This is the fallback for the truncation bug in the vendored official
    parser.py (see :func:`_iter_brace_matched_objects`).  It is a no-op when
    the stdout carries no mocha reporter JSON (e.g. Go / jest instances), so it
    is safe to call unconditionally.
    """
    out: dict[str, str] = {}
    if not combined or '"stats"' not in combined:
        return out
    key_status = {"passes": "PASSED", "pending": "SKIPPED", "failures": "FAILED"}
    for blob in _iter_brace_matched_objects(combined, '"stats"'):
        try:
            obj = json.loads(blob)
        except Exception:
            continue
        if not isinstance(obj, dict) or "stats" not in obj:
            continue
        for key, status in key_status.items():
            for test in obj.get(key) or []:
                if not isinstance(test, dict):
                    continue
                name = _mocha_name(test)
                if not name.strip(" |"):
                    continue
                # failures win over passes if a name appears twice (retries).
                if name in out and out[name] == "FAILED":
                    continue
                out[name] = status
    return out


# jest --verbose output symbols (same symbol set as the official parser.py + failure symbols ✕/✗).
_JEST_PASS_MARKS = ("✓", "✔")           # ✓ ✔
_JEST_FAIL_MARKS = ("✕", "✗", "✘")  # ✕ ✗ ✘
_JEST_SKIP_MARKS = ("○", "✎")            # ○ ✎
_JEST_FILE_PAT = re.compile(r"^(?:PASS|FAIL)\s+(\S+)")


def _jest_status_from_stdout(combined: str) -> dict[str, str]:
    """Rebuild name->status from jest ``--verbose`` human-readable output.

    The official parser.py only parses ``PASS <file>`` file blocks and ignores
    ``FAIL <file>`` blocks, so whenever an F2P fails in the same test file (->
    jest marks that file FAIL), the passing P2P cases inside the block are all
    swallowed, and the pre-check stage falsely reports P2P_PRE_FAIL. Here we do
    not distinguish PASS/FAIL blocks and collect every ``✓/✕/○`` case,
    constructing ``file | describe | case`` names consistent with the dataset.

    No-op for Go / mocha-json / pytest output (no jest file headers or symbol
    lines).
    """
    out: dict[str, str] = {}
    if not combined:
        return out
    cur_file = ""
    cur_suite = ""
    for raw in combined.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        m = _JEST_FILE_PAT.match(stripped)
        if m:
            cur_file = m.group(1)
            cur_suite = ""
            continue
        if not cur_file:
            continue
        mark = stripped[0]
        if mark in _JEST_PASS_MARKS or mark in _JEST_FAIL_MARKS or mark in _JEST_SKIP_MARKS:
            # Case line: strip the leading symbol + trailing duration "(N ms)".
            case = stripped[1:].strip()
            case = re.sub(r"\s*\(\d+(?:\.\d+)?\s*m?s\)\s*$", "", case).strip()
            if not case:
                continue
            if mark in _JEST_PASS_MARKS:
                st = "PASSED"
            elif mark in _JEST_FAIL_MARKS:
                st = "FAILED"
            else:
                st = "SKIPPED"
            name = f"{cur_file} | {cur_suite} | {case}" if cur_suite else f"{cur_file} | {case}"
            # Failure wins (on same-name retries, FAILED is not overwritten by PASSED).
            if name in out and out[name] == "FAILED":
                continue
            out[name] = st
        elif line[:1] in (" ", "\t") and not stripped.startswith(("●", ">", "at ", "Received", "Expected")):
            # describe block title: indented, no symbol, and not a failure-detail block (starts with ●).
            indent = len(line) - len(line.lstrip())
            if indent <= 4:
                cur_suite = stripped
    return out


# ospec (the assertion-style test framework used by ts repos like tutanota) output semantics:
#   - on failure, prints each ``<describe> > <case>:`` + error stack line by line;
#   - on success, prints *only* the aggregate line ``All N assertions passed (old style total: M)``,
#     never printing any individual passing case name.
# The official parser.py is a hardcoded stub that only produces suite-level names (on matching
# ``All N passed`` it appends a hardcoded ``... | test suite`` list), so it never gets case-level
# nodes -> own_f2p case-level nodes are all NOT_FOUND. Here we recover case-level status per ospec's
# own semantics (parallel to the mocha/jest recovery helpers).
_OSPEC_PASS_AGG = re.compile(r"All \d+ assertions passed")
_OSPEC_FAIL_AGG = re.compile(r"\d+ out of \d+ assertions failed")


def _ospec_status_from_stdout(combined: str, nodes: list[str]) -> dict[str, str]:
    """Recover case-level name->status from raw ospec stdout by ``<describe> > <case>``.

    Criteria (honest and conservative):
      - if the node's failure block header ``<describe> > <case>:`` appears in combined -> FAILED;
      - otherwise, if this run has *zero failures* (``All N assertions passed`` appears and no
        ``K out of N assertions failed``) -> PASSED. test_patch has already merged the new cases into
        the full api/client suite (run_script.sh runs the whole suite), so ``All N passed`` is the
        authoritative guarantee of "zero failures in the suite including the new cases", and ospec
        does not print case names on success anyway.
      - in all other cases (there is a failing run but this node is not in a failure block, so we
        cannot confirm whether this node ran), draw no conclusion, return None, hand back to the
        NOT_FOUND logic, to avoid falsely marking a genuinely missing node PASSED.

    No-op for Go / mocha-json / jest / pytest output (no ospec assertions aggregate line).
    """
    out: dict[str, str] = {}
    if not combined or "assertions" not in combined:
        return out
    has_pass_agg = bool(_OSPEC_PASS_AGG.search(combined))
    has_fail_agg = bool(_OSPEC_FAIL_AGG.search(combined))
    if not has_pass_agg and not has_fail_agg:
        return out  # non-ospec output
    for node in nodes:
        parts = [p.strip() for p in str(node).split(" | ") if p.strip()]
        if len(parts) < 2:
            continue
        case = parts[-1]
        describe = parts[-2]
        # ospec failure block header: "<describe> > <case>:" (separator is " > ", not the dataset's " | ")
        fail_pat = re.compile(re.escape(describe) + r"\s*>\s*" + re.escape(case) + r"\s*:")
        if fail_pat.search(combined):
            out[node] = "FAILED"
        elif has_pass_agg and not has_fail_agg:
            out[node] = "PASSED"
    return out


def _node_aliases(node: str) -> list[str]:
    text = str(node or "").strip()
    if not text:
        return []
    aliases = [text]
    if " | " in text:
        parts = [part.strip() for part in text.split(" | ") if part.strip()]
        aliases.extend(parts)
        aliases.append(parts[0])
        aliases.append(" | ".join(parts[1:]))
    if "::" in text:
        aliases.append(text.rsplit("::", 1)[-1])
        aliases.append(text.split("::", 1)[0])
    return list(dict.fromkeys(alias for alias in aliases if alias))


def _status_from_aliases(node: str, parsed_status: dict[str, str]) -> str | None:
    aliases = _node_aliases(node)
    for alias in aliases:
        if alias in parsed_status:
            return parsed_status[alias]
    for parsed_name, status in parsed_status.items():
        parsed_aliases = set(_node_aliases(parsed_name))
        if any(alias in parsed_aliases for alias in aliases):
            return status
    return None


def _fallback_status_from_output(node: str, combined: str) -> str | None:
    """Infer a missing official parser result from raw stdout/stderr.

    Several Pro parser.py files only emit passed tests. In seed precheck this
    turns genuine F2P failures into NOT_FOUND unless we recover the status from
    the official raw output.
    """
    text = combined or ""
    if not text:
        return None
    aliases = _node_aliases(node)
    for alias in aliases:
        escaped = re.escape(alias)
        if re.search(rf"---\s+FAIL:\s+{escaped}\b", text):
            return "FAILED"
        if re.search(rf"---\s+PASS:\s+{escaped}\b", text):
            return "PASSED"
    if " | " in node:
        parts = [part.strip() for part in node.split(" | ") if part.strip()]
        test_file = parts[0] if parts else ""
        test_case = parts[-1] if len(parts) > 1 else ""
        if test_file and test_case and test_case in text:
            escaped_file = re.escape(test_file)
            if re.search(rf"(?m)^FAIL\s+.*{escaped_file}\b", text):
                return "FAILED"
            if re.search(rf"(?m)^PASS\s+.*{escaped_file}\b", text):
                return "PASSED"
    return None


def run_pro_official_tests(
    session: DockerSession,
    *,
    instance: dict,
    test_patch: Any,
    requested_nodes: list[str],
    base_commit: str,
    cwd: str,
    timeout_sec: int,
    apply_unified_diff,
    reset_eval_test_files,
) -> tuple[dict[str, str], dict[str, dict], str]:
    """Run SWE-bench Pro tests with the official per-instance run_script/parser.

    Official SWE-bench Pro does not infer test commands from repo_language.  It
    ships one run_script.sh and parser.py per instance.  For the original dataset
    test patch, official evaluation checks out target test files via the last
    line of before_repo_set_cmd after applying the model patch.  For synthetic
    test patches generated by our trace pipeline, we still apply the unified diff
    directly, then run the same official test harness.
    """
    instance_id = str(instance.get("instance_id") or "")
    if not instance_id:
        raise PatchApplyError("Pro instance is missing instance_id")
    if not official_scripts_available(instance_id):
        raise PatchApplyError(f"official SWE-bench Pro scripts not found for {instance_id}")

    nodes = [str(n) for n in requested_nodes if str(n)]
    test_patch_parts = _patch_parts(test_patch)
    official_test_patch = _same_patch(test_patch_parts, instance.get("test_patch"))
    selected = selected_tests_for_instance(instance, nodes, official_test_patch=official_test_patch)

    if official_test_patch:
        _apply_official_test_state(session, instance, cwd=cwd)
    else:
        for part in test_patch_parts:
            apply_unified_diff(session, part, cwd=cwd)

    try:
        _write_official_workspace(session, instance_id)
        arg = ",".join(selected)
        cmd = (
            f"cd {shlex.quote(cwd)} && "
            f"bash /workspace/run_script.sh {shlex.quote(arg)} "
            f"> /workspace/stdout.log 2> /workspace/stderr.log; "
            f"python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json"
        )
        try:
            p = session.exec(cmd, timeout=timeout_sec)
        except subprocess.TimeoutExpired as exc:
            output = ""
            if exc.output:
                output += exc.output.decode() if isinstance(exc.output, bytes) else str(exc.output)
            if exc.stderr:
                output += "\n" + (exc.stderr.decode() if isinstance(exc.stderr, bytes) else str(exc.stderr))
            status = {n: "TIMEOUT" for n in nodes}
            details = {
                n: {"status": "TIMEOUT", "returncode": None, "output_excerpt": output or "[timeout]"}
                for n in nodes
            }
            return status, details, output or "[timeout]"

        stdout = _read_workspace_file(session, "/workspace/stdout.log")
        stderr = _read_workspace_file(session, "/workspace/stderr.log")
        if not stdout and not stderr and p.returncode != 0:
            stderr = p.stderr or p.stdout or ""
        combined = stdout + "\n" + stderr
        parsed_status = _status_from_output_json(_read_output_json(session))
        # The vendored official parser.py truncates mocha --reporter=json output
        # at the first nested array (non-greedy `\[.*?\]`), yielding an empty
        # output.json so every node looks NOT_FOUND even though the suite ran.
        # Recover the full reporter JSON ourselves via brace-matching and fill in
        # any names the official parser missed. No-op for Go/jest stdout.
        mocha_status = _mocha_status_from_stdout(combined)
        if mocha_status:
            for name, st in mocha_status.items():
                parsed_status.setdefault(name, st)
        # jest --verbose: the official parser.py skips FAIL file blocks and swallows the
        # passing P2P inside them -> pre-check falsely reports P2P_PRE_FAIL. Add a fallback
        # that does not distinguish PASS/FAIL blocks. No-op for Go/mocha/pytest output.
        jest_status = _jest_status_from_stdout(combined)
        if jest_status:
            for name, st in jest_status.items():
                parsed_status.setdefault(name, st)
        # ospec (tutanota etc.): the official parser.py is a suite-level hardcoded stub, so
        # case-level nodes are all NOT_FOUND; on success ospec only prints the ``All N
        # assertions passed`` aggregate and no case names. Recover case-level own_f2p status
        # from raw stdout per ospec semantics. No-op for non-ospec.
        ospec_status = _ospec_status_from_stdout(combined, nodes)
        if ospec_status:
            for name, st in ospec_status.items():
                parsed_status.setdefault(name, st)
        status: dict[str, str] = {}
        details: dict[str, dict] = {}
        for node in nodes:
            node_status = _status_from_aliases(node, parsed_status)
            if node_status is None:
                node_status = _fallback_status_from_output(node, combined)
            if node_status is None:
                node_status = "NOT_FOUND"
            status[node] = node_status
            details[node] = {
                "status": node_status,
                "returncode": p.returncode,
                "output_excerpt": combined,
            }
        return status, details, combined
    finally:
        # Use the Pro-specific in-place reset instead of the passed-in shared
        # reset_eval_test_files: the shared one runs `git checkout <ref> -- <f>`
        # (O_CREAT|O_EXCL) which hits the fuse-overlayfs EEXIST race on
        # sweap-images under concurrent IO. _pro_reset_test_files overwrites in
        # place (O_TRUNC) and is immune to that race. The reset_eval_test_files
        # parameter is retained for interface compatibility but intentionally
        # unused here.
        _pro_reset_test_files(
            session, test_patch=test_patch_parts,
            base_commit=base_commit or "HEAD", cwd=cwd,
        )
