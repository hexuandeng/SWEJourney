from __future__ import annotations

import shlex
import subprocess
from typing import Any

from swebench_dockerlib import DockerSession, PatchApplyError


def is_verified_instance(
    instance: dict | None,
    *,
    image: str | None = None,
    runner_kind: str | None = None,
) -> bool:
    instance = instance or {}
    if str(runner_kind or "").strip():
        return str(runner_kind).strip() == "pytest"
    image = str(image or "").strip()
    return (
        not str(instance.get("dockerhub_tag") or "").strip()
        and "sweb.eval." in image
    )


def _patch_parts(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(part) for part in raw if str(part).strip()]
    if isinstance(raw, str) and raw.strip():
        return [raw]
    return []


def _test_spec(instance: dict):
    from swebench.harness.constants import MAP_REPO_TO_EXT
    from swebench.harness.test_spec.test_spec import TestSpec

    return TestSpec(
        instance_id=str(instance.get("instance_id") or ""),
        repo=str(instance.get("repo") or ""),
        version=str(instance.get("version") or ""),
        repo_script_list=[],
        eval_script_list=[],
        env_script_list=[],
        arch="x86_64",
        FAIL_TO_PASS=list(instance.get("FAIL_TO_PASS_parsed") or []),
        PASS_TO_PASS=list(instance.get("PASS_TO_PASS_parsed") or []),
        language=MAP_REPO_TO_EXT[str(instance.get("repo") or "")],
        docker_specs={},
        namespace=None,
    )


def run_verified_official_tests(
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
    """Run a Verified instance with SWE-bench's repo/version test command."""
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
    from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
    from swebench.harness.test_spec.python import get_test_directives

    repo = str(instance.get("repo") or "")
    version = str(instance.get("version") or "")
    try:
        specs = MAP_REPO_VERSION_TO_SPECS[repo][version]
        parser = MAP_REPO_TO_PARSER[repo]
    except KeyError as exc:
        raise PatchApplyError(
            f"SWE-bench Verified test spec missing for repo={repo!r} version={version!r}"
        ) from exc

    parts = _patch_parts(test_patch)
    reset_eval_test_files(
        session,
        test_patch=parts,
        base_commit=base_commit or "HEAD",
        cwd=cwd,
    )
    for part in parts:
        apply_unified_diff(session, part, cwd=cwd)

    nodes = [str(node) for node in requested_nodes if str(node)]
    try:
        directives = get_test_directives(
            {**instance, "repo": repo, "version": version, "test_patch": "\n".join(parts)}
        )
        test_cmds = specs["test_cmd"]
        if isinstance(test_cmds, str):
            test_cmds = [test_cmds]
        suffix = " ".join(shlex.quote(directive) for directive in directives)
        commands = [
            f"{command} {suffix}".strip()
            for command in test_cmds
        ]
        command = " ; ".join(commands)
        cmd = (
            "source /opt/miniconda3/bin/activate && "
            "conda activate testbed && "
            f"cd {shlex.quote(cwd)} && {command}"
        )
        try:
            proc = session.exec(cmd, timeout=timeout_sec)
            output = (proc.stdout or "") + "\n" + (proc.stderr or "")
            parsed = parser(output, _test_spec(instance))
        except subprocess.TimeoutExpired as exc:
            output = ""
            if exc.output:
                output += exc.output.decode() if isinstance(exc.output, bytes) else str(exc.output)
            if exc.stderr:
                output += "\n" + (
                    exc.stderr.decode() if isinstance(exc.stderr, bytes) else str(exc.stderr)
                )
            status = {node: "TIMEOUT" for node in nodes}
            details = {
                node: {
                    "status": "TIMEOUT",
                    "returncode": None,
                    "output_excerpt": output or "[timeout]",
                }
                for node in nodes
            }
            return status, details, output or "[timeout]"

        status = {node: parsed.get(node, "NOT_FOUND") for node in nodes}
        details = {
            node: {
                "status": status[node],
                "returncode": proc.returncode,
                "output_excerpt": output,
            }
            for node in nodes
        }
        return status, details, output
    finally:
        reset_eval_test_files(
            session,
            test_patch=parts,
            base_commit=base_commit or "HEAD",
            cwd=cwd,
        )
