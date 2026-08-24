#!/usr/bin/env python3
"""
usersim_main.py — standalone user-sim runner (evaluation side, all user-sim logic).

Two-process architecture (see USER_SIM_STANDALONE_PLAN.md):
  - Assistant: a mini-swe-agent HTTP server (minisweagent.run.assistant_server),
    holding one agent + one container, driven over HTTP.
  - User-sim (this process): generates clean user requests via LLMGateway(sglang),
    runs the three tools (run_subtask_test / mark_subtask / read_status) against
    the assistant's live container, maintains the clean stream, and dumps traj.json.

Main loop:
  GREETING -> driver.next_user_message(clean) -> client.message(text) -> visible
  reply -> clean stream -> ... until all subtasks resolved / confidence floor /
  max_user_turns.

Usage
-----
python usersim_main.py \
    --trace /.../instance_xxx.json \
    --sim-config /.../user_sim_sglang_28150_coop_v2.yaml \
    -c swebench.yaml -c runs/v4pro_hxsz_28150.yaml ... \
    --out /.../runs/user_sim_xxx --port 8799
  # spawns the assistant server itself; pass --assistant-url to use an external one.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import subprocess
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
# Relative-path wiring: depends on ../trace_synth (agent_trace/local_test_executor/swebench_dockerlib etc.)
# and ../mini_swe_agent (minisweagent.*), and puts this package on sys.path. swebench comes from the pip package.
import config as _cfg  # noqa: E402
_cfg.setup_paths()

import yaml  # noqa: E402

from usersim_scenario import (  # noqa: E402
    build_subtask_specs,
    build_user_scenario,
    _enabled_ok_attempts,
)
from usersim_driver import UserSimDriver  # noqa: E402
from assistant_client import AssistantClient  # noqa: E402
from live_subtask_test_tool import RunSubtaskTestTool, MarkSubtaskTool, ReadStatusTool  # noqa: E402
from swebench_dockerlib import DockerSession  # noqa: E402
from bigtask_eval import BigTaskEvaluator, run_final_eval  # noqa: E402

logger = logging.getLogger("usersim_main")

MINISWE_DIR = str(_cfg.MINISWE_DIR)   # vendored mini_swe_agent (../mini_swe_agent)
PYTHON = sys.executable               # use the current interpreter to spawn the assistant server subprocess

# ─── Assistant-model switcher ────────────────────────────────────────────────
# Switches ONLY the "assistant agent" model (the user-sim model is still decided by
# the --sim-config YAML and is unaffected by this).
# Usage (pick one):
#   --assistant-model <KEY>              switch by alias from the table below (switches model_name + api_base together)
#   --assistant-model <name>@<base_url>  freeform passthrough (e.g. openai/Foo@http://h:port/v1)
# Implementation: parsed into two `-c` deep-merge overrides sent to the assistant server —
#   model.model_name=<name>  /  model.model_kwargs.api_base=<url>
# This -c key=value path uses the same mechanism as resume's `-c agent.wall_time_limit_seconds=...`;
# deep merge only changes these two keys, keeping model_kwargs siblings like temperature / max_tokens / drop_params.
# litellm openai-compatible endpoint: model_name needs the "openai/" prefix, api_base ends with /v1.
ASSISTANT_MODELS = {
    # alias: (model_name, api_base) — loaded from config.yaml (config.load_endpoints).
    # model_name needs the "openai/" prefix for litellm; api_base should end with /v1.
    # Fill in your own endpoints in config.yaml, or pass --assistant-model
    # name@base_url freeform without editing anything.
    alias: (entry.get("model", ""), entry.get("api_base", ""))
    for alias, entry in (_cfg.load_endpoints().get("assistant_models") or {}).items()
    if isinstance(entry, dict)
}


def resolve_assistant_model(choice: str) -> list[str]:
    """Parse the --assistant-model value into the `-c` override list for the assistant server.

    Supports: a registry alias (a key of ASSISTANT_MODELS), or freeform `name@base_url`.
    Returns [] when unspecified (no override).
    """
    choice = (choice or "").strip()
    if not choice:
        return []
    if choice in ASSISTANT_MODELS:
        name, base = ASSISTANT_MODELS[choice]
    elif "@" in choice:
        name, base = choice.split("@", 1)
        name, base = name.strip(), base.strip()
        if not name or not base:
            raise SystemExit(f"--assistant-model freeform is of the form name@base_url, got: {choice!r}")
    else:
        keys = ", ".join(ASSISTANT_MODELS)
        raise SystemExit(f"--assistant-model unknown alias {choice!r}; choose [{keys}] or use name@base_url")
    return [f"model.model_name={name}", f"model.model_kwargs.api_base={base}"]

# All user-sim conversation logs live under evaluation/logs/<run-name>/.
EVAL_LOG_ROOT = _THIS_DIR / "logs"


class ConvLog:
    """Human-readable transcript of EVERYTHING both sides exchanged.

    Writes two files under evaluation/logs/<name>/:
      - conversation.log : turn-by-turn, the clean user request, the assistant's
        visible reply, and the user-sim's internal tool calls (name/args/result),
        plus the raw model output (with [[confidence:N]]).
      - exchanges.jsonl  : one JSON object per logged event (machine-readable).
    """

    def __init__(self, name: str):
        self.dir = EVAL_LOG_ROOT / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.txt = self.dir / "conversation.log"
        self.jsonl = self.dir / "exchanges.jsonl"
        # truncate
        self.txt.write_text("", encoding="utf-8")
        self.jsonl.write_text("", encoding="utf-8")

    def _w(self, s: str):
        with self.txt.open("a", encoding="utf-8") as f:
            f.write(s + "\n")

    def _j(self, obj: dict):
        with self.jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def header(self, meta: dict):
        self._w("=" * 100)
        self._w(f"USER-SIM CONVERSATION LOG  —  {meta.get('name')}")
        for k in ("instance_id", "persona", "lang", "n_subtasks", "assistant_url", "container",
                  "dimensions", "rubric"):
            if k in meta:
                v = meta[k]
                self._w(f"  {k}: {json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v}")
        self._w("=" * 100)
        self._j({"event": "header", **meta})

    def usersim_turn(self, turn: int, out: dict):
        """Log the user-sim side of a turn: what it was prompted with, the tools it
        called (+results), and the clean user message it produced."""
        self._w("")
        self._w("-" * 100)
        self._w(f"[TURN {turn}] === USER-SIM (model generates the next user message) ===")
        self._w(f"--- prompt sent to user-sim (transcript + [USER] instruction) ---")
        self._w(out.get("prompt_user", ""))
        for i, tc in enumerate(out.get("tool_calls") or []):
            self._w(f"--- user-sim TOOL CALL #{i+1}: {tc.get('name')} args={json.dumps(tc.get('arguments'), ensure_ascii=False)}")
            self._w(f"    -> result: {tc.get('result')}")
        self._w(f"--- user-sim RAW output (with confidence) ---")
        self._w(out.get("raw_text") or "(empty)")
        rub = out.get("rubric")
        if rub is not None:
            self._w(f"--- rubric verdict (attempts={out.get('rubric_attempts')}) ---")
            self._w(f"    pass={rub.get('overall_pass')} avg={rub.get('avg')} "
                    f"scores={json.dumps(rub.get('scores'), ensure_ascii=False)}")
            for it in (rub.get("issues") or []):
                self._w(f"    issue: {it}")
        self._w(f">>> USER → ASSISTANT (clean, conf={out.get('confidence')}):")
        self._w(out.get("text") or "")
        self._j({"event": "usersim_turn", "turn": turn, "prompt_user": out.get("prompt_user"),
                 "tool_calls": out.get("tool_calls"), "raw_text": out.get("raw_text"),
                 "clean_text": out.get("text"), "confidence": out.get("confidence"),
                 "rubric": out.get("rubric"), "rubric_attempts": out.get("rubric_attempts"),
                 "rubric_history": out.get("rubric_history")})

    def assistant_turn(self, turn: int, sent: str, resp: dict):
        self._w("")
        self._w(f"[TURN {turn}] === ASSISTANT (mini-swe-agent via HTTP) ===")
        self._w(f"--- request sent to assistant ---")
        self._w(sent)
        self._w(f"<<< ASSISTANT → USER (visible reply, exit={resp.get('exit_status')}, "
                f"submission_len={len(resp.get('submission') or '')}):")
        self._w(resp.get("reply") or "")
        self._j({"event": "assistant_turn", "turn": turn, "request": sent,
                 "reply": resp.get("reply"), "exit_status": resp.get("exit_status"),
                 "submission": resp.get("submission"), "n_messages": resp.get("n_messages")})

    def status(self, turn: int, status: dict):
        self._w(f"    [status@turn{turn}] {json.dumps(status, ensure_ascii=False)}")
        self._j({"event": "status", "turn": turn, "status": dict(status)})

    def footer(self, reason: str, status: dict, confidence: int):
        self._w("")
        self._w("=" * 100)
        self._w(f"STOP: {reason}  final_confidence={confidence}")
        self._w(f"final_status: {json.dumps(status, ensure_ascii=False)}")
        self._w("=" * 100)
        self._j({"event": "footer", "stop_reason": reason, "final_status": dict(status),
                 "final_confidence": confidence})


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _read_server_port(port_file: Path, proc, max_wait: float = 900.0, poll: float = 0.5) -> int:
    """Poll the port file the server writes, reading back the real port the kernel
    assigned at the instant of bind.

    On the server side --port 0 → the kernel assigns a port when ThreadingHTTPServer
    binds → written to port_file. This eliminates the TOCTOU race of the driver
    pre-allocating a port. If the server subprocess exits before writing the port
    (e.g. container creation failed), raise early instead of idly waiting until timeout.
    """
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        try:
            txt = port_file.read_text().strip()
            if txt:
                return int(txt)
        except (FileNotFoundError, ValueError):
            pass
        rc = proc.poll()
        if rc is not None:
            raise RuntimeError(f"assistant server exited before writing the port (rc={rc}); see stdout.log")
        time.sleep(poll)
    raise TimeoutError(f"assistant server did not write the port within {max_wait}s ({port_file})")


def _borrow_session(container_id: str, image: str, login_shell: bool) -> DockerSession:
    """A DockerSession that *borrows* the assistant server's container.

    Mirrors trace_runner_v2._make_executor_session: stateless `docker exec` by
    container name/id; _started=False so stop()/__del__ never `docker rm` the
    assistant's live container.
    """
    sess = DockerSession.__new__(DockerSession)
    sess.image = image
    sess.container_name = container_id
    sess.keep_on_fail = True
    sess.login_shell = login_shell
    sess._started = False
    sess._failed = False
    return sess


# ── resume helpers ──────────────────────────────────────────────────────────

def _read_container_id(out_dir: Path) -> str:
    """Parse the container ID the last assistant server started, for --resume attach.

    Prefer reading resume_container.json (the new container ID reproduced by
    rebuild_resume_containers.py): after a machine swap / docker rebuild the old
    container is gone, and the new container ID reproduced by replaying the first K
    turns of commands lands in this sidecar.
    Only changes behavior when that file exists; otherwise fully falls back to
    parsing the old cid from stdout.log (original logic unchanged).
    """
    import re
    sidecar = out_dir / "resume_container.json"
    if sidecar.is_file():
        try:
            cid = (json.loads(sidecar.read_text()).get("container_id") or "").strip()
            if cid:
                return cid
        except Exception:
            pass
    log = out_dir / "stdout.log"
    if not log.is_file():
        return ""
    cid = ""
    for m in re.finditer(r"container=([0-9a-f]{12,})", log.read_text(errors="replace")):
        cid = m.group(1)
    return cid


def _container_remaining_seconds(container_id: str) -> int:
    """How much longer the container can live (seconds) = StartedAt + sleep duration − now.

    The container is started via `docker run -d ... sleep <container_timeout>` and
    exits automatically when it's up. Compute the remaining lifetime from inspect's
    StartedAt and the sleep value in the launch args, without relying on any yaml.
    Returns 0 on parse failure (the caller treats it as "not enough time": don't
    enter the loop, wrap up directly).
    """
    import re
    from datetime import datetime, timezone
    try:
        out = subprocess.run(
            ["docker", "inspect", "-f",
             "{{.State.StartedAt}}|{{range .Args}}{{.}} {{end}}", container_id],
            capture_output=True, text=True, timeout=60, check=True,
        ).stdout.strip()
        started_raw, args_raw = out.split("|", 1)
        # sleep duration: Args look like "sleep 100h"; supports 100h / 90m / 3600 / 1d etc.
        m = re.search(r"(\d+)\s*([smhd]?)\b", args_raw)
        if not m:
            return 0
        val = int(m.group(1)); unit = m.group(2) or "s"
        secs = val * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
        # StartedAt: RFC3339; the nanosecond digits must be truncated to microseconds for fromisoformat to parse
        s = started_raw.replace("Z", "+00:00")
        s = re.sub(r"\.(\d{6})\d+", r".\1", s)
        started = datetime.fromisoformat(s)
        now = datetime.now(timezone.utc)
        return max(0, int(secs - (now - started).total_seconds()))
    except Exception:
        return 0


def _count_assistant_turns(assistant_traj: Path) -> int:
    """Number of turns the assistant traj has completed = count of entries with role==exit
    and exit_status==ReplyToUser.

    On each user turn the agent ends with one reply_to_user (producing one such exit
    entry). When a kill lands mid-way through a bash step, the last turn has no such
    exit, so this count is exactly "the number of turns that actually completed and
    replied to the user".
    Returns 0 when unreadable (after the caller takes min, K=0 → treated as no complete
    turn; the logic to run this cell from scratch is handled by the upper layer).
    """
    try:
        d = json.loads(assistant_traj.read_text())
    except Exception:
        return 0
    return sum(1 for m in d.get("messages", [])
               if m.get("role") == "exit"
               and (m.get("extra") or {}).get("exit_status") == "ReplyToUser")


def _truncate_main_to_k(clean: list, sidecar: list, k: int) -> tuple[list, list]:
    """Trim the main traj's clean stream to "exactly the first k complete turns (user→assistant pairs)".

    clean = [system, u0, a0, u1, a1, …]. Keep system + the first k (user, assistant)
    pairs, discarding everything after the k-th pair (including a dangling lone user /
    the extra turn the main traj has beyond the assistant).
    sidecar is trimmed to k entries in sync. Returns (clean, sidecar), guaranteed to
    be strictly alternating and paired.
    """
    out = []
    seen_assistant = 0
    for m in clean:
        role = m.get("role")
        if role == "system":
            out.append(m)
            continue
        if seen_assistant >= k:
            break  # already collected k complete turns; drop everything after
        out.append(m)
        if role == "assistant":
            seen_assistant += 1
    # If it happens to stop on a user at the end (won't happen since we break before the k-th turn's
    # assistant is added; defensive)
    while out and out[-1].get("role") == "user":
        out.pop()
    return out, list(sidecar[:k])



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--sim-config", required=True, help="user_sim YAML (contains the user_sim: section)")
    ap.add_argument("-c", "--config", action="append", default=[], help="assistant config spec (may be repeated)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--assistant-url", default="", help="URL of an externally started assistant server; empty = spawn one in this process")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0, help="assistant server port; 0 = random")
    ap.add_argument("--model", default="", help="override assistant model_name (name only, does not change the endpoint)")
    ap.add_argument("--assistant-model", default="",
                    help="switch the assistant agent model (switches model_name+api_base together). "
                         "Takes a registry alias [%s], or freeform 'name@base_url'." % ", ".join(ASSISTANT_MODELS))
    ap.add_argument("--log-name", default="", help="the evaluation/logs/<name>/ conversation-log directory name; empty = use the out directory name")
    ap.add_argument("--resume", action="store_true",
                    help="resume: restore the clean conversation / tri-state / confidence / turn count from an existing traj in the --out directory, "
                         "attach the assistant's live container (no git reset), and continue the closed loop from the next turn (not from scratch).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    trace = json.loads(Path(args.trace).read_text())
    instance = trace.get("instance", {})
    instance_id = instance.get("instance_id", Path(args.trace).stem)
    cfg = trace.get("config", {})
    cwd = cfg.get("cwd", "/app")
    pytest_bin = cfg.get("pytest_bin", "/opt/miniconda3/envs/testbed/bin/pytest")
    login_shell = cfg.get("test_runner_kind") != "pro_official"
    base_commit = instance.get("base_commit", "")
    image = cfg.get("image", "")

    sim_cfg = dict((yaml.safe_load(Path(args.sim_config).read_text()) or {}).get("user_sim", {}))

    # ── scenario + tri-state ───────────────────────────────────────────────
    specs = build_subtask_specs(trace)
    if not specs:
        logger.error("No enabled&ok ATTEMPT found; giving up.")
        sys.exit(2)
    status = {s["id"]: "pending" for s in specs}
    subtask_specs = {s["id"]: s for s in specs}
    persona = sim_cfg.get("persona") or ""
    scenario = build_user_scenario(specs, persona)

    # ── resume: restore clean conversation / tri-state / confidence / turn count from the dumped traj ──
    # On resume, read back the last dumped state; attach the assistant's live container;
    # continue from the next complete turn. All restore variables are computed here first,
    # then used per the resume branch throughout below.
    resume_clean: list = []
    resume_sidecar: list = []
    resume_status: dict | None = None
    resume_confidence: int | None = None
    resume_attach_cid: str = ""
    resume_wall_limit: int | None = None
    resume_keep_turns: int = 0   # K=min(main traj complete turns, assistant completed turns), sent to the assistant server
    traj_path_early = out_dir / f"{instance_id}.traj.json"
    asst_traj_early = out_dir / "assistant" / f"{instance_id}.assistant.traj.json"
    if args.resume:
        if not traj_path_early.is_file():
            logger.error("[resume] missing %s, cannot resume — please run from scratch instead.", traj_path_early)
            sys.exit(5)
        prev = json.loads(traj_path_early.read_text())
        prev_msgs = prev.get("messages") or []
        prev_sidecar = prev.get("usersim_tool_sidecar") or []
        # Align the two streams "to the shorter": usersim_main is the sole arbiter, computing K=min(N,M).
        #   N = main traj completed turns = number of assistant reply entries in clean
        #   M = assistant traj completed turns = number of ReplyToUser exit entries (a kill mid-bash loses one turn)
        # The main traj is trimmed to K here; the same K is sent to the assistant server to trim its own traj.
        n_main = sum(1 for m in prev_msgs if m.get("role") == "assistant")
        m_asst = _count_assistant_turns(asst_traj_early)
        resume_keep_turns = min(n_main, m_asst)
        if resume_keep_turns <= 0:
            logger.error("[resume] no complete aligned turn (main=%d, assistant=%d) — no resumable state, "
                         "please run from scratch instead.", n_main, m_asst)
            sys.exit(5)
        resume_clean, resume_sidecar = _truncate_main_to_k(prev_msgs, prev_sidecar, resume_keep_turns)
        prev_info = prev.get("info") or {}
        resume_status = dict(prev_info.get("final_status") or {})
        resume_confidence = prev_info.get("final_confidence")
        resume_attach_cid = _read_container_id(out_dir)
        if not resume_attach_cid:
            logger.error("[resume] no container ID parsed from stdout.log, cannot attach — please run from scratch instead.")
            sys.exit(5)
        # No longer time-limited: the wall clock is globally disabled (wall_time_limit_seconds=0),
        # and resume no longer overrides the container's remaining lifetime onto the agent's wall clock.
        # Container survival is backstopped by docker container_timeout(240h), and the closed loop
        # converges via max_user_turns=3×number of subtasks.
        resume_wall_limit = None
        # Override the initial all-pending with the restored tri-state (driver / three tools share the same status reference)
        if resume_status:
            for k in status:
                if k in resume_status:
                    status[k] = resume_status[k]
        logger.info("[resume] aligned K=min(main%d, asst%d)=%d; container=%s (no time limit); "
                    "conf=%s, status=%s",
                    n_main, m_asst, resume_keep_turns, resume_attach_cid,
                    resume_confidence,
                    json.dumps(status, ensure_ascii=False))

    # ── assistant server: spawn in this process or external ────────────────
    server_proc = None
    if args.assistant_url:
        base_url = args.assistant_url.rstrip("/")
    else:
        # No longer let the driver pre-allocate the port (the old _free_port() has a TOCTOU race:
        # between close and the server actually binding the port is unowned, and under high concurrency
        # two cells grab the same port → the latter crashes on bind, the former gets wrongly connected
        # → data cross-talk pollution). Switch to --port 0, letting the server's kernel assign it at
        # the instant of bind, then read back the real port from the port file the server writes to build base_url.
        server_out_dir = out_dir / "assistant"
        port_file = server_out_dir / "port"
        try:
            port_file.unlink()  # clear last time's leftover, to avoid reading a stale port
        except FileNotFoundError:
            pass
        except Exception:  # noqa: BLE001
            pass
        # user-sim assistant server (adds the reply_to_user tool). Local copy
        # under evaluation/agent/, NOT minisweagent.run.assistant_server, so the
        # mini-swe-agent package is left untouched. cwd stays MINISWE_DIR below
        # so relative -c configs (swebench.yaml / runs/*.yaml) still resolve.
        usersim_server = str(Path(__file__).resolve().parent / "agent" / "usersim_assistant_server.py")
        cmd = [PYTHON, usersim_server, args.trace,
               "-o", str(server_out_dir), "--host", args.host, "--port", "0"]
        for c in args.config:
            cmd += ["-c", c]
        # Resume: attach the live container (skip creation + skip git reset startup), inject the last
        # assistant history; and override the agent's wall clock to "the container's remaining lifetime",
        # so it inherits rather than resetting to a fresh 96h.
        if args.resume:
            cmd += ["--attach-container", resume_attach_cid,
                    "--resume-traj", str(out_dir / "assistant" / f"{instance_id}.assistant.traj.json"),
                    "--resume-keep-turns", str(resume_keep_turns)]
            if resume_wall_limit is not None:
                cmd += ["-c", f"agent.wall_time_limit_seconds={resume_wall_limit}"]
        if args.model:
            cmd += ["-m", args.model]
        # Model switcher: parse --assistant-model into two `-c` deep-merge overrides (model_name+api_base).
        # Placed after the other -c so it is sent last → highest priority, overriding the built-in endpoint/model in v4flash_hxsz.yaml.
        for ov in resolve_assistant_model(args.assistant_model):
            cmd += ["-c", ov]
        logger.info("[usersim] spawning assistant server: %s", " ".join(cmd))
        server_proc = subprocess.Popen(cmd, cwd=MINISWE_DIR)
        # Poll the port file to read back the real port the server's kernel assigned (the server may be slow to build the container; allow ample time).
        real_port = _read_server_port(port_file, server_proc, max_wait=900.0)
        base_url = f"http://{args.host}:{real_port}"

    client = AssistantClient(base_url)
    health = client.wait_ready(max_wait=900.0, expect_instance=instance_id)
    container_id = health["container"]
    logger.info("[usersim] assistant ready: container=%s url=%s", container_id, base_url)

    # ── three tools: point at the assistant's live container (borrow a DockerSession) ──
    sess = _borrow_session(container_id, image, login_shell)
    sim_instance = dict(instance)
    sim_instance["_config_image"] = image
    sim_instance["_test_runner_kind"] = cfg.get("test_runner_kind")

    run_test_tool = RunSubtaskTestTool(
        session=sess, subtask_specs=subtask_specs, status=status, instance=sim_instance,
        base_commit=base_commit, cwd=cwd, pytest_bin=pytest_bin,
        timeout_sec=int(sim_cfg.get("timeout_sec", 1800)),
    )
    mark_tool = MarkSubtaskTool(subtask_specs=subtask_specs, status=status)
    read_tool = ReadStatusTool(subtask_specs=subtask_specs, status=status)

    # ── private per-big-task accounting (not in the conversation stream; results only land in a host file, never shown to usersim/agent) ──
    # Trigger: when all subtasks of a big task reach a terminal state (done/failed, no pending), run the hidden
    # f2p/p2p once for each subtask of that big task (reusing the run_test_tool above: same snapshot/restore +
    # executor core, testing the agent's current live container, REPORT-only, does not change status, does not pollute the workspace).
    # Accounting only, used to compute "the correctness rate of the agent's code at the end of this interaction".
    bigtask_eval = BigTaskEvaluator(
        specs=specs, status=status, run_test_tool=run_test_tool,
        out_path=out_dir / "bigtask_eval.json", instance_id=instance_id,
    )

    driver = UserSimDriver(
        scenario=scenario, run_test_tool=run_test_tool, mark_tool=mark_tool, read_tool=read_tool,
        status=status, sim_cfg=sim_cfg, lang=str(sim_cfg.get("lang", "en")),
        prompt_path=sim_cfg.get("prompt_path"), max_tool_steps=int(sim_cfg.get("max_tool_steps", 16)),
    )
    # User-turn cap = 3 × number of subtasks (no longer time-limited; the closed loop converges by this).
    # max_user_turns in sim_cfg is deprecated: the cap is decided dynamically by this instance's subtask scale,
    # so a big task is allowed more interactions.
    max_user_turns = int(3.0 * len(specs))
    big_task_id = specs[0]["big_task_id"]
    logger.info("[usersim] driver ready: model=%s lang=%s subtasks=%d",
                getattr(driver.llm._gateway, "model", "?"), driver.lang, len(specs))
    # Resume: restore confidence (the driver initializes to confidence_init; override it here with the last value)
    if args.resume and resume_confidence is not None:
        driver.confidence = driver._clamp_confidence(int(resume_confidence))

    # Detailed conversation log (all information exchanged by both sides) → evaluation/logs/<name>/
    log_name = args.log_name or out_dir.name
    convlog = ConvLog(log_name)
    convlog.header({"name": log_name, "instance_id": instance_id, "persona": persona,
                    "lang": driver.lang, "n_subtasks": len(specs),
                    "assistant_url": base_url, "container": container_id,
                    "dimensions": sim_cfg.get("dimensions") or {},
                    "rubric": {"enabled": driver.rubric_enabled,
                               "max_attempts": driver.rubric_max_attempts,
                               "min_avg": driver.rubric_min_avg}})
    logger.info("[usersim] conversation log → %s", convlog.txt)

    # ── clean stream + main loop ─────────────────────────────────────────────
    if args.resume:
        # Resume: continue from the history trimmed to a complete-turn boundary (resume_clean[0] is last time's system).
        clean = list(resume_clean)
        tool_sidecar = list(resume_sidecar)
    else:
        clean: list[dict] = [{"role": "system", "content": driver.system_prompt}]
        tool_sidecar: list[dict] = []
    stop_reason = "max_user_turns"
    traj_path = out_dir / f"{instance_id}.traj.json"

    # The assistant agent's "fatal" exit statuses: once hit, the agent terminates this turn
    # without reply_to_user (hit the wall clock), and any subsequent continue_with will only
    # re-read the last old assistant text (idle spinning). Once detected, end the closed loop
    # immediately, and never keep sending messages to an already-dead agent.
    #
    # Note: LimitsExceeded (step_limit / cost_limit) is NO LONGER treated as fatal. cost_limit
    # is set to 0 (unlimited), so the only trigger for LimitsExceeded is step_limit=250 — produced
    # when the agent runs a full 250 steps in a turn without replying. In that case the agent
    # process is still alive and the container state is intact, so the closed loop should continue
    # to the next user turn (converged by max_user_turns), rather than break globally the moment
    # the step cap is hit. The wall clock is disabled (wall_time_limit_seconds=0), so TimeExceeded
    # won't actually fire either, but keep it as the sole fatal backstop (in case the wall clock is
    # explicitly enabled somewhere).
    _FATAL_EXIT_STATUSES = {"TimeExceeded"}

    def dump(reason: str):
        # All generation inputs for the user-sim side, so the renderer can show
        # every system/user prompt that fed the user-sim model (the assistant-side
        # system/instance prompts live in the sibling assistant traj).
        usersim_prompts = {
            "system": driver.system_prompt,
            "user_instruction_template": getattr(driver._template, "user_instruction", "") or "",
            "greeting": driver.greeting,
        }
        if getattr(driver, "_verify_tpl", None):
            usersim_prompts["verify_system"] = driver._verify_tpl[0]
            usersim_prompts["verify_user_template"] = driver._verify_tpl[1]
        data = {
            "info": {
                "instance_id": instance_id, "user_sim": True, "user_sim_standalone": True,
                "user_sim_model": sim_cfg.get("model_name"), "persona": persona,
                "dimensions": sim_cfg.get("dimensions") or {},
                "lang": driver.lang, "n_subtasks": len(specs), "final_status": status,
                "stop_reason": reason, "final_confidence": driver.confidence,
                "usersim_prompts": usersim_prompts,
            },
            "messages": clean, "usersim_tool_sidecar": tool_sidecar,
            "trajectory_format": "mini-swe-agent-1.1",
        }
        traj_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    try:
        _TIMEOUT_FORCED_MSG = {
            "en": ("You have timed out. Please immediately report your current progress "
                   "and what you have completed so far. Do not continue processing."),
        }

        def assistant_message(text: str, *, first: bool):
            """Call the assistant. On timeout, lower confidence by 10 and re-send a
            forced progress-report message, repeating as long as it keeps timing out.

            Once confidence has reached the floor, stop retrying and return a synthetic
            timeout reply (never raises). The main loop then exits normally via
            confidence_exhausted() on the next turn, so the finally-block tests run.
            """
            try:
                return client.message(text, first=first, big_task_id=big_task_id)
            except (TimeoutError, OSError) as e:
                logger.warning("[usersim] assistant call timed out/interrupted (%r)", e)
                forced = _TIMEOUT_FORCED_MSG.get(driver.lang, _TIMEOUT_FORCED_MSG["en"])
                while True:
                    driver.confidence = max(driver.confidence_floor, driver.confidence - 10)
                    logger.info("[usersim] confidence lowered to %d (floor=%d)",
                                driver.confidence, driver.confidence_floor)
                    if driver.confidence <= driver.confidence_floor:
                        logger.warning("[usersim] confidence has bottomed out — stop retrying, inject a timeout reply, gracefully end the closed loop")
                        msg = ("[System notice] Sorry, this request took too long on my end and has timed out "
                               "(still no result after 24 hours). This conversation ends here.")
                        return {"reply": msg, "exit_status": "assistant_timeout"}
                    logger.info("[usersim] sending forced report message, asking the assistant to report progress immediately")
                    try:
                        return client.message(forced, first=False, big_task_id=big_task_id)
                    except (TimeoutError, OSError) as e2:
                        logger.warning("[usersim] forced report message timed out again (%r) — keep lowering confidence and retrying", e2)
                        continue

        _assistant_dead = False
        if args.resume:
            # Resume: skip the opening (turn 0 was already produced and dumped last time). clean/sidecar/tri-state/
            # confidence have been restored, so continue directly from the next complete turn. When the container's
            # remaining time is insufficient, the loop below either does not enter (start_turn out of range) or
            # confidence_exhausted on the first turn also gracefully wraps up via finally.
            start_turn = resume_keep_turns  # = len(tool_sidecar); K complete turns done, continue from turn K
            logger.info("[resume] skipping the opening, continuing from turn=%d (%d turns already done)",
                        start_turn, start_turn)
            if driver.all_resolved():
                stop_reason = "all_subtasks_resolved"
        else:
            # Opening: generate the first user message → first=true
            first = driver.next_user_message(clean)
            tool_sidecar.append({"turn": 0, "confidence": first["confidence"],
                                 "tool_calls": first["tool_calls"],
                                 "prompt_user": first.get("prompt_user", ""),
                                 "full_messages": first.get("full_messages", [])})
            convlog.usersim_turn(0, first)
            opening = (first["text"] or "").strip()
            clean.append({"role": "user", "content": opening})
            logger.info("[usersim turn 0] user opening: %s (conf=%d)", opening[:160], first["confidence"])
            # Private accounting: after the opening, if any big task has already reached a terminal state (very rare) run it once (result not in the conversation stream).
            bigtask_eval.check_and_evaluate(turn=0)
            dump(stop_reason)

            resp = assistant_message(opening, first=True)
            clean.append({"role": "assistant", "content": resp.get("reply", "")})
            convlog.assistant_turn(0, opening, resp)
            convlog.status(0, status)
            logger.info("[usersim turn 0] assistant exit=%s", resp.get("exit_status"))
            dump(stop_reason)
            # Closed-loop guard: skip the entire conversation loop when the assistant agent exits fatally
            # (see the same criterion at the top of the loop below; if hit at turn 0, _assistant_dead=True and the for-loop body is simply not entered).
            _assistant_dead = resp.get("exit_status") in _FATAL_EXIT_STATUSES
            if _assistant_dead:
                stop_reason = f"assistant_{resp.get('exit_status')}"
                logger.warning("[usersim] assistant fatal exit (%s) at turn 0 — the agent has terminated, "
                               "end the closed loop immediately, no more idle spinning", resp.get("exit_status"))
            start_turn = 1

        for turn in range(start_turn, max_user_turns + 1):
            if _assistant_dead:
                break
            if driver.all_resolved():
                stop_reason = "all_subtasks_resolved"; break

            out = driver.next_user_message(clean)
            tool_sidecar.append({"turn": turn, "confidence": out["confidence"],
                                 "tool_calls": out["tool_calls"],
                                 "prompt_user": out.get("prompt_user", ""),
                                 "full_messages": out.get("full_messages", [])})
            convlog.usersim_turn(turn, out)

            # Private accounting: this turn usersim may have used mark_subtask to push the last subtask of some big task
            # into a terminal state. Before the agent changes the container next turn, run f2p/p2p accounting once for the
            # just-finished big task, testing exactly the agent's code "at the moment that big task just finished". Results only land in a file, not the conversation stream.
            bigtask_eval.check_and_evaluate(turn=turn)

            if driver.confidence_exhausted():
                stop_reason = "confidence_zero"; break

            user_text = (out["text"] or "").strip()
            clean.append({"role": "user", "content": user_text})
            logger.info("[usersim turn %d] user: %s (conf=%d)", turn, user_text[:160], out["confidence"])
            dump(stop_reason)

            resp = assistant_message(user_text, first=False)
            clean.append({"role": "assistant", "content": resp.get("reply", "")})
            convlog.assistant_turn(turn, user_text, resp)
            convlog.status(turn, status)
            logger.info("[usersim turn %d] assistant exit=%s", turn, resp.get("exit_status"))
            dump(stop_reason)

            if resp.get("exit_status") in _FATAL_EXIT_STATUSES:
                stop_reason = f"assistant_{resp.get('exit_status')}"
                logger.warning("[usersim] assistant fatal exit (%s) at turn %d — the agent has terminated, "
                               "end the closed loop immediately, no more idle spinning", resp.get("exit_status"), turn)
                break

            if driver.all_resolved():
                stop_reason = "all_subtasks_resolved"; break
    finally:
        # Wrap-up accounting once more: capture the last big task that just reached a terminal state
        # but wasn't accounted yet when exiting via all_resolved / confidence_zero / max_user_turns (already-accounted ones are not re-run).
        try:
            bigtask_eval.check_and_evaluate(turn=-1)
        except Exception as exc:
            logger.warning("[bigtask-eval] wrap-up accounting exception (ignored): %r", exc)
        try:
            run_final_eval(specs, status, run_test_tool, out_dir / "final_eval.json", instance_id)
        except Exception as exc:
            logger.warning("[final-eval] final test exception (ignored): %r", exc)
        dump(stop_reason)
        convlog.footer(stop_reason, status, driver.confidence)
        n_done = sum(1 for v in status.values() if v == "done")
        n_failed = sum(1 for v in status.values() if v == "failed")
        logger.info("[usersim] done stop=%s done=%d failed=%d/%d → %s",
                    stop_reason, n_done, n_failed, len(specs), traj_path)
        try:
            client.shutdown()
        except Exception:
            pass
        if server_proc is not None:
            try:
                server_proc.wait(timeout=30)
            except Exception:
                server_proc.kill()


if __name__ == "__main__":
    main()
