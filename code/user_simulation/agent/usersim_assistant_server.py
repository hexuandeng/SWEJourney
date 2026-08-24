#!/usr/bin/env python3
"""usersim_assistant_server.py — user-sim variant of mini-swe-agent's
assistant_server.

This is a LOCAL COPY of minisweagent.run.assistant_server (NOT an edit of it),
living under user_simulation/agent/ so the user-sim branch can use a
different agent class + reply handling without touching mini-swe-agent.

Differences vs the upstream assistant_server:
  - Uses ``UserSimAgent`` instead of ``TraceRunnerAgentV2`` (adds the
    reply_to_user tool that ends a turn).
  - ``visible_agent_text`` returns the assistant's reply_to_user message when
    the turn ended via that tool (exit_status == "ReplyToUser"), instead of the
    last assistant chat content.

Everything else (HTTP API, container ownership, lifecycle) is identical.

Usage
-----
python <this file> /path/to/trace.json \
    -c <assistant config yaml> ... --host 127.0.0.1 --port 8799
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import typer

# Make the user_simulation/ dir importable so the `agent` package + config resolve
# both when launched as a script (python .../usersim_assistant_server.py) and as a
# module. Then config.setup_paths() puts ../mini_swe_agent (minisweagent.*) and
# ../trace_synth (agent_trace/local_test_executor/...) on sys.path — independent of
# cwd or any pip install, so a clean open-source checkout resolves correctly.
_EVALUATION_DIR = str(Path(__file__).resolve().parent.parent)
if _EVALUATION_DIR not in sys.path:
    sys.path.insert(0, _EVALUATION_DIR)

import config as _cfg  # noqa: E402
_cfg.setup_paths()

from minisweagent.config import get_config_from_spec
from minisweagent.models import get_model
from minisweagent.run.benchmarks.swebench import get_sb_environment
from minisweagent.run.trace_runner import extract_tasks
from minisweagent.utils.log import add_file_handler
from minisweagent.utils.serialize import recursive_merge

from agent.usersim_agent import UserSimAgent

logger = logging.getLogger(__name__)


def visible_agent_text(agent, result: dict) -> str:
    """The user-visible slice of the agent's latest turn.

    User-sim branch: if the turn ended via reply_to_user, that message IS the
    visible reply. Otherwise fall back to the last non-empty assistant content
    (e.g. a limit/time exit, which has no reply_to_user message).
    """
    res = result or {}
    if res.get("exit_status") == "ReplyToUser":
        reply = (res.get("reply") or "").strip()
        if reply:
            return reply

    last_assistant = ""
    for m in reversed(agent.messages):
        if m.get("role") == "assistant" and (m.get("content") or "").strip():
            last_assistant = m["content"].strip()
            break
    if last_assistant:
        return last_assistant
    return "(I've handled this round — let me know what else you need.)"


class AssistantSession:
    """Owns the single agent + container for the server's lifetime."""

    def __init__(self, trace_path: Path, config_spec: list[str], model_name: str | None,
                 output_dir: Path, *, attach_container_id: str = "",
                 resume_traj: Path | None = None, resume_keep_turns: int = 0):
        output_dir.mkdir(parents=True, exist_ok=True)
        add_file_handler(output_dir / "assistant_server.log")

        self.trace = json.loads(Path(trace_path).read_text())
        self.instance = self.trace.get("instance", {})
        self.instance_id = self.instance.get("instance_id", Path(trace_path).stem)

        tasks = extract_tasks(self.trace)
        if not tasks:
            raise RuntimeError(f"No enabled ATTEMPT tasks in {trace_path}")
        self.tasks = tasks

        configs = [get_config_from_spec(spec) for spec in config_spec]
        if model_name:
            configs.append({"model": {"model_name": model_name}})
        # resume: attach to a still-alive container (inheriting its filesystem state +
        # remaining wall-clock), and have get_sb_environment skip env_startup_command (no git reset --hard).
        if attach_container_id:
            configs.append({"environment": {"attach_container_id": attach_container_id}})
        self.config = recursive_merge(*configs)

        self.model = get_model(config=self.config.get("model", {}))
        self.env = get_sb_environment(self.config, self.instance)

        self.traj_path = output_dir / f"{self.instance_id}.assistant.traj.json"
        agent_kwargs = dict(self.config.get("agent", {}))
        agent_kwargs["output_path"] = self.traj_path
        self.agent = UserSimAgent(self.model, self.env, **agent_kwargs)

        self._lock = threading.Lock()
        self._started = False  # whether a fresh session has begun (run() called)

        # resume: inject the previous agent.messages so that the first /message goes through
        # continue_with (resuming history) instead of run (clearing and starting over). Prefer
        # resume_traj, otherwise default to traj_path.
        # The two streams align on a "shorter wins" basis: usersim_main has already computed
        # K=resume_keep_turns and passes it down; here we truncate our own messages up to (and
        # including) the "K-th ReplyToUser exit" — discarding the trailing unpaired
        # tool_call/tool left over when a kill happened mid bash step (injecting them would be
        # rejected by the API), so the injected history ends on a clean exit boundary.
        if attach_container_id:
            src = resume_traj if resume_traj else self.traj_path
            try:
                prev = json.loads(Path(src).read_text())
                prev_msgs = prev.get("messages") or []
                kept = self._truncate_assistant_to_k(prev_msgs, resume_keep_turns)
                if kept:
                    self.agent.messages = kept
                    self._started = True
                    logger.info("[usersim-assistant-server] resume: injected %d/%d history messages"
                                " (truncated to the %d-th ReplyToUser exit; src=%s)",
                                len(kept), len(prev_msgs), resume_keep_turns, src)
                    # Continue (rather than overwrite) query_snapshots: after truncating the
                    # previous snapshots to the aligned K turns, inject them back into the model
                    # proxy's _query_snapshots and preset _current_turn_idx to K-1, so that the
                    # resumed run's first continue_with increments to K -> the new snapshots'
                    # turn_idx seamlessly follow the old ones (0..K-1, K, K+1, ...). This way,
                    # when the run finishes and writes back to traj, the snapshots are the complete
                    # "old+new" concatenation rather than only the segment after resuming.
                    self._resume_query_snapshots(prev, resume_keep_turns)
                else:
                    logger.warning("[usersim-assistant-server] resume: no messages after truncation"
                                   " (K=%d, %d original), treating as a fresh session", resume_keep_turns, len(prev_msgs))
            except Exception as exc:
                logger.warning("[usersim-assistant-server] resume: failed to read/truncate history (%s): %r, treating as a fresh session",
                               src, exc)

        logger.info("[usersim-assistant-server] ready: instance=%s container=%s",
                    self.instance_id, self.container_id)

    @staticmethod
    def _truncate_assistant_to_k(messages: list, k: int) -> list:
        """Truncate up to (and including) the "k-th ReplyToUser exit", discarding everything after it.

        Each user turn has the agent end with one reply_to_user call -> one message with
        role==exit / exit_status==ReplyToUser. Keep the first k such exits and all messages
        between them, and cut off all messages after the k-th exit (the unpaired tool_call/
        tool left over when a kill happened mid bash step). Returns an empty list when k<=0
        or there are fewer than k exits (the caller then treats it as a fresh session).

        Note: any exit messages left in the injected history are automatically filtered out by
        _CompressingModelProxy._compress (which never sends any role==exit to the API), so
        ending on an exit does not affect the next call.
        """
        if k <= 0:
            return []
        kept = []
        seen = 0
        for m in messages:
            kept.append(m)
            if m.get("role") == "exit" and (m.get("extra") or {}).get("exit_status") == "ReplyToUser":
                seen += 1
                if seen >= k:
                    return kept
        # fewer than k exits (should not happen, since usersim_main already took the min) -> treat as unalignable
        return []

    def _resume_query_snapshots(self, prev_traj: dict, k: int) -> None:
        """Continue query_snapshots: inject the previously persisted snapshots (truncated to the aligned K turns) and preset the turn counter.

        query_snapshots is an in-memory record inside the _CompressingModelProxy process, lost
        on restart. If a resume does not restore it, then when the run finishes and writes back
        to traj only the segment after resuming remains (turn_idx counting from 1 again), and the
        old snapshots for turns 0..K-1 are overwritten and lost.

        Here we inject the part of the previous traj's query_snapshots with "turn_idx <= K-1" back
        into agent.model._query_snapshots, and preset agent._current_turn_idx to K-1. The resumed
        run's first /message goes through continue_with -> _current_turn_idx increments to K, and
        the new snapshots' turn_idx continue from K, seamlessly concatenating with the old
        snapshots (0..K-1) into a complete sequence.

        Note: the snapshots corresponding to the truncated "incomplete final turn" (kill happened
        mid bash step) are discarded too, consistent with the messages truncation boundary (the
        K-th ReplyToUser exit). best-effort: any step failing only logs a warning and does not
        affect the resume itself (worst case, falls back to replay rendering).
        """
        try:
            prev_snaps = prev_traj.get("query_snapshots") or []
            # keep only the first K already-aligned turns (turn_idx 0..K-1)
            kept_snaps = [s for s in prev_snaps
                          if isinstance(s, dict) and isinstance(s.get("turn_idx"), int)
                          and s["turn_idx"] <= k - 1]
            proxy = self.agent.model  # _TruncatingProxy(_CompressingModelProxy)
            if not hasattr(proxy, "_query_snapshots"):
                logger.warning("[usersim-assistant-server] resume: model has no _query_snapshots, "
                               "skipping snapshot continuation (will fall back to replay rendering)")
                return
            proxy._query_snapshots = list(kept_snaps)
            # preset the turn counter: so the resumed run's first continue_with increments to K (continuing after K-1)
            self.agent._current_turn_idx = k - 1
            logger.info("[usersim-assistant-server] resume: continuing query_snapshots -- injected old snapshots "
                        "%d/%d (turn_idx<=%d), _current_turn_idx preset to %d, "
                        "new snapshots will continue from turn %d", len(kept_snaps), len(prev_snaps),
                        k - 1, k - 1, k)
        except Exception as exc:  # noqa: BLE001  best-effort, does not block the resume
            logger.warning("[usersim-assistant-server] resume: snapshot continuation failed %r, "
                           "resume continues (this cell will fall back to replay rendering)", exc)

    @property
    def container_id(self) -> str | None:
        return getattr(self.env, "container_id", None)

    @property
    def default_big_task_id(self) -> str:
        return self.tasks[0].get("big_task_id", "UNKNOWN")

    def message(self, text: str, first: bool, big_task_id: str | None) -> dict:
        bid = big_task_id or self.default_big_task_id
        with self._lock:
            self.agent.n_calls = 0
            if first or not self._started:
                result = self.agent.run(text, big_task_id=bid)
                self._started = True
            else:
                result = self.agent.continue_with(text, big_task_id=bid)
            reply = visible_agent_text(self.agent, result)
            return {
                "reply": reply,
                "exit_status": (result or {}).get("exit_status", ""),
                "submission": (result or {}).get("submission", ""),
                "n_messages": len(self.agent.messages),
            }

    def reset(self, big_task_id: str | None) -> dict:
        with self._lock:
            self._started = False
        return {"ok": True}

    def shutdown(self) -> None:
        try:
            close = getattr(self.env, "cleanup", None) or getattr(self.env, "stop", None)
            if callable(close):
                close()
        except Exception as exc:  # best-effort
            logger.warning("[usersim-assistant-server] container cleanup failed: %s", exc)


def make_handler(session: AssistantSession, server_ref: dict):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet default logging
            logger.debug("[usersim-assistant-server] " + fmt, *args)

        def _send(self, code: int, payload: dict):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                return {}
            raw = self.rfile.read(n)
            try:
                return json.loads(raw.decode("utf-8"))
            except Exception:
                return {}

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "container": session.container_id,
                                 "n_messages": len(session.agent.messages),
                                 "instance_id": session.instance_id})
            elif self.path == "/container":
                self._send(200, {"container": session.container_id,
                                 "image": session.config.get("agent", {}).get("image")
                                          or session.trace.get("config", {}).get("image"),
                                 "cwd": session.trace.get("config", {}).get("cwd", "/app")})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                if self.path == "/message":
                    body = self._read_json()
                    text = str(body.get("text") or "")
                    first = bool(body.get("first", False))
                    bid = body.get("big_task_id")
                    out = session.message(text, first=first, big_task_id=bid)
                    self._send(200, out)
                elif self.path == "/reset":
                    body = self._read_json()
                    self._send(200, session.reset(body.get("big_task_id")))
                elif self.path == "/shutdown":
                    self._send(200, {"ok": True})
                    server_ref["stop"] = True
                    threading.Thread(target=server_ref["server"].shutdown, daemon=True).start()
                else:
                    self._send(404, {"error": "not found"})
            except Exception as exc:
                logger.exception("[usersim-assistant-server] request error")
                self._send(500, {"error": repr(exc)})

    return Handler


app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
    trace_path: Path = typer.Argument(..., help="trace JSON path"),
    output: str = typer.Option("", "-o", "--output", help="output directory"),
    config_spec: list[str] = typer.Option([], "-c", "--config", help="config spec (can be given multiple times)"),
    model: str | None = typer.Option(None, "-m", "--model", help="model name (overrides config)"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8799, "--port"),
    attach_container: str = typer.Option("", "--attach-container",
        help="resume: attach to this still-alive container ID (skip creating a new container + skip env_startup_command)"),
    resume_traj: str = typer.Option("", "--resume-traj",
        help="resume: inject history messages from this assistant.traj.json (defaults to the same-named file in the output directory)"),
    resume_keep_turns: int = typer.Option(0, "--resume-keep-turns",
        help="resume: two-stream alignment K -- truncate up to (and including) the K-th ReplyToUser exit; usersim_main computes the min and passes it down"),
):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    out_dir = Path(output) if output else Path("runs") / "assistant_server"
    session = AssistantSession(
        trace_path, list(config_spec), model, out_dir,
        attach_container_id=attach_container,
        resume_traj=Path(resume_traj) if resume_traj else None,
        resume_keep_turns=resume_keep_turns,
    )

    server_ref: dict = {}
    httpd = ThreadingHTTPServer((host, port), make_handler(session, server_ref))
    server_ref["server"] = httpd
    # The real listening port: with --port 0 the kernel assigns it at the moment of bind,
    # read back from the socket (eliminates the TOCTOU race of the driver pre-allocating a
    # port -- two cells can no longer get the same port).
    real_port = httpd.server_address[1]
    # Write the real port to a port file in the output directory, for the driver to read back
    # and build base_url (does not change the stdout redirect; server logs still go into the
    # cell's stdout.log, no regression). Atomic write: write to a temp file first then rename.
    try:
        port_file = out_dir / "port"
        tmp = out_dir / "port.tmp"
        tmp.write_text(str(real_port))
        tmp.replace(port_file)
    except Exception as exc:  # noqa: BLE001  best-effort, does not block startup
        logger.warning("[usersim-assistant-server] failed to write port file: %r", exc)
    logger.info("[usersim-assistant-server] listening on http://%s:%d (container=%s)",
                host, real_port, session.container_id)
    print(f"[assistant-server] READY http://{host}:{real_port} container={session.container_id}", flush=True)
    try:
        httpd.serve_forever()
    finally:
        session.shutdown()
        logger.info("[usersim-assistant-server] stopped")


if __name__ == "__main__":
    app()
