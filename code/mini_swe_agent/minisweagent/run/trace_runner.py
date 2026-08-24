#!/usr/bin/env python3
"""Trace runner: replay an agent_trace.v2 file through mini-swe-agent.

Each enabled ATTEMPT event in the trace becomes one or more mini-swe-agent tasks.
All tasks share a single Docker environment (environment state is continuous).
History is accumulated across all turns (full-history / scheme B).

History compression rules applied at every API call
(to tool-result / role="tool" messages in self.messages):

  ① In-progress current turn    →  keep full (messages being built this turn)
  ② Last completed turn ("the previous step's")  →  keep full
  ③ Same big_task_id, older turns     →  keep, but truncate content to HALF_MAX chars
  ④ Different big_task (≥2 behind)    →  skip entirely (not sent to LLM)
  ⑤ Any message ≥3 big_tasks behind  →  skip entirely (all roles)
  ✗ role="exit" messages              →  drop entirely (API doesn't accept this role)

Usage
-----
python -m minisweagent.run.trace_runner \\
    /path/to/trace.json \\
    -o runs/trace_out \\
    -c swebench.yaml \\
    -c runs/v4flash_hxsz.yaml
"""

from __future__ import annotations

import copy
import json
import logging
import traceback
from pathlib import Path

import litellm
import typer

from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.models import get_model
from minisweagent.run.benchmarks.swebench import get_sb_environment
from minisweagent.run.multi_turn import MultiTurnAgent
from minisweagent.utils.log import add_file_handler
from minisweagent.utils.serialize import UNSET, recursive_merge

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "swebench.yaml"

# ── History compression budget (layer ②): high/low watermark hysteresis ──────
# To maximize Anthropic prefix-cache hits, we use "high/low watermark hysteresis" rather than a single threshold:
#   · trimming is triggered only when the total history chars > _HISTORY_HIGH_CHARS (high watermark);
#   · once triggered, delete from the oldest turn group in one pass until ≤ _HISTORY_LOW_CHARS (low watermark).
# This leaves a (HIGH-LOW) margin after deletion, so several subsequent turns of history stay below the high
# watermark with the prefix completely unchanged → hitting the prefix cache the whole time, until it grows past
# the high watermark again and one more deletion happens.
# The old scheme was a single threshold: once crossed, every turn had to delete the oldest span to compress
# back → the prefix changed every turn and the cache was invalidated turn by turn ("add one span, delete one
# span"), which was both slow and expensive.
_HISTORY_HIGH_CHARS: int = 300_000
_HISTORY_LOW_CHARS: int = 150_000
# In the degraded state (force=True, this cell hit context-400), requirement ② only protects the "latest N
# assistant steps after the last user"; earlier intermediate steps are opened up as deletable. In the normal
# state, all assistants after that user are still protected.
_FORCE_KEEP_LAST_ASSTS: int = 30
# When "discarding" a span of history: delete these messages outright (removing the assistant.tool_calls
# arguments along with them), and insert a **single** placeholder assistant message at the deleted position
# (content=_DROPPED_TOOL_STUB, no tool_calls), indicating a span of history was deleted here. A run of
# consecutive deletions gets only one placeholder, not one per message.
_DROPPED_TOOL_STUB: str = "[history-elided]"

# ── Reactive context-400 handling (assistant code-helper endpoint side) ──────
# Normally history compression still follows the high/low watermark above (>256000 trims to 160000), and the
# default max_tokens comes from the yaml (32768). Only when a request truly hits a "context-overflow 400" do we
# reactively degrade: force this turn's history down to the low watermark 160000 (force=True, bypassing the
# high-watermark check, reusing the same deletion logic) + temporarily lower max_tokens to _FORCED_MAX_TOKENS
# and retry once. Once this cell has hit a 400, every subsequent request from this cell keeps using this set of
# degraded parameters (proxy instance flag self._forced_max_tokens) until the process ends → does not affect the
# next cell/user. If it still overflows after the degraded retry, raise TimeoutError, handed off to
# usersim_main.py's assistant_message fallback (forced report) for a graceful wrap-up.
_FORCED_MAX_TOKENS: int = 16_384
# Recognizing an endpoint context-overflow 400: litellm classifies an endpoint 400 as BadRequestError, and only
# when its message contains any of the keywords below is it judged a "context overflow" (avoiding falsely
# swallowing length-unrelated 400s, such as tool schema errors).
_CTX_400_MARKERS: tuple[str, ...] = (
    "maximum context length",
    "exceeds the model",
    "context length",
    "context_length_exceeded",
    "too many tokens",
    "reduce the length",
)


def _is_context_length_400(exc: Exception) -> bool:
    """True only when exc is litellm's context-overflow 400 (BadRequestError + length keyword)."""
    if not isinstance(exc, litellm.exceptions.BadRequestError):
        return False
    msg = (str(getattr(exc, "message", "") or "") + " " + str(exc)).lower()
    return any(marker in msg for marker in _CTX_400_MARKERS)



# ── Snapshot slimming: drop redundant fields before archiving to disk ─────────
# query_snapshots record the EXACT model input for each call (for rendering).
# Every prior assistant message in that input carries its `extra.response`
# (the full litellm completion dump), and each snapshot re-stores the whole
# accumulated history → the `.assistant.traj.json` is dominated by verbatim
# duplicates. We trim two purely-redundant pieces from `extra.response`:
#   · choices[].message — a verbatim copy of the top-level message's
#     content/tool_calls; the only field anything reads off a choice is
#     `finish_reason` (analyze_3models / extract_case), which we KEEP.
#   · a set of vLLM null placeholders nothing in this project reads.
# KEPT (consumed downstream): response.usage / created / id / model / object,
# and choices[].finish_reason. Pruning runs on the deep-copied snapshot ONLY —
# never on the messages sent to the API nor on the live agent history.
_RESPONSE_NULL_PLACEHOLDERS = (
    "prompt_logprobs", "prompt_token_ids", "kv_transfer_params",
    "moderation", "service_tier", "system_fingerprint",
)


def _slim_snapshot_messages(messages: list[dict]) -> None:
    """In-place trim of redundant `extra.response` fields on a snapshot copy."""
    for m in messages:
        if not isinstance(m, dict):
            continue
        extra = m.get("extra")
        if not isinstance(extra, dict):
            continue
        resp = extra.get("response")
        if not isinstance(resp, dict):
            continue
        for choice in resp.get("choices") or []:
            if isinstance(choice, dict):
                choice.pop("message", None)  # verbatim dup of top-level message
        for k in _RESPONSE_NULL_PLACEHOLDERS:
            resp.pop(k, None)


# ── Model proxy that compresses history before each API call ──────────────────

class _CompressingModelProxy:
    """Wraps a Model to apply history compression before every ``query()`` call.

    Compression is driven by per-message metadata stored in the owning
    ``TraceRunnerAgent``.
    """

    def __init__(self, real_model, agent: "TraceRunnerAgent") -> None:
        self._model = real_model
        self._agent = agent
        # Per-query snapshots of the EXACT message list sent to the underlying
        # model (after compression + truncation). This is the real model input —
        # not the post-hoc accumulated agent.messages — recorded for rendering.
        # Each entry: {"turn_idx": int, "messages": [...compressed...]}.
        self._query_snapshots: list[dict] = []
        # Reactive context-400 degradation flag: None=normal (default max_tokens + high-watermark compression);
        # once this cell hits a context-overflow 400, set to _FORCED_MAX_TOKENS, after which every query from
        # this cell keeps "forcing to the low watermark + using the degraded max_tokens" until the process ends.
        self._forced_max_tokens: int | None = None

    # ---- public interface (delegates everything except query) ---------------

    def _record_snapshot(self, compressed: list[dict]) -> None:
        """Record a snapshot of this call's real model input (for the renderer to align 1:1 per turn).
        Recorded only after a successful send and a returned result, guaranteeing at most 1 per query() and no double-recording.
        Side effect: the FormatError path (model replied but format parsing failed) no longer leaves a snapshot —
        this only affects that turn's rendering snapshot, not the agent main loop / evaluation (the main loop uses agent.messages)."""
        # Slim the archived copy (drop redundant extra.response.choices[].message
        # dup + null placeholders) — the `compressed` list sent to the model is
        # left untouched.
        snapshot_msgs = copy.deepcopy(compressed)
        _slim_snapshot_messages(snapshot_msgs)
        self._query_snapshots.append({
            "turn_idx": getattr(self._agent, "_current_turn_idx", 0),
            "messages": snapshot_msgs,
        })

    def query(self, messages: list[dict], **kwargs) -> dict:
        # If this cell has previously hit a context-overflow 400 → keep degrading: force to the low watermark + lower max_tokens.
        forced = self._forced_max_tokens is not None
        send_kwargs = dict(kwargs)
        if forced:
            send_kwargs["max_tokens"] = self._forced_max_tokens
        compressed = self._compress(messages, force=forced)
        try:
            result = self._model.query(compressed, **send_kwargs)
            self._record_snapshot(compressed)   # record only after success (at most 1 per query, no double-record)
            return result
        except Exception as exc:
            if not _is_context_length_400(exc):
                raise  # non-"context-overflow 400" (e.g. tool schema error / FormatError) re-raised as-is, behavior unchanged.
            # Reactive degradation: this cell keeps using the degraded max_tokens afterward; for this call, force to the low watermark then retry once.
            # The failed first attempt is not snapshotted — only the successful retry below is.
            self._forced_max_tokens = _FORCED_MAX_TOKENS
            logger.warning(
                "[ctx-400] assistant endpoint context-overflow 400, reactive degradation: force to low watermark %d chars + "
                "max_tokens=%d, retrying once. (%r)",
                _HISTORY_LOW_CHARS, _FORCED_MAX_TOKENS, exc,
            )
            retry_kwargs = dict(kwargs)
            retry_kwargs["max_tokens"] = _FORCED_MAX_TOKENS
            retry_compressed = self._compress(messages, force=True)
            try:
                result = self._model.query(retry_compressed, **retry_kwargs)
                self._record_snapshot(retry_compressed)   # record only on a successful retry, still 1
                return result
            except Exception as exc2:
                if not _is_context_length_400(exc2):
                    raise
                # Still overflowing after degraded compression + lowered max_tokens → raise TimeoutError, handed
                # off to usersim_main.py's assistant_message fallback (forced report) for a graceful wrap-up.
                logger.error(
                    "[ctx-400] still context-overflow 400 after degraded retry → raising TimeoutError, handing off to the forced-report fallback. (%r)",
                    exc2,
                )
                raise TimeoutError(
                    f"assistant context-length 400 persists after forced compression "
                    f"to {_HISTORY_LOW_CHARS} chars + max_tokens={_FORCED_MAX_TOKENS}: {exc2}"
                ) from exc2

    def serialize(self) -> dict:
        """Merge the underlying model's serialized state with the recorded
        per-query input snapshots (surfaced at the traj top level so the
        renderer can read data['query_snapshots'])."""
        base = {}
        real_serialize = getattr(self._model, "serialize", None)
        if callable(real_serialize):
            base = real_serialize() or {}
        return recursive_merge(base, {"query_snapshots": self._query_snapshots})

    def reset_query_snapshots(self) -> None:
        self._query_snapshots = []

    def __getattr__(self, name: str):
        return getattr(self._model, name)

    # ---- compression logic --------------------------------------------------

    def _force_deletable_positions(self, result: list[dict]) -> set[int]:
        """A subclass-overridable hook: return a set of message indices to "forcibly unprotect and make deletable".

        Default empty set → is_protected behaves exactly as before (no change to this class / parent-class path).
        The user-sim subclass overrides it to mark consecutive No-tool-call FormatError users and their
        immediately-preceding degenerate assistant as deletable (the current turn / latest one is still protected, see the subclass implementation). system is never unprotected.
        """
        return set()

    def _compress(self, messages: list[dict], *, force: bool = False) -> list[dict]:
        """Total-budget history compression (replaces the old turn/big-task scheme).

        force=True (after hitting context-400): lower the trigger line from the high watermark to the low
        watermark, i.e. ">low watermark trims to ≤low watermark", bypassing the high-watermark check, reusing the
        same deletion logic, forcing history down to the minimal budget for the retry.

        Rules (applies to ALL modes):
          - role="exit"   : always removed (no LLM API accepts this role).
          - PROTECTED — always kept full, never deleted:
              * role="system"
              * every role="user"   (the simulated-user turns)
              * the FINAL turn: the last role="assistant" message AND every message
                after it (its trailing tool results) — keeps the final result intact
                and avoids orphaning its tool_calls.
          - EVERYTHING ELSE is deletable: earlier assistant messages and their tool
            results. Deletion is done in whole TURN GROUPS (an assistant message +
            the tool messages that pair with its tool_calls) OLDEST FIRST, until the
            running total of all message contents fits within ``_HISTORY_LOW_CHARS``
            (only triggered when the total first exceeds ``_HISTORY_HIGH_CHARS``).
          - Deletion REMOVES the messages entirely — including the assistant's
            ``tool_calls`` payload (the old scheme only blanked ``content`` and left
            the tool_call arguments behind). Because we always drop a complete turn
            group, no tool message is ever orphaned from its assistant.tool_calls.
          - A run of CONSECUTIVE deleted messages collapses into a SINGLE placeholder
            assistant message (``content=_DROPPED_TOOL_STUB``, no tool_calls) marking
            that a span of history was dropped — not one stub per deleted message.
        """
        # First pass: drop exit messages entirely (no pairing to preserve).
        result: list[dict] = [m for m in messages if m.get("role") != "exit"]
        n = len(result)

        # Locate the last assistant message — it and everything after it is protected.
        last_assistant_pos = -1
        for pos in range(n - 1, -1, -1):
            if result[pos].get("role") == "assistant":
                last_assistant_pos = pos
                break

        # Locate the last user message — it and every assistant after it is protected
        # (requirement ②: the last user turn and all its subsequent assistant replies are always kept).
        last_user_pos = -1
        for pos in range(n - 1, -1, -1):
            if result[pos].get("role") == "user":
                last_user_pos = pos
                break

        # Narrowing of requirement ② in the force state (this cell hit ctx-400 degradation): the assistant after
        # the last user may run dozens of steps autonomously, and keeping them all would blow past the low
        # watermark and make deletion impossible. So when force=True, only protect the [latest
        # _FORCE_KEEP_LAST_ASSTS assistant steps] — keeping recent context for the model to continue, while
        # opening up earlier intermediate steps as deletable (deleted oldest first). Normal state (force=False) still protects them all.
        recent_assts_after_user: set[int] = set()
        if force and last_user_pos >= 0:
            asst_after = [pos for pos in range(last_user_pos + 1, n)
                          if result[pos].get("role") == "assistant"]
            recent_assts_after_user = set(asst_after[-_FORCE_KEEP_LAST_ASSTS:])

        # Each user turn's "final user-facing reply" = the last assistant after that user and before the next
        # user (or the end). These are the top-level non-collapsed per-turn replies in the HTML and must be kept
        # (requirement ①: the last assistant after each user is never deleted).
        turn_final_assistant: set[int] = set()
        last_asst_in_seg = -1
        for pos in range(n):
            role = result[pos].get("role")
            if role == "user":
                if last_asst_in_seg >= 0:
                    turn_final_assistant.add(last_asst_in_seg)
                last_asst_in_seg = -1
            elif role == "assistant":
                last_asst_in_seg = pos
        if last_asst_in_seg >= 0:  # the last assistant of the trailing segment
            turn_final_assistant.add(last_asst_in_seg)

        # Subclass hook: the set of indices to forcibly unprotect and make deletable (default empty → behavior unchanged).
        force_deletable = self._force_deletable_positions(result)

        def is_protected(pos: int, msg: dict) -> bool:
            role = msg.get("role")
            # Forcibly deletable: indices named by the subclass (e.g. consecutive No-tool-call FormatError users
            # and their immediately-preceding degenerate assistant) are unprotected — but system is never unprotected.
            if pos in force_deletable and role != "system":
                return False
            if role in ("system", "user"):
                return True
            # Final turn: last assistant + its trailing tool results.
            if last_assistant_pos >= 0 and pos >= last_assistant_pos:
                return True
            # Requirement ①: each user turn's final assistant reply (the top-level non-collapsed one in the HTML).
            if pos in turn_final_assistant:
                return True
            # Requirement ②: protection of assistants after the last user.
            #   · Normal state (force=False): [all] assistants after that user are kept (original semantics).
            #   · Degraded state (force=True, this cell hit context-400): keep only the [latest
            #     _FORCE_KEEP_LAST_ASSTS steps] of assistants (see recent_assts_after_user above), opening up
            #     earlier intermediate steps as deletable — giving the model recent context to continue while
            #     deleting the long tail that blows past the low watermark oldest-first. The latest one is still
            #     covered by the last_assistant segment, and this turn's final reply is still covered by requirement ①.
            if last_user_pos >= 0 and pos > last_user_pos and role == "assistant":
                if not force:
                    return True
                if pos in recent_assts_after_user:
                    return True
            return False

        # Assign each message to a TURN GROUP: an assistant starts a new group and
        # the tool messages following it (its tool_call results) join that group;
        # system/user are their own singleton groups. Deleting always operates on a
        # whole group so an assistant.tool_calls and its tool results go together.
        group = [0] * n
        gid = 0
        for pos in range(n):
            role = result[pos].get("role")
            if role == "tool":
                group[pos] = gid  # attach to the current assistant's group
            else:
                gid += 1
                group[pos] = gid

        # A group is protected if ANY of its messages is protected.
        protected_groups = {group[pos] for pos in range(n)
                            if is_protected(pos, result[pos])}

        # Group order by first appearance (oldest first).
        order: list[int] = []
        seen = set()
        for pos in range(n):
            if group[pos] not in seen:
                seen.add(group[pos])
                order.append(group[pos])

        # High/low watermark hysteresis: trimming is triggered only when history > the high watermark; once
        # triggered, delete from the oldest turn group until ≤ the low watermark. Deletion leaves a (HIGH-LOW)
        # margin, so subsequent turns keep the prefix unchanged and hit the prefix cache, until it grows past the
        # high watermark again and one more deletion happens.
        # force=True (forced compression after context-400): the trigger line drops to the low watermark, ">low watermark trims to ≤low watermark".
        # Watermark accounting basis: content + serialized tool_calls chars (tool_calls carry full commands / file
        # content with high token density; counting only content would badly underestimate the real payload →
        # even trimming to the low watermark could still hit a context-overflow 400). Both the initial sum and the
        # deletion decrements go through this function, keeping running symmetric and drift-free.
        def _msg_chars(m: dict) -> int:
            tc = m.get("tool_calls")
            tc_len = len(json.dumps(tc, ensure_ascii=False)) if tc else 0
            return len(m.get("content") or "") + tc_len

        deleted = [False] * n
        running = sum(_msg_chars(m) for m in result)
        trigger = _HISTORY_LOW_CHARS if force else _HISTORY_HIGH_CHARS
        if running > trigger:
            for g in order:
                if running <= _HISTORY_LOW_CHARS:
                    break
                if g in protected_groups:
                    continue
                for pos in range(n):
                    if group[pos] == g and not deleted[pos]:
                        deleted[pos] = True
                        running -= _msg_chars(result[pos])

        # Rebuild, collapsing each run of consecutive deleted messages into a SINGLE
        # placeholder assistant message (no tool_calls).
        out: list[dict] = []
        i = 0
        while i < n:
            if deleted[i]:
                out.append({"role": "assistant", "content": _DROPPED_TOOL_STUB})
                while i < n and deleted[i]:
                    i += 1
            else:
                out.append(result[i])
                i += 1

        return out



# ── TraceRunnerAgent ──────────────────────────────────────────────────────────

class TraceRunnerAgent(MultiTurnAgent):
    """MultiTurnAgent with per-turn metadata tracking + smart history compression.

    Tracks which ``continue_with()`` turn and ``big_task_id`` each message
    belongs to, then applies :class:`_CompressingModelProxy` to reduce context
    length when querying the LLM.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Metadata parallel list; index i corresponds to self.messages[i]
        self._msg_meta: list[dict] = []
        self._current_turn_idx: int = 0
        self._current_big_task_id: str | None = None
        # Replace model with compressing proxy
        self.model = _CompressingModelProxy(self.model, self)

    # ---- override add_messages to maintain metadata -------------------------

    def add_messages(self, *messages: dict) -> list[dict]:
        result = super().add_messages(*messages)
        for _ in messages:
            self._msg_meta.append(
                {
                    "turn_idx": self._current_turn_idx,
                    "big_task_id": self._current_big_task_id,
                }
            )
        return result

    # ---- override run/continue_with to set metadata context -----------------

    def run(self, task: str = "", *, big_task_id: str | None = None, **kwargs) -> dict:
        """Start fresh: reset turn counter + metadata, then run normally."""
        self._current_turn_idx = 0
        self._msg_meta = []
        self._current_big_task_id = big_task_id
        reset = getattr(self.model, "reset_query_snapshots", None)
        if callable(reset):
            reset()
        return super().run(task, **kwargs)

    def continue_with(
        self,
        user_message: str,
        loaded_messages: list[dict] | None = None,
        *,
        big_task_id: str | None = None,
    ) -> dict:
        """Increment turn counter, optionally update big_task_id, then continue."""
        self._current_turn_idx += 1
        if big_task_id is not None:
            self._current_big_task_id = big_task_id
        return super().continue_with(user_message, loaded_messages=loaded_messages)


# ── Trace parsing ─────────────────────────────────────────────────────────────

def extract_tasks(trace: dict) -> list[dict]:
    """Return ordered list of tasks extracted from enabled ATTEMPT events.

    Each entry::

        {
            "big_task_id": str,
            "subtask_id":  str,
            "content":     str,   # text of the user message
            "is_seed":     bool,
        }

    All ATTEMPT events (including SEED_TASK) yield one entry per non-empty
    message in ``user_query``.  The first event whose subtask_id is
    ``"SEED_TASK"`` sets ``is_seed=True``.
    Events with ``enabled=False`` are skipped.
    """
    tasks: list[dict] = []
    for event in trace.get("events", []):
        if event.get("kind") != "ATTEMPT":
            continue
        if not event.get("enabled", False):
            continue

        big_task_id: str = event.get("big_task_id", "UNKNOWN")
        subtask_id: str = event.get("subtask_id", "")
        is_seed: bool = subtask_id == "SEED_TASK"

        # ``user_query`` may be a single message dict (agent_trace.v2 / Pro
        # traces) or a list of message dicts. Normalize to a list so iteration
        # below is uniform.
        raw_uq = event.get("user_query")
        if isinstance(raw_uq, dict):
            user_query = [raw_uq]
        elif isinstance(raw_uq, list):
            user_query = raw_uq
        else:
            user_query = []

        for msg in user_query:
            content = (msg.get("content") or "").strip()
            if content:
                tasks.append(
                    {
                        "big_task_id": big_task_id,
                        "subtask_id": subtask_id,
                        "content": content,
                        "is_seed": is_seed,
                    }
                )
    return tasks


# ── Main runner ───────────────────────────────────────────────────────────────

def run_trace(
    trace_path: Path,
    output_dir: Path,
    config_spec: list[str],
    model_name: str | None = None,
) -> dict:
    """Replay all tasks from *trace_path* through a shared mini-swe-agent session.

    Returns the ``extra`` dict of the final agent exit message.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    add_file_handler(output_dir / "trace_runner.log")

    trace = json.loads(Path(trace_path).read_text())
    instance: dict = trace.get("instance", {})
    instance_id: str = instance.get("instance_id", Path(trace_path).stem)

    tasks = extract_tasks(trace)
    if not tasks:
        logger.warning("No enabled ATTEMPT tasks found in %s", trace_path)
        return {}

    logger.info(
        "Loaded %d tasks for instance '%s' from %s", len(tasks), instance_id, trace_path
    )

    # ── Build merged config ──────────────────────────────────────────────────
    configs = [get_config_from_spec(spec) for spec in config_spec]
    if model_name:
        configs.append({"model": {"model_name": model_name}})
    config = recursive_merge(*configs)

    # ── Create model + environment (environment shared across all tasks) ─────
    model = get_model(config=config.get("model", {}))
    env = get_sb_environment(config, instance)

    # ── Create agent ─────────────────────────────────────────────────────────
    traj_path = output_dir / f"{instance_id}.traj.json"
    agent_kwargs = dict(config.get("agent", {}))
    agent_kwargs["output_path"] = traj_path

    agent = TraceRunnerAgent(model, env, **agent_kwargs)

    # ── Execute tasks ─────────────────────────────────────────────────────────
    seed = tasks[0]
    if not seed["is_seed"]:
        logger.warning(
            "First task is not SEED_TASK (subtask_id=%s); running it as seed anyway.",
            seed["subtask_id"],
        )

    logger.info("[0/%d] SEED  subtask_id=%s", len(tasks) - 1, seed["subtask_id"])
    result: dict = {}
    try:
        result = agent.run(seed["content"], big_task_id=seed["big_task_id"])
        logger.info("[0/%d] exit_status=%s", len(tasks) - 1, result.get("exit_status"))
    except Exception as exc:
        logger.error("SEED task crashed: %s\n%s", exc, traceback.format_exc())
        raise

    for idx, task in enumerate(tasks[1:], start=1):
        logger.info(
            "[%d/%d] subtask_id=%-20s  big_task_id=%s",
            idx,
            len(tasks) - 1,
            task["subtask_id"],
            task["big_task_id"],
        )
        try:
            result = agent.continue_with(
                task["content"],
                big_task_id=task["big_task_id"],
            )
            logger.info(
                "[%d/%d] exit_status=%s", idx, len(tasks) - 1, result.get("exit_status")
            )
        except Exception as exc:
            logger.error(
                "Task %d (%s) crashed: %s\n%s",
                idx,
                task["subtask_id"],
                exc,
                traceback.format_exc(),
            )
            raise

    logger.info("Finished.  Trajectory → %s", traj_path)
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

_CONFIG_SPEC_HELP = (
    "Path to YAML config files, filenames, or key=value pairs. "
    "Multiple -c flags are recursively merged (later overrides earlier). "
    "Example: -c swebench.yaml -c runs/v4flash_hxsz.yaml"
)


@app.command()
def main(
    trace_path: Path = typer.Argument(..., help="Path to the agent_trace.v2 JSON file"),
    output: str = typer.Option("", "-o", "--output", help="Output directory"),
    config_spec: list[str] = typer.Option(
        [str(DEFAULT_CONFIG_FILE)], "-c", "--config", help=_CONFIG_SPEC_HELP
    ),
    model: str | None = typer.Option(None, "-m", "--model", help="Model name override"),
) -> None:
    """Replay an agent trace through mini-swe-agent."""
    output_dir = Path(output) if output else Path("runs") / Path(trace_path).stem
    run_trace(
        trace_path=trace_path,
        output_dir=output_dir,
        config_spec=config_spec,
        model_name=model,
    )


if __name__ == "__main__":
    app()
