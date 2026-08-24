"""Input-truncation model proxy for benchmark runs.

Adapts trace_runner_v2's two input-truncation layers into a self-contained version independent of trace
metadata, applying the following to messages before every model.query():

  ① Overall-history budget "high/low watermark hysteresis" (_HISTORY_HIGH_CHARS / _HISTORY_LOW_CHARS):
     trimming is triggered only when the total history chars > the high watermark (default 256000); once
     triggered, delete whole TURN GROUPS (one assistant + its following tool results) starting from the oldest,
     **deleting in one pass until ≤ the low watermark (default 160000)** before stopping, leaving a (HIGH-LOW)
     margin, so that subsequent turns of history stay below the high watermark with the prefix completely
     unchanged → hitting the prefix cache the whole time, until it grows past the high watermark again and one
     more deletion happens.
     (The old scheme was a single threshold of 256000: once crossed, every turn had to delete the oldest span
      to compress back → the prefix changed every turn and the cache was invalidated turn by turn, "add one
      span, delete one span", which was both slow and expensive.) Never deleted:
       - role="system"
       - all role="user"
       - the last assistant and every message after it (preserving the final turn, avoiding orphaned tool_calls)
     role="exit" is always deleted (the API doesn't accept it). A run of consecutive deletions collapses into a
     single placeholder assistant message (content=_DROPPED_STUB, no tool_calls).

  ② Per-message PER_MSG_MAX_CHARS (default 6000 chars) truncation, inserting the minimal elision marker
     ``... ...`` at the truncation point (no char count / explanatory text, to avoid the model learning it as a
     text pattern and echoing it back):
       - role="tool"      : keep the earliest PER_MSG_MAX_CHARS chars (head)
       - role="assistant" : keep only the latest PER_MSG_MAX_CHARS chars of content (tail),
                            tool_calls are kept as-is
       - role="user"      : not truncated

Behavior matches v2's _CompressingModelProxy + _TruncatingProxy, only dropping v2's agent-metadata-dependent
query_snapshots / turn-alignment logic.
"""

from __future__ import annotations

# Per-message truncation (layer ②) has been consolidated into the shared module, reusing the same
# implementation as trace_runner_v2. PER_MSG_MAX_CHARS is imported from there (still 6000, behavior unchanged).
from minisweagent.models.utils.history_truncate import PER_MSG_MAX_CHARS, truncate_msg

HISTORY_BUDGET_CHARS: int = 256_000  # kept for backward-compat naming, retained as an alias for the high watermark
# High/low watermark hysteresis (see module docstring ①): triggered only when > HIGH, deletes until ≤ LOW.
# The upstream model endpoint's input cap is about 256000, hence high watermark 256000, low watermark 160000.
_HISTORY_HIGH_CHARS: int = 256_000
_HISTORY_LOW_CHARS: int = 160_000
_DROPPED_STUB: str = "[history-elided]"


def _compress_history(messages: list[dict]) -> list[dict]:
    """Overall-history budget compression (high/low watermark hysteresis), deleting the oldest turn group."""
    result: list[dict] = [m for m in messages if m.get("role") != "exit"]
    n = len(result)
    if n == 0:
        return result

    last_assistant_pos = -1
    for pos in range(n - 1, -1, -1):
        if result[pos].get("role") == "assistant":
            last_assistant_pos = pos
            break

    def is_protected(pos: int, msg: dict) -> bool:
        role = msg.get("role")
        if role in ("system", "user"):
            return True
        if last_assistant_pos >= 0 and pos >= last_assistant_pos:
            return True
        return False

    # turn group: an assistant starts a new group, its following tool results join that group; system/user are each their own singleton.
    group = [0] * n
    gid = 0
    for pos in range(n):
        role = result[pos].get("role")
        if role == "tool":
            group[pos] = gid
        else:
            gid += 1
            group[pos] = gid

    protected_groups = {group[pos] for pos in range(n) if is_protected(pos, result[pos])}

    order: list[int] = []
    seen: set[int] = set()
    for pos in range(n):
        if group[pos] not in seen:
            seen.add(group[pos])
            order.append(group[pos])

    deleted = [False] * n
    running = sum(len(m.get("content") or "") for m in result)
    # Hysteresis: only enter trimming when over the high watermark; once entered, keep deleting until ≤ the low watermark.
    if running > _HISTORY_HIGH_CHARS:
        for g in order:
            if running <= _HISTORY_LOW_CHARS:
                break
            if g in protected_groups:
                continue
            for pos in range(n):
                if group[pos] == g and not deleted[pos]:
                    deleted[pos] = True
                    running -= len(result[pos].get("content") or "")

    out: list[dict] = []
    i = 0
    while i < n:
        if deleted[i]:
            out.append({"role": "assistant", "content": _DROPPED_STUB})
            while i < n and deleted[i]:
                i += 1
        else:
            out.append(result[i])
            i += 1
    return out


def _truncate_msg(msg: dict) -> dict:
    """[migrated] The per-message truncation implementation has been consolidated into history_truncate.truncate_msg.

    This thin wrapper is kept only for compatibility with any possibly-existing old references; new code should
    use truncate_msg directly. Behavior is byte-for-byte identical to before the migration.
    """
    return truncate_msg(msg)


class TruncatingModelProxy:
    """Wraps a model, doing the same two-layer input truncation as v2 before query(). All other attributes pass through."""

    def __init__(self, real_model) -> None:
        self._model = real_model

    def query(self, messages: list[dict], **kwargs) -> dict:
        compressed = _compress_history(messages)
        truncated = [truncate_msg(m) for m in compressed]
        return self._model.query(truncated, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._model, name)
