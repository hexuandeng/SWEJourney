"""Per-message truncation — the single implementation shared by trace_runner_v2 and truncating_proxy.

Motivation for the extraction: trace_runner_v2._TruncatingProxy._truncate (gold-prefix replay path) and
truncating_proxy._truncate_msg (benchmark sampling path) used to be two byte-for-byte equivalent copies,
differing only in layout (early-return vs. reassign-then-return). To avoid reinventing the wheel, this pure
logic is consolidated into this module, and both sites import and call it directly. **The behavior is exactly
identical to the two old copies, with no difference whatsoever.**

Note: this module only handles the "per-message truncation" layer. The overall-history "high/low watermark
hysteresis" compression (_compress) is deliberately different between the two paths (single-turn benchmark sampling
protects 3 messages vs. multi-turn replay protects 5), and is not shared here.

Truncation rules (PER_MSG_MAX_CHARS, default 6000 chars):
  - role="tool"      : keep the **earliest** PER_MSG_MAX_CHARS chars (head), append ``\\n... ...`` at the tail
  - role="assistant" : keep only the **latest** PER_MSG_MAX_CHARS chars of content (tail),
                       prepend ``... ...\\n`` at the head; tool_calls are kept as-is
  - role="user"      : not truncated

The elision marker is the minimal ``... ...`` (no char count / explanatory text, to avoid the model
learning it as a text pattern and echoing it back).
"""

from __future__ import annotations

# Per-message content cap (in chars). Both paths have historically used 6000.
PER_MSG_MAX_CHARS: int = 6_000


def truncate_msg(msg: dict) -> dict:
    """Truncate a single message's content by role, returning a new dict (does not mutate the argument).

    tool keeps the head, assistant.content keeps the tail, user is untouched; ``... ...`` is inserted at the
    truncation point. Returned as-is when not over the limit or when the role is out of scope.
    """
    role = msg.get("role", "")

    if role == "tool":
        content: str = msg.get("content") or ""
        if len(content) > PER_MSG_MAX_CHARS:
            # Minimal elision marker: no char count / Chinese explanation, to avoid the model learning it as a text pattern and echoing it back.
            notice = "\n... ..."
            return {**msg, "content": content[:PER_MSG_MAX_CHARS] + notice}

    elif role == "assistant":
        content = msg.get("content") or ""
        if content and len(content) > PER_MSG_MAX_CHARS:
            notice = "... ...\n"
            # Only truncate content; tool_calls are kept as-is
            return {**msg, "content": notice + content[-PER_MSG_MAX_CHARS:]}

    # role="user" and others: not truncated, return as-is
    return msg
