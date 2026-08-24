"""reply_to_user tool: schema + action parsing for the user-sim assistant.

This is the user-sim counterpart to mini-swe-agent's
``models/utils/actions_toolcall.py``. We do NOT edit that file; instead we
expose a second tool (``reply_to_user``) alongside ``bash`` and a parser that
accepts both. The model imports BASH_TOOL / format_toolcall_observation_messages
from mini-swe-agent so behaviour for bash stays identical.

Semantics (user-sim branch):
  - ``reply_to_user`` is how the assistant talks to the (simulated) user —
    either to ask a clarifying question or to deliver its final answer.
  - Calling it ENDS the current turn and hands control back to the user
    (the turn-ending logic lives in UserSimAgent.execute_actions).
  - ``reply_to_user`` fully replaces the old ``COMPLETE_TASK_AND_SUBMIT...``
    bash-submission path for ending a turn.
  - If the model emits NO tool call at all, we raise FormatError with a message
    telling it to use ``reply_to_user`` (to talk to the user) or ``bash`` (to
    run a command) — mini-swe-agent then appends that as a user message and the
    model retries.
"""

from __future__ import annotations

import json

from jinja2 import StrictUndefined, Template

from minisweagent.exceptions import FormatError
from minisweagent.models.utils.actions_toolcall import BASH_TOOL  # reused as-is

REPLY_TO_USER_TOOL = {
    "type": "function",
    "function": {
        "name": "reply_to_user",
        "description": (
            "Send a message to the user — either to ask a clarifying question "
            "or to deliver your final answer for this turn. Calling this tool "
            "ENDS your turn and hands control back to the user, who will read "
            "your message and respond. Use this (not a bash echo) whenever you "
            "want to talk to the user or finish the current request."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to show the user (your question or final answer).",
                }
            },
            "required": ["message"],
        },
    },
}

# Tools advertised to the user-sim assistant model.
USERSIM_TOOLS = [BASH_TOOL, REPLY_TO_USER_TOOL]

# Marker key used on a parsed action to denote a reply_to_user call.
REPLY_ACTION_KEY = "reply_to_user"

# When consecutive FormatErrors (the model gave no tool call at all / gave an illegal
# tool call) reach this cap, UserSimAgent stops re-prompting and instead takes the raw
# text of the last degenerate assistant as this turn's final reply
# (see usersim_agent.UserSimAgent.query). The single source of truth lives here, for
# usersim_agent to import, and is also embedded in the _NO_TOOLCALL_ERROR prompt text
# below to avoid the number drifting in two places.
FORMAT_ERROR_GIVEUP = 5

_NO_TOOLCALL_ERROR = (
    "No tool calls found in the response. Every response MUST include at least "
    "one tool call. To run a shell command, call the `bash` tool. To talk to "
    "the user — ask a clarifying question or give your final answer — call the "
    "`reply_to_user` tool (this ends your turn and returns control to the user). "
    f"Note: if you fail to make a valid tool call {FORMAT_ERROR_GIVEUP} times in a "
    "row, your latest response will be delivered to the user as your final answer "
    "and your turn will end automatically."
)


def _format_error(format_error_template: str, error: str) -> FormatError:
    return FormatError(
        {
            "role": "user",
            "content": Template(format_error_template, undefined=StrictUndefined).render(
                error=error, actions=[]
            ),
            "extra": {"interrupt_type": "FormatError"},
        }
    )


def parse_actions_with_reply(tool_calls: list, *, format_error_template: str) -> list[dict]:
    """Parse tool calls, accepting both ``bash`` and ``reply_to_user``.

    Returns a list of action dicts:
      - bash:          {"command": <str>, "tool_call_id": <id>}
      - reply_to_user: {"reply_to_user": <str>, "tool_call_id": <id>}

    Raises FormatError (with a reply_to_user-aware message) when no tool call is
    present, or when a call is malformed / names an unknown tool.
    """
    if not tool_calls:
        raise _format_error(format_error_template, _NO_TOOLCALL_ERROR)

    actions: list[dict] = []
    for tool_call in tool_calls:
        name = tool_call.function.name
        error_msg = ""
        args: dict = {}
        try:
            args = json.loads(tool_call.function.arguments)
        except Exception as e:  # noqa: BLE001
            error_msg = f"Error parsing tool call arguments: {e}."

        if name == "bash":
            if not isinstance(args, dict) or "command" not in args:
                error_msg += "Missing 'command' argument in bash tool call."
            if error_msg:
                raise _format_error(format_error_template, error_msg.strip())
            actions.append({"command": args["command"], "tool_call_id": tool_call.id})
        elif name == "reply_to_user":
            if not isinstance(args, dict) or "message" not in args:
                error_msg += "Missing 'message' argument in reply_to_user tool call."
            if error_msg:
                raise _format_error(format_error_template, error_msg.strip())
            actions.append({REPLY_ACTION_KEY: args["message"], "tool_call_id": tool_call.id})
        else:
            error_msg += (
                f"Unknown tool '{name}'. Only 'bash' and 'reply_to_user' are available."
            )
            raise _format_error(format_error_template, error_msg.strip())

    return actions
