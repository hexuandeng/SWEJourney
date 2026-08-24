"""UserSimAgent — TraceRunnerAgentV2 whose assistant can talk to the user via
the ``reply_to_user`` tool, which ends the turn.

Subclass-only; nothing in mini-swe-agent is modified.

Behaviour added on top of TraceRunnerAgentV2:
  - The underlying model is UserSimLitellmModel (advertises reply_to_user),
    re-wrapped in the same _TruncatingProxy the V2 agent uses, so history
    compression + per-message truncation are preserved.
  - execute_actions is overridden: a ``reply_to_user`` action is NOT run in the
    shell. Instead we
        1. emit a paired tool-result message for it (an ack), keeping the
           tool_call/tool-result pairing valid so the next API call's history
           is not rejected, then
        2. append a ``role="exit"`` message with exit_status="ReplyToUser" and
           extra.reply = the message the assistant sent the user.
    The base run/continue_with loop breaks as soon as messages[-1] is an exit,
    so a reply_to_user call ends the turn and hands control back to the user.
  - bash actions behave exactly as before (env.execute + observation message).

If the model produces neither bash nor reply_to_user, the model layer raises
FormatError (see reply_tool.parse_actions_with_reply) which mini-swe-agent turns
into a user message prompting it to use one of the two tools.
"""

from __future__ import annotations

import logging

from minisweagent.exceptions import FormatError, InterruptAgentFlow
from minisweagent.models.utils.history_truncate import PER_MSG_MAX_CHARS as _PER_MSG_MAX_CHARS
from minisweagent.run.trace_runner_v2 import TraceRunnerAgentV2, _TruncatingProxy

from agent.reply_tool import REPLY_ACTION_KEY, FORMAT_ERROR_GIVEUP as _FORMAT_ERROR_GIVEUP
from agent.usersim_model import UserSimLitellmModel

logger = logging.getLogger("usersim_agent")

# Ack shown to the model as the tool-result for a reply_to_user call. Kept short
# (it lives in history); its only job is to satisfy tool_call pairing.
_REPLY_ACK = "[Message delivered to the user. Your turn has ended; awaiting their reply.]"

# The consecutive-FormatError giveup cap _FORMAT_ERROR_GIVEUP is imported from
# reply_tool (single source of truth); that same constant is also embedded in the
# English prompt text of reply_tool._NO_TOOLCALL_ERROR, so that "the count written in
# the prompt" and "the count at which query() actually gives up" always stay in sync
# and never drift in two places.


class _UserSimTruncatingProxy(_TruncatingProxy):
    """user-sim-specific truncating proxy: on top of the parent's two-layer
    compression, it additionally marks consecutive No-tool-call FormatError users
    (``extra.interrupt_type=="FormatError"``) and the immediately preceding degenerate
    assistant as "deletable", handing them to the parent _compress's high/low-watermark
    whole-group deletion logic for cleanup.

    Only _force_deletable_positions is overridden (a neutral hook provided by
    trace_runner._CompressingModelProxy); deletion timing / watermarks / fold
    placeholders all follow the parent unchanged, no compression strategy is altered.
    A FormatError user has no tool_call pairing and a degenerate assistant has no
    tool_calls -> deletion never orphans a tool_call. The current turn / latest message
    is still protected by the parent's is_protected as usual (last assistant, everything
    after the last user, each turn's final assistant); this only affects older history
    buildup.
    """

    def _force_deletable_positions(self, result: list[dict]) -> set[int]:
        deletable: set[int] = set()
        for pos, msg in enumerate(result):
            if msg.get("role") != "user":
                continue
            if (msg.get("extra") or {}).get("interrupt_type") != "FormatError":
                continue
            deletable.add(pos)
            # The immediately preceding assistant (i.e. the degenerate reply that
            # triggered this FormatError) is also released from protection.
            j = pos - 1
            while j >= 0:
                if result[j].get("role") == "assistant":
                    deletable.add(j)
                    break
                # If a tool is sandwiched in between (should not happen, since a
                # FormatError means no tool call), skip it and keep scanning backward.
                if result[j].get("role") in ("system", "user"):
                    break
                j -= 1
        return deletable


class UserSimAgent(TraceRunnerAgentV2):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Consecutive-FormatError counter (scope: all FormatErrors; reset to zero as
        # soon as a valid tool call is produced).
        self._consecutive_format_errors = 0
        # After TraceRunnerAgentV2.__init__ the proxy stack is
        # _TruncatingProxy(real_model). We swap it for the user-sim variant that
        # also frees No-tool-call history from protection (Change ①), preserving
        # the same underlying real model + all compression/truncation behaviour.
        real_model = self._unwrap_real_model(self.model)
        if not isinstance(real_model, UserSimLitellmModel):
            real_model = self._coerce_to_usersim_model(real_model)
        self.model = _UserSimTruncatingProxy(real_model, self)

    @staticmethod
    def _unwrap_real_model(model):
        """Peel proxy layers (._model) until reaching the underlying real model."""
        seen = 0
        while hasattr(model, "_model") and seen < 8:
            model = model._model
            seen += 1
        return model

    @staticmethod
    def _coerce_to_usersim_model(real_model):
        """Return a UserSimLitellmModel mirroring ``real_model``'s config."""
        if isinstance(real_model, UserSimLitellmModel):
            return real_model
        cfg = getattr(real_model, "config", None)
        kwargs = cfg.model_dump() if cfg is not None else {}
        logger.info("[usersim] wrapping %s as UserSimLitellmModel", type(real_model).__name__)
        return UserSimLitellmModel(**kwargs)

    def query(self) -> dict:
        """On top of DefaultAgent.query, layer on "consecutive-FormatError counting + giveup at the cap".

        Normal / successful reply_to_user -> reset the counter, return as-is.
        FormatError (the model gave no tool call / gave an illegal tool call) -> counter +1:
          · < _FORMAT_ERROR_GIVEUP: re-raise as-is (the parent main loop appends a "No tool calls" re-prompt and retries, behaviour unchanged).
          · >= _FORMAT_ERROR_GIVEUP: stop re-prompting; take the raw text of that last degenerate assistant as this turn's final reply,
            build a role="exit" (exit_status="ReplyToUser", reply=raw text, extra.format_error_giveup=True),
            and instead raise InterruptAgentFlow(exit_msg) -> the main loop's add_messages lands the exit -> this turn ends on that exit.

        Note: the parent DefaultAgent.query's except block has already added this degenerate assistant into self.messages
        (e.assistant_message), so its content is available at giveup time; at this point we do NOT append the No-tool-call user re-prompt.
        """
        try:
            message = super().query()
        except FormatError as e:
            self._consecutive_format_errors += 1
            if self._consecutive_format_errors < _FORMAT_ERROR_GIVEUP:
                raise
            # At the cap: reset to zero and take the raw text of that last degenerate assistant as the final reply.
            self._consecutive_format_errors = 0
            am = getattr(e, "assistant_message", None)
            reply_text = ((am or {}).get("content") or "").strip()
            # The degenerate reply at giveup time is often the overly long <think> monologue the model
            # spits out when it "gives no tool call"; landing it directly into history as the final reply
            # would blow up the context (precisely one of the triggers of ctx-400). Truncate to
            # PER_MSG_MAX_CHARS characters (keep the tail, same policy as per-message assistant truncation),
            # only truncating when over the limit.
            if len(reply_text) > _PER_MSG_MAX_CHARS:
                reply_text = "... ...\n" + reply_text[-_PER_MSG_MAX_CHARS:]
            logger.warning(
                "[usersim] %d consecutive FormatErrors (no valid tool call) -> taking the last degenerate reply as the final reply"
                " (exit_status=ReplyToUser, format_error_giveup=True)",
                _FORMAT_ERROR_GIVEUP,
            )
            exit_msg = self.model.format_message(
                role="exit",
                content=reply_text,
                extra={
                    "exit_status": "ReplyToUser",
                    "submission": "",
                    "reply": reply_text,
                    "format_error_giveup": True,
                },
            )
            raise InterruptAgentFlow(exit_msg) from e
        self._consecutive_format_errors = 0
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        """Run bash actions normally; turn a reply_to_user action into a turn-ending exit."""
        actions = message.get("extra", {}).get("actions", [])

        outputs: list[dict] = []
        reply_text: str | None = None
        for action in actions:
            if REPLY_ACTION_KEY in action:
                # Do not touch the shell; produce an ack output so the paired
                # tool-result message renders cleanly.
                outputs.append({"output": _REPLY_ACK, "returncode": 0, "exception_info": ""})
                if reply_text is None:  # first reply in the turn wins
                    reply_text = action.get(REPLY_ACTION_KEY) or ""
            else:
                outputs.append(self.env.execute(action))

        # Paired tool-result messages for every action (keeps tool_call pairing
        # valid so the next call's compressed history is accepted by the API).
        obs = self.add_messages(
            *self.model.format_observation_messages(message, outputs, self.get_template_vars())
        )

        if reply_text is not None:
            exit_msg = self.model.format_message(
                role="exit",
                content=reply_text,
                extra={
                    "exit_status": "ReplyToUser",
                    "submission": "",
                    "reply": reply_text,
                },
            )
            obs = obs + self.add_messages(exit_msg)

        return obs
