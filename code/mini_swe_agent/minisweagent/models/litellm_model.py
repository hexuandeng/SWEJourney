import json
import logging
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import litellm
from pydantic import BaseModel

from minisweagent.exceptions import FormatError
from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.actions_toolcall import (
    BASH_TOOL,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)
from minisweagent.models.utils.anthropic_utils import _reorder_anthropic_thinking_blocks
from minisweagent.models.utils.cache_control import set_cache_control
from minisweagent.models.utils.openai_multimodal import expand_multimodal_content
from minisweagent.models.utils.retry import retry

logger = logging.getLogger("litellm_model")


def _find_degen_cut(s: str) -> int | None:
    """Locate the offset where degeneration (long repetition) starts; return None if there is no long repetition.

    Degeneration anchors (validated with zero false positives on Kimi-K3's 825 finish_reason=length samples):
      · some single character repeats consecutively >= 100 times, or
      · after whitespace splitting, the same short token (<=4 chars) repeats consecutively >= 20 times.
    """
    if not s:
        return None
    n = len(s)
    # single-character long repetition
    i = 0
    while i < n:
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        if j - i >= 100:
            return i
        i = j if j > i else i + 1
    # short-token consecutive repetition (e.g. "@ @ @ @ ...")
    m = re.search(r"(\S{1,4})(?:\s+\1){19,}", s)
    if m:
        return m.start()
    return None


def _truncate_degenerate(text: str, finish_reason: str | None) -> str:
    """When hitting max_tokens (finish_reason=length), scrub the degenerate tail to prevent @@@ garbage from backfilling into the next turn and triggering a cascade.

    Rule (aligned with the user): when finish_reason=length, the cut position is the earlier of the "degeneration
    repetition start" and the "first half" — the degenerate tail is always cut cleanly, and a length response
    keeps at most its first half. Non-length responses are returned as-is.
    """
    if finish_reason != "length" or not text:
        return text
    cut = _find_degen_cut(text)
    half = len(text) // 2
    cut = min(cut, half) if cut is not None else half
    return text[:cut]


_THINK_INNER = re.compile(r"<think>((?:(?!<think>).)*?)</think>", re.S)
_THINK_TAG = re.compile(r"</?think>")
THINK_MAX_CHARS = 20000  # ~p99 of a normal model's reasoning length; thinking beyond this is treated as degeneration and hard-cut.


def _collapse_think(content: str, keep_inner: bool) -> str:
    """Collapse the proliferation of <think> tags in content down to at most one think block, and cap the thinking volume.

    Kimi-K2.7 re-echoes the <think> tags it emitted into content on the previous turn as context, turn by turn,
    snowballing a single message to hundreds of thousands ~ millions of chars (nested open tags
    <think><think>… with only one matching </think>, the middle filled with whitespace), blowing up the input
    and triggering a cascade. Cleanup rules:

      · keep_inner=True (reasoning is empty, thinking is mixed into the content body, as in K2.7):
        among all **innermost** paired <think>…</think>, take the one with the longest content (where the real
        thinking is), cap its content at THINK_MAX_CHARS, and re-wrap it into a single <think>…</think>; all
        other <think>/</think> tags in the body (including nested outer ones and orphan tags) are stripped.
      · keep_inner=False (reasoning is non-empty, the real thinking is in the reasoning field, wrapped separately
        by an outer wrapper): all <think>/</think> tags in the content body are stripped (the real thinking is
        not here).

    Either way, in the final content <think> and </think> each appear at most once.
    """
    if "<think>" not in content and "</think>" not in content:
        return content
    if keep_inner:
        blocks = list(_THINK_INNER.finditer(content))
        best = max((m.group(1) for m in blocks), key=lambda s: len(s.strip()), default="")
        # best may still contain literal <think>/</think> inside; strip them first to ensure only the wrapper pair of tags remains.
        best = _THINK_TAG.sub("", best).strip()
        if len(best) > THINK_MAX_CHARS:
            best = best[:THINK_MAX_CHARS]
        # Strip all think tags in the body, then re-wrap the real thinking into a single block placed at the front.
        body = _THINK_TAG.sub("", content)
        return (f"<think>\n{best}\n</think>\n" if best else "") + body
    # reasoning non-empty: the think tags in the body are all re-echoed garbage, strip them all (the wrapper is added separately by the outer layer)
    return _THINK_TAG.sub("", content)


class LitellmModelConfig(BaseModel):
    model_name: str
    """Model name. Highly recommended to include the provider in the model name, e.g., `anthropic/claude-sonnet-4-5-20250929`."""
    model_kwargs: dict[str, Any] = {}
    """Additional arguments passed to the API."""
    litellm_model_registry: Path | str | None = os.getenv("LITELLM_MODEL_REGISTRY_PATH")
    """Model registry for cost tracking and model metadata. See the local model guide (https://mini-swe-agent.com/latest/models/local_models/) for more details."""
    set_cache_control: Literal["default_end"] | None = None
    """Set explicit cache control markers, for example for Anthropic models"""
    cost_tracking: Literal["default", "ignore_errors"] = os.getenv("MSWEA_COST_TRACKING", "default")
    """Cost tracking mode for this model. Can be "default" or "ignore_errors" (ignore errors/missing cost info)"""
    format_error_template: str = "{{ error }}"
    """Template used when the LM's output is not in the expected format."""
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    """Template used to render the observation after executing an action."""
    multimodal_regex: str = ""
    """Regex to extract multimodal content. Empty string disables multimodal processing."""


class LitellmModel:
    abort_exceptions: list[type[Exception]] = [
        litellm.exceptions.UnsupportedParamsError,
        litellm.exceptions.NotFoundError,
        litellm.exceptions.PermissionDeniedError,
        litellm.exceptions.ContextWindowExceededError,
        litellm.exceptions.AuthenticationError,
        KeyboardInterrupt,
    ]

    def __init__(self, *, config_class: Callable = LitellmModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        if self.config.litellm_model_registry and Path(self.config.litellm_model_registry).is_file():
            litellm.utils.register_model(json.loads(Path(self.config.litellm_model_registry).read_text()))

    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            return litellm.completion(
                model=self.config.model_name,
                messages=messages,
                tools=[BASH_TOOL],
                **(self.config.model_kwargs | kwargs),
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        prepared = [{k: v for k, v in msg.items() if k != "extra"} for msg in messages]
        prepared = _reorder_anthropic_thinking_blocks(prepared)
        return set_cache_control(prepared, mode=self.config.set_cache_control)

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                response = self._query(self._prepare_messages_for_api(messages), **kwargs)
        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        message = response.choices[0].message.model_dump()
        finish_reason = response.choices[0].finish_reason
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        content = _truncate_degenerate(message.get("content") or "", finish_reason)
        if reasoning:
            # The real thinking is in the reasoning field: wrap it into the sole <think> wrapper at the outer level,
            # and strip all <think> tags the model emitted itself in the content body (all re-echoed garbage).
            # reasoning itself may also contain literal <think>/</think>; strip them first to ensure the wrapper is the only source.
            reasoning = _THINK_TAG.sub("", _truncate_degenerate(reasoning, finish_reason)).strip()
            think_block = f"<think>\n{reasoning}\n</think>\n"
            content = _collapse_think(content, keep_inner=False).strip()
            content = think_block + content if not content.startswith(think_block) else content
        else:
            # reasoning is empty (e.g. Kimi-K2.7): thinking is mixed into the content body; take the innermost
            # longest think block (where the real thinking is), re-wrap it into a single block, and strip all other think tags.
            content = _collapse_think(content, keep_inner=True).strip()
        # Cap the whole content at ~p99: K2.7's body degeneration (studying / "high-school stage" re-echoing,
        # blockquote flooding, etc., falling outside </think>) can't be stopped by tag cleanup, so hard-cut the
        # final volume here as a fallback. Normal models rarely hit this cap (reasoning p99 ≈ THINK_MAX_CHARS).
        if len(content) > THINK_MAX_CHARS:
            content = content[:THINK_MAX_CHARS]
        message["content"] = content
        # The thinking now lives only inside `content` (subject to truncation).
        # Drop every other place litellm duplicates it — the standalone
        # `reasoning_content`/`reasoning` fields AND `provider_specific_fields`
        # (whose `reasoning` key is a third verbatim copy). Otherwise those copies
        # are re-sent every turn and accumulate unbounded, blowing the input-token cap.
        message.pop("reasoning_content", None)
        message.pop("reasoning", None)
        message.pop("provider_specific_fields", None)
        # Also drop `provider_specific_fields` from the response snapshot archived
        # under `extra` (both the choice level and its nested message level). Nothing
        # in this project reads it and the OpenAI-compatible send path never consumes
        # it, so it is dead weight — its `reasoning` key is a third verbatim copy of
        # the thinking. `reasoning_content`/`reasoning` are intentionally KEPT in extra.
        response_dump = response.model_dump()
        for choice in response_dump.get("choices") or []:
            if isinstance(choice, dict):
                choice.pop("provider_specific_fields", None)
                resp_msg = choice.get("message")
                if isinstance(resp_msg, dict):
                    resp_msg.pop("provider_specific_fields", None)
        try:
            actions = self._parse_actions(response)
        except FormatError as e:
            message["extra"] = {
                "actions": [],
                "response": response_dump,
                **cost_output,
                "timestamp": time.time(),
            }
            e.assistant_message = message
            raise
        message["extra"] = {
            "actions": actions,
            "response": response_dump,
            **cost_output,
            "timestamp": time.time(),
        }
        return message

    def _calculate_cost(self, response) -> dict[str, float]:
        try:
            cost = litellm.cost_calculator.completion_cost(response, model=self.config.model_name)
            if cost <= 0.0:
                raise ValueError(f"Cost must be > 0.0, got {cost}")
        except Exception as e:
            cost = 0.0
            if self.config.cost_tracking != "ignore_errors":
                msg = (
                    f"Error calculating cost for model {self.config.model_name}: {e}, perhaps it's not registered? "
                    "You can ignore this issue from your config file with cost_tracking: 'ignore_errors' or "
                    "globally with export MSWEA_COST_TRACKING='ignore_errors'. "
                    "Alternatively check the 'Cost tracking' section in the documentation at "
                    "https://klieret.short.gy/mini-local-models. "
                    " Still stuck? Please open a github issue at https://github.com/SWE-agent/mini-swe-agent/issues/new/choose!"
                )
                logger.critical(msg)
                raise RuntimeError(msg) from e
        return {"cost": cost}

    def _parse_actions(self, response) -> list[dict]:
        """Parse tool calls from the response. Raises FormatError if unknown tool."""
        tool_calls = response.choices[0].message.tool_calls or []
        return parse_toolcall_actions(tool_calls, format_error_template=self.config.format_error_template)

    def format_message(self, **kwargs) -> dict:
        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Format execution outputs into tool result messages."""
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }
