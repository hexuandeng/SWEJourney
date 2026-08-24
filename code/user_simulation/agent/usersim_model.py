"""UserSimLitellmModel — LitellmModel that also advertises the reply_to_user tool.

Subclass-only; mini-swe-agent's LitellmModel is untouched. Selected via the
``model_class`` config key in the assistant config yaml, e.g.::

    model:
      model_class: "agent.usersim_model.UserSimLitellmModel"
      model_name: "openai/your-assistant-model"
      ...

Differences vs LitellmModel:
  - ``_query`` advertises BOTH ``bash`` and ``reply_to_user`` tools.
  - ``_parse_actions`` accepts either tool (via parse_actions_with_reply).

Turn-ending behaviour (a reply_to_user call ends the turn) lives in
UserSimAgent.execute_actions; the model only parses the calls.
"""

from __future__ import annotations

import json
import logging
import time

import litellm
import requests

from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_model import LitellmModel

from agent.reply_tool import USERSIM_TOOLS, parse_actions_with_reply

logger = logging.getLogger("usersim_model")

# Hard input-length gate. The model's true context window is 1,048,576 tokens;
# letting a request approach that limit (a) gets rejected by vLLM anyway and
# (b) has been observed to trigger a GPU MMU fault (Xid 31) in the fp8/MTP
# kernels, taking down the whole DP serving group. A prompt this long almost
# always means a previous attempt's context ran away, so we treat it as a
# failure of this turn rather than forwarding it to the server.
MAX_INPUT_TOKENS = 800_000


def _count_input_tokens(model_name: str, messages: list[dict]) -> int:
    """Best-effort input token count; never raises (falls back to a char heuristic)."""
    try:
        return int(litellm.token_counter(model=model_name, messages=messages))
    except Exception:
        chars = 0
        for m in messages:
            c = m.get("content")
            if isinstance(c, str):
                chars += len(c)
            elif isinstance(c, list):
                for part in c:
                    if isinstance(part, dict):
                        chars += len(str(part.get("text", "")))
        return chars // 4  # ~4 chars/token


# ============================================================================
# /v1/responses bridge (enabled only when model_kwargs.use_responses_api=True)
# ----------------------------------------------------------------------------
# Background: under this gateway's /v1/chat/completions, function tools require
# reasoning_effort='none' (otherwise 400), which effectively turns off reasoning.
# gpt-5.6 is a reasoning model; with reasoning off it barely scratches the surface
# and does not dig deep. The gateway's /v1/responses, by contrast, allows function
# tools to coexist with an arbitrary reasoning.effort.
# Here, inside _query, we translate "chat messages + chat-tools" into the native
# Responses format and send it out, then translate output[] back into a litellm
# ModelResponse (with choices[0].message.tool_calls), so the base class's
# query()/_parse_actions/pricing/archiving all need no changes. Every turn resends
# everything (does not use previous_response_id), consistent with the existing
# harness's behaviour of resending the full messages each turn.
# The switch is off by default -> other models (which do not set this field) still
# go through the original litellm.completion, zero impact.
# ============================================================================

def _chat_tools_to_responses(tools: list[dict]) -> list[dict]:
    """chat tools ({'type':'function','function':{name,...}}) -> responses top-level name format."""
    out = []
    for t in tools:
        fn = t.get("function", {})
        out.append({
            "type": "function",
            "name": fn.get("name"),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {}),
        })
    return out


def _messages_to_responses_input(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """chat messages -> (instructions, input[]). Resends everything each turn; system folds into instructions."""
    instructions: str | None = None
    items: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            instructions = (instructions or "") + (m.get("content") or "")
        elif role == "user":
            items.append({
                "type": "message", "role": "user",
                "content": [{"type": "input_text", "text": m.get("content") or ""}],
            })
        elif role == "assistant":
            txt = m.get("content")
            if txt:
                items.append({
                    "type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": txt}],
                })
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function", {})
                items.append({
                    "type": "function_call",
                    "name": fn.get("name"),
                    "arguments": fn.get("arguments") or "{}",
                    "call_id": tc.get("id"),
                })
        elif role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": m.get("tool_call_id"),
                "output": m.get("content") or "",
            })
        # other internal roles (such as exit) are not sent to the model
    return instructions, items


def _responses_json_to_model_response(d: dict, model_name: str):
    """responses output[] -> litellm ModelResponse (choices[0].message.tool_calls)."""
    from litellm.types.utils import (
        ModelResponse, Choices, Message, Usage,
        ChatCompletionMessageToolCall, Function,
    )
    tool_calls = []
    text = None
    for it in d.get("output", []):
        typ = it.get("type")
        if typ == "function_call":
            tool_calls.append(ChatCompletionMessageToolCall(
                id=it.get("call_id"),
                type="function",
                function=Function(name=it.get("name"), arguments=it.get("arguments") or "{}"),
            ))
        elif typ == "message":
            for c in it.get("content", []):
                if c.get("type") == "output_text":
                    text = (text or "") + c.get("text", "")
    msg = Message(role="assistant", content=text, tool_calls=tool_calls or None)
    finish = "tool_calls" if tool_calls else "stop"
    choice = Choices(index=0, message=msg, finish_reason=finish)
    u = d.get("usage", {}) or {}
    # responses' input_tokens already includes cached_tokens (cached is a subset of input),
    # so prompt_tokens takes input_tokens directly, without adding them up again (otherwise
    # double-counting); the cached portion is attached separately to
    # prompt_tokens_details.cached_tokens for pricing at the $0.5 discount.
    itd = u.get("input_tokens_details", {}) or {}
    cached = itd.get("cached_tokens", 0) or 0
    usage = Usage(
        prompt_tokens=u.get("input_tokens", 0) or 0,
        completion_tokens=u.get("output_tokens", 0) or 0,
        total_tokens=u.get("total_tokens", 0) or 0,
    )
    if cached:
        try:
            from litellm.types.utils import PromptTokensDetailsWrapper
            usage.prompt_tokens_details = PromptTokensDetailsWrapper(cached_tokens=cached)
        except Exception:
            usage.prompt_tokens_details = {"cached_tokens": cached}
    return ModelResponse(
        id=d.get("id", ""), choices=[choice], usage=usage,
        model=model_name, object="chat.completion",
    )


def _query_via_responses(model_name: str, messages: list[dict], model_kwargs: dict):
    """Go through /v1/responses: send the request and translate it back into a ModelResponse. api_base/api_key are taken from model_kwargs."""
    api_base = model_kwargs["api_base"].rstrip("/")
    # api_base looks like http://host:port/v1; the responses endpoint is <base>/responses
    url = api_base + "/responses"
    api_key = model_kwargs["api_key"]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    instructions, items = _messages_to_responses_input(messages)
    body: dict = {
        "model": model_name.split("/")[-1],  # strip the "openai/" prefix
        "input": items,
        "tools": _chat_tools_to_responses(USERSIM_TOOLS),
    }
    if instructions:
        body["instructions"] = instructions
    reasoning = model_kwargs.get("reasoning")
    if reasoning:
        body["reasoning"] = reasoning
    if model_kwargs.get("max_output_tokens"):
        body["max_output_tokens"] = model_kwargs["max_output_tokens"]
    if model_kwargs.get("parallel_tool_calls") is not None:
        body["parallel_tool_calls"] = model_kwargs["parallel_tool_calls"]
    # prompt cache lifetime: the gateway responses currently only supports ttl="30m" (default
    # is about 5min, which task 3's slow turns easily let expire). Injected only when
    # model_kwargs.cache_ttl is explicitly given; otherwise unaffected.
    cache_ttl = model_kwargs.get("cache_ttl")
    if cache_ttl:
        body["prompt_cache_options"] = {"ttl": cache_ttl}

    timeout = model_kwargs.get("request_timeout", 7200)
    r = requests.post(url, headers=headers, json=body, timeout=timeout)
    if r.status_code == 401:
        raise litellm.exceptions.AuthenticationError(
            message=f"/v1/responses 401: {r.text[:300]}",
            model=model_name, llm_provider="openai",
        )
    if r.status_code != 200:
        raise RuntimeError(f"/v1/responses HTTP {r.status_code}: {r.text[:500]}")
    return _responses_json_to_model_response(r.json(), model_name)



# ============================================================================
# Vertex passthrough Anthropic messages bridge (only when model_kwargs.use_anthropic_messages=True)
# ----------------------------------------------------------------------------
# Background: the plain /v1/chat/completions route was measured to have prompt cache
# read constantly at 0 (creation works, read never hits), yielding no cost savings.
# Whereas the Vertex passthrough route
#   <your-gateway>/v1/publishers/anthropic/models/<model>:streamRawPredict
# goes through the native Anthropic messages protocol, where prefix cache read was
# measured to actually hit (probe: turn 2 cache_read=4566, only incremental writes).
# Here we translate chat messages/tools into an Anthropic request and send it out, then
# translate the response back into a litellm ModelResponse (with
# choices[0].message.tool_calls), so the base class's query()/_parse_actions/pricing/
# archiving all need no changes. Every turn resends everything (consistent with harness
# behaviour).
# [Keep parallel] Do not set disable_parallel_tool_use: consistent with the opus5 original
#   litellm branch's behaviour; when the model emits N tool_use in one turn, on feedback the
#   N tool_result are merged into the single user message that immediately follows
#   (required by the Anthropic protocol, otherwise 400). The switch is off by default ->
#   other models go through the original branch, zero impact.
# ============================================================================

def _chat_tools_to_anthropic(tools: list[dict]) -> list[dict]:
    """chat tools ({'type':'function','function':{name,description,parameters}}) ->
    Anthropic top-level {name,description,input_schema} format."""
    out = []
    for t in tools:
        fn = t.get("function", {})
        out.append({
            "name": fn.get("name"),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {}) or {"type": "object", "properties": {}},
        })
    return out


def _messages_to_anthropic(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """chat messages -> (system_blocks, anthropic_messages).
    system folds into a separate system; user->text block; assistant(content+tool_calls)->
    text + tool_use blocks; consecutive tool -> merged into multiple tool_result blocks of one
    user message (the key to keeping parallelism: the message immediately following the
    assistant carries all tool_result together)."""
    system_blocks: list[dict] = []
    out: list[dict] = []
    pending_results: list[dict] = []

    def flush_results():
        if pending_results:
            out.append({"role": "user", "content": list(pending_results)})
            pending_results.clear()

    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            flush_results()
            if isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "text":
                        system_blocks.append({"type": "text", "text": b.get("text", "")})
                    elif isinstance(b, str):
                        system_blocks.append({"type": "text", "text": b})
            elif content:
                system_blocks.append({"type": "text", "text": content})
        elif role == "tool":
            # accumulate into multiple tool_result within one user message (parallel-tool pairing).
            pending_results.append({
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id"),
                "content": content or "",
            })
        elif role == "user":
            flush_results()
            blocks: list[dict] = []
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        blocks.append({"type": "text", "text": part.get("text", "")})
                    elif isinstance(part, str):
                        blocks.append({"type": "text", "text": part})
            elif content:
                blocks.append({"type": "text", "text": content})
            if not blocks:
                blocks.append({"type": "text", "text": content or ""})
            out.append({"role": "user", "content": blocks})
        elif role == "assistant":
            flush_results()
            blocks = []
            if isinstance(content, str) and content.strip():
                blocks.append({"type": "text", "text": content})
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id"),
                    "name": fn.get("name"),
                    "input": args,
                })
            if not blocks:
                blocks.append({"type": "text", "text": content or ""})
            out.append({"role": "assistant", "content": blocks})
        # other internal roles (exit, etc.) are not sent to the model
    flush_results()
    return system_blocks, out


def _attach_anthropic_cache_breakpoint(messages: list[dict], ttl: str = "5m") -> list[dict]:
    """Copy extend verbatim: attach an ephemeral breakpoint to the last cacheable block of messages
    (caches the entire prefix up to the previous turn, so the next turn can hit on read). Modifies in place (messages is freshly built)."""
    cache_control = {"type": "ephemeral", "ttl": ttl}
    for mi in range(len(messages) - 1, -1, -1):
        content = messages[mi].get("content")
        if not isinstance(content, list):
            continue
        for bi in range(len(content) - 1, -1, -1):
            blk = content[bi]
            if not isinstance(blk, dict) or blk.get("type") == "thinking":
                continue
            blk["cache_control"] = dict(cache_control)
            return messages
    return messages


def _anthropic_json_to_model_response(d: dict, model_name: str):
    """Anthropic content[] -> litellm ModelResponse (choices[0].message.tool_calls)."""
    from litellm.types.utils import (
        ModelResponse, Choices, Message, Usage,
        ChatCompletionMessageToolCall, Function,
    )
    tool_calls = []
    text_parts: list[str] = []
    for blk in d.get("content", []) or []:
        typ = blk.get("type")
        if typ == "text":
            text_parts.append(blk.get("text", ""))
        elif typ == "tool_use":
            tool_calls.append(ChatCompletionMessageToolCall(
                id=blk.get("id"),
                type="function",
                function=Function(
                    name=blk.get("name"),
                    arguments=json.dumps(blk.get("input") or {}, ensure_ascii=False),
                ),
            ))
        # thinking block (adaptive) is not returned, ignored
    text = "".join(text_parts) if text_parts else None
    msg = Message(role="assistant", content=text, tool_calls=tool_calls or None)
    finish = "tool_calls" if tool_calls else "stop"
    choice = Choices(index=0, message=msg, finish_reason=finish)
    u = d.get("usage", {}) or {}
    inp = u.get("input_tokens", 0) or 0
    cache_read = u.get("cache_read_input_tokens", 0) or 0
    cache_write = u.get("cache_creation_input_tokens", 0) or 0
    out_tok = u.get("output_tokens", 0) or 0
    usage = Usage(
        prompt_tokens=inp + cache_read + cache_write,
        completion_tokens=out_tok,
        total_tokens=inp + cache_read + cache_write + out_tok,
    )
    # Expose the cache hit amount for downstream stats (failure is non-fatal).
    try:
        from litellm.types.utils import PromptTokensDetailsWrapper
        usage.prompt_tokens_details = PromptTokensDetailsWrapper(cached_tokens=cache_read)
    except Exception:
        try:
            usage.prompt_tokens_details = {"cached_tokens": cache_read}
        except Exception:
            pass
    return ModelResponse(
        id=d.get("id", ""), choices=[choice], usage=usage,
        model=model_name, object="chat.completion",
    )


def _query_via_anthropic_messages(model_name: str, messages: list[dict], model_kwargs: dict):
    """Go through the Vertex passthrough :streamRawPredict (native Anthropic protocol + prefix cache breakpoint)."""
    url = model_kwargs["anthropic_messages_url"]
    api_key = model_kwargs["api_key"]
    # Task-dimension account affinity (cache_task_id): pin all requests of the same cell to the
    # same underlying account -> preserve prefix cache; different cells use different ids ->
    # spread across different healthy accounts, sidestepping a given account's on-demand 400 /
    # easing 429.
    # Appended to the tail of the Bearer query string only when the override layer/caller
    # explicitly gives cache_task_id; absent the key, behaviour is unchanged.
    cache_task_id = model_kwargs.get("cache_task_id")
    if cache_task_id:
        sep = "&" if "?" in api_key else "?"
        api_key = f"{api_key}{sep}cache_task_id={cache_task_id}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    system_blocks, amsgs = _messages_to_anthropic(messages)
    amsgs = _attach_anthropic_cache_breakpoint(amsgs, ttl=model_kwargs.get("cache_ttl", "5m"))
    body: dict = {
        "anthropic_version": model_kwargs.get("anthropic_version", "vertex-2023-10-16"),
        "max_tokens": model_kwargs.get("max_tokens", 32768),
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": model_kwargs.get("reasoning_effort", "xhigh")},
        "tools": _chat_tools_to_anthropic(USERSIM_TOOLS),
        "messages": amsgs,
    }
    if system_blocks:
        body["system"] = system_blocks
    # The aws_third /v1/messages entry needs a platform model marker in the body (litellm uses
    # it to route). Injected only when the override layer explicitly gives model_marker;
    # the polaris/google passthrough channels do not provide this key and behaviour is unchanged.
    model_marker = model_kwargs.get("model_marker")
    if model_marker:
        body["model"] = model_marker

    timeout = model_kwargs.get("request_timeout", 7200)
    r = requests.post(url, headers=headers, json=body, timeout=timeout)
    if r.status_code == 401:
        raise litellm.exceptions.AuthenticationError(
            message=f":streamRawPredict 401: {r.text[:300]}",
            model=model_name, llm_provider="openai",
        )
    if r.status_code != 200:
        raise RuntimeError(f":streamRawPredict HTTP {r.status_code}: {r.text[:500]}")
    return _anthropic_json_to_model_response(r.json(), model_name)


class UserSimLitellmModel(LitellmModel):
    # Set by query() around each _query() call so we can attach precise wall-clock
    # timing of the ASSISTANT model's own inference (excludes bash exec and the
    # user-sim side, which never go through this method).
    _last_timing: dict | None = None

    def _query(self, messages: list[dict[str, str]], **kwargs):
        n_input = _count_input_tokens(self.config.model_name, messages)
        if n_input >= MAX_INPUT_TOKENS:
            # In LitellmModel.abort_exceptions -> aborts the turn without retry
            # and without ever sending the oversized request to the server.
            raise litellm.exceptions.ContextWindowExceededError(
                message=(
                    f"Input context {n_input} tokens >= cap {MAX_INPUT_TOKENS}; "
                    "treating as failure (runaway context from prior attempts)."
                ),
                model=self.config.model_name,
                llm_provider="openai",
            )
        # Precise wall-clock around the pure inference call. perf_counter for the
        # latency (monotonic, not affected by clock jumps); time.time() for the
        # absolute start/end epoch so timelines can be reconstructed & aligned.
        t_start_epoch = time.time()
        t0 = time.perf_counter()
        try:
            merged_kwargs = self.config.model_kwargs | kwargs
            if merged_kwargs.get("use_anthropic_messages"):
                # opus5-specific: go through the Vertex passthrough :streamRawPredict (native
                # Anthropic protocol + prefix cache breakpoint); measured cache read actually
                # hits and saves cost.
                resp = _query_via_anthropic_messages(
                    self.config.model_name, messages, merged_kwargs
                )
            elif merged_kwargs.get("use_responses_api"):
                # gpt-5.6-sol-specific: go through /v1/responses to preserve reasoning + function tools.
                resp = _query_via_responses(
                    self.config.model_name, messages, merged_kwargs
                )
            else:
                resp = litellm.completion(
                    model=self.config.model_name,
                    messages=messages,
                    tools=USERSIM_TOOLS,
                    **merged_kwargs,
                )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e
        finally:
            latency = time.perf_counter() - t0
            t_end_epoch = time.time()
            self._last_timing = {
                "start_epoch": t_start_epoch,
                "end_epoch": t_end_epoch,
                "latency_s": latency,
                "n_input_tokens_est": n_input,
            }
            logger.info(
                "[assistant-timing] model=%s start=%.6f end=%.6f latency_s=%.3f n_input_est=%d",
                self.config.model_name, t_start_epoch, t_end_epoch, latency, n_input,
            )
        return resp

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        self._last_timing = None
        try:
            message = super().query(messages, **kwargs)
        except FormatError as e:
            # Attach timing to the assistant_message the base class stashed on the
            # exception, so even format-error turns carry precise timing.
            am = getattr(e, "assistant_message", None)
            if isinstance(am, dict) and self._last_timing is not None:
                am.setdefault("extra", {})["timing"] = self._last_timing
            raise
        if self._last_timing is not None:
            message.setdefault("extra", {})["timing"] = self._last_timing
        return message

    def _parse_actions(self, response) -> list[dict]:
        tool_calls = response.choices[0].message.tool_calls or []
        return parse_actions_with_reply(
            tool_calls, format_error_template=self.config.format_error_template
        )
