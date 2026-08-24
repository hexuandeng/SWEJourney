"""
usersim_driver.py
=================

Standalone user-simulator driver — the user-sim side, fully decoupled from
mini-swe-agent. (The coding *assistant* side stays on mini-swe-agent; this driver
only replaces "how the next user message is generated".)

Message-construction flow (see USER_SIM_STANDALONE_PLAN.md §1):

- One shared CLEAN stream is maintained by the caller (the runner): role ∈
  {system, user, assistant}, clean[0] = system (the rendered user_sim_call.txt),
  then user/assistant alternate. The synthetic GREETING is NOT in the clean
  stream — it is only the first transcript line.

- To generate the next USER message, this driver sends the user-sim model exactly
  TWO messages:
      [ {system: <rendered [SYSTEM]>},
        {user:   <transcript> + "\n\n" + <[USER] continuation instruction>} ]
  where <transcript> renders the running dialogue as plain text:
      Assistant: <GREETING>
      User: <u1>
      Assistant: <a1>
      ...
      Assistant: <latest assistant reply>

- The model may freely call its three tools (run_subtask_test / mark_subtask /
  read_status) any number of times. We loop: while the reply carries tool_use →
  execute the tools, feed results back, ask again; once it returns a tool-free
  message (end_turn) we take its text-after-thinking as the clean user request
  (then strip the [[confidence:N]] tag).

Termination (decided by the runner, surfaced via this driver's state):
  - all subtasks resolved (none "pending"), OR
  - confidence <= floor, OR
  - max_user_turns.
"""

from __future__ import annotations

import copy
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from user_simulator import (  # noqa: E402
    SglangGatewayClient,
    UserScenario,
    load_call_template,
    parse_confidence,
    strip_confidence,
    _expand_includes,
)
from live_subtask_test_tool import (  # noqa: E402
    RunSubtaskTestTool,
    MarkSubtaskTool,
    ReadStatusTool,
)

DEFAULT_PROMPT_EN = _THIS_DIR / "prompts" / "user_sim_call.txt"


def _json_from_text(text: str) -> Any:
    """Lenient JSON extraction: strip code fences / prose, parse the first object.

    Tolerates the verifier wrapping its JSON in ```json ... ``` or adding stray
    text before/after. Raises ValueError if no JSON object can be parsed.
    """
    if not text or not text.strip():
        raise ValueError("empty verifier output")
    s = text.strip()
    # strip ```json ... ``` / ``` ... ``` fences
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", s, re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    # fall back to the outermost {...} span
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(s[start:end + 1])
    raise ValueError("no JSON object found in verifier output")

# Transcript labels.
_LABELS = {
    "en": {"assistant": "Assistant", "user": "User"},
}

# Per-item cap (characters) for tool results on the driver side. When
# run_subtask_test runs the full test suite, a single stdout can reach 300k+
# characters (measured: astropy once produced 313,295 characters); appending it
# verbatim into the driver's tool conversation context would blow the prompt past
# the endpoint limit (ctx-400 -> RuntimeError crashing the process, which only
# happens for the repository_steward persona, the only one that enables
# run_subtask_test). Mirror the assistant-side history_truncate.truncate_msg tool
# rule: keep the first _TOOL_RESULT_MAX_CHARS characters (the head contains the
# PASS/FAIL summary), then append a minimal elision marker ``\n... ...`` at the
# tail. Only the messages copy fed to the model is truncated; the tool_trace
# archive keeps the full content.
_TOOL_RESULT_MAX_CHARS: int = 20_000


def _truncate_tool_result(result: str) -> str:
    """Truncate a tool result per the assistant-side tool rule: keep head + append
    ``\\n... ...`` at the tail.

    Returns unchanged if not over the limit. The elision marker matches
    history_truncate (no character count / explanation, to avoid the model
    echoing it as a text pattern)."""
    if result and len(result) > _TOOL_RESULT_MAX_CHARS:
        return result[:_TOOL_RESULT_MAX_CHARS] + "\n... ..."
    return result


class UserSimDriver:
    """Generates the next clean user message from the running clean stream.

    The driver owns: the user-sim LLM gateway, the three tools + shared status
    dict, the rendered [SYSTEM] prompt + [USER] continuation instruction, the
    GREETING, and the running confidence value.
    """

    def __init__(
        self,
        *,
        scenario: UserScenario,
        run_test_tool: RunSubtaskTestTool,
        mark_tool: MarkSubtaskTool,
        read_tool: ReadStatusTool,
        status: dict[str, str],
        sim_cfg: dict,
        lang: str = "en",
        prompt_path: "str | Path | None" = None,
        max_tool_steps: int = 16,
    ):
        self.scenario = scenario
        all_tools = [run_test_tool, mark_tool, read_tool]
        # Per-persona tool gating: sim_cfg["tools_disabled"] is an explicit list of
        # tool names this persona may NOT use (e.g. a non-validating user must not
        # call run_subtask_test). Disabled tools are removed from BOTH the advertised
        # schema list and the dispatch map, so the model never sees them and any
        # leaked call is reported as unavailable (see _exec_tool).
        disabled = {str(x).strip() for x in (sim_cfg.get("tools_disabled") or []) if str(x).strip()}
        self.disabled_tools = disabled
        self.tools = [t for t in all_tools if t.name not in disabled]
        self._tool_map = {t.name: t for t in self.tools}
        if disabled:
            print(f"[user-sim] tools_disabled for this persona: {sorted(disabled)} "
                  f"-> active tools: {sorted(self._tool_map.keys())}", flush=True)
        self.status = status
        # English-only: the user-sim prompts ship in English (see prompts/).
        self.lang = "en"
        self.labels = _LABELS[self.lang]
        self.max_tool_steps = int(max_tool_steps)

        # Resolve prompt file (explicit override > bundled English default).
        if prompt_path:
            p = Path(prompt_path)
        else:
            p = DEFAULT_PROMPT_EN
        self._template = load_call_template(p)

        # Build the user-sim LLM gateway (reuses LLMGateway sglang via SglangGatewayClient).
        sg_kwargs = {
            k: v for k, v in sim_cfg.items()
            if k in ("endpoint_url", "api_base", "request_timeout")
        }
        self.llm = SglangGatewayClient(model=sim_cfg.get("model_name") or None, **sg_kwargs)

        # Confidence tracking.
        self.confidence_init = int(sim_cfg.get("confidence_init", 50))
        self.confidence = self.confidence_init
        self.confidence_floor = int(sim_cfg.get("confidence_floor", 0))
        # Hard bound: a turn's committed confidence may not move more than
        # ``confidence_max_delta`` from the previous committed value. Anything
        # beyond is clamped to exactly that delta (in the same direction).
        self.confidence_max_delta = int(sim_cfg.get("confidence_max_delta", 10))

        # Render system prompt (fill template_vars + runtime persona).
        self.system_prompt = self._render_system_prompt(sim_cfg)

        # ---- rubric verification (modeled on synthesize_flows.PlanFlow) ----
        # Each generated candidate user message is scored by a rubric verifier
        # (verify_user_msg.txt). If it does not pass, we regenerate with the
        # rubric feedback injected, up to `rubric_max_attempts` total attempts.
        # On exhaustion we fall back to the last (best-effort) candidate.
        self._dimensions = dict(sim_cfg.get("dimensions") or {})
        self._persona_text = str(getattr(self.scenario, "persona", "") or "")
        # ---- fixed persona-reflection round ----
        # After a candidate user message is generated (and BEFORE confidence is
        # committed), run exactly ONE extra LLM turn that tells the model its draft
        # does NOT fit the current persona and asks it to revise accordingly. The
        # wording lives in the prompt txt ([REFLECT] block, {{PERSONA}}/{{DRAFT}}
        # placeholders); a built-in zh/en default is used if the block is absent.
        # Always on unless explicitly disabled via sim_cfg["reflect"]["enabled"]=false.
        _ref = dict(sim_cfg.get("reflect") or {})
        self.reflect_enabled = bool(_ref.get("enabled", True))
        rub = dict(sim_cfg.get("rubric") or {})
        self.rubric_enabled = bool(rub.get("enabled", False))
        self.rubric_max_attempts = int(rub.get("max_attempts", 3))
        self.rubric_min_avg = float(rub.get("min_avg", 4.0))
        self.rubric_floor = int(rub.get("min_dim_floor", 2))  # any dim <= floor fails
        self.verify_max_attempts = int(rub.get("verify_max_attempts", 3))
        # Verifier prompt path (bundled English default; explicit override allowed).
        vp = rub.get("verify_prompt_path")
        if vp:
            self._verify_prompt_path = Path(vp)
        else:
            self._verify_prompt_path = _THIS_DIR / "prompts" / "verify_user_msg.txt"
        # Verifier LLM client: reuse the user-sim gateway unless a separate
        # endpoint is configured under rubric.verify_endpoint_url / verify_model.
        if self.rubric_enabled:
            v_url = rub.get("verify_endpoint_url") or sim_cfg.get("endpoint_url")
            v_model = rub.get("verify_model") or sim_cfg.get("model_name")
            if rub.get("verify_endpoint_url") and rub.get("verify_endpoint_url") != sim_cfg.get("endpoint_url"):
                self.verify_llm = SglangGatewayClient(
                    model=v_model or None, endpoint_url=v_url,
                    request_timeout=sim_cfg.get("request_timeout"),
                )
            else:
                self.verify_llm = self.llm  # share the same gateway
            self._verify_tpl = self._load_verify_prompt(self._verify_prompt_path)
        else:
            self.verify_llm = None
            self._verify_tpl = None

    @staticmethod
    def _load_verify_prompt(path: "str | Path") -> tuple[str, str]:
        """Parse a verifier prompt file into (system, user_template).

        Unlike the call template, the verifier prompt has only [SYSTEM] and
        [USER] blocks (no [GREETING]); {{INPUT_JSON}} stays unfilled here.

        `{{INCLUDE <relative-path>}}` directives are expanded first (relative to
        the prompt file's directory), so the verifier can share the same
        persona-dimensions partial as the call template.
        """
        p = Path(path)
        raw = _expand_includes(p.read_text(encoding="utf-8"), p.resolve().parent)
        sys_m = re.search(r"^\[SYSTEM\]\s*(.*?)\s*^\[/SYSTEM\]", raw, re.DOTALL | re.MULTILINE)
        usr_m = re.search(r"^\[USER\]\s*(.*?)\s*^\[/USER\]", raw, re.DOTALL | re.MULTILINE)
        if not sys_m or not usr_m:
            raise ValueError(f"[user-sim] verify prompt missing [SYSTEM]/[USER]: {path}")

        def _strip_meta(t: str) -> str:
            return "\n".join(ln for ln in t.splitlines() if not ln.lstrip().startswith("#%")).strip()

        return _strip_meta(sys_m.group(1)), _strip_meta(usr_m.group(1))

    # -- prompt rendering ----------------------------------------------------

    def _render_system_prompt(self, sim_cfg: dict) -> str:
        text = self._template.system_template
        # 1) Dimension conditionals: keep only the {{#if_D* <value>}}...{{/if_D*}}
        #    block whose <value> matches the chosen value for that dimension in
        #    sim_cfg["dimensions"]. Unselected blocks (and dimensions with no value)
        #    are dropped, mirroring synthesize_trace_workspace's if-block rendering.
        text = self._render_dimension_conditionals(text, sim_cfg.get("dimensions") or {})
        # 1b) Tool conditionals: keep `{{#if_tool <name>}}...{{/if_tool}}` blocks
        #     only when that tool is active for this persona; drop the block
        #     entirely when the tool is in tools_disabled. This removes the
        #     disabled tool's prompt description (e.g. run_subtask_test) for
        #     personas that may not call it, matching the tool-dict gating in
        #     __init__ (self.tools / self._tool_map).
        text = self._render_tool_conditionals(text, getattr(self, "disabled_tools", set()))
        # 2) Plain placeholder substitution (scenario vars + init confidence).
        #    Persona behaviour now lives entirely in the prompt txt's {{#if_D*}}
        #    dimension blocks; there is no {{PERSONA}} slot and no code-injected
        #    persona-guidelines slot anymore.
        sub = {
            "INIT_CONFIDENCE": str(getattr(self, "confidence_init", 50)),
        }
        sub.update({str(k): str(v) for k, v in (self.scenario.template_vars or {}).items()})
        for key, value in sub.items():
            text = text.replace("{{" + key + "}}", value)
        return text.strip()

    @staticmethod
    def _render_dimension_conditionals(text: str, dimensions: dict) -> str:
        """Render `{{#if_D<n> <value>}}...{{/if_D<n>}}` blocks against the chosen
        per-dimension values.

        ``dimensions`` keys may be either the short tag (``D1``) or the full name
        (``D1_agent_response_preference``); we index by the ``D<n>`` prefix. For
        each dimension tag present in the template, the block whose declared value
        equals the chosen value is kept (body only); every other block for that
        tag is dropped. Dimensions absent from the config drop ALL their blocks.
        """
        # Map short tag (D1..D5) -> chosen value.
        chosen: dict[str, str] = {}
        for k, v in (dimensions or {}).items():
            tag = str(k).split("_", 1)[0].strip()
            if tag:
                chosen[tag] = str(v).strip()

        def make_re(tag: str) -> "re.Pattern[str]":
            return re.compile(
                r"\{\{#if_" + re.escape(tag) + r"\s+([A-Za-z0-9_-]+)\}\}(.*?)\{\{/if_"
                + re.escape(tag) + r"\}\}",
                re.DOTALL,
            )

        # Discover every dimension tag referenced in the template.
        tags = sorted(set(re.findall(r"\{\{#if_(D\d+)\s", text)))
        for tag in tags:
            want = chosen.get(tag)

            def repl(m: "re.Match[str]", _want=want) -> str:
                value = m.group(1).strip()
                body = m.group(2).strip()
                return body if (_want is not None and value == _want) else ""

            text = make_re(tag).sub(repl, text)
        return text

    @staticmethod
    def _render_tool_conditionals(text: str, disabled_tools: set) -> str:
        """Render `{{#if_tool <name>}}...{{/if_tool}}` blocks.

        For each block, the body is kept verbatim when ``<name>`` is NOT in
        ``disabled_tools`` (the tool is active for this persona), and dropped
        entirely otherwise. This is the prompt-side counterpart of the tool-dict
        gating in ``__init__``: a persona whose ``tools_disabled`` lists a tool
        neither sees its callable schema NOR its prompt description.
        """
        disabled = {str(x).strip() for x in (disabled_tools or set())}
        block_re = re.compile(
            r"\{\{#if_tool\s+([A-Za-z0-9_-]+)\}\}(.*?)\{\{/if_tool\}\}",
            re.DOTALL,
        )

        def repl(m: "re.Match[str]") -> str:
            name = m.group(1).strip()
            body = m.group(2)
            return body if name not in disabled else ""

        return block_re.sub(repl, text)

    @property
    def greeting(self) -> str:
        return self._template.greeting

    # -- transcript ----------------------------------------------------------

    def _render_transcript(self, clean: list[dict[str, Any]],
                           truncate_n: int = 0, truncate_chars: int = 500) -> str:
        """Render the clean stream (excluding clean[0]=system) as plain text,
        with the GREETING as the first Assistant line.

        ``truncate_n`` > 0 is the context-overflow fallback (see ``_chat_shrink``):
        the EARLIEST ``truncate_n`` assistant turns (the agent's oldest replies,
        typically the biggest code/diff dumps) are clipped to their first
        ``truncate_chars`` characters. The GREETING is never counted/clipped; only
        the agent's real replies in ``clean`` are. This shrinks ONLY the rendered
        copy fed to the user-sim model — the caller's ``clean`` (main-traj stream)
        is read-only here and stays full-fidelity."""
        lines = [f"{self.labels['assistant']}: {self.greeting}"]
        a_seen = 0
        for m in clean:
            role = m.get("role")
            if role == "system":
                continue
            content = (m.get("content") or "").strip()
            if not content:
                continue
            label = self.labels.get(role)
            if label is None:
                continue
            if role == "assistant":
                a_seen += 1
                if a_seen <= truncate_n and len(content) > truncate_chars:
                    content = content[:truncate_chars] + " …[truncated]"
            lines.append(f"{label}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _is_context_overflow(exc: Exception) -> bool:
        """True iff ``exc`` is the user-sim endpoint's 'input too long' 400.

        Matches the server message shape, e.g.:
          LLM http 400: {... "Input length (750988 tokens) exceeds the maximum
          allowed length (621306 tokens)..."}
        We match on the stable phrasing rather than the (variable) token counts."""
        s = str(exc).lower()
        return ("exceeds the maximum allowed length" in s
                or ("input length" in s and "maximum" in s)
                or "context length" in s)

    def _chat_shrink(self, messages: list[dict[str, Any]],
                     tools: Optional[list[dict[str, Any]]],
                     clean: list[dict[str, Any]]) -> Any:
        """``self.llm.chat`` with context-overflow fault tolerance.

        On a 400 'input too long' from the user-sim endpoint, clip the earliest
        not-yet-clipped assistant turn in the transcript to 500 chars and retry,
        progressively (earliest → later), until the request fits or every
        assistant turn has been clipped. ONLY ``messages[1]`` (the transcript+
        instruction user message) is re-rendered in place; the appended tail
        (drafts / tool results at index ≥2) and the caller's ``clean`` main-traj
        stream are never mutated, so truncation affects this one outbound
        user-sim request only and does not persist to later turns."""
        n_assistant = sum(1 for m in clean if m.get("role") == "assistant")
        truncate_n = 0
        while True:
            try:
                return self.llm.chat(messages=messages, tools=tools)
            except Exception as exc:
                if not self._is_context_overflow(exc) or truncate_n >= n_assistant:
                    raise
                truncate_n += 1
                messages[1] = {
                    "role": "user",
                    "content": self._render_transcript(clean, truncate_n)
                    + "\n\n" + self._cur_instruction,
                }
                print(f"[user-sim][shrink] 400 context overflow; clipped earliest "
                      f"{truncate_n}/{n_assistant} assistant turn(s) to 500 char, "
                      f"retrying", flush=True)

    def _build_messages(self, clean: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Build the per-generation STARTING context: [system, user(transcript+instruction)].

        This is the ONE place a fresh context is created for a single
        ``next_user_message()`` call. Every in-call retry (rubric regeneration,
        forced persona reflection) then APPENDS onto this same list, so the model
        sees its own prior drafts, tool calls and the director feedback as one
        continuous, role-alternating conversation. The context is discarded when
        ``next_user_message()`` returns; the following turn rebuilds from scratch
        (the new transcript already contains the just-accepted user message plus
        the coding assistant's reply).
        """
        transcript = self._render_transcript(clean)
        instruction = self._template.user_instruction or (
            "Above is the conversation so far. You are the User. Continue by writing "
            "ONLY your next single user message to the Assistant."
        )
        # Inject the running confidence so the model fine-tunes from its previous
        # value instead of re-deciding blind. On the first turn self.confidence is
        # still confidence_init (50). Placeholders are authored in the prompt txt:
        #   {{PREV_CONFIDENCE}} -> last accepted turn's confidence (50 on turn 1)
        #   {{INIT_CONFIDENCE}} -> the configured initial confidence (50)
        instruction = (
            instruction
            .replace("{{PREV_CONFIDENCE}}", str(self.confidence))
            .replace("{{INIT_CONFIDENCE}}", str(self.confidence_init))
        )
        user_content = transcript + "\n\n" + instruction
        # Stash the rendered instruction tail + the structured source so the
        # context-overflow fallback (_chat_shrink) can re-render messages[1] with
        # progressively-clipped early assistant turns without rebuilding here.
        self._cur_instruction = instruction
        self._cur_clean = clean
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _regen_instruction_msg(self, rubric_feedback: str) -> dict[str, Any]:
        """The rubric-regeneration director turn, appended onto the running
        generation context.

        The rejected draft is ALREADY the last assistant message in that context
        (``_run_generation_loop`` appends its final draft), so this only adds the
        user-role rewrite instruction — yielding natural
        assistant(draft)->user(regen)->assistant(draft2) alternation. Wording
        lives in the prompt txt ([REGEN] block, {{RUBRIC_FEEDBACK}} placeholder).
        """
        regen_tpl = getattr(self._template, "regen_instruction", "") or (
            "Your previous draft of the next user message (shown above) was "
            "rejected by a quality check. Rewrite it to fix EVERY problem below, "
            "while staying fully in character and never leaking the implementation "
            "path. Output ONLY the improved next user message (with the required "
            "[[confidence:N]] tag at the end):\n{{RUBRIC_FEEDBACK}}"
        )
        regen_content = regen_tpl.replace("{{RUBRIC_FEEDBACK}}", rubric_feedback)
        return {"role": "user", "content": regen_content}

    # -- tool execution ------------------------------------------------------

    def _exec_tool(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self._tool_map.get(name)
        if tool is None:
            if name in getattr(self, "disabled_tools", set()):
                return json.dumps(
                    {"ok": False,
                     "error": f"tool '{name}' is disabled for your persona; do not use it",
                     "available_tools": sorted(self._tool_map.keys())},
                    ensure_ascii=False,
                )
            return json.dumps(
                {"ok": False, "error": f"unknown tool: {name}",
                 "available_tools": sorted(self._tool_map.keys())},
                ensure_ascii=False,
            )
        try:
            return tool.run(arguments or {})
        except Exception as e:  # keep the driver alive; surface as grounded evidence
            return json.dumps({"ok": False, "error": repr(e)}, ensure_ascii=False)

    def _clamp_confidence(self, new_conf: int) -> int:
        """Hard-bound the committed confidence to within ``confidence_max_delta``
        of the previous committed value (``self.confidence``). A model that jumps
        further than the allowed delta is pulled back to exactly +/-delta in the
        same direction, then floored at ``confidence_floor`` and capped at 100.
        """
        prev = int(self.confidence)
        delta = int(self.confidence_max_delta)
        bounded = max(prev - delta, min(prev + delta, int(new_conf)))
        bounded = max(self.confidence_floor, min(100, bounded))
        if bounded != int(new_conf):
            print(f"[user-sim][confidence] clamp {new_conf} -> {bounded} "
                  f"(prev={prev}, max_delta={delta})", flush=True)
        return bounded

    def _ask_missing_confidence(self, messages: list[dict[str, Any]],
                                raw_text: str) -> Optional[int]:
        """Ask one extra internal-only question when the visible user turn omitted
        ``[[confidence:N]]``.

        This must not rewrite the user message sent to the coding assistant. The
        extra LLM call only supplies the confidence value; if it still fails to
        provide one, the caller falls back to ``self.confidence``.
        """
        if not (raw_text or "").strip():
            return None
        ask = (
            "Your previous user message omitted the required [[confidence:N]] tag. "
            "Do not rewrite that user message and do not add any dialogue content; "
            "based only on that message and the current context, output one "
            "confidence tag in exactly this format: [[confidence:N]], where N is "
            "an integer from 0 to 100."
        )
        try:
            repair_messages = (
                copy.deepcopy(messages)
                + [{"role": "assistant", "content": raw_text},
                   {"role": "user", "content": ask}]
            )
            msg = self.llm.chat(messages=repair_messages, tools=None)
            conf = parse_confidence(msg.content or "")
            if conf is None:
                print("[user-sim][confidence] missing after repair prompt; reuse previous",
                      flush=True)
            return conf
        except Exception as e:
            print(f"[user-sim][confidence] repair prompt error: {e!r}; reuse previous",
                  flush=True)
            return None

    # -- main generation -----------------------------------------------------

    def _run_generation_loop(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Run the generate->tool->converge loop, APPENDING onto ``messages``.

        ``messages`` is owned by the caller (``next_user_message``) and threads
        through every in-call retry: the model's tool calls, tool results and its
        final tool-free draft are all appended here, so a subsequent rubric-regen
        or reflection turn sees this draft + its tool evidence as prior context.
        The context is role-alternating by construction (user->assistant->tool->
        assistant->user->...), so no consecutive same-role messages arise.

        Does NOT mutate self.confidence (committed by next_user_message on the
        accepted candidate). Returns:
          { "text", "raw_text", "confidence", "prompt_user", "tool_calls",
            "full_messages" }
        """
        # prompt_user = the first user turn's content (the rendered transcript +
        # instruction); kept for the sidecar/renderer.
        prompt_user = ""
        for m in messages:
            if m.get("role") == "user":
                prompt_user = m.get("content") or ""
                break
        tool_schemas = [t.openai_schema() for t in self.tools]
        tool_trace: list[dict[str, Any]] = []

        for _step in range(self.max_tool_steps + 1):
            msg = self._chat_shrink(messages, tool_schemas, getattr(self, "_cur_clean", []))

            if msg.is_tool_call():
                # Record the assistant tool-call turn (OpenAI-flat shape) so the
                # model sees its own calls + their results on the next iteration.
                openai_tool_calls = []
                tool_result_msgs = []
                for tc in (msg.tool_calls or []):
                    call_id = tc.id or f"call_{uuid.uuid4().hex[:12]}"
                    args = tc.arguments if isinstance(tc.arguments, dict) else {}
                    openai_tool_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(args, ensure_ascii=False)},
                    })
                    result = self._exec_tool(tc.name, args)
                    tool_result_msgs.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": _truncate_tool_result(result),
                    })
                    tool_trace.append({"name": tc.name, "arguments": args, "result": result})
                messages.append({"role": "assistant", "content": None, "tool_calls": openai_tool_calls})
                messages.extend(tool_result_msgs)
                continue

            # Tool-free message → this is the clean user request. Append it onto
            # the running context so any following regen/reflect turn revises THIS
            # exact draft (with its tool evidence) instead of a rebuilt context.
            raw = msg.content or ""
            messages.append({"role": "assistant", "content": raw})
            conf = parse_confidence(raw)
            if conf is None:
                conf = self._ask_missing_confidence(messages, raw)
            clean_text = strip_confidence(raw)
            return {"text": clean_text, "raw_text": raw,
                    "confidence": conf if conf is not None else self.confidence,
                    "prompt_user": prompt_user, "tool_calls": tool_trace,
                    "full_messages": copy.deepcopy(messages)}

        # Exhausted tool budget without a plain message → neutral fallback.
        # Wording lives in the prompt txt ([FALLBACK] block).
        fallback_text = getattr(self._template, "fallback_text", "") or (
            "Let me know once you have an update; I'll verify it then."
        )
        messages.append({"role": "assistant", "content": fallback_text})
        return {
            "text": fallback_text,
            "raw_text": "",
            "confidence": self.confidence,
            "full_messages": copy.deepcopy(messages),
            "prompt_user": prompt_user,
            "tool_calls": tool_trace,
        }

    def _reflect_candidate(self, messages: list[dict[str, Any]],
                           candidate: dict[str, Any]) -> dict[str, Any]:
        """Run ONE fixed self-reflection turn over an already-generated candidate,
        APPENDING onto the SAME ``messages`` context that produced it.

        Tells the model its draft does not fit the current persona and asks it to
        revise. Tools ARE open here (same loop as _run_generation_loop): if the
        persona reflection makes the model decide it should verify before
        speaking, it can call run_subtask_test / mark_subtask / read_status
        silently and then produce a message grounded in the result — instead of
        being forced to narrate "let me check" because tools were unavailable.
        Returns a NEW candidate dict (same shape as _run_generation_loop) with the
        revised text/raw_text/confidence. On any failure, returns the original
        candidate unchanged (never blocks the run).
        """
        draft_raw = candidate.get("raw_text") or candidate.get("text") or ""
        if not draft_raw.strip():
            return candidate
        try:
            # The accepted draft is ALREADY the last assistant message in
            # ``messages`` (appended by _run_generation_loop). We only append the
            # reflect director turn (user role) and keep running on the SAME
            # context, so the model revises that exact draft + its tool evidence.
            reflect_tpl = getattr(self._template, "reflect_instruction", "") or (
                "Your previous draft of the next user message (shown above) does NOT "
                "fit the current user persona. Revise it to match the persona "
                "exactly — its tone, technical level, phrasing habits and degree of "
                "cooperation — while never leaking any implementation path or test "
                "detail. Output ONLY the revised next user message (keep the "
                "[[confidence:N]] tag at the end):\n{{DRAFT}}"
            )
            # The reflect template may include the shared persona-dimensions
            # partial (via {{INCLUDE ...}}, already spliced at load time), which
            # carries {{#if_D*}} conditionals. Render those against THIS persona's
            # dimensions first (keep only the chosen value's block), exactly like
            # the system prompt, then fill the draft/persona placeholders.
            reflect_tpl = self._render_dimension_conditionals(reflect_tpl, self._dimensions)
            reflect_content = (reflect_tpl
                               .replace("{{DRAFT}}", draft_raw)
                               .replace("{{PREV_CONFIDENCE}}", str(self.confidence))
                               .replace("{{INIT_CONFIDENCE}}", str(self.confidence_init))
                               .replace("{{PERSONA}}", self._persona_text))
            messages.append({"role": "user", "content": reflect_content})
            # Carry forward the draft's tool trace, then keep appending any tool
            # calls the reflection turn makes.
            tool_trace: list[dict[str, Any]] = list(candidate.get("tool_calls") or [])
            tool_schemas = [t.openai_schema() for t in self.tools]

            raw = ""
            for _step in range(self.max_tool_steps + 1):
                msg = self._chat_shrink(messages, tool_schemas, getattr(self, "_cur_clean", []))
                if msg.is_tool_call():
                    openai_tool_calls = []
                    tool_result_msgs = []
                    for tc in (msg.tool_calls or []):
                        call_id = tc.id or f"call_{uuid.uuid4().hex[:12]}"
                        args = tc.arguments if isinstance(tc.arguments, dict) else {}
                        openai_tool_calls.append({
                            "id": call_id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(args, ensure_ascii=False)},
                        })
                        result = self._exec_tool(tc.name, args)
                        tool_result_msgs.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": _truncate_tool_result(result),
                        })
                        tool_trace.append({"name": tc.name, "arguments": args, "result": result})
                    messages.append({"role": "assistant", "content": None, "tool_calls": openai_tool_calls})
                    messages.extend(tool_result_msgs)
                    continue
                # Tool-free message → the revised user request.
                raw = msg.content or ""
                break
            if not raw.strip():
                return candidate
            # Append the revised draft onto the shared context too, keeping it
            # role-alternating and ready for any (hypothetical) further append.
            messages.append({"role": "assistant", "content": raw})
            conf = parse_confidence(raw)
            if conf is None:
                conf = self._ask_missing_confidence(messages, raw)
            clean_text = strip_confidence(raw)
            return {
                "text": clean_text,
                "raw_text": raw,
                "confidence": conf if conf is not None else candidate.get("confidence", self.confidence),
                "prompt_user": candidate.get("prompt_user", ""),
                "tool_calls": tool_trace,
                "full_messages": copy.deepcopy(messages),
                "reflected": True,
                "pre_reflect_text": candidate.get("text", ""),
            }
        except Exception as e:  # never block the run on reflection failure
            print(f"[user-sim][reflect] error: {e!r}", flush=True)
            return candidate

    def next_user_message(self, clean: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate the next clean user turn, optionally gated by a rubric verifier.

        Modeled on synthesize_flows.PlanFlow: generate a candidate, score it with
        the rubric verifier, and on failure regenerate with the rubric feedback
        injected — up to `rubric_max_attempts` total. On exhaustion, fall back to
        the last candidate (best-effort). When rubric is disabled, this is a single
        pass through generation (+ optional reflection).

        All in-call work (initial generation, every rubric regeneration, and the
        forced reflection) shares ONE ``messages`` context built here. Each step
        appends onto it, so the model always sees its own prior drafts, the tool
        evidence it gathered, and the director feedback as a single role-
        alternating conversation. The context is discarded on return; the next
        turn rebuilds from the updated transcript.

        Return dict adds rubric bookkeeping:
          { ...candidate fields..., "rubric": <verdict|None>,
            "rubric_attempts": <int>, "rubric_history": [verdict, ...] }
        """
        # The single per-call generation context. Every step below appends onto
        # this exact list (never rebuilds it), so retries accumulate in-context.
        messages = self._build_messages(clean)

        if not self.rubric_enabled or self.verify_llm is None:
            out = self._run_generation_loop(messages)
            if self.reflect_enabled:
                out = self._reflect_candidate(messages, out)
            self.confidence = self._clamp_confidence(int(out["confidence"]))
            out["confidence"] = self.confidence
            out["rubric"] = None
            out["rubric_attempts"] = 1
            out["rubric_history"] = []
            return out

        history: list[dict[str, Any]] = []
        last_out: dict[str, Any] | None = None
        last_verdict: dict[str, Any] | None = None
        for attempt_no in range(1, self.rubric_max_attempts + 1):
            # First attempt runs on the base context; each retry has already had
            # its regen director turn appended at the end of the previous
            # iteration (after the rejected draft, which the loop appended itself).
            out = self._run_generation_loop(messages)
            last_out = out
            try:
                verdict = self._run_rubric(clean, out["text"])
            except Exception as e:  # verifier failure → accept candidate, don't block run
                print(f"[user-sim][rubric] verifier error attempt {attempt_no}: {e!r}", flush=True)
                verdict = None
            last_verdict = verdict
            if verdict is not None:
                history.append(verdict)
            if verdict is None or verdict.get("overall_pass"):
                break
            if attempt_no >= self.rubric_max_attempts:
                break
            feedback = self.format_feedback(verdict)
            # Append the regen director turn onto the SAME context. The rejected
            # draft is already the last assistant message (appended by the
            # generation loop), so the model revises that exact text + its tool
            # evidence — no rebuild, natural assistant->user->assistant alternation.
            messages.append(self._regen_instruction_msg(feedback))
            print(
                f"[user-sim][rubric] FAIL attempt {attempt_no}/{self.rubric_max_attempts} "
                f"avg={verdict.get('avg')} scores={verdict.get('scores')}",
                flush=True,
            )

        assert last_out is not None
        # Fixed persona-reflection round on the rubric-accepted candidate, before
        # committing confidence. Preserve the rubric verdict/history on the result.
        if self.reflect_enabled:
            reflected = self._reflect_candidate(messages, last_out)
            if reflected is not last_out:
                reflected["rubric"] = last_verdict
                reflected["rubric_attempts"] = len(history) if history else 1
                reflected["rubric_history"] = history
                self.confidence = self._clamp_confidence(int(reflected["confidence"]))
                reflected["confidence"] = self.confidence
                return reflected
        self.confidence = self._clamp_confidence(int(last_out["confidence"]))
        last_out["confidence"] = self.confidence
        last_out["rubric"] = last_verdict
        last_out["rubric_attempts"] = len(history) if history else 1
        last_out["rubric_history"] = history
        return last_out

    # -- rubric verification (modeled on PlanFlow) ---------------------------

    # The 7 dimensions of D2 interaction authenticity (one-to-one with the scores schema in verify_user_msg.txt).
    _RUBRIC_DIMENSIONS = (
        "user_language_naturalness",      # D2-01
        "non_oracle_input",               # D2-02
        "disclosure_pacing",              # D2-03
        "context_reference",              # D2-04
        "clarification_closure",          # D2-05
        "interaction_granularity",        # D2-06
        "role_boundary_clarity",          # D2-22
    )

    def _build_verify_messages(self, clean: list[dict[str, Any]], candidate: str) -> list[dict[str, Any]]:
        """Render the verify_user_msg prompt with the input-JSON payload filled.

        Persona/dimensions are NOT passed as JSON keys; instead the verifier
        prompt's own `{{#if_D*}}` blocks are rendered against this user's
        configured dimension values (if/else style), so only the actually-chosen
        persona axes survive in the prompt. The full conversation history is
        passed (no truncation) so the verifier can judge multi-turn references,
        disclosure pacing, clarification closure, and feedback-driven turns.
        """
        system, user_tpl = self._verify_tpl
        # Full transcript as plain lines (GREETING first, like the gen-side render),
        # newest last; no [-N:] truncation.
        lines: list[str] = [f"{self.labels['assistant']}: {self.greeting}"]
        for m in clean:
            if m.get("role") == "system":
                continue
            content = (m.get("content") or "").strip()
            if not content:
                continue
            label = self.labels.get(m.get("role"))
            if label is None:
                continue
            lines.append(f"{label}: {content}")
        transcript = "\n".join(lines)
        counts: dict[str, int] = {}
        for v in self.status.values():
            counts[v] = counts.get(v, 0) + 1
        payload = {
            "subtask_status": {
                "subtasks": dict(self.status),
                "counts": counts,
            },
            "transcript": transcript,
            "candidate_message": candidate,
        }
        # Render the persona if/else blocks in BOTH system and user templates
        # against this user's configured dimensions, then fill INPUT_JSON.
        system = self._render_dimension_conditionals(system or "", self._dimensions)
        user = self._render_dimension_conditionals(user_tpl, self._dimensions)
        user = user.replace("{{INPUT_JSON}}", json.dumps(payload, ensure_ascii=False, indent=2))
        msgs: list[dict[str, Any]] = []
        if system.strip():
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        return msgs

    def _run_rubric(self, clean: list[dict[str, Any]], candidate: str) -> dict[str, Any]:
        """Score one candidate message; multi-attempt on malformed verifier JSON."""
        messages = self._build_verify_messages(clean, candidate)
        last_err: Exception | None = None
        for attempt_no in range(1, self.verify_max_attempts + 1):
            if attempt_no > 1 and last_err is not None:
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous output was not valid JSON or failed validation: "
                        f"{type(last_err).__name__}: {last_err}. Return ONLY a JSON object "
                        "matching the schema; no markdown, no code block, no extra text."
                    ),
                })
            msg = self.verify_llm.chat(messages=messages, tools=None)
            raw = msg.content or ""
            try:
                obj = _json_from_text(raw)
                verdict = self.validate_verdict(obj)
                return self._apply_pass_rule(verdict)
            except Exception as e:
                last_err = e
                messages.append({"role": "assistant", "content": raw})
        raise RuntimeError(f"rubric verifier failed: {type(last_err).__name__}: {last_err}")

    @classmethod
    def validate_verdict(cls, verdict_obj: Any) -> dict[str, Any]:
        """Validate the rubric verdict shape; host computes overall_pass.

        overall_pass == (avg >= min_avg) and (min(scores) > floor). Since the
        class doesn't know the instance thresholds here, we attach raw scores/avg
        and let the caller's instance thresholds decide."""
        if not isinstance(verdict_obj, dict):
            raise ValueError("verify output must be a JSON object")
        raw_scores = verdict_obj.get("scores")
        if not isinstance(raw_scores, dict):
            raise ValueError("verify scores must be a JSON object with the seven dimensions")
        scores: dict[str, int] = {}
        for dim in cls._RUBRIC_DIMENSIONS:
            val = raw_scores.get(dim)
            if isinstance(val, bool) or not isinstance(val, int):
                raise ValueError(f"verify scores.{dim} must be an integer in [1,5]")
            if val < 1 or val > 5:
                raise ValueError(f"verify scores.{dim} must be in [1,5], got {val}")
            scores[dim] = val
        avg = sum(scores.values()) / len(cls._RUBRIC_DIMENSIONS)
        issues = [str(x) for x in (verdict_obj.get("issues") or [])
                  if isinstance(x, str) and x.strip()]
        low_dims = [d for d, s in scores.items() if s < 4]
        if low_dims and not issues:
            raise ValueError(
                "verify issues must be non-empty because some dimension scored < 4: "
                + ", ".join(low_dims))
        issues_blob = "\n".join(issues)
        missing = [d for d in low_dims if d not in issues_blob]
        if missing:
            raise ValueError(
                "verify issues must name every dimension scoring < 4; missing: "
                + ", ".join(missing))
        return {
            "scores": scores,
            "avg": round(avg, 4),
            "issues": issues,
            "diagnostic_feedback_for_retry": str(verdict_obj.get("diagnostic_feedback_for_retry") or ""),
        }

    def _apply_pass_rule(self, verdict: dict[str, Any]) -> dict[str, Any]:
        """Stamp overall_pass onto a validated verdict using instance thresholds."""
        scores = verdict.get("scores") or {}
        avg = verdict.get("avg", 0.0)
        verdict["min_avg"] = self.rubric_min_avg
        verdict["min_dim_floor"] = self.rubric_floor
        overall = bool(scores) and (avg >= self.rubric_min_avg) and (min(scores.values()) > self.rubric_floor)
        verdict["overall_pass"] = overall
        return verdict

    @staticmethod
    def format_feedback(verdict: dict[str, Any]) -> str:
        """Render a rubric failure into text fed back to the next regeneration."""
        lines: list[str] = []
        scores = verdict.get("scores") or {}
        if scores:
            lines.append("[dimension scores]")
            lines.extend(f"- {d}: {s}" for d, s in scores.items())
            if verdict.get("avg") is not None:
                lines.append(
                    f"- average: {verdict.get('avg')} "
                    f"(need >= {verdict.get('min_avg', 4.0)} and no dimension <= "
                    f"{verdict.get('min_dim_floor', 1)})"
                )
        issues = verdict.get("issues") or []
        if issues:
            lines.append("[issues — fix every one of these in your next message]")
            lines.extend(f"- {it}" for it in issues)
        diag = str(verdict.get("diagnostic_feedback_for_retry") or "").strip()
        if diag:
            lines.append("[diagnostic feedback]")
            lines.append(diag)
        return "\n".join(lines).strip()

    # -- termination ---------------------------------------------------------

    def all_resolved(self) -> bool:
        """True once no subtask is left pending (all done or failed)."""
        return all(v != "pending" for v in self.status.values()) if self.status else False

    def confidence_exhausted(self) -> bool:
        return self.confidence <= self.confidence_floor
