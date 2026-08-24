"""
Tau2/Tau3-style user simulator for coding agents.

Trimmed/adapted from the reference simulator the project provided. Faithful to
the reference design except for the project-specific adaptations called out in
runs/USER_SIM_PLAN.md:

- The simulated user is a stateful participant, not a static prompt.
- The user has hidden task-specific scenario instructions.
- The user can have its own tools, and tool results ground later user messages.
- The orchestrator is half-duplex: agent -> user -> agent, with optional
  user-tool hops hidden from the coding agent.
- The user stops by emitting special tokens: ###STOP###, ###TRANSFER###,
  ###OUT-OF-SCOPE###.
- When calling the user LLM, actual roles are flipped (UserState.flip_roles()).

Project-specific changes vs. reference:
- Dropped ``RunUserCommandTool`` (unused in this branch; live testing goes
  through ``LiveSubtaskTestTool`` in live_subtask_test_tool.py).
- Injected the D2 behavior-dimension table into the global guidelines, as a
  hard behavior spec for the simulated user (see CODING_USER_D2_GUIDELINES).

Dependency:
    pip install litellm
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import textwrap
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional, Protocol, Sequence

STOP = "###STOP###"
TRANSFER = "###TRANSFER###"
OUT_OF_SCOPE = "###OUT-OF-SCOPE###"

# The confidence value the simulated user self-assesses at the end of every "visible" message (absolute value 0-100).
# Parsed by the harness and used to decide whether to stop; it is stripped out before being shown to the coding agent.
CONFIDENCE_RE = re.compile(r"\[\[\s*confidence\s*:\s*(\d{1,3})\s*\]\]", re.IGNORECASE)


def parse_confidence(text: Optional[str]) -> Optional[int]:
    """Extract the integer of the last [[confidence:N]] from a visible user message, clamped to 0-100.

    Returns None when there is no marker (the harness treats this as "the model gave
    no confidence this turn" and reuses the previous turn's value).
    """
    if not text:
        return None
    matches = CONFIDENCE_RE.findall(text)
    if not matches:
        return None
    try:
        val = int(matches[-1])
    except (ValueError, TypeError):
        return None
    return max(0, min(100, val))


def strip_confidence(text: Optional[str]) -> str:
    """Remove all [[confidence:N]] markers, yielding the text actually shown to the coding agent."""
    if not text:
        return ""
    return CONFIDENCE_RE.sub("", text).strip()

Role = Literal["system", "user", "assistant", "tool"]
ToolRequestor = Literal["assistant", "user"]


# -----------------------------------------------------------------------------
# Message and tool-call models
# -----------------------------------------------------------------------------


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")
    requestor: ToolRequestor = "assistant"

    def to_openai_tool_call(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }

    @classmethod
    def from_openai_tool_call(cls, raw: Any, requestor: ToolRequestor) -> "ToolCall":
        # Works with both dict responses and LiteLLM/OpenAI objects.
        if isinstance(raw, dict):
            call_id = raw.get("id") or f"call_{uuid.uuid4().hex[:12]}"
            fn = raw.get("function", {})
            name = fn.get("name")
            arg_s = fn.get("arguments") or "{}"
        else:
            call_id = getattr(raw, "id", None) or f"call_{uuid.uuid4().hex[:12]}"
            fn = getattr(raw, "function")
            name = getattr(fn, "name")
            arg_s = getattr(fn, "arguments", "{}") or "{}"
        try:
            args = json.loads(arg_s) if isinstance(arg_s, str) else dict(arg_s)
        except json.JSONDecodeError:
            args = {"_raw_arguments": arg_s}
        return cls(id=call_id, name=name, arguments=args, requestor=requestor)


@dataclass
class Message:
    role: Role
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    requestor: Optional[ToolRequestor] = None
    error: bool = False
    timestamp: float = field(default_factory=time.time)

    def has_text_content(self) -> bool:
        return self.content is not None and self.content.strip() != ""

    def is_tool_call(self) -> bool:
        return bool(self.tool_calls)

    def validate_single_mode(self) -> None:
        if self.is_tool_call() and self.has_text_content():
            raise UserProtocolError("message has both text content and tool calls")
        if not self.is_tool_call() and not self.has_text_content() and self.role != "tool":
            raise UserProtocolError("message is empty")

    def to_openai(self) -> dict[str, Any]:
        if self.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": self.tool_call_id,
                "name": self.name,
                "content": self.content or "",
            }
        out: dict[str, Any] = {"role": self.role, "content": self.content or ""}
        if self.tool_calls:
            out["tool_calls"] = [tc.to_openai_tool_call() for tc in self.tool_calls]
        return out


class UserProtocolError(Exception):
    pass


# Names of the user-side tools. A well-behaved model triggers these via
# structured tool_call blocks; some sglang deployments instead degrade the call
# to PLAIN TEXT in `content` (e.g. literally "run_subtask_test(1.1)"). Such text
# would otherwise be returned as a visible user message and shipped to the coding
# agent — meaningless to it and a leak of the hidden-verification mechanism.
_USER_TOOL_NAMES = ("run_subtask_test", "run_subtask_tests", "mark_subtask", "read_status")
# Match content that contains a bare function-call to one of those tools:
#   run_subtask_test(1.1) | run_subtask_test(subtask_id="S1") | read_status() | mark_subtask(1.2,"done")
# We do NOT anchor at the start, because the call is frequently wrapped in a tag
# or lead-in the model invented — a Chinese-worded "tool-call" tag
# `<tool-call>read_status()</tool-call>`, `<tool_call>run_subtask_test(1.1)</tool_call>`,
# a fullwidth-bracket `[tool-call] mark_subtask(...)`, or a Chinese lead-in
# `invoke tool: run_subtask_test(1.1)`. Instead we require a left boundary that is NOT
# an identifier char (start-of-string, whitespace, or any punctuation/bracket/CJK
# tag char) so `myrun_subtask_test(` etc. is not caught, then `name` + `(`.
_USER_TOOL_NAMES = ("run_subtask_test", "run_subtask_tests", "mark_subtask", "read_status")
_TEXT_PSEUDO_TOOLCALL_RE = re.compile(
    r"(?:^|[^0-9A-Za-z_])(?:" + "|".join(re.escape(n) for n in _USER_TOOL_NAMES) + r")\s*\(",
    re.IGNORECASE,
)


def _is_text_pseudo_tool_call(content: Optional[str]) -> bool:
    """True if `content` looks like a user-side tool invocation emitted as text.

    Used as a parse-time fallback: when the model degrades a tool_use block to a
    plain-text function call, we treat the turn as invalid (like an empty turn)
    and let the client's retry loop ask the model to emit a real tool_call or a
    natural-language message instead.
    """
    if not content:
        return False
    return _TEXT_PSEUDO_TOOLCALL_RE.search(content) is not None


# Some reasoning models leak their chain-of-thought into `content`, wrapped in
# <think>...</think> (or variants). The ACTUAL user-sim message is only the text
# AFTER the closing think tag; everything up to and including it is reasoning and
# must never be shipped to the coding agent. We strip on the LAST closing tag so
# nested/duplicate think blocks all collapse, and tolerate tag spelling variants
# (<think>, <thinking>, <reasoning>) and CJK brackets. The two escaped
# code points U+601D U+8003 are the Chinese word for "thinking", and U+3010
# U+3011 are the fullwidth brackets; they are kept (as \u escapes) so the
# matcher still recognizes CJK thinking tags written with Chinese glyphs.
_THINK_CLOSE_RE = re.compile(
    "</\\s*(?:think|thinking|reasoning)\\s*>|</\\s*\u601d\u8003\\s*>|\u3010/?\u601d\u8003\u3011",
    re.IGNORECASE,
)


def _strip_leaked_thinking(content: Optional[str]) -> Optional[str]:
    """Return only the visible output that follows a leaked </think> block.

    If no closing think tag is present, `content` is returned unchanged. If a tag
    is present, everything up to and including the LAST closing tag is dropped and
    the remainder is returned (stripped). If nothing follows the tag (the model
    emitted only reasoning), returns None so the caller treats it as an empty turn.
    """
    if not content:
        return content
    matches = list(_THINK_CLOSE_RE.finditer(content))
    if not matches:
        return content
    tail = content[matches[-1].end():].strip()
    return tail or None


class UserState:
    """State held by the user simulator.

    The stored messages are in real conversation roles:
      - agent/coding-assistant messages are AssistantMessage-like: role="assistant"
      - simulated-user messages are role="user"
      - user-tool results are role="tool", requestor="user"

    Before sending history to the user LLM, roles are flipped so the simulator
    can generate as the assistant role (mirrors tau2's UserState.flip_roles()).
    """

    def __init__(self, system_prompt: str, messages: Optional[list[Message]] = None):
        self.system_messages = [Message(role="system", content=system_prompt)]
        self.messages: list[Message] = messages or []

    def flip_roles(self) -> list[Message]:
        flipped: list[Message] = []
        for msg in self.messages:
            if msg.role == "user":
                # Actual simulated user becomes assistant from the user-LLM POV.
                flipped.append(
                    Message(
                        role="assistant",
                        content=msg.content,
                        tool_calls=msg.tool_calls,
                        timestamp=msg.timestamp,
                    )
                )
            elif msg.role == "assistant":
                # Actual coding agent becomes user from the user-LLM POV.
                if msg.is_tool_call():
                    raise UserProtocolError(
                        "agent tool calls should not be sent directly to the user simulator; "
                        "execute them in the coding-agent environment first."
                    )
                flipped.append(Message(role="user", content=msg.content, timestamp=msg.timestamp))
            elif msg.role == "tool":
                if msg.requestor != "user":
                    raise UserProtocolError("only user-tool results belong in user simulator history")
                flipped.append(
                    Message(
                        role="tool",
                        content=msg.content,
                        tool_call_id=msg.tool_call_id,
                        name=msg.name,
                        requestor=msg.requestor,
                        error=msg.error,
                        timestamp=msg.timestamp,
                    )
                )
            else:
                raise UserProtocolError(f"unexpected message role: {msg.role}")
        return flipped

    def llm_messages(self) -> list[dict[str, Any]]:
        return [m.to_openai() for m in self.system_messages + self.flip_roles()]


# -----------------------------------------------------------------------------
# Task/persona schema, aligned with tau2 Task.UserScenario
# -----------------------------------------------------------------------------


@dataclass
class StructuredCodingUserInstructions:
    domain: str = "coding_agent"
    reason_for_call: str = ""
    known_info: Optional[str] = None
    unknown_info: Optional[str] = None
    task_instructions: str = ""

    def __str__(self) -> str:
        lines = [f"Domain: {self.domain}"]
        lines.append("Reason for call:\n" + textwrap.indent(self.reason_for_call, "\t"))
        if self.known_info:
            lines.append("Known info:\n" + textwrap.indent(self.known_info, "\t"))
        if self.unknown_info:
            lines.append("Unknown info:\n" + textwrap.indent(self.unknown_info, "\t"))
        lines.append("Task instructions:\n" + textwrap.indent(self.task_instructions, "\t"))
        return "\n".join(lines)


@dataclass
class UserScenario:
    instructions: StructuredCodingUserInstructions | str
    persona: Optional[str] = None
    # Per-instance values for the prompt-file placeholders ({{PERSONA}},
    # {{NUM_BIG_TASKS}}, {{NUM_SUBTASKS}}, {{HIDDEN_CHECKLIST}}).
    # Populated by trace_runner_v2._build_user_scenario. When non-empty, the
    # simulator fills the loaded call template from these instead of using the
    # legacy str(scenario) rendering.
    template_vars: dict = field(default_factory=dict)

    def __str__(self) -> str:
        parts: list[str] = []
        if self.persona:
            parts.append("Persona:")
            parts.append(textwrap.indent(self.persona, "\t"))
        parts.append("Instructions:")
        parts.append(textwrap.indent(str(self.instructions), "\t"))
        return "\n".join(parts)


class Verbosity(str, Enum):
    STANDARD = "standard"
    MINIMAL = "minimal"


@dataclass
class PersonaConfig:
    verbosity: Verbosity = Verbosity.STANDARD

    def to_guidelines_text(self) -> str:
        if self.verbosity != Verbosity.MINIMAL:
            return ""
        return textwrap.dedent(
            """
            ## MINIMAL VERBOSITY
            You are terse in your responses.
            - When a 1-2 word response is sufficient, respond with only those 1-2 words.
            - When a short phrase is sufficient, respond with the phrase instead of a full sentence.
            - Avoid pleasantries or elaboration unless specifically needed.
            - Still provide necessary technical details, error snippets, or acceptance criteria when the agent asks.
            """
        ).strip()


# -----------------------------------------------------------------------------
# D2 behavior-dimension spec (injected into every system prompt).
# Per runs/USER_SIM_PLAN.md §4.5: this is a hard behavior requirement for the
# simulated user — it must "act" like a real developer along these dimensions.
# It is a behavior constraint, NOT a self-scoring instruction.
# -----------------------------------------------------------------------------


CODING_USER_D2_GUIDELINES = """
# Realism Requirements (D2 behavior dimensions)
You MUST make every message obey these behavioral requirements. They describe how
a real human developer talks to a coding agent. Do not evaluate yourself against
them; just behave this way.

- D2-01 User language naturalness: Talk like a real developer — colloquial, elliptical,
  with back-references and fuzzy phrasing. Never use benchmark/template boilerplate.
- D2-02 Non-oracle input: Give only the goal and boundaries. NEVER leak the
  implementation path (do not hand over "modify private function X + use helper Y +
  this DOM structure"-style answers).
- D2-03 Progressive information disclosure: Disclose requirements across turns. Do NOT dump everything
  in the first turn; add constraints / exceptions / corrections / format demands later.
- D2-04 Context reference and elision: Frequently use long-range references and elision such as
  "this one / the previous one / the left one".
- D2-05 Clarification closure: When you clarify, the clarification must actually change the
  later implementation — not a formal Q&A ritual.
- D2-06 Granularity switching: Switch naturally between goal-level / file-level / line-level.
- D2-07 Preference and standard drift: Across turns you may change style preferences, output
  format, or priorities.
- D2-08 Execution-feedback-driven dialogue: Key turning points are driven by artifacts — stack
  traces, failing tests, coverage, runtime traces (couples with your live test results).
- D2-09 Suggestion adoption and review behavior: Naturally mix accept / reject / verify-before-accept.
- D2-10 Trust calibration: Your trust level drives how deeply you review / how many times
  you re-check (couples with your internal confidence in the agent).
- D2-15 Interaction-rhythm authenticity: Vary turn count and pacing with task complexity; not a
  fixed template.
- D2-17 Collaboration-mode-switching authenticity: When you "know how", use the AI as an accelerator;
  when you "don't know how", use it as an explorer — switch naturally.
""".strip()


# -----------------------------------------------------------------------------
# Tau-style global prompts, adapted to coding-agent simulation
# -----------------------------------------------------------------------------


CODING_USER_GUIDELINES = f"""
# User Simulation Guidelines
You are playing the role of a real user asking a coding agent to modify, debug, or implement software.
Your goal is to simulate realistic coding-agent interactions while following the specific scenario instructions.

## Core Principles
- Generate one message at a time, maintaining natural conversation flow.
- Strictly follow the scenario instructions you have received.
- Never make up or hallucinate information not provided in the scenario instructions.
- Information that is not provided in the scenario instructions should be considered unknown or unavailable.
- Avoid repeating the exact instructions verbatim. Use paraphrasing and natural language to convey the same information.
- Disclose information progressively. Wait for the coding agent to ask for specific information before providing it.
- Do not solve the coding task yourself. You are the user, not the coding agent.
- If the agent proposes a behavior that conflicts with the hidden requirements, correct it naturally and reveal only the necessary missing requirement.

## Coding-Agent Specific Behavior
- Start with a realistic user request, not a full hidden specification dump.
- Give repository paths, stack traces, expected behavior, reproduction steps, or constraints only when the agent asks or when they become necessary.
- If the agent asks for information that is listed as unknown, say you do not know or do not have it.
- If the agent says the task is complete, only agree when all scenario instructions and acceptance criteria are satisfied.
- If the agent asks you to review or run something, respond based only on scenario instructions or tool results, not guesses.

## Task Completion
- Continue the conversation until you are satisfied or you have lost confidence in the agent.
- When the instruction goal is satisfied (or you decide to stop for any in-character reason), generate the '{STOP}' token to end the conversation.
- If you are transferred to another agent/system, generate the '{TRANSFER}' token to indicate the transfer.
- If the scenario does not provide enough information for you to continue, generate the '{OUT_OF_SCOPE}' token to end the conversation.

{CODING_USER_D2_GUIDELINES}

Remember: create realistic, natural conversations while strictly adhering to the provided instructions and maintaining character consistency.
""".strip()


CODING_USER_GUIDELINES_WITH_TOOLS = f"""
# User Simulation Guidelines with Tools
You are playing the role of a real user asking a coding agent to modify, debug, or implement software.
Your goal is to simulate realistic coding-agent interactions while following the specific scenario instructions.
You have some user-side tools to perform actions on your end, such as running hidden tests to verify the agent's work.

## Core Principles
- Generate one message at a time, maintaining natural conversation flow.
- At each turn you can either:
  - Send a message to the coding agent.
  - Make a tool call to perform an action requested by the coding agent or necessary to answer its question / verify its work.
- You cannot do both at the same time.
- Strictly follow the scenario instructions you have received.
- Never make up or hallucinate information not provided in the scenario instructions.
- Information that is not provided in the scenario instructions should be considered unknown or unavailable.
- Never make up the results of tool calls. Ground your later responses in the actual tool results.
- If you made an error in a tool call and get an error message, fix the error and try again when possible.
- All information you provide to the coding agent must be grounded in the scenario instructions or tool results.
- Avoid repeating the exact instructions verbatim. Use paraphrasing and natural language.
- Disclose information progressively. Wait for the coding agent to ask for specific information before providing it.
- Ask clarifying questions if you do not know what action to take.
- Your messages when performing tool calls are not displayed to the coding agent; only messages without tool calls are displayed to the coding agent.

## Verification Procedure (each turn, in this exact order)
- FIRST, re-judge the status of every subtask in your hidden checklist as one of:
  done / pending / to_test, based on what the agent just said.
- THEN, for every subtask you marked to_test, call the `run_subtask_tests`
  tool (one subtask_id at a time) to actually verify it against the hidden tests.
- Ground your next visible message in those real test results.

## Coding-Agent Specific Behavior
- Start with a realistic user request, not a full hidden specification dump.
- Give repository paths, stack traces, expected behavior, reproduction steps, or constraints only when the agent asks or when they become necessary.
- If the agent asks for information that is listed as unknown, say you do not know or do not have it.
- If the agent says the task is complete, only agree once your own tests confirm it.
- If a tool result contradicts the agent's claim, report the contradiction naturally and continue.

## Task Completion
- Continue the conversation until you are satisfied or you have lost confidence in the agent.
- When you decide to stop (because the goal is satisfied, or for any in-character reason such as losing patience), generate the '{STOP}' token to end the conversation.
- If you are transferred to another agent/system, generate the '{TRANSFER}' token to indicate the transfer. Only do this after the agent clearly indicates transfer.
- If the scenario does not provide enough information for you to continue, generate the '{OUT_OF_SCOPE}' token to end the conversation.

{CODING_USER_D2_GUIDELINES}

Remember: create realistic, natural conversations while strictly adhering to the provided instructions and maintaining character consistency.
""".strip()


CONFIDENCE_SELF_REPORT_GUIDELINES = """
# Confidence Self-Report (REQUIRED on every visible message)
After deciding what to say to the coding agent, judge how much confidence you —
acting in character — currently have that the agent will successfully complete the
WHOLE task to your satisfaction. Express it as an ABSOLUTE integer from 0 to 100:

- 100 = fully confident the agent is on track / has finished correctly.
- ~50 = mixed; some progress but real doubts remain.
- 0  = you have lost all confidence and want to give up.

Rules:
- Append the tag `[[confidence:N]]` (N is your 0-100 integer) at the VERY END of
  every visible message you send to the coding agent. Exactly one tag per message.
- Base N on the real evidence so far: live test results (your `run_subtask_tests`
  calls), how many hidden subtasks are actually done, and whether the agent is
  making genuine progress or going in circles. Lower it when turns pass without
  real progress; raise it when tests pass and subtasks get done.
- This is YOUR in-character trust calibration (couples with D2-10). Do NOT explain
  the number or mention the tag in prose — just append it.
- When you also emit a stop token (e.g. ###STOP###), still append the confidence
  tag (typically high if satisfied, low if you gave up).
- Do NOT put a confidence tag on user-tool calls — only on visible messages.
""".strip()


SYSTEM_PROMPT_TEMPLATE = """
{global_user_sim_guidelines}

{confidence_self_report_guidelines}

{runtime_persona_guidelines}

{scenario_instructions}
""".strip()


# -----------------------------------------------------------------------------
# Prompt-file loader (trace_synth/synthesize_trace_workspace.render_prompt style)
#
# The single source of truth for the simulated-user call is the prompt file
# prompts/user_sim_call.txt. We load it, strip loader-meta lines (those starting
# with the "#%" sentinel — section dividers / reference notes, NOT part of the
# prompt; markdown "#"/"##" headers are KEPT), and split it on the
# [SYSTEM]...[/SYSTEM] and [GREETING]...[/GREETING] markers.
# -----------------------------------------------------------------------------

DEFAULT_CALL_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "user_sim_call.txt"

# NOTE: anchor markers at start-of-line (re.MULTILINE). The file header documents
# the block names as "#   [SYSTEM] ... [/SYSTEM]" comment lines; without the ^
# anchor the non-greedy regex would match those doc-comment occurrences first and
# capture garbage (this previously made the GREETING render as "markers." etc.).
_SYSTEM_BLOCK_RE = re.compile(r"^\[SYSTEM\]\s*(.*?)\s*^\[/SYSTEM\]", re.DOTALL | re.MULTILINE)
_GREETING_BLOCK_RE = re.compile(r"^\[GREETING\]\s*(.*?)\s*^\[/GREETING\]", re.DOTALL | re.MULTILINE)
_USER_BLOCK_RE = re.compile(r"^\[USER\]\s*(.*?)\s*^\[/USER\]", re.DOTALL | re.MULTILINE)
_REGEN_BLOCK_RE = re.compile(r"^\[REGEN\]\s*(.*?)\s*^\[/REGEN\]", re.DOTALL | re.MULTILINE)
_FALLBACK_BLOCK_RE = re.compile(r"^\[FALLBACK\]\s*(.*?)\s*^\[/FALLBACK\]", re.DOTALL | re.MULTILINE)
_REFLECT_BLOCK_RE = re.compile(r"^\[REFLECT\]\s*(.*?)\s*^\[/REFLECT\]", re.DOTALL | re.MULTILINE)
_INCLUDE_RE = re.compile(r"\{\{INCLUDE\s+([^}]+?)\}\}")


def _strip_loader_meta(text: str) -> str:
    """Drop lines whose first non-space chars are the "#%" sentinel."""
    kept = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#%")]
    return "\n".join(kept).strip()


@dataclass
class CallTemplate:
    """Parsed user_sim_call.txt. All wording sent to the user-sim model lives in
    the prompt file — there is no hardcoded prompt text in the driver:
      system_template   [SYSTEM]   — the system prompt (placeholders + {{#if_D*}})
      greeting          [GREETING] — the opening assistant line
      user_instruction  [USER]     — per-turn continuation instruction
      regen_instruction [REGEN]    — rubric-feedback rewrite message ({{RUBRIC_FEEDBACK}})
      fallback_text     [FALLBACK] — neutral reply when the tool budget is exhausted
      reflect_instruction [REFLECT] — fixed self-reflection: revise the draft so it
                                      matches the persona ({{DRAFT}} placeholder)
    """

    system_template: str
    greeting: str
    user_instruction: str = ""
    regen_instruction: str = ""
    fallback_text: str = ""
    reflect_instruction: str = ""


def _expand_includes(raw: str, base_dir: Path, _depth: int = 0) -> str:
    """Expand `{{INCLUDE <relative-path>}}` directives in a prompt file.

    The named file is read (relative to base_dir) and its content spliced in
    verbatim, with its own loader-meta (`#%`) lines stripped first so shared
    partials can carry documentation headers. Includes may nest (a partial may
    itself INCLUDE another); recursion is bounded to avoid include cycles.
    Missing include targets raise FileNotFoundError so a typo fails loudly.
    """
    if _depth > 8:
        raise RuntimeError("[user-sim] INCLUDE nesting too deep (possible cycle)")

    def _repl(m: "re.Match[str]") -> str:
        rel = m.group(1).strip()
        inc_path = (base_dir / rel)
        if not inc_path.exists():
            raise FileNotFoundError(f"[user-sim] INCLUDE target not found: {inc_path} (in {base_dir})")
        text = inc_path.read_text(encoding="utf-8")
        text = _strip_loader_meta(text)
        # resolve nested includes relative to the included file's own dir
        return _expand_includes(text, inc_path.resolve().parent, _depth + 1)

    return _INCLUDE_RE.sub(_repl, raw)


def load_call_template(path: "str | Path | None" = None) -> CallTemplate:
    """Read and parse the call prompt file. Raises if a required block is missing.

    [SYSTEM] and [GREETING] are required. [USER]/[REGEN]/[FALLBACK]/[REFLECT] are
    optional for backward compatibility — empty string if absent (the driver
    supplies a minimal built-in default only when a block is missing).

    `{{INCLUDE <relative-path>}}` directives are expanded first (relative to the
    prompt file's directory), so shared partials (e.g. persona_dimensions.txt)
    can be spliced into multiple blocks before block parsing.
    """
    p = Path(path) if path else DEFAULT_CALL_PROMPT_PATH
    raw = p.read_text(encoding="utf-8")
    raw = _expand_includes(raw, p.resolve().parent)
    sys_m = _SYSTEM_BLOCK_RE.search(raw)
    if not sys_m:
        raise ValueError(f"[user-sim] no [SYSTEM]...[/SYSTEM] block in prompt file: {p}")
    greet_m = _GREETING_BLOCK_RE.search(raw)
    if not greet_m:
        raise ValueError(f"[user-sim] no [GREETING]...[/GREETING] block in prompt file: {p}")
    user_m = _USER_BLOCK_RE.search(raw)
    regen_m = _REGEN_BLOCK_RE.search(raw)
    fallback_m = _FALLBACK_BLOCK_RE.search(raw)
    reflect_m = _REFLECT_BLOCK_RE.search(raw)
    return CallTemplate(
        system_template=_strip_loader_meta(sys_m.group(1)),
        greeting=_strip_loader_meta(greet_m.group(1)),
        user_instruction=_strip_loader_meta(user_m.group(1)) if user_m else "",
        regen_instruction=_strip_loader_meta(regen_m.group(1)) if regen_m else "",
        fallback_text=_strip_loader_meta(fallback_m.group(1)) if fallback_m else "",
        reflect_instruction=_strip_loader_meta(reflect_m.group(1)) if reflect_m else "",
    )


# -----------------------------------------------------------------------------
# LLM client
# -----------------------------------------------------------------------------


class LLMClient(Protocol):
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> Message:
        ...


class LiteLLMClient:
    """Small adapter around LiteLLM, matching tau2's use of OpenAI-style tools."""

    def __init__(
        self,
        model: str = "gpt-4.1-2025-04-14",
        temperature: float = 0.0,
        **completion_kwargs: Any,
    ):
        self.model = model
        self.temperature = temperature
        self.completion_kwargs = completion_kwargs

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> Message:
        try:
            from litellm import completion
        except ImportError as e:
            raise RuntimeError("Install litellm first: pip install litellm") from e

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            **self.completion_kwargs,
        }
        if tools:
            kwargs["tools"] = tools
        resp = completion(**kwargs)
        choice = resp.choices[0]
        raw_msg = choice.message

        content = getattr(raw_msg, "content", None)
        raw_tool_calls = getattr(raw_msg, "tool_calls", None) or []
        tool_calls = [ToolCall.from_openai_tool_call(tc, requestor="user") for tc in raw_tool_calls]
        return Message(role="user", content=content, tool_calls=tool_calls or None)


class OpenAIClient:
    """Direct OpenAI-compatible client (no litellm).

    Talks straight to an OpenAI ``/chat/completions`` endpoint (e.g. a vLLM
    server). Same ``chat()`` contract as LiteLLMClient, so
    the simulator is agnostic to which client backs it.

    The model string may carry a ``openai/`` prefix (litellm convention); it is
    stripped here since the OpenAI SDK wants the bare model id.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
        request_timeout: Optional[float] = None,
        seed: Optional[int] = None,
        # Swallow litellm-only kwargs (e.g. drop_params) so callers can pass the
        # same sim_cfg dict to either client without filtering.
        **_ignored: Any,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("Install openai first: pip install openai") from e

        self.model = model.split("/", 1)[1] if model.startswith("openai/") else model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self._client = OpenAI(
            base_url=api_base or os.environ.get("OPENAI_BASE_URL"),
            api_key=api_key or os.environ.get("OPENAI_API_KEY") or "dummy",
            timeout=request_timeout if request_timeout is not None else 1800.0,
            max_retries=2,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> Message:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if tools:
            kwargs["tools"] = tools
        resp = self._client.chat.completions.create(**kwargs)
        raw_msg = resp.choices[0].message

        content = getattr(raw_msg, "content", None)
        raw_tool_calls = getattr(raw_msg, "tool_calls", None) or []
        tool_calls = [ToolCall.from_openai_tool_call(tc, requestor="user") for tc in raw_tool_calls]
        return Message(role="user", content=content, tool_calls=tool_calls or None)


class GptGatewayClient:
    """gpt54 client replicating synthesize_agent_trace_v2's LLMGateway openai path.

    Faithful to ``synthesize_gateway.LLMGateway.chat_once`` for the openai
    backend (OpenAI-compatible chat/completions gateway), with one deliberate
    change: the reference raises on ``tools`` for the openai backend, but the
    simulated user needs tool calls (``run_subtask_tests``), so this client
    *does* forward ``tools``/``tool_choice`` using the standard OpenAI
    tool-calling protocol.

    Wire details copied from the reference:
      - payload uses ``max_completion_tokens`` + ``reasoning_effort`` (not
        ``max_tokens``/``temperature``), matching the azure_openai gpt-5.4 path.
      - system role is mapped to ``developer`` (openai_messages_from_chat).
      - auth header ``Bearer {app_id}:{api_key}?provider=..&model=..&timeout=..``.
      - retry on 429/5xx/overloaded with a fixed backoff.
    """

    # Defaults for a standard OpenAI-compatible chat/completions endpoint.
    # Override per instance via constructor args or the env vars below.
    DEFAULT_MODEL = "gpt-4o"
    DEFAULT_PROVIDER = "openai"
    DEFAULT_URL = os.environ.get("LLM_GATEWAY_URL", "https://api.openai.com/v1/chat/completions")
    DEFAULT_APP_ID = os.environ.get("LLM_APP_ID", "")
    LLM_MAX_TOKENS = 65536
    _RETRYABLE = ("429", "500", "502", "503", "504", "529", "overloaded", "rate")

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,  # accepted but unused (reasoning model)
        app_id: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        usage: Optional[str] = None,
        max_tokens: Optional[int] = None,
        request_timeout: Optional[float] = None,
        reasoning_effort: str = "medium",
        **_ignored: Any,
    ):
        self.model = (model or self.DEFAULT_MODEL).strip()
        self.provider = (provider or self.DEFAULT_PROVIDER).strip()
        self.endpoint_url = (endpoint_url or self.DEFAULT_URL).strip()
        self.app_id = (app_id or os.environ.get("LLM_APP_ID") or self.DEFAULT_APP_ID).strip()
        self.api_key = (api_key or os.environ.get("ANTHROPIC_API_KEY") or "").strip()
        self.usage_mode = None if usage is None else str(usage).strip()
        self.max_tokens = int(max_tokens) if max_tokens else self.LLM_MAX_TOKENS
        self.request_timeout = int(request_timeout) if request_timeout else 3600
        self.reasoning_effort = reasoning_effort

    @staticmethod
    def _openai_messages_from_chat(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Mirror LLMGateway.openai_messages_from_chat (system → developer)."""
        out: list[dict[str, Any]] = []
        for msg in messages:
            role = str(msg.get("role") or "user")
            new = dict(msg)  # preserve tool_calls / tool_call_id / name as-is
            if role == "system":
                new["role"] = "developer"
            elif role not in {"user", "assistant", "developer", "tool"}:
                new["role"] = "user"
            out.append(new)
        return out

    def _is_retryable(self, text: str) -> bool:
        low = text.lower()
        return any(tok in low for tok in self._RETRYABLE)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> Message:
        import requests

        payload: dict[str, Any] = {
            "model": self.model,
            "max_completion_tokens": self.max_tokens,
            "messages": self._openai_messages_from_chat(messages),
            "reasoning_effort": self.reasoning_effort,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        usage_query = f"&usage={self.usage_mode}" if self.usage_mode else ""
        auth_query = (
            f"?provider={self.provider}&model={self.model}{usage_query}&timeout=600"
        )
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.app_id}:{self.api_key}{auth_query}",
        }

        last_error: Optional[Exception] = None
        for retry_idx in range(5):
            try:
                resp = requests.post(
                    self.endpoint_url,
                    headers=headers,
                    json=payload,
                    timeout=self.request_timeout,
                )
                try:
                    body = resp.json()
                except Exception as exc:
                    raise RuntimeError(
                        f"gpt54 invalid json (status={resp.status_code}): {resp.text[:500]}"
                    ) from exc
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"gpt54 http {resp.status_code}: "
                        f"{json.dumps(body, ensure_ascii=False)[:500]}"
                    )
                break
            except Exception as exc:
                last_error = exc
                if retry_idx >= 4 or not self._is_retryable(str(exc)):
                    raise
                time.sleep(60)
        else:  # pragma: no cover
            assert last_error is not None
            raise last_error

        choices = body.get("choices") or []
        raw_msg = (choices[0] or {}).get("message") if choices else {}
        raw_msg = raw_msg or {}

        content = raw_msg.get("content")
        if isinstance(content, list):  # some gateways return content blocks
            parts = [b.get("text", "") for b in content if isinstance(b, dict)]
            content = "\n".join(p for p in parts if p)
        raw_tool_calls = raw_msg.get("tool_calls") or []
        tool_calls = [ToolCall.from_openai_tool_call(tc, requestor="user") for tc in raw_tool_calls]
        return Message(role="user", content=content, tool_calls=tool_calls or None)


class DeepSeekGatewayClient:
    """deepseek-v4-pro client replicating synthesize_gateway's deepseek path.

    Faithful to ``synthesize_gateway.LLMGateway.chat_once`` for the ``deepseek``
    backend (DeepSeek, OpenAI Chat Completions standard protocol over an
    OpenAI-compatible gateway). Same ``chat()`` contract as the other clients so
    the simulator is agnostic to which backs it.

    Wire details copied from the reference (see synthesize_agent_trace_v2.py /
    synthesize_gateway.py):
      - URL is the same chat/completions endpoint as the openai path.
      - model marker is the configured DeepSeek model.
      - payload is standard OpenAI Chat Completions: ``{model, max_tokens,
        messages, stream:false, tools?, tool_choice?}`` (NOT max_completion_tokens
        / reasoning_effort — that is the azure gpt-5.4 path).
      - system role stays ``system`` (deepseek_messages_from_chat keeps it),
        unlike the gpt54 path which maps system→developer.
      - auth header is ``Bearer {app_id}:{api_key}`` with NO query string.
      - retry on 429/5xx/overloaded with a fixed backoff.
    """

    # Defaults for a standard OpenAI-compatible chat/completions endpoint.
    # Override per instance via constructor args or the env vars below.
    DEFAULT_MODEL = "deepseek-chat"
    DEFAULT_PROVIDER = "deepseek"
    DEFAULT_URL = os.environ.get("LLM_GATEWAY_URL", "https://api.openai.com/v1/chat/completions")
    DEFAULT_APP_ID = os.environ.get("LLM_APP_ID", "")
    LLM_MAX_TOKENS = 65536
    _RETRYABLE = ("429", "500", "502", "503", "504", "529", "overloaded", "rate")

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,  # accepted but unused
        app_id: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        usage: Optional[str] = None,
        max_tokens: Optional[int] = None,
        request_timeout: Optional[float] = None,
        **_ignored: Any,
    ):
        self.model = (model or self.DEFAULT_MODEL).strip()
        self.provider = (provider or self.DEFAULT_PROVIDER).strip()
        self.endpoint_url = (endpoint_url or self.DEFAULT_URL).strip()
        self.app_id = (app_id or os.environ.get("LLM_APP_ID") or self.DEFAULT_APP_ID).strip()
        self.api_key = (api_key or os.environ.get("ANTHROPIC_API_KEY") or "").strip()
        self.usage_mode = None if usage is None else str(usage).strip()
        self.max_tokens = int(max_tokens) if max_tokens else self.LLM_MAX_TOKENS
        self.request_timeout = int(request_timeout) if request_timeout else 3600

    @staticmethod
    def _deepseek_messages_from_chat(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Pass standard OpenAI-shape messages through, keeping system as-is.

        The simulator already produces flat OpenAI-style messages (string
        content + optional tool_calls / tool_call_id), so unlike the reference's
        Anthropic-block converter we only need to normalize roles and drop empty
        content. system stays system (deepseek accepts it natively).
        """
        out: list[dict[str, Any]] = []
        for msg in messages:
            role = str(msg.get("role") or "user")
            if role not in {"system", "user", "assistant", "tool", "developer"}:
                role = "user"
            if role == "developer":
                role = "system"
            new = dict(msg)
            new["role"] = role
            out.append(new)
        return out

    def _is_retryable(self, text: str) -> bool:
        low = text.lower()
        return any(tok in low for tok in self._RETRYABLE)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> Message:
        import requests

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self._deepseek_messages_from_chat(messages),
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        # deepseek standard protocol: Bearer APP_ID:API_KEY with NO query string.
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.app_id}:{self.api_key}",
        }

        last_error: Optional[Exception] = None
        for retry_idx in range(5):
            try:
                resp = requests.post(
                    self.endpoint_url,
                    headers=headers,
                    json=payload,
                    timeout=self.request_timeout,
                )
                try:
                    body = resp.json()
                except Exception as exc:
                    raise RuntimeError(
                        f"deepseek invalid json (status={resp.status_code}): {resp.text[:500]}"
                    ) from exc
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"deepseek http {resp.status_code}: "
                        f"{json.dumps(body, ensure_ascii=False)[:500]}"
                    )
                break
            except Exception as exc:
                last_error = exc
                if retry_idx >= 4 or not self._is_retryable(str(exc)):
                    raise
                time.sleep(60)
        else:  # pragma: no cover
            assert last_error is not None
            raise last_error

        choices = body.get("choices") or []
        raw_msg = (choices[0] or {}).get("message") if choices else {}
        raw_msg = raw_msg or {}

        content = raw_msg.get("content")
        if isinstance(content, list):  # some gateways return content blocks
            parts = [b.get("text", "") for b in content if isinstance(b, dict)]
            content = "\n".join(p for p in parts if p)
        raw_tool_calls = raw_msg.get("tool_calls") or []
        tool_calls = [ToolCall.from_openai_tool_call(tc, requestor="user") for tc in raw_tool_calls]
        return Message(role="user", content=content, tool_calls=tool_calls or None)


class SglangGatewayClient:
    """sglang (self-hosted V4Pro) client that *reuses* synthesize_gateway.LLMGateway.

    We do NOT re-implement the sglang wire protocol here. Instead we import the
    project's existing ``LLMGateway`` (trace_synth/synthesize_gateway.py) and drive its
    ``backend="sglang"`` path verbatim — same payload (``thinking=True`` /
    ``reasoning_effort="max"`` / ``priority=100`` / ``max_tokens=65536``), same
    no-auth POST, same prefix-cache self-estimation and usage/cost
    accounting. That keeps a single source of truth for the sglang protocol.

    Two adaptations are needed to fit the simulator's ``chat()->Message`` contract:

      1. ``LLMGateway.chat_once`` is ``async`` and returns a *raw body dict* that,
         for sglang, has already been translated into Anthropic shape
         (``anthropic_body_from_sglang`` → ``{role, content:[{type:text}|
         {type:tool_use,id,name,input}], ...}``). We call it with
         ``asyncio.run(...)`` and parse that Anthropic body back into a ``Message``.

      2. ``LLMGateway.deepseek_messages_from_chat`` (used by the sglang path) expects
         *Anthropic-block* messages, whereas the simulator hands us *flat OpenAI*
         messages (``{role:"tool", tool_call_id,...}`` and assistants with a flat
         ``tool_calls`` list). We convert flat→Anthropic-block before handing them
         to the gateway so tool-call pairing (user-side run_subtask_tests) survives.

    The gateway is constructed once with harmless no-op hooks (``log_step`` /
    ``trim_tool_output``) and an empty ``prompt_usage_types`` — its usage ledger
    runs locally and is never checkpointed (``usage_checkpoint_path=None``).
    """

    DEFAULT_MODEL = "your-user-sim-model"

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,  # accepted; sglang path ignores temperature
        endpoint_url: Optional[str] = None,
        api_base: Optional[str] = None,
        request_timeout: Optional[float] = None,
        **_ignored: Any,
    ):
        import sys as _sys

        # Ensure ../trace_synth (home of synthesize_gateway + agent_trace etc.) is
        # importable. Other modules do this at import time via config.setup_paths();
        # guard here for standalone use of SglangGatewayClient. swebench comes from pip.
        import config as _cfg
        _cfg.setup_paths()
        from synthesize_gateway import LLMGateway  # noqa: E402

        # Accept either a full chat/completions endpoint or a /v1 base; normalize
        # to the full chat/completions URL the gateway POSTs to.
        url = (endpoint_url or "").strip()
        if not url and api_base:
            base = api_base.strip().rstrip("/")
            url = base + ("/chat/completions" if not base.endswith("/chat/completions") else "")
        if not url:
            raise ValueError("SglangGatewayClient requires endpoint_url or api_base")

        bare_model = (model or self.DEFAULT_MODEL).strip()
        if bare_model.startswith("openai/"):
            bare_model = bare_model.split("/", 1)[1]

        self._gateway = LLMGateway(
            app_id=None,
            api_key="",
            provider="sglang",
            model=bare_model,
            usage=None,
            backend="sglang",
            endpoint_url=url,
            anthropic_url=url,
            openai_url=url,
            prompt_usage_types=(),
            log_step=(lambda *a, **k: None),
            trim_tool_output=(lambda text, **k: text),
            request_timeout_sec=int(request_timeout) if request_timeout else 3600,
        )

    @staticmethod
    def _openai_messages_to_anthropic_blocks(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Flat OpenAI messages → Anthropic-block messages the gateway expects.

        - system stays system (string content).
        - assistant: text → [{type:text}]; flat tool_calls → [{type:tool_use,...}].
        - tool result ({role:"tool", tool_call_id, content}) → merged into a user
          message as [{type:tool_result, tool_use_id, content}] (the gateway's
          deepseek_messages_from_chat re-expands tool_result back to {role:tool}).
        """
        out: list[dict[str, Any]] = []
        for msg in messages:
            role = str(msg.get("role") or "user")
            if role == "tool":
                out.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id") or "",
                        "content": msg.get("content") or "",
                    }],
                })
                continue
            tool_calls = msg.get("tool_calls") or []
            if role == "assistant" and tool_calls:
                blocks: list[dict[str, Any]] = []
                text = msg.get("content")
                if isinstance(text, str) and text.strip():
                    blocks.append({"type": "text", "text": text})
                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    raw_args = fn.get("arguments")
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) and raw_args.strip() else (raw_args or {})
                    except Exception:
                        args = {}
                    if not isinstance(args, dict):
                        args = {}
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id") or "",
                        "name": str(fn.get("name") or ""),
                        "input": args,
                    })
                out.append({"role": "assistant", "content": blocks})
                continue
            # Plain system / user / assistant (string content).
            out.append({"role": role, "content": msg.get("content") or ""})
        return out

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> Message:
        import asyncio

        anthropic_msgs = self._openai_messages_to_anthropic_blocks(messages)

        # The simulator enforces single-mode (a user message may carry EITHER text
        # OR tool calls, never both, and never be empty). The user-sim model
        # occasionally emits a tool call wrapped in a reasoning preamble (text +
        # tool_use) or an
        # empty turn. Retry once on empty, then normalize in _parse_body so the
        # contract always holds (drop preamble text when tool calls are present;
        # supply a benign fallback only when truly empty).
        last_msg: Optional[Message] = None
        for _attempt in range(2):
            body = asyncio.run(
                self._gateway.chat_once(
                    anthropic_msgs,
                    tools=tools,  # gateway.deepseek_tool accepts {type:function} schemas
                    usage_stage="USER_SIM",
                )
            )
            last_msg = self._parse_body(body)
            # Accept a real tool_call, or a visible message that is NOT a
            # text-degraded tool call. A bare "run_subtask_test(...)" in content
            # means the model failed to emit a structured tool_use — retry so it
            # produces a real tool_call or natural-language reply instead of
            # shipping the pseudo-call to the coding agent.
            if last_msg.is_tool_call():
                return last_msg
            if last_msg.has_text_content() and not _is_text_pseudo_tool_call(last_msg.content):
                return last_msg
        # Both attempts produced nothing usable — emit a neutral nudge so the
        # conversation can continue instead of crashing the run.
        return Message(
            role="user",
            content="Please continue working on the task; let me know once you have an update.",
            tool_calls=None,
        )

    @staticmethod
    def _parse_body(body: dict[str, Any]) -> Message:
        # sglang body is already Anthropic-shaped: {content:[text|tool_use], ...}.
        content_blocks = body.get("content") or []
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            btype = str(block.get("type") or "")
            if btype == "text":
                t = block.get("text")
                if isinstance(t, str) and t.strip():
                    text_parts.append(t)
            elif btype == "tool_use":
                args = block.get("input") if isinstance(block.get("input"), dict) else {}
                tool_calls.append(ToolCall(
                    name=str(block.get("name") or ""),
                    arguments=args,
                    id=str(block.get("id") or f"call_{uuid.uuid4().hex[:12]}"),
                    requestor="user",
                ))
        if tool_calls:
            # Single-mode: a tool-calling turn must NOT also carry text. Any text is
            # the model's reasoning preamble; drop it and keep only the tool calls.
            return Message(role="user", content=None, tool_calls=tool_calls)
        content = "\n".join(text_parts) if text_parts else None
        # Strip any leaked chain-of-thought: the visible message is only what
        # follows the closing </think> tag. Reasoning-only turns collapse to None,
        # which the chat() retry loop treats as an (invalid) empty turn.
        content = _strip_leaked_thinking(content)
        return Message(role="user", content=content, tool_calls=None)


# -----------------------------------------------------------------------------
# User-side tools for coding-agent simulation
# -----------------------------------------------------------------------------


class UserTool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    def run(self, arguments: dict[str, Any]) -> str:
        pass


class RunHiddenTestsTool(UserTool):
    """Run user-owned validation commands (local subprocess version).

    Retained from the reference for completeness / standalone demos. The live
    coding-agent branch uses LiveSubtaskTestTool (live_subtask_test_tool.py)
    instead, which runs against the agent's actual docker container.
    """

    name = "run_hidden_tests"
    description = "Run one configured hidden validation command by name and return stdout/stderr/exit code."
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of the configured test command to run, e.g. 'default' or 'regression'.",
            }
        },
        "required": ["name"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        repo_root: str | Path,
        commands: dict[str, str | Sequence[str]],
        timeout_sec: int = 120,
        max_output_chars: int = 12000,
        env: Optional[dict[str, str]] = None,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.commands = commands
        self.timeout_sec = timeout_sec
        self.max_output_chars = max_output_chars
        self.env = env or {}

    def run(self, arguments: dict[str, Any]) -> str:
        name = str(arguments.get("name", ""))
        if name not in self.commands:
            return json.dumps(
                {
                    "ok": False,
                    "error": f"unknown hidden test command: {name}",
                    "available_names": sorted(self.commands.keys()),
                },
                ensure_ascii=False,
            )
        cmd = self.commands[name]
        shell = isinstance(cmd, str)
        argv = cmd if shell else list(cmd)
        try:
            proc = subprocess.run(
                argv,
                cwd=self.repo_root,
                shell=shell,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                env={**os.environ, **self.env},
            )
            stdout = (proc.stdout or "")[-self.max_output_chars:]
            stderr = (proc.stderr or "")[-self.max_output_chars:]
            return json.dumps(
                {
                    "ok": proc.returncode == 0,
                    "exit_code": proc.returncode,
                    "stdout_tail": stdout,
                    "stderr_tail": stderr,
                },
                ensure_ascii=False,
            )
        except subprocess.TimeoutExpired as e:
            return json.dumps(
                {
                    "ok": False,
                    "error": "timeout",
                    "timeout_sec": self.timeout_sec,
                    "stdout_tail": (e.stdout or "")[-self.max_output_chars:],
                    "stderr_tail": (e.stderr or "")[-self.max_output_chars:],
                },
                ensure_ascii=False,
            )


# -----------------------------------------------------------------------------
# The tau-style coding user simulator
# -----------------------------------------------------------------------------


class CodingUserSimulator:
    def __init__(
        self,
        llm: LLMClient,
        scenario: UserScenario,
        tools: Optional[list[UserTool]] = None,
        persona_config: Optional[PersonaConfig] = None,
        seed: Optional[int] = None,
        validate_communication: bool = True,
        prompt_path: "str | Path | None" = None,
        cooperativeness_guidelines: "str | None" = None,
    ):
        self.llm = llm
        self.scenario = scenario
        self.tools = tools or []
        self.persona_config = persona_config or PersonaConfig()
        # Optional per-persona behavior guidelines (e.g. cooperative vs
        # uncooperative). When provided, this text REPLACES the persona_config
        # guidelines in the {{RUNTIME_PERSONA_GUIDELINES}} slot. When None, the
        # legacy persona_config.to_guidelines_text() path is used unchanged.
        self.cooperativeness_guidelines = (
            cooperativeness_guidelines.strip()
            if isinstance(cooperativeness_guidelines, str) and cooperativeness_guidelines.strip()
            else None
        )
        self.seed = seed
        self.validate_communication = validate_communication
        self._tool_map = {tool.name: tool for tool in self.tools}
        # Load the single-source-of-truth call template (system prompt + greeting).
        # If the file is missing/unparseable we fall back to the legacy in-code
        # templates so the simulator never hard-fails on a prompt-file issue.
        try:
            self._call_template: Optional[CallTemplate] = load_call_template(prompt_path)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[user-sim][warn] failed to load call template ({exc}); "
                  f"falling back to in-code prompt constants", flush=True)
            self._call_template = None
        if seed is not None:
            random.seed(seed)

    @property
    def opening_greeting(self) -> str:
        """The synthetic agent greeting that opens the conversation.

        Sourced from the [GREETING] block of the call prompt file; falls back to
        a built-in default if the file could not be loaded.
        """
        if self._call_template is not None:
            return self._call_template.greeting
        return "Hi, I'm the assistant who'll help you modify the code. What would you like me to start with?"

    @property
    def global_guidelines(self) -> str:
        return CODING_USER_GUIDELINES_WITH_TOOLS if self.tools else CODING_USER_GUIDELINES

    @property
    def system_prompt(self) -> str:
        # Preferred path: render the loaded call template by substituting the
        # per-instance placeholders (trace_synth/synthesize_trace_workspace style).
        if self._call_template is not None:
            text = self._call_template.system_template
            persona_guidelines = (
                self.cooperativeness_guidelines
                if self.cooperativeness_guidelines is not None
                else self.persona_config.to_guidelines_text()
            )
            sub = {
                "RUNTIME_PERSONA_GUIDELINES": persona_guidelines,
            }
            sub.update({str(k): str(v) for k, v in (self.scenario.template_vars or {}).items()})
            for key, value in sub.items():
                text = text.replace("{{" + key + "}}", value)
            return text.strip()
        # Fallback: legacy in-code assembly (no prompt file available).
        runtime_guidelines = (
            self.cooperativeness_guidelines
            if self.cooperativeness_guidelines is not None
            else self.persona_config.to_guidelines_text()
        )
        return SYSTEM_PROMPT_TEMPLATE.format(
            global_user_sim_guidelines=self.global_guidelines,
            confidence_self_report_guidelines=CONFIDENCE_SELF_REPORT_GUIDELINES,
            runtime_persona_guidelines=runtime_guidelines,
            scenario_instructions=str(self.scenario),
        )

    def get_init_state(self, message_history: Optional[list[Message]] = None) -> UserState:
        message_history = message_history or []
        for msg in message_history:
            if not is_valid_user_history_message(msg):
                raise UserProtocolError(f"invalid user history message: {msg}")
        return UserState(system_prompt=self.system_prompt, messages=list(message_history))

    @staticmethod
    def is_stop(message: Message) -> bool:
        if message.is_tool_call() or not message.content:
            return False
        return STOP in message.content or TRANSFER in message.content or OUT_OF_SCOPE in message.content

    def generate_next_message(self, incoming: Message, state: UserState) -> tuple[Message, UserState]:
        if incoming.role == "assistant":
            incoming.validate_single_mode()
        if incoming.role == "tool" and incoming.requestor != "user":
            raise UserProtocolError("user simulator can only receive user-tool results")

        # Store the agent's visible message or the user-tool result.
        state.messages.append(incoming)

        response = self.llm.chat(
            messages=state.llm_messages(),
            tools=[tool.openai_schema() for tool in self.tools] or None,
        )
        response.role = "user"
        if response.tool_calls:
            for tc in response.tool_calls:
                tc.requestor = "user"
        if self.validate_communication:
            response.validate_single_mode()
        state.messages.append(response)
        return response, state

    def execute_user_tool_calls(self, user_tool_message: Message) -> list[Message]:
        if not user_tool_message.tool_calls:
            return []
        out: list[Message] = []
        for call in user_tool_message.tool_calls:
            tool = self._tool_map.get(call.name)
            if tool is None:
                result = json.dumps(
                    {"ok": False, "error": f"unknown user tool: {call.name}"},
                    ensure_ascii=False,
                )
                out.append(
                    Message(
                        role="tool",
                        content=result,
                        tool_call_id=call.id,
                        name=call.name,
                        requestor="user",
                        error=True,
                    )
                )
                continue
            try:
                result = tool.run(call.arguments)
                error = False
            except Exception as e:  # Keep simulator running; tool failure becomes grounded evidence.
                result = json.dumps({"ok": False, "error": repr(e)}, ensure_ascii=False)
                error = True
            out.append(
                Message(
                    role="tool",
                    content=result,
                    tool_call_id=call.id,
                    name=call.name,
                    requestor="user",
                    error=error,
                )
            )
        return out

    def next_visible_user_message(
        self,
        agent_text: str,
        state: UserState,
        max_user_tool_rounds: int = 16,
    ) -> tuple[Message, UserState]:
        """Convenience adapter for coding-agent harnesses.

        Input: a visible message from the coding agent.
        Output: the next visible user message.

        If the user LLM decides to call user-side tools, this method executes
        those tools and feeds the tool results back to the simulator until it
        produces text. The intermediate user tool-call messages are preserved in
        state but are not returned to the coding agent.

        ``max_user_tool_rounds`` is generous (default 16) because the verification
        procedure may legitimately test many to_test subtasks one at a time
        before emitting a visible message.
        """
        msg = Message(role="assistant", content=agent_text)
        for _ in range(max_user_tool_rounds + 1):
            user_msg, state = self.generate_next_message(msg, state)
            if not user_msg.is_tool_call():
                return user_msg, state
            tool_results = self.execute_user_tool_calls(user_msg)
            if not tool_results:
                raise UserProtocolError("user made a tool call but no tool result was produced")
            if len(tool_results) > 1:
                for tr in tool_results[:-1]:
                    state.messages.append(tr)
                msg = tool_results[-1]
            else:
                msg = tool_results[0]
        raise UserProtocolError("user tool loop exceeded max_user_tool_rounds")


def is_valid_user_history_message(message: Message) -> bool:
    return (
        message.role == "user"
        or (message.role == "assistant" and not message.is_tool_call())
        or (message.role == "tool" and message.requestor == "user")
    )
