from __future__ import annotations

import json

import agent_trace as at
from dataclasses import dataclass
from typing import Any

from synthesize_flow_utils import BaseFlow, GeneratedJSONParseError, json_from_text
from inspect import isawaitable
from synthesize_augment_utils import AugmentSupport



@dataclass
class ToolLoopConfig:
    """Configuration for the generation loop with docker_cli."""

    enable_docker_cli: bool = False
    max_tool_calls: int = 0
    remind_every_tool_calls: int = 0
    reminder_context: str = ""
    # Overrides used only for the "first deepseek round": trigger point + wrap-up reminder (does not expose remaining count).
    first_round_max_tool_calls: int = 0
    first_round_remind_start_tool_calls: int = 0
    first_round_remind_every_tool_calls: int = 0


@dataclass
class SchemaOps:
    """Entry points for PLAN / ATTEMPT schema coercion and validation."""

    coerce_event: Any
    validate_event: Any
    is_request_instruction_subtask: Any = None



class CodeValidationFailed(RuntimeError):
    """Indicates that ATTEMPT finished local code execution but code validation did not pass.

    Raised inside `quality_validate`, caught by `BaseFlow.run()` as one functional retry:
    the next `generate(...)` regenerates patch+f2p and appends a new ATTEMPT;
    the current failed event stays in the trace as failure history for the next round's prompt.
    """

    def __init__(self, *, result: dict[str, Any], message: str):
        super().__init__(message)
        self.result = result


class RubricFailed(RuntimeError):
    """Indicates that ATTEMPT code execution passed, but the rubric verifier's semantic check did not.

    Also caught by `BaseFlow.run()` as one functional retry, triggering regeneration of patch+f2p.
    """

    def __init__(self, *, verdict: dict[str, Any], message: str = ""):
        super().__init__(message or "ATTEMPT rubric verification failed")
        self.verdict = verdict


_INFRA_ERROR_MARKERS = (
    "llm http",          # upstream HTTP error wrapped by the gateway (e.g. "LLM http 400: ...")
    "http 400", "http 408", "http 409", "http 425",
    "http 429",          # rate limiting
    "http 500", "http 502", "http 503", "http 504",
    # Residual fallback: if some path still raises the old-style "invalid json (status=5xx)" (a proxy 504 HTML error page),
    # also treat it as a gateway infra error and re-run, rather than mistaking it for a generation-quality issue handed to json_repair / burning an ATTEMPT slot.
    "invalid json (status=502", "invalid json (status=503", "invalid json (status=504",
    "timeout", "timed out",
    "connection", "connect error", "connection reset", "connection aborted",
    "temporarily unavailable", "service unavailable", "bad gateway",
    "must be followed by tool messages",  # deepseek/openai protocol-state error (not a generation-quality issue)
)


def _is_infra_error(exc: BaseException | None) -> bool:
    """Conservatively decide whether an exception is a "transport/gateway infra error" rather than a generation-quality issue.

    Purpose: infra errors (deepseek 400 protocol state, rate limiting, network timeouts, etc.) should not be treated
    as JSON format errors handed to openai for tool-free repair, nor should they burn a real ATTEMPT retry slot. On a hit,
    it should "re-run tool-enabled generation" without incrementing attempt_no.

    The decision is conservative: return True only when the exception type is RuntimeError/OSError/timeout/connection-related
    and its message text hits an explicit transport/gateway marker word; ordinary ValueError (schema failure),
    GeneratedJSONParseError, CodeValidationFailed, and RubricFailed are never treated as infra.
    """
    if exc is None:
        return False
    # Explicitly exclude generation-quality exceptions (even if their message happens to contain words like "timeout", it is not infra).
    if isinstance(exc, (GeneratedJSONParseError, CodeValidationFailed, RubricFailed)):
        return False
    # Explicit transport-layer exception types (no need to inspect the message).
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    msg = str(exc).lower()
    return any(marker in msg for marker in _INFRA_ERROR_MARKERS)


class AugmentFiltered(RuntimeError):
    """Indicates that augment candidates had no survivor after docker semantic filtering.

    Caught by `BaseFlow.run()` as one retry: the next augment `generate(...)` reads the
    failure attribution in `.debug` and regenerates the candidate own_p2p / test_patch.
    """

    def __init__(self, *, debug: dict[str, Any], message: str = ""):
        super().__init__(message or "augment produced no survivor after docker filter")
        self.debug = debug


class PlanNeedsRefinement(RuntimeError):
    """PLAN passed schema validation, but triggers a difficulty-strengthening retry.

    Raised only when `refine_plan=True` and the current round is the first pass, caught by `BaseFlow.run()`:
    the next `generate(...)` feeds the draft to the model for difficulty strengthening.
    `_draft_plan` is stored on the PlanFlow instance and returned as a fallback if all strengthening rounds fail.
    """


class PlanFlow(BaseFlow):
    """Generate one PLAN event, and retry only when the PLAN's JSON/schema validation fails.

    Prompt organization:
    - The first round directly uses the already-rendered PLAN prompt pair `(system, user)`.
    - This prompt is built from the full visible history; for a repair, it additionally carries
      a `failure_brief` tail. Thus PLAN sees the global history across BigTasks, rather
      than a local slice of a single BigTask.
    - The first round is allowed to use `docker_cli`.
    - Retries do not re-organize the prompt from scratch; instead they attach the previous bad JSON as an
      assistant message, then append a host-written repair note that clearly tells the model which
      validation failed.

    Validation contract:
    - The returned object must be coercible into the standard PLAN schema.
    - The host validates that the fixed decision fields (`reason`, `big_task_id`, `round`)
      are not tampered with by the model, and continues to enforce the full schema
      constraints of `validate_plan_event(...)`.
    - A validation failure here only means "the PLAN's JSON/schema is invalid"; it does not involve code,
      test, or rubric verification.
    """

    # User-message template for the second strengthening round; {{DRAFT_PLAN_JSON}} is replaced at runtime with the draft JSON string.
    _REFINE_TEMPLATE: str = (
        "================================================================\n"
        "[Second strengthening] The PLAN draft you just produced passed schema validation; now perform difficulty strengthening.\n"
        "================================================================\n"
        "Before outputting the final strengthened PLAN, you may still further explore the code via docker_cli "
        "to confirm the feasibility of the strengthening direction; tool usage is the same as before, up to 20 more calls.\n\n"
        "Below is the draft you just generated:\n\n"
        "{{DRAFT_PLAN_JSON}}\n\n"
        "Your task is to **rewrite a significantly harder version on top of this**, with these requirements:\n\n"
        "1. **Keep big_task_id / reason / round unchanged** (byte-equal).\n"
        "2. **Difficulty must take a qualitative leap** -- not just tweaking the description wording, but a substantive upgrade along at least one of the following dimensions:\n"
        "   - raise requirement_depth by one or more levels (e.g. behavior_adjustment -> feature_addition, "
        "or feature_addition -> refactor_or_migration / release_or_roadmap);\n"
        "   - expand change_scope (e.g. single_file -> multi_file -> cross_component);\n"
        "   - clearly raise the abstraction level of constraint complexity or goal combination.\n"
        "3. **The main direction must be sharply distinct from the historical BigTasks** -- if the draft already achieves directional differentiation, "
        "deepen it further on that basis; if the draft's directional differentiation is unclear, it must be corrected.\n"
        "4. uplift_proof must be updated in sync to reflect the actual improvement after strengthening.\n"
        "5. subtasks_blueprint must be adjusted accordingly -- after the difficulty increase, subtasks should be more complex with a wider impact.\n"
        "6. The output format and schema requirements remain exactly the same; output only the strengthened PLAN JSON, with no explanation."
    )

    @staticmethod
    def retry_user_message(error: Exception, bad_json: dict, *, attempt_no: int, max_attempts: int) -> str:
        """Build the error note fed back to the model on a PLAN retry."""
        return (
            "The previous PLAN output was not valid JSON, or did not pass the host's _validate_plan_event(plan_obj) check; you need to fix it directly.\n"
            f"This is PLAN generation attempt {attempt_no}/{max_attempts}; you may still call docker_cli this round to further explore the code and confirm the fix direction (same usage as the first round, up to 20 more calls).\n"
            f"Validation error reason: {type(error).__name__}: {error}\n"
            "The previous JSON was:\n"
            f"{json.dumps(bad_json, ensure_ascii=False, indent=2)}\n\n"
            "After completing any necessary exploration, return only one complete, corrected PLAN JSON object; no markdown, no explanation, no code block, no extra text before or after."
        )

    def __init__(
        self, *, client, prompt: tuple[str | None, str], trace, plan_decision, schema_ops: SchemaOps,
        repair_client=None, tool_loop: ToolLoopConfig | None = None, max_attempts: int = 3, json_repair=None,
        output_path=None, write_json=None, refine_plan: bool = False,
        verify_client=None, verify_max_attempts: int = 2, max_rubric_failures: int = 2,
        workspace=None, first_call_client=None,
    ):
        """Save PlanFlow dependencies and initialize the first-round message context.

        `output_path` / `write_json` are optional; when provided, a schema-validation failure also persists a disabled
        PLAN event to disk, aiding auditing and test_failure_details aggregation.

        When `refine_plan=True`, once the first round passes schema, one difficulty-strengthening retry is triggered (except
        for repair); if all strengthening fails, the first-round draft is returned as a fallback.

        rubric verification dependencies (same names and meanings as AttemptFlow):
        - `verify_client`: the LLM client that runs the rubric verifier; when None, rubric is skipped (dry-run / not configured).
        - `workspace`: builds the verify_plan prompt.
        - `verify_max_attempts`: the multi-round cap for the verifier's JSON format validation.
        - `max_rubric_failures`: used only for log counting; the real retry cap is controlled by `max_attempts` (same as attempt).
        """
        super().__init__(max_attempts=max_attempts, json_repair=json_repair)
        self.client = client
        # The client used for the first main generation (attempt_no==1); falls back to the main client when None.
        self.first_call_client = first_call_client or client
        self.trace = trace
        self.plan_decision = plan_decision
        self.schema_ops = schema_ops
        self.tool_loop = tool_loop or ToolLoopConfig()
        self.repair_client = repair_client
        self.output_path = output_path
        self.write_json = write_json
        self.refine_plan = refine_plan
        self.verify_client = verify_client
        self.verify_max_attempts = verify_max_attempts
        self.max_rubric_failures = max_rubric_failures
        self._rubric_failure_count = 0
        # rubric runs unconditionally: it also runs when schema fails; the feedback is stashed here and appended by the
        # schema-retry branch when feeding back to the model, ensuring the feedback always reaches the next fix step.
        self._last_rubric_feedback: str = ""
        self.workspace = workspace
        self._draft_plan: dict | None = None
        system, user = prompt
        self.messages: list[dict[str, Any]] = []
        if system is not None:
            self.messages.append({"role": "system", "content": system})
        self.messages.append({"role": "user", "content": user})
        # Save the initial messages to rebuild the conversation context for refinement rounds.
        self._initial_messages: list[dict[str, Any]] = [dict(m) for m in self.messages]

    async def generate(self, attempt_no: int, last_obj: Any, last_error: Exception | None) -> Any:
        """Run one PLAN prompt.

        First round:
        - Takes the main generation path, deciding whether to allow docker tool calls based on configuration.

        Refinement round (triggered by first-round schema pass + refine_plan=True):
        - Builds on the initial message context, appending the first-round draft (assistant) and the strengthening instruction (user).
        - Opens docker_cli, up to 20 calls, so the model can further explore code feasibility.

        Schema-error retry round:
        - Feeds the previous invalid JSON and the host-written error note back to the model.
        - If a repair client is configured, the retry uses the repair client; otherwise it reuses the main client.
        """
        if attempt_no == 1:
            # PLAN's first round directly uses the main client (opus), not first_call_client (sglang):
            # PLAN blueprint quality has a large global impact, so the strong model is used from the first round to avoid the sglang blueprint being repeatedly refined/retried.
            print(f"[synthesize] step plan-generate: starting  phase=original attempt={attempt_no}")
            result = await self.client.call_json_with_docker_cli_messages(
                self.messages,
                trace=self.trace,
                enable_docker_cli=self.tool_loop.enable_docker_cli,
                event_kind="PLAN",
                max_tool_calls=self.tool_loop.first_round_max_tool_calls or self.tool_loop.max_tool_calls,
                remind_every_tool_calls=self.tool_loop.first_round_remind_every_tool_calls or self.tool_loop.remind_every_tool_calls,
                remind_start_tool_calls=self.tool_loop.first_round_remind_start_tool_calls,
                wrap_up_reminder=bool(self.tool_loop.first_round_remind_start_tool_calls),
            )
            print(f"[synthesize] step plan-generate: done  phase=original attempt={attempt_no}")
            return result
        assert last_obj is not None and last_error is not None
        # Refinement round: the first-round draft passed schema, triggering difficulty strengthening.
        if self._draft_plan is not None and isinstance(last_error, PlanNeedsRefinement):
            draft_json = json.dumps(self._draft_plan, ensure_ascii=False, indent=2)
            refine_content = self._REFINE_TEMPLATE.replace("{{DRAFT_PLAN_JSON}}", draft_json)
            self.messages = list(self._initial_messages) + [
                {"role": "assistant", "content": json.dumps(self._draft_plan, ensure_ascii=False)},
                {"role": "user", "content": refine_content},
            ]
            print(f"[synthesize] step plan-generate: starting  phase=refinement attempt={attempt_no}")
            result = await self.client.call_json_with_docker_cli_messages(
                self.messages,
                trace=self.trace,
                enable_docker_cli=True,
                event_kind="PLAN",
                max_tool_calls=20,
                remind_every_tool_calls=self.tool_loop.remind_every_tool_calls,
            )
            print(f"[synthesize] step plan-generate: done  phase=refinement attempt={attempt_no}")
            return result
        # rubric-failure retry round: at this point the PLAN event is not yet in the trace (it is appended only after the flow returns),
        # so no build_plan_prompt rebuild is needed; reuse the message thread, just appending the previous plan + feedback.
        if isinstance(last_error, RubricFailed):
            self.messages.append({"role": "assistant", "content": json.dumps(last_obj, ensure_ascii=False)})
            self.messages.append({
                "role": "user",
                "content": self.rubric_retry_user_message(last_error.verdict),
            })
            print(f"[synthesize] step plan-generate: starting  phase=rubric-retry attempt={attempt_no}")
            result = await self.client.call_json_with_docker_cli_messages(
                self.messages,
                trace=self.trace,
                enable_docker_cli=True,
                event_kind="PLAN",
                max_tool_calls=20,
                remind_every_tool_calls=self.tool_loop.remind_every_tool_calls,
            )
            print(f"[synthesize] step plan-generate: done  phase=rubric-retry attempt={attempt_no}")
            return result
        # schema-error retry round.
        self.messages.append({"role": "assistant", "content": json.dumps(last_obj, ensure_ascii=False)})
        retry_content = self.retry_user_message(
            last_error, last_obj, attempt_no=attempt_no, max_attempts=self.max_attempts,
        )
        # rubric runs unconditionally: if the previous schema failure still ran rubric, feed that feedback back too,
        # ensuring rubric feedback always reaches the next fix step (per user requirement).
        if self._last_rubric_feedback:
            retry_content += (
                "\n\nAdditionally, although the previous round did not pass schema, the host still ran the task-authenticity rubric on it; "
                "please also improve on the following feedback when rewriting (threshold: the mean across the 6 dimensions must be >= 4.0 and no single dimension == 1):\n"
                f"{self._last_rubric_feedback}"
            )
            self._last_rubric_feedback = ""
        self.messages.append({"role": "user", "content": retry_content})
        print(f"[synthesize] step plan-generate: starting  phase=schema-retry attempt={attempt_no}")
        result = await self.client.call_json_with_docker_cli_messages(
            self.messages,
            trace=self.trace,
            enable_docker_cli=True,
            event_kind="PLAN",
            max_tool_calls=20,
            remind_every_tool_calls=self.tool_loop.remind_every_tool_calls,
        )
        print(f"[synthesize] step plan-generate: done  phase=schema-retry attempt={attempt_no}")
        return result

    async def quality_validate(self, obj: Any) -> Any:
        """Run host-side PLAN validation.

        The checks performed here include:
        - Canonical coercion into the exact PLAN shape expected by the trace.
        - Fixed identity fields from host decision must remain unchanged.
        - Full PLAN schema checks delegated to `validate_plan_event(...)`,
          including blueprint structure and other event-level constraints.
        - rubric LLM verification (mirroring AttemptFlow): after schema passes and refine is no longer triggered,
          run the 6-dimension authenticity rubric on the whole PLAN blueprint; if it does not meet the bar, raise RubricFailed to trigger re-planning.

        Option A: if output_path / write_json were passed, a schema-validation failure also persists the
        disabled PLAN event (result.ok=False + test_failure_details) before re-raising;
        if not passed, it raises directly, matching the original behavior.
        """
        schema_error: Exception | None = None
        plan_obj: dict | None = None
        try:
            plan_obj = self.schema_ops.coerce_event(
                obj,
                big_task_id=self.plan_decision["big_task_id"],
                reason=self.plan_decision["reason"],
                round_no=int(self.plan_decision["round"]),
            )
            self.schema_ops.validate_event(plan_obj)
        except Exception as _err:
            schema_error = _err

        if schema_error is not None:
            _bid = str(self.plan_decision.get("big_task_id") or "")
            # rubric runs unconditionally (per user requirement): even if schema fails, as long as an evaluable
            # plan_obj can be coerced, still run rubric and stash the feedback in self._last_rubric_feedback for the schema
            # retry branch to feed back to the model; if coercion also fails (plan_obj is None), there is no evaluable object, so skip.
            self._last_rubric_feedback = ""
            schema_verdict: dict | None = None
            if self.verify_client is not None and plan_obj is not None:
                print(f"[synthesize] step plan-rubric: starting (schema-failed)  big_task_id={_bid}", flush=True)
                try:
                    schema_verdict = await self._run_plan_rubric_verification(plan_obj)
                except Exception as ve:
                    msg = f"plan rubric verifier raised {type(ve).__name__}: {ve}"
                    print(f"[synthesize] step plan-rubric: verifier-error  big_task_id={_bid} {msg[:200]}", flush=True)
                    schema_verdict = None
                if schema_verdict is not None:
                    self._last_rubric_feedback = self.format_plan_feedback(schema_verdict)
                    if not schema_verdict.get("overall_pass"):
                        self._rubric_failure_count += 1
                    print(
                        f"[synthesize] step plan-rubric: done (schema-failed)  big_task_id={_bid}"
                        f"  avg={schema_verdict.get('avg')}  scores={schema_verdict.get('scores')}",
                        flush=True,
                    )
            if self.output_path is not None and self.write_json is not None:
                # Option A: persist the schema error as a disabled PLAN event.
                err_msg = str(schema_error)
                if plan_obj is not None:
                    stub = plan_obj
                else:
                    stub = {
                        "kind": "PLAN",
                        "big_task_id": self.plan_decision.get("big_task_id") or "",
                        "reason": self.plan_decision.get("reason") or "",
                        "round": int(self.plan_decision.get("round") or 0),
                        "enabled": False,
                        "natural_language_goal": "",
                        "acceptance_criteria": "",
                        "test_patch": [],
                        "dim_tags": {
                            "context_scope": "no_context",
                            "change_scope": "no_change",
                            "requirement_depth": "explicit_bug",
                            "knowledge_domains": [],
                            "new_task_angle": [],
                        },
                        "own_f2p": [],
                        "own_p2p": [],
                        "subtasks_blueprint": [],
                        "result": None,
                    }
                stub["enabled"] = False
                _details = [{
                    "node_id": "<schema_validation>",
                    "phase": "schema_validation",
                    "status": "failed",
                    "expectation_error": err_msg,
                    "output_excerpt": err_msg,
                }]
                if self._last_rubric_feedback:
                    _details.append({
                        "node_id": "<rubric_verification>",
                        "phase": "rubric_verification",
                        "status": "failed" if (schema_verdict is None or not schema_verdict.get("overall_pass")) else "passed",
                        "expectation_error": self._last_rubric_feedback,
                        "output_excerpt": self._last_rubric_feedback,
                    })
                _result = {
                    "ok": False,
                    "error_kind": "schema_validation_failed",
                    "failing_node_ids": [],
                    "test_failure_details": _details,
                }
                if schema_verdict is not None:
                    _result["rubric_verification"] = schema_verdict
                stub["result"] = _result
                self.trace["events"].append(stub)
                self.write_json(self.output_path, self.trace)
            raise schema_error

        # Trigger refinement: first round passed schema, refine_plan=True, not a repair, and no draft stored yet.
        if (
            self.refine_plan
            and self._draft_plan is None
            and self.plan_decision.get("reason") != "repair"
        ):
            self._draft_plan = plan_obj
            print(
                f"[synthesize] step plan-refine-trigger: ok  "
                f"big_task_id={plan_obj.get('big_task_id', '?')}  "
                f"subtasks={len(plan_obj.get('subtasks_blueprint', []))}  "
                f"draft accepted -> starting refinement pass"
            )
            raise PlanNeedsRefinement("schema ok; triggering one refinement pass")

        # (4) rubric LLM verification: runs only after schema passes and refine is no longer triggered (the refine draft
        # is not verified, only the final version is). Skipped when verify_client is None (dry-run / not configured),
        # preserving backward compatibility. The verdict is written back to plan_obj.result regardless of pass/fail.
        if self.verify_client is None:
            return plan_obj
        _bid = str(self.plan_decision.get("big_task_id") or "")
        print(f"[synthesize] step plan-rubric: starting  big_task_id={_bid}", flush=True)
        verdict = await self._run_plan_rubric_verification(plan_obj)
        result = dict(plan_obj.get("result") or {})
        result["rubric_verification"] = verdict
        result["ok"] = bool(verdict.get("overall_pass"))
        plan_obj["result"] = result
        if not verdict.get("overall_pass"):
            self._rubric_failure_count += 1
            feedback = self.format_plan_feedback(verdict)
            print(
                f"[synthesize] step plan-rubric: FAIL"
                f"  big_task_id={_bid}"
                f"  avg={verdict.get('avg')}"
                f"  scores={verdict.get('scores')}"
                f"  rubric_failures={self._rubric_failure_count}/{self.max_rubric_failures}"
                f"  issues={feedback!r}",
                flush=True,
            )
            raise RubricFailed(verdict=verdict)
        print(f"[synthesize] step plan-rubric: ok  big_task_id={_bid}", flush=True)
        return plan_obj

    def normalize_retry_obj(self, obj: Any) -> Any:
        """Best-effort normalize failed output into a stable object for direct reference in the next retry.

        If even normalization fails, keep the original object so the retry prompt faithfully shows what the model
        actually returned last round.
        """
        raw = super().normalize_retry_obj(obj)
        if not isinstance(raw, dict):
            return raw
        try:
            return self.schema_ops.coerce_event(
                dict(raw),
                big_task_id=self.plan_decision["big_task_id"],
                reason=self.plan_decision["reason"],
                round_no=int(self.plan_decision["round"]),
                )
        except Exception:
            return raw

    # ---- rubric verification (mirrors AttemptFlow's same-named methods; the only difference is the verdict schema uses 6-dimension scoring) ----
    _RUBRIC_DIMENSIONS = (
        "problem_statement_clarity",
        "problem_statement_no_leakage",
        "cross_file_collaboration",
        "skill_compositionality",
        "economic_professional_value",
        "repo_responsibility_alignment",
    )

    @staticmethod
    def verify_retry_user_message(error: Exception, bad_json: dict, *, attempt_no: int, max_attempts: int) -> str:
        """Build the retry note for when the rubric verifier's output format is wrong (mirrors AttemptFlow.verify_retry_user_message)."""
        return (
            "The previous VERIFY_PLAN output was not valid JSON, or did not pass the host check; you need to fix it directly.\n"
            f"This is VERIFY_PLAN generation attempt {attempt_no}/{max_attempts}; no tool calls are allowed this round.\n"
            f"Validation error reason: {type(error).__name__}: {error}\n"
            "The previous JSON was:\n"
            f"{json.dumps(bad_json, ensure_ascii=False, indent=2)}\n\n"
            "Return only one complete, corrected JSON object; no markdown, no explanation, no code block, no extra text before or after."
        )

    @classmethod
    def validate_plan_verdict(cls, verdict_obj: Any) -> dict:
        """Validate the shape of the PLAN rubric verifier JSON, and let the host compute overall_pass by rule.

        The only difference from attempt is the verdict schema: 6 dimensions scored 1-5. The model only scores + writes issues,
        it does not output overall_pass. The host computes it itself: overall_pass == (avg >= 4.0 and min(scores) > 1).
        For any dimension scoring < 4, the reason for the low score must appear in issues (there is no longer a low_score_reasons field).
        """
        if not isinstance(verdict_obj, dict):
            raise ValueError("VERIFY_PLAN output must be a JSON object")
        raw_scores = verdict_obj.get("scores")
        if not isinstance(raw_scores, dict):
            raise ValueError("VERIFY_PLAN scores must be a JSON object with the six dimensions")
        scores: dict[str, int] = {}
        for dim in cls._RUBRIC_DIMENSIONS:
            val = raw_scores.get(dim)
            if isinstance(val, bool) or not isinstance(val, int):
                raise ValueError(f"VERIFY_PLAN scores.{dim} must be an integer in [1,5]")
            if val < 1 or val > 5:
                raise ValueError(f"VERIFY_PLAN scores.{dim} must be in [1,5], got {val}")
            scores[dim] = val
        # The host computes overall_pass by rule, not delegating it to the model.
        avg = sum(scores.values()) / len(cls._RUBRIC_DIMENSIONS)
        overall_pass = (avg >= 4.0) and (min(scores.values()) > 1)
        issues = [
            str(x) for x in (verdict_obj.get("issues") or [])
            if isinstance(x, str) and x.strip()
        ]
        # For every dimension scoring < 4, the low-score reason must be written into issues (fed to the next PLAN refinement round).
        low_dims = [dim for dim, score in scores.items() if score < 4]
        if low_dims and not issues:
            raise ValueError(
                "VERIFY_PLAN issues must be non-empty because some dimension scored < 4: "
                + ", ".join(low_dims)
            )
        issues_blob = "\n".join(issues)
        missing = [dim for dim in low_dims if dim not in issues_blob]
        if missing:
            raise ValueError(
                "VERIFY_PLAN issues must explicitly name every dimension scoring < 4; "
                f"missing reasons for: {', '.join(missing)}"
            )
        return {
            "scores": scores,
            "avg": round(avg, 4),
            "overall_pass": overall_pass,
            "issues": issues,
            "diagnostic_feedback_for_retry": str(
                verdict_obj.get("diagnostic_feedback_for_retry") or ""
            ),
        }

    @staticmethod
    def format_plan_feedback(verdict: dict) -> str:
        """Summarize a rubric failure into a text summary for the model (mirrors AttemptFlow.format_feedback)."""
        lines: list[str] = []
        scores = verdict.get("scores") or {}
        if scores:
            lines.append("[dimension scores]")
            lines.extend(f"- {dim}: {score}" for dim, score in scores.items())
            if verdict.get("avg") is not None:
                lines.append(f"- average: {verdict.get('avg')} (need >= 4.0 and no dimension == 1)")
        issues = verdict.get("issues") or []
        if issues:
            lines.append("[issues (includes the reason for every dimension scoring < 4 — address these in the next plan)]")
            lines.extend(f"- {item}" for item in issues)
        diagnostic = str(verdict.get("diagnostic_feedback_for_retry") or "").strip()
        if diagnostic:
            lines.append("[diagnostic feedback]")
            lines.append(diagnostic)
        return "\n".join(lines).strip()

    def rubric_retry_user_message(self, verdict: dict) -> str:
        """The user message fed back to the model to re-plan after a rubric failure."""
        return (
            "The previous PLAN passed schema, but did not pass the task-authenticity rubric verification; you need to re-plan a more authentic, "
            "more engineering-valuable PLAN. Here is the verification feedback:\n\n"
            f"{self.format_plan_feedback(verdict)}\n\n"
            "Make substantive improvements to the low-scoring dimensions (for example: make the problem statement clearer and better-bounded, avoid leaking the specific fix, "
            "let cross-file collaboration be driven by real dependencies rather than artificially spread out, raise skill compositionality, align with repo responsibility, ensure there is a real beneficiary), "
            "but never leak the solution in the problem statement. Keep big_task_id / reason / round unchanged.\n"
            "Return only one complete, corrected PLAN JSON object; no markdown, no explanation, no code block, no extra text before or after."
        )

    async def _run_plan_rubric_verification(self, plan_obj: dict) -> dict:
        """Run the rubric verifier on a schema-passing PLAN and return the validated verdict.

        Reuses the multi-round validation skeleton of AttemptFlow._run_rubric_verification, only replacing:
        the prompt builder (build_verify_plan_prompt), usage_stage (VERIFY_PLAN), and
        the validation function (validate_plan_verdict).
        """
        if self.verify_client is None or self.workspace is None:
            raise RuntimeError("PlanFlow verify_client and workspace are required for rubric verification")

        prompt_name = "verify_plan.txt"
        system, user = self.workspace.build_verify_plan_prompt(
            current_plan=plan_obj,
            category=str(plan_obj.get("reason") or self.plan_decision.get("reason") or "plan"),
            prompt_name=prompt_name,
            trace=self.trace,
        )
        messages: list[dict[str, Any]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        last_obj: Any = None
        last_error: Exception | None = None
        for attempt_no in range(1, self.verify_max_attempts + 1):
            try:
                if attempt_no > 1:
                    assert last_obj is not None and last_error is not None
                    messages.append({"role": "assistant", "content": json.dumps(last_obj, ensure_ascii=False)})
                    messages.append({
                        "role": "user",
                        "content": self.verify_retry_user_message(
                            last_error, last_obj, attempt_no=attempt_no, max_attempts=self.verify_max_attempts,
                        ),
                    })
                stage = "VERIFY_PLAN" if attempt_no == 1 else "VERIFY_PLAN_VALIDATION_RETRY"
                body = await self.verify_client.chat_once(messages, usage_stage=stage)
                raw_obj = json_from_text(self.verify_client.client_response_text(body))
                return self.validate_plan_verdict(raw_obj)
            except Exception as e:
                last_obj = raw_obj if 'raw_obj' in locals() else None
                if not isinstance(last_obj, dict):
                    last_obj = {"_raw": last_obj}
                last_error = e
        assert last_error is not None
        raise RuntimeError(f"{type(last_error).__name__}: {last_error}")

    async def run(self) -> tuple[Any | None, str | None]:
        """Run the PLAN generation retry loop; if all refinement rounds fail, fall back to returning the first-round draft."""
        result, error = await super().run()
        if result is None and error is not None and self._draft_plan is not None:
            # All strengthening rounds failed: fall back to the first-round schema-passing draft.
            draft_subtasks = self._draft_plan.get("subtasks_blueprint") or []
            draft_summary = "\n".join(
                f"  [{i+1}] id={s.get('id')} category={s.get('category')} text={str(s.get('text') or '')[:120]}"
                for i, s in enumerate(draft_subtasks)
            )
            print(
                f"[synthesize] step plan-refine: fallback  "
                f"big_task_id={self._draft_plan.get('big_task_id', '?')}  "
                f"refinement all failed -> using draft plan\n"
                f"  error: {error}\n"
                f"  draft subtasks ({len(draft_subtasks)}):\n{draft_summary}\n"
                f"  draft_plan_json: {json.dumps(self._draft_plan, ensure_ascii=False)}",
                flush=True,
            )
            return self._draft_plan, None
        # Strengthening succeeded: attach the draft to a top-level field of the final plan event for comparison reference.
        if result is not None and self._draft_plan is not None and result is not self._draft_plan:
            result["draft_plan"] = self._draft_plan
            print(
                f"[synthesize] step plan-refine: ok  "
                f"big_task_id={result.get('big_task_id', '?')}  "
                f"draft_subtasks={len(self._draft_plan.get('subtasks_blueprint', []))}  "
                f"final_subtasks={len(result.get('subtasks_blueprint', []))}  "
                f"draft saved to result.draft_plan"
            )
        return result, error


class AttemptUserQueryFlow(BaseFlow):
    """Generate `ATTEMPT.user_query` and `ATTEMPT.acceptance_criteria`, and validate the format.

    This is the first half of a v2 ATTEMPT, deliberately split from patch/test generation.

    Prompt organization:
    - The prompt is rendered from the current BigTask's history slice, the current target subtask, and the
      failure brief belonging only to the current subtask.
    - This step does not generate patch / own_f2p / own_p2p.
    - On retry it attaches the previous bad JSON and the host-written format-error note.

    Validation contract:
    - `user_query` must be a dict containing exactly the two fields `role`/`content`.
    - `role` must be `user`.
    - `content` must be a non-empty string.
    - `acceptance_criteria` must be a non-empty string.

    This validation only serves the user_query generation stage. If it fails, only this step is retried;
    it does not proceed into patch+f2p synthesis.
    """

    @staticmethod
    def retry_user_message(error: Exception, bad_json: dict, *, attempt_no: int, max_attempts: int) -> str:
        """Build the format-error note fed back to the model on a user_query retry."""
        return (
            "The previous ATTEMPT user query output was not valid JSON, or did not pass the host check; you need to fix it directly.\n"
            f"This is ATTEMPT user query generation attempt {attempt_no}/{max_attempts}; no tool calls are allowed this round.\n"
            f"Validation error reason: {type(error).__name__}: {error}\n"
            "The previous JSON was:\n"
            f"{json.dumps(bad_json, ensure_ascii=False, indent=2)}\n\n"
            "Return only one complete, corrected JSON object; no markdown, no explanation, no code block, no extra text before or after."
        )

    @staticmethod
    def validate_user_query(query_obj: Any, *, subtask: dict) -> dict:
        """Validate the format of the user_query + acceptance_criteria fields, returning a normalized result dict."""
        if not isinstance(query_obj, dict):
            raise ValueError("ATTEMPT user query output must be a JSON object")
        user_query = query_obj.get("user_query")
        if not isinstance(user_query, dict) or not user_query:
            raise ValueError("user_query must be a non-empty dict with role/content")
        keys = set(user_query.keys())
        if keys != {"role", "content"}:
            raise ValueError(
                f"user_query must contain exactly role/content, got {sorted(keys)}"
            )
        if user_query.get("role") != "user":
            raise ValueError("user_query.role must be user")
        content = user_query.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("user_query.content must be non-empty string")
        acceptance_criteria = query_obj.get("acceptance_criteria")
        if not isinstance(acceptance_criteria, str) or not acceptance_criteria.strip():
            raise ValueError("acceptance_criteria must be non-empty string")
        return {
            "user_query": {"role": "user", "content": content},
            "acceptance_criteria": acceptance_criteria,
        }

    def __init__(
        self, *, client, prompt: tuple[str | None, str], subtask: dict,
        max_attempts: int = 3, json_repair=None,
        trace=None, big_task_id: str | None = None, output_path=None, write_json=None,
    ):
        """Save the dependencies needed for user_query generation and prepare the base message list.

        `trace` / `big_task_id` / `output_path` / `write_json` are optional; when provided,
        a schema-validation failure also persists a disabled stub ATTEMPT event to disk.
        """
        super().__init__(max_attempts=max_attempts, json_repair=json_repair)
        self.client = client
        self.subtask = subtask
        self.trace = trace
        self.big_task_id = big_task_id
        self.output_path = output_path
        self.write_json = write_json
        system, user = prompt
        self.messages: list[dict[str, Any]] = []
        if system is not None:
            self.messages.append({"role": "system", "content": system})
        self.messages.append({"role": "user", "content": user})

    async def generate(self, attempt_no: int, last_obj: Any, last_error: Exception | None) -> Any:
        """Run the user_query prompt.

        On retry it attaches the previous invalid JSON and the host-written format-requirement note,
        explicitly restating the exact shape constraints of user_query.
        """
        if attempt_no > 1:
            assert last_obj is not None and last_error is not None
            self.messages.append({"role": "assistant", "content": json.dumps(last_obj, ensure_ascii=False)})
            self.messages.append({
                "role": "user",
                "content": self.retry_user_message(
                    last_error, last_obj, attempt_no=attempt_no, max_attempts=self.max_attempts,
                ),
            })
        stage = "ATTEMPT_USER_QUERY" if attempt_no == 1 else "ATTEMPT_USER_QUERY_VALIDATION_RETRY"
        sid = str(self.subtask.get("id") or "")
        print(f"[synthesize] step user-query-generate: starting  attempt={attempt_no} subtask_id={sid}", flush=True)
        body = await self.client.chat_once(self.messages, usage_stage=stage)
        result = json_from_text(self.client.client_response_text(body))
        print(f"[synthesize] step user-query-generate: done  attempt={attempt_no} subtask_id={sid}", flush=True)
        return result

    def quality_validate(self, obj: Any) -> Any:
        """Run strict `user_query` format validation for the current subtask.

        Option A: if trace / big_task_id / output_path / write_json were passed,
        a schema-validation failure persists a disabled stub ATTEMPT event (result.ok=False +
        test_failure_details) before re-raising; if not passed, it raises directly.
        """
        schema_error: Exception | None = None
        validated: dict | None = None
        try:
            validated = self.validate_user_query(
                obj, subtask=self.subtask,
            )
        except Exception as _err:
            schema_error = _err

        if schema_error is not None:
            can_persist = (
                self.trace is not None
                and self.big_task_id is not None
                and self.output_path is not None
                and self.write_json is not None
            )
            if can_persist:
                err_msg = str(schema_error)
                stub: dict = {
                    "kind": "ATTEMPT",
                    "big_task_id": self.big_task_id,
                    "subtask_id": str(self.subtask.get("id") or ""),
                    "enabled": False,
                    "natural_language_goal": str(self.subtask.get("text") or ""),
                    "test_patch": [],
                    "acceptance_criteria": "",
                    "own_f2p": [],
                    "own_p2p": [],
                    "user_query": {},
                    "patch": [{"format": "unified_diff", "diff": ""}],
                    "result": {
                        "ok": False,
                        "error_kind": "schema_validation_failed",
                        "failing_node_ids": [],
                        "test_failure_details": [{
                            "node_id": "<schema_validation>",
                            "phase": "schema_validation",
                            "status": "failed",
                            "expectation_error": err_msg,
                            "output_excerpt": err_msg,
                        }],
                    },
                }
                self.trace["events"].append(stub)
                self.write_json(self.output_path, self.trace)
            raise schema_error

        return validated


class AttemptFlow(BaseFlow):
    """Generate the second half of ATTEMPT: fields such as patch / test_patch / own_f2p.

    This flow now carries two layers of host validation at once:
    1. Main generation stage: produce and validate the ATTEMPT JSON/schema.
    2. Embedded rubric verification stage: after code execution passes, call the verifier prompt and
       also fold the verifier's verdict into the same flow's async `quality_validate(...)`.

    This way ATTEMPT's "generation + schema validation + verifier JSON validation" is consolidated into one class;
    the outer orchestration only decides when to run code validation, when to trigger this embedded
    verifier branch, and whether a verifier failure goes back to patch+f2p retry or is directly fatal.
    """

    @staticmethod
    def retry_user_message(error: Exception, bad_json: dict, *, attempt_no: int, max_attempts: int) -> str:
        """Build the schema-error note for a retry of the ATTEMPT implementation stage."""
        return (
            "The previous ATTEMPT output was not valid JSON, or did not pass the host's _validate_attempt_event(attempt_obj) check; you need to fix it directly.\n"
            f"This is ATTEMPT generation attempt {attempt_no}/{max_attempts}; no tool calls are allowed this round, especially do not call docker_cli/tool_calls.\n"
            f"Validation error reason: {type(error).__name__}: {error}\n"
            "The previous JSON was:\n"
            f"{json.dumps(bad_json, ensure_ascii=False, indent=2)}\n\n"
            "Return only one complete, corrected ATTEMPT JSON object; no markdown, no explanation, no code block, no extra text before or after."
        )

    @staticmethod
    def verify_retry_user_message(error: Exception, bad_json: dict, *, attempt_no: int, max_attempts: int) -> str:
        """Build the retry note for when the rubric verifier's output format is wrong."""
        return (
            "The previous VERIFY_ATTEMPT output was not valid JSON, or did not pass the host check; you need to fix it directly.\n"
            f"This is VERIFY_ATTEMPT generation attempt {attempt_no}/{max_attempts}; no tool calls are allowed this round.\n"
            f"Validation error reason: {type(error).__name__}: {error}\n"
            "The previous JSON was:\n"
            f"{json.dumps(bad_json, ensure_ascii=False, indent=2)}\n\n"
            "Return only one complete, corrected JSON object; no markdown, no explanation, no code block, no extra text before or after."
        )

    @staticmethod
    def validate_verdict(verdict_obj: Any) -> dict:
        """Validate the shape and internal consistency of the verifier JSON."""
        if not isinstance(verdict_obj, dict):
            raise ValueError("VERIFY_ATTEMPT output must be a JSON object")
        for key in ("alignment_pass", "self_contained_pass", "overall_pass"):
            if not isinstance(verdict_obj.get(key), bool):
                raise ValueError(f"VERIFY_ATTEMPT {key} must be boolean")
        verdict = {
            "alignment_pass": verdict_obj["alignment_pass"],
            "self_contained_pass": verdict_obj["self_contained_pass"],
            "overall_pass": verdict_obj["overall_pass"],
            "alignment_issues": [
                str(x) for x in (verdict_obj.get("alignment_issues") or [])
                if isinstance(x, str) and x.strip()
            ],
            "self_containedness_issues": [
                str(x) for x in (verdict_obj.get("self_containedness_issues") or [])
                if isinstance(x, str) and x.strip()
            ],
            "diagnostic_feedback_for_retry": str(
                verdict_obj.get("diagnostic_feedback_for_retry") or ""
            ),
        }
        expected_overall = verdict["alignment_pass"] and verdict["self_contained_pass"]
        if verdict["overall_pass"] != expected_overall:
            raise ValueError(
                "VERIFY_ATTEMPT overall_pass must equal alignment_pass && self_contained_pass"
            )
        if not verdict["alignment_pass"] and not verdict["alignment_issues"]:
            raise ValueError("VERIFY_ATTEMPT alignment_issues must be non-empty when alignment fails")
        if not verdict["self_contained_pass"] and not verdict["self_containedness_issues"]:
            raise ValueError(
                "VERIFY_ATTEMPT self_containedness_issues must be non-empty when self-containedness fails"
            )
        return verdict

    @staticmethod
    def format_feedback(verdict: dict) -> str:
        """Summarize a rubric failure into a text summary written back to ATTEMPT.result."""
        lines: list[str] = []
        alignment = verdict.get("alignment_issues") or []
        self_contained = verdict.get("self_containedness_issues") or []
        if alignment:
            lines.append("[alignment issues]")
            lines.extend(f"- {item}" for item in alignment)
        if self_contained:
            lines.append("[self-containedness issues]")
            lines.extend(f"- {item}" for item in self_contained)
        diagnostic = str(verdict.get("diagnostic_feedback_for_retry") or "").strip()
        if diagnostic:
            lines.append("[diagnostic feedback]")
            lines.append(diagnostic)
        return "\n".join(lines).strip()

    @classmethod
    def apply_verification_result(
        cls, trace: dict, *, attempt_index: int, verdict: dict,
        extra_failure_details: list[dict] | None = None,
    ) -> None:
        """Write the rubric verification result back to ATTEMPT.result.

        rubric passed: only append rubric_verification; ok/enabled keep the code-execution result unchanged.
        rubric failed: set ok to False, enabled to False, and write error_kind /
        test_failure_details, marking it pending-retry or, when max_rubric_failures lets it through, restored later.

        `extra_failure_details`: entries additionally folded into test_failure_details on rubric failure.
        A typical source is the detailed error info of failed F2P nodes trimmed during the PARTIAL-KEEP stage --
        on rubric failure this F2P failure evidence is also relayed back to the model to provide more repair clues.
        """
        attempt = (trace.get("events") or [])[attempt_index]
        result = dict(attempt.get("result") or {})
        result["rubric_verification"] = verdict
        if verdict.get("overall_pass"):
            attempt["result"] = result
            return

        feedback = cls.format_feedback(verdict)
        result["ok"] = False
        result["failing_node_ids"] = []
        result["error_kind"] = "rubric_verification_failed"
        details: list[dict] = [{
            "node_id": "<rubric_verification>",
            "phase": "rubric_verification",
            "status": "failed",
            "expectation_error": feedback,
            "output_excerpt": feedback,
        }]
        for d in (extra_failure_details or []):
            if isinstance(d, dict):
                details.append(dict(d))
        result["test_failure_details"] = details
        attempt["result"] = result
        attempt["enabled"] = False

    def __init__(
        self, *, client, prompt: tuple[str | None, str], trace, big_task_id: str, subtask: dict,
        user_query: dict, acceptance_criteria: str, schema_ops: SchemaOps,
        tool_loop: ToolLoopConfig | None = None, max_attempts: int = 3, json_repair=None,
        repair_client=None, verify_client=None, verify_max_attempts: int = 2,
        max_rubric_failures: int = 2,
        rubric_max_rounds: int = 2,
        inner_repair_max: int = 3,
        output_path=None, write_json=None, load_trace=None, executor=None,
        enable_execute: bool = False, workspace=None,
        prompt_name: str = "generate_f2p.txt",
        first_call_client=None,
    ):
        """Save the ATTEMPT implementation-stage dependencies and prepare the first-round messages.

        This also holds all the dependencies needed to consolidate the three validation layers
        "format -> code execution -> rubric" into `quality_validate`:

        - `output_path` / `write_json` / `load_trace`: append the candidate event, persist, and
          reload the trace with result after code execution.
        - `executor`: the file-based code execution entry point (signature same as
          `run_pending_attempts(path, *, stop_on_fail, verbose)`).
        - `enable_execute`: in dry-run (`--no-execute`), only do schema validation.
        - `workspace`: internally build the verify prompt and rebuild the impl prompt on functional retry.
        """
        super().__init__(max_attempts=max_attempts, json_repair=json_repair)
        self.max_rubric_failures = max_rubric_failures
        self._rubric_failure_count = 0
        # rubric runs only in the first rubric_max_rounds retry rounds; later rounds (e.g. rounds 3-5 of 5)
        # only do schema + code-execution validation, skipping the rubric semantic check. run() writes
        # the current attempt_no to self._current_attempt_no at the start of each round, and quality_validate decides based on it.
        self.rubric_max_rounds = rubric_max_rounds
        self._current_attempt_no = 1
        # The inner "does not consume a retry slot" repair cap in generate (JSON parse failure / infra error).
        self.inner_repair_max = max(1, int(inner_repair_max))
        # PARTIAL-KEEP snapshot: when own_f2p is let through with half passing, record the trimmed failed nodes
        # and their detailed error info, to relay back to the model together on rubric failure (providing more repair clues).
        self._partial_keep_dropped_f2p: list[str] = []
        self._partial_keep_dropped_details: list[dict] = []
        self._partial_keep_dropped_status: dict[str, str] = {}
        self.client = client
        self.trace = trace
        self.big_task_id = big_task_id
        self.subtask = subtask
        self.user_query = user_query
        self.acceptance_criteria = acceptance_criteria
        self.schema_ops = schema_ops
        self.tool_loop = tool_loop or ToolLoopConfig()
        self.repair_client = repair_client
        self.verify_client = verify_client
        self.verify_max_attempts = verify_max_attempts
        # The client used for the first 3 main-generation rounds (sglang); falls back to the main client when None. From round 4 on, the main client (opus) is used.
        self.first_call_client = first_call_client or client
        self.output_path = output_path
        self.write_json = write_json
        self.load_trace = load_trace
        self.executor = executor
        self.enable_execute = enable_execute
        self.workspace = workspace
        self.prompt_name = prompt_name
        self._allow_user_query_override = False
        system, user = prompt
        self.messages: list[dict[str, Any]] = []
        if system is not None:
            self.messages.append({"role": "system", "content": system})
        self.messages.append({"role": "user", "content": user})

    def _gen_client_for(self, attempt_no: int):
        """Two-stage cascade client selection: rounds 1-3 use sglang (first_call_client), from round 4 fall back to the main client (opus)."""
        if attempt_no <= 3:
            return self.first_call_client
        return self.client

    async def _generate_with_tools(self, attempt_no: int, last_error: Exception | None, *, inner: int) -> Any:
        """Generation with docker_cli tools (original prompt on first round / rebuilt impl prompt on retry rounds).

        - attempt_no==1: use the original messages from construction (first round).
        - attempt_no>1: rebuild the impl prompt from the latest trace (functional retry or infra/degraded re-run),
          so the model re-synthesizes after reading the failure history. The client is decided by attempt_no<=2 (deepseek/opus).
        """
        sid = str(self.subtask.get("id") or "")
        if attempt_no == 1:
            phase = "original"
            print(f"[synthesize] step attempt-generate: starting  phase={phase} attempt={attempt_no} inner={inner} subtask_id={sid}", flush=True)
            result = await self.first_call_client.call_json_with_docker_cli_messages(
                self.messages,
                trace=self.trace,
                enable_docker_cli=self.tool_loop.enable_docker_cli,
                event_kind="ATTEMPT",
                max_tool_calls=self.tool_loop.first_round_max_tool_calls or self.tool_loop.max_tool_calls,
                remind_every_tool_calls=self.tool_loop.first_round_remind_every_tool_calls or self.tool_loop.remind_every_tool_calls,
                remind_start_tool_calls=self.tool_loop.first_round_remind_start_tool_calls,
                wrap_up_reminder=bool(self.tool_loop.first_round_remind_start_tool_calls),
                reminder_context=self.tool_loop.reminder_context,
            )
            print(f"[synthesize] step attempt-generate: done  phase={phase} attempt={attempt_no} inner={inner} subtask_id={sid}", flush=True)
            return result
        # attempt_no>1: rebuild the impl prompt (with tools). Functional retry reads the failure history; infra/degraded re-run also takes this path.
        self._allow_user_query_override = True
        self.trace = self.load_trace(self.output_path) if self.load_trace is not None else self.trace
        system, user = self.workspace.build_attempt_impl_prompt(
            trace=self.trace,
            target={"big_task_id": self.big_task_id, "subtask_id": self.subtask["id"]},
            user_query=self.user_query,
            acceptance_criteria=self.acceptance_criteria,
            prompt_name=self.prompt_name,
        )
        self.messages = []
        if system is not None:
            self.messages.append({"role": "system", "content": system})
        self.messages.append({"role": "user", "content": user})
        if isinstance(last_error, CodeValidationFailed):
            phase = "functional-retry-code"
        elif isinstance(last_error, RubricFailed):
            phase = "functional-retry-rubric"
        else:
            phase = "functional-retry-degraded"
        print(f"[synthesize] step attempt-generate: starting  phase={phase} attempt={attempt_no} inner={inner} subtask_id={sid}", flush=True)
        # Two-stage cascade main generation: rounds 1-3 = first (sglang) / from round 4 = main client (opus).
        # Rounds 1-3 (_first_round) reuse the first_round tool cap/reminders.
        _first_round = attempt_no <= 3
        gen_client = self._gen_client_for(attempt_no)
        result = await gen_client.call_json_with_docker_cli_messages(
            self.messages,
            trace=self.trace,
            enable_docker_cli=self.tool_loop.enable_docker_cli,
            event_kind="ATTEMPT",
            max_tool_calls=(
                (self.tool_loop.first_round_max_tool_calls or self.tool_loop.max_tool_calls)
                if _first_round else self.tool_loop.max_tool_calls
            ),
            remind_every_tool_calls=(
                (self.tool_loop.first_round_remind_every_tool_calls or self.tool_loop.remind_every_tool_calls)
                if _first_round else self.tool_loop.remind_every_tool_calls
            ),
            remind_start_tool_calls=(
                self.tool_loop.first_round_remind_start_tool_calls if _first_round else 0
            ),
            wrap_up_reminder=bool(_first_round and self.tool_loop.first_round_remind_start_tool_calls),
            reminder_context=self.tool_loop.reminder_context,
        )
        print(f"[synthesize] step attempt-generate: done  phase={phase} attempt={attempt_no} inner={inner} subtask_id={sid}", flush=True)
        return result

    async def _generate_schema_repair(self, attempt_no: int, last_obj: Any, last_error: Exception | None, *, inner: int) -> Any:
        """Tool-free openai JSON/schema repair: continue the message thread, attaching the previous bad JSON + repair note.

        Used only for pure-format issues like "JSON parse failure"; does not run tools, executor, or rubric.
        This path does not consume an ATTEMPT retry slot (handled by the inner loop of generate).
        """
        self.messages.append({"role": "assistant", "content": json.dumps(last_obj, ensure_ascii=False)})
        self.messages.append({
            "role": "user",
            "content": self.retry_user_message(
                last_error, last_obj, attempt_no=attempt_no, max_attempts=self.max_attempts,
            ),
        })
        chat_client = self.repair_client or self.client
        sid = str(self.subtask.get("id") or "")
        print(f"[synthesize] step attempt-generate: starting  phase=schema-retry-inner attempt={attempt_no} inner={inner} subtask_id={sid}", flush=True)
        body = await chat_client.chat_once(self.messages, usage_stage="ATTEMPT_VALIDATION_RETRY")
        result = json_from_text(chat_client.client_response_text(body))
        print(f"[synthesize] step attempt-generate: done  phase=schema-retry-inner attempt={attempt_no} inner={inner} subtask_id={sid}", flush=True)
        return result

    async def generate(self, attempt_no: int, last_obj: Any, last_error: Exception | None) -> Any:
        """Run the ATTEMPT implementation prompt, with a built-in "does not consume a retry slot" repair loop.

        The outer `BaseFlow.run()` increments attempt_no only on a **functional failure** (CodeValidationFailed / RubricFailed)
        or when this method **throws after its inner loop is exhausted** (degradation). This method internally absorbs two kinds of
        non-generation-quality failures, each of which does not consume a slot and is counted independently within an attempt (reset after an attempt retry):

        - **JSON parse failure** (`GeneratedJSONParseError`) -> switch to tool-free openai schema repair,
          continuing the message thread to fix the JSON.
        - **infra error** (deepseek 400 protocol state / rate limiting / network timeout, etc., `_is_infra_error`)
          -> re-run "tool-enabled generation" (not tool-free openai repair, avoiding a mismatched remedy).

        After `inner_repair_max` (default 3) consecutive rounds still yield no parseable result -> throw the last error,
        and the outer run() increments attempt_no (degrading to one normal tool-enabled attempt).
        """
        sid = str(self.subtask.get("id") or "")
        # First generation mode: attempt==1 or a functional failure -> with tools; the rest (including degraded bubble-up) also use tools.
        mode = "tools"
        inner_last_obj = last_obj
        inner_last_err = last_error
        for inner in range(1, self.inner_repair_max + 1):
            try:
                if mode == "schema_repair":
                    return await self._generate_schema_repair(attempt_no, inner_last_obj, inner_last_err, inner=inner)
                return await self._generate_with_tools(attempt_no, inner_last_err, inner=inner)
            except GeneratedJSONParseError as je:
                # JSON parse failure: does not consume a slot, switch to tool-free openai repair.
                inner_last_err = je
                inner_last_obj = self.normalize_retry_obj({"_raw": getattr(je, "raw_text", "")})
                mode = "schema_repair"
                print(
                    f"[synthesize] step attempt-generate-inner: json-parse FAIL → schema-repair  "
                    f"subtask_id={sid} attempt={attempt_no} inner={inner}/{self.inner_repair_max} "
                    f"error={str(je)[:200]}",
                    flush=True,
                )
                continue
            except Exception as ex:
                if _is_infra_error(ex):
                    # infra error: does not consume a slot, re-run tool-enabled generation (not tool-free repair).
                    inner_last_err = ex
                    mode = "tools"
                    print(
                        f"[synthesize] step attempt-generate-inner: infra FAIL → tool-regen  "
                        f"subtask_id={sid} attempt={attempt_no} inner={inner}/{self.inner_repair_max} "
                        f"error={type(ex).__name__}: {str(ex)[:200]}",
                        flush=True,
                    )
                    continue
                # Other non-JSON/infra exceptions: re-raise directly, handled by run() as usual.
                raise
        # Inner loop exhausted: throw the last error, triggering outer degradation (attempt_no+1, tool-enabled functional retry).
        print(
            f"[synthesize] step attempt-generate-inner: exhausted  "
            f"subtask_id={sid} attempt={attempt_no} inner_repair_max={self.inner_repair_max} "
            f"final_error={type(inner_last_err).__name__ if inner_last_err else 'none'} -> degrading to one normal attempt",
            flush=True,
        )
        raise inner_last_err or RuntimeError("ATTEMPT generate inner repair exhausted without result")


    async def quality_validate(self, obj: Any) -> Any:
        """ATTEMPT's single validation entry point: format validation -> code execution validation -> rubric LLM verification.

        All three validation layers "raise on finding a problem", handled uniformly by the retry skeleton of `BaseFlow.run()`:

        - (1) format (schema) failure -> raise `ValueError`.
          In execute mode it also persists the disabled ATTEMPT event to disk (Option A),
          ensuring schema errors also enter the test_failure_details history;
          in dry-run mode it raises directly, producing no orphan event.
        - (2) code execution failure -> raise `CodeValidationFailed` (the event is already appended, kept on disk as failure history).
        - (3) rubric failure -> raise `RubricFailed` (the event is already appended, kept on disk as failure history).

        In `--no-execute` dry-run, only (1) format validation is done, then the normalized event is returned directly.
        """
        # At the start of each validation round, first clear the previous round's PARTIAL-KEEP snapshot to avoid old evidence bleeding into this round.
        self._partial_keep_dropped_f2p = []
        self._partial_keep_dropped_details = []
        self._partial_keep_dropped_status = {}
        # (1) Format validation: side-effect-free, must complete before append.
        schema_error: Exception | None = None
        attempt_obj: dict | None = None
        try:
            attempt_obj = self.schema_ops.coerce_event(
                obj,
                big_task_id=self.big_task_id,
                subtask=self.subtask,
                user_query=(
                    None if self._allow_user_query_override else self.user_query
                ),
                acceptance_criteria=self.acceptance_criteria,
            )
            self.schema_ops.validate_event(attempt_obj)
            if self._allow_user_query_override:
                self.user_query = dict(attempt_obj["user_query"])
                self._allow_user_query_override = False
            if self.schema_ops.is_request_instruction_subtask(self.subtask) and not attempt_obj.get("own_f2p"):
                raise ValueError(
                    "ATTEMPT validation failed: request_or_instruction requires non-empty own_f2p; regenerate this ATTEMPT"
                )
        except Exception as _err:
            schema_error = _err

        # rubric runs unconditionally (per user requirement): on schema failure / code execution failure, rubric still runs,
        # and its feedback is merged with the schema/code failure details into the same event's result.test_failure_details,
        # so the next round's "functional retry" prompt (reading the trace's failure_brief) can see all the feedback at once.
        sid = str(self.subtask.get("id") or self.subtask.get("subtask_id") or "")

        # dry-run (--no-execute): no trace persistence or retry loop, keep the original behavior --
        # schema failure raises directly, schema pass returns directly, without running executor / rubric.
        if not self.enable_execute:
            if schema_error is not None:
                print(f"[synthesize] step attempt-validate: FAIL  phase=schema  subtask_id={sid} error={str(schema_error)[:200]}", flush=True)
                raise schema_error
            return attempt_obj

        # (1) Persist the pending event to disk and determine code_ok.
        code_ok = True
        if schema_error is not None:
            # schema failure: persist a disabled stub (result carries schema failure details), then continue to rubric.
            print(f"[synthesize] step attempt-validate: FAIL  phase=schema  subtask_id={sid} error={str(schema_error)[:200]}", flush=True)
            err_msg = str(schema_error)
            if attempt_obj is not None:
                event_obj = attempt_obj
            else:
                event_obj = {
                    "kind": "ATTEMPT",
                    "big_task_id": self.big_task_id,
                    "subtask_id": str(self.subtask.get("id") or ""),
                    "enabled": False,
                    "natural_language_goal": str(self.subtask.get("text") or ""),
                    "test_patch": [],
                    "acceptance_criteria": self.acceptance_criteria or "",
                    "own_f2p": [],
                    "own_p2p": [],
                    "user_query": self.user_query or {},
                    "patch": [{"format": "unified_diff", "diff": ""}],
                    "result": None,
                }
            event_obj["enabled"] = False
            event_obj["result"] = {
                "ok": False,
                "error_kind": "schema_validation_failed",
                "failing_node_ids": [],
                "test_failure_details": [{
                    "node_id": "<schema_validation>",
                    "phase": "schema_validation",
                    "status": "failed",
                    "expectation_error": err_msg,
                    "output_excerpt": err_msg,
                }],
            }
            idx = len(self.trace["events"])
            self.trace["events"].append(event_obj)
            self.write_json(self.output_path, self.trace)
            code_ok = False
        else:
            # (2) append + persist (persist only after schema passes).
            idx = len(self.trace["events"])
            self.trace["events"].append(attempt_obj)
            self.write_json(self.output_path, self.trace)

            # (3) Code execution validation: the file-based executor runs the current pending ATTEMPT in place and backfills result.
            # Non-code subtasks also go through here: the executor applies their patch (empty or not) and, via
            # cumulative regression, checks that previously passing tests in the BigTask still pass; it just does not declare
            # own_f2p pytest nodes (here own_f2p is an acceptance-criteria string).
            _exe_sid = sid
            print(f"[synthesize] step attempt-execute: starting  subtask_id={_exe_sid}", flush=True)
            self.executor(str(self.output_path), stop_on_fail=False, verbose=True)
            self.trace = self.load_trace(self.output_path)
            current_attempt = self.trace["events"][idx]
            result = dict(current_attempt.get("result") or {})
            if result.get("ok") is not True:
                # Partial keep: attempted only when the failure type is a pure test failure (PYTEST_FAIL). Other hard errors
                # (TEST_PATCH_APPLY_FAIL / COLLECT_MISS / TIMEOUT / PATCH_* etc.) retry as usual.
                # Rule: among own_f2p nodes with per_node_status==PASSED, if they are >= half of the total,
                # trim own_f2p to keep only these passing nodes, clear result and re-run the executor for a clean snapshot,
                # then continue to rubric; otherwise handle as a code failure as usual.
                _own_f2p = [str(n) for n in (current_attempt.get("own_f2p") or [])]
                _per_node = result.get("per_node_status") or {}
                _passing_f2p = [n for n in _own_f2p if str(_per_node.get(n) or "").upper() == "PASSED"]
                if (
                    result.get("error_kind") == "PYTEST_FAIL"
                    and len(_own_f2p) > len(_passing_f2p)  # there really are failing nodes to trim
                    and len(_passing_f2p) * 2 >= len(_own_f2p)
                ):
                    _dropped = [n for n in _own_f2p if n not in set(_passing_f2p)]
                    # Let it through when half pass, but the detailed error info of the trimmed failed nodes must be preserved:
                    # if rubric later fails, this F2P failure evidence is relayed back to the model together, providing more
                    # repair clues (per user requirement). Snapshot it here before clearing result.
                    _dropped_set = set(_dropped)
                    _kept_failure_details: list[dict] = []
                    for _d in (result.get("test_failure_details") or []):
                        if not isinstance(_d, dict):
                            continue
                        if str(_d.get("node_id") or "") in _dropped_set:
                            _kept_failure_details.append(dict(_d))
                    self._partial_keep_dropped_f2p = list(_dropped)
                    self._partial_keep_dropped_details = _kept_failure_details
                    self._partial_keep_dropped_status = {
                        n: str((result.get("per_node_status") or {}).get(n) or "")
                        for n in _dropped
                    }
                    print(
                        f"[synthesize] step attempt-execute: PARTIAL-KEEP  subtask_id={_exe_sid} "
                        f"keep_f2p={len(_passing_f2p)}/{len(_own_f2p)} dropped={_dropped[:5]}",
                        flush=True,
                    )
                    current_attempt["own_f2p"] = _passing_f2p
                    current_attempt["result"] = None
                    current_attempt["enabled"] = True
                    self.write_json(self.output_path, self.trace)
                    # Re-run the executor: the current attempt is now pending (result=None), so it will be re-executed,
                    # keeping only the retained own_f2p nodes, thereby producing ok=True and the correct post_snapshot.
                    self.executor(str(self.output_path), stop_on_fail=False, verbose=True)
                    self.trace = self.load_trace(self.output_path)
                    current_attempt = self.trace["events"][idx]
                    result = dict(current_attempt.get("result") or {})
                if result.get("ok") is not True:
                    failing = list(result.get("failing_node_ids") or [])
                    print(f"[synthesize] step attempt-execute: FAIL  subtask_id={_exe_sid} error_kind={result.get('error_kind')} failing={failing[:3]}", flush=True)
                    # Code execution failure: mark enabled=False and persist; but do not raise immediately, continue to rubric.
                    current_attempt["enabled"] = False
                    self.write_json(self.output_path, self.trace)
                    code_ok = False
            if code_ok:
                print(f"[synthesize] step attempt-execute: ok  subtask_id={_exe_sid}", flush=True)
            event_obj = self.trace["events"][idx]

        # (4) rubric LLM verification: runs unconditionally (even if schema / code already failed), and the verdict is merged
        # with existing failure details and relayed back to the retry model. Skipped when verify_client is None (dry-run / not configured).
        # Also: rubric runs only in the first rubric_max_rounds retry rounds; beyond that (e.g. rounds 3-5 of 5)
        # rubric is skipped, relying only on schema + code execution, to avoid later retries being repeatedly stuck by rubric.
        verdict: dict | None = None
        _rubric_round_ok = self._current_attempt_no <= self.rubric_max_rounds
        if self.verify_client is not None and _rubric_round_ok:
            print(f"[synthesize] step attempt-rubric: starting  subtask_id={sid}", flush=True)
            try:
                verdict = await self._run_rubric_verification(event_obj)
            except Exception as ve:
                # The verifier itself still failed after multiple rounds: synthesize a failed verdict so the feedback can still be relayed to the model.
                msg = f"rubric verifier raised {type(ve).__name__}: {ve}"
                print(f"[synthesize] step attempt-rubric: verifier-error  subtask_id={sid} {msg[:200]}", flush=True)
                verdict = {
                    "alignment_pass": False,
                    "self_contained_pass": False,
                    "overall_pass": False,
                    "alignment_issues": [msg],
                    "self_containedness_issues": [],
                    "diagnostic_feedback_for_retry": msg,
                }
        elif self.verify_client is not None and not _rubric_round_ok:
            print(
                f"[synthesize] step attempt-rubric: skipped  subtask_id={sid} "
                f"attempt={self._current_attempt_no}>rubric_max_rounds={self.rubric_max_rounds} "
                f"(schema+code validation only)",
                flush=True,
            )

        # (5) Merge and persist: fold the rubric verdict + feedback into the current event's result, coexisting with
        # schema/code failure details, ensuring the functional-retry prompt can read all feedback at once.
        self.trace = self.load_trace(self.output_path)
        rubric_ok = (verdict is None) or bool(verdict.get("overall_pass"))
        _extra_details = self._build_partial_keep_failure_details()
        self._merge_attempt_result(
            attempt_index=idx,
            schema_failed=(schema_error is not None),
            code_ok=code_ok,
            verdict=verdict,
            rubric_ok=rubric_ok,
            extra_failure_details=_extra_details,
        )
        self.write_json(self.output_path, self.trace)

        # (6) Verdict: success only when all three gates (schema / code / rubric) pass.
        if schema_error is None and code_ok and rubric_ok:
            print(f"[synthesize] step attempt-rubric: ok  subtask_id={sid}", flush=True)
            return self.trace["events"][idx]

        feedback = AttemptFlow.format_feedback(verdict) if verdict is not None else ""
        # Raise RubricFailed only when the problem is "rubric-only" (so it can be hit by the max_rubric_failures early-accept logic);
        # otherwise (schema / code also failed) raise CodeValidationFailed, to avoid mistakenly accepting a code-failed event.
        if schema_error is None and code_ok and not rubric_ok:
            self._rubric_failure_count += 1
            print(
                f"[synthesize] step rubric-verify: FAIL"
                f"  subtask={self.subtask.get('subtask_id', '?')}"
                f"  alignment={verdict.get('alignment_pass')}"
                f"  self_contained={verdict.get('self_contained_pass')}"
                f"  rubric_failures={self._rubric_failure_count}/{self.max_rubric_failures}"
                f"  issues={feedback!r}",
                flush=True,
            )
            raise RubricFailed(verdict=verdict)
        print(
            f"[synthesize] step attempt-validate: FAIL(combined)"
            f"  subtask={self.subtask.get('subtask_id', '?')}"
            f"  schema_failed={schema_error is not None}"
            f"  code_ok={code_ok}"
            f"  rubric_ok={rubric_ok}"
            f"  issues={feedback!r}",
            flush=True,
        )
        merged_result = dict(self.trace["events"][idx].get("result") or {})
        raise CodeValidationFailed(
            result=merged_result,
            message=(
                f"ATTEMPT validation failed (combined): "
                f"schema_failed={schema_error is not None} code_ok={code_ok} rubric_ok={rubric_ok}"
            ),
        )

    def _merge_attempt_result(
        self, *, attempt_index: int, schema_failed: bool, code_ok: bool,
        verdict: dict | None, rubric_ok: bool, extra_failure_details: list[dict] | None = None,
    ) -> None:
        """Merge schema/code failure details + rubric verdict/feedback back into the current ATTEMPT.result.

        - All three gates pass: only append rubric_verification; ok/enabled keep the code-execution result unchanged.
        - Any gate fails: set ok to False, enabled to False, append the rubric feedback and PARTIAL-KEEP
          failure details into test_failure_details (coexisting with existing schema/code failure details),
          and error_kind takes priority schema > code > rubric.
        """
        events = self.trace.get("events") or []
        if attempt_index >= len(events):
            return
        event = events[attempt_index]
        result = dict(event.get("result") or {})
        if verdict is not None:
            result["rubric_verification"] = verdict
        passed = (not schema_failed) and code_ok and rubric_ok
        if passed:
            event["result"] = result
            return
        details = list(result.get("test_failure_details") or [])
        if verdict is not None and not rubric_ok:
            feedback = AttemptFlow.format_feedback(verdict)
            details.append({
                "node_id": "<rubric_verification>",
                "phase": "rubric_verification",
                "status": "failed",
                "expectation_error": feedback,
                "output_excerpt": feedback,
            })
        for d in (extra_failure_details or []):
            if isinstance(d, dict):
                details.append(dict(d))
        result["test_failure_details"] = details
        result["ok"] = False
        result["failing_node_ids"] = list(result.get("failing_node_ids") or [])
        if schema_failed:
            result["error_kind"] = "schema_validation_failed"
        elif not code_ok:
            if not result.get("error_kind"):
                result["error_kind"] = "code_validation_failed"
        else:  # rubric-only failure
            result["error_kind"] = "rubric_verification_failed"
        event["result"] = result
        event["enabled"] = False

    def normalize_retry_obj(self, obj: Any) -> Any:
        """Best-effort normalize a failed ATTEMPT into a stable object for reference in the retry prompt."""
        raw = super().normalize_retry_obj(obj)
        if not isinstance(raw, dict):
            return raw
        try:
            return self.schema_ops.coerce_event(
                dict(raw), big_task_id=self.big_task_id, subtask=self.subtask, user_query=self.user_query,
                acceptance_criteria=self.acceptance_criteria,
            )
        except Exception:
            return raw

    def _append_generation_failure_event(
        self,
        *,
        attempt_no: int,
        phase: str,
        error: Exception,
        raw_text: str = "",
        raw_obj: Any = None,
    ) -> None:
        """Persist a generation-side failure so retry prompts can see it."""
        if not self.enable_execute or self.write_json is None or self.output_path is None:
            return
        if self.load_trace is not None:
            try:
                self.trace = self.load_trace(self.output_path)
            except Exception:
                pass
        sid = str(self.subtask.get("id") or self.subtask.get("subtask_id") or "")
        err_text = f"{type(error).__name__}: {error}"
        raw_excerpt = raw_text or ""
        if not raw_excerpt and raw_obj is not None:
            try:
                raw_excerpt = json.dumps(raw_obj, ensure_ascii=False, indent=2)
            except Exception:
                raw_excerpt = str(raw_obj)
        if len(raw_excerpt) > 12000:
            raw_excerpt = raw_excerpt[:6000] + "\n...[raw output truncated]...\n" + raw_excerpt[-6000:]
        expectation_error = (
            f"ATTEMPT generation failed before executor; attempt={attempt_no}/{self.max_attempts}; "
            f"phase={phase}; error={err_text}"
        )
        if raw_excerpt:
            expectation_error += "\n\nRaw model/repair output excerpt:\n" + raw_excerpt
        stub = {
            "kind": "ATTEMPT",
            "big_task_id": self.big_task_id,
            "subtask_id": sid,
            "enabled": False,
            "natural_language_goal": str(self.subtask.get("text") or ""),
            "test_patch": [],
            "acceptance_criteria": self.acceptance_criteria or "",
            "own_f2p": [],
            "own_p2p": [],
            "user_query": self.user_query or {},
            "patch": [{"format": "unified_diff", "diff": ""}],
            "rationale": (
                f"Diagnostic event persisted by host because ATTEMPT generation failed "
                f"before a valid executable event was available: {phase}."
            ),
            "result": {
                "ok": False,
                "error_kind": phase,
                "failing_node_ids": [],
                "test_failure_details": [{
                    "node_id": f"<{phase}>",
                    "phase": phase,
                    "status": "failed",
                    "expectation_error": expectation_error,
                    "output_excerpt": raw_excerpt,
                }],
            },
        }
        self.trace.setdefault("events", []).append(stub)
        self.write_json(self.output_path, self.trace)
        print(
            f"[synthesize] step attempt-diagnostic: ok  subtask_id={sid} "
            f"attempt={attempt_no}/{self.max_attempts} phase={phase} "
            f"event_index={len(self.trace.get('events') or []) - 1} "
            f"error={err_text[:300]}",
            flush=True,
        )

    async def run(self) -> tuple[Any | None, str | None]:
        """Override BaseFlow.run() to terminate early when consecutive rubric failures reach max_rubric_failures.

        The rest of the logic is identical to BaseFlow.run(); only an early-exit check is added in the RubricFailed branch.
        """
        last_obj: Any = None
        last_error: Exception | None = None
        for attempt_no in range(1, self.max_attempts + 1):
            raw_obj: Any = None
            self._current_attempt_no = attempt_no
            sid = str(self.subtask.get("id") or self.subtask.get("subtask_id") or "")
            print(
                f"[synthesize] step attempt-impl-loop: starting  "
                f"subtask_id={sid} attempt={attempt_no}/{self.max_attempts} "
                f"last_error={type(last_error).__name__ if last_error else 'none'}",
                flush=True,
            )
            try:
                raw_obj = await self.generate(attempt_no, last_obj, last_error)
                validated = self.quality_validate(raw_obj)
                if isawaitable(validated):
                    validated = await validated
                print(
                    f"[synthesize] step attempt-impl-loop: ok  "
                    f"subtask_id={sid} attempt={attempt_no}/{self.max_attempts}",
                    flush=True,
                )
                return validated, None
            except GeneratedJSONParseError as e:
                print(
                    f"[synthesize] step attempt-json-parse: FAIL  "
                    f"subtask_id={sid} attempt={attempt_no}/{self.max_attempts} "
                    f"error={type(e).__name__}: {str(e)[:300]}",
                    flush=True,
                )
                self._append_generation_failure_event(
                    attempt_no=attempt_no,
                    phase="attempt_json_parse_failed",
                    error=e,
                    raw_text=e.raw_text,
                )
                repaired_obj: Any = None
                try:
                    repaired_obj = await self.try_json_repair(e)
                    print(
                        f"[synthesize] step attempt-json-repair: ok  "
                        f"subtask_id={sid} attempt={attempt_no}/{self.max_attempts}",
                        flush=True,
                    )
                    validated = self.quality_validate(repaired_obj)
                    if isawaitable(validated):
                        validated = await validated
                    print(
                        f"[synthesize] step attempt-impl-loop: ok  "
                        f"subtask_id={sid} attempt={attempt_no}/{self.max_attempts} source=json-repair",
                        flush=True,
                    )
                    return validated, None
                except GeneratedJSONParseError as repair_parse_error:
                    print(
                        f"[synthesize] step attempt-json-repair: FAIL  "
                        f"subtask_id={sid} attempt={attempt_no}/{self.max_attempts} "
                        f"phase=parse error={type(repair_parse_error).__name__}: {str(repair_parse_error)[:300]}",
                        flush=True,
                    )
                    self._append_generation_failure_event(
                        attempt_no=attempt_no,
                        phase="attempt_json_repair_parse_failed",
                        error=repair_parse_error,
                        raw_text=repair_parse_error.raw_text,
                    )
                    last_obj = self.normalize_retry_obj({"_raw": repair_parse_error.raw_text})
                    last_error = repair_parse_error
                except Exception as repair_error:
                    if repaired_obj is None:
                        print(
                            f"[synthesize] step attempt-json-repair: FAIL  "
                            f"subtask_id={sid} attempt={attempt_no}/{self.max_attempts} "
                            f"phase=call error={type(repair_error).__name__}: {str(repair_error)[:300]}",
                            flush=True,
                        )
                        self._append_generation_failure_event(
                            attempt_no=attempt_no,
                            phase="attempt_json_repair_failed",
                            error=repair_error,
                            raw_text=e.raw_text,
                        )
                        last_obj = self.normalize_retry_obj({"_raw": e.raw_text})
                        last_error = repair_error
                        continue
                    print(
                        f"[synthesize] step attempt-json-repair-validate: FAIL  "
                        f"subtask_id={sid} attempt={attempt_no}/{self.max_attempts} "
                        f"error={type(repair_error).__name__}: {str(repair_error)[:300]}",
                        flush=True,
                    )
                    last_obj = self.normalize_retry_obj(repaired_obj)
                    last_error = repair_error
            except RubricFailed as e:
                last_obj = self.normalize_retry_obj(raw_obj)
                last_error = e
                if self._rubric_failure_count >= self.max_rubric_failures:
                    print(
                        f"[synthesize] rubric failure limit reached"
                        f"  subtask={self.subtask.get('subtask_id', '?')}"
                        f"  rubric_failures={self._rubric_failure_count}/{self.max_rubric_failures}"
                        f"  — accepting last rubric-failed attempt as ok=True",
                        flush=True,
                    )
                    # rubric retry limit exceeded: change the ok of this last rubric-failed event to True
                    # and restore enabled to True, then return it as a success. The error metadata was already written
                    # into result by apply_verification_result (rubric_verification /
                    # test_failure_details), kept in the file for later analysis.
                    if self.load_trace is not None and self.output_path is not None:
                        self.trace = self.load_trace(self.output_path)
                    sid = str(self.subtask.get("id") or self.subtask.get("subtask_id") or "")
                    # Find the last event belonging to this subtask with error_kind=rubric_verification_failed
                    accept_idx: int | None = None
                    for i in range(len(self.trace.get("events") or []) - 1, -1, -1):
                        ev = self.trace["events"][i]
                        ev_result = ev.get("result") or {}
                        if (
                            ev.get("kind") == "ATTEMPT"
                            and str(ev.get("subtask_id") or "") == sid
                            and ev_result.get("error_kind") == "rubric_verification_failed"
                        ):
                            accept_idx = i
                            break
                    if accept_idx is not None:
                        ev = self.trace["events"][accept_idx]
                        ev_result = dict(ev.get("result") or {})
                        ev_result["ok"] = True
                        ev["result"] = ev_result
                        ev["enabled"] = True
                        if self.write_json is not None and self.output_path is not None:
                            self.write_json(self.output_path, self.trace)
                        return ev, None
                    # Not found (should not happen in theory): continue on the normal failure path
            except Exception as e:
                print(f"[synthesize] step attempt-impl: FAIL  attempt={attempt_no} subtask_id={sid} error={type(e).__name__}: {str(e)[:200]}", flush=True)
                last_obj = self.normalize_retry_obj(raw_obj)
                last_error = e
        print(
            f"[synthesize] step attempt-impl-loop: exhausted  "
            f"subtask_id={str(self.subtask.get('id') or self.subtask.get('subtask_id') or '')} "
            f"attempts={self.max_attempts} final_error={type(last_error).__name__ if last_error else 'none'}: {str(last_error)[:500] if last_error else ''}",
            flush=True,
        )
        return None, f"{type(last_error).__name__}: {last_error}"


    def _build_partial_keep_failure_details(self) -> list[dict]:
        """Organize the failed F2P nodes trimmed during the PARTIAL-KEEP stage into test_failure_details entries.

        own_f2p is let through when half pass (partial keep), but the detailed error info of the trimmed failed nodes should not
        be discarded. Here they are organized into result.test_failure_details-compatible entries; on rubric failure
        they are folded into result for the functional-retry prompt to reference, providing more repair clues.
        """
        dropped = list(self._partial_keep_dropped_f2p or [])
        if not dropped:
            return []
        kept_details = {
            str(d.get("node_id") or ""): d
            for d in (self._partial_keep_dropped_details or [])
            if isinstance(d, dict)
        }
        out: list[dict] = []
        summary = (
            "This round own_f2p was let through with half passing (PARTIAL-KEEP); the following F2P nodes still did not pass "
            "after applying the model patch and have been trimmed from own_f2p. On rubric failure this failure evidence is provided too, to aid diagnosis:"
            f" dropped={dropped}"
        )
        out.append({
            "node_id": "<partial_keep_dropped_f2p>",
            "phase": "partial_keep_dropped_f2p",
            "status": "failed",
            "expectation_error": summary,
            "output_excerpt": summary,
        })
        for node in dropped:
            d = kept_details.get(node)
            status = str((self._partial_keep_dropped_status or {}).get(node) or "")
            if isinstance(d, dict):
                entry = dict(d)
                entry.setdefault("phase", "partial_keep_dropped_f2p")
                entry["node_id"] = node
                out.append(entry)
            else:
                msg = (
                    f"F2P node {node} had status {status or 'NON-PASSED'} after applying the model patch, "
                    "and was trimmed from own_f2p during the PARTIAL-KEEP stage (no separate test_failure_details record)."
                )
                out.append({
                    "node_id": node,
                    "phase": "partial_keep_dropped_f2p",
                    "status": "failed",
                    "expectation_error": msg,
                    "output_excerpt": msg,
                })
        return out

    async def _run_rubric_verification(self, current_attempt: dict) -> dict:
        """Run the rubric verifier on an ATTEMPT that passed code execution, returning the validated verdict.

        The verify prompt is built internally here (exposing only the current ATTEMPT), so the caller no longer needs to
        pass in `verify_prompt` in advance. If the verifier JSON is still invalid after multiple rounds, it raises, bubbling up to
        `BaseFlow.run()` to be counted as a functional retry.
        """
        if self.verify_client is None or self.workspace is None:
            raise RuntimeError("AttemptFlow verify_client and workspace are required for rubric verification")

        prompt_name = "verify_attempt.txt"
        system, user = self.workspace.build_verify_attempt_current_only_prompt(
            current_attempt=current_attempt,
            category=str(self.subtask.get("category") or "unknown"),
            prompt_name=prompt_name,
            trace=self.trace,
        )
        messages: list[dict[str, Any]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        last_obj: Any = None
        last_error: Exception | None = None
        for attempt_no in range(1, self.verify_max_attempts + 1):
            try:
                if attempt_no > 1:
                    assert last_obj is not None and last_error is not None
                    messages.append({"role": "assistant", "content": json.dumps(last_obj, ensure_ascii=False)})
                    messages.append({
                        "role": "user",
                        "content": self.verify_retry_user_message(
                            last_error, last_obj, attempt_no=attempt_no, max_attempts=self.verify_max_attempts,
                        ),
                    })
                stage = "VERIFY_ATTEMPT" if attempt_no == 1 else "VERIFY_ATTEMPT_VALIDATION_RETRY"
                body = await self.verify_client.chat_once(messages, usage_stage=stage)
                raw_obj = json_from_text(self.verify_client.client_response_text(body))
                return self.validate_verdict(raw_obj)
            except Exception as e:
                last_obj = raw_obj if 'raw_obj' in locals() else None
                if not isinstance(last_obj, dict):
                    last_obj = {"_raw": last_obj}
                last_error = e
        assert last_error is not None
        raise RuntimeError(f"{type(last_error).__name__}: {last_error}")


class AugmentFlow(BaseFlow):
    """Generate augmentation candidates after an ATTEMPT has already passed.

    Prompt organization:
    - The message list for the current augmentation target is prepared in advance by the caller.
    - The augment prompt's scope is narrower than the ATTEMPT main generation: it cares about the current target event,
      rather than synthesizing a new event.
    - The first round is allowed to use `docker_cli`.
    - On retry it attaches the previous augmentation JSON and the host-written failure note; this
      failure may come from schema or from docker-side candidate/test validation.

    Validation contract:
    - The output must contain only the augmentation fields the host expects: `own_p2p` and `test_patch`.
    - This layer only validates the JSON shape; semantic filtering, deduplication, old-test protection, docker candidate
      validation, and attributing "new test_patch broke old tests" vs "the new candidate itself is wrong" are all
      handled in the subsequent caller logic.

    Subsequent error attribution rules:
    - If old tests break after adding the new test_patch, the problem is in the new `test_patch`.
    - If only the new candidate fails, the problem is in the new P2P candidate.
    """

    @staticmethod
    def validate_output(aug_obj: Any) -> dict:
        """Only validate the top-level JSON shape of augmentation, making no semantic judgment."""
        if not isinstance(aug_obj, dict):
            raise ValueError("AUGMENT_TESTS output must be a JSON object")
        return {
            "own_p2p": [
                str(x) for x in (aug_obj.get("own_p2p") or [])
                if isinstance(x, str) and x.strip()
            ],
            "test_patch": [
                str(x) for x in (aug_obj.get("test_patch") or [])
                if isinstance(x, str) and x.strip()
            ],
        }

    @staticmethod
    def retry_reason(debug: dict) -> str:
        """Translate the augment failure reason into an actionable retry note for the model."""
        skipped = str(debug.get("skipped") or "")
        guard_failed = list(debug.get("guard_failed") or [])
        pre_not_failing = list(debug.get("pre_not_failing") or [])
        pre_failed = list(debug.get("pre_failed") or [])
        collect_miss = list(debug.get("collect_miss") or [])
        run_failed = list(debug.get("run_failed") or [])
        unkept = list(debug.get("unkept_defined_tests") or [])
        per_errors: dict[str, str] = dict(debug.get("per_candidate_errors") or {})

        def fmt_per_errors(intro: str) -> str:
            if not per_errors:
                return intro
            lines = [intro]
            for node, reason in per_errors.items():
                lines.append(f"  - {node!r}: {reason}")
            return "\n".join(lines)

        if skipped == "all_overlap":
            duplicates = list(debug.get("duplicates") or [])
            if per_errors:
                return fmt_per_errors(
                    "All of your candidate own_p2p entries duplicate existing tests and add no new nodes. The reason for each duplicate is below:"
                )
            return (
                "The candidate own_p2p you generated completely duplicate existing tests, adding no new nodes. "
                f"The duplicate candidates are: {duplicates}. "
                "Generate brand-new P2P test nodes that do not overlap with the existing own_f2p / own_p2p."
            )
        if skipped == "has_duplicates":
            if per_errors:
                return fmt_per_errors(
                    "Some of your candidate own_p2p entries duplicate existing tests and need to be fixed one by one. The reason for each duplicate is below:"
                )
            return (
                "Some of your candidate own_p2p entries duplicate existing tests. "
                "Generate brand-new P2P test nodes that do not overlap at all with the existing own_f2p / own_p2p."
            )
        if skipped == "smoke_fail":
            return (
                "smoke import failed after applying the patch, which means the test_patch you added introduced a syntax error or a wrong import path. "
                "Check the import statements, class names, and function definitions in test_patch to ensure the file can be parsed normally by Python."
            )
        if skipped == "existing_own_pre_expectation_failed":
            return (
                "This is an existing old test failing, which means the problem is in the test_patch you added, not the new candidate nodes. "
                f"Before applying the patch, these old tests did not meet the pre-fix expectation: {guard_failed}. "
                "Modify the test_patch you added; do not break the pre-fix behavior of the existing own_f2p / own_p2p."
            )
        if skipped == "existing_own_post_expectation_failed":
            return (
                "This is an existing old test failing, which means the problem is in the test_patch you added, not the new candidate nodes. "
                f"After applying the patch, these old tests did not pass: {guard_failed}. "
                "Modify the test_patch you added to ensure the existing own_f2p / own_p2p all still pass after the patch."
            )
        if skipped == "new_test_patch_has_unkept_tests":
            if per_errors:
                return fmt_per_errors(
                    f"The new test_patch defines the following test functions, but they do not appear in the own_p2p candidates; "
                    f"add them to own_p2p, or delete the corresponding definitions from test_patch. The reason for each is below:"
                )
            return (
                "The new test_patch defines new test functions, but these tests did not form final kept candidates. "
                f"The new tests not kept are: {unkept}. Delete the irrelevant tests or add the correct candidate node-ids."
            )
        if skipped == "below_partial_keep_threshold":
            passing = debug.get("passing")
            total = debug.get("total")
            head = (
                f"The number of candidate own_p2p that passed verification this round is {passing}/{total}, "
                "not reaching the partial-keep threshold (need >= half of the total candidates). "
                "Fix the failing candidates below, or add more semantically valid, non-duplicate new candidates, "
                "so that the final passing candidates reach the threshold."
            )
            if per_errors:
                return fmt_per_errors(head + " The specific reason for each failing candidate is below:")
            if collect_miss or run_failed:
                return (
                    head
                    + f" New candidates that cannot be collected: {collect_miss};"
                    + f" new candidates that can be collected but do not pass after the fix: {run_failed}."
                )
            rejected_structural = list(debug.get("rejected_structural") or [])
            if rejected_structural:
                return head + f" Candidates discarded for duplicating existing tests: {rejected_structural}."
            return head
        if skipped == "no candidate after pre filter" or skipped == "candidates_failed":
            if per_errors:
                return fmt_per_errors(
                    "The new candidate nodes you output did not pass verification (note: the existing old tests are all fine, the problem is only in the new candidates)."
                    " The specific failure reason for each candidate is below:"
                )
            return (
                "The new candidate nodes you output are themselves semantically invalid (note: the existing old tests are all fine, the problem is only in the new candidates). "
                f"New F2P candidates that should fail before the fix but actually passed: {pre_not_failing}; "
                f"new P2P candidates that should pass before the fix but actually failed: {pre_failed}. "
                "Fix the candidate node-ids you output or the test_patch that goes with them."
            )
        if skipped == "candidates_failed_post_test":
            if per_errors:
                return fmt_per_errors(
                    "All of the new candidate nodes you output failed post-patch verification (note: the existing old tests are all fine, the problem is only in the new candidates)."
                    " The specific failure reason for each candidate is below:"
                )
            return (
                "The new candidate nodes you output did not pass docker verification (note: the existing old tests are all fine, the problem is only in the new candidates). "
                f"New candidates that cannot be collected: {collect_miss}; "
                f"new candidates that can be collected but do not pass after the fix: {run_failed}. "
                "Fix the candidate node-ids you output or the test_patch that goes with them."
            )
        if collect_miss or run_failed:
            return (
                "The new candidate nodes you output did not pass docker verification (note: the existing old tests are all fine, the problem is only in the new candidates). "
                f"New candidates that cannot be collected: {collect_miss}; "
                f"new candidates that can be collected but do not pass after the fix: {run_failed}. "
                "Fix the candidate node-ids you output or the test_patch that goes with them."
            )
        reason_part = f"reason code: {skipped!r}" if skipped else "no reason code"
        return (
            f"augment docker verification failed ({reason_part}). "
            f"\nDiagnostic info:\n{json.dumps(debug, ensure_ascii=False, indent=2)}"
        )

    @classmethod
    def retry_user_message(cls, debug: dict, bad_json: dict, *, attempt_no: int, max_attempts: int) -> str:
        """Build the retry note for when augment fails schema or docker verification."""
        return (
            "The previous AUGMENT_TESTS output did not pass the host check or docker verification; you need to fix it directly.\n"
            f"This is AUGMENT_TESTS generation attempt {attempt_no}/{max_attempts}; you may still call docker_cli this round.\n"
            f"Failure reason: {cls.retry_reason(debug)}\n"
            "The previous JSON was:\n"
            f"{json.dumps(bad_json, ensure_ascii=False, indent=2)}\n\n"
            "[Mandatory] Before resubmitting, you must call patch_skills to distill this failure into a lesson written into augment_p2p.md, "
            "to avoid future AUGMENT_TESTS repeating the same mistake. WARNING - key requirement: this experience library will be read and reused by "
            "later tasks of [other instances / other repos / other languages], so the lesson must be written to be [general and transferable], and must never be "
            "bound to the specific details of the current task "
            "(do not hardcode specific file names, function names, or domain nouns like ISBN/openlibrary). "
            "Correct way to write it: first use square brackets to state the applicable premise \"[When: <in what scenario / what type of repo / what test framework does this lesson apply>]\", "
            "then write \"-> <what should be done>\"; abstract the failure cause into a general response to a class of problems, rather than restating the details of this one time. "
            "Inherit and refine existing entries rather than piling up duplicates, keeping it faithful and actionable. This step is mandatory on every failure retry.\n\n"
            "Return only one complete, corrected JSON object; no markdown, no explanation, no code block, no extra text before or after."
        )

    def __init__(
        self, *, client, messages: list[dict[str, Any]], trace: dict,
        scope: str, target_event: dict, tool_loop: ToolLoopConfig | None = None,
        max_attempts: int = 3, output_path=None, write_json=None, verbose: bool = True,
        first_call_client=None, json_repair=None,
    ):
        """Save the augment generation-stage dependencies and copy the retryable message list.

        docker semantic filtering, survivor merging, and persistence are now all consolidated into this class's
        `quality_validate(...)`; therefore this additionally holds `output_path` / `write_json`.
        """
        super().__init__(max_attempts=max_attempts, json_repair=json_repair)
        self.client = client
        # The client used for the first 3 main-generation rounds (sglang); from round 4 fall back to the main client (opus).
        self.first_call_client = first_call_client or client
        self.messages = [dict(m) for m in messages]
        self.trace = trace
        self.scope = scope
        self.target_event = target_event
        self.tool_loop = tool_loop or ToolLoopConfig()
        self.output_path = output_path
        self.write_json = write_json
        self.verbose = verbose
        self._last_debug: dict[str, Any] = {}  # the debug of the last docker filter, for writing diagnostics after a permanent failure

    async def generate(self, attempt_no: int, last_obj: Any, last_error: Exception | None) -> Any:
        """Run the augmentation prompt.

        On retry it keeps the same augmentation target unchanged, and feeds the docker-filter failure attribution (from
        `AugmentFiltered.debug`) back to the model.
        """
        if attempt_no > 1:
            assert last_obj is not None and last_error is not None
            debug = last_error.debug if isinstance(last_error, AugmentFiltered) else {"skipped": str(last_error)}
            self.messages.append({"role": "assistant", "content": json.dumps(last_obj, ensure_ascii=False)})
            self.messages.append({
                "role": "user",
                "content": self.retry_user_message(
                    debug,
                    last_obj,
                    attempt_no=attempt_no,
                    max_attempts=self.max_attempts,
                ),
            })
        _aug_sid = str(self.target_event.get("subtask_id") or "-")
        _aug_bid = str(self.target_event.get("big_task_id") or "")
        print(f"[synthesize] step augment-generate: starting  attempt={attempt_no} scope={self.scope} big_task_id={_aug_bid} subtask_id={_aug_sid}", flush=True)
        # Two-stage cascade: rounds 1-3 use sglang (first_call_client), from round 4 fall back to the main client (opus).
        gen_client = (
            self.first_call_client if attempt_no <= 3
            else self.client
        )
        _first_round = attempt_no <= 3
        return await gen_client.call_json_with_docker_cli_messages(
            self.messages,
            trace=self.trace,
            enable_docker_cli=self.tool_loop.enable_docker_cli,
            event_kind="AUGMENT_TESTS",
            max_tool_calls=(
                (self.tool_loop.first_round_max_tool_calls or self.tool_loop.max_tool_calls)
                if _first_round else self.tool_loop.max_tool_calls
            ),
            remind_every_tool_calls=(
                (self.tool_loop.first_round_remind_every_tool_calls or self.tool_loop.remind_every_tool_calls)
                if _first_round else self.tool_loop.remind_every_tool_calls
            ),
            remind_start_tool_calls=(
                self.tool_loop.first_round_remind_start_tool_calls if _first_round else 0
            ),
            wrap_up_reminder=bool(_first_round and self.tool_loop.first_round_remind_start_tool_calls),
            reminder_context=(
                f"{self.scope} {self.target_event.get('big_task_id')}/"
                f"{self.target_event.get('subtask_id') or '-'}"
            ),
            docker_cli_session_cls=self._make_augment_session_cls(),
        )

    def _make_augment_session_cls(self):
        """Build the augment-specific docker_cli session class.

        The baseline state required for augment generation:
          base_commit + history (stop_before the current ATTEMPT) with successfully applied model patches
          + the current ATTEMPT's test_patch (so the model sees existing tests, avoiding rewrites/conflicts)
          but **excluding** the current ATTEMPT's model patch (the `patch` field).

        This way, when generating new P2P, the model sees the code state before the patch was applied, making it easier to identify
        existing behaviors "already passing before the patch", consistent with the pre-check stage of docker_filter_extras.

        - `_replay_patches()` is overridden, using stop_before_event=the current ATTEMPT.
        - Historical test_patches are still not applied; if the current ATTEMPT has a test_patch, it is additionally applied.
        - After reset_worktree the session is cleared, and the next start() re-applies, ensuring the environment
          after reset is consistent with the initial environment.
        - The test-side flow (docker_filter_extras) is unaffected, still following the original apply order.
        """
        test_patch_parts = AugmentSupport.normalize_test_patch_list(self.target_event.get("test_patch"))
        from local_test_executor import apply_eval_test_patch, replay_model_patches

        base_cls = type(self.client).GenerationDockerCliSession
        _tp = list(test_patch_parts)
        _trace = self.trace
        _target_event = self.target_event
        _bid = self.target_event.get("big_task_id")
        _sid = self.target_event.get("subtask_id") or "-"

        class _AugmentDockerCliSession(base_cls):
            def _replay_patches(self) -> list:
                # Only replay the historical successful patches before the current ATTEMPT, excluding the current ATTEMPT's model patch.
                return replay_model_patches(
                    self.session, self.trace,
                    stop_before_event=_target_event,
                    verbose=False,
                )

            def start(self) -> None:
                was_none = self.session is None
                super().start()
                if was_none and _tp:
                    inst = _trace.get("instance", {}) or {}
                    apply_eval_test_patch(
                        self.session,
                        test_patch=_tp,
                        base_commit=inst.get("base_commit", ""),
                        cwd=self.cwd,
                    )
                    print(
                        f"[synthesize] step augment-generate: applied current test_patch to "
                        f"pre-patch worktree  big_task_id={_bid} subtask_id={_sid}",
                        flush=True,
                    )

        return _AugmentDockerCliSession

    def quality_validate(self, obj: Any) -> Any:
        """augment's single validation entry point: schema shape -> structural filter -> docker semantic filter -> survivor merge and persist.

        - schema shape error -> raise `ValueError` (retried by `BaseFlow.run`).
        - no candidate after structural filter -> return normally (treated as nothing to augment, no more retries).
        - no survivor after docker filter -> raise `AugmentFiltered(debug=...)` (retried by `BaseFlow.run`,
          the next generate reads debug and regenerates candidates).
        - has survivor -> merge back into `target_event` and persist, return a result summary.
        """
        try:
            aug_obj = self.validate_output(obj)
        except Exception as schema_err:
            # schema validation failure: record it in _last_debug, ensuring it can be written into test_failure_details on a permanent failure.
            self._last_debug = {"skipped": f"schema_error: {schema_err}"}
            raise
        bid = self.target_event.get("big_task_id")
        sid = self.target_event.get("subtask_id") or "-"

        raw_p2p = list(aug_obj["own_p2p"])
        raw_test_patch = list(aug_obj["test_patch"])
        merged_test_patch = AugmentSupport.dedup_patch_fragments(
            AugmentSupport.normalize_test_patch_list(self.target_event.get("test_patch")) + raw_test_patch
        )

        _f2p_clean, p2p_clean, rejected_reasons = AugmentSupport.filter_extras_against_known(
            self.trace, self.target_event, self.trace.get("instance") or {},
            candidates_f2p=[], candidates_p2p=raw_p2p,
        )
        # Structural filter: rejected candidates (exact node-id duplicates or self-duplicates) are discarded directly, no longer all-or-nothing.
        # Whether to accept "partial keep" is decided later by the unified threshold (passing count >= half of the total candidates).
        if not p2p_clean:
            if self.verbose:
                print(f"[augment] {self.scope} {bid}/{sid}: no candidate after structural filter (all overlap)")
            _overlap_debug: dict[str, Any] = {"skipped": "all_overlap", "duplicates": raw_p2p}
            self._last_debug = _overlap_debug
            raise AugmentFiltered(debug=_overlap_debug)

        if self.verbose:
            print(f"[augment] {self.scope} {bid}/{sid}: {len(raw_p2p)} raw p2p, "
                  f"{len(p2p_clean)} p2p candidates -> running docker pytest filter")

        survivors_f2p, survivors_p2p, debug = AugmentSupport.docker_filter_extras(
            self.trace, self.target_event,
            candidates_f2p=[],
            candidates_p2p=p2p_clean,
            candidate_test_patch=merged_test_patch,
            include_prior_regression=False,
            verbose=self.verbose,
        )
        # No survivor after docker filter, retry directly.
        if not survivors_p2p:
            print(
                f"[synthesize] step augment-partial-keep: NONE  {self.scope} {bid}/{sid} "
                f"pass=0/{len(raw_p2p)} reason={AugmentFlow.retry_reason(debug)}",
                flush=True,
            )
            self._last_debug = debug
            raise AugmentFiltered(debug=debug)

        # Partial-keep threshold: the denominator is the model's total candidate count raw_p2p, and the numerator is the final survivors.
        # As long as the passing count reaches half of the total candidates, accept partial keep; otherwise retry, asking the model to regenerate.
        _total_p2p = len(raw_p2p)
        _passing_p2p = len(survivors_p2p)
        _augment_status = "ok"
        if _passing_p2p < _total_p2p:
            if not (_passing_p2p * 2 >= _total_p2p):
                print(
                    f"[synthesize] step augment-partial-keep: BELOW-THRESHOLD  {self.scope} {bid}/{sid} "
                    f"pass={_passing_p2p}/{_total_p2p} (need >=half); retrying",
                    flush=True,
                )
                _thresh_debug: dict[str, Any] = {
                    **debug,
                    "skipped": "below_partial_keep_threshold",
                    "passing": _passing_p2p,
                    "total": _total_p2p,
                    "rejected_structural": list(rejected_reasons.keys()),
                }
                self._last_debug = _thresh_debug
                raise AugmentFiltered(debug=_thresh_debug)
            _augment_status = "partial_ok"
            print(
                f"[synthesize] step augment-partial-keep: ACCEPTED  {self.scope} {bid}/{sid} "
                f"pass={_passing_p2p}/{_total_p2p} dropped={[n for n in raw_p2p if n not in set(survivors_p2p)][:5]}",
                flush=True,
            )
        else:
            print(
                f"[synthesize] step augment-partial-keep: ALL-PASS  {self.scope} {bid}/{sid} "
                f"pass={_passing_p2p}/{_total_p2p}",
                flush=True,
            )

        # Has survivor: first merge back into the event and persist.
        if self.scope == "PLAN":
            cur_f2p, cur_p2p = at.plan_own_tests(self.target_event)
        else:
            cur_f2p, cur_p2p = at.attempt_own_tests(self.target_event)
        self.target_event["test_patch"] = merged_test_patch
        self.target_event["own_f2p"] = list(cur_f2p)
        self.target_event["own_p2p"] = AugmentSupport.dedup_keep_order(list(cur_p2p) + survivors_p2p)
        ev_result = dict(self.target_event.get("result") or {})
        ev_result["augment_status"] = _augment_status
        self.target_event["result"] = ev_result
        self.write_json(self.output_path, self.trace)
        if self.verbose:
            print(f"[augment]   appended {len(survivors_p2p)} p2p to event")

        # unkept_defined_tests: the survivors are persisted, but test_patch still has undeclared functions.
        # Raise after persisting the survivors, so the model adds the missed functions into own_p2p next round;
        # the persist action is not rolled back -- on retry, filter_extras_against_known folds the just-written
        # own_p2p nodes into forbidden, avoiding the model re-declaring already-merged survivors.
        unkept = list(debug.get("unkept_defined_tests") or [])
        if unkept:
            if self.verbose:
                print(
                    f"[augment] {self.scope} {bid}/{sid}: "
                    f"{len(unkept)} unkept defined test(s) after merge; raising for retry"
                )
            self._last_debug = debug
            raise AugmentFiltered(debug=debug)

        return {"added_f2p": survivors_f2p, "added_p2p": survivors_p2p, **debug}

    def normalize_retry_obj(self, obj: Any) -> Any:
        """Even if the model returns a non-dict, wrap it into a JSON-friendly shape that can enter the retry prompt."""
        raw = super().normalize_retry_obj(obj)
        if isinstance(raw, dict):
            return raw
        return {"_raw": raw}


class AugmentOrchestrator:
    """Responsible for the current event-level augmentation loop and retry behavior.

    Most of the augment-specific prompt organization and docker validation logic has already been pushed down into
    `AugmentSupport`. This mainly retains only three things:
    - retry boundary control
    - merging survivors back into the event
    - trace persistence
    """

    @classmethod
    async def run(
        cls,
        *,
        client,
        trace: dict,
        target_event: dict,
        scope: str,
        output_path,
        verbose: bool,
        write_json,
        max_tool_calls: int,
        remind_every_tool_calls: int,
        first_round_max_tool_calls: int = 0,
        first_round_remind_start_tool_calls: int = 0,
        first_round_remind_every_tool_calls: int = 0,
        max_attempts: int = 3,
        first_call_client=None,
        json_repair=None,
    ) -> dict:
        """Only do eligibility judgment + assemble prompt + call `AugmentFlow.run()`.

        Generation, schema validation, docker semantic filtering, survivor merging, and persistence are all consolidated into
        `AugmentFlow.quality_validate(...)`; this method no longer carries any validation or write-back.
        """
        if scope not in {"PLAN", "ATTEMPT"}:
            raise ValueError(f"scope must be PLAN or ATTEMPT, got {scope!r}")
        if scope == "ATTEMPT":
            res = target_event.get("result") or {}
            if res.get("ok") is not True:
                return {"skipped": "event not ok"}
            category = AugmentSupport.attempt_blueprint_category(trace, target_event)
            own_f2p, own_p2p = at.attempt_own_tests(target_event)
            recent_patches_summary: list[dict] = []
        else:
            category = "request_or_instruction"
            own_f2p, own_p2p = at.plan_own_tests(target_event)
            recent_patches_summary = [
                AugmentSupport.summarize_attempt_patch(ev.get("patch"))
                for ev in (trace.get("events") or [])
                if ev.get("kind") == "ATTEMPT"
                and at.is_enabled(ev)
                and ev.get("big_task_id") == target_event.get("big_task_id")
                and (ev.get("result") or {}).get("ok") is True
            ]
        augment_payload = {
            "test_runner": client.workspace.test_runner_info(trace),
            "patch": AugmentSupport.augment_patch_payload(
                trace, target_event, scope=scope,
            ),
            "task_summary": AugmentSupport.augment_task_summary(
                target_event,
                scope=scope,
                recent_patches_summary=recent_patches_summary,
            ),
            "category": category,
            "current_test_patch": AugmentSupport.normalize_test_patch_list(target_event.get("test_patch")),
            "current_own_f2p": list(own_f2p),
            "current_own_p2p": list(own_p2p),
            "all_declared_own": list(AugmentSupport.all_declared_own(trace)),
        }
        system, user = client.workspace.build_augment_prompt(
            payload=augment_payload, prompt_name="augment_p2p.txt",
        )
        messages: list[dict] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        flow = AugmentFlow(
            client=client,
            first_call_client=first_call_client,
            messages=messages,
            trace=trace,
            scope=scope,
            target_event=target_event,
            tool_loop=ToolLoopConfig(
                enable_docker_cli=True,
                max_tool_calls=max_tool_calls,
                remind_every_tool_calls=remind_every_tool_calls,
                first_round_max_tool_calls=first_round_max_tool_calls,
                first_round_remind_start_tool_calls=first_round_remind_start_tool_calls,
                first_round_remind_every_tool_calls=first_round_remind_every_tool_calls,
            ),
            max_attempts=max_attempts,
            output_path=output_path,
            write_json=write_json,
            verbose=verbose,
            json_repair=json_repair,
        )
        initial_p2p_count = len(own_p2p)
        result, error = await flow.run()
        if result is None:
            # augment exhausted all retries: augment is treated as an optional enhancement and does not change the target event's success semantics.
            # Only record diagnostics (augment_status + test_failure_details) for later auditing, and continue to the next subtask.
            last_debug = flow._last_debug
            current_p2p_count = len(target_event.get("own_p2p") or [])
            ev_result = dict(target_event.get("result") or {})
            ev_result["augment_status"] = (
                "partial_ok_exhausted" if current_p2p_count > initial_p2p_count else "failed_empty"
            )
            ev_result["augment_error"] = error or "augment failed"
            if last_debug:
                existing_tfd: list[dict] = list(ev_result.get("test_failure_details") or [])
                existing_tfd.append({
                    "node_id": "<augment_docker_filter>",
                    "phase": "augment_docker_filter",
                    "status": "failed",
                    "expectation_error": AugmentFlow.retry_reason(last_debug),
                    "output_excerpt": json.dumps(last_debug, ensure_ascii=False),
                })
                ev_result["test_failure_details"] = existing_tfd
            target_event["result"] = ev_result
            write_json(output_path, trace)
            print(
                f"[synthesize] step augment-exhausted: KEEP-EVENT  "
                f"big_task_id={target_event.get('big_task_id')} subtask_id={target_event.get('subtask_id') or '-'} "
                f"own_p2p={len(target_event.get('own_p2p') or [])} ok=unchanged reason={error or 'augment failed'}",
                flush=True,
            )
            return {"added_f2p": [], "added_p2p": [], "skipped": error or "augment failed"}
        return result
