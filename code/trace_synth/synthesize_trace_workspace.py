from __future__ import annotations

import json
import os
import random
import re
import ast
from pathlib import Path
from typing import Any

import agent_trace as at


# ── new_task_angle T=2 softmax-smoothed sampling distribution ────────────────
# Macro-avg percentages with override (61.7% -> 19.7%) then softmax T=2.
# Source: classify/distribution_report.md  §New Task Angle T=2 column.
#
# 20-category pool (was 16). Four angles that the classifier previously folded
# into neighbours via analyze_distribution.py DIM_REMAP_LABELS were broken out
# as independent categories (business-rule/algorithm, query/search, automation,
# external-integration). Their raw macro share is tiny, so after softmax the two
# lowest (business-rule, external-integration) were floored to 1.14% each and the
# whole column re-normalised to 100% to keep them samplable by PLAN.
_ANGLE_LABELS_T2: list[tuple[str, float]] = [
    ("Feature addition and capability expansion",                    10.35),
    ("Defect fixing and semantic correctness alignment",             10.26),
    ("Support range expansion, compatibility adaptation, and technical migration",        9.22),
    ("Architecture refactoring, abstraction decoupling, and code structure organization",         7.00),
    ("Authentication, authorization, security, and compliance governance",              6.75),
    ("User interface, pages, and interaction development",                6.11),
    ("Configuration management, load priority, and runtime effect governance",     6.00),
    ("Persistence, caching, data consistency, and migration governance",       5.47),
    ("Data ingestion, parsing, extraction, cleaning, and transformation",         5.46),
    ("Observability, logging, monitoring, and auditing",              4.03),
    ("Automation scripts, workflows, scheduling, and background execution",          4.01),
    ("Environment setup, build deployment, and runtime infrastructure",         3.97),
    ("Query, filtering, search, retrieval, and statistics capability building",          3.76),
    ("Testing, evaluation, verification, and quality gates",              3.69),
    ("Documentation, specifications, and knowledge preservation",             3.09),
    ("Repository collaboration, version, and release governance",                2.98),
    ("Performance, stability, concurrency, and resource optimization",             2.94),
    ("Interface contracts, data models, and parameter/naming semantic adjustments",    2.64),
    ("Business rules, algorithm strategies, and state logic adjustments",          1.14),
    ("External systems, platforms, SDKs, and model service integration",           1.14),
]


_ANGLE_LABEL_DEFS: dict[str, str] = {
    "Defect fixing and semantic correctness alignment":
        "Fix behavioral errors, boundary defects, type errors, or semantic inconsistencies in existing code, strictly aligning the actual behavior of functions/modules with their specifications, documentation, or user expectations.",
    "Feature addition and capability expansion":
        "Introduce entirely new feature points, new parameters, new options, or extended coverage scenarios for existing interfaces in the current system, while maintaining backward compatibility of existing behavior.",
    "Support range expansion, compatibility adaptation, and technical migration":
        "Broaden the system's support for new versions, new platforms, new language features, or new runtime environments; or migrate legacy code to newer framework/library versions, eliminating deprecated API usage.",
    "Configuration management, load priority, and runtime effect governance":
        "Adjust configuration item definitions, loading order, and override priorities, ensuring that various sources (environment variables, files, CLI, defaults) take effect as expected at runtime.",
    "Architecture refactoring, abstraction decoupling, and code structure organization":
        "Reorganize module boundaries, introduce abstraction layers, extract reusable components, or eliminate circular dependencies, optimizing code structure and maintainability without changing external behavior.",
    "Environment setup, build deployment, and runtime infrastructure":
        "Configure build scripts, CI/CD pipelines, container images, dependency management, or local development environments, ensuring the project can be built, tested, and deployed reproducibly.",
    "Testing, evaluation, verification, and quality gates":
        "Add or improve test cases, benchmarks, end-to-end validation, or code quality check rules, increasing coverage and establishing clear quality baselines.",
    "User interface, pages, and interaction development":
        "Develop or improve frontend pages, UI components, interaction flows, or user experience, including layout adjustments, style optimizations, and interaction logic refinements.",
    "Performance, stability, concurrency, and resource optimization":
        "Identify and eliminate performance bottlenecks, memory leaks, race conditions, or resource waste, enabling the system to run stably and efficiently under high load, large data volumes, or high-concurrency scenarios.",
    "Documentation, specifications, and knowledge preservation":
        "Write or update code comments, API documentation, design specifications, usage tutorials, or contribution guides, helping developers understand system behavior and decision rationale.",
    "Repository collaboration, version, and release governance":
        "Standardize branching strategies, version number management, Changelog maintenance, release processes, or dependency version locking, ensuring version history is clear and traceable during multi-person collaboration.",
    "Interface contracts, data models, and parameter/naming semantic adjustments":
        "Modify API parameter names, data structure fields, return types, or protocol formats, making interface contracts more accurately reflect business semantics while handling backward compatibility issues.",
    "Authentication, authorization, security, and compliance governance":
        "Implement or fix identity authentication, access control, key management, or security vulnerabilities, ensuring the system meets security policies and compliance requirements.",
    "Persistence, caching, data consistency, and migration governance":
        "Handle database schema changes, cache invalidation strategies, transaction consistency, or data migration scripts, ensuring the reliability of the persistence layer as business requirements evolve.",
    "Data ingestion, parsing, extraction, cleaning, and transformation":
        "Implement reading of data from external sources, format parsing, field extraction, noise filtering, or structural transformation, providing normalized data that meets downstream processing expectations.",
    "Observability, logging, monitoring, and auditing":
        "Introduce or improve log recording, metric instrumentation, distributed tracing, or audit logs, making the system's runtime state observable and diagnosable in production environments.",
    "Automation scripts, workflows, scheduling, and background execution":
        "Codify manual or ad-hoc steps into scripts, jobs, pipelines, cron/scheduled tasks, background daemons, or repeatable automation chains, enabling unattended, batch, or long-running execution.",
    "Query, filtering, search, retrieval, and statistics capability building":
        "Strengthen retrieval, filtering, sorting, pagination, aggregation, search, matching, or statistics over data or content, making it easier for users or upstream systems to find the results they need.",
    "Business rules, algorithm strategies, and state logic adjustments":
        "Change decision rules, computation formulas, sorting/routing/allocation strategies, thresholds, state machines, or business semantics so outputs follow a new business or mathematical definition.",
    "External systems, platforms, SDKs, and model service integration":
        "Integrate external platforms, third-party services, open APIs, SDKs, message channels, cloud capabilities, or AI/model services, completing authentication, request wrapping, data adaptation, and call-chain wiring.",
}


def _sample_angle_candidates(k: int = 4) -> str:
    """Weighted sample k labels (without replacement) from _ANGLE_LABELS_T2.

    Returns a newline-separated bullet list with definitions, e.g.:
        - Defect fixing and semantic correctness alignment: fix behavioral errors in existing code...
        - Feature addition and capability expansion: introduce in the current system...
        ...
    """
    labels = [lbl for lbl, _ in _ANGLE_LABELS_T2]
    weights = [w for _, w in _ANGLE_LABELS_T2]
    chosen: list[str] = []
    pool_labels = list(labels)
    pool_weights = list(weights)
    k = min(k, len(pool_labels))
    for _ in range(k):
        total = sum(pool_weights)
        r = random.random() * total
        cumulative = 0.0
        idx = 0
        for i, w in enumerate(pool_weights):
            cumulative += w
            if r <= cumulative:
                idx = i
                break
        chosen.append(pool_labels[idx])
        pool_labels.pop(idx)
        pool_weights.pop(idx)
    return "\n".join(
        f"  - {lbl}: {_ANGLE_LABEL_DEFS[lbl]}" for lbl in chosen
    )


class TraceWorkspace:
    """Responsible for trace read/write, prompt input assembly, history slicing, and payload organization.

    This object is the boundary between the "raw trace storage" and the "projected view the prompt sees".
    Different phases intentionally see history at different granularities:
    - PLAN sees the full enabled history across BigTasks
    - the ATTEMPT-series prompts only see the current BigTask + the current subtask's failure_brief
    - AUGMENT sees the finer-grained target-centric payload the caller prepared in advance

    Concentrating these rules here avoids scattering ad-hoc history-slicing logic throughout the top-level main loop.
    """

    _SYSTEM_BLOCK_RE = re.compile(
        r"\[SYSTEM\]\s*(.*?)\s*\[/SYSTEM\]\s*\[USER\]\s*(.*?)\s*\[/USER\]\s*\Z",
        re.DOTALL,
    )
    _IF_REASON_RE = re.compile(
        r"\{\{#if_reason\s+([A-Za-z0-9_-]+)\}\}(.*?)\{\{/if_reason\}\}",
        re.DOTALL,
    )
    _UNLESS_REASON_RE = re.compile(
        r"\{\{#unless_reason\s+([A-Za-z0-9_-]+)\}\}(.*?)\{\{/unless_reason\}\}",
        re.DOTALL,
    )
    _IF_FAILURE_BRIEF_EMPTY_RE = re.compile(
        r"\{\{#if_failure_brief_empty\}\}(.*?)\{\{/if_failure_brief_empty\}\}",
        re.DOTALL,
    )
    _IF_FAILURE_BRIEF_NONEMPTY_RE = re.compile(
        r"\{\{#if_failure_brief_nonempty\}\}(.*?)\{\{/if_failure_brief_nonempty\}\}",
        re.DOTALL,
    )
    _IF_CATEGORY_RE = re.compile(
        r"\{\{#if_category\s+([A-Za-z0-9_-]+)\}\}(.*?)\{\{/if_category\}\}",
        re.DOTALL,
    )
    _IF_TEST_RUNNER_RE = re.compile(
        r"\{\{#if_test_runner\s+([A-Za-z0-9_-]+)\}\}(.*?)\{\{/if_test_runner\}\}",
        re.DOTALL,
    )

    def __init__(self, *, prompt_dir: str | Path, schema_version: str):
        """Record the prompt directory and the current trace schema version."""
        self.prompt_dir = Path(prompt_dir)
        self.schema_version = schema_version

    def read_json(self, path: str | Path) -> Any:
        """Read a JSON file."""
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def write_json(self, path: str | Path, obj: Any) -> None:
        """Atomically write an object as a UTF-8 JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            errors="backslashreplace",
        )
        os.replace(tmp, path)

    def node_list(self, raw: Any) -> list[str]:
        """Normalize the test-node fields in an instance into a uniform list[str]."""
        if raw is None:
            return []
        if isinstance(raw, list):
            return [str(x) for x in raw]
        if isinstance(raw, str):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(raw)
                    if isinstance(parsed, list):
                        return [str(x) for x in parsed]
                except Exception:
                    pass
            return [raw]
        return []

    def instance_subset(self, instance: dict) -> dict:
        """Extract the subset of the instance that actually needs to be persisted into the trace."""
        keys = [
            "repo", "instance_id", "base_commit", "problem_statement", "hints_text",
            "version", "difficulty", "FAIL_TO_PASS", "PASS_TO_PASS", "patch", "test_patch",
            "dockerhub_tag", "repo_language", "source", "before_repo_set_cmd",
            "selected_test_files_to_run",
        ]
        out = {k: instance[k] for k in keys if k in instance}
        fail_to_pass = instance.get("FAIL_TO_PASS")
        if fail_to_pass is None:
            fail_to_pass = instance.get("fail_to_pass")
        pass_to_pass = instance.get("PASS_TO_PASS")
        if pass_to_pass is None:
            pass_to_pass = instance.get("pass_to_pass")
        out.setdefault("FAIL_TO_PASS", fail_to_pass)
        out.setdefault("PASS_TO_PASS", pass_to_pass)
        out["FAIL_TO_PASS_parsed"] = self.node_list(fail_to_pass)
        out["PASS_TO_PASS_parsed"] = self.node_list(pass_to_pass)
        return out

    def init_trace(
        self, instance: dict, *, image: str, pytest_bin: str, cwd: str, smoke_import: str | None,
    ) -> dict:
        """Construct a fresh empty trace based on the raw instance."""
        from swe_bench_pro_runner import is_pro_instance

        runner_kind = (
            "pro_official"
            if is_pro_instance(instance, image=image)
            else "pytest"
        )
        return {
            "schema": self.schema_version,
            "instance": self.instance_subset(instance),
            "config": {
                "image": image,
                "pytest_bin": pytest_bin,
                "cwd": cwd,
                "smoke_import": smoke_import,
                "test_runner_kind": runner_kind,
            },
            "state": {"current_phase": "INIT", "terminal": None},
            "events": [],
        }

    def is_trace_payload(self, obj: Any) -> bool:
        """Decide whether the input object is already a trace payload rather than a raw instance."""
        return (
            isinstance(obj, dict)
            and isinstance(obj.get("instance"), dict)
            and isinstance(obj.get("config"), dict)
            and isinstance(obj.get("events"), list)
        )

    def prepare_resume_trace(
        self, trace: dict, *, image: str, pytest_bin: str, cwd: str, smoke_import: str | None,
    ) -> dict:
        """Bring an existing trace into a state that can be resumed."""
        trace = json.loads(json.dumps(trace, ensure_ascii=False))
        trace.setdefault("schema", self.schema_version)
        trace.setdefault("events", [])
        trace.setdefault("instance", {})
        cfg = trace.setdefault("config", {})
        cfg["image"] = image or cfg.get("image")
        cfg["pytest_bin"] = pytest_bin or cfg.get("pytest_bin")
        cfg["cwd"] = cwd or cfg.get("cwd")
        cfg["smoke_import"] = smoke_import
        from swe_bench_pro_runner import is_pro_instance
        cfg["test_runner_kind"] = (
            "pro_official"
            if is_pro_instance(trace.get("instance"), image=cfg.get("image"))
            else "pytest"
        )
        state = trace.setdefault("state", {})
        state["current_phase"] = "RESUME"
        state["terminal"] = None
        state.pop("last_error", None)
        return trace

    def load_prompt(self, name: str) -> str:
        """Read a template text from the prompt directory."""
        return (self.prompt_dir / name).read_text(encoding="utf-8").strip()

    def render_reason_conditionals(self, text: str, payload: dict) -> str:
        """Render the `if_reason/unless_reason` conditional blocks in the template."""
        reason = str((payload.get("decision") or {}).get("reason") or "")

        def repl(match: re.Match[str]) -> str:
            expected = match.group(1)
            body = match.group(2).strip()
            return body if expected == reason else ""

        text = self._IF_REASON_RE.sub(repl, text)

        def unless_repl(match: re.Match[str]) -> str:
            expected = match.group(1)
            body = match.group(2).strip()
            return "" if expected == reason else body

        return self._UNLESS_REASON_RE.sub(unless_repl, text)

    def render_failure_brief_conditionals(self, text: str, payload: dict) -> str:
        """Render the corresponding conditional block based on whether failure_brief is empty."""
        events = ((payload.get("failure_brief") or {}).get("events") or [])
        has_failure = bool(events)
        text = self._IF_FAILURE_BRIEF_EMPTY_RE.sub(
            lambda m: m.group(1).strip() if not has_failure else "",
            text,
        )
        return self._IF_FAILURE_BRIEF_NONEMPTY_RE.sub(
            lambda m: m.group(1).strip() if has_failure else "",
            text,
        )

    def render_category_conditionals(self, text: str, payload: dict) -> str:
        """Render the branch conditional blocks in the template based on category."""
        category = str(payload.get("category") or "")
        return self._IF_CATEGORY_RE.sub(
            lambda m: m.group(2).strip() if m.group(1) == category else "",
            text,
        )

    def render_test_runner_conditionals(self, text: str, payload: dict) -> str:
        """Render the branch conditional blocks in the template based on the test runner type."""
        runner = payload.get("test_runner") or {}
        kind = str(runner.get("kind") or "")
        return self._IF_TEST_RUNNER_RE.sub(
            lambda m: m.group(2).strip() if m.group(1) == kind else "",
            text,
        )

    def render_prompt(self, name: str, payload: dict, *, extras: dict | None = None) -> tuple[str | None, str]:
        """Render the template, payload, and extra substitutions together into a `(system, user)` prompt."""
        text = self.load_prompt(name).replace(
            "{{INPUT_JSON}}",
            json.dumps(payload, ensure_ascii=False, indent=2),
        )
        if extras:
            for key, value in extras.items():
                text = text.replace("{{" + key + "}}", value)
        text = self.render_reason_conditionals(text, payload)
        text = self.render_failure_brief_conditionals(text, payload)
        text = self.render_category_conditionals(text, payload)
        text = self.render_test_runner_conditionals(text, payload)
        m = self._SYSTEM_BLOCK_RE.search(text)
        if m:
            return (m.group(1).strip() or None, m.group(2).strip())
        return None, text

    def test_runner_info(self, trace: dict | None) -> dict:
        """Generate prompt-readable test runner metadata.

        Verified uses pytest node-ids; SWE-bench Pro uses the official per-instance
        run_script.sh/parser.py. The two have different test identifier formats and
        validation methods, so the prompt must branch explicitly to avoid the model
        treating every instance as pytest.
        """
        trace = trace or {}
        inst = trace.get("instance") or {}
        cfg = trace.get("config") or {}
        image = str(cfg.get("image") or "")
        from swe_bench_pro_runner import is_pro_instance
        is_pro = is_pro_instance(
            inst,
            image=image,
            runner_kind=cfg.get("test_runner_kind"),
        )
        if is_pro:
            return {
                "kind": "pro_official",
                "instance_id": str(inst.get("instance_id") or ""),
                "repo_cwd": str(cfg.get("cwd") or "/app"),
                "repo_language": str(inst.get("repo_language") or ""),
                "identifier_format": (
                    "Use exact names emitted by this instance's official parser.py. "
                    "Common formats include Go test function names like TestFoo, "
                    "JS/TS names like path/to/test_file.ts | suite | test, and "
                    "pytest-style node ids only when that Pro parser emits them."
                ),
                "execution_model": (
                    "The executor applies test_patch, copies "
                    "swe_bench_pro_official/run_scripts/<instance_id>/run_script.sh "
                    "and parser.py into /workspace, runs run_script.sh, then validates "
                    "own_f2p/own_p2p against parser.py output names."
                ),
            }
        return {
            "kind": "pytest",
            "instance_id": str(inst.get("instance_id") or ""),
            "repo_cwd": str(cfg.get("cwd") or "/testbed"),
            "repo_language": str(inst.get("repo_language") or "python"),
            "identifier_format": (
                "Use full pytest node ids such as "
                "path/to/test_file.py::TestClass::test_method or "
                "path/to/test_file.py::test_func[param]."
            ),
            "execution_model": (
                "The executor temporarily applies test_patch, runs pytest collect/run "
                "on the declared node ids, then resets the test patch files."
            ),
        }

    # Character cap for a single failure text (expectation_error / output_excerpt) before it enters the prompt.
    # For large repos (e.g. teleport), a single `go test ./...` full stdout can reach 250K characters, and
    # expectation_error and output_excerpt are often verbatim identical, so accumulating N failures can blow
    # the prompt past the 1M token limit. Here we slim it in two layers: (1) dedup repeated fields; (2) head/tail-preserving truncation per entry.
    _FAILURE_TEXT_LIMIT = 8000      # single-text cap
    _FAILURE_TEXT_HEAD = 2000       # head characters kept when truncating
    _FAILURE_TEXT_TAIL = 6000       # tail characters kept when truncating (failure info skews toward the tail)

    @classmethod
    def _truncate_failure_text(cls, text: str) -> str:
        """Apply head/tail-preserving truncation to a single failure text (head 2K + tail 6K)."""
        if not isinstance(text, str) or len(text) <= cls._FAILURE_TEXT_LIMIT:
            return text
        head = text[: cls._FAILURE_TEXT_HEAD]
        tail = text[-cls._FAILURE_TEXT_TAIL :]
        dropped = len(text) - cls._FAILURE_TEXT_HEAD - cls._FAILURE_TEXT_TAIL
        return f"{head}\n...[truncated {dropped} chars]...\n{tail}"

    def _project_failure_details(self, res: dict) -> list[dict]:
        """Keep only the most critical information about failing nodes, for use by the retry prompt.

        When the executor writes the ExecuteReport it has already written all structured
        information (per-node entries + [DATA BUG] guidance entries) directly into
        test_failure_details. Here we slim it in two layers before passing it through
        (this only affects the prompt projection, not the full data the executor persists):
          (1) dedup: when output_excerpt and expectation_error are verbatim identical, drop output_excerpt;
          (2) truncate: apply head/tail-preserving truncation to expectation_error / output_excerpt when they exceed 8000 characters.
        """
        out: list[dict] = []
        for entry in res.get("test_failure_details") or []:
            if not isinstance(entry, dict):
                out.append(entry)
                continue
            slim = dict(entry)
            ee = slim.get("expectation_error")
            oe = slim.get("output_excerpt")
            # (1) dedup: if fully identical, no longer carry output_excerpt again
            if isinstance(oe, str) and oe == ee:
                slim.pop("output_excerpt", None)
                oe = None
            # (2) head/tail truncation
            if isinstance(ee, str):
                slim["expectation_error"] = self._truncate_failure_text(ee)
            if isinstance(oe, str):
                slim["output_excerpt"] = self._truncate_failure_text(oe)
            out.append(slim)
        return out

    def project_result(self, res: Any) -> Any:
        """Compress the executor's large result object into the failure summary the prompt needs.

        All failure information ([DATA BUG] validation hints, per-node entries) is presented
        uniformly through test_failure_details; pytest_tail / notes are no longer exposed separately.
        """
        if not isinstance(res, dict):
            return res
        return {
            "ok": res.get("ok"),
            "error_kind": res.get("error_kind"),
            "failing_node_ids": res.get("failing_node_ids") or [],
            "test_failure_details": self._project_failure_details(res),
        }

    def event_view(self, ev: dict, *, mode: str, omit_result: bool = False) -> dict:
        """Project a trace event into a prompt-visible view according to the given mode.

        Different phases must trade off between "information sufficiency" and "context locality":
        - `plan_full`: for PLAN, keeps the full enabled history across BigTasks
        - `attempt_big_task`: for the ATTEMPT series, keeps only the current BigTask slice
        - `augment_light`: for augment, keeps only a lightweight view related to the current event

        omit_result=True: does not output the result field (all history events are ok, so it carries no information and is too long).
        """
        kind = ev.get("kind")
        if kind == "PLAN":
            d = {
                "kind": "PLAN",
                "big_task_id": ev.get("big_task_id"),
                "reason": ev.get("reason"),
                "round": ev.get("round"),
                "natural_language_goal": ev.get("natural_language_goal"),
                "acceptance_criteria": ev.get("acceptance_criteria"),
                "test_patch": ev.get("test_patch"),
                "dim_tags": ev.get("dim_tags"),
                "own_f2p": ev.get("own_f2p"),
                "own_p2p": ev.get("own_p2p"),
                "subtasks_blueprint": ev.get("subtasks_blueprint"),
            }
            if not omit_result:
                d["result"] = self.project_result(ev.get("result"))
            return d
        if kind == "ATTEMPT":
            base = {
                "kind": "ATTEMPT",
                "big_task_id": ev.get("big_task_id"),
                "subtask_id": ev.get("subtask_id"),
                "natural_language_goal": ev.get("natural_language_goal"),
                "test_patch": ev.get("test_patch"),
                "acceptance_criteria": ev.get("acceptance_criteria"),
                "own_f2p": ev.get("own_f2p"),
                "own_p2p": ev.get("own_p2p"),
                "user_query": ev.get("user_query"),
                "patch": ev.get("patch"),
                "rationale": ev.get("rationale"),
            }
            if not omit_result:
                base["result"] = self.project_result(ev.get("result"))
            if mode == "augment_light":
                aug = {
                    "kind": "ATTEMPT",
                    "big_task_id": base["big_task_id"],
                    "subtask_id": base["subtask_id"],
                    "patch": base["patch"],
                    "test_patch": base["test_patch"],
                    "own_f2p": base["own_f2p"],
                    "own_p2p": base["own_p2p"],
                }
                if not omit_result:
                    aug["result"] = self.project_result(ev.get("result"))
                return aug
            return base
        return {"kind": kind}

    def enabled_events(self, trace: dict) -> list[dict]:
        """Take out all enabled events in the trace."""
        return [ev for ev in (trace.get("events") or []) if at.is_enabled(ev)]

    def event_views(self, events: list[dict], *, mode: str, omit_result: bool = False) -> list[dict]:
        """Batch-project a list of events into prompt views for the given mode."""
        return [self.event_view(ev, mode=mode, omit_result=omit_result) for ev in events]

    def split_plan_history_failure(self, trace: dict, decision: dict) -> tuple[list[dict], list[dict]]:
        """Slice the history trunk and the repair failure_brief according to the PLAN branch rules.

        - history: enabled events before start (clean history that other BigTasks have already passed).
        - failure_brief: events within the current big_task scope that are enabled=False or
          result.ok=False (matching either condition qualifies), so the repair prompt can see the
          complete failure information, including detailed error logs from a subtask's repeated failures.
        """
        if decision.get("reason") != "repair":
            return self.enabled_events(trace), []
        bid = decision.get("big_task_id")
        all_events = trace.get("events") or []
        # Among all events (including disabled), find the starting position of the repair big_task PLAN
        start = None
        for i, ev in enumerate(all_events):
            if ev.get("kind") == "PLAN" and ev.get("big_task_id") == bid:
                start = i
                break
        if start is None:
            return self.enabled_events(trace), []
        # History prefix: enabled events before start
        prefix_events = [ev for ev in all_events[:start] if at.is_enabled(ev)]
        # Failure tail: events within the current big_task scope that are enabled=False or result.ok=False
        failure_events = [
            ev for ev in all_events[start:]
            if not at.is_enabled(ev) or (ev.get("result") or {}).get("ok") is False
        ]
        return prefix_events, failure_events

    def split_attempt_history_failure(
        self, trace: dict, *, big_task_id: str, subtask_id: str,
    ) -> tuple[list[dict], list[dict]]:
        """Separate the current BigTask's normal history from a subtask's failed attempts.

        - history: enabled=True events within the current big_task, excluding all events of the
          current subtask (the PLAN event and the successful ATTEMPTs of other subtasks stay in history).
        - failure_brief: events within the current (big_task_id, subtask_id) scope that are
          enabled=False or result.ok=False (matching either), so the retry prompt can see the
          subtask's complete failure history (including detailed error logs from three failures).

        This is the single source of truth for the ATTEMPT / augment retry prompts.
        """
        all_events = trace.get("events") or []

        # failure_brief: events within the current subtask that are enabled=False or result.ok=False
        failure_events = [
            ev for ev in all_events
            if ev.get("big_task_id") == big_task_id
            and ev.get("subtask_id") == subtask_id
            and (not at.is_enabled(ev) or (ev.get("result") or {}).get("ok") is False)
        ]

        # history: PLAN events of prior BigTasks (so ATTEMPT knows the historical goals and test boundaries)
        # + enabled events within the current BigTask (excluding the current subtask)
        prev_plan_events = [
            ev for ev in all_events
            if at.is_enabled(ev)
            and ev.get("kind") == "PLAN"
            and ev.get("big_task_id") != big_task_id
        ]
        current_history_events = [
            ev for ev in all_events
            if at.is_enabled(ev)
            and ev.get("big_task_id") == big_task_id
            and ev.get("subtask_id") != subtask_id
        ]
        history_events = prev_plan_events + current_history_events

        return history_events, failure_events

    def assert_attempt_failure_split(
        self, history_events: list[dict], *, big_task_id: str, subtask_id: str,
    ) -> None:
        """Verify the ATTEMPT failure split did not mix the current failure into the history trunk."""
        if not history_events:
            return
        ev = history_events[-1]
        if (
            ev.get("kind") == "ATTEMPT"
            and ev.get("big_task_id") == big_task_id
            and ev.get("subtask_id") == subtask_id
            and (ev.get("result") or {}).get("ok") is False
        ):
            raise RuntimeError(
                "ATTEMPT failure split invariant violated: current subtask failure "
                "must be placed in failure_brief.events, not history.events"
            )

    def failure_brief(self, events: list[dict], *, mode: str) -> dict:
        """Render the failure tail using the same projection mode as the parent prompt."""
        return {"events": self.event_views(events, mode=mode)}

    def build_plan_prompt(
        self,
        *,
        trace: dict,
        decision: dict,
        prompt_name: str,
    ) -> tuple[str | None, str]:
        """Render the PLAN prompt based on the full history and, only in the repair scenario, the failure_brief.

        PLAN is the only phase that always sees the full enabled history, which conforms to the
        host rule: PLAN needs to advance across completed BigTasks, not just look at the current local slice.
        """
        history_events, failure_events = self.split_plan_history_failure(trace, decision)
        payload = {
            "decision": decision,
            "history": {"events": self.event_views(history_events, mode="plan_full", omit_result=True)},
            "failure_brief": self.failure_brief(failure_events, mode="plan_full"),
        }
        # Persist angle candidates: sample once per trace, reuse on every
        # subsequent build_plan_prompt call (e.g. repair / escalate) so the
        # model always sees the same 5 candidates throughout a single run.
        state = trace.setdefault("state", {})
        if "angle_candidates" not in state:
            state["angle_candidates"] = _sample_angle_candidates()
        return self.render_prompt(
            prompt_name, payload,
            extras={"ANGLE_CANDIDATES": state["angle_candidates"]},
        )

    def build_attempt_prompt_payload(
        self,
        *,
        trace: dict,
        target: dict,
    ) -> tuple[dict, dict]:
        """Construct the shared payload for the ATTEMPT-series prompts based only on the current BigTask history.

        The three steps -- user_query generation, implementation generation, rubric verification --
        all reuse this same input perspective: they see only the current BigTask slice and the
        current subtask's failure tail.
        """
        history_events, failure_events = self.split_attempt_history_failure(
            trace,
            big_task_id=target.get("big_task_id"),
            subtask_id=target.get("subtask_id"),
        )
        self.assert_attempt_failure_split(
            history_events,
            big_task_id=target.get("big_task_id"),
            subtask_id=target.get("subtask_id"),
        )
        latest_plan = next(
            (
                ev for ev in reversed(history_events)
                if ev.get("kind") == "PLAN"
                and ev.get("big_task_id") == target.get("big_task_id")
                and at.is_enabled(ev)
            ),
            {},
        )
        target_subtask = next(
            (
                st for st in (latest_plan.get("subtasks_blueprint") or [])
                if isinstance(st, dict) and st.get("id") == target.get("subtask_id")
            ),
            {},
        )
        payload = {
            "test_runner": self.test_runner_info(trace),
            "history": {"events": self.event_views(history_events, mode="attempt_big_task", omit_result=True)},
            "target": {
                "big_task_id": target.get("big_task_id"),
                "subtask_id": target.get("subtask_id"),
                "subtask_text": target_subtask.get("text"),
                "subtask_category": target_subtask.get("category") or "",
            },
            "failure_brief": self.failure_brief(failure_events, mode="attempt_big_task"),
        }
        extras = {
            "TARGET_SUBTASK_ID": str(target.get("subtask_id") or ""),
            "TARGET_SUBTASK_TEXT": str(target_subtask.get("text") or ""),
            "TARGET_SUBTASK_CATEGORY": json.dumps(target_subtask.get("category") or "", ensure_ascii=False),
        }
        return payload, extras

    def build_attempt_user_query_prompt(
        self,
        *,
        trace: dict,
        target: dict,
        prompt_name: str,
    ) -> tuple[str | None, str]:
        """Render the ATTEMPT first-step prompt, generating `user_query` and `acceptance_criteria`."""
        payload, extras = self.build_attempt_prompt_payload(
            trace=trace,
            target=target,
        )
        return self.render_prompt(prompt_name, payload, extras=extras)

    def build_attempt_impl_prompt(
        self,
        *,
        trace: dict,
        target: dict,
        user_query: dict,
        acceptance_criteria: str,
        prompt_name: str,
    ) -> tuple[str | None, str]:
        """Render the ATTEMPT second-step prompt, reusing the already-fixed `user_query` and `acceptance_criteria`."""
        payload, extras = self.build_attempt_prompt_payload(
            trace=trace,
            target=target,
        )
        extras = dict(extras)
        extras["ATTEMPT_USER_QUERY_JSON"] = json.dumps(user_query, ensure_ascii=False, indent=2)
        extras["ATTEMPT_ACCEPTANCE_CRITERIA"] = acceptance_criteria
        return self.render_prompt(prompt_name, payload, extras=extras)

    def build_augment_prompt(
        self,
        *,
        payload: dict,
        prompt_name: str,
    ) -> tuple[str | None, str]:
        """Render the augmentation prompt based on the already-compressed payload.

        augment intentionally does not reuse the full PLAN/ATTEMPT prompt structure; instead it
        consumes the current-target-perspective payload the caller has organized in advance.
        """
        return self.render_prompt(prompt_name, payload)

    def build_verify_attempt_prompt(
        self,
        *,
        trace: dict,
        target: dict,
        current_attempt: dict,
        prompt_name: str,
    ) -> tuple[str | None, str]:
        """Render the rubric verification prompt for the current ATTEMPT."""
        payload, extras = self.build_attempt_prompt_payload(
            trace=trace,
            target=target,
        )
        payload = {
            "test_runner": payload.get("test_runner") or self.test_runner_info(trace),
            "history": payload["history"],
            "target": payload["target"],
            "current_attempt": self.event_view(current_attempt, mode="attempt_big_task"),
        }
        return self.render_prompt(prompt_name, payload, extras=extras)

    def build_verify_attempt_current_only_prompt(
        self,
        *,
        current_attempt: dict,
        category: str,
        prompt_name: str,
        trace: dict | None = None,
    ) -> tuple[str | None, str]:
        """Render the prompt for the minimal-input rubric verifier, exposing only the current ATTEMPT.

        The verify prompt needs to know which of the four categories the current ATTEMPT belongs to,
        so that the `{{#if_category ...}}` conditional blocks in the template can land precisely on
        the true branch. Here we directly use the blueprint category passed in by the caller, rather
        than heuristically inferring it again from the ATTEMPT event shape.
        """
        payload = {
            "category": str(category or "unknown"),
            "test_runner": self.test_runner_info(trace),
            "current_attempt": self.event_view(current_attempt, mode="attempt_big_task"),
        }
        return self.render_prompt(prompt_name, payload)

    def build_verify_plan_prompt(
        self,
        *,
        current_plan: dict,
        category: str,
        prompt_name: str,
        trace: dict | None = None,
    ) -> tuple[str | None, str]:
        """Render the prompt for the PLAN rubric verifier, exposing only the current PLAN blueprint.

        Mirrors build_verify_attempt_current_only_prompt: projects the current PLAN into the
        plan_full view (including natural_language_goal / acceptance_criteria / dim_tags /
        subtasks_blueprint) and hands it to verify_plan.txt for 6-dimensional task-authenticity scoring.
        """
        payload = {
            "category": str(category or "unknown"),
            "test_runner": self.test_runner_info(trace),
            "current_plan": self.event_view(current_plan, mode="plan_full", omit_result=True),
        }
        return self.render_prompt(prompt_name, payload)
