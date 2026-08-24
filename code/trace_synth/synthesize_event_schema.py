from __future__ import annotations

from typing import Any


PLAN_SUBTASK_CATEGORIES = {
    "request_or_instruction",
}


class EventSchema:
    """Host-side event normalization and validation rules shared by the various flows.

    In v2 this class carries two kinds of responsibility:
    1. Even if a generation stage returns half-baked JSON, it must be normalized into a
       uniform canonical shape before actually being written into the trace; this class
       handles that normalization.
    2. Some semantics are host policy and should not be left to the model's discretion. For
       example, non-code subtasks are forced into empty ATTEMPTs, a normal ATTEMPT's `own_p2p`
       is deferred to augment, and a rollback ATTEMPT moves the model-generated `own_f2p` into
       `own_p2p` and then clears `own_f2p`. This keeps the generation interface uniform while
       the execution semantics still preserve rollback-specific checks.
    """

    @staticmethod
    def strip_forbidden(ev: dict) -> dict:
        """Remove host fields and runtime fields that must not appear in a persisted event."""
        ev = dict(ev)
        for k in ("seq", "ts", "big_task", "input", "attempt_no", "duration_sec", "executed_at"):
            ev.pop(k, None)
        res = ev.get("result")
        if isinstance(res, dict):
            res.pop("duration_sec", None)
        if ev.get("kind") == "PLAN":
            for st in ev.get("subtasks_blueprint", []) or []:
                st.pop("enabled", None)
        return ev

    @staticmethod
    def collect_plan_errors(ev: dict) -> list[str]:
        """Collect all of a PLAN's schema errors, rather than returning early at the first one."""
        errors: list[str] = []
        required = [
            "kind", "big_task_id", "reason", "round", "enabled",
            "natural_language_goal", "acceptance_criteria",
            "test_patch", "dim_tags", "own_f2p", "own_p2p",
            "subtasks_blueprint", "result",
        ]
        for k in required:
            if k not in ev:
                errors.append(f"PLAN missing {k}")
        if ev.get("kind") != "PLAN":
            errors.append("PLAN.kind must be PLAN")
        if not isinstance(ev.get("test_patch"), list):
            errors.append("PLAN.test_patch must be list[str]")
        elif any(not isinstance(x, str) for x in ev.get("test_patch")):
            errors.append("PLAN.test_patch elements must be str")
        expected_dims = {"context_scope", "change_scope", "requirement_depth", "knowledge_domains", "new_task_angle"}
        if set(ev.get("dim_tags") or {}) != expected_dims:
            errors.append(f"PLAN.dim_tags must contain exactly {sorted(expected_dims)}")
        subtasks = ev.get("subtasks_blueprint")
        if not isinstance(subtasks, list) or not subtasks:
            errors.append("PLAN.subtasks_blueprint must be non-empty list")
            subtasks = []
        category_set: set[str] = set()
        for i, st in enumerate(subtasks, 1):
            if not isinstance(st, dict):
                errors.append(f"subtasks_blueprint item #{i} must be object: {st}")
                continue
            forbidden = set(st) - {"id", "text", "category"}
            if forbidden:
                errors.append(f"subtasks_blueprint item #{i} has forbidden keys {sorted(forbidden)}: {st}")
            if "category" not in st:
                errors.append(f"subtasks_blueprint item #{i} missing category: {st}")
            category = st.get("category")
            if not isinstance(category, str) or not category:
                errors.append(f"subtasks_blueprint item #{i} category must be non-empty string: {st}")
                continue
            if category not in PLAN_SUBTASK_CATEGORIES:
                errors.append(f"subtasks_blueprint item #{i} category has unknown value {category!r}: {st}")
            category_set.add(category)
        return errors

    @classmethod
    def validate_plan_event(cls, ev: dict) -> None:
        """If the PLAN has any schema errors, aggregate them and raise at once."""
        errors = cls.collect_plan_errors(ev)
        if errors:
            joined = "\n".join(f"- {msg}" for msg in errors)
            raise ValueError(f"PLAN validation failed with {len(errors)} error(s):\n{joined}")

    @staticmethod
    def collect_attempt_errors(ev: dict) -> list[str]:
        """Collect all of an ATTEMPT's schema errors, rather than returning early at the first one."""
        errors: list[str] = []
        required = [
            "kind", "big_task_id", "subtask_id", "enabled", "natural_language_goal",
            "test_patch", "acceptance_criteria", "own_f2p", "own_p2p", "user_query",
            "patch", "result",
        ]
        for k in required:
            if k not in ev:
                errors.append(f"ATTEMPT missing {k}")
        allowed = set(required) | {"rationale"}
        extra = set(ev) - allowed
        if extra:
            errors.append(f"ATTEMPT has forbidden top-level keys: {sorted(extra)}")
        if ev.get("kind") != "ATTEMPT":
            errors.append("ATTEMPT.kind must be ATTEMPT")
        if ev.get("enabled") is not True:
            errors.append("ATTEMPT.enabled must be true")
        if ev.get("result") is not None:
            errors.append("ATTEMPT.result must be null")
        if not isinstance(ev.get("own_f2p"), list):
            errors.append("ATTEMPT.own_f2p must be list[str]")
        elif any(not isinstance(x, str) or not x.strip() for x in ev.get("own_f2p")):
            errors.append("ATTEMPT.own_f2p elements must be non-empty str")
        if not isinstance(ev.get("own_p2p"), list):
            errors.append("ATTEMPT.own_p2p must be list[str]")
        elif any(not isinstance(x, str) or not x.strip() for x in ev.get("own_p2p")):
            errors.append("ATTEMPT.own_p2p elements must be non-empty str")
        if not isinstance(ev.get("test_patch"), list):
            errors.append("ATTEMPT.test_patch must be list[str]")
        elif any(not isinstance(x, str) or not x.strip() for x in ev.get("test_patch")):
            errors.append("ATTEMPT.test_patch elements must be non-empty str")
        user_query = ev.get("user_query")
        patch = ev.get("patch")
        if not isinstance(user_query, dict) or not user_query:
            errors.append("ATTEMPT.user_query must be non-empty dict")
        else:
            forbidden = set(user_query) - {"role", "content"}
            if forbidden:
                errors.append(f"user_query has forbidden keys {sorted(forbidden)}")
            if user_query.get("role") != "user":
                errors.append("user_query.role must be user")
            if not isinstance(user_query.get("content"), str) or not user_query.get("content"):
                errors.append("user_query.content must be non-empty string")
        if not isinstance(patch, list):
            errors.append("ATTEMPT.patch must be list of SubPatch objects")
            patch = []
        for j, subpatch in enumerate(patch, 1):
            if not isinstance(subpatch, dict):
                errors.append(f"ATTEMPT.patch item #{j} must be object")
                continue
            fmt = subpatch.get("format")
            if fmt not in {"unified_diff", "env_setup"}:
                errors.append(f"ATTEMPT.patch item #{j} has unknown format {fmt!r}")
                continue
            allowed_keys = (
                {"format", "diff"}
                if fmt == "unified_diff"
                else {"format", "commands"}
            )
            extra_keys = set(subpatch) - allowed_keys
            if extra_keys:
                errors.append(
                    f"ATTEMPT.patch item #{j} has forbidden keys {sorted(extra_keys)}"
                )
            if fmt == "unified_diff":
                if not isinstance(subpatch.get("diff"), str):
                    errors.append(
                        f"ATTEMPT.patch item #{j}.diff must be str"
                    )
            else:
                commands = subpatch.get("commands")
                if not isinstance(commands, list):
                    errors.append(
                        f"ATTEMPT.patch item #{j}.commands must be list[str]"
                    )
                elif any(
                    not isinstance(command, str) or not command.strip()
                    for command in commands
                ):
                    errors.append(
                        f"ATTEMPT.patch item #{j}.commands elements must be non-empty str"
                    )
        return errors

    @classmethod
    def validate_attempt_event(cls, ev: dict) -> None:
        """If the ATTEMPT has any schema errors, aggregate them and raise at once."""
        errors = cls.collect_attempt_errors(ev)
        if errors:
            joined = "\n".join(f"- {msg}" for msg in errors)
            raise ValueError(f"ATTEMPT validation failed with {len(errors)} error(s):\n{joined}")

    @staticmethod
    def coerce_plan_event(
        ev: dict, *, big_task_id: str, reason: str, round_no: int,
    ) -> dict:
        """Normalize a raw PLAN JSON into the standard shape used for final persistence."""
        ev = EventSchema.strip_forbidden(ev)
        ev["kind"] = "PLAN"
        ev["big_task_id"] = big_task_id
        ev["reason"] = reason
        ev["round"] = round_no
        ev["enabled"] = True
        ev.setdefault("id", big_task_id)
        ev.pop("seed", None)
        ev["test_patch"] = []
        ev["own_f2p"] = []
        ev["own_p2p"] = []
        dims = dict(ev.get("dim_tags") or {})
        ev["dim_tags"] = {
            "context_scope": str(dims.get("context_scope") or "no_context"),
            "change_scope": str(dims.get("change_scope") or "no_change"),
            "requirement_depth": str(dims.get("requirement_depth") or "explicit_bug"),
            "knowledge_domains": list(dims.get("knowledge_domains") or []),
            "new_task_angle": list(dims.get("new_task_angle") or []),
        }
        ev.setdefault("uplift_proof", [])
        ev["result"] = None
        fixed = []
        for i, st in enumerate(ev.get("subtasks_blueprint") or [], 1):
            if not isinstance(st, dict):
                raise ValueError(f"PLAN.subtasks_blueprint item #{i} must be object")
            fixed.append({
                "id": str(st.get("id") or f"S{i}"),
                "text": str(st.get("text") or ""),
                "category": st.get("category") or "request_or_instruction",
            })
        ev["subtasks_blueprint"] = fixed
        return ev

    @staticmethod
    def coerce_patch_list(patch) -> list:
        """Normalize the patch field into a list of SubPatch shapes; supply an empty diff placeholder when missing."""
        if patch is None:
            return [{"format": "unified_diff", "diff": ""}]
        if not isinstance(patch, list):
            raise ValueError("ATTEMPT.patch must be list of SubPatch objects")
        return [dict(item) if isinstance(item, dict) else item for item in patch]

    @staticmethod
    def coerce_user_query(user_query) -> dict:
        """Normalize user_query into a standard `{role, content}` dict."""
        if not isinstance(user_query, dict):
            raise ValueError("ATTEMPT.user_query must be dict")
        return {"role": user_query.get("role"), "content": user_query.get("content")}

    @classmethod
    def coerce_attempt_event(
        cls, ev: dict, *, big_task_id: str, subtask: dict, test_patch: Any = None,
        user_query: dict | None = None,
        acceptance_criteria: str | None = None,
    ) -> dict:
        """Normalize a raw ATTEMPT JSON into the standard shape used for final persistence.

        This also enforces several v2 host policies here:
        - a normal ATTEMPT's `own_p2p` is not generated here, but left to augment
        """
        ev = cls.strip_forbidden(ev)
        ev["kind"] = "ATTEMPT"
        ev["big_task_id"] = big_task_id
        ev["subtask_id"] = str(subtask["id"])
        ev["enabled"] = True
        ev["natural_language_goal"] = str(subtask.get("text") or "")
        test_patch_list = test_patch if test_patch is not None else ev.get("test_patch")
        if test_patch_list is None:
            test_patch_list = []
        if not isinstance(test_patch_list, list):
            raise ValueError("ATTEMPT.test_patch must be list[str]")
        ev["test_patch"] = list(test_patch_list)
        if acceptance_criteria is not None:
            ev["acceptance_criteria"] = acceptance_criteria
        else:
            ev.setdefault("acceptance_criteria", "This subtask's changes are complete and its tests are declared passing")
        own_f2p = ev.get("own_f2p", [])
        own_p2p = ev.get("own_p2p", [])
        if not isinstance(own_f2p, list):
            raise ValueError("ATTEMPT.own_f2p must be list[str]")
        if not isinstance(own_p2p, list):
            raise ValueError("ATTEMPT.own_p2p must be list[str]")
        ev["own_f2p"] = list(own_f2p)
        ev["own_p2p"] = list(own_p2p)
        if user_query is None:
            if "user_query" not in ev:
                ev["user_query"] = {}
            ev["user_query"] = cls.coerce_user_query(ev.get("user_query"))
        else:
            ev["user_query"] = cls.coerce_user_query(user_query)
        ev["patch"] = cls.coerce_patch_list(ev.get("patch"))
        ev.setdefault("rationale", "")
        ev["result"] = None
        ev["own_p2p"] = []
        return ev

    @staticmethod
    def is_request_instruction_subtask(subtask: dict) -> bool:
        """Determine whether this subtask belongs to the normal implementation-type request_or_instruction branch."""
        return subtask.get("category") == "request_or_instruction"
