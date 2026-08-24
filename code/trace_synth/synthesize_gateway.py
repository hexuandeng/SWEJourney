from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shlex
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import requests

import tool_detail_log
from swebench_dockerlib import DockerSession
from local_test_executor import bootstrap_session, replay_model_patches
from synthesize_flow_utils import (
    find_invalid_surrogate,
    one_line,
    printable_text,
    sanitize_json_strings,
)
from synthesize_gateway_utils import call_json_with_docker_cli_messages as gateway_call_json_with_docker_cli_messages
from swe_bench_pro_runner import is_pro_instance
from skills_manager import SkillsManager, stage_file_for_event_kind

# Default Anthropic API version header value (official Messages API).
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
LLM_MAX_TOKENS = 65536
# Extended-thinking budget for the Anthropic official Messages API. Must be
# >=1024 and < max_tokens; the remaining tokens are available for the reply.
LLM_THINKING_BUDGET_TOKENS = 32768
DEFAULT_LLM_BACKEND = "anthropic"
# Local OpenAI-compatible server tuning (does not affect other backends):
# - Output cap defaults to 65536; reasoning_effort is always max; scheduling priority=100.
SGLANG_MAX_OUTPUT_TOKENS = 65536
# sglang server-side hard per-request context limit (input+output). Before sending, do a
# "shrink first, then clamp" fallback based on this: when the budget is below 65536 (prompt
# too long) -> first shrink history once via 90th-percentile truncation; if still insufficient,
# clamp output to [16384, 65536]. Output floor is 16384 (no smaller); anything longer is left to
# the server to error normally and the framework to self-heal.
SGLANG_CONTEXT_LIMIT = 321536
SGLANG_OUTPUT_SAFETY_MARGIN = 4096
SGLANG_MIN_OUTPUT_TOKENS = 16384
SGLANG_PRIORITY = 100


def _approx_sglang_tokens(text: str) -> int:
    """Conservative token estimate: char count / 3.5 (more pessimistic than /4; better to under-allocate output than to overshoot the limit)."""
    return int(len(text) / 3.5) if text else 0


def _shrink_sglang_prompt_once(messages: list[dict]) -> bool:
    """One-shot prompt shrink: for assistant text and tool return results, take the "90th-percentile
    char count" **per category** as the cap, truncate over-long ones from the tail (keep the front),
    and insert a truncation marker. The two categories have very different length distributions (tool
    docker output is often far longer than assistant reasoning), so percentiles are computed separately
    to avoid one category's threshold being skewed by the other. Only these two verbose history
    categories are truncated; system / user instructions are never touched. Modifies messages in place
    (these dicts are freshly created by deepseek_messages_from_chat, so the caller's original chat is
    unaffected). Runs only once; returns whether any truncation happened."""
    changed = False
    for role in ("assistant", "tool"):
        segs = [
            (i, len(m["content"]))
            for i, m in enumerate(messages)
            if m.get("role") == role
            and isinstance(m.get("content"), str)
            and m["content"]
        ]
        if len(segs) < 2:
            continue
        lengths = sorted(ln for _, ln in segs)
        cap = lengths[int(0.9 * (len(lengths) - 1))]  # 90th-percentile char count for this category
        for i, ln in segs:
            if ln > cap:
                c = messages[i]["content"]
                messages[i]["content"] = c[:cap] + f"\n...[truncated {ln - cap} chars, over {role} history 90th percentile]..."
                changed = True
    return changed


def default_endpoint_url(
    *,
    backend: str,
    endpoint_url: str | None,
    anthropic_url: str,
    openai_url: str,
) -> str:
    """Pick the default endpoint by backend; an explicit endpoint_url wins."""
    if endpoint_url:
        return endpoint_url
    norm = str(backend or "anthropic").strip().lower()
    if norm in ("openai", "deepseek", "local", "sglang"):
        # OpenAI-compatible Chat Completions endpoints (official OpenAI, or a
        # self-hosted / local server such as vLLM / SGLang).
        return openai_url
    return anthropic_url
_DOCKER_CLI_ONLY = [
    {
        "type": "function",
        "name": "docker_cli",
        "description": (
            "Run one shell command inside the current task execution environment. "
            "Use it to inspect, edit, validate, or perform any command needed before "
            "writing the PLAN/ATTEMPT. "
            "State accumulates across calls — if you apply a patch in one call, "
            "subsequent calls see the modified files. "
            "Use reset_worktree to restore the environment to the clean baseline when needed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to run from the configured repo cwd.",
                }
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
]
_DOCKER_CLI_WITH_RESET = _DOCKER_CLI_ONLY + [
    {
        "type": "function",
        "name": "reset_worktree",
        "description": (
            "Restore the worktree to the clean baseline state for the current generation "
            "session. Any modifications made by previous docker_cli calls "
            "(git apply, file edits, etc.) are discarded. "
            "Use this to get a clean environment for verifying that your patch applies "
            "correctly before submitting the final ATTEMPT JSON."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
]
# PLAN exposes only docker_cli, not reset_worktree; ATTEMPT / AUGMENT_TESTS expose both.
PLAN_DOCKER_CLI_TOOL = _DOCKER_CLI_ONLY
IMPL_DOCKER_CLI_TOOL = _DOCKER_CLI_WITH_RESET

# read_skills / patch_skills: exposed only in the three stages PLAN / ATTEMPT / AUGMENT_TESTS,
# letting the model read and enrich the host-side persistent engineering-experience library (skills directory).
_SKILLS_TOOLS = [
    {
        "type": "function",
        "name": "read_skills",
        "description": (
            "Read the persistent engineering-experience library for the current stage. "
            "Returns this stage's specialized file. "
            "Call this early so you can reuse and build on prior lessons."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "patch_skills",
        "description": (
            "Apply a unified-diff patch to the skills library to inherit and enrich the "
            "experience notes. You may only edit the current stage's specialized file. "
            "Produce the diff the same way you would for "
            "code (e.g. via git diff); do not hand-edit hunk headers. On apply failure the "
            "git error is returned so you can regenerate the diff and retry."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "patch": {
                    "type": "string",
                    "description": "A unified diff (git apply compatible) editing this stage's specialized file.",
                }
            },
            "required": ["patch"],
            "additionalProperties": False,
        },
    },
]


class LLMGateway:
    """Responsible for request formatting, retries, and the docker tool-call loop.

    The rest of the synthesis code should not care whether a given step goes over an
    Anthropic/Bedrock-style interface or OpenAI chat completions; nor should it care
    about docker_cli's multi-round tool loop. These transport-layer responsibilities
    are centralized here, so a flow only needs to choose:
    - to make a plain chat request
    - or to start a JSON-generation session with an optional docker_cli round
    """

    def __init__(
        self,
        *,
        app_id: str | None,
        api_key: str,
        provider: str,
        model: str,
        usage: str | None,
        backend: str,
        endpoint_url: str | None,
        anthropic_url: str,
        openai_url: str,
        prompt_usage_types: tuple[str, ...],
        log_step,
        trim_tool_output,
        usage_owner: "LLMGateway | None" = None,
        request_timeout_sec: int = 3600,
        anthropic_version: str = DEFAULT_ANTHROPIC_VERSION,
    ):
        """Inject logging/trimmer/timeout, and record model identity, backend, endpoint, and usage aggregation config."""
        self.log_step = log_step
        self.trim_tool_output = trim_tool_output
        self.request_timeout_sec = request_timeout_sec
        self.app_id = str(app_id or "").strip()
        self.api_key = api_key
        self.provider = str(provider or "").strip()
        self.model = str(model or "").strip()
        self.usage_mode = None if usage is None else str(usage).strip()
        self.backend = str(backend or "anthropic").strip().lower()
        # `local` is the public name for a self-hosted OpenAI-compatible server;
        # internally it reuses the OpenAI-compatible chat path (aliased `sglang`).
        if self.backend == "local":
            self.backend = "sglang"
        self.anthropic_version = str(anthropic_version or DEFAULT_ANTHROPIC_VERSION).strip()
        self.endpoint_url = default_endpoint_url(
            backend=self.backend,
            endpoint_url=endpoint_url,
            anthropic_url=anthropic_url,
            openai_url=openai_url,
        )
        self.usage_owner = usage_owner
        self.workspace = None
        # A local self-hosted server is reached directly (no outbound proxy) and
        # may run slow reasoning generations, so disable any inherited proxy and
        # extend the request timeout for this backend.
        if self.backend == "sglang":
            self.request_proxies: dict[str, str | None] | None = {"http": None, "https": None}
            self.request_timeout_sec = max(request_timeout_sec, 7200)
        else:
            self.request_proxies = None
        # The local server does not return prompt_tokens_details.cached_tokens, so
        # the cache-hit amount is estimated: remember the previous serialized
        # OpenAI messages and take the longest-common-prefix ratio next round
        # (see _estimate_sglang_cached_tokens).
        self._sglang_prev_prompt_text: str | None = None
        # skills experience-library directory (host side, permanently retained). Injected by
        # set_skills_dirs; when None, the read_skills / patch_skills tools are not exposed.
        self.skills_dir: Path | None = None
        self.skills_history_dir: Path | None = None
        self.prompt_usage_types = tuple(prompt_usage_types)
        self.usage_checkpoint_path: Path | None = None
        self.account_id: str | None = None
        self.account_id_cache_path = self._default_account_id_cache_path()
        self._load_account_id_cache()
        self.usage: dict[str, Any] = {
            "model": self.model,
            "provider": self.provider,
            "backend": self.backend,
            "endpoint_url": self.endpoint_url,
            "cost": 0.0,
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "total_tokens": 0,
            "by_stage": {},
            "by_prompt_type": {
                prompt_type: {
                    "cost": 0.0,
                    "requests": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "total_tokens": 0,
                }
                for prompt_type in self.prompt_usage_types
            },
        }

    def set_usage_checkpoint(self, path: str | Path | None) -> None:
        """Set the usage checkpoint file path; cumulative usage will be written back into it later."""
        self.usage_checkpoint_path = Path(path) if path else None

    def set_skills_dirs(self, skills_dir: str | Path | None, history_dir: str | Path | None) -> None:
        """Inject the host-side skills experience-library directory and its history snapshot directory.

        If either is None, the read_skills / patch_skills tools will not be exposed.
        """
        self.skills_dir = Path(skills_dir) if skills_dir else None
        self.skills_history_dir = Path(history_dir) if history_dir else None

    def _default_account_id_cache_path(self) -> Path:
        """Generate a stable account_id cache file path for the current app/provider/model."""
        cache_root = Path(
            os.environ.get("LLM_ACCOUNT_ID_CACHE_DIR")
            or (Path(os.environ.get("TMPDIR") or "/tmp") / "llm_account_ids")
        )
        cache_root.mkdir(parents=True, exist_ok=True)
        key = "\n".join([
            self.endpoint_url,
            self.app_id,
            self.provider,
            self.model,
            self.backend,
            self.usage_mode or "",
        ])
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return cache_root / f"{digest}.txt"

    def _load_account_id_cache(self) -> None:
        """Restore the last-obtained account_id from the hash-named cache file."""
        try:
            if not self.account_id_cache_path.exists():
                return
            account_id = self.account_id_cache_path.read_text(encoding="utf-8").strip()
            if account_id:
                self.account_id = account_id
        except Exception:
            return

    def remember_account_id(self, account_id: str | None) -> None:
        """Remember the X-Account-Id returned in the response headers, so subsequent requests always hit the same account."""
        value = str(account_id or "").strip()
        if not value:
            return
        self.account_id = value
        tmp_path = self.account_id_cache_path.with_name(
            f"{self.account_id_cache_path.name}.{os.getpid()}.tmp"
        )
        try:
            tmp_path.write_text(value + "\n", encoding="utf-8")
            os.replace(tmp_path, self.account_id_cache_path)
        except Exception:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

    def load_usage_summary(self, summary: Any) -> None:
        """Restore the previously accumulated usage summary from an existing trace."""
        if not isinstance(summary, dict):
            return
        merged = self.usage_summary()
        for key, value in summary.items():
            if key in {"by_stage", "by_prompt_type"} and isinstance(value, dict):
                current = merged.setdefault(key, {})
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict):
                        current.setdefault(subkey, {}).update(subvalue)
                    else:
                        current[subkey] = subvalue
            else:
                merged[key] = value
        merged.setdefault("by_stage", {})
        by_prompt_type = merged.setdefault("by_prompt_type", {})
        for prompt_type in self.prompt_usage_types:
            by_prompt_type.setdefault(prompt_type, {
                "cost": 0.0,
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "total_tokens": 0,
            })
        self.usage = merged

    def checkpoint_usage(self) -> None:
        """Write the current usage summary back into the checkpoint trace file."""
        if self.usage_checkpoint_path is None:
            return
        try:
            path = self.usage_checkpoint_path
            if not path.exists():
                return
            trace = json.loads(path.read_text(encoding="utf-8"))
            if not (
                isinstance(trace, dict)
                and isinstance(trace.get("instance"), dict)
                and isinstance(trace.get("config"), dict)
                and isinstance(trace.get("events"), list)
            ):
                return
            trace.setdefault("state", {})["llm_usage"] = self.usage_summary()
            path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            return

    # ── Price table (CNY per million tokens; convention cost_stored/1e6 = CNY, same scale as deepseek) ──
    # deepseek/sglang responses carry or self-compute cost; the anthropic (Vertex opus) and azure openai
    # gateway responses **do not return cost**. This converts the official USD price x exchange rate into
    # "CNY per million", and self-computes it in record_usage when the body has no cost.
    # Exchange rate: USD->CNY ~= 7.2 (estimation basis, same below; >200K long-context opus is actually more expensive, not broken out).
    _USD_TO_CNY = 7.2
    # Claude Opus 4.8:in $5 / out $25 / cache_read $0.5 / cache_write(5m) $6.25 per M
    _OPUS_PRICE_INPUT_PER_M = 5.0 * _USD_TO_CNY        # 36.0
    _OPUS_PRICE_OUTPUT_PER_M = 25.0 * _USD_TO_CNY      # 180.0
    _OPUS_PRICE_CACHE_READ_PER_M = 0.5 * _USD_TO_CNY   # 3.6
    _OPUS_PRICE_CACHE_WRITE_PER_M = 6.25 * _USD_TO_CNY # 45.0
    # Azure GPT-5.4:in $2.5 / out $15 / cached_in $0.25 per M
    _AZURE_PRICE_INPUT_PER_M = 2.5 * _USD_TO_CNY       # 18.0
    _AZURE_PRICE_OUTPUT_PER_M = 15.0 * _USD_TO_CNY     # 108.0
    _AZURE_PRICE_CACHE_READ_PER_M = 0.25 * _USD_TO_CNY # 1.8

    @classmethod
    def _estimate_cost_from_tokens(
        cls,
        *,
        backend: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read: int,
        cache_write: int,
    ) -> float:
        """Self-compute cost via the price table for gateways that do not return cost (anthropic Vertex opus / azure openai).

        The return value is already multiplied by 1e6 (same as deepseek, cost_stored/1e6 = CNY). Returns 0 when no price table matches.
        """
        mdl = (model or "").lower()
        if backend == "anthropic" or "opus" in mdl or "claude" in mdl:
            return (
                input_tokens * cls._OPUS_PRICE_INPUT_PER_M
                + output_tokens * cls._OPUS_PRICE_OUTPUT_PER_M
                + cache_read * cls._OPUS_PRICE_CACHE_READ_PER_M
                + cache_write * cls._OPUS_PRICE_CACHE_WRITE_PER_M
            )
        if backend == "openai" or "gpt" in mdl or "azure" in mdl:
            return (
                input_tokens * cls._AZURE_PRICE_INPUT_PER_M
                + output_tokens * cls._AZURE_PRICE_OUTPUT_PER_M
                + cache_read * cls._AZURE_PRICE_CACHE_READ_PER_M
            )
        return 0.0

    @staticmethod
    def _usage_int(value: Any) -> int:
        """Convert a usage field to int as robustly as possible."""
        try:
            return int(value or 0)
        except Exception:
            return 0

    @staticmethod
    def _usage_float(value: Any) -> float:
        """Convert a usage/cost field to float as robustly as possible."""
        try:
            return float(value or 0)
        except Exception:
            return 0.0

    @staticmethod
    def _nested_token_sum(value: Any) -> int:
        """Recursively sum nested token fields, handling nested structures like cache_creation."""
        if isinstance(value, dict):
            return sum(LLMGateway._nested_token_sum(v) for v in value.values())
        if isinstance(value, list):
            return sum(LLMGateway._nested_token_sum(v) for v in value)
        return LLMGateway._usage_int(value)

    @staticmethod
    def _prompt_type_for_stage(stage: str) -> str | None:
        """Collapse a finer-grained stage name into a coarser-grained prompt type."""
        stage = str(stage or "").upper()
        if stage.startswith("PLAN"):
            return "PLAN"
        if stage.startswith("VERIFY_ATTEMPT"):
            return "VERIFY_ATTEMPT"
        if stage.startswith("ATTEMPT"):
            return "ATTEMPT"
        if stage.startswith("AUGMENT"):
            return "AUGMENT_TESTS"
        return None

    def _add_usage_delta(self, stage: str, delta: dict[str, Any]) -> None:
        """Merge one request's usage delta into the total table, by_stage, and by_prompt_type."""
        if "cost" in delta:
            delta["cost"] = self._usage_float(delta.get("cost"))
        for key, value in delta.items():
            if key == "cost":
                self.usage[key] = self._usage_float(self.usage.get(key)) + value
            else:
                self.usage[key] = self._usage_int(self.usage.get(key)) + value
        by_stage = self.usage.setdefault("by_stage", {})
        stage_usage = by_stage.setdefault(stage, {
            "cost": 0.0,
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "total_tokens": 0,
        })
        for key, value in delta.items():
            if key == "cost":
                stage_usage[key] = self._usage_float(stage_usage.get(key)) + value
            else:
                stage_usage[key] = self._usage_int(stage_usage.get(key)) + value
        prompt_type = self._prompt_type_for_stage(stage)
        if prompt_type:
            by_prompt_type = self.usage.setdefault("by_prompt_type", {})
            prompt_usage = by_prompt_type.setdefault(prompt_type, {
                "cost": 0.0,
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "total_tokens": 0,
            })
            for key, value in delta.items():
                if key == "cost":
                    prompt_usage[key] = self._usage_float(prompt_usage.get(key)) + value
                else:
                    prompt_usage[key] = self._usage_int(prompt_usage.get(key)) + value

    def record_usage(self, body: dict[str, Any], *, stage: str) -> None:
        """Extract usage/cost from one model response and accumulate it into the ledger."""
        raw = body.get("usage") or {}
        if not isinstance(raw, dict):
            return
        input_tokens = self._usage_int(raw.get("input_tokens") or raw.get("prompt_tokens"))
        output_tokens = self._usage_int(raw.get("output_tokens") or raw.get("completion_tokens"))
        cache_creation = self._usage_int(raw.get("cache_creation_input_tokens"))
        if not cache_creation:
            cache_creation = self._nested_token_sum(raw.get("cache_creation"))
        cache_read = self._usage_int(raw.get("cache_read_input_tokens"))
        cost = self._usage_float(body.get("cost"))
        # anthropic (Vertex opus) / azure openai gateway responses do not return cost.
        # Here, when cost is missing, self-compute it via the price table (deepseek/sglang already carry cost).
        if not cost:
            cost = self._estimate_cost_from_tokens(
                backend=self.backend,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read=cache_read,
                cache_write=cache_creation,
            )
        total_tokens = (
            self._usage_int(raw.get("total_tokens"))
            or input_tokens + output_tokens + cache_creation + cache_read
        )
        delta = {
            "cost": cost,
            "requests": 1,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": cache_creation,
            "cache_read_input_tokens": cache_read,
            "total_tokens": total_tokens,
        }
        self._add_usage_delta(stage, delta)
        # Print this call's cache-hit info on every call (including the stage's cumulative), for real-time observation of the implicit-cache effect.
        stage_acc = (self.usage.get("by_stage") or {}).get(stage) or {}
        prompt_in = input_tokens + cache_read  # cache_read has been split out of input; add it back for total request input
        hit_pct = (cache_read / prompt_in * 100.0) if prompt_in else 0.0
        print(
            f"[synthesize] llm cache stage={stage} backend={getattr(self, 'backend', '?')}"
            f"  this_call: cache_read={cache_read} cache_write={cache_creation}"
            f" input={input_tokens} output={output_tokens} hit={hit_pct:.1f}%"
            f"  | stage_total: cache_read={stage_acc.get('cache_read_input_tokens', 0)}"
            f" cache_write={stage_acc.get('cache_creation_input_tokens', 0)}"
            f" input={stage_acc.get('input_tokens', 0)} req={stage_acc.get('requests', 0)}",
            flush=True,
        )
        if self.usage_owner is not None:
            self.usage_owner._add_usage_delta(stage, dict(delta))
            self.usage_owner.checkpoint_usage()
        else:
            self.checkpoint_usage()

    def usage_summary(self) -> dict[str, Any]:
        """Return a deep copy of the usage ledger, to prevent external in-place mutation of internal state."""
        return json.loads(json.dumps(self.usage, ensure_ascii=False))

    @staticmethod
    def anthropic_tool(tool: Any) -> Any:
        """Convert a generic tool schema into the shape required by the Anthropic/Bedrock interface."""
        if not isinstance(tool, dict):
            return tool
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            fn = tool["function"]
            return {
                "name": fn.get("name"),
                "description": fn.get("description"),
                "input_schema": fn.get("parameters") or {},
            }
        if "parameters" in tool and "input_schema" not in tool:
            out = dict(tool)
            out["input_schema"] = out.pop("parameters")
            out.pop("type", None)
            return out
        return tool

    @staticmethod
    def request_json_sync(
        method: str,
        url: str,
        headers: dict[str, str],
        *,
        payload: dict[str, Any] | None = None,
        timeout: int = 3600,
        proxies: dict[str, str | None] | None = None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Synchronously make one HTTP JSON request, wrapping any non-200/non-JSON result into an exception.

        When `proxies` is passed explicitly (e.g. a direct connection to a self-hosted sglang),
        the outbound HTTP proxy is disabled; otherwise slow generations hit the proxy idle-timeout and
        get a 504 HTML error page.
        """
        response = requests.request(
            method, url, headers=headers, json=payload, timeout=timeout, proxies=proxies
        )
        # Judge by status code first: any non-200 (including proxy 504 HTML error pages, gateway 5xx) is
        # uniformly reported as "LLM http {status}", so the upper layer's is_retryable_llm_error /
        # _is_infra_error correctly recognizes it as a retryable infrastructure error, rather than being
        # thrown as "invalid json" by a failed `.json()` parse first -- which would be mistaken for invalid
        # model content, burning an ATTEMPT retry and wrongly going through json_repair.
        if response.status_code != 200:
            raise RuntimeError(
                f"LLM http {response.status_code}: {printable_text(response.text[:500])}"
            )
        try:
            body = response.json()
        except Exception as exc:
            raise RuntimeError(
                f"LLM invalid json (status={response.status_code}): {printable_text(response.text[:500])}"
            ) from exc
        return body, dict(response.headers)

    @staticmethod
    def is_retryable_llm_error(exc: Exception) -> bool:
        """Decide whether a given model-request error is worth automatically retrying."""
        text = printable_text(exc).lower()
        return (
            "llm http 429" in text
            or "llm http 529" in text
            or "overloaded" in text
            or "rate" in text
            or "llm http 500" in text
            or "llm http 502" in text
            or "llm http 503" in text
            or "llm http 504" in text
            # Fallback: even if some path still throws the old-style "invalid json (status=504)", or the
            # proxy/network layer directly throws a connection/timeout error, treat these as retryable
            # infrastructure errors too.
            or "invalid json (status=502" in text
            or "invalid json (status=503" in text
            or "invalid json (status=504" in text
            or "timed out" in text
            or "timeout" in text
            or "connection" in text
        )

    @staticmethod
    def _anthropic_text_block(text: Any) -> dict[str, Any]:
        """Uniformly convert plain text into an Anthropic text block, to make it easy to attach an explicit cache breakpoint."""
        return {"type": "text", "text": printable_text(text)}

    @classmethod
    def _anthropic_content_blocks(cls, *, role: str, content: Any) -> list[dict[str, Any]]:
        """Normalize a single message's content into a block array."""
        if isinstance(content, list):
            if role == "assistant":
                content_error = find_invalid_surrogate(content, path="$.messages[].content")
                if content_error is not None:
                    raise RuntimeError(
                        "Refusing to forward non-standard Anthropic assistant content: "
                        f"{content_error}"
                    )
                return [dict(block) if isinstance(block, dict) else {"type": "text", "text": printable_text(block)} for block in content]
            normalized = sanitize_json_strings(content)
            return [dict(block) if isinstance(block, dict) else {"type": "text", "text": printable_text(block)} for block in normalized]
        return [cls._anthropic_text_block(content)]

    @staticmethod
    def _is_cacheable_anthropic_block(block: Any) -> bool:
        """Decide whether an Anthropic content block may carry an explicit cache_control."""
        if not isinstance(block, dict):
            return False
        block_type = str(block.get("type") or "")
        if block_type == "thinking":
            return False
        if block_type == "text":
            return bool(str(block.get("text") or "").strip())
        return True

    @classmethod
    def _attach_ephemeral_cache_breakpoint(
        cls,
        *,
        system_blocks: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        ttl: str = "5m",
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Per the official explicit-breakpoint semantics, attach cache_control only to the last cacheable block."""
        cache_control = {"type": "ephemeral", "ttl": ttl}
        for msg_idx in range(len(messages) - 1, -1, -1):
            content = messages[msg_idx].get("content") or []
            if not isinstance(content, list):
                continue
            for block_idx in range(len(content) - 1, -1, -1):
                block = content[block_idx]
                if not cls._is_cacheable_anthropic_block(block):
                    continue
                messages[msg_idx] = dict(messages[msg_idx])
                new_content = list(content)
                new_block = dict(block)
                new_block["cache_control"] = dict(cache_control)
                new_content[block_idx] = new_block
                messages[msg_idx]["content"] = new_content
                return system_blocks, messages
        for block_idx in range(len(system_blocks) - 1, -1, -1):
            block = system_blocks[block_idx]
            if not cls._is_cacheable_anthropic_block(block):
                continue
            new_system = list(system_blocks)
            new_block = dict(block)
            new_block["cache_control"] = dict(cache_control)
            new_system[block_idx] = new_block
            return new_system, messages
        return system_blocks, messages

    @classmethod
    def anthropic_messages_from_chat(cls, messages: list[dict]) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]]]:
        """Arrange a generic chat message list into an Anthropic-style request, placing the breakpoint on the last cacheable block."""
        system_blocks: list[dict[str, Any]] = []
        out: list[dict[str, Any]] = []
        for msg in messages:
            role = str(msg.get("role") or "user")
            content = msg.get("content", "")
            if role == "system":
                if isinstance(content, list):
                    normalized = sanitize_json_strings(content)
                    for block in normalized:
                        if isinstance(block, dict):
                            system_blocks.append(dict(block))
                        else:
                            system_blocks.append(cls._anthropic_text_block(block))
                else:
                    system_blocks.append(cls._anthropic_text_block(content))
                continue
            if role not in {"user", "assistant"}:
                role = "user"
            out.append({"role": role, "content": cls._anthropic_content_blocks(role=role, content=content)})
        system_blocks, out = cls._attach_ephemeral_cache_breakpoint(system_blocks=system_blocks, messages=out)
        return (system_blocks or None), out

    @staticmethod
    def anthropic_response_text(body: dict[str, Any]) -> str:
        """Extract the visible text from an Anthropic-style response."""
        content = body.get("content")
        if isinstance(content, list):
            parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            ]
            if parts:
                return "\n".join(parts)
        return ""

    @staticmethod
    def openai_messages_from_chat(messages: list[dict]) -> list[dict[str, Any]]:
        """Arrange a generic chat message list into OpenAI chat completions style."""
        out: list[dict[str, Any]] = []
        for msg in messages:
            role = str(msg.get("role") or "user")
            content = msg.get("content", "")
            if isinstance(content, str):
                normalized = printable_text(content)
            else:
                normalized = json.dumps(sanitize_json_strings(content), ensure_ascii=False)
            if role == "system":
                out.append({"role": "developer", "content": normalized})
            elif role in {"user", "assistant", "developer"}:
                out.append({"role": role, "content": normalized})
            else:
                out.append({"role": "user", "content": normalized})
        return out

    @staticmethod
    def _tool_result_text(content: Any) -> str:
        """Normalize a tool_result's content into a plain text string."""
        if isinstance(content, str):
            return printable_text(content)
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(printable_text(block["text"]))
                elif isinstance(block, str):
                    parts.append(printable_text(block))
            return "\n".join(parts)
        if content is None:
            return ""
        return printable_text(json.dumps(content, ensure_ascii=False))

    # ---- deepseek (OpenAI Chat Completions standard protocol) bidirectional translation ----
    # Internal messages are always Anthropic-shaped; here they are converted <-> OpenAI chat
    # messages/tool_calls, and the response is translated back to Anthropic shape, so the upper
    # tool loop is unaware. The deepseek standard-protocol response **does not return cost**, so
    # cost is self-computed via the price table (units: see _DEEPSEEK_PRICE_*).

    # Price table: CNY per million tokens. Convention cost_stored/1e6 = CNY (same scale as other backends).
    _DEEPSEEK_PRICE_INPUT_PER_M = 3.0       # uncached input
    _DEEPSEEK_PRICE_OUTPUT_PER_M = 6.0      # output
    _DEEPSEEK_PRICE_CACHED_PER_M = 0.025    # cache-hit input

    @classmethod
    def deepseek_messages_from_chat(cls, messages: list[dict]) -> list[dict[str, Any]]:
        """Convert internal Anthropic-shaped chat messages into OpenAI Chat Completions messages.

        - system -> {role:system}
        - user/assistant string content is kept as-is; block arrays are decomposed by type:
          text -> text accumulation; assistant's tool_use -> message.tool_calls;
          user's tool_result -> a standalone {role:tool, tool_call_id, content} message.
        """
        out: list[dict[str, Any]] = []
        for msg in messages:
            role = str(msg.get("role") or "user")
            content = msg.get("content", "")
            if role == "system":
                text = content if isinstance(content, str) else cls._deepseek_text_from_blocks(content)
                out.append({"role": "system", "content": printable_text(text)})
                continue
            if not isinstance(content, list):
                out.append({"role": role if role in {"user", "assistant"} else "user",
                            "content": printable_text(content)})
                continue
            # block array: first collect text / tool_use / tool_result.
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            tool_result_msgs: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    t = printable_text(block)
                    if t.strip():
                        text_parts.append(t)
                    continue
                btype = str(block.get("type") or "")
                if btype == "text":
                    t = printable_text(block.get("text") or "")
                    if t.strip():
                        text_parts.append(t)
                elif btype == "tool_use":
                    args = block.get("input") if isinstance(block.get("input"), dict) else {}
                    tool_calls.append({
                        "id": str(block.get("id") or ""),
                        "type": "function",
                        "function": {
                            "name": str(block.get("name") or ""),
                            "arguments": json.dumps(args, ensure_ascii=False),
                        },
                    })
                elif btype == "tool_result":
                    tool_result_msgs.append({
                        "role": "tool",
                        "tool_call_id": str(block.get("tool_use_id") or ""),
                        "content": cls._tool_result_text(block.get("content")),
                    })
                elif btype == "thinking":
                    continue
                else:
                    text_parts.append(printable_text(json.dumps(block, ensure_ascii=False)))
            if role == "assistant":
                am: dict[str, Any] = {"role": "assistant"}
                am["content"] = "\n".join(text_parts) if text_parts else ""
                if tool_calls:
                    am["tool_calls"] = tool_calls
                out.append(am)
                # An assistant message will not simultaneously carry tool_result; ignore tool_result_msgs.
            else:
                # user role: first send text (if any), then send tool result messages one by one.
                if text_parts:
                    out.append({"role": "user", "content": "\n".join(text_parts)})
                out.extend(tool_result_msgs)
        return out

    @staticmethod
    def _deepseek_text_from_blocks(content: Any) -> str:
        """Extract plain text from a block array/string (used for messages without tool semantics, e.g. system)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for b in content:
                if isinstance(b, dict) and isinstance(b.get("text"), str):
                    parts.append(b["text"])
                elif isinstance(b, str):
                    parts.append(b)
            return "\n".join(parts)
        return printable_text(content)

    @staticmethod
    def deepseek_tool(tool: Any) -> dict[str, Any]:
        """Convert a generic tool schema into a single OpenAI tools function declaration."""
        if not isinstance(tool, dict):
            return {}
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            fn = tool["function"]
            name, desc, params = fn.get("name"), fn.get("description"), fn.get("parameters")
        else:
            name = tool.get("name")
            desc = tool.get("description")
            params = tool.get("parameters") or tool.get("input_schema")
        params = params if isinstance(params, dict) else {"type": "object", "properties": {}}
        return {
            "type": "function",
            "function": {"name": name, "description": desc or "", "parameters": params},
        }

    @classmethod
    def anthropic_body_from_deepseek(cls, body: dict[str, Any]) -> dict[str, Any]:
        """Translate an OpenAI Chat Completions response into an Anthropic-shaped body.

        Produces {role, content[], stop_reason, usage, cost}, where content may contain text / tool_use,
        so the upper tool loop (which only understands the Anthropic shape) can feed it back and parse
        tool_use as-is. cost is self-computed.
        """
        choices = body.get("choices") or []
        choice = choices[0] if isinstance(choices, list) and choices else {}
        msg = (choice.get("message") or {}) if isinstance(choice, dict) else {}
        out_content: list[dict[str, Any]] = []
        text = msg.get("content")
        if isinstance(text, str) and text.strip():
            out_content.append({"type": "text", "text": text})
        has_tool_use = False
        for tc in (msg.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) and raw_args.strip() else (raw_args or {})
            except Exception:
                args = {}
            if not isinstance(args, dict):
                args = {}
            has_tool_use = True
            out_content.append({
                "type": "tool_use",
                "id": str(tc.get("id") or f"deepseek-{uuid.uuid4().hex[:24]}"),
                "name": str(fn.get("name") or ""),
                "input": args,
            })
        stop_reason = "tool_use" if has_tool_use else "end_turn"
        usage = cls._deepseek_usage_to_anthropic(body.get("usage"))
        out = {
            "role": "assistant",
            "content": out_content,
            "stop_reason": stop_reason,
            "usage": usage,
        }
        # The standard protocol does not return cost: self-compute via the price table (CNY per million tokens), storing as cost/1e6 = CNY.
        out["cost"] = (
            usage.get("input_tokens", 0) * cls._DEEPSEEK_PRICE_INPUT_PER_M
            + usage.get("cache_read_input_tokens", 0) * cls._DEEPSEEK_PRICE_CACHED_PER_M
            + usage.get("output_tokens", 0) * cls._DEEPSEEK_PRICE_OUTPUT_PER_M
        )
        return out

    @classmethod
    def _deepseek_usage_to_anthropic(cls, raw: Any) -> dict[str, Any]:
        """Map OpenAI usage into the fields that record_usage understands.

        prompt_tokens includes cached_tokens; split it out to align with input(uncached)/cache_read semantics.
        """
        if not isinstance(raw, dict):
            return {}
        prompt = raw.get("prompt_tokens") or 0
        completion = raw.get("completion_tokens") or 0
        total = raw.get("total_tokens") or 0
        details = raw.get("prompt_tokens_details") or {}
        cached = (details.get("cached_tokens") if isinstance(details, dict) else 0) or 0
        non_cached_input = prompt - cached if prompt and cached else prompt
        return {
            "input_tokens": non_cached_input,
            "output_tokens": completion,
            "cache_read_input_tokens": cached,
            "cache_creation_input_tokens": 0,
            "total_tokens": total,
        }

    # ---- local OpenAI-compatible server: same protocol as deepseek, but does
    #      not return cached_tokens ----
    # Price table follows deepseek's official prices (input 3.0 / cached 0.025 / output 6.0, CNY per million tokens).
    # Cache-hit amount is self-estimated: SGLang RadixAttention reuses by token prefix; the agent loop only
    # appends messages at the tail each round, so "the longest-common-prefix length ratio between this round's
    # request text and the previous round's request text" ~= the hit-prefix ratio; multiplied by this round's
    # prompt_tokens gives the estimated cache_read. This is an optimistic estimate (assuming the prefix was not
    # evicted), which basically holds on a dedicated large-cache machine. Marked as estimated in logs/usage to
    # avoid confusion with a real billing backend.

    @staticmethod
    def _sglang_prompt_fingerprint(payload_messages: list[dict]) -> str:
        """Serialize this round's OpenAI messages into stable text, for longest-common-prefix comparison."""
        try:
            return json.dumps(payload_messages, ensure_ascii=False, sort_keys=False)
        except Exception:
            return str(payload_messages)

    @staticmethod
    def _longest_common_prefix_len(a: str, b: str) -> int:
        """Longest-common-prefix char count of two strings."""
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def _estimate_sglang_cached_tokens(self, prompt_tokens: int, cur_prompt_text: str) -> int:
        """Estimate this round's cache-hit token count based on the longest-common-prefix ratio with the previous request."""
        prev = self._sglang_prev_prompt_text
        cached = 0
        if prev and prompt_tokens > 0 and cur_prompt_text:
            lcp = self._longest_common_prefix_len(prev, cur_prompt_text)
            frac = lcp / len(cur_prompt_text) if cur_prompt_text else 0.0
            # ratio x this round's prompt_tokens, floored, and not exceeding prompt_tokens.
            cached = min(prompt_tokens, int(prompt_tokens * frac))
        # Remember this round's request text for next round's comparison.
        self._sglang_prev_prompt_text = cur_prompt_text
        return cached

    def anthropic_body_from_sglang(
        self, body: dict[str, Any], *, cur_prompt_text: str
    ) -> dict[str, Any]:
        """Translate an sglang(OpenAI) response into Anthropic shape.

        Follows the deepseek translation, but since sglang does not return cached_tokens, first use the
        prefix estimate to fill back usage.prompt_tokens_details.cached_tokens, then reuse deepseek's
        usage/cost computation, ensuring cache hits are also priced at 0.025 (identical to deepseek).
        """
        raw_usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        prompt_tokens = int(raw_usage.get("prompt_tokens") or 0)
        details = raw_usage.get("prompt_tokens_details")
        already = (details.get("cached_tokens") if isinstance(details, dict) else 0) or 0
        if not already:
            # The server did not return cached_tokens -> self-estimate and stuff it back, so the deepseek translation/pricing reuses it transparently.
            est = self._estimate_sglang_cached_tokens(prompt_tokens, cur_prompt_text)
            new_details = dict(details) if isinstance(details, dict) else {}
            new_details["cached_tokens"] = est
            new_details["cached_tokens_estimated"] = True
            raw_usage = dict(raw_usage)
            raw_usage["prompt_tokens_details"] = new_details
            body = dict(body)
            body["usage"] = raw_usage
        else:
            # The server actually returned cached_tokens (future version): use it directly, just update the prefix state.
            self._sglang_prev_prompt_text = cur_prompt_text
        return self.anthropic_body_from_deepseek(body)

    def client_response_text(self, body: dict[str, Any]) -> str:
        """Extract the final text from the response body, depending on backend type."""
        if getattr(self, "backend", DEFAULT_LLM_BACKEND) == "openai":
            choices = body.get("choices") or []
            if not isinstance(choices, list) or not choices:
                return ""
            message = (choices[0] or {}).get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return printable_text(content)
            if isinstance(content, list):
                parts: list[str] = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    block_text = block.get("text")
                    if isinstance(block_text, str) and block_text:
                        parts.append(printable_text(block_text))
                return "\n".join(parts)
            return ""
        return self.anthropic_response_text(body)

    async def request_with_retry(self, fn, *args, operation: str, **kwargs):
        """Execute one request, with unified retry on transient errors."""
        last_error: Exception | None = None
        for retry_idx in range(5):
            try:
                return await asyncio.to_thread(fn, *args, **kwargs)
            except Exception as exc:
                last_error = exc
                if not self.is_retryable_llm_error(exc) or retry_idx >= 4:
                    raise
                print(
                    f"[synthesize][warn] LLM temporary error during {operation}; retry {retry_idx + 1}/5 after 60s: {printable_text(exc)}",
                    flush=True,
                )
                await asyncio.sleep(60)
        assert last_error is not None
        raise last_error

    async def chat_once(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        usage_stage: str = "UNKNOWN",
    ) -> dict[str, Any]:
        """Make one single-round chat request to the specified backend, and record usage."""
        backend = getattr(self, "backend", DEFAULT_LLM_BACKEND)
        if backend == "openai":
            if tools:
                raise RuntimeError("OpenAI backend paths in v2 do not support tool calls")
            payload = {
                "model": self.model,
                "max_completion_tokens": LLM_MAX_TOKENS,
                "messages": self.openai_messages_from_chat(messages),
                "reasoning_effort": "medium",
            }
        elif backend in ("deepseek", "sglang"):
            # deepseek / sglang standard protocol (OpenAI Chat Completions): supports tool calls + automatic prefix cache.
            payload = {
                "model": self.model,
                "max_tokens": SGLANG_MAX_OUTPUT_TOKENS if backend == "sglang" else LLM_MAX_TOKENS,
                "messages": self.deepseek_messages_from_chat(messages),
                "stream": False,
            }
            if backend == "sglang":
                # Enable thinking (this deployment uses a boolean thinking, does not accept {"type":"enabled"});
                # reasoning_effort is always max; priority scheduling priority 100.
                payload["thinking"] = True
                payload["reasoning_effort"] = "max"
                payload["priority"] = SGLANG_PRIORITY
                # Context fallback: compute the budget first; if below 65536 -> first shrink history once, then clamp output by the remaining budget.
                def _budget():
                    approx = _approx_sglang_tokens(
                        self._sglang_prompt_fingerprint(payload["messages"])
                    )
                    return SGLANG_CONTEXT_LIMIT - approx - SGLANG_OUTPUT_SAFETY_MARGIN
                budget = _budget()
                if budget < SGLANG_MAX_OUTPUT_TOKENS:
                    if _shrink_sglang_prompt_once(payload["messages"]):
                        new_budget = _budget()
                        print(
                            f"[synthesize][sglang] prompt shrunk (90pct trim): "
                            f"budget {budget}→{new_budget} tokens",
                            flush=True,
                        )
                        budget = new_budget
                clamped = max(SGLANG_MIN_OUTPUT_TOKENS,
                              min(SGLANG_MAX_OUTPUT_TOKENS, budget))
                if clamped < payload["max_tokens"]:
                    print(
                        f"[synthesize][sglang] max_tokens clamped "
                        f"{payload['max_tokens']}→{clamped} (long prompt; floor {SGLANG_MIN_OUTPUT_TOKENS})",
                        flush=True,
                    )
                payload["max_tokens"] = clamped
            if tools is not None:
                decls = [d for d in (self.deepseek_tool(t) for t in tools) if d.get("function", {}).get("name")]
                if decls:
                    payload["tools"] = decls
                    payload["tool_choice"] = "auto"
        else:
            system, anthropic_messages = self.anthropic_messages_from_chat(messages)
            payload = {
                "model": self.model,
                "max_tokens": LLM_MAX_TOKENS,
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": LLM_THINKING_BUDGET_TOKENS,
                },
                "messages": anthropic_messages,
            }
            if system:
                payload["system"] = system
            if tools is not None:
                payload["tools"] = [self.anthropic_tool(tool) for tool in tools]

        outbound_error = find_invalid_surrogate(payload)
        if outbound_error is not None:
            raise RuntimeError(
                f"Refusing to submit non-standard {backend} JSON payload: {outbound_error}"
            )
        # Standard vendor authentication:
        #   - anthropic  → x-api-key + anthropic-version headers (official Messages API)
        #   - openai / local (OpenAI-compatible) → Authorization: Bearer <key>
        #     A local self-hosted server needing no auth simply leaves the key empty.
        headers = {"Content-Type": "application/json"}
        if backend == "anthropic":
            if self.api_key:
                headers["x-api-key"] = self.api_key
            headers["anthropic-version"] = self.anthropic_version
        else:
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
        body, response_headers = await self.request_with_retry(
            self.request_json_sync,
            "POST",
            self.endpoint_url,
            headers,
            payload=payload,
            timeout=self.request_timeout_sec,
            proxies=getattr(self, "request_proxies", None),
            operation=f"messages stage={usage_stage} backend={backend}",
        )
        # deepseek / sglang (local OpenAI-compatible) native responses are first
        # translated into Anthropic shape, then usage is recorded / tool loop fed.
        if backend == "deepseek":
            body = self.anthropic_body_from_deepseek(body)
        elif backend == "sglang":
            # The local server does not return cached_tokens: estimate the cache
            # hit from the longest common prefix vs the previous prompt, then reuse
            # the deepseek translation / cost path.
            cur_prompt_text = self._sglang_prompt_fingerprint(payload.get("messages") or [])
            body = self.anthropic_body_from_sglang(body, cur_prompt_text=cur_prompt_text)
        self.record_usage(body, stage=usage_stage)
        return body

    @staticmethod
    def tool_call_command(tool_call: dict) -> tuple[str, str, str | None]:
        """Parse a docker_cli command out of one tool_use block."""
        call_id = str(tool_call.get("id") or "").strip()
        name = str(tool_call.get("name") or "").strip()
        args = tool_call.get("input") or {}
        if not isinstance(args, dict):
            return call_id, "", f"invalid input for {name}: expected object"
        if name != "docker_cli":
            return call_id, "", f"unknown tool {name!r}; only docker_cli is available"
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            return call_id, "", "docker_cli requires non-empty string argument: command"
        return call_id, command, None

    class GenerationDockerCliSession:
        """A shared docker session reused across repeated docker_cli calls within a single generation."""

        def __init__(self, trace: dict, trim_tool_output):
            """Record the trace, trimmer, and container working directory and other session config."""
            self.trace = trace
            self.trim_tool_output = trim_tool_output
            self.cfg = trace["config"]
            self.cwd = self.cfg.get("cwd", "/testbed")
            self.session: DockerSession | None = None

        def start(self) -> None:
            """Start the container and replay to the "starting code state of the current generation session"."""
            if self.session is not None:
                return
            login_shell = not is_pro_instance(
                self.trace.get("instance"),
                image=self.cfg.get("image"),
                runner_kind=self.cfg.get("test_runner_kind"),
            )
            self.session = DockerSession(
                image=self.cfg["image"],
                keep_on_fail=False,
                login_shell=login_shell,
            )
            self.session.start()
            replay_events = [
                f"{ev.get('big_task_id')}/{ev.get('subtask_id')}"
                for ev in (self.trace.get("events") or [])
                if ev.get("kind") == "ATTEMPT"
                and ev.get("enabled", True)
                and (ev.get("result") or {}).get("ok") is True
            ]
            print(
                "[synthesize] docker_cli session start: "
                f"container={self.session.container_name} image={self.cfg.get('image')} "
                f"cwd={self.cwd} runner={self.cfg.get('test_runner_kind')} "
                f"login_shell={login_shell} replay_ok_attempts={replay_events}",
                flush=True,
            )
            bootstrap_session(self.session, self.trace)
            applied = self._replay_patches()
            print(
                "[synthesize] docker_cli session baseline ready: "
                f"container={self.session.container_name} applied_files={applied}",
                flush=True,
            )

        def _replay_patches(self) -> list:
            """Replay historically successful patches into the current session. Subclasses may override to limit the replay scope."""
            return replay_model_patches(self.session, self.trace, verbose=False)

        def run(self, command: str, *, timeout_sec: int = 150) -> dict:
            """Execute one docker_cli command in the shared container and return the observation.

            State accumulates across multiple calls within the same generation: if the previous command
            modified files (e.g. git apply), the next command will see the modified state.
            To return to the clean baseline state defined by the current generation session, call the reset_worktree tool.
            """
            self.start()
            assert self.session is not None
            try:
                proc = self.session.exec(f"cd {shlex.quote(self.cwd)} && {command}", timeout=timeout_sec)
            except subprocess.TimeoutExpired as exc:
                # A command timeout should not crash the entire attempt or burn retry quota: like returncode!=0,
                # feed "this command timed out" back to the model as a normal observation, so it retries with a
                # lighter command. The container stays untouched (the same session continues to be reused); the
                # timed-out process stays inside the container but does not affect subsequent exec.
                partial_out = exc.output or ""
                if isinstance(partial_out, bytes):
                    partial_out = partial_out.decode(errors="replace")
                partial_err = exc.stderr or ""
                if isinstance(partial_err, bytes):
                    partial_err = partial_err.decode(errors="replace")
                timeout_msg = (
                    f"docker_cli command timed out after {timeout_sec}s and was killed. "
                    f"The container session is preserved; pick a lighter/faster command "
                    f"(e.g. narrow the scope, add head/tail, avoid full-repo scans or hanging processes) and retry."
                )
                trimmed_out = self.trim_tool_output(str(partial_out))
                trimmed_err = self.trim_tool_output(str(partial_err))
                print(
                    "[synthesize] docker_cli exec "
                    f"rc=TIMEOUT timeout_sec={timeout_sec} stdout_len={len(str(partial_out))} "
                    f"stderr_len={len(str(partial_err))} cmd={one_line(command, limit=160)}",
                    flush=True,
                )
                tool_detail_log.log_text(
                    f"docker_cli exec container={self.session.container_name} cwd={self.cwd} "
                    f"returncode=TIMEOUT timeout_sec={timeout_sec} command={command}",
                    f"--- TIMEOUT after {timeout_sec}s ---\n"
                    f"--- partial stdout (raw_len={len(str(partial_out))}) ---\n{partial_out}\n\n"
                    f"--- partial stderr (raw_len={len(str(partial_err))}) ---\n{partial_err}",
                )
                return {
                    "tool": "docker_cli",
                    "command": command,
                    "container": self.session.container_name,
                    "cwd": self.cwd,
                    "image": self.cfg.get("image"),
                    "test_runner_kind": self.cfg.get("test_runner_kind"),
                    "returncode": None,
                    "timed_out": True,
                    "timeout_sec": timeout_sec,
                    "stdout": trimmed_out,
                    "stderr": (timeout_msg if not trimmed_err else trimmed_err + "\n" + timeout_msg),
                    "stdout_raw_len": len(str(partial_out)),
                    "stderr_raw_len": len(str(partial_err)),
                    "stdout_trimmed": len(trimmed_out) != len(str(partial_out)),
                    "stderr_trimmed": len(trimmed_err) != len(str(partial_err)),
                }
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            trimmed_stdout = self.trim_tool_output(stdout)
            trimmed_stderr = self.trim_tool_output(stderr)
            print(
                "[synthesize] docker_cli exec "
                f"rc={proc.returncode} stdout_len={len(stdout)} stderr_len={len(stderr)} "
                f"cmd={one_line(command, limit=160)}",
                flush=True,
            )
            tool_detail_log.log_text(
                f"docker_cli exec container={self.session.container_name} cwd={self.cwd} "
                f"returncode={proc.returncode} command={command}",
                f"--- stdout (raw_len={len(stdout)}) ---\n{stdout}\n\n"
                f"--- stderr (raw_len={len(stderr)}) ---\n{stderr}",
            )
            return {
                "tool": "docker_cli",
                "command": command,
                "container": self.session.container_name,
                "cwd": self.cwd,
                "image": self.cfg.get("image"),
                "test_runner_kind": self.cfg.get("test_runner_kind"),
                "returncode": proc.returncode,
                "stdout": trimmed_stdout,
                "stderr": trimmed_stderr,
                "stdout_raw_len": len(stdout),
                "stderr_raw_len": len(stderr),
                "stdout_trimmed": len(trimmed_stdout) != len(stdout),
                "stderr_trimmed": len(trimmed_stderr) != len(stderr),
            }

        def reset(self) -> dict:
            """Destroy the current container; on the next run() it is rebuilt from the baseline.

            More reliable than filesystem-snapshot restore: does not depend on in-container tar/mv operations and will not hang.
            On the next run() call, the container is automatically rebuilt and historically successful patches are replayed.
            """
            if self.session is not None:
                session = self.session
                self.session = None
                print(
                    "[synthesize] reset_worktree: stopping current docker_cli container "
                    f"container={session.container_name} image={self.cfg.get('image')} "
                    f"cwd={self.cwd} runner={self.cfg.get('test_runner_kind')}",
                    flush=True,
                )
                try:
                    session.stop(timeout=30)
                except Exception as exc:
                    print(
                        f"[synthesize] warning: reset_worktree container stop failed"
                        f" — {printable_text(exc)}",
                        flush=True,
                    )
            return {
                "tool": "reset_worktree",
                "status": "ok",
                "image": self.cfg.get("image"),
                "cwd": self.cwd,
                "test_runner_kind": self.cfg.get("test_runner_kind"),
                "message": (
                    "Container destroyed. Worktree will be rebuilt from the clean "
                    "baseline for this generation session on the next docker_cli call."
                ),
            }

        def close(self) -> None:
            """Forcibly destroy the container; directly docker rm -f."""
            if self.session is None:
                return
            session = self.session
            try:
                try:
                    session.stop(timeout=60)
                except Exception as exc:
                    print(
                        f"[synthesize] warning: docker rm -f timed out or failed"
                        f" — container {session.container_name} may leak: "
                        f"{printable_text(exc)}",
                        flush=True,
                    )
            finally:
                self.session = None

    async def call_json_with_docker_cli_messages(
        self,
        messages: list[dict],
        *,
        trace: dict,
        enable_docker_cli: bool,
        event_kind: str,
        max_tool_calls: int,
        remind_every_tool_calls: int,
        remind_start_tool_calls: int = 0,
        wrap_up_reminder: bool = False,
        reminder_context: str = "",
        docker_cli_session_cls=None,
    ) -> Any:
        """Reuse the shared docker_cli session main loop from utils.

        `docker_cli_session_cls` defaults to `GenerationDockerCliSession` (the standard generation baseline
        where all successful ATTEMPT model patches are applied and test_patch is not applied). augment
        generation needs an environment that "additionally applies the current ATTEMPT's test_patch", which
        the caller can pass in as a custom subclass override; when None, it falls back to the default behavior.
        """
        tool_list = PLAN_DOCKER_CLI_TOOL if event_kind == "PLAN" else IMPL_DOCKER_CLI_TOOL
        # Only expose read_skills / patch_skills when the skills directory has been injected and the current stage is a skills stage.
        skills_manager = None
        if (
            self.skills_dir is not None
            and self.skills_history_dir is not None
            and stage_file_for_event_kind(event_kind) is not None
        ):
            _inst = trace.get("instance")
            _instance_id = (
                _inst.get("instance_id") if isinstance(_inst, dict) else _inst
            ) or "unknown"
            skills_manager = SkillsManager(
                skills_dir=self.skills_dir,
                history_dir=self.skills_history_dir,
                instance=str(_instance_id),
            )
            tool_list = list(tool_list) + _SKILLS_TOOLS
        return await gateway_call_json_with_docker_cli_messages(
            messages=messages,
            trace=trace,
            enable_docker_cli=enable_docker_cli,
            event_kind=event_kind,
            max_tool_calls=max_tool_calls,
            remind_every_tool_calls=remind_every_tool_calls,
            remind_start_tool_calls=remind_start_tool_calls,
            wrap_up_reminder=wrap_up_reminder,
            reminder_context=reminder_context,
            # deepseek/sglang uses the OpenAI Chat Completions protocol; putting the reminder text and the
            # tool_result in the same user message breaks the assistant(tool_calls)->tool adjacency constraint, so disable reminder injection.
            suppress_reminders=self.backend in ("deepseek", "sglang"),
            chat_once=self.chat_once,
            client_response_text=self.client_response_text,
            tool_call_command=self.tool_call_command,
            log_step=self.log_step,
            trim_tool_output=self.trim_tool_output,
            docker_cli_tool=tool_list,
            docker_cli_session_cls=docker_cli_session_cls or self.GenerationDockerCliSession,
            skills_manager=skills_manager,
        )
