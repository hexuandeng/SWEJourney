#!/usr/bin/env python3
"""Runtime path & endpoint configuration for user_simulation (all resolved relative to this file, so the tree can be relocated as a whole).

Directory layout (relative to SWEJourney/code/user_simulation/):

    ../trace_synth          synthesis engine (reuses its agent_trace / local_test_executor /
                            swebench_dockerlib / swe_bench_*_runner low-level modules)
    ../mini_swe_agent       vendored minisweagent (assistant-server engine + docker backend)
    ../../data/trace        raw trace input (--trace points at one task inside it)
    ../../data/usersim_final run output root (final_eval.json / bigtask_eval.json, consumed by the scorer)
    ./prompts               4 persona sim_*.yaml + user_sim_call / persona_dimensions
    ./logs                  user-sim conversation log root
    ./config.yaml           endpoints & model configuration (single source of truth)

Endpoints, model names, gateway URL and proxy all live in config.yaml — fill in
your own OpenAI-compatible endpoints there (or pass --assistant-model name@base_url
freeform at runtime).

Environment overrides:
    SWEJOURNEY_SYNTH_DIR    override the trace_synth dependency directory
    SWEJOURNEY_MINISWE_DIR  override the mini_swe_agent directory
    SWEJOURNEY_DATA_DIR     override the data root (trace input + usersim_final output)
    SWEJOURNEY_USERSIM_CONFIG  override the config.yaml path
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# -- Dependency directories --
TRACE_SYNTH_DIR = Path(os.environ.get("SWEJOURNEY_SYNTH_DIR", HERE / ".." / "trace_synth")).resolve()
MINISWE_DIR = Path(os.environ.get("SWEJOURNEY_MINISWE_DIR", HERE / ".." / "mini_swe_agent")).resolve()

# -- Data root --
DATA_DIR = Path(os.environ.get("SWEJOURNEY_DATA_DIR", HERE / ".." / ".." / "data")).resolve()
TRACE_IN_DIR = DATA_DIR / "trace"
USERSIM_OUT_DIR = DATA_DIR / "usersim_final"

# -- In-package resources --
PROMPTS_DIR = HERE / "prompts"
LOG_ROOT = HERE / "logs"
ENDPOINTS_FILE = Path(os.environ.get("SWEJOURNEY_USERSIM_CONFIG", HERE / "config.yaml")).resolve()


def setup_paths() -> None:
    """Insert the dependency directories at the head of sys.path so this package's modules can import their low-level deps.

    - trace_synth: agent_trace / local_test_executor / swebench_dockerlib /
      swe_bench_*_runner / user_simulator, etc.;
    - mini_swe_agent: minisweagent.* (the assistant-server engine) — skipped when
      ``minisweagent`` is already importable (e.g. after ``pip install -e .``);
    - this package itself (HERE): usersim_scenario / usersim_driver / agent subpackage / user_simulator, etc.
    Also requires the swebench package (swebench.harness.*, see ../trace_synth/requirements.txt).
    """
    import importlib.util
    dirs = [str(TRACE_SYNTH_DIR), str(HERE)]
    # Only vendor the bundled mini_swe_agent onto sys.path if it isn't installed.
    if importlib.util.find_spec("minisweagent") is None:
        dirs.insert(0, str(MINISWE_DIR))
    for p in dirs:
        if p not in sys.path:
            sys.path.insert(0, p)


_ENDPOINTS_CACHE: dict | None = None


def load_endpoints() -> dict:
    """Load config.yaml (assistant-model registry, gateway defaults, proxy).

    Returns a dict with keys ``assistant_models`` / ``gateway`` / ``docker_proxy``.
    A missing file or missing keys degrade to empty defaults, so the caller's own
    fallbacks (env vars / CLI args) still apply. Result is cached per process.
    """
    global _ENDPOINTS_CACHE
    if _ENDPOINTS_CACHE is not None:
        return _ENDPOINTS_CACHE
    data: dict = {}
    try:
        import yaml
        with open(ENDPOINTS_FILE, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        data = {}
    data.setdefault("assistant_models", {})
    data.setdefault("gateway", {})
    data.setdefault("docker_proxy", "")
    _ENDPOINTS_CACHE = data
    return data
