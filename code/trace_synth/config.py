#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime configuration loader for the trace-synthesis engine.

All deployment-specific values (API endpoints, model names, credentials env var
names, dataset/docker paths) live in ``config.yaml`` next to this file so the
open-source code carries no hard-coded internal endpoints or absolute paths.

Precedence (highest first):
    1. environment variable (when a field declares an ``*_env`` name)
    2. value in ``config.yaml``
    3. the built-in default in this module

Only :func:`load_config` reads the YAML; the rest of the pipeline imports the
resolved values from here, so there is a single source of truth.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception as _exc:  # pragma: no cover - surfaced lazily on first use
    yaml = None  # noqa: N816
    _YAML_IMPORT_ERROR = _exc
else:
    _YAML_IMPORT_ERROR = None

HERE = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("SWEJOURNEY_SYNTH_CONFIG", HERE / "config.yaml"))

# ── built-in defaults (used when config.yaml is absent or a key is missing) ──
# Endpoints default to the public vendor APIs; the local model defaults to a
# localhost OpenAI-compatible server. Override any of these in config.yaml.
_DEFAULTS: dict[str, Any] = {
    "clients": {
        # Main client: Anthropic official Messages API.
        "main": {
            "backend": "anthropic",
            "base_url": "https://api.anthropic.com/v1/messages",
            "model": "claude-opus-4-1",
            "api_key_env": "ANTHROPIC_API_KEY",
            "anthropic_version": "2023-06-01",
        },
        # JSON-repair client: OpenAI official Chat Completions API.
        "repair": {
            "backend": "openai",
            "base_url": "https://api.openai.com/v1/chat/completions",
            "model": "gpt-4o",
            "api_key_env": "OPENAI_API_KEY",
        },
        # First-call (cheap) client: a self-hosted / local OpenAI-compatible
        # server (e.g. vLLM / SGLang). Points at localhost by default.
        "first_call": {
            "backend": "local",
            "base_url": "http://127.0.0.1:8000/v1/chat/completions",
            "model": "local-model",
            "api_key_env": "LOCAL_API_KEY",
        },
    },
    "paths": {
        "dataset": "",
        "full_dataset": "",
        "subset_jsonl": "",
        "docker_pro_dir": "",
        "docker_verified_dir": "",
        "hf_home": "",
    },
    "runtime": {
        "python": "",   # empty → sys.executable
        "proxy": "",     # empty → no proxy for docker containers
        "usage": "1",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Return a copy of ``base`` with ``override`` merged in recursively."""
    out = dict(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_config(path: str | Path | None = None) -> dict:
    """Load and merge ``config.yaml`` over the built-in defaults.

    Missing file → defaults only. Missing keys → filled from defaults.
    """
    cfg_path = Path(path) if path else CONFIG_PATH
    data: dict = {}
    if cfg_path.is_file():
        if yaml is None:
            raise RuntimeError(
                "pyyaml is required to read config.yaml but is not installed "
                f"(import error: {_YAML_IMPORT_ERROR}). Install it via "
                "`pip install pyyaml` or delete config.yaml to use defaults."
            )
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    return _deep_merge(_DEFAULTS, data)


def _resolve_env(value: str) -> str:
    """Expand ``${ENV}`` / ``$ENV`` references inside a config string."""
    if not isinstance(value, str) or "$" not in value:
        return value
    return os.path.expandvars(value)


def client_config(name: str, cfg: dict | None = None) -> dict:
    """Return the resolved config for one client, with the API key looked up.

    The returned dict adds an ``api_key`` field resolved from the env var named
    by ``api_key_env`` (empty string if unset). ``base_url``/``model`` support
    ``${ENV}`` expansion so a single yaml can be reused across deployments.
    """
    cfg = cfg or load_config()
    raw = dict((cfg.get("clients") or {}).get(name) or {})
    raw.setdefault("backend", "openai")
    for key in ("base_url", "model", "anthropic_version"):
        if key in raw and isinstance(raw[key], str):
            raw[key] = _resolve_env(raw[key])
    key_env = str(raw.get("api_key_env") or "").strip()
    raw["api_key"] = os.environ.get(key_env, "") if key_env else ""
    return raw


def path_config(cfg: dict | None = None) -> dict:
    """Return resolved deployment paths (env-expanded)."""
    cfg = cfg or load_config()
    out = dict((cfg.get("paths") or {}))
    for key, val in list(out.items()):
        out[key] = _resolve_env(val) if isinstance(val, str) else val
    return out


def runtime_config(cfg: dict | None = None) -> dict:
    """Return resolved runtime settings (python interpreter, proxy, usage)."""
    cfg = cfg or load_config()
    out = dict((cfg.get("runtime") or {}))
    for key, val in list(out.items()):
        out[key] = _resolve_env(val) if isinstance(val, str) else val
    return out
