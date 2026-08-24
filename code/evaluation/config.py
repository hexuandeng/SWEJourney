#!/usr/bin/env python3
"""evaluation runtime path config.

Precedence for every path (highest first):
    1. the shared environment variable (SWEJOURNEY_*), when set
    2. the matching key in ``config.yaml`` next to this file
    3. the built-in default, resolved relative to this file (so the tree is portable)

This mirrors the two-tier scheme used by trace_synth/config.py and
trace_clean/config.py, so all four subdirs are configured the same way (a
per-subdir ``config.yaml`` + SWEJOURNEY_* env overrides). Nothing here is secret.

Directory conventions (relative to SWEJourney/code/evaluation/):

    ../../benchmark               the sole evaluation input: final_f2p/final_p2p (+ own_* in events)
                                  —— the denominators/calibration for D1/D3/D4 are all reverse-derived from here (no longer reading probes under data/)
    ../../data/usersim_final      root of user-sim run outputs: <model>/<instance>/<persona>/
                                  {final_eval,bigtask_eval}.json (consumed by the scorer)
    ../../data/                   root for cost cache / csv / md outputs

config.yaml keys (all under ``paths:``, all optional): data_dir / benchmark_dir.

Environment overrides (take precedence over config.yaml):
    SWEJOURNEY_EVAL_CONFIG     override the config.yaml path itself
    SWEJOURNEY_DATA_DIR        override the data root (usersim_final / cost cache)
    SWEJOURNEY_BENCHMARK_DIR   override the benchmark root
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - surfaced lazily on first use
    yaml = None  # noqa: N816

HERE = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("SWEJOURNEY_EVAL_CONFIG", HERE / "config.yaml"))


def _yaml_paths() -> dict:
    """Return the ``paths:`` mapping from config.yaml (empty dict if absent)."""
    if yaml is None or not CONFIG_PATH.is_file():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return dict(data.get("paths") or {})


_P = _yaml_paths()


def _pick(env: str, yaml_key: str, default: Path) -> Path:
    """env var > config.yaml paths.<yaml_key> > built-in default."""
    val = os.environ.get(env) or (_P.get(yaml_key) or "")
    return Path(val).resolve() if val else Path(default).resolve()


DATA_DIR = _pick("SWEJOURNEY_DATA_DIR", "data_dir", HERE / ".." / ".." / "data")

# benchmark is the sole evaluation input: the denominators/calibration for D1/D3/D4 (bad-node exclusion,
# p2p→f2p reclassification) are all reverse-derived from final_f2p/final_p2p (+ own_* in events) of
# benchmark/*.json, no longer reading the intermediate-state probes under data/.
BENCHMARK_DIR = _pick("SWEJOURNEY_BENCHMARK_DIR", "benchmark_dir", HERE / ".." / ".." / "benchmark")

TRACES_DIR = BENCHMARK_DIR       # count_traces counts the number of files; the own_* reverse-derived by both loaders also come from benchmark's events
ALLGOLD_DIR = BENCHMARK_DIR      # load_allgold_exclude reverse-derives the bad-node set from benchmark (full set of events own_* − final lists)
SEEDPROBE_DIR = BENCHMARK_DIR    # load_seedprobe_reclass reverse-derives the reclassification set from benchmark (own_p2p ∩ final_f2p)
USERSIM_OUT_DIR = DATA_DIR / "usersim_final"
