#!/usr/bin/env python3
"""Runtime path configuration for trace_clean.

Precedence for every path (highest first):
    1. the shared environment variable (SWEJOURNEY_*), when set
    2. the matching key in ``config.yaml`` next to this file
    3. the built-in default, resolved relative to this file (so the tree is portable)

This mirrors the two-tier scheme used by trace_synth/config.py, so all four
subdirs are configured the same way (a per-subdir ``config.yaml`` + SWEJOURNEY_*
env overrides). Nothing here is secret; the file carries no absolute paths.

Directory layout (relative to SWEJourney/code/trace_clean/):

    ../trace_synth          synthesis engine (this package reuses its 5 low-level modules; no copy is vendored)
    ../../data/trace        raw agent_trace.v2 trace input (where trace_synth's output lands)
    ../../data/_base_seed_probe  base+SEED-gold starting-state probe results (intermediate, produced by this package's probe)
    ../../data/_allgold_probe    all-gold final-state probe results (intermediate, produced by this package's probe)
    ../../benchmark         final cleaned output: 30 clean JSON files (one per task)

config.yaml keys (all under ``paths:``, all optional): trace_synth_dir /
data_dir / trace_in_dir / benchmark_dir.

Environment overrides (take precedence over config.yaml):
    SWEJOURNEY_CLEAN_CONFIG  override the config.yaml path itself
    SWEJOURNEY_TRACE_IN      override the raw trace input directory
    SWEJOURNEY_DATA_DIR      override the intermediate (probe) output root
    SWEJOURNEY_BENCHMARK_DIR override the benchmark output root
    SWEJOURNEY_SYNTH_DIR     override the trace_synth dependency directory
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - surfaced lazily on first use
    yaml = None  # noqa: N816

HERE = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("SWEJOURNEY_CLEAN_CONFIG", HERE / "config.yaml"))


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


# -- Dependency: the synthesis-engine directory (reuses its swebench_dockerlib /
#    local_test_executor / agent_trace / swe_bench_*_runner; sys.path points here on import) --
TRACE_SYNTH_DIR = _pick("SWEJOURNEY_SYNTH_DIR", "trace_synth_dir", HERE / ".." / "trace_synth")

# -- Data root: raw trace input plus the two probe intermediates all live under data/ --
DATA_DIR = _pick("SWEJOURNEY_DATA_DIR", "data_dir", HERE / ".." / ".." / "data")
TRACE_IN_DIR = _pick("SWEJOURNEY_TRACE_IN", "trace_in_dir", DATA_DIR / "trace")
SEEDPROBE_DIR = DATA_DIR / "_base_seed_probe"
ALLGOLD_DIR = DATA_DIR / "_allgold_probe"

# -- Output: benchmark root (holds only the final cleaned 30 clean JSON files) --
# Same env var as evaluation/config.py so a single override keeps both stages in sync.
BENCH_OUT_DIR = _pick("SWEJOURNEY_BENCHMARK_DIR", "benchmark_dir", HERE / ".." / ".." / "benchmark")


def ensure_synth_on_path() -> None:
    """Insert the trace_synth directory at the head of sys.path so probes can import its low-level modules."""
    import sys
    p = str(TRACE_SYNTH_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
