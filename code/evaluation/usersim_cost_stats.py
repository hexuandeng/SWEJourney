#!/usr/bin/env python3
"""Compute per-cell token usage / wall-clock time / ¥ estimate for the user-sim closed-loop evaluation.

Data source: data/usersim_final/<model>/<instance>/<persona>/assistant/*.assistant.traj.json
  Each message's extra.response.usage = {prompt_tokens, completion_tokens,
  cached_tokens(inside prompt_tokens_details)}; extra.response.timestamp = epoch seconds.
  info.model_stats is untrustworthy (hardcodes api_calls=2); we MUST walk messages and accumulate one by one.

Conventions (confirmed by user):
  - Emit both the raw token counts AND the ¥ estimate;
  - time = wall-clock span (difference between the first and last call timestamp of the cell);
  - grouping = model × persona, averaged over instances.
  - [Convention A] count only the assistant-side (evaluated model) tokens; the user-sim side is not counted.
  - [Convention Y] avg_cost_usd() only counts completed cells that have a final_eval.json (same denominator as the six-dimension table).

Unit price: official real prices (see the URLs and the A/B/C reliability grading in evaluation/COST_ESTIMATION_NOTES.md).
  Single source of truth = PRICE_CNY below (yuan / million tokens); USD = ¥ ÷ 6.8.
  2026-08-05: model directory names have been changed to clean alias names, count increased to 12, price table rebuilt for the current 12 keys.

cache fitting (explicitly stated by user 2026-08-05, applied uniformly to all models):
  every time the user-sim actually speaks → the cache prefix is cleared; the i-th call within a turn hits
  = min(prompt_i, prompt_{i-1}+completion_{i-1}); the framework-injected "Tool call error:"
  format-retries do NOT count as a new turn and do NOT clear the prefix (they are retries within the same turn).

Reusable interface (for the "$/cell" column of evaluation/stat_all_personas_final.py):
  avg_cost_usd(models=..., personas=...) -> {(model,persona): {avg_usd,n,total_usd}}

Usage:
  python evaluation/usersim_cost_stats.py
  # optional: --src <dir> --csv <out.csv> --workers N
"""
import argparse
import csv
import json
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Relative-path wiring: ROOT = SWEJourney/ (this file is at SWEJourney/code/evaluation/);
# data/usersim_final / data/*cost* cache etc. follow the ROOT/data semantics, consistent with config.DATA_DIR.
ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "data" / "usersim_final"
USD_CNY = 6.8

# ============================ unit price (single source of truth) ============================
# User convention: use the official-site [USD] price, computing the cache discount. So PRICE_USD is the only source of truth, ¥ = $ × 6.8.
# Three-tier (cache_hit, cache_miss_input, output), USD / million tokens. Billing (real-bill convention):
#   cost = eff_cached*hit + (prompt - eff_cached)*miss + completion*out
# ⚠️ 2026-08-05: directory names changed to clean alias names, models 7→12.
#   User clarified these are [public, real models, just renamed with aliases] → all prices were looked up online (not fabricated).
# Source URLs and the A/B/C reliability grading are in evaluation/COST_ESTIMATION_NOTES.md.
#   Grade A · officially confirmed item by item: Flash / V4-Pro / Kimi-K2.7 / MiMo / Hy3
#     — V4-Pro & MiMo official ¥3/¥0.025/¥6 → ÷6.8 = 0.4167/0.003472/0.8333
#       (the earlier synth table mistakenly wrote 0.435/0.87, now corrected to the official ¥).
#     — Kimi is officially priced in USD directly, $0.19/$0.95/$4.0 (K2.7-Code).
#   Grade B · official + third-party: Qwen3.5-397B / Qwen3.5-122B / Qwen3.6-35B (output price / hit involves third-party or estimation).
#   Grade C · officially unconfirmed / version uncertain (placeholder, to be decided): GLM-5.1 / GLM-5.2 / Hy3-Preview / gpt-5.6-sol.
PRICE_USD: dict[str, tuple] = {
    # (cache_hit, cache_miss_input, output)  —— USD / 1M tokens
    "DeepSeek-V4-Flash":   (0.007,     0.22,    0.66),       # off-peak official USD price(2026-08): hit 0.007/miss 0.22/output 0.66
    "DeepSeek-V4-Pro":     (0.022,     0.66,    1.98),       # off-peak official USD price(2026-08): hit 0.022/miss 0.66/output 1.98
    "Kimi-K2.7":           (0.19,      0.95,    4.00),       # A official USD K2.7-Code
    "MiMo-V2.5-Pro":       (0.0036,    0.435,   0.87),       # A official ¥0.025/¥3/¥6 ÷6.8
    "Hy3":                 (0.033,     0.132,   0.528),      # A official ¥0.25/¥1/¥4 ÷6.8
    "Qwen3.5-397B-A17B":   (0.022,     0.111,   3.60),       # B input official ¥0.8; output Novita $3.6
    "Qwen3.5-122B-A10B":   (0.06,      0.30,    2.40),       # B Novita third-party convention
    "Qwen3.6-35B-A3B":     (0.006,     0.028,   0.28),       # B Alibaba Flash tier; hit approx 20%-off estimate
    "GLM-5.1":             (0.26,      1.40,    4.40),       # C uncertain placeholder
    "GLM-5.2":             (0.26,      1.40,    4.40),       # C uncertain placeholder
    "Hy3-Preview":         (0.033,     0.132,   0.528),      # C preview-version price inferred
    "gpt-5.6-sol":         (0.50,      5.00,    30.0),       # C GPT-5 two price sets, uncertain
    "Claude-Opus-5":       (0.50,      5.00,    25.0),       # official Opus price(provided by user 2026-08-07): cache-read 0.5/input 5/output 25; cache-write (10/6.25) not counted for this model
}
# Derived CNY prices (yuan / million tokens, same hit/miss/out order), for legacy ¥ reports; = $ × 6.8.
PRICE_CNY: dict[str, tuple] = {
    m: tuple(round(x * USD_CNY, 4) for x in usd) for m, usd in PRICE_USD.items()
}
# Unit price contains approximate components / officially unconfirmed → mark ~ in report (see Grade C in COST_ESTIMATION_NOTES.md)
PRICE_APPROX = {"Qwen3.5-397B-A17B", "Qwen3.5-122B-A10B", "Qwen3.6-35B-A3B",
                "GLM-5.1", "GLM-5.2", "Hy3-Preview", "gpt-5.6-sol"}

# ---------------- cache-write unit price (USD / million tokens) ----------------
# Anthropic 5m ephemeral official premium: write = miss × 1.25; for 1h ephemeral it is × 2.5.
# Other channels (including OpenAI /v1/responses) whose gateway does not return a separate cache_creation_input_tokens
# field → treated as write=miss (no premium, equivalently folded into miss).
# Models not appearing in this table automatically fall back to PRICE_USD[model][1] (miss unit price).
WRITE_PRICE_USD: dict[str, float] = {
    "Claude-Opus-5": 6.25,   # 5m ephemeral official premium (yaml cache_ttl:"5m")
    # other models not listed → write_p = miss_p, see _write_price()
}

# ---------------- read/write/miss steady-state ratios (as a fraction of the full prompt) ----------------
# Billing convention: cost = prompt_tot * (read_pct*hit + write_pct*write_p + miss_pct*miss)
#                 + out_tot * out_p
# Ratio source: per-traj find the position k of "first cache_read>0", and the steady-state-segment ratios accumulated from k onward
# (excluding the polluted segment where the pre-08-08 old code's /v1/chat channel had read constantly 0).
#
# Models not in this table → follow the _scan_traj_cost fitting result (cached/prompt as read_pct,
# write_pct=0), behavior exactly equivalent to the old formula cached*hit + fresh*miss + out*outp.
CACHE_RATIO: dict[str, tuple] = {
    # (read_pct, write_pct)  —— miss_pct = 1 - read - write
    "Claude-Opus-5": (0.7564, 0.1875),   # measured on the steady-state segment of 23 done cells (the write>0 segment comes from Anthropic's official fields)
    "gpt-5.6-sol":   (0.8457, 0.0000),   # measured on the steady-state segment of 39 done cells (responses channel has no separate write)
}


def _write_price(model: str) -> float:
    """This model's write (cache_creation) unit price (USD / 1M tokens). Default = miss unit price."""
    if model in WRITE_PRICE_USD:
        return WRITE_PRICE_USD[model]
    return PRICE_USD[model][1]  # miss


def _cost_usd(model: str, cached: int, fresh: int, out: int) -> float:
    """Unified cost formula: tokens use the fitting from _scan_traj_cost (cached/fresh/out),
    then per the steady-state ratios in CACHE_RATIO extrapolate prompt_tot into the three buckets read/write/miss.

    * model ∈ CACHE_RATIO → use the hardcoded ratio anchors (Opus5 / gpt-5.6-sol);
    * model ∉ CACHE_RATIO → fitted ratios: read_pct = cached/prompt_tot, write_pct=0,
      in which case the result is exactly equivalent to the old formula cached*hit + fresh*miss + out*outp.
    """
    hit, miss, outp = PRICE_USD[model]
    write_p = _write_price(model)
    prompt_tot = cached + fresh
    if prompt_tot <= 0:
        return out * outp / 1e6
    if model in CACHE_RATIO:
        read_pct, write_pct = CACHE_RATIO[model]
        miss_pct = max(0.0, 1.0 - read_pct - write_pct)
    else:
        # fitted ratios: equivalent to the original formula
        read_pct = cached / prompt_tot
        write_pct = 0.0
        miss_pct = fresh / prompt_tot
    read_tok = prompt_tot * read_pct
    write_tok = prompt_tot * write_pct
    miss_tok = prompt_tot * miss_pct
    return (read_tok * hit + write_tok * write_p + miss_tok * miss
            + out * outp) / 1e6

# User explicitly required on 2026-08-05: cache hit is [uniformly] computed via the fitting principle (clear on
# each user-sim arrival, reuse prefix within a turn via min(prompt_i, prev_prompt+prev_completion)), applied to all models——
# no longer distinguishing "whether the backend returns cached_tokens". So the estimation set = all models.
# (The original logic only did structural estimation for Flash/V4Pro and used backend measurements for the rest; now uniformly fitted per user's convention.)
ESTIMATED_CACHE_MODELS = set(PRICE_CNY.keys())

# Display names on the page (kept consistent with render_all_usersim_final.DISPLAY_NAME).
# 2026-08-05: directory names are already clean alias names, display name matching the key is enough (mapping kept in case a few need beautifying).
DISPLAY_NAME = {
    "DeepSeek-V4-Flash":   "DeepSeek-V4-Flash",
    "DeepSeek-V4-Pro":     "DeepSeek-V4-Pro",
    "Hy3":                 "Hy3",
    "Hy3-Preview":         "Hy3-Preview",
    "Kimi-K2.7":           "Kimi-K2.7",
    "MiMo-V2.5-Pro":       "MiMo-V2.5-Pro",
    "GLM-5.1":             "GLM-5.1",
    "GLM-5.2":             "GLM-5.2",
    "Qwen3.5-397B-A17B":   "Qwen3.5-397B-A17B",
    "Qwen3.5-122B-A10B":   "Qwen3.5-122B-A10B",
    "Qwen3.6-35B-A3B":     "Qwen3.6-35B-A3B",
    "gpt-5.6-sol":         "gpt-5.6-sol",
}
PERSONA_LABEL = {
    "non_programming_user":    "Non-programming professional",
    "product_stakeholder":     "Product / business requester",
    "repo_newcomer_engineer":  "Engineering-capable, new to this repo",
    "repository_steward":      "Repository steward, architecture gatekeeper",
}


def _msg_text(m: dict) -> str:
    """Get a message's text content (content may be a str or a [{text:...}] list)."""
    c = m.get("content")
    if isinstance(c, list):
        return " ".join(x.get("text", "") for x in c if isinstance(x, dict))
    return c if isinstance(c, str) else ""


# Prefix of the framework-injected "format-error retry" message (when the model does not output tool_call in the
# right format, the framework injects a message with role=="user" reporting the error to make it retry). These are
# not real user-sim utterances and MUST be excluded when counting "turns", but they did trigger real API calls and
# really burned tokens, so cost/tokens are still counted in full.
RETRY_PREFIX = "Tool call error:"


def scan_cell(args) -> dict:
    """Parse a single assistant traj, returning that cell's usage/time/cost/turn count.

    done=True only when a final_eval.json exists in the same directory (the truly-finished marker, see task3 memory).
    Unfinished/half-done cells still have their usage parsed but are filtered out in the aggregation.

    A turn = user-sim sends one request + agent replies once (which may contain multiple tool calls in between).
    Determination: role=="user" with no tool_call_id and content not starting with "Tool call error:".
    """
    model, instance, persona, path = args
    # final_eval.json / bigtask_eval.json are at the same level as assistant/ (under the persona directory)
    pdir = Path(path).parent.parent
    fin_path = pdir / "final_eval.json"
    big_path = pdir / "bigtask_eval.json"
    done = fin_path.exists()

    # ── accuracy (convention confirmed by user 2026-07-20) ──
    # Final accuracy: top-level n_f2p_pass/total, n_p2p_pass/total in final_eval.json (uniform re-test after everything is done).
    # Intermediate accuracy: bigtask_eval.json evaluates each big_task on the spot at completion, accumulating pass and total separately for f2p/p2p
    #   (evaluated_at_turn snapshot, reflecting the instantaneous state "right when that block was just finished"; compare with the final to see regression).
    # total=0 means the cell has no corresponding data (not counted in the denominator).
    fin_f2p_p = fin_f2p_t = fin_p2p_p = fin_p2p_t = 0
    mid_f2p_p = mid_f2p_t = mid_p2p_p = mid_p2p_t = 0
    if fin_path.exists():
        try:
            fe = json.load(open(fin_path))
            fin_f2p_p = fe.get("n_f2p_pass", 0) or 0
            fin_f2p_t = fe.get("n_f2p_total", 0) or 0
            fin_p2p_p = fe.get("n_p2p_pass", 0) or 0
            fin_p2p_t = fe.get("n_p2p_total", 0) or 0
        except Exception as e:
            sys.stderr.write(f"[warn] final_eval {model}/{instance}/{persona}: {e!r}\n")
    if big_path.exists():
        try:
            be = json.load(open(big_path))
            for bt in be.get("big_tasks", []) or []:
                if not isinstance(bt, dict):
                    continue
                mid_f2p_p += bt.get("n_f2p_pass", 0) or 0
                mid_f2p_t += bt.get("n_f2p_total", 0) or 0
                mid_p2p_p += bt.get("n_p2p_pass", 0) or 0
                mid_p2p_t += bt.get("n_p2p_total", 0) or 0
        except Exception as e:
            sys.stderr.write(f"[warn] bigtask_eval {model}/{instance}/{persona}: {e!r}\n")
    tin = tout = tcached = ncall = 0
    user_turns = 0      # count of real user-sim utterance turns (excluding format retries)
    retry_errors = 0    # count of framework-injected "Tool call error:" retries
    ts_min = ts_max = None
    ok = True

    # ── the per-turn cache-clearing estimation convention (only used for Flash/V4Pro) ──
    # Physical model: during the wait before the user-sim returns, concurrent pressure evicts this cell's KV prefix
    #   (LRU eviction by vLLM APC / SGLang RadixAttention) → cross-turn cache goes to zero;
    #   only consecutive tool calls within the same turn share the prefix. So reset the prefix baseline at every real user-sim turn.
    # The i-th call within a turn can hit the prefix = min(prompt_i, prev_prompt + prev_completion).
    prev_in_turn = None   # the (prompt+completion) token count of the previous call in this turn
    est_cached = 0
    try:
        d = json.load(open(path))
        for m in d.get("messages", []):
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            # Turn boundary: a real user-sim utterance → a new turn, cache prefix cleared.
            if role == "user" and not m.get("tool_call_id"):
                if _msg_text(m).lstrip().startswith(RETRY_PREFIX):
                    retry_errors += 1        # format retry, not a real turn; do not reset the cache prefix
                else:
                    user_turns += 1
                    prev_in_turn = None      # new turn → cross-turn cache goes to zero
            ex = m.get("extra") or {}
            r = ex.get("response")
            if r:
                u = r.get("usage")
                if u:
                    ncall += 1
                    p = u.get("prompt_tokens", 0) or 0
                    c = u.get("completion_tokens", 0) or 0
                    tin += p
                    tout += c
                    pd = u.get("prompt_tokens_details") or {}
                    tcached += (pd.get("cached_tokens") or 0)
                    if prev_in_turn is not None:
                        est_cached += min(p, prev_in_turn)
                    prev_in_turn = p + c
                # timestamp is in extra.timestamp (epoch seconds); fall back to response.created.
                t = ex.get("timestamp") or r.get("created")
                if t:
                    ts_min = t if ts_min is None else min(ts_min, t)
                    ts_max = t if ts_max is None else max(ts_max, t)
    except Exception as e:
        ok = False
        sys.stderr.write(f"[warn] {model}/{instance}/{persona}: {e!r}\n")
    wall = (ts_max - ts_min) if (ts_min and ts_max) else 0.0

    # Effective cache for billing: Flash/V4Pro backends do not return it → use the per-turn-cleared structural estimate;
    # the other 5 models' backends really return cached_tokens → use the measurement directly.
    eff_cached = est_cached if model in ESTIMATED_CACHE_MODELS else tcached
    price = PRICE_CNY.get(model)
    cny = None
    if price:
        hit, miss, out = price
        miss_tok = max(0, tin - eff_cached)
        cny = (eff_cached * hit + miss_tok * miss + tout * out) / 1e6
    return {
        "model": model, "instance": instance, "persona": persona,
        "api_calls": ncall, "prompt_tok": tin, "completion_tok": tout,
        "cached_tok": tcached, "est_cached_tok": est_cached,
        "eff_cached_tok": eff_cached,
        "user_turns": user_turns, "retry_errors": retry_errors,
        "wall_sec": round(wall, 1),
        "cny": (round(cny, 4) if cny is not None else None),
        # raw accuracy counts (numerator/denominator stored separately; accumulate per group then compute the ratio, to avoid the "mean of ratios" bias)
        "fin_f2p_p": fin_f2p_p, "fin_f2p_t": fin_f2p_t,
        "fin_p2p_p": fin_p2p_p, "fin_p2p_t": fin_p2p_t,
        "mid_f2p_p": mid_f2p_p, "mid_f2p_t": mid_f2p_t,
        "mid_p2p_p": mid_p2p_p, "mid_p2p_t": mid_p2p_t,
        "ok": ok, "done": done,
    }


# ==================== reusable cost interface (for the "$/cell" column of the six-dimension table) ====================
# Cache-token extraction results (key = traj path, includes mtime/size; as evaluation progresses, changed cells are incrementally re-scanned automatically).
COST_CACHE = ROOT / "data" / "usersim_cost_token_cache.json"


def _scan_traj_cost(path):
    """Stream (ijson C-backend) scan a single traj → (fresh_input, cached_input, output) tokens.

    ⚠️ The cache-fitting convention is exactly the same as scan_cell above (change either place and you MUST sync the other):
      · real user-sim utterance (role==user and not 'Tool call error:') → clear the prefix baseline (cross-turn cache goes to zero);
      · framework 'Tool call error:' retry (also role==user) does not clear (it is a format retry within the same turn);
      · a tool result is role==tool (not user), which inherently does not trigger clearing;
      · the i-th call within a turn hits = min(prompt_i, prompt_{i-1}+completion_{i-1}).
    Convention A: count only the assistant side —— only accumulate tokens from assistant replies bearing extra.response.usage.
    Streaming parse: a traj can reach the GB scale (a 1.85GB one was seen); json.load would blow up memory, so use yajl2_c to read and discard on the fly.
    """
    import ijson.backends.yajl2_c as _ij
    prev_total = 0
    fresh = cached = out = 0
    with open(path, "rb") as fh:
        for msg in _ij.items(fh, "messages.item"):
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "user" and not msg.get("tool_call_id"):
                c = msg.get("content")
                if isinstance(c, list):
                    c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
                if not (isinstance(c, str) and c.lstrip().startswith(RETRY_PREFIX)):
                    prev_total = 0          # real user-sim turn → cache prefix cleared
                continue
            ex = msg.get("extra") or {}
            resp = ex.get("response") if isinstance(ex, dict) else None
            u = resp.get("usage") if isinstance(resp, dict) else None
            if not u:
                continue
            p = u.get("prompt_tokens") or 0
            comp = u.get("completion_tokens") or 0
            hit = 0 if prev_total == 0 else min(prev_total, p)
            cached += hit
            fresh += p - hit
            out += comp
            prev_total = p + comp
    return fresh, cached, out


def avg_cost_usd(models=None, personas=None, workers=16, src=None,
                 use_cache=True, verbose=False, by_instance=False):
    """Average cost per (model, persona) (USD/cell). Convention A (assistant only) + Y (completed cells only).

    Returns {(model, persona): {"avg_usd": float, "n": int, "total_usd": float}}.
      · completed = a final_eval.json exists under the persona directory (fully aligned with the six-dimension table's denominator).
      · cost = (cached*hit + fresh*miss + out*out_price)/1e6, unit price = PRICE_USD[model].
      · token extraction has an mtime/size cache (COST_CACHE); the done state is queried live each time (final_eval may appear later).
    Models not in the price table PRICE_USD are skipped directly (not entering the return).

    When by_instance=True, each entry additionally carries "by_inst": {instance: usd} (per-cell detail,
    for downstream re-aggregation/estimation over an instance subset); off by default, existing callers unaffected.
    """
    src = Path(src) if src else SRC
    want_m = set(models) if models else None
    want_p = set(personas) if personas else None

    jobs = []  # (model, persona, path, done)
    for mdir in sorted(p for p in src.iterdir() if p.is_dir()):
        model = mdir.name
        if (want_m and model not in want_m) or model not in PRICE_USD:
            continue
        for idir in sorted(p for p in mdir.iterdir() if p.is_dir()):
            for pdir in sorted(p for p in idir.iterdir() if p.is_dir()):
                persona = pdir.name
                if want_p and persona not in want_p:
                    continue
                trajs = list((pdir / "assistant").glob("*.assistant.traj.json"))
                if not trajs:
                    continue
                jobs.append((model, persona, str(trajs[0]),
                             (pdir / "final_eval.json").exists()))

    cache = {}
    if use_cache and COST_CACHE.exists():
        try:
            cache = json.load(open(COST_CACHE))
        except Exception:
            cache = {}

    # Only (re)scan trajs whose cache is missing or whose mtime/size changed
    todo = []
    stat_of = {}
    for (_m, _p, path, _d) in jobs:
        if not _d:
            continue  # Convention Y: only scan completed cells that have final_eval, skip live cells still running (avoid reading a never-ending giant live traj and hanging)
        try:
            stt = os.stat(path)
        except OSError:
            continue
        stat_of[path] = (stt.st_mtime, stt.st_size)
        c = cache.get(path)
        if not (c and c.get("mtime") == stt.st_mtime and c.get("size") == stt.st_size):
            todo.append(path)

    if todo:
        if verbose:
            print(f"[cost] need to (re)scan {len(todo)}/{len(jobs)} trajs, {workers} processes streaming-parse…",
                  flush=True)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_scan_traj_cost, p): p for p in todo}
            n = 0
            for fut in as_completed(futs):
                path = futs[fut]
                try:
                    fresh, cached_, out = fut.result()
                except Exception as e:
                    sys.stderr.write(f"[cost warn] {path}: {e!r}\n")
                    continue
                mt, sz = stat_of[path]
                cache[path] = {"mtime": mt, "size": sz,
                               "fresh": fresh, "cached": cached_, "out": out}
                n += 1
                if verbose and (n % 20 == 0 or n == len(todo)):
                    print(f"  [cost {n}/{len(todo)}]", flush=True)
        if use_cache:
            try:
                json.dump(cache, open(COST_CACHE, "w"))
            except Exception as e:
                sys.stderr.write(f"[cost warn] failed to write cache: {e!r}\n")

    agg = {}
    for (model, persona, path, done) in jobs:
        if not done:
            continue
        c = cache.get(path)
        if not c:
            continue
        # Unified cost formula (applies to all models):
        #   * Opus5 / gpt-5.6-sol use the CACHE_RATIO hardcoded ratios (including Opus write=$6.25/M premium);
        #   * the other 12 models use fitted ratios (read=cached/prompt, write=0), equivalent to the old formula.
        usd = _cost_usd(model, c["cached"], c["fresh"], c["out"])
        a = agg.setdefault((model, persona), {"n": 0, "total_usd": 0.0, "by_inst": {}})
        a["n"] += 1
        a["total_usd"] += usd
        if by_instance:
            # instance = .../<model>/<instance>/<persona>/assistant/*.traj.json → parents[2]
            inst = Path(path).parents[2].name
            a["by_inst"][inst] = usd
    out = {}
    for k, a in agg.items():
        d = {"n": a["n"], "total_usd": a["total_usd"],
             "avg_usd": (a["total_usd"] / a["n"] if a["n"] else 0.0)}
        if by_instance:
            d["by_inst"] = a["by_inst"]
        out[k] = d
    return out


def collect_jobs(src: Path):
    jobs = []
    for mdir in sorted(p for p in src.iterdir() if p.is_dir()):
        model = mdir.name
        for idir in sorted(p for p in mdir.iterdir() if p.is_dir()):
            instance = idir.name
            for pdir in sorted(p for p in idir.iterdir() if p.is_dir()):
                persona = pdir.name
                adir = pdir / "assistant"
                if not adir.is_dir():
                    continue
                trajs = list(adir.glob("*.assistant.traj.json"))
                if trajs:
                    jobs.append((model, instance, persona, str(trajs[0])))
    return jobs


def fmt_int(n):
    return f"{n:,}"


def _acc(p, t):
    """Pass-rate percentage string; denominator 0 (no data for this item) → '—'."""
    return f"{p / t * 100:.0f}%" if t else "—"


def _disp(model):
    return DISPLAY_NAME.get(model, model)


def write_md(path, grid_rows, model_agg, tot, all_models, all_personas,
             n_total, n_done):
    """Write the statistics into a markdown report: summary table + detail table + pricing conventions + pitfalls."""
    L = []
    A = L.append
    A("# User-Sim Closed-Loop Evaluation · Token / Time / Cost Statistics Report\n")
    A(f"> Data source: `data/usersim_final/<model>/<instance>/<persona>/assistant/*.assistant.traj.json`  \n"
      f"> Snapshot convention: of all {n_total} cells, **only the {n_done} completed ones are counted**"
      f"(only cells with a `final_eval.json` count as truly finished; evaluation in progress, unfinished ones not counted).  \n"
      f"> Pricing: official three-tier (cache hit / cache miss input / output), exchange rate USD→CNY = {USD_CNY}.\n")

    # ── I. summary by model ──
    A("## I. Summary by model (completed cells only)\n")
    A("> Accuracy is a micro average (accumulate pass/total within a group, then compute the ratio). mid = intermediate accuracy (`bigtask_eval.json`, "
      "each big_task evaluated on the spot at completion and accumulated); final = final accuracy (`final_eval.json`, uniform re-test after everything is done). "
      "f2p = pass rate of the tests for the bug being fixed, p2p = regression rate of tests that should have kept passing. **high mid, low final = regression** (later changes broke what was previously done right).\n")
    A("| Model | Completed cells | Avg turns | Total retries | Total in_tok | Total out_tok | Cache hit% | Total wall(h) | Avg wall(min/cell) | Avg ¥/cell | Total ¥ | mid f2p | mid p2p | final f2p | final p2p |")
    A("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for model in all_models:
        m = model_agg.get(model)
        if not m:
            A(f"| {_disp(model)} | — | — | — | — | — | — | — | — | — | — | — | — | — | — |")
            continue
        approx = "~" if model in PRICE_APPROX else ""
        est = "*" if model in ESTIMATED_CACHE_MODELS else ""
        cache_pct = m["cache"] / max(1, m["in"]) * 100
        A(f"| {_disp(model)} | {m['n']} | {m['turns']/m['n']:.1f} | {fmt_int(m['retry'])} "
          f"| {fmt_int(m['in'])} | {fmt_int(m['out'])} "
          f"| {cache_pct:.0f}%{est} | {m['wall']/3600:.1f} | {m['wall']/m['n']/60:.1f} "
          f"| {approx}¥{m['cny']/m['n']:.3f} | {approx}¥{m['cny']:.2f} "
          f"| {_acc(m['mid_f2p_p'],m['mid_f2p_t'])} | {_acc(m['mid_p2p_p'],m['mid_p2p_t'])} "
          f"| {_acc(m['fin_f2p_p'],m['fin_f2p_t'])} | {_acc(m['fin_p2p_p'],m['fin_p2p_t'])} |")
    if tot:
        A(f"| **Total** | **{tot['n']}** | **{tot['turns']/tot['n']:.1f}** | **{fmt_int(tot['retry'])}** "
          f"| **{fmt_int(tot['in'])}** | **{fmt_int(tot['out'])}** "
          f"| — | **{tot['wall']/3600:.1f}** | **{tot['wall']/tot['n']/60:.1f}** "
          f"| — | **¥{tot['cny']:.2f}** "
          f"| **{_acc(tot['mid_f2p_p'],tot['mid_f2p_t'])}** | **{_acc(tot['mid_p2p_p'],tot['mid_p2p_t'])}** "
          f"| **{_acc(tot['fin_f2p_p'],tot['fin_f2p_t'])}** | **{_acc(tot['fin_p2p_p'],tot['fin_p2p_t'])}** |")
    A("")

    # ── II. model × persona detail ──
    A("## II. Model × persona detail (single-cell average)\n")
    A("> Groups with none completed show `—`. A `*` on cache% means structural estimate (see convention III). "
      "Turns = count of real user-sim utterance turns (format retries already removed); retries = count of framework-injected `Tool call error:` wasted turns.\n")
    A("| Model | Persona | N | Avg turns | Avg retries | Calls/avg | in_tok/avg | out_tok/avg | Cache% | Wall(min)/avg | ¥/avg | mid f2p | mid p2p | final f2p | final p2p |")
    A("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for model, persona, d in grid_rows:
        pl = PERSONA_LABEL.get(persona, persona)
        if d is None:
            A(f"| {_disp(model)} | {pl} | — | — | — | — | — | — | — | — | — | — | — | — | — |")
            continue
        est = "*" if d["est"] else ""
        cny_s = f"{d['approx']}¥{d['cny']:.3f}" if d["cny"] is not None else "N/A"
        A(f"| {_disp(model)} | {pl} | {d['n']} | {d['turns']:.1f} | {d['retry']:.1f} "
          f"| {d['call']:.0f} | {d['in']:,.0f} | {d['out']:,.0f} | {d['cache']:.0f}%{est} "
          f"| {d['wall']:.1f} | {cny_s} "
          f"| {_acc(d['mid_f2p_p'],d['mid_f2p_t'])} | {_acc(d['mid_p2p_p'],d['mid_p2p_t'])} "
          f"| {_acc(d['fin_f2p_p'],d['fin_f2p_t'])} | {_acc(d['fin_p2p_p'],d['fin_p2p_t'])} |")
    A("")

    # ── III. pricing conventions ──
    A("## III. Pricing conventions and unit-price table\n")
    A("Three-tier billing formula (real-bill convention):\n")
    A("```\ncost = cached_tok × hit + (prompt_tok − cached_tok) × miss + completion_tok × out\n```\n")
    A("Unit price (yuan / million tokens, = official-site USD price × 6.8; USD source of truth see COST_ESTIMATION_NOTES.md):\n")
    A("| Model | Cache hit | Cache miss input | Output | Note |")
    A("|---|--:|--:|--:|---|")
    note = {
        "DeepSeek-V4-Flash": "A official ¥0.02/1/2",
        "DeepSeek-V4-Pro":   "A official ¥0.025/3/6",
        "Kimi-K2.7":         "A official $0.19/0.95/4.0×6.8",
        "MiMo-V2.5-Pro":     "A official ¥0.025/3/6",
        "Hy3":               "A official ¥0.25/1/4",
        "Qwen3.5-397B-A17B": "B input official ¥0.8; output Novita",
        "Qwen3.5-122B-A10B": "B Novita third-party",
        "Qwen3.6-35B-A3B":   "B Alibaba Flash tier; hit approx 20%-off estimate",
        "GLM-5.1":           "~ C official page only has the vision version, uncertain",
        "GLM-5.2":           "~ C official page only has the vision version, uncertain",
        "Hy3-Preview":       "~ C preview-version price inferred",
        "gpt-5.6-sol":       "~ C GPT-5 two price sets, tier uncertain",
    }
    for model in all_models:
        hit, miss, out = PRICE_CNY[model]
        A(f"| {_disp(model)} | ¥{hit} | ¥{miss} | ¥{out} | {note.get(model,'')} |")
    A("")

    # ── IV. cache estimation convention (key point) ──
    A("## IV. Cache back-estimation convention for Flash / V4Pro (key point)\n")
    A("**Problem**: most backends return the real hit count in "
      "`prompt_tokens_details.cached_tokens` (measured hit rate 76–90%); but the **local vLLM** and "
      "**local SGLang** endpoints do not return this field → computing directly would show 0% cache and overestimate cost in full. "
      "**So only these two kinds of local endpoints use structural estimation, everything else uses the backend measurement.**\n")
    A("**Back-estimation convention (clear every turn, reuse only within a turn)**: a turn = user-sim sends one request + agent replies once "
      "(which may contain multiple tool calls in between). Under the 7×8 high-concurrency pressure, in the gap where the agent waits for the user-sim to return, "
      "this cell's KV prefix gets evicted from VRAM by other requests (LRU eviction of vLLM APC / SGLang RadixAttention), "
      "so **cross-turn cache is basically zeroed**; only **consecutive tool calls within the same turn** stably share the prefix. So reset the baseline per turn:\n")
    A("```\nat every real user-sim utterance → clear the prefix baseline\n"
      "the i-th call within a turn hits = min(prompt_i, prompt_{i-1} + completion_{i-1})\n"
      "est_cached = Σ of all hittable prefixes within turns\n```\n")
    A("> Note: turn boundaries only recognize **real user-sim utterances**; the framework-injected `Tool call error:` format-retry messages "
      "do not count as a new turn and do not clear the prefix (they are retries within the same turn).\n")
    A("**Why not use the theoretical upper bound of 'append-only, no eviction throughout'**: an early version assumed the whole cell's prompt "
      "grows append-only and monotonically and the cache never expires, deriving ~99% hit. This was falsified on 5 models that return cached—"
      "real backends only have 76–90%, 10–32% below the theoretical upper bound (exactly the KV eviction loss during cross-turn waits). "
      "The 'clear every turn' convention models this eviction explicitly and is closer to reality than the theoretical upper bound.\n")
    A("| Model returning cached | Backend-measured hit% | No-eviction upper bound% throughout | Measured/upper bound |")
    A("|---|--:|--:|--:|")
    for mdl, act, est, ratio in [
        ("GLM-5.2-FP8", 90.0, 99.3, 0.906),
        ("Hy3-FP8", 86.6, 99.7, 0.869),
        ("Kimi-K2.7-FP8", 80.0, 99.5, 0.804),
        ("Hy3-Preview-FP8", 75.9, 99.7, 0.761),
        ("Qwen3.5-397B-A17B-FP8", 77.6, 99.5, 0.780),
    ]:
        A(f"| {mdl} | {act:.1f}% | {est:.1f}% | {ratio:.3f} |")
    A("")
    A("**Conclusion**: Flash/V4Pro's cache%/¥ use the 'clear every turn' structural estimate (cache% marked `*`). This is "
      "more conservative than the 'no-eviction upper bound throughout' and more optimistic than 'zero cache, full charge'; it is the convention closest to reality given no measured data; "
      "the real bill may still be slightly higher because the within-turn prefix is also partly evicted. When comparing across models, mind the two `*` columns.\n")

    # ── V. accuracy convention ──
    A("## V. Accuracy convention (intermediate vs final)\n")
    A("This evaluation is **progressive multi-subtask**: one instance is split into a SEED + several BIG_tasks, and the agent completes them step by step over turns. "
      "The two evaluation files correspond to two points in time:\n")
    A("- **Intermediate accuracy** (`bigtask_eval.json`): each big_task is **evaluated once on the spot** at the turn it completes (`evaluated_at_turn`), "
      "reflecting \"whether it was right right when that block was just done\". Aggregation convention: accumulate all big_tasks' `n_f2p_pass`/`n_f2p_total` "
      "separately and then compute the ratio (p2p likewise).\n")
    A("- **Final accuracy** (`final_eval.json`): a **uniform re-test** after all subtasks are done, reflecting \"whether it was right at final delivery\", "
      "taking the top-level `n_f2p_pass/total`, `n_p2p_pass/total`.\n")
    A("**f2p vs p2p**: f2p (fail-to-pass) = the tests for the bug being fixed; the higher the pass rate, the more it shows the feature was truly built; "
      "p2p (pass-to-pass) = tests that should have kept passing, a drop in pass rate is a **regression**—breaking what was previously right.\n")
    A("**micro average**: all ratios use \"accumulate pass within a group ÷ accumulate total\", not \"compute the ratio per cell first then take the mean\"—"
      "the latter would inflate the weight of small-denominator cells. Cells with denominator 0 (no data for this item) are not counted.\n")
    A("**How to read \"right in the middle, dropping at the end\"**: if a model's intermediate f2p is high but its final f2p is clearly lower, it means that while doing the later big_tasks it "
      "broke what it had done right earlier (classic: modifying shared code introduces a regression). mid≈final means the process is stable with no obvious regression.\n")

    # ── VI. pitfalls ──
    A("## VI. Pitfalls and caveats\n")
    A("1. **`info.model_stats` is untrustworthy** (hardcodes api_calls=2, instance_cost=0); we MUST walk messages and accumulate `extra.response.usage` one by one.\n")
    A("2. **The `cost` field is always 0.0**: all models are self-hosted (vLLM/SGLang) and do not return API billing.\n")
    A("3. **The timestamp is in `extra.timestamp` (epoch seconds)**, not `extra.response.timestamp` (always None); fall back to `response.created`.\n")
    A("4. **Wall clock = the time difference between the cell's first and last API call**, including tool execution / queuing / multithreaded overlap, **not pure inference time**.\n")
    A("5. **Half-done traj**: an in-progress cell's traj is an intermediate snapshot refreshed each turn and may be truncated (JSONDecodeError); it has been skipped; the numbers in this report change as evaluation progresses.\n")
    A("6. **The usersim (user simulator) side records no usage**; cost comes only from the assistant side.\n")
    A("7. **`Tool call error:` format retries pollute the turn count**: some cells (most notably the openlibrary/teleport/astropy ones under Hy3-FP8) "
      "repeatedly fail to emit tool_call in the right format, and the framework injects an error message with `role==\"user\"` each time to make it retry, which can loop from hundreds to nearly a thousand times per cell. "
      "These are **not real user-sim turns** and have been removed from the turn count (see the 'Total retries' column); but they really triggered API calls and really burned tokens, "
      "so tokens/cost are still counted in full. An abnormally high retry count itself exposes the model's instruction-following deficiency.\n")

    Path(path).write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--csv", default=str(ROOT / "data" / "usersim_cost_stats.csv"))
    ap.add_argument("--md", default=str(ROOT / "data" / "usersim_cost_report.md"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    jobs = collect_jobs(Path(args.src))
    print(f"found {len(jobs)} cells, {args.workers} processes parsing in parallel…", flush=True)

    rows = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(scan_cell, j): j for j in jobs}
        for fut in as_completed(futs):
            rows.append(fut.result())
            done += 1
            if done % 10 == 0 or done == len(jobs):
                print(f"  [{done}/{len(jobs)}]", flush=True)

    rows.sort(key=lambda r: (r["model"], r["instance"], r["persona"]))

    # detail CSV (all cells, including unfinished; the done column distinguishes them)
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "instance", "persona", "done", "api_calls",
                    "user_turns", "retry_errors",
                    "prompt_tok", "completion_tok", "cached_tok",
                    "wall_sec", "cny_est",
                    "mid_f2p_pass", "mid_f2p_total", "mid_p2p_pass", "mid_p2p_total",
                    "fin_f2p_pass", "fin_f2p_total", "fin_p2p_pass", "fin_p2p_total"])
        for r in rows:
            w.writerow([r["model"], r["instance"], r["persona"],
                        int(r["done"]), r["api_calls"],
                        r["user_turns"], r["retry_errors"],
                        r["prompt_tok"], r["completion_tok"], r["cached_tok"],
                        r["wall_sec"], r["cny"] if r["cny"] is not None else "",
                        r["mid_f2p_p"], r["mid_f2p_t"], r["mid_p2p_p"], r["mid_p2p_t"],
                        r["fin_f2p_p"], r["fin_f2p_t"], r["fin_p2p_p"], r["fin_p2p_t"]])
    n_done = sum(1 for r in rows if r["done"])
    print(f"\ndetail written: {args.csv}  ({len(rows)} rows, of which completed = {n_done})\n")

    # ===== count only the finished cells (those with final_eval.json) =====
    done_rows = [r for r in rows if r["done"]]

    ALL_MODELS = sorted(PRICE_CNY)
    ALL_PERSONAS = ["non_programming_user", "product_stakeholder",
                    "repo_newcomer_engineer", "repository_steward"]

    grp: dict = {}
    for r in done_rows:
        grp.setdefault((r["model"], r["persona"]), []).append(r)

    print("=" * 166)
    print("[Model × Persona] completed cells only (with final_eval.json); none completed = -")
    print("-" * 166)
    print(f"{'Model':<26}{'Persona':<24}{'N':>3} {'AvgTurns':>6} {'AvgRetry':>6} {'Calls/avg':>7} "
          f"{'in_tok/avg':>13} {'out_tok/avg':>12} {'Cache%':>6} {'wall min/avg':>10} {'¥/avg':>10} "
          f"{'mid f2p':>7}{'mid p2p':>7}{'fin f2p':>7}{'fin p2p':>7}")
    print("-" * 166)
    model_agg: dict = {}
    grid_rows: list = []   # for md: one row per (model,persona)
    for model in ALL_MODELS:
        for persona in ALL_PERSONAS:
            cells = grp.get((model, persona))
            if not cells:
                print(f"{model:<26}{persona:<24}{'-':>3} {'-':>6} {'-':>6} {'-':>7} "
                      f"{'-':>13} {'-':>12} {'-':>6} {'-':>10} {'-':>10} "
                      f"{'-':>7}{'-':>7}{'-':>7}{'-':>7}")
                grid_rows.append((model, persona, None))
                continue
            n = len(cells)
            a_turns = statistics.mean(c["user_turns"] for c in cells)
            a_retry = statistics.mean(c["retry_errors"] for c in cells)
            a_call = statistics.mean(c["api_calls"] for c in cells)
            a_in = statistics.mean(c["prompt_tok"] for c in cells)
            a_out = statistics.mean(c["completion_tok"] for c in cells)
            # cache% uses the effective cache (backend measurement first, otherwise structural estimate); estimators marked *.
            a_cache = (sum(c["eff_cached_tok"] for c in cells) /
                       max(1, sum(c["prompt_tok"] for c in cells)) * 100)
            a_wall = statistics.mean(c["wall_sec"] for c in cells) / 60
            cnys = [c["cny"] for c in cells if c["cny"] is not None]
            a_cny = statistics.mean(cnys) if cnys else None
            # accuracy: accumulate pass/total within a group separately then compute the ratio (micro average, not "mean of ratios")
            g = {k: sum(c[k] for c in cells) for k in
                 ("fin_f2p_p", "fin_f2p_t", "fin_p2p_p", "fin_p2p_t",
                  "mid_f2p_p", "mid_f2p_t", "mid_p2p_p", "mid_p2p_t")}
            est = model in ESTIMATED_CACHE_MODELS
            approx = "~" if model in PRICE_APPROX else ""
            cache_s = f"{a_cache:.0f}%" + ("*" if est else "")
            cny_s = f"{approx}¥{a_cny:.3f}" if a_cny is not None else "N/A"
            print(f"{model:<26}{persona:<24}{n:>3} {a_turns:>6.1f} {a_retry:>6.1f} "
                  f"{a_call:>7.0f} {a_in:>13,.0f} {a_out:>12,.0f} {cache_s:>6} "
                  f"{a_wall:>10.1f} {cny_s:>10} "
                  f"{_acc(g['mid_f2p_p'],g['mid_f2p_t']):>7}{_acc(g['mid_p2p_p'],g['mid_p2p_t']):>7}"
                  f"{_acc(g['fin_f2p_p'],g['fin_f2p_t']):>7}{_acc(g['fin_p2p_p'],g['fin_p2p_t']):>7}")
            grid_rows.append((model, persona, {
                "n": n, "turns": a_turns, "retry": a_retry,
                "call": a_call, "in": a_in, "out": a_out,
                "cache": a_cache, "wall": a_wall, "cny": a_cny,
                "est": est, "approx": approx, **g}))
            m = model_agg.setdefault(model, {"n": 0, "in": 0, "out": 0,
                                             "cache": 0, "wall": 0, "cny": 0,
                                             "turns": 0, "retry": 0,
                                             "fin_f2p_p": 0, "fin_f2p_t": 0,
                                             "fin_p2p_p": 0, "fin_p2p_t": 0,
                                             "mid_f2p_p": 0, "mid_f2p_t": 0,
                                             "mid_p2p_p": 0, "mid_p2p_t": 0})
            m["n"] += n
            m["in"] += sum(c["prompt_tok"] for c in cells)
            m["out"] += sum(c["completion_tok"] for c in cells)
            m["cache"] += sum(c["eff_cached_tok"] for c in cells)
            m["wall"] += sum(c["wall_sec"] for c in cells)
            m["cny"] += sum(cnys)
            m["turns"] += sum(c["user_turns"] for c in cells)
            m["retry"] += sum(c["retry_errors"] for c in cells)
            for k in ("fin_f2p_p", "fin_f2p_t", "fin_p2p_p", "fin_p2p_t",
                      "mid_f2p_p", "mid_f2p_t", "mid_p2p_p", "mid_p2p_t"):
                m[k] += g[k]

    print("=" * 166)
    print("\n[Summary by model] completed cells only (cumulative + single-cell average)")
    print("-" * 156)
    print(f"{'Model':<26}{'cells':>6}{'AvgTurns':>7}{'TotRetry':>8}{'Tot in_tok':>16}{'Tot out_tok':>13}"
          f"{'Cache%':>7}{'Tot wall h':>9}{'Avg wall min':>10}{'Avg ¥/cell':>11}{'Tot ¥':>11}"
          f"{'mid f2p':>7}{'mid p2p':>7}{'fin f2p':>7}{'fin p2p':>7}")
    print("-" * 156)
    for model in ALL_MODELS:
        m = model_agg.get(model)
        approx = "~" if model in PRICE_APPROX else ""
        if not m:
            print(f"{model:<26}{'-':>6}{'-':>7}{'-':>8}{'-':>16}{'-':>13}{'-':>7}"
                  f"{'-':>9}{'-':>10}{'-':>11}{'-':>11}"
                  f"{'-':>7}{'-':>7}{'-':>7}{'-':>7}")
            continue
        est = model in ESTIMATED_CACHE_MODELS
        cache_pct = m["cache"] / max(1, m["in"]) * 100
        cache_s = f"{cache_pct:.0f}%" + ("*" if est else "")
        print(f"{model:<26}{m['n']:>6}{m['turns']/m['n']:>7.1f}{m['retry']:>8,}"
              f"{m['in']:>16,}{m['out']:>13,}"
              f"{cache_s:>7}{m['wall']/3600:>9.1f}{m['wall']/m['n']/60:>10.1f}"
              f"{approx+'¥'+format(m['cny']/m['n'],'.3f'):>11}"
              f"{approx+'¥'+format(m['cny'],'.2f'):>11}"
              f"{_acc(m['mid_f2p_p'],m['mid_f2p_t']):>7}{_acc(m['mid_p2p_p'],m['mid_p2p_t']):>7}"
              f"{_acc(m['fin_f2p_p'],m['fin_f2p_t']):>7}{_acc(m['fin_p2p_p'],m['fin_p2p_t']):>7}")
    print("-" * 156)
    tot = None
    if model_agg:
        tc = sum(m["n"] for m in model_agg.values())
        ti = sum(m["in"] for m in model_agg.values())
        to = sum(m["out"] for m in model_agg.values())
        tw = sum(m["wall"] for m in model_agg.values())
        ty = sum(m["cny"] for m in model_agg.values())
        tturns = sum(m["turns"] for m in model_agg.values())
        tretry = sum(m["retry"] for m in model_agg.values())
        tacc = {k: sum(m[k] for m in model_agg.values()) for k in
                ("fin_f2p_p", "fin_f2p_t", "fin_p2p_p", "fin_p2p_t",
                 "mid_f2p_p", "mid_f2p_t", "mid_p2p_p", "mid_p2p_t")}
        tot = {"n": tc, "in": ti, "out": to, "wall": tw, "cny": ty,
               "turns": tturns, "retry": tretry, **tacc}
        print(f"{'Total':<26}{tc:>6}{tturns/tc:>7.1f}{tretry:>8,}{ti:>16,}{to:>13,}"
              f"{'':>7}{tw/3600:>9.1f}{tw/tc/60:>10.1f}"
              f"{'':>11}{'¥'+format(ty,'.2f'):>11}"
              f"{_acc(tacc['mid_f2p_p'],tacc['mid_f2p_t']):>7}{_acc(tacc['mid_p2p_p'],tacc['mid_p2p_t']):>7}"
              f"{_acc(tacc['fin_f2p_p'],tacc['fin_f2p_t']):>7}{_acc(tacc['fin_p2p_p'],tacc['fin_p2p_t']):>7}")
    print("\nNotes: ① Only cells with final_eval.json are counted (evaluation in progress, unfinished ones not counted)."
          "\n    ② Turns = count of real user-sim utterance turns (one request sent + agent replies once = one turn);"
          "\n       framework-injected 'Tool call error:' format retries have been removed (when the model does not emit tool_call in the right format, the framework returns an error and retries)."
          "\n       The 'Total retries' column is the count of these wasted turns—they really burned tokens, so cost/tokens are still counted in full."
          "\n    ③ ¥ = official three-tier billing (cache hit / cache miss input / output), exchange rate 6.8. A ~ means the unit price contains an approximation"
          "(Preview-HX≈Hy3 official price, GLM per the public tier)."
          "\n    ④ A * on cache% means structural estimate (Flash/V4Pro backends do not return cached_tokens): convention = clear every turn,"
          "reuse prefix only across tool calls within a turn; the other 5 models use the backend-measured cached_tokens."
          "\n    ⑤ Wall clock = the time difference between the cell's first and last API call, including tool execution / queuing / multithreaded overlap, not pure inference time."
          "\n    ⑥ Accuracy (micro, accumulate pass/total within a group then compute the ratio): mid = bigtask_eval, each big_task evaluated on the spot at completion and accumulated;"
          "\n       final = final_eval, uniform re-test after everything is done. f2p = pass rate of the tests for the bug being fixed, p2p = regression rate of tests that should have kept passing."
          "\n       'right in the middle, dropping at the end' is a regression—later changes broke what was previously done right.")

    # ===== generate the markdown report =====
    write_md(args.md, grid_rows, model_agg, tot, ALL_MODELS, ALL_PERSONAS,
             n_total=len(rows), n_done=n_done)
    print(f"\nmarkdown report written: {args.md}")


if __name__ == "__main__":
    main()
