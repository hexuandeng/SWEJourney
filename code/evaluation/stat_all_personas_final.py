#!/usr/bin/env python3
"""Multi-assistant-model comparison version: for each model × 4 personas, compute the "four dimensions" of accuracy.

Reuses the algorithm of stat_all_personas.py, only changing the output root directory to carry a model dimension:
    OUTROOT/<model>/<instance>/<persona>/{final_eval,bigtask_eval}.json
Within a [single model sub-root], the hierarchy is exactly the same as v3 (cell_id = second-/third-to-last segment), so
compute_persona is parameterized with an added root (=OUTROOT/<model>); the outer layer, after discovering the model
sub-directories, prints one table per model, and finally appends a "model × dimension" cross-model comparison table (taking the 4-persona total).

Four dimensions: D1 process·f2p / D2 process·p2p / D3 final·all-big·f2p / D4 final·all-big·p2p.
"""
from __future__ import annotations
import argparse, json, glob, os, sys
from collections import defaultdict

# Relative-path wiring (see config.py): benchmark is the sole evaluation input (final_f2p/p2p + events own_*),
# the bad-node exclusion set / the p2p→f2p reclassification set are all reverse-derived from benchmark; the run outputs are still under ../../data/usersim_final.
import config as _cfg
OUTROOT = str(_cfg.USERSIM_OUT_DIR)      # user-sim run outputs: <model>/<instance>/<persona>/
BENCHMARK_DIR = str(_cfg.BENCHMARK_DIR)  # the sole evaluation input: reverse-derive the exclude/reclass sets + count the number of cells
TRACES = str(_cfg.TRACES_DIR)            # = BENCHMARK_DIR (count_traces counts the number of files)
ALLGOLD_DIR = str(_cfg.ALLGOLD_DIR)      # = BENCHMARK_DIR (alias kept for backward compatibility)
SEEDPROBE_DIR = str(_cfg.SEEDPROBE_DIR)  # = BENCHMARK_DIR (alias kept for backward compatibility)
ORDER = ["non_programming_user", "product_stakeholder",
         "repo_newcomer_engineer", "repository_steward"]

_ds_cache: dict[str, dict] = {}


_bench_sets_cache: dict | None = None


def _benchmark_sets() -> tuple[set, set]:
    """Scan benchmark/*.json once, reverse-deriving the two sets that are [element-for-element equivalent] to the old probe version (cached).

    benchmark is the sole evaluation input: each file's top-level final_f2p/final_p2p is a list of [{"sid","node"}] objects,
    and events retain the own_f2p/own_p2p of every ok subtask. From this we reverse-derive:
      · allgold_exclude = the cell's (own_f2p ∪ own_p2p) − (final_f2p ∪ final_p2p)
                          -- bad nodes that still don't pass even with full gold and have been removed from final (not in the list = excluded).
      · seed_reclass    = nodes in own_p2p that appear in final_f2p -- ones whose base+SEED baseline did not pass and have been shifted into f2p.
    key = (cell_id, sid, node), exactly isomorphic to the scoring key. cell_id = the file stem with the `instance_` prefix stripped.
    """
    global _bench_sets_cache
    if _bench_sets_cache is not None:
        return _bench_sets_cache
    exclude: set = set()
    reclass: set = set()
    for path in glob.glob(f"{BENCHMARK_DIR}/*.json"):
        stem = os.path.basename(path)[:-len(".json")]
        if stem == "_SUMMARY":
            continue
        cell_id = stem[len("instance_"):] if stem.startswith("instance_") else stem
        try:
            d = json.load(open(path))
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] benchmark read failed {stem}: {e}")
            continue
        own_f2p: set = set()
        own_p2p: set = set()
        for e in d.get("events", []):
            if e.get("kind") != "ATTEMPT":
                continue
            sid = f"{e.get('big_task_id')}/{e.get('subtask_id')}"
            for node in (e.get("own_f2p") or []):
                own_f2p.add((cell_id, sid, node))
            for node in (e.get("own_p2p") or []):
                own_p2p.add((cell_id, sid, node))
        final_f2p = {(cell_id, it["sid"], it["node"]) for it in d.get("final_f2p", [])}
        final_p2p = {(cell_id, it["sid"], it["node"]) for it in d.get("final_p2p", [])}
        own_all = own_f2p | own_p2p
        final_all = final_f2p | final_p2p
        exclude |= (own_all - final_all)                       # not in final = excluded bad node
        reclass |= {t for t in own_p2p if t in final_f2p}      # own_p2p yet in f2p = reclassification
    _bench_sets_cache = (exclude, reclass)
    return _bench_sets_cache


def load_allgold_exclude() -> set:
    """Reverse-derive from benchmark the exclusion set of bad nodes that "still don't pass even with full gold" (see _benchmark_sets for details).

    key = (cell_id, sid, node), exactly isomorphic to the scoring key. These nodes have been removed from benchmark's final_f2p/final_p2p
    (not in the list = excluded), reverse-derived from (full own set − full final set). Used for the exclusion in dimensions 3/4 (and dimensions 1/2 share the same-source denominator).
    """
    return _benchmark_sets()[0]


def load_seedprobe_reclass() -> set:
    """Reverse-derive from benchmark the reclassification set of "non-SEED p2p that did not pass on the base+SEED baseline" (see _benchmark_sets for details).

    key = (cell_id, sid, node). These originally belonged to own_p2p, yet appear in benchmark's final_f2p -- i.e. already shifted into f2p.
    Orthogonal to allgold_exclude (bad nodes are absent from final, so neither set contains them, and they naturally never enter this set).
    """
    return _benchmark_sets()[1]


def dataset_big_totals(cell_id: str, seed_reclass: set | None = None) -> dict:
    """The dataset-side f2p/p2p denominator per big (used by D1/D2).

    When seed_reclass is not None: nodes in own_p2p that hit the reclassification set (non-SEED p2p whose baseline did not pass)
    are counted into that big's f2p denominator instead, while the remaining own_p2p are still counted as p2p; all own_f2p count as f2p.
    This way the D1/D2 denominators also move together with the "p2p→f2p shift", consistent with the numerator convention.
    The cache key carries whether reclass is enabled, to avoid the two conventions cross-contaminating.
    """
    ck = (cell_id, seed_reclass is not None)
    if ck in _ds_cache:
        return _ds_cache[ck]
    path = f"{TRACES}/{cell_id}.json"
    try:
        d = json.load(open(path))
    except FileNotFoundError:
        print(f"  [warn] missing trace: {cell_id}")
        _ds_cache[ck] = {}
        return {}
    seen = {}
    for e in d.get("events", []):
        if e.get("kind") != "ATTEMPT" or not e.get("enabled", False):
            continue
        key = (e["big_task_id"], e["subtask_id"])
        if key in seen:
            continue
        seen[key] = (e.get("own_f2p") or [], e.get("own_p2p") or [])
    totals = defaultdict(lambda: {"f2p": 0, "p2p": 0})
    for (big, sub), (own_f2p, own_p2p) in seen.items():
        sid = f"{big}/{sub}"
        totals[big]["f2p"] += len(own_f2p)
        for node in own_p2p:
            if seed_reclass is not None and (cell_id, sid, node) in seed_reclass:
                totals[big]["f2p"] += 1
            else:
                totals[big]["p2p"] += 1
    out = dict(totals)
    _ds_cache[ck] = out
    return out


def big_sort_key(big_id: str) -> int:
    if big_id == "SEED":
        return -1
    if big_id.startswith("BIG_"):
        try:
            return int(big_id.split("_", 1)[1])
        except ValueError:
            return 0
    return 0


def prefix_of(subtask_id: str) -> str:
    return subtask_id.split("/", 1)[0]


def compute_persona(root: str, persona: str, allowed: set | None = None,
                    allgold_exclude: set | None = None,
                    seed_reclass: set | None = None) -> dict:
    """For a single persona under a single model sub-root `root`, compute the four dimensions (D1/D2/D3/D4).

    root = OUTROOT/<model>; its inner structure <instance>/<persona>/ matches v3, so
    cell_id is still taken as the second-to-last segment of the final path, and the third-to-last of the bigtask path.

    When allowed is not None, only count cells with cell_id ∈ allowed (used by --common, the common completed set).

    When allgold_exclude is not None, the (cell_id, sid, node) bad nodes it contains are [excluded from all of dimensions 1/2/3/4]
    -- these nodes still don't pass even with full gold, so they are data/environment problems and should not be charged to the model.

    When seed_reclass is not None, the (cell_id, sid, node) it contains are "non-SEED p2p that
    did not pass on the base+SEED baseline"; in [all four dimensions] that node is shifted from p2p and merged into f2p (numerator and denominator move together).
    Orthogonal to allgold_exclude: during scoring, allgold bad nodes are skipped first, then seed reclassification is applied to the rest.

    ─── New convention for D1/D2 (process f2p/p2p) (starting 2026-08-10) ───────────────────────────────
    Denominator: exactly the same as D3/D4 = all nodes of [all bigs (including SEED)] in final_eval,
          after allgold exclusion + seed_reclass shift.
    Numerator: for the same (cell_id, sid, node)--
          · if that node has a record in this cell's bigtask_eval.json → use the bigtask status;
          · otherwise (bt did not measure it / or the whole cell has no bt file, e.g. SEED is naturally not in bt)
            → fall back to the status of that node in final_eval.
    This way the process denominator is comparable and from the same source as the result denominator, and nodes that are "not judged in the process" don't lose their denominator for nothing.
    """
    last_pass = {"f2p": 0, "p2p": 0}     # D3/D4 numerator
    last_total = {"f2p": 0, "p2p": 0}    # D3/D4 denominator
    proc_pass = {"f2p": 0, "p2p": 0}     # D1/D2 numerator (new convention)
    proc_total = {"f2p": 0, "p2p": 0}    # D1/D2 denominator (new convention = D3/D4 denominator)

    def eff(kind: str, cell_id: str, sid: str, node: str) -> str:
        """Count "non-SEED p2p whose baseline did not pass" nodes as f2p. The rest are left as-is."""
        if (kind == "p2p" and seed_reclass is not None
                and (cell_id, sid, node) in seed_reclass):
            return "f2p"
        return kind

    fe_cells = sorted(glob.glob(f"{root}/*/{persona}/final_eval.json"))
    bt_cells = sorted(glob.glob(f"{root}/*/{persona}/bigtask_eval.json"))

    for fe_path in fe_cells:
        cell_dir = fe_path.rsplit("/", 1)[0]
        cell_id = cell_dir.split("/")[-2]
        if allowed is not None and cell_id not in allowed:
            continue
        final = json.load(open(fe_path))
        subtasks = final.get("subtasks", [])

        # First collect the scoring status of all (sid, node) from this cell's bigtask_eval into one dict,
        # for the "use bt if present, else use final" fallback of the D1/D2 numerator below.
        bt_status: dict[tuple[str, str], str] = {}
        bt_path = f"{cell_dir}/bigtask_eval.json"
        try:
            bt = json.load(open(bt_path))
            for big in bt.get("big_tasks", []):
                for st in big.get("subtasks", []):
                    sid = st["subtask_id"]
                    for kind in ("f2p", "p2p"):
                        for node, status in (st.get(kind) or {}).items():
                            bt_status[(sid, node)] = status
        except FileNotFoundError:
            pass

        # Dimensions 3/4 (final·all-big·including SEED) & dimensions 1/2 (process, same denominator + bt fallback):
        # done in the same scan of final_eval, guaranteeing the D1/D2 denominator is exactly the same source as the D3/D4 denominator.
        for st in subtasks:
            sid = st["subtask_id"]
            for kind in ("f2p", "p2p"):
                for node, status in (st.get(kind) or {}).items():
                    key = (cell_id, sid, node)
                    # Dimensions 1/2/3/4 all apply allgold exclusion.
                    if allgold_exclude is not None and key in allgold_exclude:
                        continue
                    ek = eff(kind, cell_id, sid, node)
                    # D3/D4: use the final status directly.
                    last_total[ek] += 1
                    if status == "PASSED":
                        last_pass[ek] += 1
                    # D1/D2: denominator same as D3/D4; numerator = use bt if present, else use final.
                    proc_total[ek] += 1
                    proc_status = bt_status.get((sid, node), status)
                    if proc_status == "PASSED":
                        proc_pass[ek] += 1

    def _count(cells, idx):
        if allowed is None:
            return len(cells)
        return sum(1 for c in cells if c.split("/")[idx] in allowed)

    return {
        "n_final": _count(fe_cells, -3),
        "n_bigtask": _count(bt_cells, -3),
        1: (proc_pass["f2p"], proc_total["f2p"]),
        2: (proc_pass["p2p"], proc_total["p2p"]),
        3: (last_pass["f2p"], last_total["f2p"]),
        4: (last_pass["p2p"], last_total["p2p"]),
    }


def fmt(pt: tuple) -> str:
    p, t = pt
    if not t:
        return f"{p}/{t} N/A"
    return f"{p}/{t} {100.0*p/t:.1f}%"


def fmt_est(pt: tuple, est_pct, is_full: bool, on: bool) -> str:
    """Append a "full-30 estimate" parenthesis after the measured string.

    on=False (estimation not enabled) or is_full=True (this cell already ran the full 30) → return only the measured value, omit the parenthesis.
    est_pct is None (no pool / denominator 0 / no data → cannot estimate) → append (—).
    Otherwise append (xx.x%), which may exceed 100% without truncation.
    """
    base = fmt(pt)
    if not on or is_full:
        return base
    if est_pct is None:
        return base + " (—)"
    return base + f" ({est_pct:.1f}%)"


DIM_LABELS = {
    1: "1.process·f2p(bt-scored, unmeasured nodes fall back to final)",
    2: "2.process·p2p(bt-scored, unmeasured nodes fall back to final)",
    3: "3.final·all-big·f2p",
    4: "4.final·all-big·p2p",
}

# By default only print D1/D3/D4 (process f2p / final f2p / final p2p) + mean(1/3/4) + per-cell cost;
# with --full print back the four dimensions D1–D4. The mean row is always based on (1,3,4), regardless of how many dimensions are printed.
DEFAULT_DIMS = (1, 3, 4)
AVG_DIMS = (1, 3, 4)


def avg_1_3_4(res: dict) -> float | None:
    """Arithmetic mean of the three percentages D1/D3/D4 (a dimension with N/A, i.e. denominator 0, is skipped; all missing → None).

    Shares the same convention as the mean row in print_model_table and the cross-model sorting (change one place and everything changes).
    """
    vals = [100.0 * res[d][0] / res[d][1] for d in AVG_DIMS if res[d][1]]
    return sum(vals) / len(vals) if vals else None

SHORT = {
    "non_programming_user": "non-programming user",
    "repository_steward": "repository steward",
    "product_stakeholder": "product stakeholder",
    "repo_newcomer_engineer": "newcomer engineer",
}


def discover_models() -> list[str]:
    if not os.path.isdir(OUTROOT):
        return []
    return sorted(d for d in os.listdir(OUTROOT)
                  if os.path.isdir(os.path.join(OUTROOT, d)))


def count_traces() -> int:
    """The number of traces in traces_30 (= the number of cells each model×persona should complete)."""
    n = 0
    for p in glob.glob(f"{TRACES}/*.json"):
        if "run_progress" in os.path.basename(p):
            continue
        n += 1
    return n


def cell_progress(root: str, persona: str) -> tuple[int, int]:
    """(completed, in-progress/created) -- completed = has final_eval.json; created = has a persona directory."""
    done = len(glob.glob(f"{root}/*/{persona}/final_eval.json"))
    built = len([d for d in glob.glob(f"{root}/*/{persona}")
                 if os.path.isdir(d)])
    return done, built


def print_progress(models: list[str], n_trace: int) -> None:
    """Overall completion-progress table: completed/created/expected(=n_trace) per model×persona, + global."""
    cw = 16
    width = 26 + cw * len(ORDER)
    print("=" * width)
    print(f"[completion progress] completed = has final_eval.json ; expected completion/persona = {n_trace} trace")
    print("=" * width)
    head = f"{'model':24s}"
    for p in ORDER:
        head += f"{SHORT[p]:>{cw - 4}s}"
    head += f"{'subtotal':>{cw}s}"
    print(head)
    print("-" * (width + cw))
    g_done = g_built = 0
    for m in models:
        root = os.path.join(OUTROOT, m)
        row = f"{m[:22]:24s}"
        m_done = m_built = 0
        for p in ORDER:
            done, built = cell_progress(root, p)
            m_done += done; m_built += built
            row += f"{f'{done}/{built}/{n_trace}':>{cw}s}"
        row += f"{f'{m_done}/{m_built}/{n_trace * len(ORDER)}':>{cw}s}"
        print(row)
        g_done += m_done; g_built += m_built
    print("-" * (width + cw))
    total = n_trace * len(ORDER) * len(models)
    pct_done = 100.0 * g_done / total if total else 0.0
    pct_built = 100.0 * g_built / total if total else 0.0
    print(f"format: completed/created/expected | global: completed {g_done}/{total} ({pct_done:.1f}%)  "
          f"created {g_built}/{total} ({pct_built:.1f}%)")
    print("=" * width + "\n")


def print_model_table(model: str, results: dict, cost: dict | None = None,
                      dims: tuple = DEFAULT_DIMS, ests: dict | None = None,
                      fulls: dict | None = None, est_on: bool = False,
                      cost_est: dict | None = None):
    """Print a single model's four-persona × dimension table.

    dims selects which dimensions to print (default D1/D3/D4; with --full pass the four dimensions D1–D4). After the dimension rows, one row is appended,
    "mean(1/3/4)" = the arithmetic mean of the three percentages D1/D3/D4 (always based on 1/3/4, regardless of dims;
    a dimension with denominator 0 is skipped, all three missing → N/A).
    When cost is not None ({persona: avg_usd}), another row "average cost per cell ($)" is appended.
    Cost convention A (assistant-side tokens only) + Y (only completed cells that have a final_eval, same denominator as this table's n_final);
    unit prices are the official real prices, including the cache discount, see usersim_cost_stats.PRICE_USD and COST_ESTIMATION_NOTES.md.

    When est_on, each cell that "did not run the full 30" appends a "(full-30 estimate)" parenthesis after the measured value:
      estimate = measured_S × (pool P full-30% / pool P limited-to-S%), a ratio correction, not truncated, may be >100%;
      cells that already ran the full 30 omit the parenthesis, cells with no pool / denominator 0 show (—).
      ests[persona][dim] = estimate% (or None); fulls[persona] = whether the full 30 was run.
    """
    cols = ORDER
    cw = 24 if est_on else 16
    width = 26 + cw * len(cols)
    print("\n" + "=" * width)
    ttl = f"[model={model}] user-sim four personas × dimension (numerator = PASSED nodes / denominator)"
    if est_on:
        ttl += "  parenthesis = full-30 estimate"
    print(ttl)
    print("=" * width)
    head = f"{'dimension':24s}"
    for p in cols:
        head += f"{SHORT[p]:>{cw - 4}s}"
    print(head)
    line = f"{'  completed cell(final)':24s}"
    for p in cols:
        line += f"{str(results[p]['n_final']):>{cw}s}"
    print(line)
    line = f"{'  participated in process stats(bigtask)':22s}"
    for p in cols:
        line += f"{str(results[p]['n_bigtask']):>{cw}s}"
    print(line)
    print("-" * width)
    for dim in dims:
        row = f"{DIM_LABELS[dim]:24s}"
        for p in cols:
            ep = (ests[p].get(dim) if ests else None)
            fp = (fulls[p] if fulls else False)
            row += f"{fmt_est(results[p][dim], ep, fp, est_on):>{cw}s}"
        print(row)
    # mean(1/3/4): arithmetic mean of the three percentages D1/D3/D4; a dimension with N/A (denominator 0) is skipped, all missing → N/A.
    print("-" * width)
    row = f"{'mean(1/3/4)':24s}"
    for p in cols:
        avg = avg_1_3_4(results[p])
        cell = f"{avg:.1f}%" if avg is not None else "N/A"
        if est_on and not (fulls and fulls[p]):
            evals = [ests[p][d] for d in AVG_DIMS
                     if ests and ests[p].get(d) is not None] if ests else []
            eavg = sum(evals) / len(evals) if evals else None
            cell += f" ({eavg:.1f}%)" if eavg is not None else " (—)"
        row += f"{cell:>{cw}s}"
    print(row)
    if cost is not None:
        print("-" * width)
        row = f"{'average cost per cell($)':22s}"
        for p in cols:
            v = cost.get(p)
            base = f'${v:.4f}' if v is not None else '—'
            # Cost estimation: same rule as accuracy -- full run omits the parenthesis; not full but estimable appends ($xx); not estimable (—).
            if est_on and not (fulls and fulls[p]):
                ce = (cost_est.get(p) if cost_est else None)
                base += f" (${ce:.4f})" if ce is not None else " (—)"
            row += f"{base:>{cw}s}"
        print(row)
    print("=" * width)


def completed_insts(root: str, persona: str) -> set:
    """The set of instances already completed (has final_eval.json) under this model×persona."""
    out = set()
    for fe in glob.glob(f"{root}/*/{persona}/final_eval.json"):
        out.add(fe.split("/")[-3])
    return out


def count_bigtask(root: str, persona: str) -> int:
    """The number of cells that truly participated in process stats (has bigtask_eval.json, including partials that did not reach final).

    In estimation mode the measurement is restricted to the final set (allowed=S), which would also squeeze n_bigtask down to n_final;
    but the "participated in process stats(bigtask)" row is a coverage metric and must retain the true (unrestricted) count, so it is counted separately.
    """
    return len(glob.glob(f"{root}/*/{persona}/bigtask_eval.json"))


def pool_ref(pool_roots: list[str], persona: str, subset: set | None,
             allgold_exclude: set | None, seed_reclass: set | None) -> dict:
    """The four-dimensional (pass,total) of the reference pool P on subset (None=each's own full 30), node-pooled (micro).

    Every model in the pool ran the full 30, so allowed=subset naturally covers any S; adding each model's
    numerator/denominator by node gives the pool-level accuracy. The allgold/seed convention is passed through exactly the same as the measurement.
    """
    agg = {d: [0, 0] for d in (1, 2, 3, 4)}
    for root in pool_roots:
        r = compute_persona(root, persona, allowed=subset,
                            allgold_exclude=allgold_exclude, seed_reclass=seed_reclass)
        for d in (1, 2, 3, 4):
            agg[d][0] += r[d][0]
            agg[d][1] += r[d][1]
    return {d: tuple(agg[d]) for d in agg}


def est_dim(measured_pt: tuple, ref_full_pt: tuple, ref_S_pt: tuple):
    """Ratio-correction estimate: measured% × (ref_full% / ref_S%).

    measured_pt = model M's (pass,total) on its final set S;
    ref_full_pt = pool P full-30's (pass,total); ref_S_pt = pool P restricted to S's (pass,total).
    Any denominator being 0, or ref_S passing nothing (=0%) → cannot estimate, return None. Not truncated, may be >100%.
    """
    mp, mt = measured_pt
    rfp, rft = ref_full_pt
    rsp, rst = ref_S_pt
    if not mt or not rft or not rst or rsp == 0:
        return None
    return 100.0 * (mp / mt) * ((rfp / rft) / (rsp / rst))


def pool_cost_avg(pool_members: list, persona: str, subset: set | None,
                  cost_by_inst: dict) -> tuple:
    """The cost of reference pool P on subset (None=each's all completed cells) (micro pooled).

    Returns (total_usd, n_cells): sum the per-instance cost of each model×persona in the pool by cell.
    Pool members must be in cost_by_inst (= priced in PRICE_USD and cost already computed); those not are automatically skipped.
    Exactly isomorphic to the accuracy pool_ref (same pool, same subset S).
    """
    tot = 0.0
    n = 0
    for m in pool_members:
        bi = cost_by_inst.get(m, {}).get(persona)
        if not bi:
            continue
        for inst, usd in bi.items():
            if subset is not None and inst not in subset:
                continue
            tot += usd
            n += 1
    return tot, n


def est_cost(measured_avg, full_ct: tuple, S_ct: tuple):
    """Cost ratio-correction estimate: measured_avg × (pool P full-30 per-cell mean / pool P limited-to-S per-cell mean).

    Same logic as est_dim, only replacing "pass rate" with "average cost per cell".
    measured_avg = model M's per-cell mean on its final set S (the measured cost shown in the current table);
    full_ct/S_ct = the (total_usd, n_cells) returned by pool_cost_avg (full-30 / limited-to-S).
    Any denominator being 0, or the pool S mean being 0 → cannot estimate, return None.
    """
    ft, fn = full_ct
    st, sn = S_ct
    if measured_avg is None or fn == 0 or sn == 0:
        return None
    s_avg = st / sn
    if s_avg == 0:
        return None
    return measured_avg * ((ft / fn) / s_avg)


def main():
    ap = argparse.ArgumentParser(
        description="user-sim four-dimensional accuracy statistics (optional common completed set / select models)")
    ap.add_argument("--models", nargs="+", metavar="M",
                    help="only count these models (default: all discovered model sub-directories)")
    ap.add_argument("--exclude", nargs="+", metavar="M", default=[],
                    help="exclude these models from the statistics (e.g. --exclude Qwen3.6-35B-A3B)")
    ap.add_argument("--common", action="store_true",
                    help="only compute the common completed set: for each persona take the intersection of instances completed by all selected models")
    ap.add_argument("--full", action="store_true",
                    help="print the complete four dimensions D1–D4 (default only prints D1/D3/D4 + mean(1/3/4) + per-cell cost)")
    ap.add_argument("--cost", action="store_true",
                    help="(enabled by default, kept only for backward compatibility) the average cost per cell ($) row is now printed by default; "
                         "convention A only assistant-side + Y only completed cells, same denominator as n_final, see the price table usersim_cost_stats.PRICE_USD")
    ap.add_argument("--pool-exclude", nargs="+", metavar="M", default=[],
                    help="only affects [the full-30 estimation reference pool P]: remove these models from the ratio-correction reference pool (they don't serve as others' reference), "
                         "their own rows/measured values are still shown normally. Default empty = behavior completely unchanged. Used to remove extreme-performing models to avoid polluting the est_dim correction ratio.")
    args = ap.parse_args()

    models = discover_models()
    if args.models:
        want = set(args.models)
        models = [m for m in models if m in want]
        missing = want - set(models)
        if missing:
            print(f"[warn] these specified models have no directory under {OUTROOT}, ignored: {sorted(missing)}")
    if args.exclude:
        models = [m for m in models if m not in set(args.exclude)]
    if not models:
        print(f"[warn] no models to count (OUTROOT={OUTROOT}, empty after filtering).")
        return

    # --common: for each persona compute the instance intersection (completed by all selected models)
    common_insts: dict[str, set] | None = None
    if args.common:
        common_insts = {}
        for p in ORDER:
            inter = None
            for m in models:
                s = completed_insts(os.path.join(OUTROOT, m), p)
                inter = s if inter is None else (inter & s)
            common_insts[p] = inter or set()
        print("=" * 60)
        print(f"[--common common completed set] selected {len(models)} models: {models}")
        for p in ORDER:
            print(f"  {SHORT[p]}({p}): number of commonly completed instances = {len(common_insts[p])}")
        print(f"  total common cells = {sum(len(v) for v in common_insts.values())}"
              f"(× {len(models)} models = {sum(len(v) for v in common_insts.values())*len(models)} samples)")
        print("=" * 60 + "\n")

    # Overall completion-progress table (printed first): completed/created/expected per model×persona + global percentages.
    n_trace = count_traces()
    print_progress(models, n_trace)

    # Dynamically load the allgold bad-node exclusion set (only affects dimensions 3/4).
    allgold_exclude = load_allgold_exclude()
    print(f"[allgold] exclusion set (dimensions 3/4) bad-node count = {len(allgold_exclude)}"
          f"(still FAILED/NOT_FOUND with full gold, dynamically read from {ALLGOLD_DIR})\n")

    # Dynamically load the seed-probe reclassification set (effective across all four dimensions): non-SEED p2p that did not pass on the base+SEED baseline → merged into f2p.
    seed_reclass = load_seedprobe_reclass()
    print(f"[seedprobe] non-SEED unpassed p2p→f2p reclassified node count = {len(seed_reclass)}"
          f"(base+SEED baseline p2p not PASSED, dynamically read from {SEEDPROBE_DIR})\n")

    # Cost: call usersim_cost_stats.avg_cost_usd once to compute the average cost ($) for all (model,persona).
    # Convention A (assistant-side tokens only) + Y (only completed cells that have a final_eval) → same denominator as this table's n_final.
    # Now [printed by default] (--cost no longer needed; --cost kept only for backward compatibility).
    # by_instance=True: additionally fetch per-instance cost details, reused for cost estimation (scaled with the same pool and same S as accuracy).
    # Fault tolerance: a failure in cost extraction does not affect the main body of the four-dimensional table (print a warning and continue, without appending the cost row).
    cost_map: dict[str, dict] | None = None
    cost_by_inst: dict[str, dict] = {}  # {model: {persona: {instance: usd}}}
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import usersim_cost_stats as _C
        print(f"[cost] computing average cost per cell ($): convention A (assistant only) + Y (completed cells only), "
              f"see the price table usersim_cost_stats.PRICE_USD / COST_ESTIMATION_NOTES.md …")
        raw = _C.avg_cost_usd(models=models, verbose=True, by_instance=True)
        cost_map = {m: {} for m in models}
        for (mm, pp), d in raw.items():
            cost_map.setdefault(mm, {})[pp] = d["avg_usd"]
            cost_by_inst.setdefault(mm, {})[pp] = d.get("by_inst", {})
    except Exception as e:  # noqa: BLE001
        print(f"[cost][warn] cost computation failed, skipping the cost row: {e!r}")
        cost_map = None
        cost_by_inst = {}

    # Dimension selection: default D1/D3/D4, --full prints back the complete four dimensions D1–D4. Passed through to the single-model table and the cross-model comparison table.
    dims = (1, 2, 3, 4) if args.full else DEFAULT_DIMS

    # Estimation mode: always on by default, disabled only with --common (the common set is already a fixed shared subset, no scaling needed).
    # Once enabled: ① the measured D1 is also restricted to this model×persona's final completed set S (same-S convention as D3/D4);
    #        ② each cell that "did not run the full 30" appends a "(full-30 estimate)" after the measured value.
    est_on = not args.common

    # Each model×persona's final completed set S (in estimation mode, serves as the measurement's allowed, and also the estimation's subset).
    sub_of: dict[str, dict] = {}
    if est_on:
        for m in models:
            root = os.path.join(OUTROOT, m)
            sub_of[m] = {p: completed_insts(root, p) for p in ORDER}

    # First compute all per-model results: {model: {persona: result}}
    # Estimation mode: allowed=S (the final set) → the measured D1 also lands in the final set; the "participated in process stats(bigtask)" row
    #          still covers the true count (including partials that did not reach final), so it is overwritten back to the true value via count_bigtask.
    all_results: dict[str, dict] = {}
    for model in models:
        root = os.path.join(OUTROOT, model)
        all_results[model] = {}
        for p in ORDER:
            if est_on:
                allowed = sub_of[model][p]
            elif common_insts is not None:
                allowed = common_insts[p]
            else:
                allowed = None
            res = compute_persona(root, p, allowed=allowed,
                                  allgold_exclude=allgold_exclude,
                                  seed_reclass=seed_reclass)
            if est_on:
                res["n_bigtask"] = count_bigtask(root, p)
            all_results[model][p] = res

    # Estimation: reference pool P (independent per persona) = models that ran the full 30 (n_final==n_trace) under that persona;
    #      if P is empty, give up estimation for that persona. ref_full=pool P full-30, ref_S=pool P limited-to-S (both node-pooled micro).
    #      estimate% = measured_S% × (ref_full% / ref_S%), ratio correction, not truncated, may be >100%.
    #      Cost likewise: cost estimate = measured_S per-cell mean × (pool P full-30 per-cell mean / pool P limited-to-S per-cell mean),
    #               same pool P, same subset S, reusing the per-instance cost details cost_by_inst.
    ests: dict[str, dict] = {}
    fulls: dict[str, dict] = {}
    cost_est: dict[str, dict] = {}   # {model: {persona: estimated per-cell cost or None}}
    if est_on:
        pool_roots: dict[str, list] = {}
        pool_names: dict[str, list] = {}
        ref_full: dict[str, dict | None] = {}
        cost_full: dict[str, tuple] = {}  # pool P full-30 cost (total_usd, n_cells)
        pool_excl = set(args.pool_exclude)
        for p in ORDER:
            members = [m for m in models
                       if len(sub_of[m][p]) == n_trace and m not in pool_excl]
            proots = [os.path.join(OUTROOT, m) for m in members]
            pool_roots[p] = proots
            pool_names[p] = members
            ref_full[p] = (pool_ref(proots, p, None, allgold_exclude, seed_reclass)
                           if proots else None)
            cost_full[p] = pool_cost_avg(members, p, None, cost_by_inst)
            print(f"[estimate] {SHORT[p]}({p}): reference pool P = {len(proots)} models ran the full {n_trace}"
                  + (f" → {members}" if members
                     else "(none → give up estimation for this persona)"))
        print()
        for m in models:
            ests[m] = {p: {} for p in ORDER}
            fulls[m] = {}
            cost_est[m] = {}
            for p in ORDER:
                S = sub_of[m][p]
                is_full = (len(S) == n_trace)
                fulls[m][p] = is_full
                if is_full or not pool_roots[p] or not S:
                    ests[m][p] = {d: None for d in (1, 2, 3, 4)}
                    cost_est[m][p] = None
                    continue
                rS = pool_ref(pool_roots[p], p, S, allgold_exclude, seed_reclass)
                rF = ref_full[p]
                ests[m][p] = {d: est_dim(all_results[m][p][d], rF[d], rS[d])
                              for d in (1, 2, 3, 4)}
                # Cost estimation: same pool P, same subset S, ratio-correct the measured per-cell mean.
                measured_c = (cost_map.get(m, {}).get(p) if cost_map else None)
                cS = pool_cost_avg(pool_names[p], p, S, cost_by_inst)
                cost_est[m][p] = est_cost(measured_c, cost_full[p], cS)

    # Sort the model print order by mean(1/3/4) descending + the column order of the cross-model comparison table (user requirement 2026-08-06).
    # Sort key = the D1/D3/D4 mean after [merging numerator/denominator] across the 4 personas (model-level aggregate, not a single persona);
    # no data (None) goes to the bottom, ties are stably sorted by model name. (In estimation mode the measurement is already the final-set convention, and the sort follows.)
    def _model_avg_key(model: str):
        agg = {}
        for d in AVG_DIMS:
            agg[d] = (sum(all_results[model][p][d][0] for p in ORDER),
                      sum(all_results[model][p][d][1] for p in ORDER))
        a = avg_1_3_4(agg)
        return (-(a if a is not None else -1.0), model)
    models = sorted(models, key=_model_avg_key)

    # per-model tables (already sorted by mean descending)
    for model in models:
        print_model_table(model, all_results[model],
                          cost=(cost_map.get(model) if cost_map else None),
                          dims=dims,
                          ests=(ests.get(model) if est_on else None),
                          fulls=(fulls.get(model) if est_on else None),
                          est_on=est_on,
                          cost_est=(cost_est.get(model) if est_on else None))

    # Cross-model comparison table (take each model's 4-persona total; column order as above, by mean descending)
    cw = 30 if est_on else 22
    width = 26 + cw * len(models)
    print("\n" + "=" * width)
    title = "[cross-model comparison: for each dimension add the 4 personas' numerator/denominator; sorted by mean(1/3/4) descending"
    if args.common:
        title += "; common completed set only]"
    elif est_on:
        title += "; parenthesis = full-30 estimate (each persona's estimate% weighted by its denominator)]"
    else:
        title += "]"
    print(title)
    print("=" * width)
    head = f"{'dimension':24s}"
    for m in models:
        head += f"{m[:cw-2]:>{cw}s}"
    print(head)
    print("-" * width)

    def _xmodel_est(m: str, dim: int):
        """Model m's cross-persona full-30 estimate% on dim: each persona's "full-30 estimate%" weighted-averaged by its denominator.

        · Fully-run personas: estimate% = that persona's measured% (already the full-30 truth), weight = its denominator.
        · Not-full-but-estimable personas: estimate% = ests[m][p][dim], weight = that persona's measured (subset S) denominator.
        · Non-estimable personas (None/denominator 0): do not participate.
        All 4 personas ran the full → return 'FULL' (meaning the parenthesis should be omitted); no contribution at all → None.
        """
        all_full = all(fulls[m][p] for p in ORDER)
        if all_full:
            return "FULL"
        num = den = 0.0
        for p in ORDER:
            pt = all_results[m][p][dim]
            w = pt[1]
            if not w:
                continue
            if fulls[m][p]:
                ep = 100.0 * pt[0] / pt[1]
            else:
                ep = ests[m][p].get(dim)
                if ep is None:
                    continue
            num += ep * w
            den += w
        return (num / den) if den else None

    for dim in dims:
        row = f"{DIM_LABELS[dim]:24s}"
        for m in models:
            tp = sum(all_results[m][p][dim][0] for p in ORDER)
            tt = sum(all_results[m][p][dim][1] for p in ORDER)
            cell = fmt((tp, tt))
            if est_on:
                xe = _xmodel_est(m, dim)
                if xe == "FULL":
                    pass  # fully run → omit the parenthesis
                elif xe is None:
                    cell += " (—)"
                else:
                    cell += f" ({xe:.1f}%)"
            row += f"{cell:>{cw}s}"
        print(row)
    print("=" * width)

    # ── Per-persona cross-model comparison tables: one table per persona, rows = D1/D3/D4 + mean(1/3/4) + per-cell cost, columns = models. ──
    # Estimation is [independent per persona, per metric]: use ests[m][persona][dim] directly (scaled by that persona's own difficulty pool),
    # with no cross-persona weighting. The cost row likewise carries that persona's independent estimate (cost_est[m][persona]). The column order follows the mean-descending order.
    for persona in ORDER:
        print("\n" + "=" * width)
        title = f"[per-persona · cross-model comparison: {SHORT[persona]} ({persona}); numerator = PASSED nodes / denominator; sorted by mean(1/3/4) descending"
        if args.common:
            title += "; common completed set only]"
        elif est_on:
            title += "; parenthesis = this persona's full-30 estimate]"
        else:
            title += "]"
        print(title)
        print("=" * width)
        head = f"{'dimension':24s}"
        for m in models:
            head += f"{m[:cw-2]:>{cw}s}"
        print(head)
        # Two coverage rows: completed cell(final) / participated in process stats(bigtask)
        line = f"{'  completed cell(final)':24s}"
        for m in models:
            line += f"{str(all_results[m][persona]['n_final']):>{cw}s}"
        print(line)
        line = f"{'  participated in process stats(bigtask)':22s}"
        for m in models:
            line += f"{str(all_results[m][persona]['n_bigtask']):>{cw}s}"
        print(line)
        print("-" * width)
        for dim in DEFAULT_DIMS:
            row = f"{DIM_LABELS[dim]:24s}"
            for m in models:
                ep = (ests[m][persona].get(dim) if est_on else None)
                fp = (fulls[m][persona] if est_on else False)
                row += f"{fmt_est(all_results[m][persona][dim], ep, fp, est_on):>{cw}s}"
            print(row)
        # mean(1/3/4): arithmetic mean of measured D1/D3/D4 + arithmetic mean of estimates (same convention as the per-model table).
        print("-" * width)
        row = f"{'mean(1/3/4)':24s}"
        for m in models:
            avg = avg_1_3_4(all_results[m][persona])
            cell = f"{avg:.1f}%" if avg is not None else "N/A"
            if est_on and not fulls[m][persona]:
                evals = [ests[m][persona][d] for d in AVG_DIMS
                         if ests[m][persona].get(d) is not None]
                eavg = sum(evals) / len(evals) if evals else None
                cell += f" ({eavg:.1f}%)" if eavg is not None else " (—)"
            row += f"{cell:>{cw}s}"
        print(row)
        # Average cost per cell ($): carries that persona's independent estimate (same rule as accuracy: full run omits the parenthesis, non-estimable (—)).
        if cost_map is not None:
            print("-" * width)
            row = f"{'average cost per cell($)':22s}"
            for m in models:
                v = cost_map.get(m, {}).get(persona)
                base = f'${v:.4f}' if v is not None else '—'
                if est_on and not fulls[m][persona]:
                    ce = cost_est.get(m, {}).get(persona)
                    base += f" (${ce:.4f})" if ce is not None else " (—)"
                row += f"{base:>{cw}s}"
            print(row)
        print("=" * width)


if __name__ == "__main__":
    main()
