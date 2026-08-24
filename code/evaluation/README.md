# evaluation — Scoring the User-Simulation Runs

Reads the closed-loop run outputs produced by [`user_simulation`](../user_simulation/)
(under `data/usersim_final/<model>/<instance>/<persona>/`) together with the
published [`benchmark/`](../../benchmark/), and prints the **score** and **cost**
tables. This is the last stage of the model-evaluation pipeline
(`../../run_model_eval.sh` runs `user_simulation` first, then this).

The benchmark is the **sole source of truth**: every denominator and calibration
the scorer needs (bad-node exclusion, p2p→f2p reclassification) is reverse-derived
from each task's `final_f2p` / `final_p2p` plus the `own_f2p` / `own_p2p` carried
in its `events` — nothing under `data/` is trusted for calibration.

---

## The four dimensions

For each `model × persona`, the scorer computes four accuracy figures (pass /
total over test nodes):

| Dim | Name | Source |
|-----|------|--------|
| **D1** | process · f2p | `bigtask_eval.json` (each big-task scored when it finished), falling back to `final_eval.json` |
| **D2** | process · p2p | same, for pass-to-pass |
| **D3** | final · f2p | `final_eval.json` (every subtask re-tested at the end) |
| **D4** | final · p2p | same, for pass-to-pass |

D1 measures "did it pass along the way"; D3 measures "does it still pass once
everything is done". The default table prints **D1 / D3 / D4** plus their mean;
`--full` adds D2.

---

## Scripts

### `stat_all_personas_final.py` — the scorer

Discovers every `<model>` subdirectory under `data/usersim_final/`, and for each
one prints a `D1/D3/D4 × 4-persona` table, then a cross-model comparison table
(ranked by mean). Reads only `final_eval.json` / `bigtask_eval.json` from the run
outputs and `*.json` from the benchmark. Prints to stdout; writes no files.

```bash
cd code/evaluation
python stat_all_personas_final.py            # all models, D1/D3/D4
```

| Flag | Meaning |
|------|---------|
| `--models M …` | only score these model subdirs (default: all discovered) |
| `--exclude M …` | skip these models |
| `--common` | restrict to the instances completed by *all* selected models (disables the full-set estimation) |
| `--full` | print all four dimensions (D1–D4) instead of D1/D3/D4 |
| `--cost` | add an average-cost row (pulls from `usersim_cost_stats.avg_cost_usd`) |
| `--pool-exclude M …` | drop these models from the reference pool used to estimate full-set scores |

### `usersim_cost_stats.py` — cost accounting

Parses the assistant trajectory files
(`…/<persona>/assistant/*.assistant.traj.json`) for token usage and computes
per-cell cost, aggregated by `model × persona`. Writes a CSV and a Markdown
report, and exposes `avg_cost_usd(...)` for the scorer's cost row.

```bash
python usersim_cost_stats.py                 # → data/usersim_cost_stats.csv + data/usersim_cost_report.md
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--src DIR` | `data/usersim_final` | run-output root to scan |
| `--csv FILE` | `data/usersim_cost_stats.csv` | per-cell CSV output |
| `--md FILE` | `data/usersim_cost_report.md` | Markdown summary output |
| `--workers N` | `8` | parallel scan processes |

> Token pricing is a built-in per-model table inside the script; edit
> `PRICE_USD` there to match your own providers before reading absolute costs.

---

## Configuration

Two-tier, identical to the other subdirs (env var **>** `config.yaml` **>**
built-in default). `config.yaml` has one optional `paths:` section:

```yaml
paths:
  data_dir: ""        # empty → ../../data
  benchmark_dir: ""   # empty → ../../benchmark
```

Environment overrides: `SWEJOURNEY_EVAL_CONFIG` (path to `config.yaml`),
`SWEJOURNEY_DATA_DIR`, `SWEJOURNEY_BENCHMARK_DIR` (shared with `trace_clean`, so a
relocated benchmark stays in sync). `config.py` exports `DATA_DIR` /
`BENCHMARK_DIR` / `USERSIM_OUT_DIR` (+ the `TRACES_DIR` / `ALLGOLD_DIR` /
`SEEDPROBE_DIR` aliases, all now pointing at the benchmark).

---

## Requirements

Pure Python (no Docker): PyYAML for config, plus `ijson` for streaming the large
trajectory files in the cost pass. Run `user_simulation` first so
`data/usersim_final/` is populated.
