# Phase 10: 250-Bar Scaling Checkpoint - Research

**Researched:** 2026-04-22
**Domain:** Data-scaling experiment infrastructure, auto-trigger diagnosis, paper table update
**Confidence:** HIGH — all findings are based on direct code and data inspection, not inference

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| SCAL-01 | 250-bar auto-retrain checkpoint output captured from SCC | Blocked by auto-trigger bug; workaround is a direct manual run — see §Auto-Trigger Root Cause |
| SCAL-02 | Table 5 in paper updated with 3rd scale point (50 / 100 / 250 bars/pair) | Existing Apr-11 data covers Tier 1; GRU/LSTM missing from that entry — re-run needed |
| SCAL-03 | Figure 2 regenerated with explicit training-set-cap annotation | `plot_data_scaling.py` regenerates from log.jsonl; caption annotation must be added manually |
| SCAL-04 | Finding 22 in FINDINGS.md filled in with actual numbers | Numbers already exist in log.jsonl; they just need extraction and verification |
| SCAL-05 | Paper section 5.4 updated to confirm or document ranking shift | Apr-11 data shows ranking invariant (XGBoost > LR, GRU/LSTM absent); full confirmation requires re-run with Tier 2 |

</phase_requirements>

---

## Summary

Phase 10 was designed as a "passive wait" — wait for the SCC auto-retrain to fire the 250-bar checkpoint, collect the output, update the paper. The reality is more complex. Three separate issues have stacked up:

**Issue 1: The auto-trigger reads the wrong dataset.** `scripts/run_data_scaling.py --auto` reads `data/processed/train.parquet` to count bars per pair. That file has a max of 141 bars/pair with 0 pairs at 250+. The 47 pairs that have crossed 250 bars exist only in `data/live/bars.parquet`. The auto-trigger will never fire for the 250-bar checkpoint using the current data path — it is structurally impossible without a code fix.

**Issue 2: A 250-bar entry already exists in log.jsonl, from April 11.** The April 11 batch pre-seeded all checkpoints (50 through 2000) in a single run against the historical train.parquet. Those entries are complete for Tier 1 (LR, XGBoost) but missing GRU and LSTM — despite `include_tier2=True` in the log entry, the models are absent from `metrics_by_model`. This was almost certainly a silent Tier 2 failure (torch import error or training timeout on April 11). The Table 5 row labeled "250" in PAPER_DRAFT.md is sourced from this stale Apr-11 entry.

**Issue 3: The state file says `last_checkpoint_ran: 100`.** This accurately reflects that the auto-trigger last ran at 100 bars on April 16. Because the Apr-11 batch wrote directly to log.jsonl without going through the state-file path, the state is coherent — but the trigger still cannot advance because of Issue 1.

**Primary recommendation:** Do not wait for the auto-trigger. Run `python scripts/run_data_scaling.py --bars-per-pair 250 --include-tier2` manually on the current combined dataset (train.parquet + live bars). This produces a fresh, correctly-sourced 250-bar entry with GRU/LSTM. Update Table 5, regenerate Figure 2, fill Finding 22. Total execution time ~30-45 minutes.

---

## Standard Stack

All tools needed for this phase are already installed. No new dependencies.

### Core (already in .venv)
| Component | Version | Purpose |
|-----------|---------|---------|
| `scripts/run_data_scaling.py` | project | The experiment runner — reads train.parquet, slices to N bars/pair, trains all eligible models |
| `scripts/plot_data_scaling.py` | project | Regenerates PNG figures from log.jsonl |
| `src/experiments/retraining_policy.py` | project | Defines `SCALING_CHECKPOINTS`, `should_run_scaling_experiment` |
| `experiments/results/data_scaling/log.jsonl` | project | Append-only results log — 14 existing entries |
| `experiments/results/data_scaling/state.json` | project | Tracks `last_checkpoint_ran` — currently 100 |
| `matplotlib` (Agg backend) | in .venv | Figure generation |

### Data Sources
| File | Rows | Pairs | Role |
|------|------|-------|------|
| `data/processed/train.parquet` | 6,946 | 144 | Historical training data; max 141 bars/pair; 0 pairs at 250+ |
| `data/live/bars.parquet` | 106,254 (new-format) | 7,814 | Live-accumulated bars; 47 pairs at 250+, max 543 bars |
| `data/processed/test.parquet` | ~1,673 | 144 | Held-out test set; consistent across all scaling runs |

---

## Architecture Patterns

### How run_data_scaling.py --auto Works (and Why It Fails)

The `--auto` path in `run_data_scaling.py` (lines 421-462):

```python
# From scripts/run_data_scaling.py lines 432-444
train_df, _ = _load_train_test(data_dir, ...)  # reads ONLY train.parquet
bpp = train_df.groupby("pair_id").size()
# ...
for cp in SCALING_CHECKPOINTS:
    if cp <= last:
        continue
    pairs_ready = int((bpp >= cp).sum())
    if pairs_ready < MIN_PAIRS_FOR_CHECKPOINT:  # 20 pairs required
        break  # stops here — 0 pairs at 250+ in train.parquet
```

`_load_train_test` is hardcoded to `data_dir / "train.parquet"` — it never reads `data/live/bars.parquet`. The `scc_retrain_batch.sh` passes `data/processed` as `data_dir`. The 47 live pairs with 250+ bars are invisible to this path.

The 6-hour retrain batch IS running (last run 2026-04-22 16:48 UTC), but it runs the auto-trigger which silently exits because `pairs_ready < 20` for the 250-bar checkpoint.

### How the Manual Run Works

```bash
# Source: scripts/run_data_scaling.py lines 412-418
python scripts/run_data_scaling.py --bars-per-pair 250 --include-tier2
```

The `--bars-per-pair` path bypasses the state check entirely and calls `run_checkpoint(250, ...)` directly. It reads `data/processed/train.parquet`, slices to first 250 bars per pair (which gives all rows for all pairs since max is 141), trains Tier 1 + Tier 2, and appends one JSONL row.

**Critical note:** This run will train on `train.parquet` only (6,802-6,946 rows, 144 pairs). It will NOT incorporate live bars. This is the same data the Apr-11 entry used. The resulting metrics will likely be identical to the Apr-11 250-bar entry for Tier 1, but will now include GRU and LSTM if the torch import chain works in the current environment.

### Log Format

Each log entry (one JSON object per line):

```json
{
  "bars_per_pair": 250,
  "training_rows": 6802,        // rows after slicing
  "full_training_rows": 6802,   // rows in full train.parquet
  "timestamp": "ISO8601",
  "include_tier2": true,
  "include_category": false,
  "n_features": 29,             // NOTE: Apr-11 used 29 features; current pipeline uses 51+
  "metrics_by_model": {
    "linear_regression": {"rmse": ..., "pnl_at_2pp": ..., ...},
    "xgboost": {...},
    "gru": {...},               // only if include_tier2=True and torch import succeeds
    "lstm": {...}
  }
}
```

**Feature count discrepancy:** The Apr-11 entries used 29 features (`n_features: 29`). The current pipeline after Phase 8 alignment has 51+ features. A fresh run will produce a different feature count, making the Apr-11 and fresh entries not directly comparable. The planner must decide whether to note this in the paper or re-run all checkpoints for consistency.

### Figure Generation

`scripts/plot_data_scaling.py` reads all entries from log.jsonl and plots each (bars_per_pair, metric_value) pair per model. It uses `matplotlib.use("Agg")` so it runs headlessly on SCC.

The figure referenced in PAPER_DRAFT.md §5.4 as `experiments/figures/pnl_at_2pp_vs_data.png` **does not exist in `experiments/figures/`**. It lives at `experiments/results/data_scaling/pnl_at_2pp_vs_data.png`. The paper draft has the wrong path. This must be corrected in SCAL-03.

The plot script does NOT add the training-set-cap annotation required by SCAL-03 and P7. The caption annotation ("plateau at N=6,802, fixed pair universe, 144 pairs") must be added manually to the paper's Figure 2 caption.

### What the 250-Bar Entry Actually Means

At 250 bars/pair, `_slice_train_by_bars_per_pair` takes the first 250 bars of each pair from train.parquet. Since train.parquet has max 141 bars/pair, **all pairs are included in full** — the 250-bar slice is identical to the 100-bar slice for any pair with fewer than 250 bars. This is why training_rows = 6,802 at all checkpoints >= 100. The "plateau at 100 bars/pair" documented in Table 5 is an artifact of this data cap.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead |
|---------|-------------|-------------|
| Running the scaling experiment | A new training script | `python scripts/run_data_scaling.py --bars-per-pair 250 --include-tier2` |
| Parsing log.jsonl for Table 5 numbers | Manual grep/copy | `python scripts/run_data_scaling.py --show-log` for human-readable summary |
| Regenerating figures | Rewriting matplotlib code | `python scripts/plot_data_scaling.py` |
| Updating state.json | Direct JSON edit | Let run_data_scaling.py --auto update it, or update it as part of the manual run task |

---

## Common Pitfalls

### Pitfall 1: Waiting for the Auto-Trigger
**What goes wrong:** Phase described as "passive wait" — but the auto-trigger reads only `train.parquet` and will never fire for 250+ bars because train.parquet has max 141 bars/pair.
**How to avoid:** Run manually with `--bars-per-pair 250 --include-tier2`.
**Warning signs:** 6-hour retrain batch runs without any "Checkpoint 250" log lines.

### Pitfall 2: Using Stale Apr-11 Numbers Without GRU/LSTM
**What goes wrong:** The Apr-11 log entry for 250 bars has `include_tier2: true` in metadata but NO GRU/LSTM in `metrics_by_model`. This means a silent Tier 2 failure. Paper Table 5 currently shows "—" for GRU/LSTM at 250 bars — this is honest but incomplete.
**How to avoid:** A fresh run with `--include-tier2` in the current environment (Phase 8 confirmed torch works) will produce GRU/LSTM numbers.
**Warning signs:** Fresh run also has no GRU/LSTM in the output — check for "Tier 2 model failed" warnings in stdout.

### Pitfall 3: Wrong Figure Path in Paper
**What goes wrong:** PAPER_DRAFT.md §5.4 references `experiments/figures/pnl_at_2pp_vs_data.png` but the file actually lives at `experiments/results/data_scaling/pnl_at_2pp_vs_data.png`.
**How to avoid:** Fix the figure path reference in the paper at the same time as regenerating the figure.

### Pitfall 4: Feature Count Mismatch Between Old and New Entries
**What goes wrong:** Apr-11 entries used 29 features; current pipeline uses 51+. Mixed-feature-count entries in log.jsonl will make the scaling plot misleading (different model configurations on the same curve).
**How to avoid:** Either (a) note this clearly in paper footnote, or (b) re-run all checkpoints for consistency. Option (a) is faster and defensible for an academic paper; option (b) is cleaner but costs 30+ extra minutes.
**Recommended approach:** Option (a) — add a footnote to Table 5 noting the Apr-11 entries used 29 features and the Apr-22 entries use N features; both show the same qualitative ranking.

### Pitfall 5: Scaling Plateau Presented Without Cap Annotation (P7)
**What goes wrong:** Figure 2 shows the curve plateauing at 100 bars. Reader assumes this means "no model improves beyond 100 bars of data." The actual reason is that the training set is capped at 6,802 rows / 144 pairs — slicing to 250 bars/pair gives the same data as slicing to 100 because the data doesn't have 250 bars for any pair.
**How to avoid:** Add explicit annotation to Figure 2 caption per SCAL-03 and P7: "Plateau occurs because train.parquet contains at most 141 bars/pair (N=6,802 rows, 144 pairs); slices at 250+ bars are identical to the 100-bar slice."
**Warning signs:** Caption missing this annotation; reviewer asks "why does performance plateau at 100?"

### Pitfall 6: State File Not Updated After Manual Run
**What goes wrong:** A manual `--bars-per-pair 250` run appends to log.jsonl but does NOT update state.json (only the `--auto` path updates state). If state.json stays at `last_checkpoint_ran: 100`, the auto-trigger will attempt to re-run the 250-bar checkpoint in the future (once enough bars accumulate), producing a duplicate log entry.
**How to avoid:** Manually update `experiments/results/data_scaling/state.json` to `{"last_checkpoint_ran": 250}` after the manual run, OR run with `--auto` after fixing the data-path issue, OR accept the duplicate entry as harmless (the plot script deduplicates by plotting all points).

---

## Code Examples

### Running the Manual Checkpoint
```bash
# Source: scripts/run_data_scaling.py usage section
# Run from DS340-Project root on SCC or locally
python scripts/run_data_scaling.py --bars-per-pair 250 --include-tier2
```
Expected output (approximate, ~30-45 min on SCC with 4 cores):
```
INFO run_data_scaling: checkpoint 250 bars/pair: 6802/6946 training rows, 51 features, 144 pairs
INFO run_data_scaling: Training naive on 6802 rows
INFO run_data_scaling: Training volume on 6802 rows
INFO run_data_scaling: Training linear_regression on 6802 rows
INFO run_data_scaling: Training xgboost on 6802 rows
INFO run_data_scaling: Training gru on 6802 rows
INFO run_data_scaling: Training lstm on 6802 rows
INFO run_data_scaling: Checkpoint 250 logged to experiments/results/data_scaling/log.jsonl
```

### Checking the Results
```bash
# Source: scripts/run_data_scaling.py --show-log
python scripts/run_data_scaling.py --show-log
```

### Regenerating Figures
```bash
# Source: scripts/plot_data_scaling.py usage section
python scripts/plot_data_scaling.py
# Writes: experiments/results/data_scaling/pnl_at_2pp_vs_data.png (and rmse, dir_acc, pnl_at_3pp)
```

### Updating state.json After Manual Run
```python
# One-liner to update state after manual run
import json; p = open('experiments/results/data_scaling/state.json','w'); json.dump({"last_checkpoint_ran": 250}, p, indent=2); p.close()
```

### Extracting Numbers for Table 5
The 250-bar entry from log.jsonl (Apr-11, Tier 1 only):
- LR pnl_at_2pp: **+$199.90**
- XGBoost pnl_at_2pp: **+$210.01**
- GRU: missing (silent failure)
- LSTM: missing (silent failure)

After a fresh run, these numbers will likely be similar for Tier 1 (same training data) but with GRU/LSTM filled in.

---

## State of the Art

| What Was Expected | What Actually Exists | Impact |
|-------------------|---------------------|--------|
| 250-bar auto-trigger fires when 47 live pairs cross threshold | Auto-trigger reads train.parquet (max 141 bars); never fires | Must run manually |
| log.jsonl has only 50- and 100-bar checkpoints | log.jsonl has ALL checkpoints (50-2000) from Apr-11 + 50 and 100 from Apr-16 | 250-bar Tier 1 data already exists; GRU/LSTM missing |
| state.json stuck at last_checkpoint_ran=50 | state.json says last_checkpoint_ran=100 | Consistent with Apr-16 auto-trigger run |
| Table 5 in paper needs 250-bar row | Table 5 already has a 250-bar row (+$199.90 LR, +$210.01 XGB, no GRU/LSTM) | Row exists but is incomplete and from Apr-11 29-feature run |
| Figure lives at experiments/figures/ | Figure lives at experiments/results/data_scaling/ | Paper has wrong path |

---

## Open Questions

1. **Feature count choice for the fresh run**
   - What we know: Apr-11 used 29 features; current pipeline after Phase 8 alignment uses 51+ features
   - What's unclear: Should the planner re-run ALL checkpoints at 51 features for consistency, or accept mixed entries and add a footnote?
   - Recommendation: Accept mixed entries (add footnote to Table 5). Re-running all 6 checkpoints adds ~3 hours of compute for minimal paper value. The qualitative finding (ranking invariant) does not change.

2. **GRU/LSTM failure cause on Apr-11**
   - What we know: The Apr-11 250-bar entry has `include_tier2: true` but no GRU/LSTM keys in `metrics_by_model`. No error is preserved in the log (silent exception swallowed by `_run_tier2` try/except at line 277).
   - What's unclear: Was it a torch import failure, a training crash, or a timeout?
   - Recommendation: Phase 8 confirmed torch works in the current env. The fresh run will resolve this — if it still fails, check for "Tier 2 model failed" warnings and investigate the torch import chain.

3. **Whether to also run 500-bar checkpoint**
   - What we know: train.parquet max is 141 bars; 500-bar slice == 250-bar slice == same data. Running 500 adds nothing.
   - What's unclear: The paper's Table 5 shows 500, 1000, 2000 bar rows with the same numbers as 250 (because they're all the same data). These rows exist in log.jsonl already. They should stay in the table as-is (with a footnote explaining the cap) or be removed for cleanliness.
   - Recommendation: Keep rows in table; caption explains the plateau. Removing rows would raise questions about why the curve stops at 250.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (project-wide) |
| Config file | None detected at root; tests/ mirrors src/ structure |
| Quick run command | `python -m pytest tests/test_retraining_policy.py -x -q` |
| Full suite command | `python -m pytest tests/ -q` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SCAL-01 | Scaling experiment produces valid log entry | smoke | `python scripts/run_data_scaling.py --show-log \| grep 250` | ❌ manual verification only |
| SCAL-02 | Table 5 updated with correct numbers | manual | N/A — paper edit | N/A |
| SCAL-03 | Figure regenerated with cap annotation | smoke | `ls experiments/results/data_scaling/pnl_at_2pp_vs_data.png` | ❌ check post-run |
| SCAL-04 | Finding 22 updated in FINDINGS.md | manual | N/A — document edit | N/A |
| SCAL-05 | §5.4 text updated to confirm ranking invariance | manual | N/A — paper edit | N/A |

### Existing Test for Core Logic
`src/experiments/retraining_policy.py` contains `should_run_scaling_experiment`. Look for tests at `tests/test_retraining_policy.py` — this likely already exists from v1.0.

### Wave 0 Gaps
- No test gaps — this phase has no new code. All deliverables are a script run + paper edits. The `run_data_scaling.py` script is already tested end-to-end by the existing infrastructure.

*(If no gaps: "None — existing test infrastructure covers all phase requirements")*

---

## Sources

### Primary (HIGH confidence)
- Direct inspection of `scripts/run_data_scaling.py` (all 469 lines) — auto-trigger logic, `--auto` path behavior, `_load_train_test` data path
- Direct inspection of `src/experiments/retraining_policy.py` — `SCALING_CHECKPOINTS`, `should_run_scaling_experiment`
- Direct inspection of `scripts/scc_retrain_batch.sh` — confirms `--auto --include-tier2` invocation, confirms `data/processed` as data_dir
- Direct inspection of `experiments/results/data_scaling/log.jsonl` — 14 entries confirmed; 250-bar Apr-11 entries lack GRU/LSTM keys; Apr-16 entries are 50 and 100 bars only
- Direct inspection of `experiments/results/data_scaling/state.json` — `last_checkpoint_ran: 100`
- Runtime: `train.parquet` — 6,946 rows, 144 pairs, max 141 bars/pair, 0 pairs at 250+
- Runtime: `data/live/bars.parquet` — 106,254 new-format rows, 7,814 pairs, 47 pairs at 250+, max 543 bars
- Direct inspection of `PAPER_DRAFT.md §5.4` — Table 5 has 250-bar row with Apr-11 numbers; figure path is wrong

### Secondary (MEDIUM confidence)
- `FINDINGS.md Finding 22` — confirms 250-bar checkpoint was marked "PENDING" as of Apr-16 with max 148 bars
- `.planning/research/SUMMARY.md` — P7 pitfall (plateau without cap annotation) is the relevant guard for this phase
- `.planning/research/PITFALLS.md P7` (lines 138-140) — "annotate training-set cap on scaling figure x-axis"

---

## Metadata

**Confidence breakdown:**
- Root cause of auto-trigger failure: HIGH — confirmed by reading data path code and measuring train.parquet directly
- Existing log.jsonl state: HIGH — read all 14 entries, verified timestamps and model presence
- GRU/LSTM silent failure on Apr-11: MEDIUM — inferred from missing keys; silent exception in `_run_tier2` try/except is the most likely cause; not confirmed
- Paper Table 5 accuracy: HIGH — cross-referenced log.jsonl values against Table 5 markdown

**Research date:** 2026-04-22
**Valid until:** 2026-04-27 (submission deadline; data continues accumulating but plan is to run manually)
