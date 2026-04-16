# Architecture Research — v1.1 Integrations

**Domain:** Cross-platform prediction market arbitrage (extending existing production system)
**Researched:** 2026-04-16
**Scope:** v1.1 milestone — TFT, live-vs-backtest reconciliation, feature ablation, ensemble exploration, 250-bar checkpoint
**Confidence:** HIGH (grounded in the shipped v1.0 codebase; patterns extend established contracts rather than inventing new ones)

> This document addresses v1.1-specific integration architecture. The v1.0 greenfield architecture from 2026-04-01 remains as history for context on why the existing contracts look the way they do.

## Architectural Stance for v1.1

**Guiding principle:** the BasePredictor contract, the per-experiment script layout, and the `data/processed` -> `experiments/results/*.json` -> analysis pipeline all work. v1.1 must *fit* into that skeleton, not reshape it. Every new file listed below is either (a) a new model implementing an existing ABC, (b) a new experiment script in the existing style, or (c) a single new analysis subpackage that stitches existing artifacts together.

The deadline is April 27. Integration elegance is a means to finishing, not an end in itself.

## System Overview (v1.1 Delta)

```
                          EXISTING PIPELINE (unchanged)
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  data/raw ──> src/matching ──> data/processed ──> src/features ──> src/ │
  │                                                                 models/ │
  └─────────────────────────────────────────────────────────────────────────┘
                                      │
                   ┌──────────────────┼──────────────────┐
                   ▼                  ▼                  ▼
          ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
          │  NEW: TFT    │   │ experiments/ │   │  src/live/   │
          │  (Tier 2)    │──▶│  run_*.py    │   │  strategy    │
          └──────────────┘   │  (existing   │   │  (existing)  │
                             │   + new)     │   └──────┬───────┘
                             └──────┬───────┘          │
                                    ▼                  ▼
                     ┌─────────────────────┐   positions.db
                     │ experiments/results │   trade_log.jsonl
                     │ /{tier1,tier2,...}  │          │
                     │  *.json             │          │
                     └──────────┬──────────┘          │
                                │                     │
                                └────────┬────────────┘
                                         ▼
                           ┌──────────────────────────┐
                           │  NEW: src/analysis/      │
                           │  reconciliation.py       │
                           │  (live vs backtest)      │
                           └─────────────┬────────────┘
                                         ▼
                           ┌──────────────────────────┐
                           │ experiments/results/     │
                           │   reconciliation/        │
                           │   *.json, *.md           │
                           └──────────────────────────┘
```

Five edges of new construction; the body of the codebase sits still.

## Answers to the Six Architecture Questions

### 1. TFT integration — follow the GRU pattern, don't invent a new one

**Recommendation:** `src/models/tft.py` inherits from `BasePredictor` and exposes `fit(X, y)` / `predict(X)` in the exact row-aligned contract that `GRUPredictor` already uses. All of TFT's `TimeSeriesDataSet` / encoder-length / known-vs-observed complexity hides inside `fit()`.

**Why mimic GRU rather than invent:**
- `experiments/run_baselines.run_tier2_with_seeds` (lines 206-275 of `experiments/run_baselines.py`) already expects `(X_with_group_id_column, y_array)` input. If TFT matches this contract, it reuses the existing multi-seed harness, the existing `save_results` schema, and the existing comparison table formatter with zero modification.
- `experiments/run_walk_forward.py` (lines 260-263) already has a single branch for sequence models: "if name in ('gru', 'lstm'): pass `X_train_seq`". Adding `'tft'` to that branch is a three-character change.
- GRU has already solved every interface mismatch between "PyTorch native training loop" and "row-aligned pandas predictions" — warm-up stitching, per-group windowing, scaler caching. TFT has the same shape of problem; reusing the pattern is cheaper than re-solving it.

**How the `TimeSeriesDataSet` awkwardness gets absorbed:**

```
TFTPredictor.fit(X, y):
    1. Assert 'group_id' column (same guard as GRU, copy-paste the message)
    2. Internally build a long-format DataFrame with columns:
         - time_idx (monotonic per group_id) — derive from row order within group
         - target (y)
         - group_id
         - static_categoricals = [group_id]
         - time_varying_known_reals = ['hours_to_resolution'] if present else []
         - time_varying_unknown_reals = [all other feature_cols]
    3. Wrap in pytorch_forecasting.TimeSeriesDataSet
    4. Train with pytorch_lightning.Trainer (max_epochs, early stopping)
    5. Cache: scaler, dataset_kwargs, feature_cols, training rows
       (same _cached_train idea as GRU for warm-up stitching)
    6. Set _fitted = True

TFTPredictor.predict(X):
    1. Same group_id guard
    2. Same warm-up stitching as GRU — stitch cached training rows before test rows per group
    3. Build TimeSeriesDataSet.from_dataset(training_dataset, stitched_df,
       predict=True, stop_randomization=True)
    4. TFT forward pass on the prediction loader
    5. Slice output to return len(X) predictions, row-aligned
```

Yes, TFT needs a `time_idx` column. The orchestration script already guarantees rows are sorted by `(pair_id, time_idx)`, so `time_idx` can be derived inside `fit()` from the cumulative count within group. No upstream change needed.

**New files:**
- `src/models/tft.py` — `TFTPredictor(BasePredictor)` (new; ~250 LOC estimate)
- `experiments/run_tft.py` — thin wrapper that loads data, runs TFT with 3 seeds, saves to `experiments/results/tier2/tft.json` using existing `save_results` (new; ~80 LOC)

**Modified files:**
- `experiments/run_baselines.py` — add `TFTPredictor` import, add to `tier2` list in `build_models`, add to `model_classes` list in `run_tier2_with_seeds` (≤5 lines total)
- `experiments/run_walk_forward.py` — extend the sequence-model branch to include `'tft'` (≤3 lines)
- `_MODEL_ORDER` in `run_baselines.py` — insert `"TFT"` after `"LSTM"` (1 line)

**Data-flow:** identical to GRU. Reads `data/processed/train.parquet` + `test.parquet`, writes `experiments/results/tier2/tft.json`, feeds the cross-tier comparison table unchanged.

**Confidence:** HIGH. PyTorch Forecasting's `TimeSeriesDataSet.from_dataset()` path is exactly how production-style TFT wrappers hide the dataset plumbing. The pattern is established.

---

### 2. Reconciliation pipeline — new `src/analysis/` package, thin experiment wrapper

**Recommendation:** create `src/analysis/reconciliation.py` containing the pure-logic reconciliation functions. `experiments/run_live_reconciliation.py` is a ~40-line CLI wrapper that loads inputs, calls the analysis functions, and writes artifacts.

**Why a new `src/analysis/` package rather than stuffing it into `src/evaluation/`:**

`src/evaluation/` owns things that plug into `BasePredictor.evaluate()` — regression metrics, profit simulation, results storage. Those are in-the-loop utilities called thousands of times per experiment. Reconciliation is a different class of thing: it compares two *completed* artifact sets (live trades + backtest predictions) and produces a report. Putting it in `evaluation/` would pollute the module with "out-of-loop analytics" and force `from src.evaluation import ...` imports for scripts that have nothing to do with the predict/evaluate cycle.

Precedent: this is why `src/live/` is its own package (runtime system) and `src/experiments/retraining_policy.py` is its own module (meta-logic about when to train). A third analytical layer — "compare what happened to what we predicted would happen" — belongs at a third tier. One package, one responsibility.

**Why not `experiments/run_live_reconciliation.py` with all logic inline:**

The reconciliation logic is non-trivial (timestamp alignment, trade-to-prediction matching, multiple comparison metrics). It should be unit-testable in `tests/analysis/test_reconciliation.py` without spinning up the full experiment entrypoint. The pattern that the codebase already uses — pure-logic module in `src/`, thin CLI wrapper in `experiments/` — applies directly.

**Data-flow shape (cleanest):**

```
Inputs (read-only):
  - data/live/positions.db (SQLite: closed_positions table)
  - data/live/position_history.jsonl (close records, redundant but easier to stream)
  - data/live/bars.parquet (live feature bars)
  - models/deployed/*.pkl (the models that were live at trade time)
  - experiments/results/tier1/*.json (backtest predictions — implicit via saved models)

Process (in src/analysis/reconciliation.py):
  1. Load live closed positions -> DataFrame
     columns: [pair_id, entry_ts, exit_ts, entry_spread, exit_spread,
               realized_pnl, direction, kalshi_ticker]

  2. For each closed position, reconstruct the features the model saw at
     entry time by reading bars.parquet at that (pair_id, timestamp) row.
     This is the counterfactual "what would the backtest have done."

  3. Run the same model that was live (LR + XGB from models/deployed/)
     on those features -> predicted spread change.

  4. Compute realized spread change from bars.parquet
     (spread at exit_ts - spread at entry_ts).

  5. Emit comparison metrics:
       - live_pnl vs backtest_predicted_pnl (using same profit_sim logic)
       - live_directional_accuracy vs backtest_directional_accuracy
       - slippage (live entry price vs backtest assumed entry price at bar close)
       - execution lag (bars between signal and fill)
       - hit-rate divergence per tier

Outputs (write):
  - experiments/results/reconciliation/summary.json
  - experiments/results/reconciliation/per_position.csv
  - experiments/results/reconciliation/report.md (human-readable section)
```

Key discipline: **re-use `src.evaluation.profit_sim.simulate_profit`** when computing the backtest-counterfactual P&L. If the reconciliation report runs trades through a different P&L simulator than the backtest, the whole comparison is meaningless.

**New files:**
- `src/analysis/__init__.py`
- `src/analysis/reconciliation.py` — pure-logic module
- `experiments/run_live_reconciliation.py` — CLI wrapper
- `tests/analysis/test_reconciliation.py` — tests (see section 6 for scope)

**Modified files:** none. Inputs are all read-only.

**Data-flow relationships:**
- Reads `positions.db` via `src/live/position_manager.PositionManager` (existing accessor — don't open SQLite directly in analysis code, let the position manager hand back DataFrames)
- Reads `bars.parquet` directly (same as `src/live/strategy.py` already does)
- Loads models via `BasePredictor.load()` (existing, unchanged)
- Uses `compute_derived_features` from `src/features/engineering` (existing, unchanged)

**Confidence:** HIGH. Every component this reaches across already has a stable public accessor.

---

### 3. Feature ablation — pure experiment-level, no `fit()` signature change

**Recommendation:** feature ablation lives entirely in `experiments/run_feature_ablation.py`. It filters the feature matrix columns *before* calling `model.fit(X, y)`. Do not add a `features` parameter to `BasePredictor.fit()`.

**Why experiment-level filtering wins:**

1. **Zero invasion.** Modifying `BasePredictor.fit(X, y)` to `fit(X, y, features=None)` ripples through nine model files (linear_regression, xgboost, gru, lstm, tft, naive, volume, ppo_raw, ppo_filtered) plus every test that mocks the ABC plus the cached live `BasePredictor.load()` contract. All for functionality that can be done in one line of pandas: `X[ablation_subset]`.

2. **The contract is already "whatever columns are in X, that's the feature set."** `LinearRegressionPredictor.fit` calls `self._model.fit(X_train, y_train)` — it uses every column. `GRUPredictor.fit` uses every column except `group_id`. The filtering point is *already* at the experiment boundary. Ablation is a natural extension of that existing semantics, not a new feature.

3. **Sequence models need `group_id` preserved.** If ablation logic lives inside `fit()` and accidentally filters out `group_id`, GRU crashes with its existing guard. Keeping filtering at the experiment boundary makes this invariant obvious (and the ablation script is responsible for always appending `group_id` to sequence subsets).

**Design of `experiments/run_feature_ablation.py`:**

```
Define feature groups as module-level constants (mirror the feature taxonomy
that's already implicit in engineering.py):

FEATURE_GROUPS = {
    "prices":       [all raw price/spread columns],
    "volumes":      [kalshi_volume, poly_volume, volume_ratio, ...],
    "microstruct":  [bid_ask, order_flow_imbalance, realized_spread, ...],
    "velocity":     [spread_velocity, price_velocity_*, rolling_std, ...],
    "time":         [hours_to_resolution, time_idx, ...],
    "category":     [category_oil, category_crypto, ...],  # from features/category.py
}

For each ablation:
    - "leave_one_group_out": 6 runs, each drops one group
    - "only_one_group":      6 runs, each keeps only one group
    - "core_minimum":        greedy forward-selection floor

For each (model_class, ablation_config):
    subset = ALL_FEATURES - excluded_group
    X_train_ab = X_train[subset + (['group_id'] if sequence else [])]
    X_test_ab  = X_test[subset + (['group_id'] if sequence else [])]
    model = model_class()
    model.fit(X_train_ab, y_train)
    metrics = model.evaluate(X_test_ab, y_test, timestamps=...)
    save_results(f"{model_name}_ablate_{group}",
                 metrics, results_dir, extra={"ablated_group": group,
                                              "n_features_used": len(subset)})
```

Results land in `experiments/results/ablation/*.json` using the existing `save_results` schema. The comparison table script can then render them the same way it renders the baselines.

**Which models get ablated:** LR, XGBoost, GRU, LSTM, (TFT if Phase 11 finishes). PPO ablation is skipped — PPO training is expensive and its features are the same subset, so running the same analysis on the regression models answers the "parsimony" question at 1/20th the cost.

**New files:**
- `experiments/run_feature_ablation.py` (new; ~200 LOC)

**Modified files:** none.

**Confidence:** HIGH. This is exactly what `run_experiment2_lookback.py` and `run_experiment3_threshold.py` already do — vary one axis, re-run the same models, save to a per-experiment subdirectory.

---

### 4. Ensemble design — dedicated `src/models/ensemble.py`, not experiment glue

**Recommendation:** build `src/models/ensemble.py` containing `EnsemblePredictor(BasePredictor)` that wraps a list of child predictors. Experiments instantiate `EnsemblePredictor([LR, XGB])` like any other model.

**Why a dedicated class beats experiment-level logic:**

1. **Live deployment reuse.** `src/live/strategy.py` lines 413-420 currently hardcodes the LR + XGB average:
   ```python
   lr_pred = float(self._lr_model.predict(features)[0])
   xgb_pred = float(self._xgb_model.predict(features)[0])
   avg_pred = (lr_pred + xgb_pred) / 2.0
   ```
   If v1.1's ensemble exploration finds a better combination (e.g., LR + XGB + GRU with weights 0.4 / 0.4 / 0.2), that finding is only useful if it can be *deployed*. A `BasePredictor` subclass pickles, loads, and predicts through the same `BasePredictor.load(path).predict(X)` call that live strategy already uses. Experiment-level glue doesn't.

2. **Testability.** Ensemble logic (weight normalization, concordance handling, NaN propagation) deserves its own tests. A class has a clean seam for unit tests. Inline experiment code doesn't.

3. **Composability in ablation and walk-forward.** Walk-forward runs a model per window. If ensemble is a `BasePredictor`, it slots into the walk-forward loop unchanged. If it's experiment glue, walk-forward needs a parallel code path.

**Design:**

```python
# src/models/ensemble.py
class EnsemblePredictor(BasePredictor):
    """Weighted average ensemble of child BasePredictor instances.

    Args:
        models: list of (name, predictor, weight) tuples, or list of predictors
                with equal weighting.
        concordance_mode: 'none' | 'strict' | 'soft'.  'strict' = only emit a
                non-zero prediction when all models agree on sign (matches the
                existing live-strategy concordance check).
    """
    def fit(self, X_train, y_train):
        for _, m, _ in self._models:
            m.fit(X_train, y_train)
        self._fitted = True
        return self

    def predict(self, X):
        preds = np.stack([m.predict(X) for _, m, _ in self._models])
        weights = np.array([w for _, _, w in self._models])
        weights = weights / weights.sum()
        if self._concordance == 'strict':
            agree = np.all(np.sign(preds) == np.sign(preds[0]), axis=0)
            weighted = (weights[:, None] * preds).sum(axis=0)
            return np.where(agree, weighted, 0.0)
        return (weights[:, None] * preds).sum(axis=0)
```

**Sequence-model gotcha:** children of mixed types (GRU + LR) need identical `X.columns` or the predict step diverges. The ensemble class should require consistent input — the experiment script is responsible for passing `X_with_group_id` so both children can consume it (LR ignores the extra column; GRU requires it).

**Live-deployment implications:**

When ensemble exploration picks a winner, `src/live/retrain.py` (existing) should be updated to fit and pickle the `EnsemblePredictor` instance, and `src/live/strategy.py` should replace its hardcoded LR/XGB average with `self._ensemble_model = BasePredictor.load(model_dir / 'ensemble.pkl')`. This is a ≤20 line change post-v1.1 but the architecture supports it.

**New files:**
- `src/models/ensemble.py` — `EnsemblePredictor(BasePredictor)`
- `experiments/run_ensemble_sweep.py` — runs several ensemble variants, saves to `experiments/results/ensemble/*.json`
- `tests/models/test_ensemble.py`

**Modified files:** none in v1.1 (live-deployment swap is a v1.2 task, out of current scope).

**Confidence:** HIGH for the architecture; MEDIUM for which ensemble variant wins (empirical question).

---

### 5. Build order — mostly respects dependencies, one parallelization opportunity

**Proposed order:** Phase 8 (cleanups) -> 9 (reconciliation) -> 10 (250-bar wait) -> 11 (TFT) -> 12 (feature ablation) -> 13 (ensemble) -> 14 (paper).

**Dependency analysis:**

| Phase | Depends on | Blocks |
|-------|-----------|--------|
| 8: cleanups + re-verify | v1.0 codebase | Everything (clean baseline numbers) |
| 9: live reconciliation | Phase 8 (stable models + results) | Paper section 5 |
| 10: 250-bar checkpoint | Wall-clock (accumulation) | Paper scaling story |
| 11: TFT | Phase 8 (baseline code path) | Phase 13 (if ensemble includes TFT) |
| 12: feature ablation | Phase 8 (baseline code path) | Paper parsimony section |
| 13: ensemble | Phase 8, **optionally 11** | Paper production-ensemble section |
| 14: paper | All of the above | Submission |

**Build-order issues:**

1. **Phase 10 is pure wall-clock wait.** The 250-bar retraining policy fires when the live system accumulates enough bars; this is hours-to-days of calendar time, not development time. **Phase 10 parallelizes with 11 and 12.** Concretely: kick off TFT development (Phase 11) and feature-ablation scripting (Phase 12) while the live system accumulates the bars that Phase 10 depends on. When the 250-bar checkpoint fires (automated via `src/experiments/retraining_policy.py`), collect the metrics output — it's a few hours of analysis work, not days.

2. **Phase 13 (ensemble) depends on Phase 11 (TFT) if-and-only-if the ensemble candidate includes TFT.** The safer move: design Phase 13 to include a (LR + XGB + GRU + LSTM) variant as the baseline ensemble, and TFT-including variants as optional upgrades. If Phase 11 slips, Phase 13 still ships.

3. **Phase 9 (reconciliation) only depends on the re-verified models from Phase 8**, not on any new v1.1 models. No reason to block it on anything else.

**Recommended execution order:**

```
Phase 8 ──┬──▶ Phase 9  (reconciliation)  ──┐
          │                                  │
          ├──▶ Phase 10 (live wait, passive) ┤
          │                                  │
          ├──▶ Phase 11 (TFT)               ┤
          │                                  │
          └──▶ Phase 12 (ablation)          ┴──▶ Phase 13 (ensemble) ──▶ Phase 14 (paper)
```

Phases 9, 10, 11, 12 can all proceed in parallel after Phase 8 completes. Phase 13 depends on 8, 11, 12. Phase 14 waits for 13.

**Confidence:** HIGH. Dependencies are structural (data-flow), and the parallelization point is clear.

---

### 6. Test strategy — strict for reconciliation + ensemble, light everywhere else

**Recommendation:** three test tiers, applied by risk rather than uniformly.

| Module | Test discipline | Rationale |
|--------|----------------|-----------|
| `src/analysis/reconciliation.py` | **Full unit + integration tests** | Numbers go into the paper. Off-by-one on timestamp alignment = wrong paper. |
| `src/models/ensemble.py` | **Unit tests for weighting logic + concordance** | Candidate for live deployment; correctness here affects real trades. |
| `src/models/tft.py` | **Smoke test only** (fits on toy data, predict returns correct shape) | Behavior is delegated to pytorch_forecasting. A full test suite would rebuild what they already test. |
| `experiments/run_feature_ablation.py` | **Smoke test** (runs end-to-end on tiny subset) | Orchestration script; logic is `X[cols]` + existing model calls. |
| `experiments/run_live_reconciliation.py` | **Smoke test** (runs on 3-position fixture) | Thin CLI wrapper; logic tested in `src/analysis/`. |
| `experiments/run_tft.py` | **None** | Trivial wrapper around `run_baselines.run_tier2_with_seeds`. |
| `experiments/run_ensemble_sweep.py` | **None** | Trivial wrapper. |

**Minimum test discipline that keeps us honest:**

1. **Reconciliation must be unit-tested.** The mapping from (pair_id, entry_ts) to "the features the model saw" is fiddly. If it's wrong, every reconciliation number is wrong, and the paper's unique "live-vs-backtest" evidence collapses. Required: tests for timestamp alignment, for trades that span missing bars, for pairs with no matching backtest predictions.

2. **Ensemble weight arithmetic must be tested.** Normalize to sum-to-one, concordance short-circuit, handling child that returns NaN. Three to five tests, under 100 LOC.

3. **Smoke tests, not comprehensive coverage, for the rest.** "Runs without crashing on a 3-pair 50-row fixture" is enough to catch the class of bug that would waste a day.

**What to skip on purpose:**
- Re-testing `pytorch_forecasting` internals.
- Re-testing `BasePredictor.evaluate` (already tested in v1.0).
- Property-based tests, coverage gates, mutation testing. Out of scope for a DS340 final project.

**Where to write tests:**
- `tests/analysis/test_reconciliation.py`
- `tests/models/test_ensemble.py`
- `tests/models/test_tft.py` (smoke)
- `tests/experiments/test_feature_ablation_smoke.py`

**Confidence:** HIGH. This is the minimum that protects the paper's headline numbers without turning the last two weeks into a test-writing exercise.

---

## Integration Points Summary

### New Files (v1.1)

| Path | Purpose | Depends on |
|------|---------|-----------|
| `src/models/tft.py` | TFTPredictor(BasePredictor) | `src/models/base.py`, `src/models/sequence_utils.py`, pytorch_forecasting |
| `src/models/ensemble.py` | EnsemblePredictor(BasePredictor) | `src/models/base.py` |
| `src/analysis/__init__.py` | New package | — |
| `src/analysis/reconciliation.py` | Live vs backtest comparison logic | `src/live/position_manager`, `src/evaluation/profit_sim`, `src/features/engineering` |
| `experiments/run_tft.py` | TFT runner CLI | `src/models/tft`, `experiments/run_baselines` utilities |
| `experiments/run_feature_ablation.py` | Ablation experiment CLI | `src/evaluation/results_store`, existing models |
| `experiments/run_ensemble_sweep.py` | Ensemble sweep CLI | `src/models/ensemble`, `src/evaluation/results_store` |
| `experiments/run_live_reconciliation.py` | Reconciliation CLI wrapper | `src/analysis/reconciliation` |
| `tests/analysis/test_reconciliation.py` | Reconciliation tests | — |
| `tests/models/test_ensemble.py` | Ensemble tests | — |
| `tests/models/test_tft.py` | TFT smoke test | — |

### Modified Files (v1.1)

| Path | Change | Size |
|------|--------|------|
| `experiments/run_baselines.py` | Add TFT to `tier2` list + `_MODEL_ORDER` | ≤5 lines |
| `experiments/run_walk_forward.py` | Add `'tft'` to sequence-model branch | ≤3 lines |

### Not Modified

- `src/models/base.py` — contract is stable, ablation does NOT need a new parameter
- `src/models/linear_regression.py`, `xgboost_model.py`, `gru.py`, `lstm.py`, `naive.py`, `volume.py`, `ppo_raw.py`, `ppo_filtered.py`, `autoencoder.py`
- `src/evaluation/*` — all existing utilities consumed unchanged
- `src/live/*` — no changes for v1.1 (ensemble deployment swap is post-v1.1)
- `src/experiments/retraining_policy.py` — 250-bar checkpoint fires via existing logic
- `src/features/*` — feature set is fixed for v1.1
- `src/matching/*` — matching pipeline is frozen
- `src/data/*` — ingestion is frozen
- `scripts/*` — cron scripts unchanged
- `.github/workflows/*` — GHA fallback unchanged

## Architectural Patterns Applied

### Pattern 1: Predictor ABC as Universal Adapter

**What:** Every new model — TFT, Ensemble — implements `BasePredictor`. Anything that consumes models (experiments, live strategy, walk-forward, ablation) does so through the ABC.

**Why:** Nine model files, one interface, zero special-casing. Adding model N+1 is always "one file, one import, one list entry."

**v1.1 application:** `TFTPredictor` and `EnsemblePredictor` are both `BasePredictor` subclasses. They plug into the cross-tier comparison table, the walk-forward backtest, the feature-ablation experiment, and (prospectively) live deployment with no orchestration changes.

### Pattern 2: Pure Logic in `src/`, Thin CLI in `experiments/`

**What:** Non-trivial logic lives in `src/analysis/reconciliation.py` as testable functions. `experiments/run_live_reconciliation.py` is a 40-line CLI wrapper.

**Why:** Unit-testability without CLI overhead. Reusability from notebooks. Clear separation between "the thing that computes" and "the thing the human invokes."

**v1.1 application:** applies to reconciliation only. TFT, ensemble, and ablation have their logic inside model classes or at the experiment boundary (where `X[cols]` is sufficient), so no new `src/` module is warranted for them.

### Pattern 3: Per-Experiment Results Directory with Fixed JSON Schema

**What:** Every experiment writes `experiments/results/{experiment_name}/{model_name}.json` using `src.evaluation.results_store.save_results`. The schema includes `metrics`, `extra` (config + pnl_series), and standardized fields.

**Why:** Comparison across experiments is mechanical. The paper's tables are generated by reading all JSON files and rendering. No CSV parsing, no format migration.

**v1.1 application:** reconciliation, ablation, and ensemble all emit results in this schema. The cross-tier comparison table from `run_baselines` already handles arbitrary model names; no new formatter logic needed.

### Pattern 4: Warm-up Stitching for Row-Aligned Sequence Predictions

**What:** Sequence models (GRU, LSTM, TFT) cache training rows during `fit()` and prepend them to test rows during `predict()` so every test row gets a full-length lookback window. Row-aligned output means direct comparison with Tier 1 flat models.

**Why:** Without stitching, sequence models return `len(X) - lookback` predictions and misalign with regression models. The cross-tier comparison becomes impossible.

**v1.1 application:** TFT reuses this pattern directly. The `_cached_train` dict in `TFTPredictor` mirrors `GRUPredictor._cached_train`.

## Anti-Patterns to Avoid in v1.1

### Anti-Pattern 1: Adding a `features` parameter to `BasePredictor.fit()`

**What people do:** "Feature ablation needs to select columns — let's extend the ABC."
**Why wrong:** Ripples through nine model files, two training scripts, live strategy's pickle load, and every test. All to replace a one-liner (`X[cols]`) that belongs at the experiment boundary.
**Do this instead:** Filter columns in `experiments/run_feature_ablation.py` before calling `model.fit(X_subset, y)`.

### Anti-Pattern 2: Inline reconciliation logic in the CLI script

**What people do:** Stuff 400 lines of SQLite joins, timestamp math, and counterfactual inference into `experiments/run_live_reconciliation.py`.
**Why wrong:** Untestable. When the reconciliation number looks wrong (and it will, first time), every debugging session requires spinning up the full CLI. Unit tests are impossible.
**Do this instead:** `src/analysis/reconciliation.py` for logic; `experiments/run_live_reconciliation.py` for CLI only.

### Anti-Pattern 3: Ensemble as experiment-glue

**What people do:** Compute `(lr.predict(X) + xgb.predict(X)) / 2` directly in the experiment script.
**Why wrong:** Not deployable (live strategy has no way to consume experiment glue). Not testable as a unit. Not reusable in walk-forward or ablation.
**Do this instead:** `src/models/ensemble.EnsemblePredictor` — same `fit/predict/save/load` surface as everything else.

### Anti-Pattern 4: Reimplementing TFT's dataset plumbing upstream

**What people do:** "TFT needs `time_idx` and known/unknown feature split — let's add those columns during feature engineering so the model doesn't have to deal with it."
**Why wrong:** (a) Forces every other model to carry columns it doesn't use. (b) Couples upstream to a library decision that might change. (c) Wastes time; `time_idx` is derivable from sorted `group_id` + row order in three lines.
**Do this instead:** Hide all `TimeSeriesDataSet` preparation inside `TFTPredictor.fit()`.

### Anti-Pattern 5: Uniform test discipline

**What people do:** "Every new module gets 80% coverage."
**Why wrong:** v1.1 has two weeks. Comprehensive coverage of a TFT wrapper (mostly delegating to pytorch_forecasting) spends days to re-prove what an external library already tests. Reconciliation gets under-tested because testing budget is spent elsewhere.
**Do this instead:** Risk-weighted tests. Reconciliation (paper-critical) and ensemble (deployable) get full tests. Thin wrappers get smoke tests. Delegation layers skip tests.

## Data-Flow Diagrams for New v1.1 Capabilities

### TFT Training & Evaluation

```
data/processed/train.parquet ──┐
data/processed/test.parquet  ──┤
                               ├──▶ experiments/run_tft.py
                               │        │
                               │        ├── _build_split (shared helper)
                               │        ├── prepare_xy_for_seq (X + group_id)
                               │        ├── TFTPredictor(seed=42|123|456).fit
                               │        └── .evaluate(threshold=0.02, timestamps=...)
                               │             │
                               │             ▼
                               └──── save_results("TFT", metrics, tier2/, extra)
                                        │
                                        ▼
                              experiments/results/tier2/tft.json
                                        │
                                        ▼
                              (consumed by run_baselines --tier all comparison table,
                               run_ensemble_sweep if TFT included,
                               run_feature_ablation for TFT variants if time permits)
```

### Live vs Backtest Reconciliation

```
data/live/positions.db ────────┐
data/live/position_history.jsonl ┤
data/live/bars.parquet ────────┤
models/deployed/*.pkl ─────────┤
                               │
                               ▼
                    src/analysis/reconciliation.py
                         │
                         ├── load_closed_positions(positions.db)
                         ├── reconstruct_entry_features(bars.parquet, pair_id, entry_ts)
                         ├── counterfactual_predictions(models, features)
                         ├── counterfactual_profit_sim(preds, actuals)  ◀── uses src/evaluation/profit_sim
                         └── compute_divergence_metrics(live, counterfactual)
                                 │
                                 ▼
                   experiments/run_live_reconciliation.py (CLI)
                                 │
                                 ▼
        experiments/results/reconciliation/
                        ├── summary.json       (headline metrics)
                        ├── per_position.csv   (one row per closed trade)
                        └── report.md          (human-readable; paper section draft)
```

### Feature Ablation

```
data/processed/train.parquet ──┐
data/processed/test.parquet  ──┤
                               ├──▶ experiments/run_feature_ablation.py
                               │        │
                               │        ├── FEATURE_GROUPS = {...}  (module-level)
                               │        │
                               │        └── for (model_class, ablation) in grid:
                               │              X_subset = X[group_subset + maybe_group_id]
                               │              model.fit(X_subset, y)
                               │              metrics = model.evaluate(X_test_subset, y_test)
                               │              save_results(f"{name}_ablate_{group}", ...)
                               │
                               ▼
                     experiments/results/ablation/
                         *.json  (one per (model, ablation) pair)
                                 │
                                 ▼
                  (consumed by paper's parsimony section table generator)
```

### Ensemble Sweep

```
data/processed/train.parquet ──┐
                               ├──▶ experiments/run_ensemble_sweep.py
                               │        │
                               │        ├── define ensemble variants:
                               │        │     - LR + XGB (baseline, matches live)
                               │        │     - LR + XGB + GRU
                               │        │     - LR + XGB + GRU + LSTM
                               │        │     - LR + XGB + TFT (if Phase 11 done)
                               │        │     - LR + XGB + GRU with concordance
                               │        │
                               │        └── for variant in variants:
                               │              ens = EnsemblePredictor(variant)
                               │              ens.fit(X_train_with_group_id, y_train)
                               │              metrics = ens.evaluate(X_test, y_test)
                               │              save_results(f"Ensemble_{variant_name}", ...)
                               ▼
                     experiments/results/ensemble/
                         *.json
                                 │
                                 ▼
                  (consumed by paper + optionally fed to v1.2 live deployment)
```

## Scaling Considerations

**Actual scale for v1.1:**

| Concern | Current | v1.1 target |
|---------|---------|-------------|
| Matched pairs | ~144 (after quality filter) | ~144 (no matching work in v1.1) |
| Bars per pair (offline) | ~100-500 | unchanged |
| Bars per pair (live) | accumulating | ~250 at checkpoint |
| Feature count | 59 | 59 (no new features) |
| Model count | 9 | 11 (add TFT + Ensemble) |
| Experiment scripts | 11 | 14 (add TFT, ablation, ensemble, reconciliation) |

No scaling concerns for v1.1. All runs fit on a laptop in hours. TFT training is the slowest new component (~15 min per seed on CPU, ~3 min on GPU if available). The ablation grid is (5 groups x 5 models x leave-one-out + only-one-out) = 50 runs, each <2 min for Tier 1, so <2 hours total.

If TFT turns out to be GPU-bound, the BU SCC deployment already has GPU access; run TFT there and scp the `.json` back.

## Sources

- v1.0 `.planning/research/ARCHITECTURE.md` (2026-04-01) — establishes the contracts that v1.1 extends.
- `src/models/base.py` — actual `BasePredictor` ABC, drives question 3's "don't add a `features` parameter" conclusion.
- `src/models/gru.py` — the warm-up-stitching pattern that TFT inherits.
- `experiments/run_baselines.py` — the multi-seed harness that TFT plugs into.
- `experiments/run_walk_forward.py` — shows where TFT needs a 3-line addition to the sequence-model branch.
- `src/live/strategy.py` — shows the ensemble integration point for future deployment.
- `src/experiments/retraining_policy.py` — confirms 250-bar checkpoint fires automatically.
- PyTorch Forecasting `TimeSeriesDataSet` documentation (existing pattern for wrapping TFT; v1.0 confidence assessment still applies).

---
*Architecture research for: v1.1 integrations (TFT, reconciliation, ablation, ensemble, checkpoint)*
*Researched: 2026-04-16*
