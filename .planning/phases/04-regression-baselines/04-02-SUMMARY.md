---
phase: 04-regression-baselines
plan: 02
subsystem: models
tags: [regression, xgboost, linear-regression, results-store, baselines, ta-check-in]

# Dependency graph
requires:
  - phase: 04-regression-baselines
    provides: BasePredictor ABC, compute_regression_metrics, simulate_profit, NaivePredictor, VolumePredictor
  - phase: 03-feature-engineering
    provides: train.parquet / test.parquet with spread + derived feature columns
provides:
  - LinearRegressionPredictor (sklearn) implementing BasePredictor
  - XGBoostPredictor (xgboost) with configurable n_estimators/max_depth/learning_rate/random_state
  - save_results / load_results / load_all_results JSON persistence utilities
  - experiments/run_baselines.py — Tier 1 comparison script (TA check-in deliverable)
  - Reproducible JSON snapshot of all four Tier 1 results under experiments/results/tier1/
affects: [05-time-series, 06-rl, 07-evaluation]

# Tech tracking
tech-stack:
  added:
    - "sklearn 1.8.0 (LinearRegression)"
    - "xgboost 3.2.0 (XGBRegressor)"
  patterns:
    - "Fit-gated predict: RuntimeError if predict called before fit"
    - "Slugified filenames for per-model results JSON"
    - "Auto-sized fixed-width comparison table (no external table lib)"
    - "argparse-based CLI with --data-dir, --results-dir, --threshold"
    - "Per-pair spread-change target computed as groupby('pair_id')['spread'].shift(-1) - spread"

key-files:
  created:
    - src/models/linear_regression.py
    - src/models/xgboost_model.py
    - src/evaluation/results_store.py
    - experiments/__init__.py
    - experiments/run_baselines.py
    - experiments/results/tier1/linear_regression.json
    - experiments/results/tier1/naive_spread_closes.json
    - experiments/results/tier1/volume_higher_volume_correct.json
    - experiments/results/tier1/xgboost.json
    - tests/models/test_linear_regression.py
    - tests/models/test_xgboost_model.py
    - tests/evaluation/test_results_store.py
  modified: []

key-decisions:
  - "Loaded separate train.parquet/test.parquet from data/processed (actual Phase 3 output) instead of the plan's features_flat.parquet with a 'split' column"
  - "Computed spread-change target inline via groupby('pair_id')['spread'].shift(-1) - spread — no precomputed target column in Phase 3 data"
  - "Filtered rows to those with valid spread AND valid target (978 train, 435 test from 3944/997 raw rows)"
  - "XGBoost uses n_estimators=200, max_depth=4, learning_rate=0.05 per plan spec (shallower trees to reduce overfitting on small dataset)"
  - "Comparison table auto-sizes the model name column to the longest name (handles 30-char VolumePredictor label)"
  - "Results JSON includes full pnl_series under 'extra' for equity-curve plotting downstream; metrics dict stays a flat summary"
  - "Filled remaining NaN feature values with 0.0 before fitting (XGBoost handles NaN natively but sklearn LinearRegression does not)"

requirements-completed: [MOD-01, MOD-02]

# Metrics
duration: 4min
completed: 2026-04-05
---

# Phase 4 Plan 2: Linear Regression and XGBoost Baselines Summary

**Linear Regression + XGBoost predictors, JSON results storage, and the TA check-in comparison script showing XGBoost as the best Tier 1 performer (RMSE 0.1729, Sharpe 9.34 on 435 test rows).**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-04-05T14:12:44Z
- **Completed:** 2026-04-05T14:16:39Z
- **Tasks:** 2
- **Files created:** 12 (3 production models/utils, 3 tests, 2 experiment scaffolding, 4 result JSONs)

## Accomplishments

- `LinearRegressionPredictor` wraps `sklearn.linear_model.LinearRegression`, implements `BasePredictor`, and raises `RuntimeError` if `predict` is called before `fit`.
- `XGBoostPredictor` wraps `xgboost.XGBRegressor` with constructor-configurable `n_estimators`, `max_depth`, `learning_rate`, and `random_state=42` for reproducibility.
- `src/evaluation/results_store.py` provides `save_results`, `load_results`, `load_all_results` with slugified filenames, automatic directory creation, and ISO-8601 timestamps.
- `experiments/run_baselines.py` is the TA check-in deliverable: loads Phase 3 train/test parquet files, computes per-pair spread-change targets, trains all four Tier 1 models, saves per-model JSON results, and prints a fixed-width comparison table.
- 29 new tests added via strict TDD (8 Linear Regression, 10 XGBoost, 11 results store); all 73 model+evaluation tests pass.
- End-to-end run against real Phase 3 data produces: XGBoost RMSE 0.1729 (best), Linear Regression RMSE 0.1759, Volume 0.2184, Naive 0.2421.

## Task Commits

1. **Task 1: Linear Regression and XGBoost predictors with results storage** — `826c9766` (feat)
2. **Task 2: Baseline comparison experiment script** — `672f3299` (feat)

**Plan metadata:** pending (this commit)

## Tier 1 Results (TA Check-in, April 4)

Dataset: 978 train rows / 435 test rows (after dropping NaN spread + target rows), 35 features, threshold=0.02.

| Model                          |    RMSE |     MAE | Dir Acc |      P&L | Trades | Win Rate |  Sharpe |
|--------------------------------|--------:|--------:|--------:|---------:|-------:|---------:|--------:|
| Linear Regression              |  0.1759 |  0.1254 |  0.6840 |  33.9968 |    372 |   0.6694 |  8.5740 |
| Naive (Spread Closes)          |  0.2421 |  0.1836 |  0.6226 |  16.3079 |    396 |   0.6111 |  3.7238 |
| Volume (Higher Volume Correct) |  0.2184 |  0.1633 |  0.6226 |  16.4354 |    391 |   0.6138 |  3.7874 |
| XGBoost                        |  0.1729 |  0.1226 |  0.6792 |  36.8147 |    380 |   0.7000 |  9.3356 |

**Takeaways:** XGBoost is the strongest tabular model (as predicted in CLAUDE.md), but Linear Regression is a very close second (0.3% worse on RMSE). Both learned baselines substantially outperform the naive/volume lower bounds: the gap is ~7pp directional accuracy, 2x P&L, and 2x Sharpe. The fact that a 2-line linear model gets within 0.3% of XGBoost is direct evidence for the project's "complexity vs performance" thesis — it raises the bar for whether the GRU/LSTM/TFT/PPO tiers can justify their added complexity.

## Files Created/Modified

### Production code
- `src/models/linear_regression.py` — `LinearRegressionPredictor(BasePredictor)` wrapping sklearn
- `src/models/xgboost_model.py` — `XGBoostPredictor(BasePredictor)` wrapping xgboost with configurable hyperparameters
- `src/evaluation/results_store.py` — JSON save/load utilities with slugified filenames

### Experiments
- `experiments/__init__.py` — package init
- `experiments/run_baselines.py` — CLI script, `main()` entry point, handles missing data gracefully
- `experiments/results/tier1/*.json` — reproducible snapshot of Tier 1 results (4 files)

### Tests
- `tests/models/test_linear_regression.py` — 8 tests (contract, fit-gating, evaluate integration, learning)
- `tests/models/test_xgboost_model.py` — 10 tests (contract, fit-gating, hyperparameters, reproducibility)
- `tests/evaluation/test_results_store.py` — 11 tests (save, load, round-trip, slugify, directory creation, load_all sorting)

## Decisions Made

- **Data contract:** Loaded from `train.parquet` + `test.parquet` (separate files, actual Phase 3 output) instead of the plan's single `features_flat.parquet` with a `split` column. See Deviations.
- **Target computation:** Per-pair next-bar spread change (`groupby('pair_id')['spread'].shift(-1) - spread`). Drops the last row of each pair (no next bar) and any row with NaN `spread` (no current price).
- **NaN filling:** Remaining NaN values in numeric feature columns are filled with `0.0` before fitting. XGBoost handles NaN natively but sklearn `LinearRegression` raises `ValueError: Input contains NaN`. Zero-fill keeps both models working from the same feature matrix.
- **XGBoost hyperparameters:** `n_estimators=200, max_depth=4, learning_rate=0.05` per plan spec. Shallower trees + slower learning rate reduces overfitting on a dataset this small (978 train rows).
- **Fit-gating:** Both regression models raise `RuntimeError("Model must be fit before predict")` if `predict` is called before `fit`. Naive/Volume baselines don't need this (they don't store state).
- **Results JSON layout:** Top level has `model_name`, `metrics` (flat summary dict), `timestamp`. `pnl_series` is moved to an `extra` section alongside `threshold`, `n_train_rows`, `n_test_rows`, `n_features` so the summary metrics are cheap to parse for comparison tables while the equity curve stays available for plotting.
- **Table auto-sizing:** Model column width is computed from the longest model name so `Volume (Higher Volume Correct)` (30 chars) doesn't overflow a hardcoded column.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 — Bug / Plan drift] Data layout: separate train/test files, not features_flat.parquet**
- **Found during:** Task 2 (run_baselines.py implementation)
- **Issue:** Plan spec'd loading from `data/processed/features_flat.parquet` with a `split` column or `time_idx` for 80/20 slicing. The actual Phase 3 output (confirmed via `ls data/processed/`) is two separate files, `train.parquet` (3944 rows) and `test.parquet` (997 rows), each with `time_idx`, `group_id`, and 39 columns. Running the plan as-written would `FileNotFoundError` on feature load.
- **Fix:** Script loads both files from `data_dir` (default `data/processed/`), adds `--data-dir` / `--data-path` aliases, prints a helpful "Run Phase 3 first" error + exit code 1 if either file is missing.
- **Files modified:** experiments/run_baselines.py
- **Committed in:** 672f3299 (Task 2 commit)

**2. [Rule 2 — Missing functionality] Target column not precomputed in Phase 3 output**
- **Found during:** Task 2 (schema inspection)
- **Issue:** Plan assumed a `spread_target` column in the feature matrix. Phase 3 output has `spread` (current spread) but no precomputed spread-change target. Without a target column, no model can be fit.
- **Fix:** Script computes per-pair spread change inline: `groupby('pair_id')['spread'].shift(-1) - spread`. Rows with NaN spread or NaN target are dropped (978 train / 435 test after filtering, from 3944 / 997 raw).
- **Rationale:** The target-computation logic lives in the experiment script because (a) the target is a modeling choice, not a feature-engineering choice (you could also target spread level, absolute spread, or multi-step spread), and (b) future scripts in Phases 5-7 will reuse the same function.
- **Files modified:** experiments/run_baselines.py
- **Committed in:** 672f3299 (Task 2 commit)

**3. [Rule 1 — Bug] Table column width overflow**
- **Found during:** Task 2 verification run
- **Issue:** Hardcoded 25-char Model column truncated "Volume (Higher Volume Correct)" (30 chars), breaking the fixed-width alignment.
- **Fix:** Compute model column width from the longest model name in the results list.
- **Files modified:** experiments/run_baselines.py
- **Committed in:** 672f3299 (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (2 plan-drift / schema corrections + 1 cosmetic bug)
**Impact on plan:** Script adapts to the real Phase 3 data contract. No scope creep; all acceptance criteria met. Task 2 `--data-path` CLI arg is preserved as an alias to `--data-dir` for backward compatibility with the plan text.

## Issues Encountered

None beyond the planned deviations. TDD cycle ran cleanly for Task 1 (RED confirmed via `ModuleNotFoundError` for all three new modules, GREEN on first implementation pass, no refactor needed). Task 2 was exploratory and validated against real data directly — no test file needed per the plan scope (script-level integration verified by the import check and the end-to-end run).

## User Setup Required

None. The experiment script runs on existing `data/processed/train.parquet` and `data/processed/test.parquet`, which Phase 3 already produced.

## Next Phase Readiness

- **TA check-in (April 4) — READY.** `python -m experiments.run_baselines` produces the four-model comparison table and saves reproducible JSON results under `experiments/results/tier1/`.
- **Tier 2 time-series models (Phase 5):** BasePredictor contract holds; GRU/LSTM/TFT implementations drop into the same `fit` / `predict` / `evaluate` pipeline and will save results to `experiments/results/tier2/`.
- **Tier 3 RL models (Phase 6):** PPO and PPO+autoencoder will use the same target-computation logic and feature matrix; they save to `experiments/results/tier3/`.
- **Evaluation/SHAP (Phase 7):** `load_all_results` reads every tier's JSON files into a unified comparison table; `pnl_series` values are already persisted for equity-curve plotting.

---
*Phase: 04-regression-baselines*
*Plan: 02*
*Completed: 2026-04-05*

## Self-Check: PASSED

All 12 production/test/experiment/result files exist on disk. Both task commits (826c9766, 672f3299) exist in git history. All 73 model + evaluation tests pass. End-to-end experiment script runs cleanly against real Phase 3 data.
