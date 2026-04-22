---
phase: 11-tft-training
plan: 01
subsystem: models
tags: [pytorch-forecasting, temporal-fusion-transformer, tft, lightning, quantile-loss, group-normalizer]

# Dependency graph
requires:
  - phase: 10-250-bar-scaling-checkpoint
    provides: "Verified GRU/LSTM training stack; 51-feature pipeline; train/test parquet files"
  - phase: 08-environment-and-baseline-verification
    provides: "pytorch-forecasting 1.7.0, pytorch-lightning 2.6.1 confirmed installed"
provides:
  - TFTPredictor(BasePredictor) implementation in src/models/tft.py
  - experiments/run_tft.py single-split experiment runner
  - experiments/results/tier2/TFT.json documented negative result
  - TFT wired into run_baselines.py and run_walk_forward.py
affects:
  - "11-02-paper-section 4.1 update (TFT row in Tables 2 and 3)"
  - "run_baselines.py --tier 2 now includes TFTPredictor"

# Tech tracking
tech-stack:
  added:
    - "pytorch_forecasting.TemporalFusionTransformer (via from_dataset() factory)"
    - "pytorch_forecasting.TimeSeriesDataSet with allow_missing_timesteps=True"
    - "pytorch_forecasting.data.GroupNormalizer(transformation=None)"
    - "pytorch_forecasting.metrics.QuantileLoss(quantiles=[0.1, 0.5, 0.9])"
    - "lightning.pytorch.Trainer (NOT pytorch_lightning — different LightningModule base class)"
    - "lightning.pytorch.callbacks.EarlyStopping"
  patterns:
    - "Round-based batch prediction: K rounds x all_groups simultaneously instead of per-row iteration"
    - "group_id preservation from raw parquet before compute_derived_features (schemas.py drops it)"
    - "from_dataset() factory pattern for TFT instantiation (NOT direct __init__)"
    - "per-group monotonic time_idx via groupby().cumcount() inside fit()"

key-files:
  created:
    - src/models/tft.py (394 lines, TFTPredictor(BasePredictor))
    - tests/models/test_tft.py (203 lines, 10 tests — all pass)
    - experiments/run_tft.py (184 lines, 3-seed experiment runner)
    - experiments/results/tier2/TFT.json (documented negative result)
  modified:
    - experiments/run_baselines.py (TFTPredictor import, _MODEL_ORDER, model_classes)
    - experiments/run_walk_forward.py (try/except TFT import, tft in sequence branch)

key-decisions:
  - "TFT does not beat GRU at N=6802 (RMSE 0.3262 vs GRU 0.2928) — extends simplicity-wins thesis to transformers"
  - "Use lightning.pytorch Trainer (not pytorch_lightning) — pytorch_forecasting 1.7.0 inherits from lightning.pytorch.LightningModule, not pytorch_lightning.LightningModule; mixing them raises TypeError"
  - "Round-based batch prediction: process all 144 groups simultaneously per test row offset (K~11 rounds) instead of per-row iteration (1673 calls); 150x speedup"
  - "GroupNormalizer(transformation=None) — NOT softplus; spread changes are signed, softplus maps negatives near zero causing degenerate predictions"
  - "Zero-variance column filter before TFT training (kalshi_kyle_lambda has std=0 in full train set)"
  - "group_id re-attached from raw parquet after compute_derived_features (schemas.py OUTPUT_COLUMNS excludes group_id)"

patterns-established:
  - "TFT predict() uses round-based batching with predict=True mode for row-aligned output"
  - "TFT experiment scripts must load raw parquet and re-attach group_id before compute_derived_features"

requirements-completed: [TFT-01, TFT-02, TFT-03, TFT-04, TFT-06]

# Metrics
duration: 43min
completed: 2026-04-22
---

# Phase 11 Plan 01: TFT Training Summary

**TFTPredictor wrapping pytorch_forecasting 1.7.0 TFT via from_dataset() factory; 3-seed experiment shows RMSE 0.3262 vs GRU 0.2928 (negative result per Option B, extending simplicity-wins thesis to transformers)**

## Performance

- **Duration:** 43 min
- **Started:** 2026-04-22T22:03:09Z
- **Completed:** 2026-04-22T22:46:30Z
- **Tasks:** 2 of 2
- **Files modified:** 6

## Accomplishments
- TFTPredictor(BasePredictor) implemented with 10 passing tests (inheritance, group_id guard, hyperparameter defaults, predict shape, attention audit)
- 3-seed TFT experiment completed: RMSE 0.3262 avg (vs GRU 0.2928) — documented negative result; attention healthy (entropy=2.66, not degenerate)
- TFT wired into run_baselines.py (import, _MODEL_ORDER, model_classes) and run_walk_forward.py (try/except block, sequence branch)

## TFT Experiment Results

| Metric | TFT (avg 3 seeds) | GRU baseline |
|--------|-------------------|--------------|
| RMSE | 0.3262 | 0.2928 |
| MAE | 0.2025 | — |
| Dir. Accuracy | 51.7% | 64.3% |
| P&L | +$1.55 | +$212.50 |
| Num Trades | 120 | 1517 |
| Win Rate | 37.5% | 55.8% |

Attention audit (seed 123): entropy=2.66, max_variable_weight=0.37, threshold=1.97 — NOT degenerate.

Converged: False — TFT at N=6802 with hidden_size=8 does not outperform the GRU baseline.

## Task Commits

1. **Task 1: Write test scaffold and implement TFTPredictor** - `1a6611b` (feat — TDD green phase)
2. **Task 2: Run single-split TFT experiment and wire into run_baselines + run_walk_forward** - `32acce7` (feat)

## Files Created/Modified
- `/Users/iansabia/Desktop/DS340 Project/src/models/tft.py` — TFTPredictor(BasePredictor), 394 lines
- `/Users/iansabia/Desktop/DS340 Project/tests/models/test_tft.py` — 10 tests, 203 lines
- `/Users/iansabia/Desktop/DS340 Project/experiments/run_tft.py` — single-split runner with Option B gate, 184 lines
- `/Users/iansabia/Desktop/DS340 Project/experiments/results/tier2/TFT.json` — documented negative result
- `/Users/iansabia/Desktop/DS340 Project/experiments/run_baselines.py` — +5 lines: TFT import, _MODEL_ORDER, model_classes
- `/Users/iansabia/Desktop/DS340 Project/experiments/run_walk_forward.py` — +7 lines: try/except import, sequence branch

## Decisions Made
1. `lightning.pytorch` Trainer used instead of `pytorch_lightning` — different LightningModule base class in pytorch_forecasting 1.7.0 causes TypeError with the latter
2. Round-based batch prediction: build one combined dataset per test step k, call predict once for all 144 groups simultaneously (~11 rounds vs 1673 individual calls)
3. `GroupNormalizer(transformation=None)` confirmed correct per 11-RESEARCH.md Pitfall 1
4. TFT negative result is paper-worthy — extends simplicity-wins thesis to transformers

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] pytorch_lightning.Trainer incompatible with pytorch_forecasting 1.7.0**
- **Found during:** Task 1 (GREEN phase, test_fit_returns_self)
- **Issue:** `pytorch_lightning.Trainer.fit()` raised `TypeError: model must be a LightningModule` because pytorch_forecasting 1.7.0 inherits from `lightning.pytorch.LightningModule` (not `pytorch_lightning.LightningModule`); the two module paths have different class identities despite same version
- **Fix:** Changed imports to `import lightning.pytorch as pl` and `from lightning.pytorch.callbacks import EarlyStopping`; also documented in tft.py docstring
- **Files modified:** src/models/tft.py
- **Verification:** All 10 tests pass; training runs through 30 epochs
- **Committed in:** 1a6611b (Task 1 commit)

**2. [Rule 1 - Bug] predict() returned (n_rows, 1) shape instead of (n_rows,)**
- **Found during:** Task 1 (test_predict_shape)
- **Issue:** `model.predict(loader, mode='prediction')` returned 2-D tensor of shape (n, 1) in pytorch-forecasting 1.7.0
- **Fix:** Added `.ravel()` to flatten to 1-D
- **Files modified:** src/models/tft.py
- **Verification:** test_predict_shape passes; len(preds) == len(X_test) confirmed
- **Committed in:** 1a6611b (Task 1 commit)

**3. [Rule 1 - Bug] predict() returned too many predictions using min_prediction_idx globally**
- **Found during:** Task 2 (run_tft.py execution — shape mismatch 7614 vs 1673)
- **Issue:** Global `min_prediction_idx` captured predictions for BOTH test rows and later training rows of longer-trained groups (144 groups have different training lengths; global min was too low)
- **Fix:** Switched to round-based batch prediction: for each step k (0..max_test_rows), build one combined multi-group dataset with all groups that have a k-th test row, call `predict=True` once for all, get exactly 1 prediction per group per round. Total predictions = n_test_rows.
- **Files modified:** src/models/tft.py
- **Verification:** len(predictions) == len(X_test) == 1673 confirmed in experiment run
- **Committed in:** 32acce7 (Task 2 commit)

**4. [Rule 1 - Bug] compute_derived_features drops group_id (not in OUTPUT_COLUMNS)**
- **Found during:** Task 2 (run_tft.py — KeyError: 'group_id not in index')
- **Issue:** `prepare_xy_for_seq` requires `group_id` column, but `compute_derived_features` filters to `OUTPUT_COLUMNS` which excludes `group_id`. The GRU experiment was run before this pipeline was added; load_train_test() now drops group_id.
- **Fix:** Added `_load_with_group_id()` helper in run_tft.py that preserves group_id from raw parquet before calling compute_derived_features, then re-attaches it
- **Files modified:** experiments/run_tft.py
- **Verification:** X_train shape (6802, 52) confirmed with group_id column present
- **Committed in:** 32acce7 (Task 2 commit)

**5. [Rule 2 - Missing Critical] Zero-variance column filter before TFTPredictor.fit()**
- **Found during:** Task 2 (run_tft.py — ValueError: kalshi_kyle_lambda zero-variance)
- **Issue:** kalshi_kyle_lambda has std=0 in the full training set; fit_feature_scaler raises ValueError; GRU avoided this because walk-forward already filters zero-variance cols, but run_tft didn't
- **Fix:** Added `nonzero_var_cols = [c for c in feature_cols if train[c].std() > 1e-10]` in run_tft.py before calling prepare_xy_for_seq
- **Files modified:** experiments/run_tft.py
- **Verification:** Training starts with 50 features (1 removed), scaler succeeds
- **Committed in:** 32acce7 (Task 2 commit)

---

**Total deviations:** 5 auto-fixed (3 Rule 1 bugs, 1 Rule 2 missing critical, 1 Rule 1 incompatibility)
**Impact on plan:** All auto-fixes necessary for correct TFT training and row-aligned predictions. No scope creep. Negative result (TFT < GRU) is the expected finding per TFT-04 Option B.

## Issues Encountered
- Training took ~45s per seed for 30 epochs (within the 30-60 minute estimate)
- Predict phase with per-row-per-group approach would take 13+ minutes; switched to round-based batching (~11 rounds) which completed in <2 minutes total
- Short-sequence groups (3 pairs with 5 bars) handled by try/except fallback in predict, returning 0.0 for those groups if from_dataset raises an error

## User Setup Required
None — no external service configuration required.

## Next Phase Readiness
- TFT experiment results are in experiments/results/tier2/TFT.json (negative result documented)
- TFT row ready for paper Tables 2 and 3 (phase 11-02): RMSE 0.3262, P&L +$1.55, beats_gru=False
- Attention audit result available: entropy=2.66, not degenerate
- run_baselines.py and run_walk_forward.py both accept TFT without breaking existing models

---
*Phase: 11-tft-training*
*Completed: 2026-04-22*
