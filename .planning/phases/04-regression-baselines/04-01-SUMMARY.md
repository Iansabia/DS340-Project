---
phase: 04-regression-baselines
plan: 01
subsystem: evaluation
tags: [baselines, metrics, profit-simulation, sharpe, directional-accuracy, abc]

# Dependency graph
requires:
  - phase: 03-feature-engineering
    provides: train.parquet/test.parquet with precomputed spread column and polymarket_volume columns
provides:
  - BasePredictor ABC contract (predict, name, fit, evaluate) for every model tier
  - compute_regression_metrics (RMSE, MAE, directional accuracy)
  - simulate_profit (threshold strategy, P&L, win rate, Sharpe, equity curve)
  - NaivePredictor (full-reversion lower bound)
  - VolumePredictor (volume-weighted reversion baseline)
affects: [04-regression-models, 05-time-series, 06-rl, 07-evaluation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "BasePredictor ABC: every model must implement predict(X)->ndarray and name property"
    - "Shared evaluate() method: every model produces the same regression + trading metrics dict"
    - "252 trading-day Sharpe annualization factor as evaluation constant"
    - "TDD RED->GREEN for every production module"

key-files:
  created:
    - src/models/base.py
    - src/models/naive.py
    - src/models/volume.py
    - src/evaluation/metrics.py
    - src/evaluation/profit_sim.py
    - tests/models/conftest.py
    - tests/models/test_naive.py
    - tests/models/test_volume.py
    - tests/evaluation/conftest.py
    - tests/evaluation/test_metrics.py
    - tests/evaluation/test_profit_sim.py
  modified: []

key-decisions:
  - "BasePredictor.evaluate returns a single merged dict (regression + trading metrics) so model comparison tables are one-line lookups"
  - "Directional accuracy excludes samples where y_true == 0 (no direction to predict); returns 0.0 when all truths are zero"
  - "VolumePredictor uses volume_ratio = max_vol / total_vol (range [0.5, 1.0]) so equal volumes produce half-reversion and dominance approaches full reversion"
  - "Zero total volume predicts zero change (avoids div-by-zero, no-signal case)"
  - "Used polymarket_volume (not poly_volume) to match actual Phase 3 matched-pairs schema"
  - "Sharpe is 0.0 when num_trades < 2 or return std is 0 (undefined ratio guarded explicitly)"

patterns-established:
  - "BasePredictor contract: abstract predict + name, concrete fit (no-op default) + evaluate"
  - "Required-column validation raises ValueError in predict() with explicit column name in message"
  - "TDD workflow: write failing tests -> verify RED -> implement minimal code -> verify GREEN"
  - "Test conftest fixtures mirror Phase 3 matched-pairs column schema (kalshi_vwap/polymarket_vwap/spread/kalshi_volume/polymarket_volume)"

requirements-completed: [EVAL-01, EVAL-02, MOD-03, MOD-04]

# Metrics
duration: 4min
completed: 2026-04-05
---

# Phase 4 Plan 1: Prediction Interface and Baseline Predictors Summary

**BasePredictor ABC + regression/trading evaluation framework + Naive and Volume baselines as the lower-bound reference for every future model.**

## Performance

- **Duration:** 4 min
- **Started:** 2026-04-05T14:05:13Z
- **Completed:** 2026-04-05T14:08:58Z
- **Tasks:** 2
- **Files created:** 11 (5 production, 6 test)

## Accomplishments

- BasePredictor ABC defines the universal predict/fit/evaluate contract every Tier 1/2/3 model inherits
- compute_regression_metrics returns RMSE, MAE, and directional accuracy in a uniform dict
- simulate_profit implements the threshold-based directional trading strategy used by all models, returning P&L, trade count, win rate, annualized Sharpe, and cumulative P&L series
- NaivePredictor (full spread reversion) and VolumePredictor (volume-weighted reversion) implement BasePredictor and drop into the shared evaluate() pipeline
- 44 unit tests all pass (24 evaluation, 20 models); end-to-end smoke test against real Phase 3 train.parquet produces sensible metrics for both baselines

## Task Commits

Each task was committed atomically (TDD tests + implementation together):

1. **Task 1: Prediction interface and evaluation framework** - `78874967` (feat)
2. **Task 2: Naive and Volume baseline predictors** - `f6826ffa` (feat)

**Plan metadata:** pending (this commit)

## Files Created/Modified

### Production code
- `src/models/__init__.py` - package init
- `src/models/base.py` - BasePredictor ABC with predict/name abstract and fit/evaluate concrete
- `src/models/naive.py` - NaivePredictor (predicts -spread, full reversion to zero)
- `src/models/volume.py` - VolumePredictor (predicts -spread * volume_ratio)
- `src/evaluation/__init__.py` - package init
- `src/evaluation/metrics.py` - compute_regression_metrics(y_true, y_pred) -> dict
- `src/evaluation/profit_sim.py` - simulate_profit(predictions, actuals, threshold) -> dict

### Tests
- `tests/models/__init__.py`, `tests/evaluation/__init__.py` - test package inits
- `tests/models/conftest.py` - sample_features / sample_targets fixtures matching Phase 3 schema
- `tests/models/test_naive.py` - 9 tests for NaivePredictor
- `tests/models/test_volume.py` - 11 tests for VolumePredictor
- `tests/evaluation/conftest.py` - sample_y_true / sample_y_pred fixtures
- `tests/evaluation/test_metrics.py` - 11 tests for compute_regression_metrics
- `tests/evaluation/test_profit_sim.py` - 13 tests for simulate_profit

## Decisions Made

- **Merged metrics dict from evaluate():** A single flat dict combining regression + trading metrics means future results tables are one-line lookups per model.
- **Zero-truth handling in directional accuracy:** Samples with `y_true == 0` carry no direction and are excluded; if all truths are zero the metric returns `0.0` explicitly.
- **Volume ratio formulation:** `max(k_vol, p_vol) / (k_vol + p_vol)` lies in `[0.5, 1.0]`, so the volume baseline smoothly interpolates between half-reversion (equal volumes, no signal) and full reversion (one platform dominates). Zero total volume predicts zero change.
- **252-day Sharpe annualization:** Matches trading-day convention used in finance literature; `<2` trades or `std == 0` returns `0.0` explicitly to avoid undefined ratios.
- **polymarket_volume vs poly_volume (schema alignment):** See Deviations below.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug / Schema Mismatch] VolumePredictor uses polymarket_volume (not poly_volume)**
- **Found during:** Task 2 (VolumePredictor implementation)
- **Issue:** Plan text specified column names `poly_close` / `poly_volume`, but the actual Phase 3 matched-pairs parquet schema uses the full `polymarket_*` prefix (`polymarket_vwap`, `polymarket_volume`, `polymarket_close`, etc.) as established in Phase 2.1 / Phase 3. The `<important_context>` prompt explicitly flags this. Using `poly_volume` would make VolumePredictor incompatible with real Phase 3 data.
- **Fix:** Implemented VolumePredictor using `polymarket_volume`, tests and fixtures use `polymarket_volume`. Smoke test against real `data/processed/train.parquet` passes.
- **Files modified:** src/models/volume.py, tests/models/conftest.py, tests/models/test_volume.py
- **Verification:** `pytest tests/models/ tests/evaluation/` -> 44 passed; smoke test against train.parquet produces num_trades=888, sharpe=6.36 for VolumePredictor
- **Committed in:** f6826ffa (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 schema mismatch / bug)
**Impact on plan:** Minimal; corrects stale column naming in plan text to match the current Phase 3 data contract. No scope creep.

## Issues Encountered

None - TDD cycle ran cleanly: RED confirmed for both tasks via ModuleNotFoundError, GREEN achieved on first implementation pass, no refactor needed.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- BasePredictor contract is ready for Phase 4 Plan 2 (Linear Regression + XGBoost) to implement
- Evaluation pipeline works end-to-end on real Phase 3 data (verified with smoke test: 1001 rows, ~900 trades per baseline)
- Naive baseline produces ~0.212 RMSE, 0.63 directional accuracy on train set - establishes the performance floor every future model must beat
- Volume baseline produces ~0.192 RMSE, marginally outperforming naive - confirms the volume signal carries information

---
*Phase: 04-regression-baselines*
*Plan: 01*
*Completed: 2026-04-05*

## Self-Check: PASSED

All 11 production/test files exist. Both task commits (78874967, f6826ffa) exist in git history. All 44 tests pass.
