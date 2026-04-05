---
phase: 03-feature-engineering
plan: 01
subsystem: features
tags: [pandas, pytorch-forecasting, feature-engineering, time-series, parquet]

# Dependency graph
requires:
  - phase: 02.1-trade-based-data-reconstruction
    provides: aligned_pairs.parquet with 31 columns (4-hour VWAP bars, microstructure features)
provides:
  - compute_derived_features() for 6 microstructure-derived features
  - temporal_train_test_split() with per-pair chronological boundary
  - add_timeseries_columns() for PyTorch Forecasting time_idx/group_id
  - build_timeseries_dataset() wrapping pytorch_forecasting.TimeSeriesDataSet
  - train.parquet and test.parquet (39 columns, 3944/997 rows)
affects: [04-evaluation-framework, 05-regression-baselines, 06-time-series-models, 07-rl-models]

# Tech tracking
tech-stack:
  added: [pytorch-forecasting]
  patterns: [per-pair groupby for cross-pair isolation, per-pair chronological split, NaN-safe derived features]

key-files:
  created:
    - src/features/dataset.py
    - tests/features/test_dataset.py
    - tests/features/test_build_features.py
  modified:
    - src/features/schemas.py
    - src/features/engineering.py
    - src/features/build_features.py
    - tests/features/conftest.py
    - tests/features/test_engineering.py

key-decisions:
  - "Per-pair chronological split (not global cutoff) because pairs have different time ranges"
  - "NaN fill with 0 for TimeSeriesDataSet compatibility (pytorch_forecasting requires no NaN in feature cols)"
  - "group_id mapping fitted on train set and applied to test for consistent encoding"

patterns-established:
  - "Per-pair groupby for all rolling/diff operations to prevent cross-pair data leakage"
  - "Volume ratio uses replace([inf, -inf], NaN) after division for safety"
  - "Order flow imbalance uses .where(denom != 0, NaN) to handle zero-denominator"

requirements-completed: [FEAT-01, FEAT-02, FEAT-03, FEAT-04, FEAT-05]

# Metrics
duration: 6min
completed: 2026-04-04
---

# Phase 3 Plan 1: Feature Engineering Summary

**6 derived microstructure features (velocity, volume ratio, momentum, volatility, order flow imbalance) computed per-pair with temporal train/test split and PyTorch Forecasting TimeSeriesDataSet compatibility**

## Performance

- **Duration:** 6 min
- **Started:** 2026-04-04T23:58:42Z
- **Completed:** 2026-04-05T00:05:13Z
- **Tasks:** 2
- **Files modified:** 10 (4 deleted, 3 created, 5 rewritten)

## Accomplishments
- Rewrote feature engineering pipeline for aligned_pairs format (31-col input, 37-col output with 6 derived features)
- Implemented per-pair temporal train/test split with no-leakage assertion: 3944 train rows, 997 test rows, 17 pairs
- Added PyTorch Forecasting compatibility: time_idx (0-based per pair), group_id (stable int encoding), TimeSeriesDataSet builder
- Deleted stale Phase 3 modules (pair_loader.py, alignment.py) that operated on obsolete hourly candlestick format
- 28 tests covering all features, splits, and pipeline integration

## Task Commits

Each task was committed atomically:

1. **Task 1: Derived feature computation and schema rewrite** - `77cc17e0` (feat)
2. **Task 2: Temporal split, PyTorch Forecasting format, and CLI pipeline** - `8e9fae98` (feat)

## Files Created/Modified
- `src/features/schemas.py` - ALIGNED_COLUMNS (31), DERIVED_FEATURE_COLUMNS (6), OUTPUT_COLUMNS (37)
- `src/features/engineering.py` - compute_derived_features() with per-pair groupby isolation
- `src/features/dataset.py` - temporal_train_test_split(), add_timeseries_columns(), build_timeseries_dataset()
- `src/features/build_features.py` - CLI pipeline: load aligned_pairs, compute features, split, save
- `tests/features/conftest.py` - aligned_pairs-shaped fixture (10 rows, 2 pairs, 31 columns)
- `tests/features/test_engineering.py` - 11 tests for derived features
- `tests/features/test_dataset.py` - 11 tests for split, timeseries cols, TimeSeriesDataSet
- `tests/features/test_build_features.py` - 6 integration tests for CLI pipeline
- `src/features/pair_loader.py` - DELETED (loaded from old accepted_pairs.json format)
- `src/features/alignment.py` - DELETED (hourly alignment, replaced by Phase 2.1 aligner)
- `tests/features/test_pair_loader.py` - DELETED
- `tests/features/test_alignment.py` - DELETED

## Decisions Made
- Per-pair chronological split (not global cutoff) because pairs have different time ranges -- each pair's first 80% of bars go to train
- NaN values in numeric feature columns filled with 0.0 for TimeSeriesDataSet compatibility (pytorch_forecasting does not accept NaN in feature columns; allow_missing_timesteps only handles missing rows)
- group_id mapping fitted on train set's sorted unique pair_ids and applied consistently to test set

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] NaN values in TimeSeriesDataSet feature columns**
- **Found during:** Task 2 (build_timeseries_dataset)
- **Issue:** pytorch_forecasting.TimeSeriesDataSet raises ValueError for NaN in feature columns; allow_missing_timesteps only handles missing rows, not missing cell values
- **Fix:** Added NaN-to-0 fill for numeric feature columns inside build_timeseries_dataset()
- **Files modified:** src/features/dataset.py
- **Verification:** TimeSeriesDataSet creates successfully with 869 samples from real train data
- **Committed in:** 8e9fae98 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 bug fix)
**Impact on plan:** Essential fix for PyTorch Forecasting compatibility. No scope creep.

## Issues Encountered
None beyond the auto-fixed NaN issue above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- train.parquet (3944 rows, 39 cols) and test.parquet (997 rows, 39 cols) ready in data/processed/
- 17 pairs with non-NaN spread data available for model training
- PyTorch Forecasting TimeSeriesDataSet builder tested and functional (869 encoder-decoder samples)
- All downstream phases (evaluation, regression baselines, time series models, RL) can consume these parquets directly
- Note: ~70% of spread values are NaN (bars where one or both platforms lack data). Models should handle or filter these as appropriate.

---
*Phase: 03-feature-engineering*
*Completed: 2026-04-04*
