---
phase: 05-time-series-models
plan: 04
subsystem: experiments
tags: [gru, lstm, xgboost, comparison-table, tier2, multi-seed, cross-tier]

# Dependency graph
requires:
  - phase: 05-02
    provides: GRUPredictor model class
  - phase: 05-03
    provides: LSTMPredictor model class
  - phase: 04-01
    provides: run_baselines.py harness, Tier 1 result JSONs, BasePredictor contract
provides:
  - "--tier {1,2,both}" CLI flag on run_baselines.py
  - Cross-tier comparison table (6 models, unified)
  - tier2/gru.json and tier2/lstm.json with 3-seed aggregation
  - Tier 1 results regenerated at consistent 31-feature set
  - GRU/LSTM exports from src.models package
affects: [phase-07-experiments, phase-05-05-verification]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "prepare_xy_for_seq: group_id pass-through for Tier 2 windowing"
    - "run_tier2_with_seeds: 3-seed training with mean/std aggregation"
    - "torch.set_num_threads(1) workaround for Apple Silicon GRU segfault"
    - "torch.from_numpy instead of torch.tensor for large array conversion"

key-files:
  created:
    - experiments/results/tier2/gru.json
    - experiments/results/tier2/lstm.json
    - experiments/results/tier2/.gitkeep
    - tests/experiments/test_run_baselines_tier2.py
    - tests/experiments/__init__.py
    - .planning/phases/05-time-series-models/05-04-comparison-output.txt
  modified:
    - experiments/run_baselines.py
    - src/models/__init__.py
    - src/models/gru.py
    - src/models/lstm.py
    - src/models/sequence_utils.py
    - experiments/results/tier1/linear_regression.json
    - experiments/results/tier1/xgboost.json
    - experiments/results/tier1/naive_spread_closes.json
    - experiments/results/tier1/volume_higher_volume_correct.json

key-decisions:
  - "Dropped 4 zero-variance columns (not just 1) -- kalshi_order_flow_imbalance, kalshi_buy_volume, kalshi_sell_volume, kalshi_realized_spread all 100% zero in training data, yielding 31 features instead of CONTEXT.md's projected 34"
  - "Added torch.set_num_threads(1) in set_seed() to avoid PyTorch 2.10.0 segfault on Apple Silicon multi-threaded GRU/LSTM"
  - "Used torch.from_numpy with float32 contiguous arrays instead of torch.tensor for large sequence conversion"

patterns-established:
  - "Pattern: Tier 2 models receive X = df[feature_cols + ['group_id']] via prepare_xy_for_seq"
  - "Pattern: Multi-seed training with run_tier2_with_seeds stores seed_rmses, mean_rmse, std_rmse in extra field"
  - "Pattern: format_comparison_table(results, tier) handles all tier display modes"

requirements-completed: [MOD-05, MOD-06]

# Metrics
duration: 40min
completed: 2026-04-06
---

# Phase 5 Plan 04: Cross-Tier Comparison Harness Summary

**Extended run_baselines.py with --tier {1,2,both} CLI, trained GRU (RMSE=0.2896+/-0.0024) and LSTM (RMSE=0.2910+/-0.0004) over 3 seeds, produced unified 6-model comparison table at 31 features**

## Performance

- **Duration:** 40 min (most time spent debugging PyTorch Apple Silicon segfault)
- **Started:** 2026-04-06T00:15:11Z
- **Completed:** 2026-04-06T00:55:25Z
- **Tasks:** 3
- **Files modified:** 15

## Accomplishments
- Cross-tier comparison table with all 6 models (Naive, Volume, Linear Regression, XGBoost, GRU, LSTM) at consistent 31-feature set
- GRU RMSE=0.2896 mean +/- 0.0024 std over seeds {42, 123, 456}; LSTM RMSE=0.2910 +/- 0.0004
- Both Tier 2 models are competitive with XGBoost (0.2857) but do not beat it -- validates the complexity-vs-performance thesis
- Fixed PyTorch segfault on Apple Silicon that was blocking all Tier 2 training

## Cross-Tier Results

```
====== Cross-Tier Comparison ======

  Sharpe: annualized time-series Sharpe (4h bars, 24/7, sqrt(2190))
  Raw SR: unannualized per-trade mean/std (raw trade quality)

Model                          |    RMSE |     MAE | Dir Acc |      P&L | Trades | Win Rate |  Sharpe |  Raw SR
-------------------------------+---------+---------+---------+----------+--------+----------+---------+--------
Naive (Spread Closes)          |  0.4995 |  0.3806 |  0.5333 |  58.1205 |   1460 |   0.4795 |  5.4714 |  0.1253
Volume (Higher Volume Correct) |  0.4566 |  0.3449 |  0.5333 |  59.8120 |   1440 |   0.4806 |  5.6620 |  0.1306
Linear Regression              |  0.3081 |  0.2253 |  0.6594 | 230.1417 |   1542 |   0.5733 | 22.0102 |  0.4946
XGBoost                        |  0.2857 |  0.2216 |  0.6776 | 238.4071 |   1538 |   0.5819 | 23.1494 |  0.5216
GRU                            |  0.2928 |  0.2229 |  0.6433 | 212.5027 |   1517 |   0.5583 | 20.2405 |  0.4586
LSTM                           |  0.2915 |  0.2239 |  0.6545 | 221.8397 |   1547 |   0.5650 | 21.1169 |  0.4732
```

### Key Findings

1. **Tier 1 RMSE delta from dropping zero-variance columns:** 0.0000 (columns were all-zero, no signal)
2. **GRU RMSE (0.2896) vs CONTEXT.md predicted band (0.29-0.33):** Within band, near low end
3. **LSTM RMSE (0.2910) vs CONTEXT.md predicted band (0.30-0.34):** Within/below band (better than expected)
4. **Do Tier 2 models beat XGBoost?** No -- XGBoost (0.2857) remains best. GRU is 1.4% worse, LSTM is 1.9% worse. However both beat Linear Regression (0.3081) convincingly.
5. **Honest assessment:** The temporal structure captured by GRU/LSTM provides marginal benefit over XGBoost's feature engineering at this dataset scale (~6k training windows). This directly supports the project's complexity-vs-performance thesis: XGBoost is the sweet spot for 144 pairs.

### Feature Count Discrepancy

CONTEXT.md projected 34 features after dropping `kalshi_order_flow_imbalance`. Actual count is 31 because 3 additional Kalshi columns (`kalshi_buy_volume`, `kalshi_sell_volume`, `kalshi_realized_spread`) were also 100% zero in the training data and triggered the zero-variance safety guard in `fit_feature_scaler`. These were silently passed through in Tier 1 (no scaler) but blocked Tier 2 training. Adding all 4 zero-variance columns to `NON_FEATURE_COLUMNS` ensures both tiers use the identical feature set.

## Task Commits

Each task was committed atomically:

1. **Task 1: Integration tests (RED)** - `e49f55ef` (test)
2. **Task 2: Extend run_baselines.py + exports (GREEN)** - `2cbfa581` (feat)
3. **Task 3: End-to-end training + results** - `e8609fe8` (feat)

**Plan metadata:** (pending)

## Files Created/Modified
- `experiments/run_baselines.py` - Extended with --tier {1,2,both}, prepare_xy_for_seq, run_tier2_with_seeds, updated NON_FEATURE_COLUMNS
- `src/models/__init__.py` - Added GRUPredictor, LSTMPredictor exports
- `src/models/gru.py` - Fixed tensor conversion (from_numpy instead of torch.tensor)
- `src/models/lstm.py` - Fixed tensor conversion (from_numpy instead of torch.tensor)
- `src/models/sequence_utils.py` - Fixed set_seed() with set_num_threads(1) for Apple Silicon
- `experiments/results/tier2/gru.json` - GRU results with 3-seed aggregation
- `experiments/results/tier2/lstm.json` - LSTM results with 3-seed aggregation
- `experiments/results/tier1/*.json` - Regenerated at 31 features
- `tests/experiments/test_run_baselines_tier2.py` - 18 integration tests

## Decisions Made
- Dropped 4 zero-variance columns (not 1 as planned) because 3 additional Kalshi columns were also all-zero, yielding 31 features instead of 34
- Applied `torch.set_num_threads(1)` workaround for PyTorch 2.10.0 segfault on Apple Silicon -- affects only CPU training path
- Used `torch.from_numpy` with explicit float32 contiguous arrays instead of `torch.tensor` for robust tensor creation

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] PyTorch 2.10.0 segfault on Apple Silicon multi-threaded GRU/LSTM**
- **Found during:** Task 3 (end-to-end training)
- **Issue:** `GRUPredictor.fit()` triggered SIGSEGV (exit code 139) when PyTorch used multi-threaded OpenMP/Accelerate backend for GRU forward pass on Apple Silicon
- **Fix:** Added `torch.set_num_threads(1)` in `set_seed()` when no CUDA is available
- **Files modified:** src/models/sequence_utils.py
- **Verification:** Full 3-seed training completes without crash

**2. [Rule 1 - Bug] torch.tensor segfault on large float64 numpy arrays**
- **Found during:** Task 3 (end-to-end training)
- **Issue:** `torch.tensor(X_seq, dtype=torch.float32)` with float64 numpy arrays could segfault on some platforms
- **Fix:** Changed to `torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))` in both gru.py and lstm.py
- **Files modified:** src/models/gru.py, src/models/lstm.py
- **Verification:** Tensor creation succeeds reliably

**3. [Rule 3 - Blocking] Three additional zero-variance columns blocked Tier 2 scaler**
- **Found during:** Task 3 (end-to-end training)
- **Issue:** `fit_feature_scaler` raised ValueError for `kalshi_buy_volume`, `kalshi_sell_volume`, `kalshi_realized_spread` (all-zero, upstream Phase 3 data issue)
- **Fix:** Added all 3 to NON_FEATURE_COLUMNS alongside kalshi_order_flow_imbalance, yielding 31 features
- **Files modified:** experiments/run_baselines.py
- **Verification:** Both tiers train successfully with 31 features; Tier 1 RMSE unchanged

---

**Total deviations:** 3 auto-fixed (2 bugs, 1 blocking)
**Impact on plan:** Feature count is 31 not 34 (documented). All fixes necessary for training to complete. No scope creep.

## Issues Encountered
- PyTorch Apple Silicon segfault consumed ~25 minutes of debugging time; root cause was multi-threaded Accelerate backend incompatibility with GRU/LSTM forward pass in PyTorch 2.10.0

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 5 Plan 05 (verification) can now run: all 6 model results exist, comparison table renders
- Phase 6 (RL models) can proceed: Tier 1 + Tier 2 baselines established as comparison targets
- Phase 7 (experiments framework): cross-tier comparison table is ready for integration

---
*Phase: 05-time-series-models*
*Completed: 2026-04-06*

## Self-Check: PASSED

All 7 created files verified present. All 3 task commits (e49f55ef, 2cbfa581, e8609fe8) verified in git log.
