---
phase: 05-time-series-models
plan: 02
subsystem: models
tags: [gru, pytorch, recurrent, time-series, tdd]

# Dependency graph
requires:
  - phase: 05-01
    provides: "Shared sequence utilities (create_sequences, EarlyStopping, set_seed, get_device, fit_feature_scaler, apply_feature_scaler)"
  - phase: 04
    provides: "BasePredictor contract, evaluation pipeline (metrics + profit_sim)"
provides:
  - "GRUPredictor class -- Tier 2 recurrent baseline for spread-change prediction"
  - "TDD test suite with 14 tests covering interface, shape, reproducibility, group_id contract, padding"
affects: [05-03, 05-04, 05-05, 07]

# Tech tracking
tech-stack:
  added: []
  patterns: ["Warm-up stitching for row-aligned sequence predictions", "Internal windowing with cached train data", "group_id guard pattern for Tier 2 models"]

key-files:
  created:
    - src/models/gru.py
    - tests/models/test_gru.py
  modified: []

key-decisions:
  - "GRU uses input dropout (nn.Dropout on features) not recurrent dropout, per CONTEXT.md D6"
  - "90/10 within-pair chronological val split for early stopping (last 10% of each group)"
  - "Warm-up stitching prepends cached train rows per group_id for full lookback windows on test data"
  - "Padding by repeating first row when total available rows < lookback, logged to _padded_pairs"

patterns-established:
  - "Tier 2 model pattern: inherit BasePredictor, require group_id, cache train data, warm-up stitch in predict()"
  - "group_id guard: raise ValueError immediately in fit/predict if group_id column missing"

requirements-completed: [MOD-05]

# Metrics
duration: 3min
completed: 2026-04-06
---

# Phase 5 Plan 2: GRU Predictor Summary

**Single-layer GRU (hidden=64) with warm-up stitching, group_id boundary enforcement, and 14-test TDD suite -- Tier 2 recurrent baseline ready for harness integration**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-06T00:08:55Z
- **Completed:** 2026-04-06T00:12:14Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- GRUPredictor implements full BasePredictor contract with all hyperparameters matching CONTEXT.md D6 + D8
- Warm-up stitching returns one prediction per input row (row-aligned with Tier 1 for direct comparison)
- group_id guards on fit() and predict() prevent silent KeyError deep in training
- Padding events logged to _padded_pairs for auditability
- 14 TDD tests all pass covering interface, shape, reproducibility, evaluate keys, group boundaries, padding

## Task Commits

Each task was committed atomically:

1. **Task 1: Write failing tests for GRUPredictor (RED)** - `42820f8e` (test)
2. **Task 2: Implement GRUPredictor (GREEN)** - `aa67ee79` (feat)

## Files Created/Modified
- `src/models/gru.py` - GRUPredictor class (Tier 2 recurrent baseline, 290 lines)
- `tests/models/test_gru.py` - 14-test TDD suite across 4 test classes

## Decisions Made
- Followed CONTEXT.md D6 + D8 exactly: hidden_size=64, num_layers=1, dropout=0.3, AdamW lr=1e-3, weight_decay=1e-4, batch_size=64, max_epochs=100, patience=10, lookback=6
- Input dropout applied via nn.Dropout on features before GRU (not recurrent dropout)
- _GRUModule is an internal class (not exported), encapsulating the PyTorch nn.Module
- Training loop is manual (no PyTorch Lightning dependency)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Training Behavior (on synthetic test data)
- 40-row synthetic dataset with 2 groups of 20 rows each, 3 features
- With max_epochs=2 and hidden_size=8: trains in <1 second
- With max_epochs=3 and hidden_size=8: seed reproducibility confirmed (identical predictions to decimal=4)
- Short groups (fewer rows than lookback=6) are gracefully skipped during training
- Novel group_ids with fewer rows than lookback trigger padding and _padded_pairs logging

## Next Phase Readiness
- GRUPredictor ready for LSTM (05-03) to follow same pattern with hidden_size=32
- Ready for harness integration (05-04) via run_baselines.py --tier 2
- evaluate() returns Tier-1-compatible keys for cross-tier comparison table

## Self-Check: PASSED

All files created and all commits verified.

---
*Phase: 05-time-series-models*
*Completed: 2026-04-06*
