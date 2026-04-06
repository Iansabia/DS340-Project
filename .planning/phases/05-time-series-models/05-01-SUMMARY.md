---
phase: 05-time-series-models
plan: 01
subsystem: models
tags: [pytorch, numpy, sklearn, standardscaler, windowing, early-stopping, tdd]

# Dependency graph
requires:
  - phase: 04-regression-baselines
    provides: BasePredictor contract, evaluation framework, test patterns
provides:
  - "Shared sequence utilities: create_sequences, EarlyStopping, set_seed, get_device, fit_feature_scaler, apply_feature_scaler"
  - "TDD test suite with 15 tests covering windowing, early stopping, seeding, device selection, and feature scaling"
affects: [05-02-PLAN (GRU), 05-03-PLAN (LSTM), 05-04-PLAN (harness)]

# Tech tracking
tech-stack:
  added: []
  patterns: [sliding-window with group-boundary respect, zero-variance guard pattern, bool-to-float pre-scaling]

key-files:
  created:
    - src/models/sequence_utils.py
    - tests/models/test_sequence_utils.py
  modified: []

key-decisions:
  - "EarlyStopping compares against best_loss (not previous loss) for min_delta threshold"
  - "create_sequences preserves first-occurrence group order via OrderedDict"
  - "fit_feature_scaler raises ValueError listing zero-variance column names to surface upstream bugs"

patterns-established:
  - "Zero-variance guard: fit_feature_scaler rejects columns with std==0 via ValueError before StandardScaler.fit"
  - "Group-boundary windowing: create_sequences loops unique group_ids, only emits windows within a single group"
  - "Bool pre-casting: bool columns cast to float {0.0, 1.0} before scaler fit/transform"

requirements-completed: [MOD-05, MOD-06]

# Metrics
duration: 3min
completed: 2026-04-06
---

# Phase 5 Plan 01: Sequence Utilities Summary

**Shared sequence model utilities (windowing, early stopping, seed, device, StandardScaler) with TDD -- 6 exports, 15 tests, zero-variance guard**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-06T00:01:16Z
- **Completed:** 2026-04-06T00:05:09Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Created `sequence_utils.py` with 6 exports consumed by GRU (05-02) and LSTM (05-03)
- Wrote 15 TDD tests covering all Validation Dimensions 1 and 3 from 05-RESEARCH.md
- Zero-variance guard in `fit_feature_scaler` surfaces upstream feature-engineering bugs loudly via ValueError

## Task Commits

Each task was committed atomically:

1. **Task 1: Write failing tests for sequence_utils (RED)** - `e9cd612f` (test)
2. **Task 2: Implement sequence_utils.py (GREEN)** - `62b3dbbf` (feat)

## Exports and Signatures

```python
def create_sequences(X, y, lookback, group_ids) -> tuple[np.ndarray, np.ndarray]
class EarlyStopping(patience=10, min_delta=1e-4)  # .step(val_loss) -> bool
def set_seed(seed: int) -> None
def get_device() -> torch.device
def fit_feature_scaler(X: pd.DataFrame, bool_cols: list[str]) -> StandardScaler
def apply_feature_scaler(X: pd.DataFrame, scaler, bool_cols: list[str]) -> np.ndarray
```

## Test Coverage

| Class | Tests | Covers |
|-------|-------|--------|
| TestCreateSequences | 4 | Shape, group boundaries, short-pair skip, target alignment |
| TestEarlyStopping | 3 | Patience trigger, reset on improvement, initial step |
| TestSetSeed | 2 | Numpy reproducibility, Torch reproducibility |
| TestGetDevice | 2 | Returns torch.device, type is cpu or cuda |
| TestFeatureScaler | 4 | Bool casting, zero-mean/unit-std, no refit on test, zero-variance error |
| **Total** | **15** | |

## Zero-Variance Guard

`fit_feature_scaler` computes column-wise std before fitting. If any column has std==0, it raises:
```
ValueError: fit_feature_scaler: zero-variance columns detected (would produce NaN after scaling): ['column_name']
```
This prevents silent NaN poisoning downstream -- upstream bugs are surfaced with the specific offending column names.

## Files Created/Modified
- `src/models/sequence_utils.py` - Shared utilities (194 lines): windowing, early stopping, seed, device, scaler
- `tests/models/test_sequence_utils.py` - TDD test suite (270 lines): 15 tests across 5 classes

## Decisions Made
- EarlyStopping compares against best_loss (not previous loss) for min_delta -- standard PyTorch convention
- create_sequences uses OrderedDict to preserve first-occurrence group order (deterministic output)
- Empty-window edge case returns correctly-shaped empty arrays (0, lookback, n_features)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed EarlyStopping test values for best_loss comparison**
- **Found during:** Task 2 (GREEN implementation)
- **Issue:** Test losses [0.5, 0.49, 0.488, ...] assumed comparison against previous loss, but EarlyStopping correctly compares against best_loss. Value 0.488 WAS a meaningful improvement from best_loss=0.5 (delta=0.012 > min_delta=0.01), so the counter never accumulated.
- **Fix:** Updated test values to correctly model no-improvement scenarios: [0.5, 0.499, 0.498, 0.497] for trigger test, [0.5, 0.3, 0.31, 0.1, 0.11, 0.12] for reset test.
- **Files modified:** tests/models/test_sequence_utils.py
- **Verification:** All 15 tests pass
- **Committed in:** 62b3dbbf (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug in test values)
**Impact on plan:** Test logic corrected to match standard EarlyStopping semantics. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `sequence_utils.py` is ready for import by 05-02 (GRU) and 05-03 (LSTM) in Wave 2
- All 6 exports have full test coverage
- No CONTEXT.md decisions D2 or D5 were violated

## Self-Check: PASSED

- [x] src/models/sequence_utils.py exists
- [x] tests/models/test_sequence_utils.py exists
- [x] 05-01-SUMMARY.md exists
- [x] Commit e9cd612f exists (Task 1 RED)
- [x] Commit 62b3dbbf exists (Task 2 GREEN)

---
*Phase: 05-time-series-models*
*Completed: 2026-04-06*
