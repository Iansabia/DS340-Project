---
phase: 13-ensemble-formalization
plan: 01
subsystem: models
tags: [ensemble, base-predictor, concordance, tdd, pickle]

# Dependency graph
requires:
  - phase: 08-environment-and-baseline-verification
    provides: BasePredictor pickle contract (save/load with isinstance guard)
  - phase: 08-environment-and-baseline-verification
    provides: src/utils/seed.set_all_seeds() reproducibility entry point
provides:
  - EnsemblePredictor(BasePredictor) class with weighted averaging and concordance gate
  - Reusable formalization of LR+XGB ensemble pattern (previously hard-coded in strategy.py)
  - 13-test TDD suite covering fit/predict/save/load/concordance behaviors
affects:
  - 13-02 (per-member feature routing; depends on EnsemblePredictor constructor contract)
  - 13-03 (weight sweep + paper §5.11; consumes EnsemblePredictor as the swept object)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Weight normalization inside predict() (not at call sites) — keeps ensemble math opaque to callers"
    - "Concordance gate as a string-mode param ('none' | 'strict') rather than boolean — extensible for future modes (e.g. 'majority')"
    - "Members fit internally by ensemble.fit() — no pre-fitted-model coupling"

key-files:
  created:
    - src/models/ensemble.py
    - tests/models/test_ensemble.py
    - .planning/phases/13-ensemble-formalization/deferred-items.md
  modified: []

key-decisions:
  - "Concordance 'strict' mode uses all-members-agree semantics (signs match element 0) rather than pairwise voting — matches strategy.py's binary LR/XGB behavior and trivially generalizes to N members"
  - "Weight normalization lives in predict(), not __init__() — lets callers reuse a single EnsemblePredictor instance across different weight semantics without surgery; also makes sweep runner trivial"
  - "set_all_seeds(self._seed) is called inside fit(), not __init__() — defers seeding until training starts so constructor stays pure"
  - "No custom __getstate__/__setstate__ — pickle works natively because all supported members (LR, XGB, GRU, LSTM, naive) are already picklable per Phase 8"

patterns-established:
  - "Stub-predictor pattern for testing concordance: _FixedSignPredictor returns a constant-signed vector, enabling deterministic sign-agreement assertions without fitting real models"
  - "TDD for composable BasePredictor subclasses — use LinearRegressionPredictor (small, deterministic, already tested) as concrete member rather than mocking"

requirements-completed: [ENSM-01, ENSM-05]

# Metrics
duration: 9min
completed: 2026-04-23
---

# Phase 13 Plan 01: EnsemblePredictor TDD Summary

**Picklable BasePredictor subclass that formalizes weighted-average ensembling with an optional sign-concordance gate — the reusable abstraction that Phase 13's weight sweep (§5.11) will consume.**

## Performance

- **Duration:** ~9 min
- **Started:** 2026-04-23T15:41:00Z
- **Completed:** 2026-04-23T15:50:51Z
- **Tasks:** 2 TDD commits (RED + GREEN; no REFACTOR needed)
- **Files created:** 3 (ensemble.py, test_ensemble.py, deferred-items.md)
- **Files modified:** 0 (ENSM-05 guard: strategy.py untouched)

## Accomplishments

- `EnsemblePredictor(BasePredictor)` class accepting arbitrary `list[BasePredictor]` members with optional per-member weights and a `concordance_mode` switch
- 13-test TDD suite (11 mandated behaviors + 2 sanity tests) all green on the GREEN commit; runs in 1.65s
- Constructor validates all four error modes: empty members, weight-length mismatch, negative weights, zero-sum weights, invalid concordance_mode
- Sign-concordance gate emits 0.0 in 'strict' mode when members disagree on sign, weighted average in 'none' mode regardless
- Save/load round-trip verified: `BasePredictor.load(path)` returns an EnsemblePredictor that passes isinstance(BasePredictor) and reproduces predictions exactly
- ENSM-05 safety verified: `git diff src/live/strategy.py` is empty

## Task Commits

Each TDD step was committed atomically:

1. **Task 1 (RED): failing test suite** — `4f8b01e` (test)
2. **Task 2 (GREEN): EnsemblePredictor implementation** — `7198b9a` (feat)

REFACTOR was not needed: the GREEN implementation already has complete module docstring, class docstring, per-method docstrings, input validation, and no dead code. All 13 tests pass without modification.

**Plan metadata:** pending (this summary + STATE.md + ROADMAP.md)

## Files Created/Modified

- `src/models/ensemble.py` — 149 lines. `EnsemblePredictor(BasePredictor)` with fit/predict/name, internal weight normalization, concordance gate, and full input validation.
- `tests/models/test_ensemble.py` — 218 lines, 13 tests. Uses shared `tiny_features`/`tiny_targets` fixtures local to the module plus a `_FixedSignPredictor` stub for deterministic sign-disagreement assertions.
- `.planning/phases/13-ensemble-formalization/deferred-items.md` — Logs 17 pre-existing, out-of-scope test failures observed during the full-suite smoke check.

## Decisions Made

- **All-members-agree concordance semantics.** 'strict' mode zeros a row iff `np.sign(preds)` is not constant across the member axis. Matches strategy.py's LR/XGB binary behavior and generalizes cleanly to N members without introducing a pairwise-voting parameter the paper would have to explain.
- **Normalize weights in predict(), not __init__().** Callers can mutate `_weights` externally (e.g. during a sweep) and the next predict() reflects it without re-instantiating the ensemble. Also simplifies the Plan 13-03 sweep runner.
- **Seed inside fit(), not __init__().** Constructor stays a pure metadata builder; `set_all_seeds(seed)` fires exactly once, right before members are trained.
- **No custom pickle hooks.** All supported member classes are already pickle-safe per Phase 8, so `BasePredictor.save/load` covers ensemble serialization transparently.

## Deviations from Plan

None — plan executed exactly as written. All 11 specified test behaviors were implemented as described; two extra sanity tests (`test_predict_before_fit_raises`, `test_inherits_base_predictor`) were added at no cost for defensive coverage.

## Issues Encountered

- **Full-suite smoke revealed 17 pre-existing failures unrelated to this plan** (PPO/matching modules missing `gymnasium`, 39-column feature-schema drift after Phase 8, legacy `test_evaluate_produces_full_metrics_dict` assertions in LR/XGB/naive/volume tests). Verified these failures are not introduced by 13-01 and logged them in `deferred-items.md` per the GSD scope-boundary rule. The 13 new ensemble tests pass cleanly.

## Deferred Issues

See `.planning/phases/13-ensemble-formalization/deferred-items.md` for the pre-existing test-suite failures observed during the smoke check. Recommended handling: dedicated tech-debt plan after Phase 13 ship.

## User Setup Required

None — no external service configuration required. The ensemble module is a pure in-process class with no new dependencies (relies only on numpy, pandas, and existing src/* modules).

## Self-Check: PASSED

- `src/models/ensemble.py` present — FOUND
- `tests/models/test_ensemble.py` present — FOUND
- `.planning/phases/13-ensemble-formalization/deferred-items.md` present — FOUND
- RED commit `4f8b01e` in git log — FOUND
- GREEN commit `7198b9a` in git log — FOUND
- `pytest tests/models/test_ensemble.py -x -q` exits 0 with 13/13 passing — VERIFIED
- `git diff src/live/strategy.py` is empty (ENSM-05) — VERIFIED

## Next Phase Readiness

- Plan 13-02 (per-member feature routing) can start immediately. Constructor contract is frozen and test-locked; 13-02 will extend `__init__` to accept an optional per-member `feature_cols` mapping without breaking any of the 13 existing tests.
- Plan 13-03 (weight sweep + §5.11 paper section) can consume `EnsemblePredictor` as a black box; the weight-normalization-in-predict() decision makes the sweep runner trivial (mutate `_weights`, call predict(), record P&L).

---
*Phase: 13-ensemble-formalization*
*Completed: 2026-04-23*
