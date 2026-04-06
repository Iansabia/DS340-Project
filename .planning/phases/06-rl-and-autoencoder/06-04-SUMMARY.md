---
phase: 06-rl-and-autoencoder
plan: 04
subsystem: models
tags: [ppo, rl, autoencoder, anomaly-detection, stable-baselines3, gymnasium, reward-masking]

# Dependency graph
requires:
  - phase: 06-rl-and-autoencoder
    provides: SpreadTradingEnv (06-01), AnomalyDetectorAutoencoder (06-02)
  - phase: 05-time-series-models
    provides: sequence_utils (fit_feature_scaler, apply_feature_scaler, set_seed)
provides:
  - PPOFilteredPredictor(BasePredictor) for Tier 4 RL + anomaly filter
  - FilteredTradingEnv(gymnasium.Wrapper) for reward masking via anomaly flags
affects: [06-05 harness integration, Phase 7 evaluation]

# Tech tracking
tech-stack:
  added: []
  patterns: [gymnasium-wrapper-reward-masking, autoencoder-gated-rl-reward]

key-files:
  created:
    - src/models/ppo_filtered.py
    - tests/models/test_ppo_filtered.py
  modified: []

key-decisions:
  - "FilteredTradingEnv implemented as gymnasium.Wrapper around SpreadTradingEnv (not subclass) for clean separation of reward masking logic"
  - "Per-pair anomaly flag arrays stored in dict keyed by group_id for O(1) lookup during env step"
  - "Predict does NOT use anomaly filter -- filtering only affects training reward, not inference"

patterns-established:
  - "Gymnasium Wrapper pattern: FilteredTradingEnv wraps base env and intercepts step() to modify reward"
  - "Autoencoder-gated reward: non-flagged bars get fixed penalty, flagged bars get normal computed reward"

requirements-completed: [MOD-10]

# Metrics
duration: 3min
completed: 2026-04-06
---

# Phase 6 Plan 04: PPO-Filtered Summary

**PPO-Filtered predictor with autoencoder reward masking: non-flagged bars get -0.01 penalty, flagged bars earn normal reward via FilteredTradingEnv gymnasium wrapper**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-06T13:50:58Z
- **Completed:** 2026-04-06T13:54:13Z
- **Tasks:** 2 (TDD RED + GREEN)
- **Files modified:** 2

## Accomplishments
- TDD RED: 10 tests covering BasePredictor contract, action mapping, evaluate keys, group_id guard, autoencoder constructor integration, and auto-training when no detector provided
- TDD GREEN: Full PPOFilteredPredictor implementation with FilteredTradingEnv wrapper passing all 10 tests
- Autoencoder integration: constructor accepts pre-trained detector OR trains one internally during fit()

## Task Commits

Each task was committed atomically:

1. **Task 1: RED -- Write PPO-Filtered tests** - `25d13a84` (test)
2. **Task 2: GREEN -- Implement PPOFilteredPredictor** - `b3836774` (feat)

## Files Created/Modified
- `src/models/ppo_filtered.py` - PPOFilteredPredictor(BasePredictor) with FilteredTradingEnv wrapper, SB3 PPO training on anomaly-filtered reward, warm-up stitching predict
- `tests/models/test_ppo_filtered.py` - 10 tests: inheritance, name, fit/predict contract, action mapping, evaluate keys, group_id guard, autoencoder integration

## Decisions Made
- FilteredTradingEnv implemented as `gymnasium.Wrapper` around SpreadTradingEnv rather than subclassing, keeping reward masking logic cleanly separated from base environment mechanics
- Per-pair anomaly flag arrays stored in a dict keyed by group_id for efficient O(1) lookup during environment step; flags computed once at fit time from autoencoder.flag_anomalies()
- Predict does NOT use anomaly filter at inference time -- the filtering only affected training reward, so the learned policy runs on all rows during prediction (per CONTEXT.md spec)
- When no anomaly_detector is provided to constructor, fit() trains an AnomalyDetectorAutoencoder internally with conservative settings (5 epochs, patience 3) for test speed

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- PPOFilteredPredictor is ready for integration into the `--tier 3` / `--tier all` harness (Plan 05)
- Same BasePredictor interface as all other tiers -- evaluate() returns identical metric keys
- All four Phase 6 model components complete: SpreadTradingEnv (01), Autoencoder (02), PPO-Raw (03), PPO-Filtered (04)

## Self-Check: PASSED
