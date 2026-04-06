---
phase: 06-rl-and-autoencoder
plan: 03
subsystem: models
tags: [ppo, reinforcement-learning, stable-baselines3, gymnasium, spread-trading]

# Dependency graph
requires:
  - phase: 06-01
    provides: SpreadTradingEnv gymnasium environment for PPO training
  - phase: 05
    provides: BasePredictor contract, sequence_utils (scaler, set_seed)
provides:
  - PPORawPredictor(BasePredictor) for Tier 3 RL evaluation
  - Action-to-prediction mapping for evaluate() compatibility
affects: [06-04, 06-05, 07]

# Tech tracking
tech-stack:
  added: [stable-baselines3 PPO]
  patterns: [RL-to-BasePredictor adapter with action-to-prediction mapping, warm-up stitching for RL predict]

key-files:
  created:
    - src/models/ppo_raw.py
    - tests/models/test_ppo_raw.py
  modified: []

key-decisions:
  - "PPO predict() tracks current_position per pair to augment observations with correct position state"
  - "Warm-up stitching reuses GRU pattern: prepend cached train rows per group_id for full lookback windows"
  - "Action-to-prediction mapping exposed as class-level _action_to_prediction dict for direct testing"

patterns-established:
  - "RL-to-BasePredictor adapter: discrete actions mapped to coarse pseudo-predictions for evaluation compatibility"
  - "Per-pair position tracking during predict: mirrors env state for correct observation augmentation"

requirements-completed: [MOD-08]

# Metrics
duration: 2min
completed: 2026-04-06
---

# Phase 6 Plan 3: PPO-Raw Predictor Summary

**SB3 PPO agent trained on SpreadTradingEnv with MlpPolicy [64,64], wrapped as BasePredictor via action-to-prediction mapping (hold->0.0, long->+0.03, short->-0.03)**

## Performance

- **Duration:** 2 min
- **Started:** 2026-04-06T13:50:35Z
- **Completed:** 2026-04-06T13:52:57Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- PPORawPredictor inherits BasePredictor and implements full fit/predict/evaluate contract
- SB3 PPO trains on SpreadTradingEnv with locked hyperparameters (lr=3e-4, 100k timesteps, ent_coef=0.005)
- Discrete actions converted to pseudo-predictions for cross-tier evaluation compatibility
- 9 tests covering inheritance, action mapping, evaluate keys, and group_id guards

## Task Commits

Each task was committed atomically:

1. **Task 1: RED -- Write PPO-Raw tests** - `d71c1599` (test)
2. **Task 2: GREEN -- Implement PPORawPredictor** - `c778c6ee` (feat)

## Files Created/Modified
- `src/models/ppo_raw.py` - PPORawPredictor(BasePredictor) wrapping SB3 PPO with action-to-prediction mapping
- `tests/models/test_ppo_raw.py` - 9 tests for BasePredictor contract, action mapping, evaluate keys, group_id guard

## Decisions Made
- PPO predict() tracks current_position per pair during inference to augment observations with correct position state (mirrors env logic)
- Warm-up stitching follows GRU pattern: cached train data prepended per group_id for full lookback windows
- Action-to-prediction mapping exposed as class-level `_action_to_prediction` dict for direct test access
- Position state for observation augmentation uses action-to-position mapping {0:0, 1:1, 2:-1} matching env's `_ACTION_TO_POSITION`

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- PPORawPredictor ready for integration into run_baselines.py --tier 3 harness (06-05)
- PPO-Filtered (06-04) can reuse the same predict() warm-up stitching pattern
- Autoencoder from 06-02 ready to be combined with PPO in 06-04

## Self-Check: PASSED

- [x] src/models/ppo_raw.py exists
- [x] tests/models/test_ppo_raw.py exists
- [x] 06-03-SUMMARY.md exists
- [x] Commit d71c1599 exists (RED tests)
- [x] Commit c778c6ee exists (GREEN implementation)

---
*Phase: 06-rl-and-autoencoder*
*Completed: 2026-04-06*
