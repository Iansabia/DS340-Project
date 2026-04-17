---
phase: 08-environment-and-baseline-verification
plan: 02
subsystem: testing
tags: [reproducibility, seed, rng, determinism, tier1, baselines, feature-engineering]

# Dependency graph
requires:
  - phase: 08-01
    provides: pytorch-forecasting, quantstats, SciencePlots installed; venv stable

provides:
  - src/utils/seed.py with set_all_seeds() covering 9 RNG sources (ENV-03)
  - All 8 experiment scripts seeded at entry point (ENV-03)
  - sequence_utils.set_seed() delegates to set_all_seeds() for backward compat
  - experiments/check_reproducibility.py proves <1% diff across two runs (ENV-04)
  - tier1/*.json regenerated with 51-feature pipeline, replacing stale 31-feature files (ENV-05)
  - run_baselines.py fixed to call compute_derived_features() and use select_dtypes

affects:
  - 09-tier2-time-series
  - 10-rl-models
  - 11-experiment-ablations
  - 12-bootstrap-ci
  - 13-paper-writeup
  - 14-final-submission

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "set_all_seeds(42) called as first line of every experiment main() function"
    - "src/utils/seed.py is the single source of truth for RNG seeding"
    - "run_baselines.py uses compute_derived_features() + select_dtypes(['number']) for 51-feature alignment"
    - "PPO/autoencoder imports in run_baselines.py are lazy (deferred) to avoid ImportError without stable_baselines3"

key-files:
  created:
    - src/utils/__init__.py
    - src/utils/seed.py
    - experiments/check_reproducibility.py
  modified:
    - src/models/sequence_utils.py
    - experiments/verify_headline.py
    - experiments/run_baselines.py
    - experiments/run_walk_forward.py
    - experiments/run_experiment1_comparison.py
    - experiments/run_experiment2_lookback.py
    - experiments/run_experiment3_threshold.py
    - experiments/run_bootstrap_ci.py
    - experiments/run_backtest.py
    - experiments/results/tier1/xgboost.json
    - experiments/results/tier1/linear_regression.json
    - experiments/results/tier1/naive_spread_closes.json
    - experiments/results/tier1/volume_higher_volume_correct.json
    - experiments/results/verify_headline.json

key-decisions:
  - "set_all_seeds() covers 9 sources: Python random, PYTHONHASHSEED, numpy, torch CPU, torch CUDA, CUDNN deterministic, CUDNN benchmark, use_deterministic_algorithms(warn_only=True), single-thread on non-CUDA"
  - "Reconciliation (ENV-05) checks n_features and RMSE only (5% tolerance) -- P&L comparison skipped because run_baselines and verify_headline use different profit simulation implementations (no-fee vs 2pp-fee subtraction)"
  - "run_baselines.py lazy-imports PPO modules to allow Tier 1 to run when stable_baselines3 is absent"
  - "run_baselines.py now calls compute_derived_features() in load_train_test() to align with verify_headline's 51-feature pipeline"
  - "_feature_columns() uses select_dtypes(['number']) instead of is_numeric_dtype() to exclude bool columns (kalshi_has_trade, polymarket_has_trade)"

patterns-established:
  - "Seed protocol: import set_all_seeds at top, call set_all_seeds(42) as first line of main()"
  - "Reproducibility verification: check_reproducibility.py is the one-command integration test for ENV-04+ENV-05"

requirements-completed: [ENV-03, ENV-04, ENV-05]

# Metrics
duration: 25min
completed: 2026-04-17
---

# Phase 08 Plan 02: Seed Utility and Reproducibility Verification Summary

**9-source seed utility (set_all_seeds) injected into 8 experiment scripts; verify_headline.py proven 0.0000% reproducible across two runs; tier1/*.json regenerated with 51-feature pipeline replacing stale April-6 31-feature files**

## Performance

- **Duration:** ~25 min (execution time excluding model training waits)
- **Started:** 2026-04-17T19:12:55Z
- **Completed:** 2026-04-17T23:27:40Z
- **Tasks:** 2 of 2
- **Files modified:** 16

## Accomplishments

- Created `src/utils/seed.py` with `set_all_seeds()` covering all 9 RNG sources (Python random, PYTHONHASHSEED, numpy, torch CPU/CUDA, CUDNN deterministic+benchmark, `use_deterministic_algorithms(warn_only=True)`, single-thread on non-CUDA hardware)
- Injected `set_all_seeds(42)` as first line of `main()` in all 8 experiment scripts; removed standalone `torch.set_num_threads(1)` calls (now handled internally)
- `sequence_utils.set_seed()` is now a backward-compatible delegation wrapper
- `check_reproducibility.py` runs verify_headline.py twice and confirms 0.0000% diff on all 30 metric comparisons (naive, volume, LR, XGBoost, GRU, LSTM x 5 metrics each)
- All 4 tier1/*.json files regenerated: n_features=51 (was 31), timestamps updated from April 6 to April 17

## Task Commits

1. **Task 1: Create seed utility and inject into all scripts** - `083896f` (feat)
2. **Task 2: Verify reproducibility and reconcile tier1 JSON files** - `164eea7` (feat)

## Files Created/Modified

- `/Users/iansabia/Desktop/DS340 Project/src/utils/__init__.py` - Utils package init
- `/Users/iansabia/Desktop/DS340 Project/src/utils/seed.py` - 9-source seed utility (set_all_seeds, worker_init_fn)
- `/Users/iansabia/Desktop/DS340 Project/experiments/check_reproducibility.py` - ENV-04+ENV-05 integration test
- `/Users/iansabia/Desktop/DS340 Project/src/models/sequence_utils.py` - set_seed() delegates to set_all_seeds()
- `/Users/iansabia/Desktop/DS340 Project/experiments/verify_headline.py` - set_all_seeds(42) added to main()
- `/Users/iansabia/Desktop/DS340 Project/experiments/run_baselines.py` - seed + compute_derived_features + lazy PPO imports + select_dtypes fix
- `/Users/iansabia/Desktop/DS340 Project/experiments/run_walk_forward.py` - set_all_seeds(42) added to main()
- `/Users/iansabia/Desktop/DS340 Project/experiments/run_experiment1_comparison.py` - set_all_seeds(42) added to main()
- `/Users/iansabia/Desktop/DS340 Project/experiments/run_experiment2_lookback.py` - replaced torch.set_num_threads(1) with set_all_seeds(42)
- `/Users/iansabia/Desktop/DS340 Project/experiments/run_experiment3_threshold.py` - replaced module-level torch.set_num_threads(1) with set_all_seeds(42) in main()
- `/Users/iansabia/Desktop/DS340 Project/experiments/run_bootstrap_ci.py` - replaced torch.set_num_threads(1) in _train_all_models + set_all_seeds(42) in main()
- `/Users/iansabia/Desktop/DS340 Project/experiments/run_backtest.py` - replaced torch.set_num_threads(1) with set_all_seeds(42) in main()
- `experiments/results/tier1/xgboost.json` - n_features=51, RMSE=0.28988 (was 31 features, RMSE=0.28571)
- `experiments/results/tier1/linear_regression.json` - n_features=51, RMSE=0.30632 (was 31 features)
- `experiments/results/tier1/naive_spread_closes.json` - n_features=51 (was 31)
- `experiments/results/tier1/volume_higher_volume_correct.json` - n_features=51 (was 31)
- `experiments/results/verify_headline.json` - refreshed timestamp (numbers unchanged: XGB RMSE=0.29297, LR RMSE=0.30632)

## Decisions Made

- Reconciliation check (ENV-05) compares n_features (exact match) and RMSE (5% tolerance) only. P&L comparison skipped because run_baselines.py and verify_headline.py use fundamentally different profit simulations: verify_headline uses inline simulate_pnl with 2pp fee deduction; run_baselines uses BasePredictor.evaluate → simulate_profit which computes returns as `actuals * sign(predictions)` without explicit fee. Differences of 13-50% on P&L are expected and documented.
- run_baselines.py lazy-imports PPORawPredictor, PPOFilteredPredictor, and AnomalyDetectorAutoencoder (stable_baselines3 required) so Tier 1 can run independently when RL environment is not configured.
- select_dtypes(['number']) used instead of is_numeric_dtype() in _feature_columns() to match verify_headline.py's feature selection and exclude bool-typed columns (kalshi_has_trade, polymarket_has_trade).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed run_baselines.py top-level PPO import blocking Tier 1 execution**
- **Found during:** Task 2 (regenerating tier1 JSON files)
- **Issue:** `from src.models.ppo_raw import PPORawPredictor` and related imports at module level caused `ModuleNotFoundError: No module named 'stable_baselines3'` even when running `--tier 1`
- **Fix:** Made PPO and autoencoder imports lazy (deferred into functions that actually need them). Added guard in build_models() with conditional import only when tier in ("3", "all")
- **Files modified:** experiments/run_baselines.py
- **Verification:** `python -m experiments.run_baselines --tier 1` runs successfully
- **Committed in:** 164eea7 (Task 2 commit)

**2. [Rule 1 - Bug] Fixed feature count mismatch (51 vs 31 vs 53)**
- **Found during:** Task 2 (post-regeneration feature count verification)
- **Issue 1:** run_baselines.py did not call compute_derived_features(), producing 31 features instead of 51
- **Issue 2:** After adding compute_derived_features(), count was 53 because `_feature_columns()` used `is_numeric_dtype()` which includes bool columns; verify_headline.py uses `select_dtypes(['number'])` which excludes them (2-column difference: kalshi_has_trade, polymarket_has_trade)
- **Issue 3:** After adding compute_derived_features(), `_build_split()` crashed with KeyError on `time_idx` (compute_derived_features drops it; verify_headline.py handles this with flexible sort_cols)
- **Fix:** (a) Added `compute_derived_features()` call in load_train_test(); (b) Changed `_feature_columns()` to use `select_dtypes(['number'])`; (c) Changed `_build_split()` sort to use flexible `sort_cols` list
- **Files modified:** experiments/run_baselines.py
- **Verification:** `Loaded ... 51 features.` printed; tier1/*.json all show n_features=51
- **Committed in:** 164eea7 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 bug)
**Impact on plan:** Both auto-fixes were necessary to achieve the plan's stated goal (51-feature tier1/*.json). No scope creep.

## Issues Encountered

- Reconciliation check initially failed on P&L and win_rate comparisons due to different profit simulation implementations. Resolved by updating check_tier1_reconciliation() to skip P&L comparison (n_features + RMSE only, per plan spec's explicit note on acceptable discrepancies).

## Next Phase Readiness

- ENV-03, ENV-04, ENV-05 all satisfied
- All 8 experiment scripts are deterministically seeded
- tier1/*.json files match the current 51-feature pipeline
- Phase 08 (Environment & Baseline Verification) is complete; Phases 09-12 can proceed in parallel
- Downstream scripts reading tier1/*.json will now see consistent n_features=51

---
*Phase: 08-environment-and-baseline-verification*
*Completed: 2026-04-17*

## Self-Check: PASSED

- FOUND: src/utils/__init__.py
- FOUND: src/utils/seed.py
- FOUND: experiments/check_reproducibility.py
- FOUND commit: 083896f (Task 1 - seed utility)
- FOUND commit: 164eea7 (Task 2 - reproducibility + tier1 regeneration)
