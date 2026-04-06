---
phase: 06-rl-and-autoencoder
verified: 2026-04-05T00:00:00Z
status: passed
score: 5/5 must-haves verified
re_verification: false
---

# Phase 6: RL and Autoencoder Verification Report

**Phase Goal:** PPO trading agents and the autoencoder anomaly detector are trained, testing whether RL and anomaly detection improve trading performance
**Verified:** 2026-04-05
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths (from ROADMAP.md Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | A custom Gym environment simulates spread trading with appropriate state space, action space, and reward function | VERIFIED | `SpreadTradingEnv` in `src/models/trading_env.py`: observation `(198,)`, `Discrete(3)` actions, reward `clip(pos * spread_change - 0.02 * |pos_change|, -0.5, 0.5)`. 11 tests all pass. |
| 2 | The autoencoder is trained on normal spread behavior and flags anomalous spread patterns via reconstruction error threshold | VERIFIED | `AnomalyDetectorAutoencoder` in `src/models/autoencoder.py`: 31->16->8->4->8->16->31 with BatchNorm + ReLU, 95th percentile threshold, `flag_anomalies()` returns bool array. 9 tests all pass. |
| 3 | PPO on raw features produces a trading policy (even if it learns "don't trade," which is a valid finding) | VERIFIED | `PPORawPredictor` in `src/models/ppo_raw.py`: SB3 PPO trained 100k timesteps on `SpreadTradingEnv`, actions mapped to `{-0.03, 0.0, +0.03}`. Result: 1656 trades, RMSE=0.3189. 9 tests pass. |
| 4 | PPO with autoencoder signal filter produces a trading policy that only acts on flagged opportunities | VERIFIED | `PPOFilteredPredictor` in `src/models/ppo_filtered.py`: `FilteredTradingEnv` wrapper overrides reward to -0.01 for non-flagged bars, PPO trained on masked reward. Result: 899 trades, Sharpe=0.79. 10 tests pass. |
| 5 | All RL models are evaluated through the existing evaluation framework with results in the comparison table | VERIFIED | Both tier3 JSONs exist with full metric schema (rmse, mae, directional_accuracy, total_pnl, num_trades, win_rate, sharpe_ratio, seeds, seed_rmses, mean_rmse, std_rmse). 8-model cross-tier table at `experiments/results/tier3/cross_tier_comparison.txt`. |

**Score:** 5/5 truths verified

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/models/trading_env.py` | SpreadTradingEnv(gymnasium.Env) | VERIFIED | 233 lines. `class SpreadTradingEnv(gymnasium.Env)` with (198,) obs, Discrete(3), 0.02 tx cost, episode cycling, short-pair skip. |
| `tests/models/test_trading_env.py` | Unit tests for env reset/step/reward/boundary (min 80 lines) | VERIFIED | 208 lines. 11 tests in 5 test classes — all pass. |
| `src/models/autoencoder.py` | AnomalyDetectorAutoencoder for anomaly flagging | VERIFIED | 315 lines. `class AnomalyDetectorAutoencoder` with `_AutoencoderModule`, 31->16->8->4->8->16->31 arch, BatchNorm, 95th percentile threshold, NOT BasePredictor. |
| `tests/models/test_autoencoder.py` | Unit tests for reconstruction, training, thresholding (min 70 lines) | VERIFIED | 200 lines. 9 tests in 5 test classes — all pass. |
| `src/models/ppo_raw.py` | PPORawPredictor(BasePredictor) for Tier 3 RL | VERIFIED | 324 lines. `class PPORawPredictor(BasePredictor)` with SB3 PPO, action mapping dict, warm-up stitching. |
| `tests/models/test_ppo_raw.py` | Unit tests for PPO-Raw BasePredictor contract (min 70 lines) | VERIFIED | 158 lines. 9 tests — all pass. |
| `src/models/ppo_filtered.py` | PPOFilteredPredictor(BasePredictor) for Tier 4 RL + anomaly filter | VERIFIED | 406 lines. `class PPOFilteredPredictor(BasePredictor)` with `FilteredTradingEnv(gymnasium.Wrapper)`, autoencoder integration, reward masking. |
| `tests/models/test_ppo_filtered.py` | Unit tests for PPO-Filtered (min 80 lines) | VERIFIED | 202 lines. 10 tests — all pass. |
| `experiments/run_baselines.py` | Extended harness with --tier 3 and --tier all | VERIFIED | Contains `run_tier3_with_seeds`, `DEFAULT_RESULTS_DIR_TIER3`, `choices=["1","2","3","both","all"]`, PPO-Raw and PPO-Filtered in `_MODEL_ORDER`. |
| `experiments/results/tier3/ppo_raw.json` | PPO-Raw evaluation results | VERIFIED | model_name="PPO-Raw", RMSE=0.3189, 1656 trades, Sharpe=14.02, seeds=[42,123,456], mean_rmse, std_rmse present. |
| `experiments/results/tier3/ppo_filtered.json` | PPO-Filtered evaluation results | VERIFIED | model_name="PPO-Filtered", RMSE=0.3268, 899 trades, Sharpe=0.79, seeds=[42,123,456], mean_rmse, std_rmse present. |
| `src/models/__init__.py` | Updated exports with all Phase 6 classes | VERIFIED | Exports PPORawPredictor, PPOFilteredPredictor, AnomalyDetectorAutoencoder in both imports and `__all__`. |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/models/trading_env.py` | `src/models/sequence_utils.py` | `from src.models.sequence_utils import fit_feature_scaler, apply_feature_scaler` | WIRED | Line 28 — both functions used in `__init__` scaler fitting. |
| `src/models/autoencoder.py` | `src/models/sequence_utils.py` | `from src.models.sequence_utils import EarlyStopping, fit_feature_scaler, apply_feature_scaler, set_seed, get_device` | WIRED | Lines 25-31 — all five names actively used in `fit()` and `compute_reconstruction_error()`. |
| `src/models/ppo_raw.py` | `src/models/trading_env.py` | `from src.models.trading_env import SpreadTradingEnv` | WIRED | Line 34. `SpreadTradingEnv` instantiated in `fit()` (line 155). |
| `src/models/ppo_raw.py` | `src/models/base.py` | `class PPORawPredictor(BasePredictor)` | WIRED | Line 44. Inheritance verified; `evaluate()` returns same metric keys as all tiers (test confirmed). |
| `src/models/ppo_filtered.py` | `src/models/trading_env.py` | `SpreadTradingEnv` instantiation with reward masking | WIRED | Line 35 import, line 248 instantiation inside `fit()`, wrapped by `FilteredTradingEnv`. |
| `src/models/ppo_filtered.py` | `src/models/autoencoder.py` | `AnomalyDetectorAutoencoder` passed to constructor or fit | WIRED | Line 28 import, line 137 constructor param, line 216 internal training fallback, line 227 `flag_anomalies()` call. |
| `src/models/ppo_filtered.py` | `src/models/base.py` | `class PPOFilteredPredictor(BasePredictor)` | WIRED | Line 100. |
| `experiments/run_baselines.py` | `src/models/ppo_raw.py` | `from src.models.ppo_raw import PPORawPredictor` | WIRED | Line 48. Used in `build_models()` and `run_tier3_with_seeds()`. |
| `experiments/run_baselines.py` | `src/models/ppo_filtered.py` | `from src.models.ppo_filtered import PPOFilteredPredictor` | WIRED | Line 49. Used in `build_models()` and `run_tier3_with_seeds()`. |
| `experiments/run_baselines.py` | `src/models/autoencoder.py` | `from src.models.autoencoder import AnomalyDetectorAutoencoder` | WIRED | Line 50. Used in `run_tier3_with_seeds()` to train shared autoencoder before PPO-Filtered seeds. |

---

## Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| MOD-08 | 06-01, 06-03, 06-05 | PPO agent acting directly on raw microstructure features | SATISFIED | `PPORawPredictor` trains SB3 PPO on `SpreadTradingEnv`; result JSON exists with 1656 trades, RMSE=0.3189 |
| MOD-09 | 06-02, 06-05 | Autoencoder trained on normal spread behavior for anomaly detection | SATISFIED | `AnomalyDetectorAutoencoder` with 31->16->8->4->8->16->31 arch, 95th percentile threshold, tested and wired into PPO-Filtered |
| MOD-10 | 06-01, 06-04, 06-05 | PPO agent with autoencoder signal filter (acts on flagged opportunities) | SATISFIED | `PPOFilteredPredictor` with `FilteredTradingEnv` reward masking; result JSON exists with 899 trades, Sharpe=0.79 |

All three Phase 6 requirements verified satisfied. No orphaned requirements from REQUIREMENTS.md traceability table (MOD-08, MOD-09, MOD-10 all listed as Phase 6, all accounted for).

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | None | — | No TODOs, FIXMEs, stubs, or placeholder returns found in any Phase 6 source file |

---

## Test Results Summary

All 39 Phase 6 tests pass (0 failures, 0 errors, 3.69s):

- `test_trading_env.py` — 11 tests: interface, reset, step, reward, episode boundaries, short-pair skipping
- `test_autoencoder.py` — 9 tests: identity, architecture, training loss decrease, threshold, anomaly flagging, reconstruction error shape
- `test_ppo_raw.py` — 9 tests: BasePredictor contract, name, fit/predict, action mapping, evaluate keys, group_id guard
- `test_ppo_filtered.py` — 10 tests: BasePredictor contract, name, fit/predict, action mapping, evaluate keys, group_id guard, autoencoder integration (constructor + internal training)

---

## Human Verification Required

### 1. PPO Training Convergence Quality

**Test:** Run `python -m experiments.run_baselines --tier 3` on a fresh install and observe training progress logs
**Expected:** PPO-Raw makes active trades (confirmed: 1656 trades), PPO-Filtered trades on reduced set (confirmed: 899 trades). Neither collapses to all-hold or all-trade.
**Why human:** Convergence behavior and reward curve shape during 100k-step training cannot be verified programmatically from saved JSON artifacts alone.

### 2. Autoencoder Anomaly Flagging Rate Appropriateness

**Test:** Print the 95th percentile threshold value and fraction of flagged bars on the test set for PPO-Filtered
**Expected:** Approximately 5% of bars flagged (95th percentile by construction). The SUMMARY documents 5% flagging rate.
**Why human:** The appropriateness of the flagging rate as an anomaly signal (vs. noise) is a qualitative judgment. The threshold is mechanically correct per the 95th percentile design but whether it captures meaningful anomalies requires domain judgment.

---

## Gaps Summary

No gaps found. All 5 success criteria are met, all 11 artifacts are substantive and wired, all 3 requirements are satisfied, and all 39 tests pass.

The phase delivers its complete goal: PPO-Raw and PPO-Filtered trading agents are trained and evaluated alongside the autoencoder anomaly detector, producing an 8-model cross-tier comparison table that empirically answers whether RL and anomaly detection improve trading performance over regression baselines. The honest finding (PPO-Raw and PPO-Filtered both underperform XGBoost) is correctly reported and documented as a valid research conclusion.

---

_Verified: 2026-04-05_
_Verifier: Claude (gsd-verifier)_
