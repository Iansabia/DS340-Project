---
phase: 05-time-series-models
verified: 2026-04-06T01:15:00Z
status: passed
score: 4/4 must-haves verified
re_verification: false
---

# Phase 5: Time Series Models Verification Report

**Phase Goal:** Recurrent and attention-based models are trained and evaluated, testing whether temporal structure improves spread prediction
**Verified:** 2026-04-06T01:15:00Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | GRU model is trained on windowed hourly sequences and evaluated through the existing evaluation framework | VERIFIED | `src/models/gru.py` (446 lines) implements GRUPredictor inheriting BasePredictor. 3-seed training results in `experiments/results/tier2/gru.json` with RMSE=0.2896 +/- 0.0024, mean_rmse/std_rmse/seed_rmses all present. 14 tests pass in `tests/models/test_gru.py`. |
| 2 | LSTM model is trained on windowed hourly sequences and evaluated through the existing evaluation framework | VERIFIED | `src/models/lstm.py` (428 lines) implements LSTMPredictor inheriting BasePredictor. 3-seed training results in `experiments/results/tier2/lstm.json` with RMSE=0.2910 +/- 0.0004, mean_rmse/std_rmse/seed_rmses all present. 14 tests pass in `tests/models/test_lstm.py`. |
| 3 | TFT model is trained via PyTorch Forecasting and evaluated through the existing framework (or explicitly deferred with documented rationale if dataset is too small or timeline is tight) | VERIFIED | TFT deferred per the roadmap's explicit deferral clause. `05-DEFERRALS.md` documents param-to-sample ratio argument (ratio ~1.9 vs 0.01 threshold), timeline argument (22 days remaining for Phases 6-8), and alternative coverage via GRU/LSTM. 9 deferral guard tests pass in `tests/planning/test_tft_deferral_documented.py`. |
| 4 | Results for all Tier 2 models appear in the same comparison table as Tier 1, enabling direct cross-tier comparison | VERIFIED | `experiments/run_baselines.py` supports `--tier {1,2,both}`. `05-04-comparison-output.txt` contains 6-model unified table (Naive, Volume, Linear Regression, XGBoost, GRU, LSTM). All 6 Tier 1 + Tier 2 result JSONs report `n_features=31`, confirming apples-to-apples comparison. |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/models/sequence_utils.py` | Shared utilities (windowing, early stopping, seed, device, scaler) | VERIFIED | 201 lines, 6 exports: `create_sequences`, `EarlyStopping`, `set_seed`, `get_device`, `fit_feature_scaler`, `apply_feature_scaler`. Zero-variance guard raises ValueError with offending column names. |
| `src/models/gru.py` | GRUPredictor class | VERIFIED | 446 lines, hidden_size=64, 1-layer, input dropout=0.3, AdamW optimizer, warm-up stitching, padded warm-up logging. Inherits BasePredictor. |
| `src/models/lstm.py` | LSTMPredictor class | VERIFIED | 428 lines, hidden_size=32, 1-layer, input dropout=0.3, AdamW optimizer, warm-up stitching, padded warm-up logging. Inherits BasePredictor. |
| `experiments/run_baselines.py` | Extended with --tier flag, cross-tier table | VERIFIED | 473 lines, `--tier {1,2,both}` CLI flag, `prepare_xy_for_seq` for group_id pass-through, `run_tier2_with_seeds` for 3-seed aggregation, `format_comparison_table` for unified display. |
| `experiments/results/tier2/gru.json` | GRU experiment results | VERIFIED | Contains metrics (RMSE, MAE, directional accuracy, P&L, etc.), seed-aggregated stats (mean_rmse=0.2896, std_rmse=0.0024, seeds=[42,123,456]), n_features=31. |
| `experiments/results/tier2/lstm.json` | LSTM experiment results | VERIFIED | Contains metrics (RMSE, MAE, directional accuracy, P&L, etc.), seed-aggregated stats (mean_rmse=0.2910, std_rmse=0.0004, seeds=[42,123,456]), n_features=31. |
| `.planning/phases/05-time-series-models/05-DEFERRALS.md` | TFT deferral rationale | VERIFIED | Documents MOD-07 deferral with param-to-sample ratio argument, timeline argument, GRU/LSTM alternative coverage, and re-examination criterion (>=20k windows). |
| `tests/models/test_sequence_utils.py` | Test suite for sequence utilities | VERIFIED | 269 lines, 15 test cases across 5 classes (TestCreateSequences, TestEarlyStopping, TestSetSeed, TestGetDevice, TestFeatureScaler). All pass. |
| `tests/models/test_gru.py` | Test suite for GRU model | VERIFIED | 207 lines, 14 test cases across 4 classes. All pass. |
| `tests/models/test_lstm.py` | Test suite for LSTM model | VERIFIED | 224 lines, 14 test cases across 4 classes. All pass. |
| `tests/planning/test_tft_deferral_documented.py` | Deferral guard tests | VERIFIED | 57 lines, 9 test cases verifying deferral file existence, content (MOD-07, roadmap clause, param-to-sample ratio, overfitting argument, GRU/LSTM alternative, timeline). All pass. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `gru.py` | `sequence_utils.py` | `from src.models.sequence_utils import create_sequences, EarlyStopping, set_seed, get_device, fit_feature_scaler, apply_feature_scaler` | WIRED | All 6 imports used in fit() and predict() methods |
| `lstm.py` | `sequence_utils.py` | `from src.models.sequence_utils import create_sequences, EarlyStopping, set_seed, get_device, fit_feature_scaler, apply_feature_scaler` | WIRED | All 6 imports used in fit() and predict() methods |
| `gru.py` | `base.py` | `from src.models.base import BasePredictor` + `class GRUPredictor(BasePredictor)` | WIRED | Inherits BasePredictor, implements name, fit, predict. evaluate() inherited. |
| `lstm.py` | `base.py` | `from src.models.base import BasePredictor` + `class LSTMPredictor(BasePredictor)` | WIRED | Inherits BasePredictor, implements name, fit, predict. evaluate() inherited. |
| `run_baselines.py` | `gru.py`, `lstm.py` | `from src.models.gru import GRUPredictor`, `from src.models.lstm import LSTMPredictor` | WIRED | Both imported and used in `build_models(tier="2")` and `run_tier2_with_seeds()` |
| `run_baselines.py` | Tier 2 result JSONs | `save_results(model_name, last_metrics, results_dir, extra=extra)` | WIRED | `gru.json` and `lstm.json` exist with full metrics and seed-aggregated stats |
| `run_baselines.py` | Cross-tier table | `format_comparison_table(combined, tier="both")` | WIRED | `05-04-comparison-output.txt` shows 6-model table with all tiers |
| `test_gru.py` | `gru.py` | `from src.models.gru import GRUPredictor` | WIRED | 14 tests exercise fit/predict/evaluate/group_id contract/padding |
| `test_lstm.py` | `lstm.py` | `from src.models.lstm import LSTMPredictor` | WIRED | 14 tests exercise fit/predict/evaluate/group_id contract/padding |
| `test_sequence_utils.py` | `sequence_utils.py` | `from src.models.sequence_utils import ...` | WIRED | 15 tests cover all 6 exports |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| MOD-05 | 05-01, 05-02, 05-04 | GRU trained on spread prediction with hourly sequences | SATISFIED | GRUPredictor trained with 3 seeds, results in tier2/gru.json (RMSE=0.2896), appears in cross-tier comparison table |
| MOD-06 | 05-01, 05-03, 05-04 | LSTM trained on spread prediction with hourly sequences | SATISFIED | LSTMPredictor trained with 3 seeds, results in tier2/lstm.json (RMSE=0.2910), appears in cross-tier comparison table |
| MOD-07 | 05-05 | TFT via PyTorch Forecasting (droppable if timeline tight) | SATISFIED (deferred) | Deferred with documented rationale in 05-DEFERRALS.md per roadmap success criterion #3's explicit deferral clause. REQUIREMENTS.md marks MOD-07 as Complete (deferral counts as disposition). |

No orphaned requirements found. REQUIREMENTS.md maps MOD-05, MOD-06, MOD-07 to Phase 5, and all three are accounted for in Phase 5 plans.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | - | - | No TODO, FIXME, HACK, PLACEHOLDER, empty return, or stub patterns found in any Phase 5 source files |

All four key source files (`sequence_utils.py`, `gru.py`, `lstm.py`, `run_baselines.py`) are clean of anti-patterns.

### Human Verification Required

### 1. Cross-Tier Comparison Table Visual Correctness

**Test:** Run `python -m experiments.run_baselines --tier both` and inspect the terminal output
**Expected:** 6-model table with consistent column alignment, GRU/LSTM showing +/- std notation, all metrics populated with non-zero values
**Why human:** Terminal formatting and visual alignment cannot be fully verified programmatically

### 2. Training Convergence Quality

**Test:** Run `python -m experiments.run_baselines --tier 2` and observe epoch-by-epoch loss printout
**Expected:** Training loss decreases over epochs; early stopping triggers before max_epochs; validation loss does not diverge
**Why human:** Loss trajectory quality (smooth convergence vs oscillation) requires human judgment

### 3. Result Reasonableness

**Test:** Compare GRU/LSTM RMSE (0.2896/0.2910) against XGBoost (0.2857) and naive baseline (0.4995)
**Expected:** Tier 2 models significantly beat naive baselines but do not dramatically outperform XGBoost (consistent with small-dataset complexity-vs-performance thesis)
**Why human:** Assessing whether results are scientifically reasonable for the project narrative requires domain judgment

### Gaps Summary

No gaps found. All four success criteria are verified:

1. GRU is trained on windowed sequences with 3-seed aggregation and evaluated via the BasePredictor evaluation framework, producing results stored in JSON format.
2. LSTM is trained identically and evaluated through the same framework.
3. TFT is explicitly deferred with a thorough, documented rationale covering dataset size (param-to-sample ratio), timeline, and alternative coverage -- exactly per the roadmap's deferral clause.
4. The cross-tier comparison table includes all 6 models (4 Tier 1 + 2 Tier 2) at the same 31-feature set, enabling direct comparison. Tier 1 was re-run to ensure feature-set parity.

All 52 tests pass (15 sequence_utils + 14 GRU + 14 LSTM + 9 deferral guard). No anti-patterns detected. All key links are wired. All three requirements (MOD-05, MOD-06, MOD-07) have clear disposition.

---

_Verified: 2026-04-06T01:15:00Z_
_Verifier: Claude (gsd-verifier)_
