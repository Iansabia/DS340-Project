---
phase: 6
slug: rl-and-autoencoder
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-06
---

# Phase 6 — Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `.venv/bin/pytest tests/models/test_trading_env.py tests/models/test_autoencoder.py tests/models/test_ppo_raw.py tests/models/test_ppo_filtered.py -x -q` |
| **Full suite command** | `.venv/bin/pytest tests/ -v` |
| **Estimated runtime** | ~30 seconds (unit), ~10 min (full with PPO training) |

## Sampling Rate

- **After every task commit:** Run quick run command
- **After each wave:** Run full suite

## Dimensions (from 06-RESEARCH.md)

### Dimension 1: Trading Environment
- test_env_resets_correctly
- test_env_step_returns_valid
- test_env_respects_episode_boundary
- test_env_reward_includes_transaction_cost

### Dimension 2: Autoencoder
- test_autoencoder_reconstruction_shape
- test_autoencoder_loss_decreases
- test_threshold_flags_outliers

### Dimension 3: PPO Predictors
- test_ppo_inherits_base_predictor
- test_predict_before_fit_raises
- test_fit_returns_self
- test_predict_returns_1d_array
- test_action_to_prediction_mapping

### Dimension 4: Integration
- test_evaluate_returns_tier1_compatible_keys

### Dimension 5: End-to-end
- test_tier3_harness_writes_json
