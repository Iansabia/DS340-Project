---
phase: 6
slug: rl-and-autoencoder
type: research
created: 2026-04-06
sources: 3 parallel research agents (Gym env, autoencoder, PPO+integration)
---

# Phase 6 Research: RL and Autoencoder

## Question Answered
**How to implement PPO trading agents + autoencoder anomaly detector for cross-platform prediction market spread arbitrage, and integrate them into the existing evaluation framework?**

---

## Data Reality (from Phase 5)

| Fact | Value |
|------|-------|
| Train/test | 6802/1673 usable rows (after target + fillna) |
| Features | 31 (after zero-variance drop) |
| Pairs | 144, median 32 bars in train |
| Bar interval | 4 hours |
| Target | `spread_change_target = spread.shift(-1) - spread` per pair |
| Best baseline | XGBoost: RMSE=0.286, per-trade Sharpe=0.52, 58% win rate |
| SB3 installed | `stable_baselines3==2.7.1`, `gymnasium==1.2.3` — already in deps |

---

## Component 1: Gymnasium Environment

### State Space
- **Shape:** `(lookback * n_features_augmented,)` = `(6 * 33,)` = `(198,)` flattened
  - 31 scaled features + 1 current_position {-1,0,+1} + 1 time_in_episode_fraction
  - Lookback=6 bars (24h) matching GRU for fair comparison
  - Flattened for SB3 MlpPolicy compatibility (SB3 expects 1D for MLP)
- **Normalization:** Train-fitted StandardScaler (reuse `fit_feature_scaler` from sequence_utils)

### Action Space
- **Discrete(3):** {0: hold, 1: long (+1), 2: short (-1)}
- Position is always ±1 when trading, 0 when holding
- Matches profit_sim logic (sign-based trading, no fractional sizing)

### Reward Function
```
reward = position * actual_spread_change - TRANSACTION_COST * |position_change|
reward = clip(reward, -0.5, +0.5)

TRANSACTION_COST = 0.02  (2pp per-side, conservative estimate for Kalshi+Polymarket)
```
- Dense per-step (not sparse end-of-episode) — critical for 32-bar episodes
- Transaction cost prevents over-trading
- Clip prevents rare large moves from destabilizing gradients

### Episode Structure
- **One episode = one pair's full training sequence** (variable length, median 32 bars)
- Per-pair envs via iteration (not VecEnv — simpler for variable-length episodes)
- Episode terminates when pair's bars are exhausted
- Between episodes, next pair is loaded with `reset()`

---

## Component 2: Autoencoder Anomaly Detector

### Architecture (point-based, NOT sequence)
```
Encoder: Linear(31→16, ReLU, BN) → Linear(16→8, ReLU, BN) → Linear(8→4)
Decoder: Linear(4→8, ReLU, BN) → Linear(8→16, ReLU, BN) → Linear(16→31)
```
- **Bottleneck = 4** (8.5:1 compression, appropriate for 6k samples)
- **Point-based** (operates on single-bar 31-dim vectors, not windows)
- **Deterministic** (not VAE — simpler, fewer hyperparams, sufficient for anomaly detection)
- **BatchNorm** stabilizes training on small batches

### Training Protocol
| Param | Value |
|-------|-------|
| Loss | MSE(input, reconstruction) |
| Optimizer | Adam, lr=1e-3 |
| Batch size | 32 |
| Max epochs | 200 |
| Early stopping | patience=20, min_delta=1e-4 on val loss |
| Val split | 90/10 within-pair chronological (same as GRU/LSTM) |

### Threshold Selection
- Compute per-sample reconstruction error on training set
- **Threshold = 95th percentile** of training errors
- Target flagging rate: 10-20% of test bars
- If <5% flagged → lower to 90th. If >30% → raise to 98th.

### Integration with PPO
- **Binary flag filter (Pattern 1):** Autoencoder flags bars → PPO reward is masked to -0.01 (small penalty) on non-flagged bars
- PPO can only earn/lose on flagged timestamps → forces focus on anomalies
- Alternative (Pattern 2: recon_error as feature 32) deferred to ablation

### Class Design
- **NOT a BasePredictor** — it's a utility/filter, not a spread predictor
- Class: `AnomalyDetectorAutoencoder` with `fit()`, `compute_reconstruction_error()`, `set_threshold()`, `flag_anomalies()`
- Lives in `src/models/autoencoder.py`

---

## Component 3: PPO Trading Agents

### Library: Stable-Baselines3
- Already installed (`stable_baselines3==2.7.1`)
- Battle-tested in 100+ RL papers, avoids implementation bugs
- Wraps Gymnasium directly

### Hyperparameters
| Param | Value | Rationale |
|-------|-------|-----------|
| Policy | MlpPolicy [64, 64] | 2 hidden layers, matches data complexity |
| lr | 3e-4 | SB3 default, conservative for small data |
| n_steps | 256 | ~8 episodes per update at 32 bars/ep |
| batch_size | 64 | 4 mini-batches per update |
| n_epochs | 4 | Reuse steps 4x per update |
| gamma | 0.99 | Long horizon for 4h bars |
| gae_lambda | 0.95 | Standard GAE |
| clip_range | 0.2 | Standard PPO |
| ent_coef | 0.005 | Moderate exploration bonus |
| total_timesteps | 100,000 | ~17 passes over training data |

### Variant 1: PPO-Raw (MOD-08)
- State: 31 features + position + time fraction (flattened 6×33 = 198-dim)
- Actions: {hold, long, short}
- Reward: dense per-step with transaction cost
- **Expected:** 75% chance converges to "always hold" (0 trades). Valid finding.

### Variant 2: PPO-Filtered (MOD-10)
- Same architecture as PPO-Raw
- Autoencoder binary flag masks reward: non-flagged bars get reward=-0.01
- PPO only earns/loses on anomalous timestamps
- **Expected:** Slightly better than PPO-Raw if autoencoder captures real signal. Still likely underperforms XGBoost.

### BasePredictor Integration
```python
class PPORawPredictor(BasePredictor):
    def fit(self, X_train, y_train):
        # y_train used for reward construction in env
        # Train SB3 PPO on per-pair episodes
    
    def predict(self, X):
        # Run policy, map actions to pseudo-predictions:
        # hold → 0.0, long → +threshold, short → -threshold
        # where threshold = 0.03 (above profit_sim's 0.02 threshold)
```
- Action-to-prediction mapping: {hold: 0.0, long: +0.03, short: -0.03}
- profit_sim sees ±0.03 as "above threshold" → triggers trades
- RMSE will be high (coarse predictions) — honest, expected
- P&L/Sharpe is the meaningful comparison metric for RL

---

## Realistic Performance Expectations

| Model | Expected RMSE | Expected Trades | Expected Per-Trade SR | Likelihood |
|-------|--------------|-----------------|----------------------|------------|
| PPO-Raw | ~0.45 | 0 (hold-only) | 0.0 | 75% |
| PPO-Raw | ~0.35 | 200-500 | 0.05-0.15 | 25% |
| PPO-Filtered | ~0.40 | 50-200 | 0.10-0.25 | 50% |
| PPO-Filtered | ~0.45 | 0 (hold-only) | 0.0 | 50% |

**Both outcomes are valid findings.** If PPO underperforms (or doesn't trade), that directly answers KG's question: "Could RL act directly on features? Could this just be regression?" Answer: at this dataset scale, regression wins.

---

## File Structure

| Path | Purpose |
|------|---------|
| `src/models/trading_env.py` | `SpreadTradingEnv(gymnasium.Env)` |
| `src/models/autoencoder.py` | `AnomalyDetectorAutoencoder` |
| `src/models/ppo_raw.py` | `PPORawPredictor(BasePredictor)` — MOD-08 |
| `src/models/ppo_filtered.py` | `PPOFilteredPredictor(BasePredictor)` — MOD-10 |
| `src/models/__init__.py` (updated) | Export new classes |
| `experiments/run_baselines.py` (updated) | `--tier 3` for RL models |
| `experiments/results/tier3/` | PPO result JSONs |
| `tests/models/test_trading_env.py` | Env tests |
| `tests/models/test_autoencoder.py` | Autoencoder tests |
| `tests/models/test_ppo_raw.py` | PPO-Raw tests |
| `tests/models/test_ppo_filtered.py` | PPO-Filtered tests |

---

## Top Gotchas

1. **Mode collapse to "hold":** PPO converges to 0 trades. Mitigate with ent_coef=0.005-0.01. If still collapses, increase to 0.02. Report honestly if it persists.
2. **SB3 MlpPolicy needs 1D input:** Flatten the (6, 33) observation to (198,). Don't use CNN policy for 2D.
3. **Episode length varies:** Pairs range 5-141 bars. Short pairs (<6 bars) must be skipped (no valid lookback window).
4. **Autoencoder threshold is arbitrary:** Validate by checking if recon_error correlates with future |spread_change|. If correlation <0.1, the filter is noise.
5. **Action→prediction mapping is lossy:** PPO's 3 discrete actions map to {-0.03, 0, +0.03}. RMSE will be inflated vs regression. Use P&L/Sharpe for fair comparison.

---

## Validation Architecture

### Dimension 1: Unit — trading env
- test_env_resets_correctly: obs shape matches, position=0
- test_env_step_returns_valid: obs, reward, done, truncated, info have correct types
- test_env_respects_episode_boundary: done=True at end of pair sequence
- test_env_reward_includes_transaction_cost: position change incurs cost

### Dimension 2: Unit — autoencoder
- test_autoencoder_reconstruction_shape: output shape == input shape
- test_autoencoder_loss_decreases: train loss < initial loss after 10 epochs
- test_threshold_flags_outliers: synthetic outlier gets flagged

### Dimension 3: Unit — PPO predictors
- test_ppo_inherits_base_predictor
- test_predict_before_fit_raises
- test_fit_returns_self
- test_predict_returns_1d_array
- test_action_to_prediction_mapping: {0→0.0, 1→+0.03, 2→-0.03}

### Dimension 4: Integration — evaluate produces metrics
- test_evaluate_returns_tier1_compatible_keys

### Dimension 5: End-to-end
- test_tier3_harness_writes_json: --tier 3 produces tier3/*.json

---

*Phase 6 research complete. Ready for CONTEXT.md + planning.*
