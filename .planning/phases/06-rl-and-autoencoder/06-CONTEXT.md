---
phase: 6
slug: rl-and-autoencoder
type: context
created: 2026-04-06
source: 3 parallel research agents + Phase 5 data verification
status: Ready for planning
---

# Phase 6: RL and Autoencoder — Context

**Central question:** Does RL (with or without anomaly detection signal filtering) improve trading performance over regression baselines?

<domain>
## Phase Boundary

**In scope:**
- Custom Gymnasium environment for spread trading (SpreadTradingEnv)
- Autoencoder anomaly detector (MOD-09)
- PPO on raw features (MOD-08) — Tier 3
- PPO with autoencoder signal filter (MOD-10) — Tier 4
- Integration into existing `--tier` harness + comparison table
- Tier 3 results JSON files

**Out of scope:**
- Hyperparameter sweeps beyond default config
- Alternative RL algorithms (DQN, A2C, SAC)
- Multi-step reward shaping experiments (Phase 7 ablation)
- Transaction cost modeling in profit_sim (Phase 7 backtesting)
</domain>

<decisions>
## Implementation Decisions (LOCKED)

### Gymnasium environment (SpreadTradingEnv)
- **State space:** Flattened `(198,)` = 6 bars × 33 channels (31 scaled features + current_position + time_fraction)
- **Lookback:** 6 bars (24h on 4h data), matching GRU for fair comparison
- **Action space:** `Discrete(3)` — {0: hold, 1: long, 2: short}
- **Reward:** `clip(position * actual_spread_change - 0.02 * |position_change|, -0.5, 0.5)`
- **Transaction cost:** 0.02 (2pp per-side) hardcoded in reward
- **Episode:** One pair's full training sequence; `reset()` loads next pair
- **Normalization:** Train-fitted StandardScaler via `fit_feature_scaler()` (reuse from sequence_utils)
- **Pairs with <lookback bars:** Skipped (same as GRU/LSTM)
- **File:** `src/models/trading_env.py`

### Autoencoder anomaly detector (MOD-09)
- **Architecture:** Point-based, deterministic (NOT VAE, NOT sequence)
- **Encoder:** Linear(31→16, ReLU, BN) → Linear(16→8, ReLU, BN) → Linear(8→4)
- **Decoder:** Linear(4→8, ReLU, BN) → Linear(8→16, ReLU, BN) → Linear(16→31)
- **Bottleneck:** 4 dimensions (8.5:1 compression)
- **Loss:** MSE(input, reconstruction)
- **Optimizer:** Adam, lr=1e-3
- **Batch size:** 32
- **Max epochs:** 200, early stopping patience=20, min_delta=1e-4
- **Val split:** 90/10 within-pair chronological (same as Tier 2)
- **Threshold:** 95th percentile of training reconstruction errors
- **Target flagging rate:** 10-20% of test bars
- **Class:** `AnomalyDetectorAutoencoder` — NOT a BasePredictor (it's a filter, not a predictor)
- **File:** `src/models/autoencoder.py`

### PPO-Raw (MOD-08)
- **Library:** Stable-Baselines3 (`PPO` with `MlpPolicy`)
- **Policy net:** [64, 64] hidden layers
- **lr:** 3e-4
- **n_steps:** 256
- **batch_size:** 64
- **n_epochs:** 4
- **gamma:** 0.99
- **gae_lambda:** 0.95
- **clip_range:** 0.2
- **ent_coef:** 0.005
- **total_timesteps:** 100,000
- **Seeds:** {42, 123, 456} (same as Tier 2, report mean±std)
- **Class:** `PPORawPredictor(BasePredictor)` in `src/models/ppo_raw.py`
- **name property:** `"PPO-Raw"`

### PPO-Filtered (MOD-10)
- Same PPO config as PPO-Raw
- Autoencoder binary flag masks reward: non-flagged bars get reward = -0.01 (small penalty to discourage holding during normal periods)
- PPO can only earn/lose on flagged timestamps
- **Class:** `PPOFilteredPredictor(BasePredictor)` in `src/models/ppo_filtered.py`
- **name property:** `"PPO-Filtered"`
- **Depends on:** trained autoencoder (passed via constructor or fit-time)

### BasePredictor integration
- `fit(X_train, y_train)`: X must contain `group_id` column (same as GRU/LSTM). y_train used for reward construction in env.
- `predict(X)`: Run trained policy per pair, convert discrete actions to pseudo-predictions:
  - hold (0) → 0.0
  - long (1) → +0.03 (above profit_sim threshold of 0.02)
  - short (2) → -0.03
- `evaluate()` inherited from BasePredictor — same metrics as all tiers
- **RMSE will be inflated** because predictions are coarse (3 values). P&L and per-trade Sharpe are the meaningful comparison metrics.

### Evaluation + comparison table
- Extend `experiments/run_baselines.py` with `--tier 3` and `--tier all`
- `--tier 3` runs PPO-Raw + PPO-Filtered with 3 seeds each
- `--tier all` (or `--tier both` extended) produces full cross-tier table: Tier 1 + 2 + 3
- Results saved to `experiments/results/tier3/ppo_raw.json` and `ppo_filtered.json`
- JSON schema identical to Tier 1/2

### Testing (TDD Iron Law)
- Tests FIRST for each component
- test_trading_env.py: env reset/step/reward/boundary tests
- test_autoencoder.py: reconstruction shape, loss decrease, threshold flagging
- test_ppo_raw.py: BasePredictor contract, action mapping, evaluate keys
- test_ppo_filtered.py: same as ppo_raw + autoencoder filter integration

### Performance expectations (document in SUMMARY)
- PPO-Raw: 75% likely converges to "always hold" (0 trades). Valid finding.
- PPO-Filtered: 50% likely learns some trading. Still likely underperforms XGBoost.
- Both outcomes answer the research question: "is RL complexity justified at this dataset scale?"

### Claude's Discretion
- Exact SB3 `learn()` callback for logging
- Whether to use SB3's VecEnv or iterate pairs manually
- Print/logging format during training
- Whether to save PPO model checkpoints (not required for paper, but nice)
</decisions>

<canonical_refs>
## Canonical References

### Project-level
- `CLAUDE.md` — RL is Tier 3/4, PPO expected to underperform, complexity-vs-performance thesis
- `.planning/REQUIREMENTS.md` — MOD-08 (PPO raw), MOD-09 (autoencoder), MOD-10 (PPO filtered)
- `.planning/ROADMAP.md` — Phase 6 goal + 5 success criteria

### Phase 6 research
- `.planning/phases/06-rl-and-autoencoder/06-RESEARCH.md` — full synthesis

### Patterns to follow from Phase 5
- `src/models/base.py` — BasePredictor contract
- `src/models/gru.py` — complex model wrapping BasePredictor (fit/predict pattern with internal training)
- `src/models/sequence_utils.py` — shared utilities (scaler, EarlyStopping, set_seed)
- `experiments/run_baselines.py` — --tier flag pattern, run_tier2_with_seeds pattern, results saving
- `experiments/results/tier2/gru.json` — JSON schema to match
- `tests/models/test_gru.py` — test pattern to mirror

### Data
- `data/processed/train.parquet` — 6946 rows / 144 pairs / 31 usable features
- `data/processed/test.parquet` — 1817 rows / 144 pairs
</canonical_refs>

<deferred>
## Deferred Ideas

- Alternative RL algorithms (DQN, A2C, SAC) — possible Phase 7 ablation
- Continuous action space (position sizing [-1, 1]) — adds complexity without evidence of benefit
- Autoencoder Pattern 2 (recon_error as feature) — Phase 7 ablation
- VAE variant — overkill for anomaly detection at 6k samples
- Multi-pair VecEnv training — more efficient but complex; iterate pairs for now
- Curriculum learning (easy pairs first) — interesting but out of scope
</deferred>

---

*Phase: 06-rl-and-autoencoder*
*Context gathered: 2026-04-06 via 3 parallel research agents*
