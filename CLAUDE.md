# DS340 Final Project: Kalshi vs. Polymarket Price Discrepancies

## Team
- Ian Sabia (U33871576)
- Alvin Jang (U64760665)

## Project Overview
Cross-platform prediction market arbitrage using ML/RL. We detect price discrepancies between Kalshi and Polymarket on matched contracts, then learn optimal trading policies. Leading architecture: PPO agent with autoencoder anomaly detector as signal filter, benchmarked against linear regression, XGBoost, GRU, LSTM, and TFT baselines.

## Deadline
- **TA Check-in:** April 4, 2026
- **Final Submission:** April 27, 2026

## Workflow: GSD + Superpowers Integration

This project uses **GSD** for macro project management (phases, roadmaps, execution tracking) and **Superpowers** for code quality within each phase (brainstorming, TDD, subagent-driven development, code review).

### When to Use Each

**GSD owns the project lifecycle:**
- `/gsd:new-project` to initialize the roadmap
- `/gsd:plan-phase N` to create detailed phase plans
- `/gsd:execute-phase N` to run phase execution
- `/gsd:progress` to check status and route to next action
- `/gsd:debug` for systematic debugging sessions
- `/gsd:quick` for small ad-hoc tasks

**Superpowers owns code quality within GSD phases:**
- **Brainstorming skill** — invoke before any design decision (model architecture, reward shaping, matching pipeline strategy). Do not write code until design is approved.
- **TDD skill** — enforce RED-GREEN-REFACTOR for the matching pipeline, data ingestion, feature engineering, and evaluation harness. These are foundational and bugs here propagate everywhere.
- **Subagent-driven development** — use when implementing models (GRU, LSTM, TFT, PPO, autoencoder). Each model gets a fresh subagent with isolated context, plus two reviewer subagents (spec compliance + code quality).
- **Code review skills** — use after completing any data pipeline or model implementation before merging.

### When to Skip Superpowers Ceremony
- Quick data exploration scripts
- One-off plotting/visualization
- Running existing models with different hyperparameters
- Writing the paper or slides
- Simple config changes

### Phase Execution Pattern

When executing a GSD phase that involves writing code:

1. **GSD creates the plan** (`/gsd:plan-phase N`)
2. **Before coding, invoke Superpowers brainstorming** if the task involves design decisions
3. **Write tests first** (Superpowers TDD) for data pipeline and core logic
4. **Implement via subagent-driven development** for model code
5. **Run Superpowers code review** on completed work
6. **GSD marks the phase complete** and updates STATE.md

## Technical Context

### Data Sources
- **Kalshi API** (`https://api.elections.kalshi.com/trade-api/v2`): Public endpoints, no auth needed. Hourly candlesticks via `period_interval=60`. Historical data for markets settled >3 months ago requires `/historical/` endpoints.
- **Polymarket API**: Three separate APIs (Gamma for metadata, CLOB for prices/orderbook, Data API for trades). Rate limits are per-10-seconds, not per-hour. CRITICAL: `/prices-history` returns empty data for resolved markets — reconstruct from trade records via Data API instead.

### Known Data Pipeline Gotchas
- Polymarket uses long numeric token IDs, not slugs. Query Gamma API first to get `clobTokenIds`.
- Kalshi live/historical data split at ~3 month cutoff. Call `/historical/cutoff` to determine which endpoint to use.
- Contract matching across platforms requires NLP — no shared identifiers. Settlement criteria differ between platforms.
- Filter out low-liquidity markets (some Kalshi hourly contracts have <10 trades).
- Kalshi candlestick OHLC fields can be null when no trades occurred in a period.

### Models & Architecture
- **Baselines:** Linear Regression, XGBoost
- **Time Series:** GRU (preferred recurrent), LSTM, TFT (via PyTorch Forecasting)
- **RL:** PPO with autoencoder anomaly detector as signal filter
- **Anomaly Detection:** Autoencoder trained on normal spread behavior, flags high reconstruction error
- All models trained from scratch on matched-pairs dataset. No pretrained models.

### Stack
- Python 3.12 with venv at `.venv/`
- PyTorch (models), PyTorch Forecasting (TFT), XGBoost/LightGBM (baselines)
- scikit-learn (evaluation), sentence-transformers (market matching)
- SHAP (interpretability), requests/pandas/numpy (data)

### Experiments
1. **Architecture Comparison:** All models on same dataset, same eval protocol
2. **Historical Window Length:** 6h, 24h, 72h, 7d lookback windows
3. **Minimum Spread Threshold:** No min, >2pp, >5pp, >10pp

### Evaluation
- Regression: RMSE, MAE, directional accuracy
- Trading: Simulated P&L, win rate, Sharpe ratio
- Baselines: Naive (spread always closes) and volume (higher-volume platform correct)
- Train on earlier data, test on recent data to avoid look-ahead bias

## Code Organization Conventions
- Data ingestion scripts in `src/data/`
- Market matching pipeline in `src/matching/`
- Feature engineering in `src/features/`
- Models in `src/models/` (one file per model class)
- Experiments in `experiments/`
- Tests in `tests/` mirroring `src/` structure
- Notebooks for exploration only in `notebooks/`
- Raw data in `data/raw/`, processed in `data/processed/`

## Important Notes
- PPO will likely underperform baselines due to small dataset size. This is expected and should be framed as a finding, not a failure.
- Transaction costs (Kalshi fees, Polymarket gas) should be acknowledged in evaluation even if not modeled.
- The novelty is in the **application domain** (cross-platform prediction market arbitrage), not the architecture (PPO + autoencoder has been published for equities).
- Settlement divergence between platforms is a real risk — document it.
