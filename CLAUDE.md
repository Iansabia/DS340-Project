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

This project uses **GSD** for macro project management (phases, roadmaps, execution tracking) and **Superpowers** for code quality enforcement (brainstorming, TDD, subagent-driven development, code review).

### Two-Tier Quality System

**Main session (full Superpowers via plugin):**
- Brainstorming skill invoked before design decisions
- Subagent-driven development for complex implementations
- Full code review ceremony after completing work
- All Superpowers skills available via the Skill tool

**GSD sub-agents (Superpowers discipline via AGENTS.md):**
- `AGENTS.md` is loaded automatically into every sub-agent
- Contains the core Superpowers rules: TDD Iron Law, verification-before-completion, self-review, report format
- Sub-agents don't have the Skill tool, but they follow the same behavioral discipline
- This ensures GSD executor agents write tested, verified, self-reviewed code

### When to Use Each

**GSD owns the project lifecycle:**
- `/gsd:new-project` to initialize the roadmap
- `/gsd:plan-phase N` to create detailed phase plans
- `/gsd:execute-phase N` to run phase execution
- `/gsd:progress` to check status and route to next action
- `/gsd:debug` for systematic debugging sessions
- `/gsd:quick` for small ad-hoc tasks

**Superpowers owns code quality within GSD phases:**
- **Brainstorming skill** — invoke in main session before any design decision (model architecture, reward shaping, matching pipeline strategy). Do not write code until design is approved.
- **TDD skill** — enforced in both main session (via Skill tool) and sub-agents (via AGENTS.md). RED-GREEN-REFACTOR for data pipeline, matching, feature engineering, evaluation, and model code.
- **Subagent-driven development** — for high-risk phases (matching pipeline, PPO + autoencoder), consider executing interactively in main session using this skill instead of `/gsd:execute-phase`, to get the full two-stage review (spec compliance + code quality).
- **Code review skills** — invoke in main session after completing any phase.

### When to Skip Superpowers Ceremony
- Quick data exploration scripts in `notebooks/`
- One-off plotting/visualization
- Running existing models with different hyperparameters
- Writing the paper or slides
- Configuration files and simple config changes

### Phase Execution Pattern

When executing a GSD phase that involves writing code:

1. **GSD creates the plan** (`/gsd:plan-phase N`)
2. **Before coding, invoke Superpowers brainstorming** in main session if the task involves design decisions
3. **Execute the phase** — either via `/gsd:execute-phase N` (sub-agents follow AGENTS.md discipline) or interactively in main session with full Superpowers for high-risk phases
4. **Run Superpowers code review** in main session on completed work
5. **GSD marks the phase complete** and updates STATE.md

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
