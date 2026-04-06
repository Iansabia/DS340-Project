# Requirements: Kalshi vs. Polymarket Price Discrepancies

**Defined:** 2026-04-01
**Core Value:** Empirically answer whether model complexity improves cross-platform prediction market arbitrage detection

## v1 Requirements

### Data Pipeline

- [x] **DATA-01**: Kalshi API adapter that handles live/historical endpoint split via `/historical/cutoff`
- [x] **DATA-02**: Polymarket API adapter that queries Gamma for metadata, CLOB for prices, Data API for trade records
- [x] **DATA-03**: Polymarket price history reconstruction from trade records for resolved markets
- [x] **DATA-04**: Rate limiting and local caching for both platform APIs
- [x] **DATA-05**: Automated retry with exponential backoff for transient API failures
- [x] **DATA-06**: Raw data storage with timestamps in `data/raw/`

### Market Matching

- [x] **MATCH-01**: Keyword-based first-pass candidate matching across platforms
- [x] **MATCH-02**: Sentence-transformer semantic similarity scoring for fuzzy matching
- [ ] **MATCH-03**: Manual curation interface for reviewing and accepting/rejecting matched pairs
- [x] **MATCH-04**: Match confidence scoring per pair
- [ ] **MATCH-05**: Settlement criteria comparison documentation for each matched pair

### Feature Engineering

- [x] **FEAT-01**: Time-aligned hourly feature vectors (spread, volume, bid-ask spread, price velocity)
- [x] **FEAT-02**: Temporal train/test split enforced (no look-ahead bias)
- [x] **FEAT-03**: Low-liquidity market filtering (remove markets with <10 trades)
- [x] **FEAT-04**: Output format compatible with PyTorch Forecasting TimeSeriesDataSet
- [x] **FEAT-05**: Processed dataset saved to `data/processed/`

### Models — Tier 1 (Regression Baselines)

- [x] **MOD-01**: Linear Regression trained on spread prediction
- [x] **MOD-02**: XGBoost trained on spread prediction
- [x] **MOD-03**: Naive baseline (spread always closes fully)
- [x] **MOD-04**: Volume baseline (higher-volume platform is always correct)

### Models — Tier 2 (Time Series)

- [x] **MOD-05**: GRU trained on spread prediction with hourly sequences
- [x] **MOD-06**: LSTM trained on spread prediction with hourly sequences
- [x] **MOD-07**: TFT via PyTorch Forecasting (droppable if timeline tight)

### Models — Tier 3 (RL)

- [x] **MOD-08**: PPO agent acting directly on raw microstructure features
- [x] **MOD-09**: Autoencoder trained on normal spread behavior for anomaly detection
- [x] **MOD-10**: PPO agent with autoencoder signal filter (acts on flagged opportunities)

### Evaluation

- [x] **EVAL-01**: Regression metrics computed for all models (RMSE, MAE, directional accuracy)
- [x] **EVAL-02**: Profit simulation for all models (P&L, win rate, Sharpe ratio)
- [ ] **EVAL-03**: SHAP interpretability analysis on best-performing models
- [ ] **EVAL-04**: Bootstrap confidence intervals on key metrics

### Experiments

- [x] **EXP-01**: Experiment 1 — Complexity-vs-performance comparison across all tiers (centerpiece)
- [x] **EXP-02**: Experiment 2 — Historical window length ablation (6h, 24h, 72h, 7d)
- [x] **EXP-03**: Experiment 3 — Minimum spread threshold ablation (no min, >2pp, >5pp, >10pp)
- [ ] **EXP-04**: Transaction cost sensitivity analysis

### Deliverables

- [ ] **DEL-01**: Final paper documenting methodology, experiments, and findings
- [ ] **DEL-02**: Lightning talk slides

## v2 Requirements

(Not applicable — single-submission academic project)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Live trading / real-money execution | Historical backtesting only — academic project |
| External event features (news, sentiment) | Microstructure-only by design to isolate signal |
| Pretrained models | All models trained from scratch per project rules |
| Third-party data aggregators | Direct API ingestion only |
| Transaction cost modeling in simulation | Acknowledged via sensitivity analysis but not modeled in P&L |
| Web dashboard or mobile app | Research project — scripts and notebooks only |
| Model ensembling | Would obscure complexity comparison — each model evaluated independently |
| Political/entertainment markets | Insufficient cross-platform overlap for matched pairs |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Complete |
| DATA-02 | Phase 1 | Complete |
| DATA-03 | Phase 1 | Complete |
| DATA-04 | Phase 1 | Complete |
| DATA-05 | Phase 1 | Complete |
| DATA-06 | Phase 1 | Complete |
| MATCH-01 | Phase 2 | Complete |
| MATCH-02 | Phase 2 | Complete |
| MATCH-03 | Phase 2 | Pending |
| MATCH-04 | Phase 2 | Complete |
| MATCH-05 | Phase 2 | Pending |
| FEAT-01 | Phase 3 | Complete |
| FEAT-02 | Phase 3 | Complete |
| FEAT-03 | Phase 3 | Complete |
| FEAT-04 | Phase 3 | Complete |
| FEAT-05 | Phase 3 | Complete |
| MOD-01 | Phase 4 | Complete |
| MOD-02 | Phase 4 | Complete |
| MOD-03 | Phase 4 | Complete |
| MOD-04 | Phase 4 | Complete |
| EVAL-01 | Phase 4 | Complete |
| EVAL-02 | Phase 4 | Complete |
| MOD-05 | Phase 5 | Complete |
| MOD-06 | Phase 5 | Complete |
| MOD-07 | Phase 5 | Complete |
| MOD-08 | Phase 6 | Complete |
| MOD-09 | Phase 6 | Complete |
| MOD-10 | Phase 6 | Complete |
| EVAL-03 | Phase 7 | Pending |
| EVAL-04 | Phase 7 | Pending |
| EXP-01 | Phase 7 | Complete |
| EXP-02 | Phase 7 | Complete |
| EXP-03 | Phase 7 | Complete |
| EXP-04 | Phase 7 | Pending |
| DEL-01 | Phase 8 | Pending |
| DEL-02 | Phase 8 | Pending |

**Coverage:**
- v1 requirements: 36 total
- Mapped to phases: 36
- Unmapped: 0

---
*Requirements defined: 2026-04-01*
*Last updated: 2026-04-01 after roadmap creation (EVAL-01/02 moved to Phase 4)*
