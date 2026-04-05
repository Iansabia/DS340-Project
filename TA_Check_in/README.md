# DS340 TA Check-in — Kalshi vs. Polymarket Price Discrepancies

**Team:** Ian Sabia (U33871576), Alvin Jang (U64760665)
**Date:** April 5, 2026
**Project progress:** 4 of 8 phases complete (50%)

---

## Research Question

Does increasing model complexity (RL, anomaly detection) improve cross-platform arbitrage detection over simpler regression approaches, and if so, when is that complexity justified?

---

## What We Built (Phases 1-4)

### Phase 1: Data Pipeline
- API adapters for Kalshi and Polymarket
- Discovered 2,069,771 Kalshi markets and 292,500 Polymarket markets
- 45 tests, all passing

### Phase 2: Market Matching
- Two-stage NLP pipeline: keyword matching + sentence-transformer semantic similarity
- Quality filters: resolution date proximity (30d), direction compatibility, threshold matching
- Scope: Economics + Crypto categories
- Result: 169 quality-filtered candidates → 77 auto-curated pairs

### Phase 2.1: Trade-Based Data Reconstruction (Inserted Urgently)
- **Problem discovered:** Kalshi's candlestick API returns null prices for 76/77 economics markets (thinly-traded)
- **Solution:** Rebuilt pipeline to pull raw trades from both platforms and reconstruct OHLCV+VWAP candles from trade records
- 4-hour granularity (optimal for thin markets per Lopez de Prado)
- VWAP as primary price, 14 microstructure features per bar
- Cross-platform alignment with forward-fill + 24h staleness decay
- Strict quality filters: min 20 trades, staleness <80%, mean |spread| <0.30, temporal overlap ≥48h, valid spread ratio ≥20%

### Phase 3: Feature Engineering
- Per-pair temporal train/test split (no look-ahead bias)
- 6 derived features: price velocity, volume ratio, spread momentum, spread volatility, order flow imbalance (×2)
- PyTorch Forecasting TimeSeriesDataSet compatibility

### Phase 4: Regression Baselines + Evaluation Framework
- BasePredictor ABC for uniform model interface
- Evaluation: RMSE, MAE, directional accuracy, profit simulation (P&L, win rate, Sharpe ratio)
- 4 models: Linear Regression, XGBoost, Naive (spread closes), Volume (higher volume correct)
- 44 tests passing

---

## Final Dataset

| Metric | Value |
|---|---|
| Aligned pairs | **12** (down from 17 after stricter filters) |
| Total observations | 1,462 (4-hour bars, all with valid spread) |
| Train rows | 1,164 |
| Test rows | 298 |
| Features per bar | 39 (31 raw + 6 derived + 2 timeseries) |

### Pair Breakdown by Topic

| Topic | Pairs |
|---|---|
| CPI/Core Inflation | 8 |
| US Unemployment (U-3) | 4 |

### Data Quality Journey

| Iteration | Pairs | Issue |
|---|---:|---|
| Initial alignment | 17 | 5 pairs had NaN spreads (no temporal overlap) |
| After temporal overlap filter | 12 | All pairs have real co-trading periods |

**Exclusion breakdown (65 pairs dropped):**
- 35 pairs: spread not mean-reverting (|spread| > 0.30)
- 19 pairs: insufficient trades (<20 on one platform)
- 5 pairs: no temporal overlap (platforms traded at different times)
- 3 pairs: missing candle data
- 2 pairs: too stale
- 1 pair: insufficient valid spread ratio

---

## Results: Tier 1 Baseline Comparison

Evaluated on held-out test set (20% of data, temporally split per-pair):

| Model | RMSE ↓ | MAE ↓ | Dir Acc ↑ | Sharpe ↑ | Win Rate |
|---|---:|---:|---:|---:|---:|
| **XGBoost** | **0.1749** | 0.1213 | **0.7053** | **9.03** | **0.702** |
| **Linear Regression** | 0.1808 | 0.1234 | 0.6667 | 8.43 | 0.691 |
| Volume Baseline | 0.2158 | 0.1557 | 0.6246 | 3.19 | 0.628 |
| Naive Baseline | 0.2346 | 0.1740 | 0.6246 | 3.20 | 0.628 |

### Key Findings

1. **XGBoost achieves 70.5% directional accuracy and Sharpe 9.03** on held-out data
2. **Both regression models beat naive baselines** by ~2.8x on Sharpe ratio
3. **Results held up on stricter data** (from 17 → 12 pairs), ruling out forward-fill artifacts
4. Regression models are genuinely competitive — important for the complexity analysis

---

## Timeline: What's Next (Phases 5-8)

- **Phase 5:** Time Series Models (GRU, LSTM, TFT)
- **Phase 6:** RL + Autoencoder (PPO on raw features, PPO + Autoencoder)
- **Phase 7:** Experiments + Interpretability (complexity comparison, window ablation, spread threshold ablation, SHAP)
- **Phase 8:** Paper + Presentation

**Final submission:** April 27, 2026

---

## Addressing KG's Feedback

Professor KG asked: *"Do all these moving parts really need to be here?"*

Our response is built into the experimental design: we test all tiers systematically.

- **Tier 1** (regression): 70.5% directional accuracy, Sharpe 9.03 — strong baseline
- **Tier 2** (GRU, LSTM, TFT): Tests whether temporal structure helps
- **Tier 3** (PPO, PPO+Autoencoder): Tests whether RL/anomaly detection adds value
- **Naive baselines**: Lower bound at Sharpe ~3.2

If simpler models win, that's the empirical answer to KG's question.

---

## Files in This Folder

- `README.md` — This document
- `linear_regression.json` — LR model results
- `xgboost.json` — XGBoost model results
- `naive_spread_closes.json` — Naive baseline results
- `volume_higher_volume_correct.json` — Volume baseline results
- `data_quality_report.json` — Phase 2.1 data quality report (12/77 pairs passed all filters)
