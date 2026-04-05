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
- 169 quality-filtered candidates → 77 auto-curated pairs

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
- 45 tests passing (includes proper time-series Sharpe calculation)

---

## Final Dataset

| Metric | Value |
|---|---|
| Aligned pairs | 12 |
| Total observations | 1,462 (4-hour bars, all with valid spread) |
| Train rows | 1,164 |
| Test rows | 298 (~48 days of trading) |
| Features per bar | 39 (31 raw + 6 derived + 2 timeseries) |

### Pair Breakdown
- CPI/Core Inflation: 8 pairs
- US Unemployment (U-3): 4 pairs

---

## Results: Tier 1 Baseline Comparison

Evaluated on held-out test set (20% of data, temporally split per-pair):

| Model | RMSE ↓ | Dir Acc ↑ | Win Rate | Raw Sharpe ↑ |
|---|---:|---:|---:|---:|
| **XGBoost** | **0.1749** | **0.7053** | **0.702** | **0.569** |
| **Linear Regression** | 0.1808 | 0.6667 | 0.691 | 0.531 |
| Volume Baseline | 0.2158 | 0.6246 | 0.628 | 0.201 |
| Naive Baseline | 0.2346 | 0.6246 | 0.628 | 0.202 |

**Raw Sharpe** = unannualized per-trade mean/std ratio. Values 0.5–1.0 indicate strong signal quality in quant finance. The naive baseline (predict "spread closes fully") represents pure mean-reversion.

### Key Findings

1. **XGBoost Raw Sharpe: 0.57** — a legitimate, strong signal (2.8× the naive baseline at 0.20)
2. **Directional accuracy 70.5%** — model correctly predicts spread movement direction
3. **Both regression models crush naive baselines** by ~2.8× on risk-adjusted return
4. **XGBoost lift over mean reversion: +8%** directional accuracy, +180% Sharpe

### Honest Caveats

- **Test period is short** (~48 days, 286 bars). Annualized Sharpe numbers would be inflated by extrapolation and are not reported.
- **Quality filter selects mean-reverting pairs** (mean |spread| < 0.30). The naive "spread closes" baseline already captures much of the signal — XGBoost adds a modest but real improvement on top.
- **The spread itself is the strongest feature** (correlation -0.43 with next-bar change). This is expected and correct — mean reversion is a real prediction market phenomenon.
- The genuine contribution is that XGBoost finds additional signal beyond simple mean reversion (2.8× Sharpe lift).

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

- **Tier 1** (regression): 70.5% directional accuracy, 2.8× Sharpe over naive — strong baseline
- **Tier 2** (GRU, LSTM, TFT): Tests whether temporal structure improves on tabular regression
- **Tier 3** (PPO, PPO+Autoencoder): Tests whether RL/anomaly detection adds value
- **Naive baselines**: Lower bound at 62.5% directional accuracy

If simpler models win or match complex ones, that IS the empirical answer to KG's question.

---

## Files in This Folder

- `README.md` — This document
- `linear_regression.json` — LR model results
- `xgboost.json` — XGBoost model results (Raw Sharpe 0.57, Dir Acc 70.5%)
- `naive_spread_closes.json` — Naive baseline results
- `volume_higher_volume_correct.json` — Volume baseline results
- `data_quality_report.json` — Phase 2.1 data quality report
