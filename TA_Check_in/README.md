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
API adapters for Kalshi and Polymarket. Discovered 2,069,771 Kalshi markets and 292,500 Polymarket markets. 45 tests passing.

### Phase 2: Market Matching
Two-stage NLP pipeline (keyword + sentence-transformer semantic similarity) with quality filters for resolution date proximity, direction compatibility, and threshold matching. Covers Economics, Crypto, and Politics categories.

### Phase 2.1: Trade-Based Data Reconstruction (Inserted Urgently)
Kalshi's candlestick API returned null prices for 76/77 economics markets. Rebuilt the entire pipeline to pull raw trades from both platforms and reconstruct OHLCV+VWAP candles from trade records at 4-hour granularity with 14 microstructure features per bar.

### Phase 3: Feature Engineering
Per-pair temporal train/test split. 6 derived features: price velocity, volume ratio, spread momentum, spread volatility, order flow imbalance (×2).

### Phase 4: Regression Baselines + Evaluation Framework
BasePredictor ABC, evaluation framework with RMSE/MAE/directional accuracy/profit simulation. Honest time-series Sharpe calculation. 45 tests passing.

---

## Final Dataset

| Metric | Value |
|---|---|
| Input matched pairs | 212 (77 economics + 135 politics, semantic matching) |
| **Aligned pairs after quality filters** | **82** |
| **Total observations** | **7,021** (4-hour bars with valid spreads) |
| Train rows | 5,498 |
| Test rows | 1,359 |
| Features per bar | 39 (31 raw + 6 derived + 2 timeseries) |

### Pipeline Funnel (212 → 82)

| Filter | Excluded |
|---|---:|
| Insufficient trades (<10 on one platform) | 56 |
| Spread not mean-reverting (|spread| > 0.60) | 29 |
| No temporal overlap (<24h co-trading) | 28 |
| Missing candle files | 20 |
| Too stale (>90%) | 7 |
| Insufficient valid spread ratio | 20 |

---

## Results: Tier 1 Baseline Comparison

Evaluated on held-out test set (20% of data, temporally split per-pair):

| Model | RMSE ↓ | Dir Acc ↑ | Win Rate | Raw Sharpe ↑ | Lift vs Naive |
|---|---:|---:|---:|---:|---:|
| **XGBoost** | **0.2791** | **0.6734** | **0.572** | **0.501** | **3.4×** |
| **Linear Regression** | 0.3085 | 0.6462 | 0.549 | 0.480 | 3.3× |
| Volume Baseline | 0.4372 | 0.5464 | 0.487 | 0.153 | 1.0× |
| Naive Baseline | 0.4894 | 0.5464 | 0.486 | 0.146 | 1.0× |

**Raw Sharpe** = unannualized per-trade mean/std ratio. Values 0.5–1.0 indicate strong signal quality in quant finance.

### Key Findings

1. **XGBoost beats naive baselines by 3.4×** on risk-adjusted returns (Raw SR 0.50 vs 0.15)
2. **Naive baseline directional accuracy is 54.6%** (only slightly above random) — this is a HARDER, more realistic dataset than pure mean-reverting pairs
3. **XGBoost finds real ML signal** — not just mean reversion (+13% over naive vs random)
4. **Linear Regression is close to XGBoost** (0.48 vs 0.50 Raw SR) — tabular signal is well-extracted by both models

### Dataset Construction: Design Choices

After initial experiments, we loosened quality filters based on real arbitrage trader insight:
- Original strict filter (mean |spread| < 0.30) selected for easily-predictable pairs → inflated baseline performance
- Loosened filter (|spread| < 0.60) includes persistent-spread markets → realistic arbitrage scenarios
- Result: 82 pairs with 7,021 observations vs 12 pairs with 1,462 observations
- ML models show stronger lift on harder data (3.4× vs 2.8× over naive)

---

## Timeline: What's Next (Phases 5-8)

- **Phase 5:** Time Series Models (GRU, LSTM, TFT) — will train on 5,498 rows
- **Phase 6:** RL + Autoencoder (PPO on raw features, PPO + Autoencoder)
- **Phase 7:** Experiments + Interpretability (complexity comparison, window ablation, spread threshold ablation, SHAP)
- **Phase 8:** Paper + Presentation

**Final submission:** April 27, 2026

---

## Addressing KG's Feedback

Professor KG asked: *"Do all these moving parts really need to be here?"*

Current data:
- **Tier 1** (regression): XGBoost 67.3% dir acc, Raw SR 0.50, **3.4× lift over naive**
- **Tier 2** (GRU, LSTM, TFT): Tests whether temporal structure improves on tabular regression
- **Tier 3** (PPO, PPO+Autoencoder): Tests whether RL/anomaly detection adds value
- **Naive baseline**: 54.6% dir acc (near random — this is hard data)

The expanded dataset makes the comparison more meaningful. If simple models win, that's still a valid answer to KG's question, but now on realistic data.

---

## Files in This Folder

- `README.md` — This document
- `linear_regression.json` — LR model results
- `xgboost.json` — XGBoost model results (Raw SR 0.50, Dir Acc 67.3%)
- `naive_spread_closes.json` — Naive baseline results
- `volume_higher_volume_correct.json` — Volume baseline results
- `data_quality_report.json` — Phase 2.1 data quality report
