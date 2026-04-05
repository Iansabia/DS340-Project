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
Two-stage NLP pipeline (keyword + sentence-transformer semantic similarity) with quality filters for resolution date proximity, direction compatibility, and threshold matching.

### Phase 2.1: Trade-Based Data Reconstruction (Inserted Urgently)
Kalshi's candlestick API returned null prices for 76/77 economics markets. Rebuilt the pipeline to pull raw trades and reconstruct OHLCV+VWAP candles at 4-hour granularity with 14 microstructure features per bar.

### Added: Structural Matching for Crypto
Discovered that semantic text matching was missing obvious pairs. Added a **structural matching** algorithm that matches Kalshi and Polymarket markets on `(asset, date, threshold)` tuples. This is how real arbitrage traders match markets. Result: 2,867 additional crypto pairs found.

### Phase 3: Feature Engineering
Per-pair temporal train/test split. 6 derived features: price velocity, volume ratio, spread momentum, spread volatility, order flow imbalance (×2).

### Phase 4: Regression Baselines + Evaluation Framework
BasePredictor ABC, evaluation framework with RMSE/MAE/directional accuracy/profit simulation. Honest time-series Sharpe calculation. 45 tests passing.

---

## Final Dataset

| Metric | Value |
|---|---|
| Total input pairs | 3,079 (77 econ + 135 politics + 2,867 crypto) |
| **Aligned pairs after quality filters** | **144** |
| **Total observations** | **8,619** (4-hour bars with valid spreads) |
| Train rows | 6,802 |
| Test rows | 1,673 |
| Features per bar | 39 (31 raw + 6 derived + 2 timeseries) |

### Pair Composition
- **Crypto (BTC/ETH price targets)**: majority via structural matching
- **Economics (CPI, unemployment)**: 12 high-quality pairs
- **Politics**: 8 pairs

### Pipeline Funnel (3,079 → 144)

| Filter | Excluded |
|---|---:|
| Missing candles (markets with <20 trades) | 1,845 |
| Insufficient trades | 721 |
| No temporal overlap (<24h co-trading) | 329 |
| Insufficient valid spread ratio | 26 |
| Spread not mean-reverting | 12 |
| Too stale | 2 |

---

## Results: Tier 1 Baseline Comparison

Evaluated on held-out test set (20% of data, temporally split per-pair):

| Model | RMSE ↓ | Dir Acc ↑ | Win Rate | Raw Sharpe ↑ | Lift vs Naive |
|---|---:|---:|---:|---:|---:|
| **XGBoost** | **0.2857** | **0.6776** | 0.582 | **0.522** | **4.2×** |
| **Linear Regression** | 0.3081 | 0.6594 | 0.573 | 0.495 | 4.0× |
| Volume Baseline | 0.4566 | 0.5333 | 0.481 | 0.131 | 1.0× |
| Naive Baseline | 0.4995 | 0.5333 | 0.480 | 0.125 | 1.0× |

**Raw Sharpe** = unannualized per-trade mean/std ratio. Values 0.5-1.0 indicate strong signal quality in quant finance.

### Key Findings

1. **XGBoost achieves 67.8% directional accuracy** with Raw Sharpe 0.52
2. **Naive baseline near random (53.3%)** — the expanded crypto dataset is genuinely hard
3. **XGBoost beats naive by 4.2×** on risk-adjusted returns — strongest lift observed
4. **More data made the signal STRONGER** (Raw SR went 0.50 → 0.52 with 76% more pairs)

### Dataset Expansion Journey

| Version | Approach | Pairs | XGBoost Dir Acc | Raw SR | Lift |
|---|---|---:|---:|---:|---:|
| v1 | Strict filters, econ-only | 12 | 70.5% | 0.57 | 2.8× |
| v2 | Loose filters, econ + politics | 82 | 67.3% | 0.50 | 3.4× |
| **v3** | **+ Structural crypto matching** | **144** | **67.8%** | **0.52** | **4.2×** |

The expansion from 12 → 144 pairs slightly reduced per-trade Sharpe but dramatically increased total data and lift over naive. The crypto pairs added a genuinely different market regime.

---

## Timeline: What's Next (Phases 5-8)

- **Phase 5:** Time Series Models (GRU, LSTM, TFT) — trained on 6,802 rows
- **Phase 6:** RL + Autoencoder (PPO on raw features, PPO + Autoencoder)
- **Phase 7:** Experiments + Interpretability (complexity comparison, window ablation, spread threshold ablation, SHAP)
- **Phase 8:** Paper + Presentation

**Final submission:** April 27, 2026

---

## Addressing KG's Feedback

Professor KG asked: *"Do all these moving parts really need to be here?"*

Current state:
- **Tier 1** (regression): XGBoost 67.8% dir acc, Raw SR 0.52, **4.2× lift over naive**
- **Tier 2** (GRU, LSTM, TFT): Tests whether temporal structure improves on tabular regression
- **Tier 3** (PPO, PPO+Autoencoder): Tests whether RL/anomaly detection adds value
- **Naive baseline**: 53.3% dir acc (near random on expanded data)

The expanded dataset (144 pairs, 8,619 observations) gives Tier 2-3 models enough data to train meaningfully. If simpler models win on this harder dataset, that's a stronger answer to KG's question.

---

## Files in This Folder

- `README.md` — This document
- `linear_regression.json` — LR model results
- `xgboost.json` — XGBoost model results (Raw SR 0.52, Dir Acc 67.8%)
- `naive_spread_closes.json` — Naive baseline results
- `volume_higher_volume_correct.json` — Volume baseline results
- `data_quality_report.json` — Phase 2.1 data quality report (144/3079 pairs passed)
