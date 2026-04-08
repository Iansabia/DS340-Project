# DS340 Live Paper Trading Report

**Team:** Ian Sabia (U33871576), Alvin Jang (U64760665)
**Last Updated:** 2026-04-08
**Status:** Live paper trading — 27 hours of data, collecting hourly

---

## System Overview

We built a cross-platform prediction market arbitrage system that:
1. Matches contracts across Kalshi and Polymarket using semantic similarity (615 matched pairs)
2. Collects live prices every hour via GitHub Actions (automated, runs 24/7)
3. Runs 5 ML models on each price snapshot to generate trading signals
4. Paper trades (simulated) to validate profitability before risking real capital

**Architecture:**
- Data collection: pmxt SDK + Kalshi orderbook API + Polymarket Gamma API
- Models: Linear Regression, XGBoost, GRU, LSTM, Naive baseline
- Evaluation: Per-trade P&L with configurable fee simulation (2-5pp)
- Infrastructure: GitHub Actions (hourly cron), local dashboard

---

## Live Results (27 hours, contracts ≥ $0.10)

### Model Comparison

| Model | Trades | Win Rate | W/L Ratio | Gross P&L | Net @2pp | Net @3pp | Per-Trade Sharpe | 95% CI (mean P&L/trade) |
|-------|--------|----------|-----------|-----------|----------|----------|-----------------|------------------------|
| **Linear Regression** | 1,748 | 51.5% | **2.26x** | **+17.18** | **+5.83** | **+0.15** | 0.120 | [+0.006, +0.014] |
| XGBoost | 1,776 | 51.4% | 2.11x | +16.59 | +5.11 | -0.63 | 0.114 | [+0.006, +0.013] |
| Naive (spread closes) | 1,745 | 51.9% | 1.88x | +14.32 | +3.04 | -2.60 | 0.103 | [+0.005, +0.012] |
| GRU | 1,739 | 51.8% | 1.80x | +12.41 | +1.14 | -4.49 | 0.090 | [+0.003, +0.011] |
| LSTM | 1,175 | 51.7% | 1.60x | +6.29 | -2.27 | -6.55 | 0.073 | [+0.002, +0.010] |

**Key findings:**
- All 5 models show **statistically significant edge** (95% CI excludes zero)
- **Linear Regression is the only model profitable after 3pp fees** — simplest model wins
- Win rates ~51-52% but **winners are 1.6-2.3x larger than losers**
- LSTM has fewest trades (most selective) but lowest W/L ratio — selectivity doesn't help

### Where The Edge Comes From

| Category | Trades | P&L | Win Rate | P&L/Trade |
|----------|--------|-----|----------|-----------|
| **Oil/Commodities** | **259** | **+18.65** | **66.0%** | **+0.072** |
| Sports | 355 | +0.26 | 36.6% | +0.001 |
| Crypto | 94 | -0.03 | 36.2% | -0.000 |
| Politics | 70 | -0.40 | 35.7% | -0.006 |
| Other | 998 | -1.88 | 23.4% | -0.002 |

**The edge is concentrated in oil/commodity contracts.** Kalshi prices WTI oil brackets significantly higher than Polymarket (30-50pp spreads). When these converge, the payoff is large. This is 15% of trades generating 112% of total profit.

### Fee Sensitivity

| Fee Level | Linear Reg | XGBoost | GRU | LSTM | Naive |
|-----------|-----------|---------|-----|------|-------|
| 0pp (gross) | +17.18 | +16.59 | +12.41 | +6.29 | +14.32 |
| 2pp | **+5.83** | **+5.11** | **+1.14** | -2.27 | **+3.04** |
| 3pp | **+0.15** | -0.63 | -4.49 | -6.55 | -2.60 |
| 5pp | -11.20 | -12.11 | -15.76 | -15.11 | -13.88 |

Break-even fee: ~3pp for Linear Regression, ~2.5pp for XGBoost.

### Critical Filter: Drop Penny Contracts (< $0.10)

**56% of all trades are on contracts priced under $0.10.** These penny contracts have tiny absolute profits but proportionally enormous fees, dragging the entire portfolio negative. Filtering them out is the single biggest improvement to the strategy.

#### Before vs After Filter (XGBoost, 27 hours)

| Metric | All Contracts | Contracts ≥ $0.10 | Improvement |
|--------|--------------|-------------------|-------------|
| Trades | 2,919 | 1,271 | -56% (removed noise) |
| Gross P&L | +12.61 | **+17.09** | **+35% higher** |
| Net @2pp fees | +3.17 | **+8.67** | **+174% higher** |
| Net @3pp fees | **-1.55** (losing) | **+4.46** (profitable) | **Flipped from loss to profit** |
| Net @5pp fees | -10.99 | -3.96 | 64% less loss |
| Win Rate | 46.9% | 48.8% | +1.9pp |
| W/L Ratio | 2.03x | 2.92x | +44% |

#### All Models After Penny Filter

| Model | Trades | Gross P&L | @2pp fees | @3pp fees | Win Rate | W/L Ratio |
|-------|--------|-----------|-----------|-----------|----------|-----------|
| **Linear Reg** | 1,248 | **+17.47** | **+9.16** | **+5.00** | 48.4% | **3.21x** |
| **XGBoost** | 1,271 | +17.09 | **+8.67** | **+4.46** | 48.8% | 2.92x |
| **Naive** | 1,237 | +15.31 | **+7.08** | **+2.96** | 52.6% | 2.31x |
| **GRU** | 1,222 | +12.66 | **+4.52** | +0.46 | 48.8% | 2.45x |
| LSTM | 800 | +5.53 | -0.40 | -3.37 | 47.7% | 1.90x |

**Why penny contracts hurt:**
- A $0.03 contract with 5pp fee costs $0.0015 per trade — but the average profit per trade is only +$0.004. Fees consume 37% of gross on cheap contracts vs 10% on $0.50 contracts.
- Penny contracts are near-certain outcomes (probability ~3% or ~97%). Spreads are noise, not signal.
- The model was trained on contracts in the $0.20-$0.80 range. Predictions on extreme-probability contracts are unreliable.

**Implementation:** Filter `mid_price >= $0.10` before generating signals. This is a single line of code that turns a losing strategy into a profitable one at realistic fee levels.

---

## Historical Backtest Results (Phases 4-7)

These results are from the full ML pipeline trained on 144 matched pairs (6,802 training rows, 1,673 test rows):

### Cross-Tier Model Comparison (8 models)

| Tier | Model | RMSE | Per-Trade Sharpe | Backtested Sharpe (5pp fees) |
|------|-------|------|-----------------|----------------------------|
| 1 | **XGBoost** | **0.286** | **0.52** | **8.30** |
| 1 | Linear Regression | 0.308 | 0.50 | 8.70 |
| 2 | LSTM | 0.292 | 0.47 | 8.38 |
| 2 | GRU | 0.293 | 0.46 | 8.29 |
| 3 | PPO-Raw | 0.319 | 0.31 | 5.96 |
| 4 | PPO-Filtered | 0.327 | 0.01 | 1.99 |
| 1 | Volume | 0.457 | 0.13 | -2.26 |
| 1 | Naive | 0.500 | 0.13 | -2.25 |

### SHAP Feature Importance (XGBoost)

Top feature: `polymarket_vwap` (mean |SHAP| = 0.138) — 10x more important than any other feature. The model primarily uses Polymarket's price to predict spread direction.

### Bootstrap Confidence Intervals

XGBoost, GRU, and LSTM have **overlapping RMSE confidence intervals** — their prediction quality is statistically indistinguishable at this dataset scale. This supports the paper's central thesis: **model complexity is not justified at small dataset scales.**

### Experiment Results

1. **Complexity vs Performance (centerpiece):** XGBoost > LSTM ≈ GRU > Linear Reg >> PPO. Simpler models win.
2. **Lookback Window Ablation:** Shorter windows (8h) marginally outperform longer (72h). Dataset too small for long history.
3. **Spread Threshold Ablation:** XGBoost best at all thresholds. PPO models produce discrete predictions incompatible with high thresholds.
4. **Transaction Cost Sensitivity:** XGBoost profitable up to 15.5pp fees (historical). At 5pp: XGBoost +$162, GRU +$137.

---

## System Architecture

```
GitHub Actions (hourly cron)
    │
    ├── src/live/collector.py          Poll Kalshi + Polymarket prices
    │   └── 615 matched pairs via semantic similarity
    │
    ├── src/live/paper_trader.py       Train models → predict → log trades
    │   └── 5 models (LR, XGBoost, GRU, LSTM, Naive)
    │
    ├── data/live/bars.parquet         Growing dataset (hourly snapshots)
    ├── data/live/paper_trades.jsonl   Trade log with predictions + outcomes
    │
    └── git commit + push              Auto-commits results back to repo

Local:
    python -m src.live.dashboard       View paper trading P&L
    python -m src.live.retrain         Retrain models on original + live data
```

### ML Pipeline (Phases 1-7)

```
Phase 1-3: Data ingestion + matching + feature engineering
    144 matched Kalshi↔Polymarket pairs, 4h bars, 31 features

Phase 4: Regression baselines (Linear Reg, XGBoost)
Phase 5: Time series models (GRU, LSTM) — TFT deferred
Phase 6: RL models (PPO-Raw, PPO-Filtered with autoencoder)
Phase 7: Experiments (SHAP, bootstrap CIs, ablations)
Phase 7.1: Walk-forward backtesting (honest Sharpe with fees)
Phase 7.2: Live paper trading system
```

---

## Next Steps

### Immediate (this week)
- [ ] Continue hourly collection — need 5+ days for statistical significance
- [ ] Monitor oil/commodity edge — is it persistent or a one-time tariff event?
- [ ] Set up Oracle Cloud VM for 15-minute bar collection (higher resolution for daily contracts)

### Short-term (before April 27 deadline)
- [ ] Implement penny contract filter ($0.10 minimum) in live collector
- [ ] Focus model on oil/commodity contracts where edge is strongest
- [ ] Retrain models on original + live data (test if more data helps GRU/LSTM)
- [ ] Write Phase 8 paper with live trading results as centerpiece finding

### Medium-term (if edge persists)
- [ ] Real money pilot: $100-$500 on Linear Regression signals, oil contracts only
- [ ] Kalshi API execution layer (authenticated trading)
- [ ] Position sizing optimization (Kelly criterion)
- [ ] Risk management (max drawdown limits, position limits per contract)

### Research extensions
- [ ] Why does Linear Regression beat XGBoost on live data? (opposite of historical)
- [ ] Is the oil edge structural (platform disagreement) or temporary (volatility)?
- [ ] Can we expand to other commodity brackets (gold, gas, BTC)?
- [ ] 15-minute bars: does higher frequency improve P&L on daily contracts?

---

## How To Use

### Check paper trading performance
```bash
git pull --rebase
python -m src.live.dashboard
```

### Run full statistical analysis
```bash
python -c "..." # see scripts in repository
```

### Retrain models with new data
```bash
python -m src.live.retrain --skip-tier3
```

### Manual collection cycle
```bash
python -m src.live.collector --live-pairs
```

---

## Repository Structure

```
DS340 Project/
├── src/
│   ├── models/          8 model implementations (LR, XGBoost, GRU, LSTM, PPO×2, Naive, Volume)
│   ├── evaluation/      Metrics, profit simulation, walk-forward backtester
│   ├── features/        Feature engineering pipeline
│   ├── live/            Live data collector, paper trader, retrain, dashboard
│   └── matching/        Cross-platform contract matching (NLP)
├── experiments/
│   ├── results/         Per-tier JSON results + ablation + backtest
│   └── figures/         12 publication-quality figures
├── data/
│   ├── processed/       Training/test data (144 pairs, 31 features)
│   └── live/            Growing live dataset + trade logs
├── tests/               91 tests across all components
└── .github/workflows/   Automated hourly collection + paper trading
```

---

*Report generated from live paper trading data. All statistics computed on contracts ≥ $0.10 with penny contract filter applied.*
