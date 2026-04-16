# Kalshi vs. Polymarket Price Discrepancies

## What This Is

A cross-platform prediction market arbitrage system for DS340 (Spring 2026). We build a data pipeline to match contracts across Kalshi and Polymarket, extract microstructure features from their price/volume data, and systematically test whether increasing model complexity (from regression to RL with anomaly detection) improves spread convergence prediction and simulated trading performance. The system is now deployed autonomously on the BU Shared Computing Cluster (SCC) with GitHub Actions fallback, trading 11,582 matched pairs on a 15-minute cycle with 6-hour auto-retrain. This is an academic project for Ian Sabia and Alvin Jang.

## Core Value

Answer the research question empirically: **does adding model complexity (RL, anomaly detection) improve cross-platform arbitrage detection over simpler regression, and when is that complexity justified?**

## Current Milestone: v1.1 — Extended Evidence & Submission

**Goal:** Strengthen every pillar of the paper's evidence base — scaling, model variety, feature understanding, execution realism — while the live system continues accumulating data. Produce a submission-ready paper (due Apr 27) that credibly signals continued deployment post-submission.

**Target features (each produces an independent paper-ready finding):**
- Priority-1 cleanups + full cross-model re-verification (LR ≡ XGBoost, GRU overfits, adaptive lookback)
- Live vs backtest reconciliation (unique evidence — almost no student paper has it)
- 250-bar data-scaling checkpoint (third scale point, tests ranking invariance)
- TFT (Temporal Fusion Transformer) — the deferred Tier-2 model
- Feature ablation study (minimum feature set, parsimony finding)
- Ensemble & deployment model exploration (evidence-based production ensemble)
- Paper finalization + 4-minute lightning-talk slides

**Non-goal:** Live-money trading. System stays on paper trading for the v1.1 scope; real-money deployment is the headline item for future-work.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Data pipeline that ingests historical market data from both Kalshi and Polymarket APIs
- [ ] Market matching pipeline (keyword + sentence-transformer semantic similarity) with manual curation pass
- [ ] Feature engineering: per-hour feature vectors from microstructure data (spread, volume, bid-ask, price velocity)
- [ ] Matched-pairs dataset with time-aligned price histories across both platforms
- [ ] Tier 1 models: Linear Regression and XGBoost trained and evaluated on spread prediction
- [ ] Tier 2 models: GRU, LSTM, and TFT trained and evaluated on spread prediction
- [ ] Tier 3 models: PPO on raw features, PPO + autoencoder signal filter
- [ ] Naive baselines: spread-always-closes and higher-volume-platform-correct
- [ ] Experiment 1 (centerpiece): Complexity-vs-performance comparison across all tiers
- [ ] Experiment 2: Historical window length ablation (6h, 24h, 72h, 7d)
- [ ] Experiment 3: Minimum spread threshold ablation (no min, >2pp, >5pp, >10pp)
- [ ] Profit simulation: simulated P&L, win rate, Sharpe ratio for each model
- [ ] SHAP interpretability analysis on best-performing models
- [ ] Final paper and lightning talk slides

### Out of Scope

- Live trading or real-money execution — this is historical backtesting only
- External event-driven features (news, sentiment) — microstructure only by design
- Pretrained financial or NLP models — all models trained from scratch
- Third-party data aggregators — direct API ingestion only
- Transaction cost modeling — acknowledged but not modeled in simulation
- Mobile app or web dashboard — this is a research project with scripts and notebooks

## Context

**Domain:** Prediction markets (Kalshi, Polymarket) are continuously tradeable contracts priced between $0-$1. Cross-platform price discrepancies create arbitrage opportunities: buy the underpriced contract on one platform, short the overpriced one on the other, profit when prices converge. Our system detects these opportunities and tests whether ML/RL can predict convergence timing.

**Professor feedback (KG):** Questioned whether all moving parts are needed. Could RL act directly on features without the autoencoder? Could this just be regression? Response: we now test all three framings (regression-only, RL-only, RL+autoencoder) and let the data answer. The project is now structured as a complexity-vs-performance tradeoff analysis.

**Data pipeline challenges (from API research):**
- Polymarket `/prices-history` returns empty data for resolved markets — must reconstruct from trade records via Data API
- Polymarket uses long numeric token IDs, not slugs — must query Gamma API first for `clobTokenIds`
- Kalshi live/historical data split at ~3 month cutoff — must call `/historical/cutoff` to determine which endpoint
- Contract matching requires NLP — no shared identifiers across platforms
- Settlement criteria differ between platforms — real arbitrage risk
- Many Kalshi hourly contracts have <10 trades — filter low-liquidity markets

**Dataset size is an open question.** The number of matchable market pairs across economics/finance and crypto categories is unknown. Could be dozens or hundreds. If very small, PPO will almost certainly underperform — which is itself a valid finding.

**Expected outcome:** Simpler models (XGBoost, GRU) likely outperform PPO due to sample efficiency. This answers the research question: complexity wasn't justified at this dataset scale. The novelty is in the application domain and systematic comparison, not the architecture.

## Constraints

- **Timeline**: TA check-in April 4 (pipeline + baselines), final submission April 27
- **Team**: 2 people (Ian Sabia, Alvin Jang), work division TBD
- **Stack**: Python 3.12, PyTorch, PyTorch Forecasting, XGBoost, scikit-learn, sentence-transformers, SHAP
- **Data**: Public APIs only, no auth required for market data
- **Compute**: Local machines, no cloud GPU budget assumed
- **Academic**: Must produce a paper and lightning talk, not just code

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Microstructure features only (no external signals) | Isolates whether arbitrage signal is self-contained within the markets | — Pending |
| Auto matching + manual curation | Sentence-transformer finds candidates, human verifies settlement criteria alignment | — Pending |
| PPO over Q-learning for RL | PPO is more stable in noisy environments, better sample efficiency | — Pending |
| Restrict to economics/finance + crypto categories | Best overlap between platforms for matched-pairs dataset | — Pending |
| Regression baselines are first-class | Per KG feedback — simpler models must be strong to justify added complexity | — Pending |
| Added PPO-without-autoencoder variant | Per KG feedback — isolates whether RL itself adds value vs. the anomaly detection layer | — Pending |

---
*Last updated: 2026-04-17 after v1.1 milestone start*
