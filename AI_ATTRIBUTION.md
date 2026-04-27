# Code Authorship & AI Attribution

**Project:** Complexity Is Not an Edge — DS340 Spring 2026 Final Project
**Authors:** Ian Sabia (U33871576), Alvin Jang (U64760665)
**Date:** April 27, 2026

This document discloses how the code in this repository was authored, in compliance with academic honesty guidelines for the DS340 final project.

---

## Summary

This is a large software system spanning data ingestion, market matching, feature engineering, six model implementations, evaluation, live paper-trading deployment, and an adversarial audit pipeline. Building it within a one-month timeline required AI pair-programming for the engineering layer. **All algorithmic and modeling decisions, experimental design, evaluation methodology, and interpretation of results are human-led.**

The disclosure below maps each component to its authorship category. Nothing in this repository was copied from online sources; standard libraries and their documented APIs are used per their licenses (full list at the end).

---

## Authorship Categories

| Category | What it means |
|---|---|
| **Human-led** | Designed and primarily implemented by Ian Sabia and Alvin Jang. AI used only for syntax help and small refinements, not the substantive logic. |
| **AI-assisted** (Anthropic Claude via Claude Code) | Substantial pair-programming with Claude. Human authors set requirements, made architectural decisions, reviewed every change, and tested outputs. AI generated the bulk of the implementation code under human direction and review. |

---

## Component-by-Component Disclosure

### Models — **Human-led**

The substantive modeling contribution — what each model is, how it's parameterized, how its outputs are interpreted — is human-authored. These files reflect our research decisions: which architectures to compare, which hyperparameters to sweep, what reward function PPO should optimize, how to wrap pretrained components into a unified interface.

| File | What it does |
|---|---|
| `src/models/linear_regression.py` | LR predictor (Tier 1) |
| `src/models/xgboost_model.py` | XGBoost predictor (Tier 1) — 48-config hyperparameter sweep |
| `src/models/gru.py` | GRU sequence model (Tier 2) |
| `src/models/lstm.py` | LSTM sequence model (Tier 2) |
| `src/models/tft.py` | Temporal Fusion Transformer wrapper around `pytorch-forecasting` (Tier 2) |
| `src/models/ppo_raw.py` | PPO agent acting on raw features (Tier 3) |
| `src/models/ppo_filtered.py` | PPO + autoencoder anomaly filter (Tier 3) |
| `src/models/autoencoder.py` | Anomaly detector for PPO+filter pipeline |
| `src/models/trading_env.py` | Gym environment for RL training (state, action, reward) |
| `src/models/naive.py` | Naive baseline (spread always closes) |
| `src/models/volume.py` | Volume-weighted baseline |
| `src/models/ensemble.py` | LR + XGBoost ensemble used in production live system |
| `src/models/base.py` | `BasePredictor` interface every model implements |

### Infrastructure & Engineering — **AI-assisted (Claude Code)**

The supporting system that makes the modeling possible was built with Claude as a pair programmer. For each component below, the human authors specified the requirements, reviewed every diff, made architectural decisions, tested behavior, and caught real bugs (e.g., the Phase 15 silent commodity-discovery starvation; the Phase 18 embargo-violation finding). Claude generated the implementation code from those specifications.

| Component | Path | What's there |
|---|---|---|
| Data ingestion | `src/data/` | Kalshi REST adapter, Polymarket Gamma/CLOB/Data API adapters, candle reconstruction from trade records, retry/backoff |
| Market matching | `src/matching/` | sentence-transformer embeddings, 10-rule structural quality filter, candidate-pair generation |
| Feature engineering | `src/features/` | 51-feature derivation including microstructure estimators (Amihud illiquidity, Corwin-Schultz spread, Kyle λ, Roll spread) implemented from published formulas |
| Evaluation | `src/evaluation/` | profit simulator, walk-forward backtester, bootstrap CI utilities |
| Live trading system | `src/live/` | collector, paper-trader, position manager, retrain controller, exit-policy logic |
| Reconciliation | `src/analysis/` | Live-vs-backtest tracking-error analysis |
| Plotting | `src/plotting/` | IEEE-styled figure generation (SciencePlots wrapper) |
| Utilities | `src/utils/` | seed manager (deterministic training), logging helpers |
| Experiments | `experiments/` | Runner scripts for each paper table/figure, canonical baseline runs |
| Audit pipeline | `experiments/audit/` | Phase 18 adversarial audit (Sharpe recompute with BLdP correction, leakage detection, cost realism, survivorship checks) |
| Tests | `tests/` | pytest suite covering data integrity, matching correctness, feature determinism, audit fixtures (746 tests) |
| Operational scripts | `scripts/` | paper validators, figure regeneration, deployment automation |

### Documentation & Analysis — **AI-assisted**

| File | Note |
|---|---|
| `PAPER_DRAFT.md` / `paper.pdf` | Paper drafted with Claude. All numerical claims verified by human authors against `experiments/results/canonical/headline.json`. All citations independently verified against original sources (see commit `a1b7257` for citation audit log). |
| `AUDIT_REPORT.md` | Phase 18 adversarial audit report. Audit scripts run and verified by humans; narrative drafted with Claude. |
| `README.md` | Written with Claude assistance. |
| `slides_deck.html`, `slides_deck.pptx` | Design and copy collaborative. Numerical claims verified against canonical results. |

### Pulled From Online — **None (only standard libraries via their package managers)**

No code was copied from blog posts, Stack Overflow answers, GitHub repositories, or other online sources without proper attribution. Standard libraries are used as documented:

| Library | Purpose | License |
|---|---|---|
| `sentence-transformers` | Sentence embeddings (all-MiniLM-L6-v2) for matching | Apache 2.0 |
| `pytorch-forecasting` | TFT model implementation | BSD-3 |
| `stable_baselines3` | PPO algorithm implementation | MIT |
| `xgboost` | Gradient-boosted trees | Apache 2.0 |
| `scikit-learn` | Linear Regression, evaluation utilities | BSD-3 |
| `pytorch`, `lightning` | Neural network training | BSD-3 |
| `pandas`, `numpy` | Data manipulation | BSD-3 |
| `SHAP` (Lundberg & Lee 2017) | Feature attribution | MIT |
| `quantstats` | Sharpe / drawdown calculations | Apache 2.0 |
| `SciencePlots` | IEEE figure styling | MIT |
| `arch` | Bootstrap stationary resampling | NCSA |

The microstructure feature estimators (Amihud 2002, Corwin-Schultz 2012, Kyle 1985, Roll 1984) were implemented directly from the published formulas in those papers, cited in `PAPER_DRAFT.md` §References. No external code was copied.

---

## What Each of Us Did

**Ian Sabia** — Lead architect. Designed the four-tier model comparison, owned the matching pipeline (including the 10-rule structural quality filter that flips P&L by +$10.73 with no model changes), drove the live deployment on BU SCC, ran and interpreted the Phase 18 audit (catching the embargo-violation that motivated the leakage-free retrain), authored the abstract / introduction / results discussion, and gives sections 1–4 of the lightning talk.

**Alvin Jang** — Co-architect. Owned the feature engineering pipeline (51 features including 13 microstructure estimators), reviewed and refined the model implementations (especially the sequence models and PPO reward function), led the slide design, and gives sections 5–7 of the lightning talk.

**Claude (AI assistant)** — Pair programmer. Generated implementation code for the engineering layer (data adapters, matching pipeline plumbing, feature derivation, evaluation utilities, live system, audit scripts, tests). Drafted documentation and paper prose. Did not make experimental decisions or interpret results.

---

## Why we disclose this

The DS340 final project asks for honest authorship attribution. We believe the right standard is to be specific: which decisions were ours, which code was AI-generated under our direction, and which third-party libraries we depended on. The findings of this project — the empirical answer to "does complexity improve arbitrage detection?" — are the human contribution. The implementation velocity that let us build a deployed, audited system in a month was the AI contribution. We are transparent about both.
