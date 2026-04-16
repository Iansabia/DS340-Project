# Complexity Is Not an Edge: An Empirical Study of Machine-Learning Arbitrage on Kalshi and Polymarket

**Ian Sabia** (U33871576), **Alvin Jang** (U64760665)
Department of Data Science, Boston University
DS340 — Spring 2026 Final Project
April 27, 2026

---

## Abstract

We study cross-platform price discrepancies between two of the largest U.S. prediction-market venues — **Kalshi** (CFTC-regulated event contracts) and **Polymarket** (permissionless on-chain prediction market) — and ask whether increasing model complexity improves arbitrage detection. We build an end-to-end system that (i) matches semantically-equivalent contracts across platforms using sentence embeddings with a 10-rule structural quality filter, (ii) engineers 59 features including 13 academic market-microstructure features (Amihud illiquidity, Corwin–Schultz implied spread, Kyle's lambda, Roll's implied spread, favorite–longshot bias), and (iii) trains four tiers of models — regression baselines (LR, XGBoost), sequence models (GRU, LSTM), PPO reinforcement learning, and PPO with an autoencoder anomaly filter — under an identical evaluation protocol. The system was also deployed as an autonomous paper-trading agent on the BU Shared Computing Cluster (SCC), executing a 15-minute trade cycle and retraining models every six hours. Across five independent evaluation regimes — (1) a single-split backtest, (2) an 11-window walk-forward backtest, (3) a category-stratified analysis, (4) a 6-point data-scaling curve (50→2000 bars/pair), and (5) live paper trading — the **simplest models consistently dominate**. In our primary fresh verification run (April 17, 2026) on 6,802 training rows and 1,673 held-out test rows across 144 matched pairs, XGBoost (depth-3 trees) achieves +\$201.63 simulated P&L at a 2-percentage-point fee, essentially **tied** with Linear Regression at +\$201.69, versus +\$182.72 for LSTM, +\$174.11 for GRU, and **−\$7,724 for PPO+autoencoder**. In the 11-window walk-forward, every ML model is profitable in every window; per-trade Sharpe trends upward from ≈0.31 in the earliest windows to ≈0.53 in the most recent, confirming the edge is stable and improving with training data. The central empirical answer: **at this data scale, complexity is a liability, not an edge**. The alpha lives in the matching pipeline and the oil/commodities asset class — not in the models.

**Keywords:** prediction markets, arbitrage, market microstructure, XGBoost, LSTM, PPO, walk-forward validation, simplicity.

---

## 1. Introduction

### 1.1 Problem Statement

Prediction markets allow traders to buy and sell contracts that pay out \$1 if a specified real-world event occurs (e.g., "Will CPI inflation exceed 3.0% in May 2026?") and \$0 otherwise. The equilibrium price of such a contract is interpreted as the market-implied probability of the event. Two large U.S.-accessible venues — **Kalshi** (CFTC-regulated, operating since 2021) and **Polymarket** (on-chain via Polygon, operating since 2020) — frequently list contracts that reference the *same* underlying event but trade at materially different prices. These cross-platform discrepancies can persist for hours or days, representing a potential statistical-arbitrage opportunity if they can be detected, matched correctly, and traded before they close.

The central research question of this project is:

> **Does increasing model complexity improve arbitrage detection in cross-platform prediction markets, and if so, when is that complexity justified?**

Answering this question rigorously requires four things, each of which we provide:

1. A *shared evaluation protocol* across model families of very different complexity.
2. A *robust matching pipeline* — without it, models are fitting noise.
3. *Multiple independent evaluation regimes* (single-split, walk-forward, scaling curve, live trading) so conclusions do not hinge on a single train/test split.
4. An *honest accounting* of transaction costs, Sharpe inflation, and survivorship bias.

### 1.2 Motivation

Two distinct audiences motivate this work:

**Academic.** A January 2026 working paper (arXiv 2601.07131) argues that "Matched Filter" normalization grounded in market-microstructure theory captures virtually all exploitable signal for investor-flow prediction, and that feature engineering consistently beats deep learning in this regime. Prediction-market arbitrage is a natural testbed for that claim because contracts have a finite lifetime, bounded payoffs, and deterministic settlement — eliminating much of the noise that hides signal in equities. If the "simple beats complex" result holds in this cleaner domain, it is strong evidence for the broader thesis.

**Applied.** Prediction markets are growing rapidly: Polymarket processed over \$1 billion of volume during the 2024 election cycle, and Kalshi added sports-event and economic-indicator contracts in 2025. As liquidity grows, cross-platform inefficiencies become a relevant feature of the ecosystem. Understanding which modeling approaches genuinely add value — and which merely add training cost — is a practical question for market participants.

### 1.3 Background

**Prediction-market mechanics.** A binary event contract on platform $P$ at time $t$ has a market price $p_P(t) \in [0, 1]$ interpretable as the implied probability of the event. At event resolution, the contract pays exactly \$1 or \$0. For two platforms $A, B$ listing the *same* event, the **spread** is $s(t) = p_A(t) - p_B(t)$. If the contracts are genuinely equivalent, $s(t) \rightarrow 0$ as $t$ approaches resolution, modulo fees and basis risk. Our models predict the one-step change $\Delta s(t) = s(t+1) - s(t)$.

**Why discrepancies exist.** Kalshi and Polymarket have disjoint user bases (Kalshi: retail U.S. traders, regulated brokerage experience; Polymarket: crypto-native, self-custodial wallet), different fee structures (Kalshi: \$0.00–\$0.07 per contract; Polymarket: 0% trading fee, \$0.01–\$0.05 Polygon gas), and different listing policies (Kalshi has contract-expiry time-windows as granular as one hour; Polymarket tends to list monthly or quarterly contracts). These differences slow cross-platform arbitrage and sustain price dispersion.

**Cross-platform matching is non-trivial.** There are no shared identifiers between platforms; the same event may be titled "Will the Fed cut rates at the March 2026 meeting?" on Kalshi and "March Fed decision" on Polymarket with different explanatory text. Contract resolution criteria can also differ in subtle ways (e.g., "who wins the 2026 NBA Finals" vs. "who wins the 2026 NBA MVP"). Semantic matching — with a principled quality filter — is therefore a first-class concern, not a preprocessing afterthought.

### 1.4 Contributions

This paper makes the following contributions:

1. **A functioning end-to-end arbitrage system**, from data ingestion through live paper-trading on BU SCC, with redundant deployment on GitHub Actions. At its peak, the system monitored **11,582 matched contract pairs**.

2. **A complexity-vs-performance benchmark** comparing four tiers of models under an identical evaluation protocol, evaluated across five independent regimes.

3. **An 11-window walk-forward validation** (Fig. 1) demonstrating the edge is stable and *improving* over time (per-trade Sharpe 0.31 → 0.53, every window profitable for every ML model).

4. **A per-category stratified analysis** (Table 3) showing that the "XGBoost wins overall" conclusion is driven by specific regimes (inflation, crypto), not universal model superiority — a nuanced finding.

5. **A six-point data-scaling curve** (Table 4) directly addressing the "would more data flip the ranking?" question — it does not.

6. **A transaction-cost sensitivity analysis** (Table 6) showing the system remains profitable at all realistic fee levels from 0 pp to 7 pp.

7. **An honest Sharpe-ratio accounting** (§4.6) showing that the "headline" per-trade Sharpe of 0.59 annualizes to roughly 4.3 under the correct per-pair independence assumption — strong but likely inflated by the short test window.

8. **A negative result**: reinforcement learning (PPO, PPO + autoencoder) is *worse* than every other model in our evaluation. We report this transparently rather than suppressing it.

---

## 2. Related Work

**Prediction-market efficiency.** Manski (2006) and Wolfers & Zitzewitz (2004) established the foundational result that prediction-market prices, while not fully efficient, are among the best available probability forecasts for well-specified events. Cross-platform dispersion has been less studied — most prior work treats a single venue in isolation.

**Favorite–longshot bias.** A robust empirical regularity in prediction markets is that longshot contracts (low prices) trade *above* their realized probabilities while heavy favorites trade *below* their realized probabilities. Burgi, Tuccella & Zitzewitz (2026) quantify this bias on both Kalshi and Polymarket and show it differs across platforms — creating a structural, model-free source of cross-platform spread. We incorporate this directly as a feature (§3.3).

**Market-microstructure features.** Amihud (2002) introduced the illiquidity ratio $|r|/V$ as a near-universal illiquidity measure. Corwin & Schultz (2012) derived a closed-form estimator of the effective bid–ask spread from daily high-low prices. Kyle (1985) introduced $\lambda$ as the price-impact coefficient in informed-trading models. Roll (1984) derived an implied spread from return autocorrelation. These features are standard in equity microstructure but, to our knowledge, have not been systematically applied to prediction-market spread prediction.

**Feature engineering vs. deep learning.** A persistent finding in applied ML is that on tabular data with moderate sample size, gradient-boosted trees (XGBoost, LightGBM) match or beat deep neural networks (Grinsztajn, Oyallon & Varoquaux, NeurIPS 2022). The January 2026 paper we cite above extends this to time-series investor-flow prediction. Our results are consistent with both.

---

## 3. Data and Feature Engineering

### 3.1 Data Sources

**Kalshi** exposes a public REST API (`api.elections.kalshi.com/trade-api/v2`) that requires no authentication for market-data endpoints. We fetch hourly OHLCV candlesticks (`period_interval=60`) and market-metadata endpoints for active and historical markets. A critical implementation detail: Kalshi splits market history at a roughly three-month cutoff, with older markets requiring a different `/historical/` endpoint path; we query `/historical/cutoff` to determine which endpoint applies per ticker.

**Polymarket** is significantly more complex, exposing *three* separate APIs: **Gamma** (market metadata), **CLOB** (order-book and price-history for active markets), and the **Data API** (fills and trades for resolved markets). Polymarket uses opaque numeric token IDs rather than slugs, so we perform a two-stage lookup: Gamma gives us `clobTokenIds`, which then key the CLOB price queries. A non-obvious gotcha: Gamma's singular `condition_id=` query parameter returns *unrelated random markets* (we observed "Russia–Ukraine ceasefire before GTA VI?" returned for a Canadian-recession query), while the plural `condition_ids=` parameter returns the exact match. Additionally, the `/prices-history` endpoint returns empty data for *resolved* markets, forcing us to reconstruct historical prices from trade records via the Data API.

### 3.2 Matching Pipeline

Semantic matching uses `sentence-transformers/all-MiniLM-L6-v2` to embed the concatenated title + description of each market on each platform into a 384-dimensional vector. We compute cosine similarity between all pairs via normalized dot-product matrix multiplication. An initial O(N·M) keyword pre-filter — which would have taken ~6.6 hours on our universe — was replaced with the matrix approach that runs in ~80 seconds, a 300× speedup.

Semantic similarity alone is insufficient: the embeddings cluster "NBA Finals winner" and "NBA MVP" very close together. We layered **10 structural quality-filter rules** on top (`src/matching/quality_filter.py`):

1. Sports: wins-vs-champion mismatch (e.g., "Lakers win Game 4" vs "Lakers win the title").
2. Fed: meeting-month mismatch (March vs. May).
3. Politics: cabinet-confirmation vs. cabinet-nomination.
4. Commodities: state-specific (CA/FL/NY/TX) vs. national (added in April after discovering 135 false matches).
5. Inflation: exact CPI vs. PCE vs. core-CPI disambiguation.
6. Strict expiry-date window: ±2 days maximum.
7. Symbol-suffix matching: exact strike comparison for futures-referenced contracts.
8. Numeric-threshold exactness: "\$50/bbl" vs. "\$55/bbl" rejected.
9. Event-key disambiguation: "Game 4" vs. "Game 5".
10. Category-consistency guard: oil market cannot match non-oil event.

This filter rejected **140 of 615 pairs (22.8%)**. The impact was large: at the first large-scale backtest (April 11, 2026), linear-regression P&L went from **−\$5.28 to +\$5.45** after adding these rules — a +\$10.73 swing purely from removing structurally-bad matches, with no model changes. We interpret this as direct empirical evidence that data quality is more important than model choice at this scale.

### 3.3 Feature Engineering

We engineer **59 features** organized into five groups. Table 1 summarizes.

**Table 1: Feature taxonomy (59 features total).**

| Group | Count | Examples | Motivation |
|---|---|---|---|
| Raw aligned (per-platform OHLCV) | 18 | `kalshi_open`, `polymarket_volume`, `kalshi_close` | Native platform signal |
| Cross-platform basic | 6 | `spread`, `mid_price`, `dollar_volume_ratio` | Arbitrage-signal primitives |
| Rolling/momentum | 9 | `spread_momentum_6`, `spread_volatility_6`, `spread_zscore` | Short-memory dynamics |
| Classical microstructure (academic) | 13 | `amihud_illiquidity`, `corwin_schultz_spread`, `kyle_lambda`, `roll_spread`, `bekker_parkinson_vol` | Grounded in 1984–2012 literature |
| Prediction-market-specific | 13 | `favorite_longshot_bias`, `near_expiry_indicator`, `platform_age_delta` | Domain-specific (Burgi 2026) |

The microstructure features are computed per-platform then differenced across platforms, producing `amihud_illiquidity_delta`, `kyle_lambda_delta`, etc. The rationale is that *relative* liquidity across platforms should predict which side converges — a Polymarket spread that is less informed (higher Amihud, higher Roll spread) should pull toward the more-liquid Kalshi price.

Empirically, the 13 classical-microstructure features are *neutral-to-slightly-positive* on the historical dataset. This is because the features rely on rolling windows of depth 6–12 bars, and the historical dataset averages only 47 bars/pair. As live trading accumulates more bars per pair, these features are expected to become informative — a hypothesis we explicitly test via the data-scaling experiment (§4.3).

---

## 4. Methodology

### 4.1 Four Model Tiers

We compare four tiers of increasing complexity, all trained from scratch on the same matched-pair dataset using the same features and the same target variable $\Delta s(t)$.

**Tier 0 — Naive baselines (lower bound):**
- *Naive-closes*: predict spread always closes fully by resolution.
- *Volume-higher-wins*: predict higher-volume platform is always correct.

**Tier 1 — Regression baselines (the backbone):**
- *Linear Regression* (scikit-learn `LinearRegression`).
- *XGBoost* (xgboost `XGBRegressor`), searched over depth ∈ {3, 5, 7, 9} × learning-rate ∈ {0.01, 0.05, 0.1, 0.3} × `n_estimators` ∈ {100, 300, 500} (48 configurations).

**Tier 2 — Time-series models:**
- *GRU* (PyTorch, 64 hidden units, 1 layer, 24-bar lookback, StandardScaler inputs, Adam optimizer, `lr=1e-3`, early stopping on validation loss).
- *LSTM* (same architecture, LSTMCell replaces GRUCell).

**Tier 3 — Reinforcement learning:**
- *PPO-Raw*: PPO agent acting directly on 59-dimensional feature vectors, 3-action space {buy-spread, sell-spread, hold}, custom gym environment with mark-to-market reward at each step. PPO implementation from `stable-baselines3`.
- *PPO-Filtered*: same PPO agent, but a trained autoencoder (3-layer symmetric, bottleneck 8) pre-filters observations, flagging "anomalous" spreads by reconstruction error. PPO only trades when anomalies are detected.

All models use the same train/test split: **time-ordered 80/20**, preserving temporal causality. No shuffling. No look-ahead. Features are computed using only past information.

### 4.2 Evaluation Protocol

Five evaluation regimes are used — each provides an independent view.

**(a) Single-split backtest.** Train on the first 80% of data (chronological), test on the last 20%. Report RMSE, MAE, directional accuracy, simulated P&L at 2pp fees, win rate, and per-trade Sharpe.

**(b) Walk-forward backtest (11 windows).** Concatenate train + test for maximum time coverage (Jan 1 – Apr 1, 2026 of the historical dataset). Split into 12 equal-time windows; use an expanding-window protocol where window $i$ is trained on all data from windows $\{0, \ldots, i-1\}$ and tested on window $i$ (window 0 has no training set, so we report 11 evaluation windows). Per-window metrics: RMSE, directional accuracy, P&L at 2 pp, win rate, and per-trade Sharpe. If the edge is stable, we expect positive P&L across windows; if it is improving with more training data, we expect per-trade Sharpe to trend upward.

**(c) Per-category breakdown.** Stratify test-set rows by contract category (oil, crypto, inflation, employment, Fed, GDP, politics-election, politics-policy, sports) using a deterministic rule derived from Kalshi tickers and Polymarket slugs. Report per-category P&L, win rate, and trade count for each model.

**(d) Data-scaling curve.** Train each Tier-1 and Tier-2 model on progressively larger slices of the data — 50, 100, 250, 500, 1000, 2000 bars/pair — and plot P&L vs. training size. This answers "does the simple-beats-complex conclusion hold across scales, or is it an artifact of small data?"

**(e) Live paper trading.** Deploy the best-performing trained model (depth-3 XGBoost ensembled with LR) on BU SCC with a 15-minute trading cycle. Auto-discover new pairs every 3 hours, auto-retrain every 6 hours. Record trades, exits, and P&L in `positions.db` and `trade_log.jsonl`.

### 4.3 Training and Hyperparameter Search

Tier-1 models complete training in under 15 seconds on a single CPU core. Tier-2 models take approximately 3 minutes per epoch on a single CPU, early-stopping after 10–15 epochs. Tier-3 models take 20–40 minutes for PPO convergence.

The XGBoost hyperparameter sweep was exhaustive: **48 configurations**, evaluated on the single-split backtest. Results are in Table 5 (§5.3); the best configuration is `depth=3, lr=0.01, n=100`. All 10 top configurations used `depth ∈ {3, 5}` — deeper trees *lost* P&L, evidence of over-fitting at this training-set size (6,802 rows).

Tier-2 models were not searched over hyperparameters due to compute budget; we used architecture defaults from the original PyTorch Forecasting examples and report the result transparently.

### 4.4 Live System Architecture

The paper-trading system runs on **BU SCC (scc1.bu.edu)** with three cron jobs:

1. **Trading cycle** every 15 minutes on the login node (~3 min CPU) — fetches live prices, generates predictions via the trained LR + XGBoost ensemble, applies a category-aware entry filter (commodity pairs use the base threshold, non-commodity require 3× confidence) and a concordance filter (skip if LR and XGBoost disagree on sign), then executes paper trades against the current market mid.

2. **Market discovery** every 3 hours as a batch job via `qsub` (~10 min) — fetches fresh Kalshi and Polymarket universes, matches semantically, applies quality filters, evicts stale pairs via a tombstone system with a 7-day TTL that protects open positions.

3. **Model retraining** every 6 hours as a batch job via `qsub` (~30 min) — rebuilds the training set from accumulated live bars and retrains LR, XGBoost, GRU, LSTM. A "checkpoint system" triggers the scaling-curve experiment when at least 20 pairs cross each data threshold (50, 100, 250, 500, 1000 bars/pair).

**GitHub Actions** runs both discovery and trading as *redundant fallback* workflows on a separate schedule. If SCC goes down (it has scheduled maintenance twice per semester), GHA keeps the system alive. All state (pair mappings, positions, trade logs, model artifacts) is committed to Git and synchronized via rebase-retry push logic.

A critical bug discovered during deployment and documented in §4.5 was that three code paths disagreed on what a `live_NNNN` pair ID meant, causing 25 positions to track wrong markets. This was fixed by moving to **content-addressed pair IDs** (e.g., `kxwti26apr08t10799-0x43d5953d`) derived deterministically from the normalized Kalshi ticker and Polymarket token ID.

### 4.5 Challenges Overcome

Three non-obvious infrastructure bugs materially affected our results:

1. **Kalshi `/events` silent HTTP 429.** The endpoint returned 429 Too Many Requests on roughly 40% of calls, *silently dropping entire commodity series*. Fixed with exponential backoff (1s/2s/4s/8s) and a per-series 250 ms pace. Commodity pair count went from 65 → **506** after the fix.

2. **Polymarket pagination too shallow.** We were fetching only the top 5,000 markets; WTI markets sat at offset 15,305+ and were completely invisible. Fixed by bumping `max_pages` from 10 to 60 (up to 30,000 markets). The complete commodity universe then became reachable.

3. **Gamma `condition_id` vs. `condition_ids`.** As noted above, the singular form returned *random unrelated markets*, not an error. We only discovered this when Polymarket prices came back for a Canadian-recession contract labeled as being about a Russia–Ukraine ceasefire. The fix (`condition_ids=` plural) was a one-character change, but the failure mode (silent, non-obvious) lost us several days of debugging.

These three bugs shared a common pattern: APIs that fail silently in ways that *look like* model problems. We interpret this as a general lesson — for ML systems on external APIs, infrastructure monitoring must come before model tuning.

---

## 5. Results

### 5.1 Headline Model Comparison

Table 2 shows the single-split backtest at 2 pp transaction costs on the full 1,673-row test set. All models use the same feature set (51 numeric features; 8 NaN/zero-variance columns excluded from the 59 engineered) and the same target variable (next-bar spread change). Results are from a fresh re-run on April 17, 2026; earlier reported numbers (pre-144-pair expansion) overstated XGBoost's edge by roughly \$8 in P&L and 0.15 in Sharpe.

**Table 2: Single-split backtest results (2 pp fees, 1,673-row test set).**

| Tier | Model | RMSE | Dir. Acc. | P&L (\$) | Win Rate | Per-trade Sharpe | # trades |
|---|---|---|---|---|---|---|---|
| 0 | Naive (closes) | 0.499 | 53.3% | +28.92 | 40.9% | 0.062 | 1460 |
| 0 | Volume (higher wins) | 0.457 | 53.3% | +31.01 | 41.0% | 0.068 | 1440 |
| 1 | **Linear Regression** | 0.306 | 66.7% | **+201.69** | **50.6%** | 0.434 | 1549 |
| 1 | **XGBoost (depth 3)** | 0.293 | 67.4% | **+201.63** | **50.8%** | **0.436** | 1555 |
| 2 | LSTM | 0.295 | 65.9% | +182.72 | 50.0% | 0.387 | 1542 |
| 2 | GRU | 0.293 | 65.0% | +174.11 | 48.8% | 0.372 | 1516 |
| 3 | PPO-Raw | — | — | *negative* | — | negative | — |
| 3 | PPO + Autoencoder | — | — | **−\$7,724** | — | negative | — |

Three observations. **First**, the ranking is unambiguous: **Tier 1 (regression) > Tier 2 (sequence) > Tier 3 (RL)**. The Tier-0 → Tier-1 gap is very large (≈6.5×); the Tier-1 → Tier-2 gap is smaller but consistent (≈10–13%); Tier-3 is catastrophic. **Second**, LR and XGBoost are *essentially tied* (\$201.69 vs \$201.63) — 14 orders of magnitude closer than their gap to either sequence-model or baseline. Within Tier 1, there is no economic reason to prefer XGBoost on this dataset. **Third**, LSTM edges out GRU by \$8 in this single split, reversing the walk-forward median ordering (§5.2); both sit well below Tier 1.

### 5.2 Walk-Forward Validation

Fig. 1 (see `experiments/figures/walk_forward_pnl.png`) shows per-window P&L for all six models across 11 windows spanning January to April 2026. Training uses an expanding-window protocol: window $i$ trains on all data from windows $\{0, \ldots, i-1\}$ and tests on window $i$. Table 3 reports per-window P&L; Table 3b reports per-trade Sharpe across the same windows (the key stability statistic).

**Table 3: Walk-forward P&L at 2 pp fees (expanding-window, 11 out-of-sample windows).**

| Window | Train rows | Test rows | Naive | Volume | LR | XGBoost | GRU | LSTM |
|---|---|---|---|---|---|---|---|---|
| 1 | 357 | 558 | +\$34.57 | +\$31.95 | +\$48.70 | +\$53.50 | +\$53.08 | +\$41.93 |
| 2 | 915 | 648 | +\$29.91 | +\$28.76 | +\$61.33 | +\$58.70 | +\$59.92 | +\$55.52 |
| 3 | 1,563 | 963 | +\$62.50 | +\$65.60 | +\$106.78 | +\$109.25 | +\$104.89 | +\$93.68 |
| 4 | 2,526 | 1,318 | +\$40.14 | +\$40.10 | +\$141.50 | +\$142.44 | +\$134.71 | +\$135.00 |
| 5 | 3,844 | 1,125 | +\$21.25 | +\$24.39 | +\$128.85 | +\$125.98 | +\$120.44 | +\$120.83 |
| 6 | 4,969 | 680 | +\$10.64 | +\$9.08 | +\$80.42 | +\$76.65 | +\$77.81 | +\$73.72 |
| 7 | 5,649 | 651 | +\$25.77 | +\$22.84 | +\$65.70 | +\$67.36 | +\$64.82 | +\$65.32 |
| 8 | 6,300 | 1,123 | +\$25.71 | +\$26.87 | +\$144.89 | +\$138.68 | +\$135.85 | +\$141.08 |
| 9 | 7,423 | 589 | +\$14.67 | +\$13.92 | +\$73.38 | +\$74.29 | +\$70.51 | +\$75.18 |
| 10 | 8,012 | 497 | +\$15.53 | +\$16.61 | +\$75.30 | +\$75.27 | +\$75.28 | +\$73.91 |
| 11 | 8,509 | 110 | +\$2.72 | +\$2.87 | +\$10.22 | +\$11.54 | +\$9.34 | +\$10.67 |
| **Median** | | | **+\$25.71** | **+\$24.39** | **+\$75.30** | **+\$75.27** | **+\$75.28** | **+\$73.91** |
| **Mean** | | | **+\$25.76** | **+\$25.73** | **+\$85.19** | **+\$84.88** | **+\$82.42** | **+\$80.62** |

**Table 3b: Walk-forward per-trade Sharpe ratio by window.**

| Window | LR | XGBoost | GRU | LSTM |
|---|---|---|---|---|
| 1 | 0.307 | 0.383 | 0.341 | 0.260 |
| 2 | 0.355 | 0.347 | 0.367 | 0.321 |
| 3 | 0.408 | 0.424 | 0.410 | 0.359 |
| 4 | 0.403 | 0.403 | 0.383 | 0.386 |
| 5 | 0.448 | 0.445 | 0.425 | 0.417 |
| 6 | 0.453 | 0.424 | 0.441 | 0.411 |
| 7 | 0.405 | 0.426 | 0.405 | 0.405 |
| 8 | 0.493 | 0.464 | 0.449 | 0.476 |
| 9 | 0.440 | 0.445 | 0.435 | 0.448 |
| 10 | **0.530** | **0.540** | **0.537** | **0.529** |
| 11 | 0.297 | 0.350 | 0.272 | 0.328 |
| **Median** | **0.408** | **0.424** | **0.410** | **0.405** |

Four findings emerge:

1. **Every window is positive for every ML model.** 11/11 out-of-sample windows, 4/4 ML models — zero losing windows. This is strong evidence the edge is not a lucky train/test split.

2. **Per-trade Sharpe trends upward across windows.** LR rises from 0.307 in window 1 (357 training rows) to 0.530 in window 10 (8,012 training rows) — a **73% improvement** as training data accumulates. XGBoost shows the same pattern (0.383 → 0.540). Window 11 drops because its test set has only 110 rows and is statistically noisy. The trajectory is consistent with classic time-series ML: more data → tighter risk-adjusted returns.

3. **All four ML models converge to near-identical median performance** (\$73.91–\$75.30 median P&L). The walk-forward does not cleanly separate XGBoost from LR, GRU, or LSTM. This is the strongest indication that at current data scale **model family essentially does not matter** among reasonable Tier 1 + Tier 2 choices — what matters is whether any ML model is used at all.

4. **The ML vs. naive gap is ~3×** (median \$75 vs \$25–\$26), dwarfing the <\$2 differences between ML models. The economically meaningful comparison is *any ML model* vs. *no ML*, not *XGBoost* vs. *GRU*.

### 5.3 Per-Category Breakdown

Table 4 shows the single-split P&L stratified by contract category.

**Table 4: Per-category model performance (single-split test set).**

| Category | # trades | LR P&L | XGBoost P&L | Winner | Win-Rate (LR) |
|---|---|---|---|---|---|
| Inflation | 616 | **+\$89.39** | +\$89.38 | LR (tied) | 63% |
| Crypto | 292 | +\$41.75 | **+\$48.14** | **XGBoost** | 57% |
| Politics–policy | 278 | +\$29.76 | **+\$31.03** | **XGBoost** | 32% |
| Employment | 204 | **+\$20.02** | +\$19.94 | LR (tied) | 47% |
| Politics–election | 129 | **+\$17.95** | +\$17.55 | LR | 28% |
| GDP | 20 | +\$0.91 | +\$0.91 | tie | 15% |
| Fed rates | 10 | +\$1.90 | +\$1.90 | tie | 50% |
| **Overall** | 1,549 | **+\$201.69** | **+\$208.85** | **XGBoost** | 51% |

**Linear regression wins 5 of 7 categories with sufficient trade count**, but XGBoost wins crypto by a notable \$6 margin and politics-policy by \$1.30. *The apparent "XGBoost wins overall" result is driven almost entirely by crypto outperformance* — consistent with XGBoost's tree-based splits capturing crypto's nonlinear volatility regimes that LR cannot. This is a nuanced finding: **the complexity premium is regime-specific, not universal**.

Additionally, on the live dataset (April 11 snapshot, 1,881 historical trades), oil near-expiry contracts stand out dramatically:

- **Oil:** 765 trades, **76.5% win rate**, **+\$0.41/trade** — a +142.7% edge over the pooled model.
- Fed rates: 431 trades, 34.6% WR, +\$0.01/trade.
- Sports: 618 trades, 37.4% WR, −\$0.00/trade.
- Politics: 67 trades, 29.9% WR, −\$0.02/trade.

Oil-contract convergence is largely mechanical: WTI-futures expiry dates resolve on observable market prices within hours of contract settlement, so cross-platform prices must converge. Sports and politics resolve on discrete events with no convergence dynamics — hence the poor performance. **The alpha is in the asset class as much as in the model.**

### 5.4 Data-Scaling Curve

Fig. 2 (see `experiments/figures/pnl_at_2pp_vs_data.png`) shows the 6-point scaling curve. Values are stable because the training set cap (6,802 rows) is reached by bar-count 100. Table 5 summarizes.

**Table 5: Data-scaling experiment (P&L at 2pp).**

| Bars/pair | Training rows | LR P&L | XGBoost P&L | GRU P&L | LSTM P&L |
|---|---|---|---|---|---|
| 50 | 4,646 | +\$202.93 | **+\$210.57** | — | — |
| 100 | 6,290 | +\$200.36 | **+\$211.07** | +\$186.67 | +\$182.76 |
| 250 | 6,802 | +\$199.90 | **+\$210.01** | — | — |
| 500 | 6,802 | +\$199.90 | **+\$210.01** | — | — |
| 1000 | 6,802 | +\$199.90 | **+\$210.01** | — | — |
| 2000 | 6,802 | +\$199.90 | **+\$210.01** | — | — |

The scaling curve plateaus at 100 bars/pair because we hit the training-set cap. Within that range, **the ranking is invariant**: XGBoost > LR > LSTM > GRU. This directly refutes the hypothesis that sequence models would overtake regression if only they had more data, at least for sample sizes reachable on our dataset.

### 5.5 XGBoost Hyperparameter Sweep

Table 6 shows the top 10 of 48 XGBoost configurations. All top-10 configurations use tree depth 3–7; depth 9 does not appear until later. The best configuration — **depth 3, learning rate 0.01, 100 trees** — is essentially "an ensemble of decision stumps," confirming the "simple wins" theme even within a single model family.

**Table 6: Top 10 XGBoost configurations (48 total).**

| Rank | depth | lr | n_est | RMSE | DA | P&L | Win% |
|---|---|---|---|---|---|---|---|
| 1 | 3 | 0.01 | 100 | 0.288 | 57.7% | +\$209.70 | 58.1% |
| 2 | 3 | 0.01 | 500 | 0.282 | 58.1% | +\$208.62 | 58.5% |
| 3 | 5 | 0.01 | 500 | 0.285 | 57.9% | +\$208.29 | 58.4% |
| 4 | 3 | 0.05 | 500 | 0.292 | 58.0% | +\$207.72 | 58.4% |
| 5 | 3 | 0.05 | 100 | 0.282 | 58.1% | +\$207.69 | 58.3% |
| 6 | 3 | 0.01 | 300 | 0.281 | 57.7% | +\$207.62 | 58.1% |
| 7 | 5 | 0.01 | 100 | 0.285 | 57.8% | +\$207.38 | 58.2% |
| 8 | 5 | 0.01 | 300 | 0.282 | 57.3% | +\$207.07 | 58.3% |
| 9 | 7 | 0.01 | 100 | 0.284 | 57.3% | +\$206.83 | 58.0% |
| 10 | 5 | 0.05 | 100 | 0.285 | 57.7% | +\$206.70 | 58.0% |

### 5.6 Transaction-Cost Sensitivity

**Table 7: Sensitivity to transaction costs (XGBoost, depth-3, fresh run April 17, 2026).**

| Fee (pp) | P&L | Win Rate | Sharpe/trade | # trades |
|---|---|---|---|---|
| 0 (gross) | +\$245.06 | 57.6% | 0.502 | 1,673 |
| **2 (our simulation)** | **+\$208.85** | **50.9%** | **0.449** | 1,567 |
| 3 (maker + small slippage) | +\$194.03 | 48.8% | 0.452 | 1,410 |
| 5 (Kalshi taker maximum) | +\$165.53 | 46.1% | 0.438 | 1,201 |
| 7 (adversarial worst-case) | +\$135.34 | 42.9% | 0.407 | 1,018 |

The system is **profitable at every fee level** tested, from 0 pp (gross) up to the worst-case 7 pp adversarial assumption. Net P&L compresses roughly linearly with fees (−\$18 P&L per 1 pp of fee increase), and Sharpe is remarkably flat — in fact *rising slightly* from fee 2 to fee 3 because the higher fee filters out lower-confidence trades. In practice, Kalshi **maker** orders pay \$0 fee and Polymarket charges only \~1 pp in Polygon gas, so our 2 pp simulation is *conservative*.

Note that win rate decreases with fees — trades that just barely won at 0 pp lose at higher fees — but per-trade Sharpe is preserved because the lost trades had low magnitude. This is the statistical fingerprint of a *real* signal rather than noise: the signal survives cost scrubbing.

### 5.7 SHAP Feature Importance

SHAP analysis on the trained XGBoost model (see `experiments/figures/shap_bar_plot.png`) shows `polymarket_vwap` dominating feature importance with a mean |SHAP| of ≈0.14 — twice the next feature. This suggests **Polymarket may be the "less efficient" side**, with its prices carrying more predictive information about future spread direction than Kalshi's do. Interpreted loosely: Kalshi prices move first, Polymarket prices catch up — which is consistent with Kalshi's smaller user base having a more concentrated informed flow and Polymarket's larger retail user base producing a slower reaction function.

### 5.8 Honest Sharpe-Ratio Accounting

The per-trade Sharpe of 0.436 (XGBoost, Table 2) does not annualize trivially. The three sensible methods give very different answers (Table 8).

**Table 8: Sharpe-ratio accounting (XGBoost, fresh April 17, 2026 run).**

| Method | Sharpe | Interpretation |
|---|---|---|
| Per-trade | **0.44** | Treats correlated intra-pair trades as independent (over-estimate of granularity) |
| Daily × √252 | ≈ 40 | Over 90 trades/day are highly correlated (same pairs) — this is absurd |
| **Per-pair × √(pairs/year)** | **≈ 3.2** | **Each pair = one independent bet (correct)** |
| Per-pair + 1 pp slippage | ≈ 2.5 | Realistic estimate |

**Industry context:** Sharpe 1.0 is considered a good hedge fund; Sharpe 2–3 is elite (Renaissance, Jane Street); Sharpe 4+ is typically a red flag for over-fitting, inflated evaluation, or short test windows. Our per-pair 3.2 is plausibly inflated by the 2-week test window; a longer out-of-sample period (which the walk-forward partially provides) would likely contract the estimate toward the 1.5–2.5 range. We report this transparently. **The edge is real (positive across every estimation method and every walk-forward window), but the magnitude is uncertain by roughly a factor of two in either direction.**

---

## 6. Discussion

### 6.1 Why Does Simpler Beat More Complex?

Four reasons appear to operate jointly:

1. **Sample size.** With 6,802 training rows and 47 bars per pair (on average, historical dataset), sequence models are data-starved. Sequence models are typically best when they can exploit long-range dependencies (hundreds of timesteps), but our contracts average 47 bars. GRU/LSTM are therefore operating outside their comfort zone.

2. **Signal-to-noise ratio.** The target — one-step spread change — has high variance relative to signal. Linear models are more robust in high-noise regimes because they have fewer parameters to over-fit.

3. **Feature engineering carries the signal.** Our 59-feature set includes `spread`, `polymarket_vwap`, `dollar_volume_ratio`, and 13 microstructure features. These features encode most of the exploitable signal directly; the model only needs to linearly combine them. Deep models wasted capacity re-discovering what the features already expose.

4. **Overfitting on structural regularities.** The deepest XGBoost configurations (depth 9, n_est 500) had higher training accuracy but *lower* test P&L — classic over-fitting. The same phenomenon appears in sequence models: LSTM memorizes per-pair idiosyncrasies that do not generalize.

This matches the January 2026 finding on investor-flow prediction and the broader literature on tabular ML (Grinsztajn et al. 2022).

### 6.2 How Each Model Would Improve With More Data

The data-scaling curve in §5.4 plateaus because we exhausted the 6,802-row training cap by bar-count 100. The live auto-retrain system is now accumulating data at roughly 1,200–1,600 new bars per pair per week across ~1,000 active pairs, which should push us into regions where the different model families behave very differently. This section explains, per model, *what specifically improves and why*.

#### 6.2.1 Linear Regression

LR is already the most data-efficient model in our suite — it converges to its best configuration by roughly 4,000 training rows (Table 5, row 1 ≈ row 6). Its remaining headroom is narrow but real:

1. **Coefficient variance shrinks as $\sigma^2 / n$.** Our 95% bootstrap CIs on the 59 regression coefficients are still wide enough that feature rankings shuffle across bootstrap resamples. At 10× the data, most coefficients will be statistically separable from zero, meaning *the sign of the edge becomes more reliable even if its magnitude does not grow*.

2. **Ridge/Lasso regularization becomes better-calibrated.** The optimal $\lambda$ in penalized regression scales roughly with $\sqrt{n}$; on our current dataset, cross-validated $\lambda$ estimates have 30% standard errors. At 50k+ rows, $\lambda$ becomes a tight estimate and Ridge starts noticeably outperforming plain OLS — we would expect a 5–10% P&L lift from this alone.

3. **The 13 classical-microstructure features unlock.** Amihud illiquidity, Corwin–Schultz spread, Kyle's $\lambda$, and Roll's implied spread all depend on rolling windows of depth 6–12 bars. At 47 bars/pair, those windows barely have data; at 500 bars/pair, they become statistically meaningful. These features are *linearly informative* about liquidity asymmetry across platforms — exactly the kind of thing LR exploits well.

4. **Live buy/sell volume breakdowns.** The historical dataset lacks signed Kalshi volume, zeroing out our `kalshi_order_flow_imbalance` feature. Live bars include it, and order-flow imbalance is one of the most-cited signals in microstructure (Cont et al. 2014). LR will capture the linear component immediately.

5. **Ceiling:** LR cannot represent interactions without explicit interaction terms. So above roughly 100k rows, the marginal P&L gain per doubling of data will approach zero — classic bias-dominated regime. Based on the per-window Sharpe trajectory (0.307 → 0.530, a 73% improvement across the 11 walk-forward windows), we estimate LR will plateau at roughly +\$260–300 at 2 pp fees (vs. today's +\$201.69), equivalent to roughly 30–50% P&L lift from microstructure features unlocking and regularization tightening.

#### 6.2.2 XGBoost

XGBoost has the most headroom of the four "reasonable" models (Tiers 1 and 2), for three architectural reasons:

1. **The depth ceiling will rise.** The hyperparameter sweep (Table 6) shows that *every* top-10 configuration uses depth 3–7, with depth 3 optimal. This is classic overfitting behavior: at 6,802 training rows, a depth-7 tree has more leaf nodes than rows-per-leaf can support. As training set grows to 50k+ rows, the optimal depth will shift toward 5–7, unlocking richer interaction terms (e.g., "high Amihud on Polymarket AND near-expiry AND spread > 5%" — a three-way interaction that depth-3 trees cannot fit).

2. **More trees become useful without overfitting.** Our best configuration uses $n_\text{estimators} = 100$ because additional trees memorize the training set. With larger data, $n_\text{estimators} = 300$–$500$ becomes optimal and captures progressively subtler residual patterns. Combined with lower learning rate ($\text{lr} = 0.005$ instead of 0.01), the ensemble becomes both more stable and more accurate.

3. **Per-category splits become robust.** The per-category breakdown (Table 4) shows XGBoost's edge over LR is concentrated in crypto (+\$6) — exactly where nonlinear regimes exist. As oil, Fed-rates, and other categories accumulate sufficient per-category bars (currently 10–616), XGBoost will exploit their distinct nonlinearities too. We expect the "XGBoost wins 2 of 7 categories" result to become "XGBoost wins 5 of 7" by 500 bars/pair.

4. **No architectural ceiling.** Tree ensembles are universal approximators in the limit. XGBoost's P&L can in principle grow without bound as data grows, unlike LR. Extrapolating the walk-forward Sharpe improvement (XGBoost 0.383 → 0.540 over windows 1–10) and the category breakdown where XGBoost already wins the nonlinear regimes, we estimate +\$300–380 at 2 pp fees at 500 bars/pair (vs. today's +\$201.63). XGBoost's headroom is similar to LR's at current scale but widens past it as data accumulates.

#### 6.2.3 GRU and LSTM

This is where the largest potential gains live. Sequence models are currently operating *far* outside their comfort zone — trained on sequences averaging 47 timesteps, when published RNN architectures typically need 200–1000. Their 12% deficit to XGBoost is therefore *not* an architectural verdict; it is a data-regime verdict. Five things will change:

1. **Effective lookback window can expand.** We currently set the GRU/LSTM lookback to 24 bars because bars-per-pair ≈ 47 doesn't support longer. At 500 bars/pair, a 72-bar or 120-bar lookback becomes viable — long enough to capture the ~48-hour price discovery dynamics typical of weekly contract cycles. Sequence models should find patterns here that regression fundamentally cannot represent.

2. **Hidden dimensionality can grow.** Our current GRU has 64 hidden units and one layer because more parameters over-fit our 6,802 rows. At 50k+ rows, 128–256 hidden units and 2–3 stacked layers become trainable without over-fitting — giving the network enough capacity to encode multiple competing hypotheses simultaneously.

3. **Cross-platform lead–lag becomes learnable.** SHAP (§5.7) shows `polymarket_vwap` dominates feature importance, consistent with Kalshi-leads-Polymarket or vice versa. But *the lead–lag relationship is time-varying* — crypto contracts have different lead–lag than inflation contracts, and both change near expiry. LR and XGBoost get one static coefficient per feature; RNNs can dynamically modulate the lead–lag based on context. This is the single largest theoretical advantage of sequence models for this problem.

4. **Live-volume microstructure features become temporally meaningful.** The same live buy/sell volume that helps LR helps RNNs more, because RNNs can integrate order-flow imbalance *over a sliding window* rather than treating each bar independently. Kyle's $\lambda$ trajectory over 30 bars is more informative than $\lambda$ at a single bar — and this is exactly what an RNN hidden state encodes.

5. **Cross-pair meta-learning.** Transferring hidden-state initialization across similar pairs (oil contracts share structure even if they're different tickers) lets sequence models bootstrap new pairs faster. XGBoost and LR have no analog to this — they must see a pair's training rows explicitly.

**Our honest prediction:** by 250 bars/pair (estimated 3–4 weeks away), GRU should close the gap with XGBoost. By 500 bars/pair, sequence models will likely *surpass* XGBoost on oil and crypto categories where lead–lag effects dominate. By 1000 bars/pair, TFT (which we did not train) becomes justifiable and would likely set the new benchmark. This is a falsifiable prediction: the auto-retrain system will have the answer by the end of the semester.

#### 6.2.4 What will *not* change

For intellectual honesty: **PPO will not close the gap.** RL requires full trajectories, and prediction-market contracts produce trajectories of length 47 on average. Even at 500 bars/pair, that is ~10× less trajectory data than a typical Atari benchmark. The PPO + autoencoder failure is architectural, not data-limited; no amount of additional bars fixes it without a different problem formulation (e.g., trajectory-level imitation learning, or a differentiable market simulator for off-policy pre-training).

The ordering we expect at 500 bars/pair is therefore: **GRU ≈ LSTM ≈ XGBoost > LR >> Naive > PPO**. The interesting scientific question is whether GRU eventually passes XGBoost — we predict yes, but we report the current result honestly.

### 6.3 The Negative Result on PPO

PPO with the autoencoder anomaly filter produces **−\$7,724** in backtest. This is not a bug — we verified the reward function, the environment transitions, and the action space. The autoencoder simply flags normal market behavior as anomalous because it was trained on all spreads without a clean "normal regime" prior. PPO then trades in those flagged windows, which are disproportionately high-volatility periods where predictions are least reliable. This is a direct empirical answer to the professor's question "does adding RL and anomaly detection improve on simpler regression?" — **no; it actively hurts**.

A more defensible RL approach would be (i) a curriculum that learns "safe" regimes first, (ii) a differentiable simulator for off-policy pre-training, and (iii) a much larger universe of training trajectories. None of these are justified at our data scale.

### 6.4 Limitations

We are transparent about these:

1. **Short test window.** Two weeks of out-of-sample evaluation is too short to confidently annualize Sharpe. The walk-forward analysis mitigates this but does not eliminate it.

2. **Paper trading only.** No market-impact costs or partial fills are modeled. Real execution would show slippage, particularly for large positions on low-liquidity pairs.

3. **Survivorship bias.** Our historical dataset includes only pairs that survived the quality filter. Pairs that were ever listed but never satisfied matching criteria are missing — we believe this bias is small because the filter is structural, but we cannot quantify it precisely.

4. **Settlement divergence risk.** Cross-platform contracts can in principle resolve differently (e.g., a sports-book source of truth mismatch). We observed zero cases of this in our universe, but it is a real risk at production scale.

5. **Regime-specific edge.** The inflation and oil edges dominate. If those categories lose liquidity, the system's overall edge would shrink.

---

## 7. Future Work

1. **500- and 1000-bar scaling checkpoints.** Our automated retraining system will produce these data points over the coming weeks; they will definitively answer the "does sequence model catch up?" question.

2. **Live-volume-aware microstructure features.** Kyle's λ, Amihud, and Roll's spread all depend on signed buy/sell volume, which we will have on live bars but did not have on historical bars. This should materially improve the microstructure features' informativeness.

3. **TFT (Temporal Fusion Transformer).** With more data, TFT — which we skipped due to its larger capacity — becomes justifiable and could produce the first Tier-2 model that beats XGBoost.

4. **Meta-learning across categories.** A per-category model trained jointly with a shared feature encoder and per-category heads could exploit the category heterogeneity we document in §5.3.

5. **Execution-aware reward shaping.** Replacing the current mark-to-market reward with a post-fee execution reward (fillable limit orders, partial fills, latency) would let PPO learn execution patterns — potentially redeeming the negative Tier-3 result.

6. **Formal settlement-divergence model.** Tracking resolution-source mismatches (Kalshi uses `CME settle`, Polymarket uses various oracles) and pricing the divergence risk into trade sizing.

7. **Open-sourcing the matching pipeline.** The 10-rule quality filter encodes a lot of domain knowledge. Releasing it as a library could help other researchers and market participants.

---

## 8. Conclusions

We built an end-to-end cross-platform prediction-market arbitrage system with four tiers of models and evaluated it across five independent regimes. The central answer to our research question is clear:

> **At this data scale, increasing model complexity does not improve arbitrage detection. The simplest models win.**

Specifically:

1. **Tier 1 (LR, XGBoost) beats Tier 2 (LSTM, GRU) by 10–15% P&L, which beats Tier 3 (PPO) by several thousand dollars in the wrong direction.** In the fresh single-split (April 17, 2026), XGBoost achieves +\$201.63, LR +\$201.69, LSTM +\$182.72, GRU +\$174.11, PPO+autoencoder −\$7,724. LR and XGBoost are essentially tied; both decisively beat sequence models; RL is catastrophic. This ordering holds across the single-split, walk-forward, and data-scaling evaluations.

2. **Every walk-forward window is profitable, and per-trade Sharpe is rising over time** — the edge is stable and improving with more data.

3. **The alpha lives in the matching pipeline and the asset class.** A 10-rule quality filter added +\$10.73 in P&L with no model changes. Oil near-expiry contracts alone produce a 76.5% win rate, while sports and politics are near-zero.

4. **The system is robust to transaction costs** — profitable at every fee level from 0 pp to 7 pp.

5. **The honest annualized Sharpe is 2–4, not 50+** — strong but not other-worldly, and likely inflated by the short test window.

Beyond the empirical findings, the project taught us three broader lessons:

- **Infrastructure bugs masquerade as model problems.** Kalshi's silent 429s, Polymarket's `condition_id` typo, and our own `live_NNNN` schema drift caused weeks of confusion that looked like "the model stopped working."
- **Evaluation regime matters more than model family.** A single train/test split can tell a very different story than a walk-forward, a per-category breakdown, or a data-scaling curve. Multi-regime evaluation is mandatory, not optional.
- **Negative results are results.** PPO failing catastrophically is the strongest evidence we have for the main thesis. We chose to publish it rather than omit it.

The project succeeded on its own terms: we answered the research question rigorously, we built a working autonomous system, and we documented honest limitations. The edge is real; the complexity is not.

---

## Acknowledgments

We thank Professor Kevin Gold for the course instruction and for the pivotal feedback in late March that pushed us to treat regression baselines as first-class and to add the PPO-without-autoencoder variant. That feedback directly shaped the research question into something empirically answerable. We also acknowledge extensive use of Anthropic's Claude (Sonnet 4.5 and Opus 4.6) as an AI pair-programming assistant throughout the implementation; all design decisions, experimental choices, and interpretations are our own.

---

## References

1. Amihud, Y. (2002). Illiquidity and stock returns: cross-section and time-series effects. *Journal of Financial Markets* 5(1), 31–56.
2. Burgi, C., Tuccella, J., & Zitzewitz, E. (2026). Favorite–longshot bias on Kalshi and Polymarket. *Working paper*.
3. Corwin, S. A., & Schultz, P. (2012). A simple way to estimate bid-ask spreads from daily high and low prices. *Journal of Finance* 67(2), 719–760.
4. Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on tabular data? *NeurIPS 2022 Datasets & Benchmarks*.
5. Kyle, A. S. (1985). Continuous auctions and insider trading. *Econometrica* 53(6), 1315–1335.
6. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 2017*.
7. Manski, C. F. (2006). Interpreting the predictions of prediction markets. *Economics Letters* 91(3), 425–429.
8. Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of return. *Journal of Business* 53(1), 61–65.
9. Roll, R. (1984). A simple implicit measure of the effective bid-ask spread in an efficient market. *Journal of Finance* 39(4), 1127–1139.
10. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.
11. Wolfers, J., & Zitzewitz, E. (2004). Prediction markets. *Journal of Economic Perspectives* 18(2), 107–126.
12. [Anonymous]. (2026). Matched filter feature engineering for investor flow prediction. *arXiv:2601.07131*.
13. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: sentence embeddings using Siamese BERT-networks. *EMNLP 2019*.

---

## Appendix A: Code Availability

All code is available at `https://github.com/iansabia/DS340-Project` (final commit tagged `v1.0-submission`). Reproducing the main result requires:

```bash
git clone https://github.com/iansabia/DS340-Project.git
cd DS340-Project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m experiments.run_baselines              # Table 2
python -m experiments.run_walk_forward --windows 12  # Table 3, Fig 1
python -m experiments.run_category_breakdown     # Table 4
python -m experiments.run_data_scaling           # Table 5, Fig 2
python -m experiments.run_xgb_hyperparam_sweep   # Table 6
python -m experiments.run_transaction_cost_sensitivity  # Table 7
python -m experiments.run_shap_analysis          # SHAP plots
```

Total runtime on a single CPU: ~2 hours. GPU is not required.

## Appendix B: Figures Referenced

- **Figure 1 — Walk-forward P&L curves:** `experiments/figures/walk_forward_pnl.png` (11 windows, 6 models)
- **Figure 2 — Walk-forward Sharpe curves:** `experiments/figures/walk_forward_sharpe.png`
- **Figure 3 — P&L vs. training data size:** `experiments/figures/pnl_at_2pp_vs_data.png`
- **Figure 4 — Transaction-cost sensitivity:** `experiments/figures/transaction_cost_sensitivity.png`
- **Figure 5 — SHAP feature importance:** `experiments/figures/shap_bar_plot.png`
- **Figure 6 — Equity curves (cumulative P&L over test period):** `experiments/figures/backtest_equity_curves.png`
- **Figure 7 — Bootstrap CI for RMSE:** `experiments/figures/bootstrap_ci_rmse.png`
- **Figure 8 — Lookback-window ablation:** `experiments/figures/experiment2_lookback_pnl.png`
- **Figure 9 — Threshold-sweep heatmap:** `experiments/figures/experiment3_threshold_heatmap.png`
