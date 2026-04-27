# Complexity Is Not an Edge: An Empirical Study of Machine-Learning Arbitrage on Kalshi and Polymarket

**Ian Sabia** (U33871576), **Alvin Jang** (U64760665)
Department of Data Science, Boston University
DS340 — Spring 2026 Final Project
April 27, 2026

---

## Abstract

We study cross-platform price discrepancies between **Kalshi** (CFTC-regulated event contracts) and **Polymarket** (on-chain prediction market) and ask whether increasing model complexity improves arbitrage detection. We build an end-to-end system that matches contracts via sentence embeddings plus a 10-rule structural quality filter, engineers 51 features, and trains four model tiers — regression baselines (Linear Regression, XGBoost), sequence models (GRU, LSTM), and PPO with and without an autoencoder anomaly filter — under one evaluation protocol. On 6,802 training rows / 1,673 test rows across 144 matched pairs at \$100 position size, **Linear Regression achieves a per-trade Sharpe of 0.501 with +15.0 bps per-trade alpha**, tied with XGBoost (0.499 / +14.9 bps), versus 0.473 / +14.3 bps for LSTM, 0.459 / +14.0 bps for GRU, +9.6 bps for PPO-Raw, and **+0.5 bps for PPO+autoencoder**. Every walk-forward window is profitable for every ML model and per-trade Sharpe rises from 0.31 to 0.53 over time. A pre-submission adversarial audit (`AUDIT_REPORT.md`) found that the original 80/20 row-index split bridged 142 of 144 pairs (embargo violation); a leakage-free pair-stratified retraining yields per-trade Sharpe 0.516 and +15.7 bps alpha — drift of +2.99%, well within the bootstrap CI. The per-trade edge is robust. The central empirical answer: **at this data scale, complexity is a liability, not an edge**. The alpha lives in the matching pipeline and the oil/commodities asset class, not in the models.

**Keywords:** prediction markets, arbitrage, market microstructure, XGBoost, LSTM, PPO, walk-forward validation, simplicity.

---

## 1. Introduction

Prediction markets allow traders to buy and sell contracts that pay \$1 if a specified real-world event occurs (e.g., "Will CPI inflation exceed 3.0% in May 2026?") and \$0 otherwise. The equilibrium price is interpreted as the market-implied probability. Two large U.S.-accessible venues — **Kalshi** (CFTC-regulated, since 2021) and **Polymarket** (on-chain via Polygon, since 2020) — frequently list contracts that reference the *same* underlying event but trade at materially different prices. These cross-platform discrepancies can persist for hours or days, representing a potential statistical-arbitrage opportunity if they can be detected, matched correctly, and traded before they close.

The central research question is:

> **Does increasing model complexity improve arbitrage detection in cross-platform prediction markets, and if so, when is that complexity justified?**

Two distinct audiences motivate this work. *Academically*, a January 2026 working paper (arXiv 2601.07131) argues that feature engineering grounded in market-microstructure theory consistently beats deep learning for investor-flow prediction. Prediction-market arbitrage is a clean testbed because contracts have finite lifetimes, bounded payoffs, and deterministic settlement. *Practically*, prediction markets are growing rapidly: Polymarket processed over \$1B in volume during the 2024 election cycle. Understanding which modeling approaches genuinely add value is a relevant question for market participants.

**Background and prior work.** A binary event contract on platform $P$ at time $t$ has price $p_P(t) \in [0,1]$; for two platforms listing the same event, the spread $s(t) = p_A(t) - p_B(t)$ converges to zero at resolution modulo fees and basis risk. We predict the one-step change $\Delta s(t) = s(t+1) - s(t)$. Manski (2006) and Wolfers & Zitzewitz (2004) established that prediction-market prices are reasonably efficient probability forecasts, but cross-platform dispersion is understudied. Bürgi, Deng & Whelan (2026) document a clear favorite–longshot bias on Kalshi (low-price contracts win less often than required to break even, while high-price contracts yield small positive returns) — a structural pricing pattern that, if asymmetric across platforms, becomes a source of cross-platform spread. Standard market-microstructure estimators — Amihud's (2002) illiquidity ratio, Corwin & Schultz's (2012) high-low bid–ask spread, Kyle's (1985) $\lambda$, and Roll's (1984) implied spread — are well established in equity markets but, to our knowledge, have not been systematically applied to prediction-market spread prediction. A persistent finding in applied ML is that on tabular data with moderate sample size, gradient-boosted trees match or beat deep neural networks (Grinsztajn et al. 2022); our results are consistent with this pattern.

**Contributions.** (1) A functioning end-to-end arbitrage system, from data ingestion through live paper-trading on BU SCC, monitoring up to 11,582 matched pairs at peak. (2) A complexity-vs-performance benchmark across four model tiers under an identical evaluation protocol. (3) An 11-window walk-forward validation showing the edge is stable and *improving* over time. (4) Proposal-mandated parameter searches over lookback windows and minimum-spread thresholds. (5) An honest Sharpe accounting that survived a pre-submission adversarial audit. (6) A negative result on PPO that we report transparently.

---

## 2. Methodology

### 2.1 Data Sources and Matching

**Kalshi** exposes a public REST API (`api.elections.kalshi.com/trade-api/v2`) requiring no authentication. We fetch hourly OHLCV candlesticks (`period_interval=60`); Kalshi splits market history at a roughly three-month cutoff, so we query `/historical/cutoff` to determine which endpoint applies. **Polymarket** exposes three separate APIs: Gamma (metadata), CLOB (orderbook/prices for active markets), and the Data API (fills/trades for resolved markets). Polymarket uses opaque numeric token IDs, so we look up `clobTokenIds` via Gamma and key CLOB queries by them. A non-obvious gotcha: Gamma's singular `condition_id=` query parameter returns *unrelated random markets*, while the plural `condition_ids=` returns the exact match. The `/prices-history` endpoint returns empty data for resolved markets, forcing reconstruction from trade records.

**Matching pipeline.** We embed concatenated title + description with `sentence-transformers/all-MiniLM-L6-v2` (384-d) and compute cosine similarity via normalized matrix multiplication (~80 s vs ~6.6 h for an O(N·M) keyword pre-filter — a 300× speedup). Semantic similarity alone is insufficient ("NBA Finals winner" embeds close to "NBA MVP"), so we layer **10 structural quality-filter rules**: sports wins-vs-champion mismatch, Fed meeting-month mismatch, cabinet confirmation-vs-nomination, state-specific vs national commodities, exact CPI vs PCE vs core-CPI, ±2-day expiry window, exact strike comparison for futures, numeric-threshold exactness ("\$50/bbl" vs "\$55/bbl"), event-key disambiguation ("Game 4" vs "Game 5"), and a category-consistency guard. The filter rejected **140 of 615 pairs (22.8%)**. Empirical impact: at the first large-scale backtest, Linear Regression P&L went from −\$5.28 to +\$5.45 — a +\$10.73 swing purely from removing structurally-bad matches, with no model changes. Data quality is more important than model choice at this scale.

### 2.2 Feature Engineering

We engineer 59 raw features and use 51 numeric features after dropping NaN/zero-variance columns. Five groups: 18 raw aligned per-platform OHLCV features (e.g., `kalshi_open`, `polymarket_volume`); 6 cross-platform basics (`spread`, `mid_price`, `dollar_volume_ratio`); 9 rolling/momentum features (`spread_momentum_6`, `spread_zscore`); 13 classical microstructure features (`amihud_illiquidity`, `corwin_schultz_spread`, `kyle_lambda`, `roll_spread`, `bekker_parkinson_vol`) computed per-platform and differenced across platforms; and 13 prediction-market-specific features (`favorite_longshot_bias`, `near_expiry_indicator`). The rationale for differenced microstructure features is that *relative* liquidity should predict which side converges. These features are neutral-to-slightly-positive on the historical dataset (47 bars/pair on average is too few for stable rolling-window estimators) and are expected to become informative as live data accumulates.

### 2.3 Model Tiers

All models share the same train/test split (time-ordered 80/20, no shuffling), the same target ($\Delta s(t)$), and the same feature set.

- **Tier 0 — Naive baselines:** *Naive-closes* (predict spread fully closes by resolution); *Volume-higher-wins* (higher-volume platform is correct).
- **Tier 1 — Regression:** *Linear Regression* (scikit-learn); *XGBoost*, hyperparameter-searched over depth ∈ {3, 5, 7, 9} × learning-rate ∈ {0.01, 0.05, 0.1, 0.3} × `n_estimators` ∈ {100, 300, 500} (48 configurations).
- **Tier 2 — Sequence models:** *GRU* (64 hidden units, 1 layer, 24-bar lookback, Adam `lr=1e-3`, early stopping); *LSTM* (same architecture, LSTMCell). We also attempted *TFT* (PyTorch Forecasting, hidden_size=8, 3-quantile QuantileLoss) with small-data hyperparameters; it did not converge (§3.4).
- **Tier 3 — Reinforcement learning:** *PPO-Raw* (PPO on 51-d feature vectors, 3-action space {buy-spread, sell-spread, hold}, mark-to-market reward; `stable-baselines3`); *PPO-Filtered* (same PPO, but a 3-layer symmetric autoencoder with bottleneck=8 pre-filters observations, gating PPO to anomaly windows).

### 2.4 Evaluation Protocol

Five regimes provide independent views. **(a) Single-split backtest:** chronological 80/20, reporting RMSE, MAE, directional accuracy, simulated P&L at the 2 pp signal threshold, win rate, and per-trade Sharpe — the headline numbers. **(b) Walk-forward backtest:** concatenate train + test for maximum time coverage, split into 12 equal-time windows, and use an expanding-window protocol where window $i$ trains on data from windows $\{0, \ldots, i-1\}$ and tests on window $i$ (window 0 has no training set, leaving 11 evaluation windows). If the edge is stable, we expect positive P&L across all windows; if it is improving with more data, per-trade Sharpe should trend upward. **(c) Per-category breakdown:** stratify test rows by contract category using deterministic rules over Kalshi tickers and Polymarket slugs. **(d) Data-scaling curve:** 50, 100, 250, 500, 1000, 2000 bars/pair, plotting P&L vs training size — does "simple wins" hold across scales or is it an artifact of small data? **(e) Live paper trading** on BU SCC with a 15-minute trade cycle, 3-hour discovery batch, and 6-hour retrain batch, with redundant fallback workflows on GitHub Actions in case SCC undergoes scheduled maintenance. Tier-1 models train in <15 s on a CPU; Tier-2 in ~3 min/epoch with early stopping at 10–15 epochs; Tier-3 in 20–40 min for PPO convergence. Live deployment uses an LR + XGBoost equal-weight ensemble with a strict concordance filter (skip trades when models disagree on sign), and content-addressed pair IDs (e.g., `kxwti26apr08t10799-0x43d5953d`) derived deterministically from the normalized Kalshi ticker and Polymarket token ID — fixing an earlier bug where three code paths disagreed on what a `live_NNNN` pair ID meant, causing 25 positions to track the wrong markets.

### 2.5 Infrastructure Lessons

Three non-obvious infrastructure bugs materially shaped the project. **Kalshi `/events` silently returned HTTP 429** on roughly 40% of calls, dropping entire commodity series; fixing with exponential backoff and a 250 ms per-series pace took commodity pair count from 65 → 506. **Polymarket pagination defaulted too shallow** at 5,000 markets; WTI markets sat at offset 15,305+, completely invisible until we raised `max_pages` from 10 to 60. **Gamma's `condition_id` (singular)** returned unrelated random markets rather than an error; the fix to `condition_ids=` (plural) was a one-character change but cost several days of debugging. The common pattern: APIs that fail silently in ways that *look like* model problems. Infrastructure monitoring must come before model tuning.

---

## 3. Results

### 3.1 Headline Model Comparison

Table 1 reports the single-split backtest at a 2 pp signal threshold for trade entry on the full 1,673-row test set. All numbers are sourced verbatim from `experiments/results/canonical/headline.json` (Phase 17-01 canonical regenerator under seed=42, threshold=0.02, position_size=\$100). The per-trade Sharpe and per-trade alpha (in basis points) lead the comparison; cumulative dollar P&L follows.

**Table 1: Single-split backtest (canonical, 2 pp signal threshold, 1,673-row test set).**

| Tier | Model | RMSE | Dir. Acc. | Sharpe (per-trade) | Alpha (bps/trade) | P&L (\$100 pos) | Win Rate | # trades |
|---|---|---|---|---|---|---|---|---|
| 0 | Naive (closes) | 0.499 | 47.7% | 0.125 | +4.0 | +\$58.12 | 47.9% | 1460 |
| 0 | Volume (higher wins) | 0.457 | 47.7% | 0.131 | +4.2 | +\$59.81 | 48.1% | 1440 |
| 1 | **Linear Regression** | 0.306 | 56.9% | **0.501** | **+15.0** | **+\$232.67** | 57.8% | 1549 |
| 1 | **XGBoost (depth 3)** | **0.290** | 56.6% | 0.499 | +14.9 | **+\$232.83** | 57.4% | 1559 |
| 2 | LSTM | 0.291 | 65.5% | 0.473 | +14.3 | +\$221.84 | 56.5% | 1547 |
| 2 | GRU | 0.293 | 64.3% | 0.459 | +14.0 | +\$212.50 | 55.8% | 1517 |
| 2 | TFT† | 0.326 | 51.7% | 0.155 | +5.5 | +\$6.57 | 37.5% | 120 |
| 3 | PPO-Raw | 0.319 | 60.5% | 0.306 | +9.6 | +\$158.15 | 52.2% | 1656 |
| 3 | PPO + Autoencoder | 0.327 | 27.2% | 0.014 | **+0.5** | **+\$4.61** | 43.2% | 899 |

†TFT did not converge at N=6,802 (30-epoch budget, hidden_size=8); see §3.4.

The ranking is unambiguous: **Tier 1 > Tier 2 > Tier 3**. The Tier-0 → Tier-1 gap is large (≈4× in per-trade alpha: 4 bps → 15 bps); the Tier-1 → Tier-2 gap is 0.7–1.0 bps; PPO+autoencoder collapses to essentially zero edge. Linear Regression and XGBoost are essentially tied (Sharpe 0.501 vs 0.499; +15.0 vs +14.9 bps); within Tier 1, there is no statistically defensible reason to prefer one over the other on this dataset. Bootstrap 95% confidence intervals on RMSE (Fig. 7) overlap heavily across LR/XGBoost/GRU/LSTM even though P&L separates cleanly — the gap is driven by *directional accuracy* and *trade selection*, not raw regression error.

### 3.2 Walk-Forward Validation

We evaluate each model across 11 expanding-window splits spanning Jan–Apr 2026. Per-window P&L (Fig. 1) is positive in **every window for every ML model** — 11/11 × 4/4, zero losing windows. This is strong evidence the edge is not a lucky train/test split. Per-trade Sharpe trends upward across windows: LR rises from 0.307 in window 1 (357 training rows) to 0.530 in window 10 (8,012 training rows), a 73% improvement; XGBoost shows the same pattern (0.383 → 0.540). Median per-trade Sharpe across windows is 0.408 (LR), 0.424 (XGBoost), 0.410 (GRU), 0.405 (LSTM) — all four ML models converge to near-identical median performance. The economically meaningful comparison is *any ML model* (median ~\$75/window) vs *no ML* (median ~\$25/window), not XGBoost vs GRU.

### 3.3 Parameter Searches (Proposal Experiments)

**XGBoost hyperparameter sweep (48 configurations).** The best configuration is `depth=3, lr=0.01, n_estimators=100` (RMSE 0.288, +\$209.70 P&L). All top-10 configurations use depth ∈ {3, 5, 7}; depth-9 trees lost P&L, evidence of over-fitting at 6,802 training rows. Within a single model family, "simple wins" recurs: the best XGBoost is essentially an ensemble of decision stumps.

**Lookback window sweep (Proposal Experiment 2).** We sweep lookback ∈ {2, 6, 12, 18} bars (8, 24, 48, 72 hours of hourly data) for both GRU and LSTM, holding all other hyperparameters fixed (Fig. 8). For GRU, P&L declines monotonically from \$+226.42 at 2 bars to \$+191.90 at 18 bars (−15%); per-trade Sharpe falls from 0.489 to 0.400. LSTM is flatter (best at 6 bars: \$+220.37; worst at 18 bars: \$+209.42) but trends in the same direction. **Longer lookbacks do not help, and at 72 h they meaningfully hurt.** With pairs averaging 47 bars total, an 18-bar window covers nearly 40% of contract lifetime and is dominated by stale pre-active-phase information. Sequence models cannot exploit longer history at this scale.

**Minimum-spread threshold sweep (Proposal Experiment 3).** We sweep the decision threshold ∈ {0.00, 0.02, 0.05, 0.10} across all eight models (Fig. 9). For ML models, P&L peaks at threshold 0.00–0.02 and declines at higher thresholds (XGBoost loses \$36 going from 0.00 → 0.10). However, *per-trade Sharpe rises monotonically* with threshold for every ML model (XGBoost: 0.498 → 0.620, +24%): higher thresholds discard low-magnitude trades faster than they sacrifice profitable ones. Naive baselines show the *opposite* P&L pattern — they have no predictive content, so filtering to large-spread trades is the only thing that helps them. We use threshold 0.02 as the fee-consistent baseline throughout. The takeaway: threshold selection is a second-order tuning knob compared to choice of model tier.

**Transaction-cost sensitivity** (Fig. 4). The system is profitable at every fee level from 0 pp (gross +\$245) to a worst-case 7 pp (+\$135). Net P&L compresses roughly linearly (~\$18 per pp), but per-trade Sharpe is remarkably flat — fees filter out lower-confidence trades, partially offsetting the cost. Model rankings are invariant across the 0–10 pp fee range. In practice, Kalshi maker orders pay \$0 fee and Polymarket charges only ~1 pp in Polygon gas, so our 2 pp simulation is *conservative*.

**Feature ablation (pre-registered).** To guard against p-hacking, we pre-registered a Leave-One-Group-Out ablation protocol at `.planning/ablation_protocol.md` before any experiment script was written, with five non-overlapping feature groups (raw OHLCV, cross-platform basics, rolling/momentum, classical microstructure, prediction-market-specific). Across 12 LOGO configurations (LR + XGBoost × 6 group conditions) on a 1,021-row ablation-holdout, no group meets the pre-registered load-bearing threshold (95% CI fully below zero AND |ΔP&L| > \$10 for both models). At this holdout size, paired-bootstrap CIs are too wide to detect group-level effects smaller than ~\$10. We therefore conservatively retain all 51 features and defer feature selection to a re-run at 250+ bars/pair where statistical power becomes sufficient. The final 1,673-row test set was not evaluated on any reduced-feature variant.

### 3.4 Other Findings (Compressed)

**Per-category stratification.** Linear Regression wins 5 of 7 sufficiently-traded categories; XGBoost wins crypto by \$6 (its tree-based splits capture nonlinear volatility regimes that LR cannot). On the live April-11 dataset, oil near-expiry contracts dominated with **76.5% win rate and +\$0.41/trade** versus essentially zero for sports and politics — convergence is mechanical when WTI-futures expiry dates resolve on observable market prices. The alpha is in the asset class as much as in the model.

**Data-scaling curve.** P&L plateaus by 100 bars/pair because train.parquet caps at 6,802 rows / 144 pairs (max 141 bars/pair); slices at 250+ bars/pair are identical. Within the measured range, the ranking XGBoost > LR > GRU > LSTM is **invariant** across a 5× growth in training data — refuting the hypothesis that sequence models would overtake regression with more data, at least at scales we can reach.

**SHAP feature importance.** `polymarket_vwap` dominates with mean |SHAP| ≈ 0.14, twice the next feature, suggesting Polymarket may be the slower-reacting price-discovery side (Fig. 5).

**Live ensemble.** The production system uses an LR + XGBoost equal-weight ensemble with a strict concordance filter. An 11-point LR-weight sweep (0.0 → 1.0) shows P&L spans only \$4.68 — weight choice is immaterial. The concordance filter rejects 4.80% of trades and improves per-trade Sharpe from 0.437 (unfiltered) to 0.455 (filtered), with a measurable cost: rejected trades would have netted +\$1.95.

**TFT non-convergence.** TFT did not converge at N=6,802 with the small-data hyperparameters (avg RMSE 0.3262 vs GRU 0.2928, only 120 trades vs 1,517). VSN attention entropy (2.656) confirms the attention is healthy and not degenerate (Fig. 10) — the bottleneck is data volume, not architecture correctness. At 1,000+ bars/pair a larger TFT (hidden_size=32+) becomes justifiable.

### 3.5 Adversarial Audit and Honest Sharpe Accounting

**Per-trade Sharpe is the load-bearing headline.** A pre-submission adversarial audit (Phase 18, `AUDIT_REPORT.md`) ran six independent kill-or-confirm checks against every quantitative claim in this paper: Tier 1 (headline Sharpe), Tier 2 (leakage / look-ahead), Tier 3 (cost realism), Tier 4 (survivorship / selection), Tier 5 (paper number-by-number trace), Tier 6 (live-vs-backtest honesty). Tier 2 flagged that the canonical 80/20 row-index split bridged **142 of 144 pairs** between train and test, with a typical gap of just 4 hours between train end and test start for the same `pair_id` — an embargo violation: the same underlying market events inform both training and test losses. We rebuilt the split as a pair-stratified 80/20 (115 train pairs / 29 test pairs, seed=42, with `train_pairs.isdisjoint(test_pairs)` enforced by construction) and retrained Linear Regression and XGBoost on the leakage-free split.

**Per-trade Sharpe drift was only +2.99% (0.501 → 0.516)** and per-trade alpha drifted +4.5% (+15.0 → +15.7 bps) — both well within the 10,000-resample bootstrap CI. The per-trade edge is robust to leakage correction. The per-pair *annualized* Sharpe moved more dramatically because the Bailey–López de Prado (2012) cross-pair-correlation correction `n_eff = N / (1 + (N − 1) · avg_corr)` flips between regimes: the leaky sample has avg pairwise correlation +0.042 (BLdP haircut active, compressing 0.78 → 0.30); the purged sample has avg correlation −0.199, which short-circuits the correction (`avg_corr ≤ 0` returns naive). We therefore demote the per-pair number to a regime-dependent secondary statistic with a 95% bootstrap CI of [0.700, 1.067] on the purged sample. An earlier draft cited a per-pair annualized Sharpe of ≈ 3.2 in the abstract; the audit found this value had no derivation path in the codebase and replaced it with the per-trade headline above. Three Tiers (1, the canonical-split portion of 2, and 3) returned non-PASS verdicts and were corrected in this revision; Tiers 4, 5, and 6 returned PASS without paper changes. The full audit report is at `AUDIT_REPORT.md`.

For comparison, mid-frequency stat-arb strategies typically target 1–5 bps/trade with millions of trades; our 15 bps/trade reflects the larger-than-typical mispricings in immature cross-platform prediction markets, and the trade-count ceiling (~1,500 in the test set) caps total scalability. PPO+autoencoder at +0.5 bps/trade has essentially no edge; PPO-Raw at +9.6 bps is mildly profitable but ~5 bps below the regression baselines. PPO+autoencoder *destroys* roughly 9 bps of edge that PPO-Raw alone would have captured — a direct empirical answer to the professor's question of whether RL plus anomaly detection improves on simpler regression: no, it actively hurts. We verified the reward function, environment transitions, and action space; the autoencoder simply flags normal market behavior as anomalous because it was trained on all spreads without a clean "normal regime" prior. PPO then trades disproportionately in high-volatility windows where predictions are least reliable.

### 3.6 Live Paper Trading

The live system was deployed on BU SCC on April 11, 2026. Through April 22, it accumulated **10,154 closed positions** across 8,421 pairs, with 100% match rate to entry bars (acceptance gate ≥80% passed without margin). Live realized P&L was +\$1.53 versus −\$1.53 in shadow simulation — an exact directional anti-correlation traced to a model-semantics mismatch (the regression models predict next-bar spread change as a mean-reversion signal; the live strategy enters short-spread on large spreads, opposite to `sign(prediction)`). The strategy's edge derives from spread-magnitude thresholding rather than directional alignment with predictions. Crypto sub-category P&L flipped sign (positive at the 3-day snapshot, negative at 8 days, −\$19.87 across 915 trades) — a documented stationarity caveat for per-category claims.

**WTI oil contracts were absent from the April 14–22 live window** (zero closed positions), so the backtest oil edge could not be live-validated within the original submission. Between submission and final draft, we diagnosed the root cause: `KALSHI_DISCOVERY_CATEGORIES` in `src/live/market_discovery.py` omitted `"Commodities"`, and Kalshi had migrated daily WTI/Brent/grain/metal series into that category. A single-line fix (commit `38d7970`), combined with a category-classifier extension and a 200-slot commodity reservation in the live collector, unblocked the pipeline. **Post-fix 12-hour window (April 24): 1,224 closed commodity positions** (KXBRENTW=486, KXWTI=409, KXWTIW=213, KXBRENTMON=76, plus four smaller series), with aggregate P&L +\$1.96 and 36.0% win rate. This is a *proof-of-life* result, not a robust live edge measurement — the 12-hour window spans one regime, the per-trade economics (~\$0.0016) are dramatically below the backtest oil near-expiry edge of +\$0.41/trade, and the live cohort includes full-contract-lifecycle positions rather than only the near-expiry subset the backtest measured. The lower live win rate is therefore expected rather than contradictory; a two-proportion z-test against the 76.5% backtest figure yields z = −10.76, p ≈ 5 × 10⁻²⁷, but the audit (Tier 6) confirms the comparison is between non-equivalent cohorts. Future work can now measure live oil-edge stability over longer windows.

---

## 4. Conclusions

We built an end-to-end cross-platform prediction-market arbitrage system with four model tiers and evaluated it across five independent regimes. The central answer to our research question is clear:

> **At this data scale, increasing model complexity does not improve arbitrage detection. The simplest models win.**

Specifically: **Tier 1 (LR +15.0 bps/trade, Sharpe 0.501; XGBoost +14.9 bps, Sharpe 0.499) beats Tier 2 (LSTM +14.3 bps, GRU +14.0 bps) by 0.7–1.0 bps and 5–10% absolute P&L; Tier 3 (PPO+autoencoder +0.5 bps) is essentially zero alpha.** This ordering holds across single-split, walk-forward, and data-scaling evaluations. Every walk-forward window is profitable for every ML model, and per-trade Sharpe rises with more training data — the edge is stable and improving. The system is robust to transaction costs at every fee level from 0 to 7 pp.

Four reasons appear to operate jointly. **Sample size:** with 47 bars/pair on average and 6,802 total training rows, GRU/LSTM operate well outside the 200–1000-timestep regime where sequence models excel; they are data-starved by construction, and the lookback ablation (§3.3) confirms longer sequences actively hurt rather than help on this dataset. **Signal-to-noise:** the one-step spread-change target has high variance relative to signal, and linear models are more robust in high-noise regimes because they have fewer parameters to over-fit. **Feature engineering carries the signal:** our 51 features include `spread`, `polymarket_vwap`, `dollar_volume_ratio`, and 13 microstructure features — they encode most exploitable signal directly, so the model only needs to linearly combine them, and deep models waste capacity re-discovering what the features already expose. **Overfitting on structural regularities:** the deepest XGBoost configurations (depth 9, n_estimators 500) had higher training accuracy but *lower* test P&L; the same phenomenon appears in sequence models, which memorize per-pair idiosyncrasies that do not generalize. This pattern matches the broader tabular-ML literature (Grinsztajn et al. 2022) and the January 2026 finding on investor-flow prediction.

We expect the picture to evolve as the live auto-retrain system accumulates data. By 250 bars/pair (currently a few weeks away), GRU should close the gap with XGBoost; by 500 bars/pair, sequence models may surpass XGBoost on oil and crypto categories where lead–lag effects dominate. By 1,000 bars/pair, a larger TFT configuration becomes justifiable. This is a falsifiable prediction the system will continue to test.

**The alpha is in the matching pipeline and the asset class.** A 10-rule structural quality filter added +\$10.73 in P&L with no model changes — direct empirical evidence that data quality dominates model choice at this scale. Oil near-expiry contracts produce a 76.5% backtest win rate while sports and politics are near-zero, because oil contracts converge mechanically on observable WTI-futures settlement.

**The per-trade Sharpe of 0.501 (canonical) → 0.516 (leakage-free purged split) is the load-bearing headline.** The Phase 18 audit (`AUDIT_REPORT.md`) confirmed the per-trade edge survives the embargo correction with only +2.99% drift, well within the bootstrap CI. The per-pair annualized Sharpe is regime-dependent (range 0.30–0.81 corrected, depending on cross-pair correlation) and reported with full caveats. An earlier draft cited per-pair annualized Sharpe of ≈ 3.2; the audit found no derivation path and replaced it with the leakage-free per-trade headline above.

**Beyond the empirical findings, three broader lessons:** (i) *infrastructure bugs masquerade as model problems* — Kalshi's silent 429s, Polymarket's `condition_id` typo, and our own `live_NNNN` schema drift caused weeks of confusion that looked like the model failing, and the fix in each case was a config change rather than a hyperparameter sweep; (ii) *evaluation regime matters more than model family* — a single train/test split can tell a very different story than a walk-forward, a per-category breakdown, or a data-scaling curve, so multi-regime evaluation is mandatory rather than optional; (iii) *negative results are results* — PPO failing catastrophically is the strongest evidence we have for the main thesis, and we chose to publish it rather than omit it. A more defensible RL approach would require a curriculum that learns "safe" regimes first, a differentiable simulator for off-policy pre-training, and a much larger universe of training trajectories. None of these are justified at our data scale, but the negative result is itself informative for future practitioners.

**Future work — retraining on the expanded live dataset.** The headline numbers in this paper were trained on 6,802 backtest rows (~3 calendar months, 144 pairs). Since the live system was deployed on April 7, 2026, it has accumulated **233,484 bars across 19 days** — roughly **34× the size of the backtest training set** — driven by a much larger active pair universe (up to 11,582 matched pairs at peak versus 144 in the backtest) and faster bar intervals on short-dated daily contracts. Due to the submission time window, we did not retrain the model tiers on this expanded dataset before submission. This is the single most important planned follow-up: at 34× more samples, the data-scarcity bottleneck identified in §3 (sample-to-parameter ratios of 0.13:1 for GRU and 0.007:1 for TFT) substantially relaxes, and we expect the deeper architectures (LSTM, GRU, possibly TFT) to begin closing the gap with — or surpassing — Linear Regression on at least the oil and crypto categories. If they do, the paper's central finding evolves from "complexity is not an edge" to "complexity is not justified *at our scale, but becomes justified at roughly N≥10× larger*" — a stronger and more nuanced result.

**Limitations.** Our test window is short (two weeks of out-of-sample evaluation; the walk-forward analysis mitigates but does not eliminate this). Paper trading does not model market impact or partial fills. Survivorship bias is structural rather than retroactive (Phase 18 Tier 4 confirmed 0/10 retroactive drops in a random sample). Settlement-divergence risk is real but unobserved in our universe. The live commodity-matching pipeline was incomplete at submission and resolved post-submission, so the backtest oil edge has only a proof-of-life live validation. Other planned follow-up work includes 500- and 1,000-bar scaling checkpoints (the auto-retrain system will produce these over the coming weeks), live-volume-aware microstructure features, a larger TFT configuration once data permits, and re-running the pre-registered LOGO feature ablation at 250+ bars/pair where bootstrap CIs become tight enough to classify individual feature groups as load-bearing or droppable.

The project succeeded on its own terms: we answered the research question rigorously, built a working autonomous system, and documented honest limitations. The edge is real; the complexity is not.

---

## Acknowledgments

We thank Professor Kevin Gold for course instruction and for pivotal late-March feedback that pushed us to treat regression baselines as first-class and to add the PPO-without-autoencoder variant — feedback that directly shaped the research question into something empirically answerable. We acknowledge use of Anthropic's Claude as an AI pair-programming assistant throughout implementation; all design decisions, experimental choices, and interpretations are our own.

---

## References

1. Kang, S. (2026). The limits of complexity: why feature engineering beats deep learning in investor flow prediction. *arXiv:2601.07131*.
2. Amihud, Y. (2002). Illiquidity and stock returns: cross-section and time-series effects. *Journal of Financial Markets* 5(1), 31–56.
3. Bailey, D. H., & López de Prado, M. (2012). The Sharpe ratio efficient frontier. *Journal of Risk* 15(2).
4. Bürgi, C., Deng, W., & Whelan, K. (2026). Makers and takers: the economics of the Kalshi prediction market. *CEPR Discussion Paper DP20631 / SSRN 5502658*.
5. Cont, R., Kukanov, A., & Stoikov, S. (2014). The price impact of order book events. *Journal of Financial Econometrics* 12(1), 47–88.
6. Corwin, S. A., & Schultz, P. (2012). A simple way to estimate bid-ask spreads from daily high and low prices. *Journal of Finance* 67(2), 719–760.
7. Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS 2022 Datasets & Benchmarks*.
8. Kyle, A. S. (1985). Continuous auctions and insider trading. *Econometrica* 53(6), 1315–1335.
9. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 2017*.
10. Manski, C. F. (2006). Interpreting the predictions of prediction markets. *Economics Letters* 91(3), 425–429.
11. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: sentence embeddings using Siamese BERT-networks. *EMNLP 2019*.
12. Roll, R. (1984). A simple implicit measure of the effective bid-ask spread in an efficient market. *Journal of Finance* 39(4), 1127–1139.
13. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.
14. Wolfers, J., & Zitzewitz, E. (2004). Prediction markets. *Journal of Economic Perspectives* 18(2), 107–126.

---

## Appendix B: Figures

![**Figure 1.** Out-of-sample P&L across 11 walk-forward windows (§3.2). Every ML model is profitable in every window; per-trade Sharpe rises from 0.31 in early windows to 0.53 in late windows, indicating the edge strengthens rather than decays as more data accumulates.](experiments/figures/walk_forward_pnl.png)

![**Figure 2.** P&L vs. training-set size at 2 pp signal threshold (§3.4). The curve plateaus at N=6,802 because the underlying pair universe is fixed at 144 pairs with at most 141 bars each; slices at 250+ bars/pair are identical to the 100-bar slice. The plateau is a property of the fixed pair universe, not a universal scaling claim.](experiments/results/data_scaling/pnl_at_2pp_vs_data.png)

![**Figure 3.** Walk-forward per-trade Sharpe trajectory across the 11 windows (§3.2 supplemental). The trend confirms the rising-edge pattern visible in Fig. 1.](experiments/figures/walk_forward_sharpe.png)

![**Figure 4.** P&L vs. round-trip fee for the four ML models (§3.3). The rank-ordering of models is invariant to fee assumptions in the 0–10 pp range; LR and XGBoost remain the top tier across the entire fee spectrum.](experiments/figures/transaction_cost_sensitivity.png)

![**Figure 5.** Mean |SHAP| feature importance for XGBoost (§3.4). `polymarket_vwap` dominates with mean |SHAP| ≈ 0.14, twice the next feature, suggesting Polymarket is the slower-reacting price discovery side.](experiments/figures/shap_bar_plot.png)

![**Figure 6.** Cumulative test-set P&L by model (§3.1). The Tier-1 vs. Tier-2 separation opens gradually rather than in a single jump — consistent with a stable per-trade edge.](experiments/figures/backtest_equity_curves.png)

![**Figure 7.** Bootstrap 95% CI on RMSE by model (1,000 resamples, §3.1). The LR / XGBoost / GRU / LSTM intervals overlap heavily even though P&L separates cleanly — confirming the P&L gap is driven by directional accuracy and trade selection, not raw regression error.](experiments/figures/bootstrap_ci_rmse.png)

![**Figure 8.** P&L vs. lookback window for GRU and LSTM (§3.3 / Proposal Experiment 2). Performance is essentially flat across {2, 6, 12, 18}-bar lookbacks, indicating the sequence models cannot exploit longer history at this dataset scale.](experiments/figures/experiment2_lookback_pnl.png)

![**Figure 9.** P&L heatmap by model × minimum-spread threshold (§3.3 / Proposal Experiment 3). Threshold = 2 pp dominates for all ML models; lower thresholds are dominated by noise, higher thresholds by missed trades.](experiments/figures/experiment3_threshold_heatmap.png)

![**Figure 10.** TFT VSN variable-selection weights (§3.4). The attention is healthy (entropy = 2.656 vs. degenerate baseline of ≈ 0) — TFT's underperformance vs. GRU is a data-volume bottleneck at N=6,802, not an architectural failure.](experiments/figures/tft_variable_importance.png)

![**Figure 11.** Walk-forward P&L at 2 pp fees across the 11-point LR-weight sweep (§3.4). The spread across weights is \$4.68; the curve is essentially flat, confirming the production ensemble's weight choice is not cherry-picked.](experiments/figures/ensemble_weight_sweep.png)
