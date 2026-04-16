# Project Roadmap & Timeline

## DS340 Final Project: Kalshi vs. Polymarket Price Discrepancies
**Team:** Ian Sabia (U33871576), Alvin Jang (U64760665)

---

## Phase 1: Foundation (March 2026)

### Data Pipeline
- Built Kalshi API client (`src/data/kalshi.py`) — hourly candlesticks, handles live/historical split
- Built Polymarket API client (`src/data/polymarket.py`) — Gamma + CLOB + Data API integration
- **Key challenge:** Polymarket uses numeric token IDs, not slugs. Required Gamma API → clobTokenIds lookup chain.
- **Key challenge:** Polymarket `/prices-history` returns empty for resolved markets. Had to reconstruct from trade records via Data API.

### Market Matching Pipeline
- Semantic matcher using sentence-transformers (`all-MiniLM-L6-v2`) for cross-platform contract matching
- Matrix cosine similarity via normalized dot product — replaced O(N*M) keyword filter that would have taken 6.6 hours with a 300x faster approach (~80 seconds)
- Quality filter (`src/matching/quality_filter.py`) — 10 structural rules (see below)

### Feature Engineering
- 31 aligned columns (OHLCV + microstructure per platform + spread)
- 6 initial derived features: price_velocity, volume_ratio, spread_momentum, spread_volatility, order_flow_imbalance (per platform)

---

## Phase 2: Model Development (Late March - Early April 2026)

### Tier 1 — Regression Baselines
- **Linear Regression:** RMSE=0.308, DA=65.9%, P&L@2pp=+$230, per-trade Sharpe=0.558
- **XGBoost:** RMSE=0.286, DA=67.8%, P&L@2pp=+$238, per-trade Sharpe=0.588
- **Naive (spread closes):** P&L@2pp=+$58 (lower bound)
- **Volume (higher volume correct):** P&L@2pp=+$60 (lower bound)
- **Finding:** Both regression baselines significantly beat naive baselines. XGBoost slightly edges LR.

### Tier 2 — Time Series Models
- **GRU:** RMSE=0.293, DA=64.3%, P&L@2pp=+$212, per-trade Sharpe=0.515
- **LSTM:** RMSE=0.292, DA=65.5%, P&L@2pp=+$222, per-trade Sharpe=0.532
- **TFT:** Not trained yet (needs more data)
- **Finding:** Sequence models have slightly better Sharpe (less volatile) but LOWER total P&L than regression. Complexity not justified at 47 bars/pair.

### Tier 3 — RL Models
- **PPO-Raw:** Trained, different eval framework
- **PPO-Filtered (+ autoencoder):** Trained. **Worst performer** — -$7,724 in backtest. The autoencoder anomaly filter actually hurts.
- **Finding:** RL is the most complex approach and performs worst. Strong negative result answering the research question.

### Experiments Completed
1. **Complexity vs Performance (Experiment 1):** All tiers on same data. XGBoost > LR > LSTM > GRU >> PPO. Simpler models win.
2. **Lookback Window (Experiment 2):** 8-24h windows beat 72h+. GRU degrades sharply at longer windows.
3. **Spread Threshold (Experiment 3):** Sweep across min-spread thresholds with heatmap visualization.
4. **Bootstrap CI:** Confidence intervals for all model metrics.
5. **SHAP Analysis:** polymarket_vwap dominates feature importance (~0.14 mean |SHAP|).
6. **Transaction Cost Sensitivity:** All ML models remain profitable up to Kalshi's 5pp taker fee.

---

## Phase 3: Live Trading System (April 9-11, 2026)

### Infrastructure
- Deployed on BU SCC (scc1.bu.edu) with 3 cron jobs:
  - Trading cycle every 15 min (login node, ~3 min CPU)
  - Market discovery every 3h (batch job via qsub, ~10 min)
  - Model retraining every 6h (batch job via qsub, ~30 min)
- GitHub Actions as redundant fallback for both discovery and trading
- SQLite-backed position manager with exit rules (take_profit, stop_loss, momentum, time_stop, resolution_proximity)

### Live Pair Universe Growth
| Date | Pairs | What changed |
|------|-------|-------------|
| Apr 9 | 615 | Initial semantic matching run |
| Apr 11 AM | 1,804 | First autonomous discovery cycle on SCC |
| Apr 11 PM | 4,753 | Fixed Kalshi 429 rate limiting + Polymarket pagination |
| Apr 12 | 11,582 | Steady-state discovery with all fixes |

### Per-Category P&L Analysis (Historical, Apr 11)
| Category | Trades | Win% | $/trade | Edge vs pooled |
|----------|--------|------|---------|----------------|
| **Oil** | 765 | **76.5%** | **+$0.41** | **+142.7%** |
| Fed rates | 431 | 34.6% | +$0.01 | -92.4% |
| Sports | 618 | 37.4% | -$0.00 | -100.1% |
| Politics | 67 | 29.9% | -$0.02 | -111.8% |

**Key finding:** Oil near-expiry contracts with deterministic convergence are the entire edge. Sports/politics are noise.

---

## Phase 4: Bug Fixes & Data Quality (April 11-12, 2026)

### Discovery Pipeline Fixes
1. **Kalshi /events HTTP 429 silently dropped** — Added exponential backoff retry (1s/2s/4s/8s) + per-series 250ms pacing. Impact: Kalshi markets fetched went from ~2000 to 7,200+.
2. **Polymarket pagination too shallow** — Bumped from 5,000 to 30,000 markets. WTI markets were at offset 15,305+ and completely invisible. Impact: Poly markets fetched went from 5,000 to 30,000.
3. **GHA push rejection race** — Added rebase-retry to both discover-markets and collect-and-trade workflows.

### Quality Filter Rules Added
- **Rule 3d (KXAAAGAS):** AAA retail gasoline state-specific (CA/FL/NY/TX) vs national Polymarket + month mismatch. Caught 135 bad pairs.

### Stale Pair Eviction
- Tombstone-based eviction with 7-day TTL. 590 ancient pairs evicted, 25 protected (open positions). Tombstones preserve array indices.

### Critical Bug: pair_id Schema Disagreement (FIXED)
- **Problem:** collector.py, strategy.py, and pair_mapping.json all disagreed on what `live_NNNN` meant. 25 open positions were referencing wrong pairs, getting wrong prices.
- **Fix:** Content-addressed pair_ids via `src/live/pair_ids.py::make_pair_id`. Format: `kxwti26apr08t10799-0x43d5953d` (matches train.parquet). All three code paths now use the same function.
- **Impact:** 25 zombie positions force-closed. Clean slate for new trading.

### Polymarket Pricing Fix
- **Problem:** Gamma API `condition_id` param returns random markets (!). `condition_ids` (plural) returns exact match.
- **Impact:** Polymarket prices went from 0/444 success to correct pricing for all pairs.

### SCC Login Node CPU Fix
- Paper trader's LSTM retraining every 15 min exceeded SCC's 15-min CPU watchdog with 10,750 pairs.
- Removed paper_trader from login-node script. Retraining moved to batch job.

---

## Phase 5: Model Improvements (April 12, 2026)

### New Features (9 added, total 46 columns)
| Feature | What it captures | Impact |
|---------|-----------------|--------|
| spread_momentum_6/12 | Medium/long-term trend | Neutral at 47 bars, expected to help at 100+ |
| spread_volatility_6 | Medium-term vol | Neutral at 47 bars |
| spread_zscore | How anomalous spread is | Mean~0, std~1 (validated) |
| spread_range | Recent trading range | Neutral |
| dollar_volume_ratio | Liquidity asymmetry | Neutral |
| trade_count_ratio | Trade activity asymmetry | Neutral |
| mid_price | Platform consensus price | Enables price_divergence_pct |
| price_divergence_pct | Relative spread size | Normalizes spread by price level |

**Finding:** New features are neutral-to-slightly-positive on 47 bars/pair historical data. Rolling windows need more data to be informative. Will show value as live bars accumulate to 100+/pair.

### XGBoost Hyperparameter Sweep (48 configs)
- **Best:** depth=3, lr=0.01, n=100 → P&L +$209.70
- **Finding:** Shallow trees (depth 3) with low learning rate consistently beat deeper models. Overfitting is the enemy at this data scale.
- **Paper finding:** More features + deep trees HURT performance. P&L dropped from $238 to $210 with expanded features on same data.

### Category-Aware Entry Filter
- Commodity categories (oil, crypto, inflation) use normal prediction threshold
- Non-commodity categories require 3x prediction confidence to enter
- Focuses capital on the categories that historically make money

### Ensemble Concordance Check
- Only enter when LR and XGBoost agree on direction
- Filtered 45 conflicted trades (2.9%), P&L impact: +$0.22 (safety net, not P&L driver)

---

## Key Findings for Presentation

### Central Research Question Answer
> **Does increasing model complexity improve arbitrage detection?**

**No, not at this data scale.** The ranking is:

| Rank | Model | Complexity | Per-trade Sharpe |
|------|-------|-----------|-----------------|
| 1 | XGBoost (depth=3) | Low | 0.588 |
| 2 | Linear Regression | Lowest | 0.558 |
| 3 | LSTM | High | 0.532 |
| 4 | GRU | High | 0.515 |
| 5 | PPO+Autoencoder | Highest | Negative |

### Supporting Evidence
1. Feature engineering beats deep learning (confirmed by Jan 2026 academic paper: arXiv 2601.07131)
2. XGBoost at depth=3 outperforms depth=7+ (overfitting kills with small data)
3. More features hurt when rolling windows lack data (47 bars vs 100+ needed)
4. Oil/commodity contracts with deterministic convergence are the real edge (76.5% WR)
5. The semantic matching + quality filter pipeline is where the actual alpha lives — not the models

### What Would Change at Scale
- More data (100+ bars/pair) would likely boost LSTM/GRU via longer lookback windows
- The new features (spread_zscore, price_divergence_pct) would become informative
- Cross-platform lead-lag signals would add genuine alpha
- But XGBoost would likely remain competitive — this is the norm in tabular ML

---

## Autonomous System Status (as of April 12, 2026)

- **Pair universe:** 11,582 total (10,967 active, 615 tombstoned)
- **Commodity pairs:** 506 (including fresh short-dated WTI)
- **Open positions:** 0 (clean slate after schema fix)
- **Models deployed:** LR + XGBoost (trained on 6,802 rows)
- **Auto-retrain:** Every 6h via SCC batch job
- **Auto-discover:** Every 3h via SCC batch job + GHA fallback
- **Trading cycle:** Every 15 min on SCC login node
- **Next milestone:** Tier 2 retrain when bars/pair hits 100 (~22h from now)

---

## Phase 6: Multi-Scale Validation (April 16, 2026)

### Walk-Forward Backtest
- Built `experiments/run_walk_forward.py` — true expanding-window walk-forward
- 5 out-of-sample windows spanning 11 weeks (Jan 12 - Apr 1, 2026)
- **Every window profitable** for both LR and XGBoost
- Per-trade Sharpe trending UP: 0.37 → 0.51 (37% improvement across windows)
- Confirms edge is not a lucky split, improves with more training data
- Output: `experiments/figures/walk_forward_*.png`, `experiments/results/walk_forward/log.jsonl`

### Per-Category Model Performance
- Built `experiments/run_category_breakdown.py` — stratifies test set by category
- **Inflation dominates** the historical edge (+$89 on 616 trades, 63% WR)
- **LR wins 5/7 categories** (employment, fed_rates, gdp, inflation, politics_election)
- **XGBoost wins 2/7 categories but by larger margins** (crypto $+48, politics_policy $+31)
- The "XGBoost best overall" result is driven entirely by crypto outperformance
- Nuanced paper claim: "XGBoost wins specific regimes, LR wins the rest"
- Output: `experiments/results/category_breakdown.json`, `category_breakdown_table.txt`

### 250-Bar Checkpoint (pending)
- Auto-retrain batch job fires every 6h on SCC
- 50-bar and 100-bar checkpoints fired on April 16
- 250-bar checkpoint pending (need 20 pairs at 250+ bars, currently max=148)
- When it fires, we'll have 3 data points (50 / 100 / 250) to confirm the complexity penalty across scales

### Current System Status (April 16, 2026)
- SCC: 48 pushes in last 12 hours (perfect uptime)
- Live pair universe: 6,397 pairs (204 commodity), 47,624 bars
- Combined P&L: +$2.69 across 2,280 closed trades overnight
- Deployed models: LR + XGBoost retrained April 16 08:39 (37k rows, 54 features)
- GRU/LSTM trained at 100-bar checkpoint, underperformed regression baselines

---

## Tools & Infrastructure
- Python 3.12, PyTorch, XGBoost, scikit-learn, sentence-transformers
- BU SCC (compute cluster) + GitHub Actions (CI/CD + fallback)
- SQLite (position management), Parquet (bar storage), JSONL (trade logs)
- Graphify knowledge graph (2,394 nodes, 3,928 edges, 128 communities)
- Claude Code for AI-assisted development
