# Roadmap: Kalshi vs. Polymarket Price Discrepancies

## Overview

This project builds an end-to-end cross-platform prediction market arbitrage pipeline, then systematically tests whether increasing model complexity (regression to recurrent networks to RL with anomaly detection) improves spread convergence prediction. The roadmap follows strict data-flow dependencies: raw ingestion feeds matching, matching feeds feature engineering, features feed all models. Phases 1-4 must complete by the April 4 TA check-in (working pipeline + baseline results). Phases 5-6 can be parallelized between team members. Phases 7-8 synthesize results into experiments and the final paper.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Data Ingestion** - Build platform API adapters and ingest historical market data from Kalshi and Polymarket
- [ ] **Phase 2: Market Matching** - Match equivalent contracts across platforms using NLP and manual curation
- [ ] **Phase 3: Feature Engineering** - Construct time-aligned hourly feature vectors and the matched-pairs dataset
- [ ] **Phase 4: Regression Baselines and Evaluation Framework** - Train Tier 1 models, build evaluation/simulation infrastructure, deliver TA check-in
- [ ] **Phase 5: Time Series Models** - Train GRU, LSTM, and TFT on spread prediction with hourly sequences
- [ ] **Phase 6: RL and Autoencoder** - Build trading environment, train autoencoder anomaly detector, train PPO variants
- [ ] **Phase 7: Experiments and Interpretability** - Run cross-tier comparison, ablation experiments, SHAP analysis, bootstrap CIs
- [ ] **Phase 8: Paper and Presentation** - Write final paper and lightning talk slides

## Phase Details

### Phase 1: Data Ingestion
**Goal**: Raw historical market data from both platforms is reliably available on disk for downstream processing
**Depends on**: Nothing (first phase)
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06
**Success Criteria** (what must be TRUE):
  1. Running the Kalshi ingestion script produces parquet files in `data/raw/` containing OHLCV candlestick data, correctly handling the live/historical endpoint split
  2. Running the Polymarket ingestion script produces parquet files in `data/raw/` containing reconstructed price histories from trade records, with metadata from Gamma API
  3. Re-running either script skips already-cached data and respects rate limits without manual intervention
  4. Both adapters recover from transient API failures via automated retry with backoff
**Plans**: 3 plans

Plans:
- [ ] 01-01-PLAN.md -- Project setup, shared infrastructure (ResilientClient, MarketDataAdapter ABC, schemas, test fixtures)
- [ ] 01-02-PLAN.md -- Kalshi adapter (market discovery, candlestick fetching with chunking, null OHLC handling)
- [ ] 01-03-PLAN.md -- Polymarket adapter (Gamma discovery, CLOB price history, Data API trade fallback)

### Phase 2: Market Matching
**Goal**: A verified registry of matched contract pairs across Kalshi and Polymarket with confidence scores and settlement documentation
**Depends on**: Phase 1
**Requirements**: MATCH-01, MATCH-02, MATCH-03, MATCH-04, MATCH-05
**Success Criteria** (what must be TRUE):
  1. The matching pipeline produces candidate pairs ranked by combined keyword and semantic similarity scores
  2. A manual curation interface allows reviewing, accepting, or rejecting each candidate pair
  3. Every accepted pair in `matched_pairs.json` has a confidence score and documented settlement criteria from both platforms
  4. The total number of matched pairs is known, enabling a go/no-go decision on project scope (if <30 pairs, TFT is dropped)
**Plans**: TBD

Plans:
- [ ] 02-01: TBD
- [ ] 02-02: TBD

### Phase 02.1: Trade-Based Data Reconstruction (INSERTED)

**Goal:** Raw trades pulled from both platforms, reconstructed into 4-hour OHLCV+VWAP candles with microstructure features, and aligned cross-platform with forward-fill and staleness decay. Output: data/processed/aligned_pairs.parquet with 50-70 usable pairs.
**Requirements**: TRADE-FETCH, TRADE-RECONSTRUCT, TRADE-ALIGN, TRADE-VALIDATE, TRADE-PIPELINE (defined by design spec)
**Depends on:** Phase 2
**Plans:** 2 plans

Plans:
- [ ] 02.1-01-PLAN.md -- Trade fetcher (Kalshi + Polymarket pagination) and candle reconstructor (4-hour OHLCV+VWAP+microstructure)
- [ ] 02.1-02-PLAN.md -- Cross-platform aligner (forward-fill, staleness decay, quality filters) and rebuild_data.py CLI pipeline

### Phase 3: Feature Engineering
**Goal**: A processed, time-aligned dataset of hourly microstructure features ready for consumption by all model tiers
**Depends on**: Phase 2
**Requirements**: FEAT-01, FEAT-02, FEAT-03, FEAT-04, FEAT-05
**Success Criteria** (what must be TRUE):
  1. `data/processed/` contains feature matrices with per-hour vectors (spread, volume, bid-ask spread, price velocity) aligned across both platforms for each matched pair
  2. Low-liquidity markets (< 10 trades) are filtered out and documented
  3. Temporal train/test split is enforced with an assertion that no training timestamp exceeds the earliest test timestamp
  4. The dataset includes `time_idx` and `group_ids` columns compatible with PyTorch Forecasting TimeSeriesDataSet format
  5. Both flat (for regression) and windowed (for time series) dataset views are available from the same underlying data
**Plans**: 2 plans

Plans:
- [ ] 03-01-PLAN.md -- Core feature pipeline (pair loader, hourly alignment, spread/microstructure features, liquidity filter, build CLI)
- [ ] 03-02-PLAN.md -- Temporal train/test split, flat and windowed dataset views, PyTorch Forecasting TimeSeriesDataSet format

### Phase 4: Regression Baselines and Evaluation Framework
**Goal**: Tier 1 models are trained and evaluated, the evaluation framework exists for all future models, and the TA check-in deliverable is ready
**Depends on**: Phase 3
**Requirements**: MOD-01, MOD-02, MOD-03, MOD-04, EVAL-01, EVAL-02
**Success Criteria** (what must be TRUE):
  1. Linear Regression and XGBoost are trained on the processed dataset and produce spread predictions on the test set
  2. Naive baselines (spread-always-closes, higher-volume-platform-correct) produce predictions on the test set
  3. All four models are evaluated with RMSE, MAE, and directional accuracy, with results in a comparison table
  4. Profit simulation runs for all four models, reporting P&L, win rate, and Sharpe ratio
  5. The evaluation framework accepts any model implementing the common prediction interface, ready for Tier 2 and Tier 3 models
**Plans**: 2 plans

Plans:
- [ ] 04-01-PLAN.md -- Prediction interface, evaluation framework (metrics + profit sim), naive and volume baselines
- [ ] 04-02-PLAN.md -- Linear Regression, XGBoost, results storage, baseline comparison experiment script

### Phase 5: Time Series Models
**Goal**: Recurrent and attention-based models are trained and evaluated, testing whether temporal structure improves spread prediction
**Depends on**: Phase 4
**Requirements**: MOD-05, MOD-06, MOD-07
**Success Criteria** (what must be TRUE):
  1. GRU model is trained on windowed hourly sequences and evaluated through the existing evaluation framework
  2. LSTM model is trained on windowed hourly sequences and evaluated through the existing evaluation framework
  3. TFT model is trained via PyTorch Forecasting and evaluated through the existing framework (or explicitly deferred with documented rationale if dataset is too small or timeline is tight)
  4. Results for all Tier 2 models appear in the same comparison table as Tier 1, enabling direct cross-tier comparison
**Plans**: TBD

Plans:
- [ ] 05-01: TBD
- [ ] 05-02: TBD

### Phase 6: RL and Autoencoder
**Goal**: PPO trading agents and the autoencoder anomaly detector are trained, testing whether RL and anomaly detection improve trading performance
**Depends on**: Phase 4
**Requirements**: MOD-08, MOD-09, MOD-10
**Success Criteria** (what must be TRUE):
  1. A custom Gym environment simulates spread trading with appropriate state space, action space, and reward function
  2. The autoencoder is trained on normal spread behavior and flags anomalous spread patterns via reconstruction error threshold
  3. PPO on raw features produces a trading policy (even if it learns "don't trade," which is a valid finding)
  4. PPO with autoencoder signal filter produces a trading policy that only acts on flagged opportunities
  5. All RL models are evaluated through the existing evaluation framework with results in the comparison table
**Plans**: TBD

Plans:
- [ ] 06-01: TBD
- [ ] 06-02: TBD

### Phase 7: Experiments and Interpretability
**Goal**: The three planned experiments are executed, SHAP analysis reveals feature importance, and all results have statistical rigor via confidence intervals
**Depends on**: Phase 5, Phase 6
**Requirements**: EXP-01, EXP-02, EXP-03, EXP-04, EVAL-03, EVAL-04
**Success Criteria** (what must be TRUE):
  1. Experiment 1 (centerpiece) produces a cross-tier complexity-vs-performance comparison with all models on identical test data
  2. Experiment 2 produces results for 6h, 24h, 72h, and 7d lookback windows showing how window length affects prediction quality
  3. Experiment 3 produces results for no minimum, >2pp, >5pp, and >10pp spread thresholds showing how filtering affects trading performance
  4. SHAP feature importance analysis is computed for the best-performing models (TreeSHAP for XGBoost, DeepExplainer or attention for neural models)
  5. Bootstrap confidence intervals are reported on key metrics, and transaction cost sensitivity analysis quantifies break-even fee levels
**Plans**: TBD

Plans:
- [ ] 07-01: TBD
- [ ] 07-02: TBD

### Phase 8: Paper and Presentation
**Goal**: The final paper and lightning talk slides are complete, presenting the complexity-vs-performance findings as an empirical contribution
**Depends on**: Phase 7
**Requirements**: DEL-01, DEL-02
**Success Criteria** (what must be TRUE):
  1. The final paper follows standard academic structure (abstract, intro, related work, methodology, experiments, results, discussion, conclusion) and presents all three experiments with figures and tables
  2. The paper frames PPO underperformance (if it occurs) as a finding about complexity-vs-performance tradeoffs, not a failure
  3. Lightning talk slides (5-10 slides) summarize the research question, methodology, key results, and takeaways
  4. Settlement divergence risk and transaction cost limitations are acknowledged in the paper discussion
**Plans**: TBD

Plans:
- [ ] 08-01: TBD
- [ ] 08-02: TBD

## Progress

**Execution Order:**
Phases 1-4 are strictly sequential (data dependencies). Phases 5 and 6 can be parallelized. Phase 7 depends on both 5 and 6. Phase 8 depends on 7.

```
1 -> 2 -> 2.1 -> 3 -> 4 -> 5 ─┐
                           └─> 6 ─┤-> 7 -> 8
```

**Milestone Deadlines:**
- TA Check-in (April 4): Phases 1-4 complete
- Final Submission (April 27): All phases complete

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Data Ingestion | 0/3 | Planning complete | - |
| 2. Market Matching | 0/0 | Not started | - |
| 2.1. Trade-Based Data Reconstruction | 0/2 | Planning complete | - |
| 3. Feature Engineering | 0/2 | Planning complete | - |
| 4. Regression Baselines and Evaluation Framework | 0/2 | Planning complete | - |
| 5. Time Series Models | 0/0 | Not started | - |
| 6. RL and Autoencoder | 0/0 | Not started | - |
| 7. Experiments and Interpretability | 0/0 | Not started | - |
| 8. Paper and Presentation | 0/0 | Not started | - |

---
*Roadmap created: 2026-04-01*
*Last updated: 2026-04-04*
