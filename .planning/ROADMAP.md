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
- [ ] **Phase 3: Feature Engineering** - Compute derived features from aligned 4-hour microstructure data, temporal split, PyTorch Forecasting format
- [x] **Phase 4: Regression Baselines and Evaluation Framework** - Train Tier 1 models, build evaluation/simulation infrastructure, deliver TA check-in
- [x] **Phase 5: Time Series Models** - Train GRU, LSTM, and TFT on spread prediction with hourly sequences
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
**Plans:** 2/2 plans complete

Plans:
- [x] 02.1-01-PLAN.md -- Trade fetcher (Kalshi + Polymarket pagination) and candle reconstructor (4-hour OHLCV+VWAP+microstructure)
- [ ] 02.1-02-PLAN.md -- Cross-platform aligner (forward-fill, staleness decay, quality filters) and rebuild_data.py CLI pipeline

### Phase 3: Feature Engineering
**Goal**: A processed, model-ready dataset with derived microstructure features computed from 4-hour aligned data, temporally split into train/test sets with PyTorch Forecasting compatibility
**Depends on**: Phase 2.1
**Requirements**: FEAT-01, FEAT-02, FEAT-03, FEAT-04, FEAT-05
**Success Criteria** (what must be TRUE):
  1. `data/processed/` contains train.parquet and test.parquet with 39 columns (31 aligned + 6 derived + time_idx + group_id) for each matched pair
  2. Low-liquidity markets (<20 trades) are filtered upstream by aligner and documented
  3. Temporal train/test split is enforced per pair with an assertion that no training timestamp exceeds the earliest test timestamp
  4. The dataset includes `time_idx` and `group_id` columns compatible with PyTorch Forecasting TimeSeriesDataSet format
  5. A `build_timeseries_dataset()` function creates a TimeSeriesDataSet from the feature matrix
**Plans**: 1 plan

Plans:
- [ ] 03-01-PLAN.md -- Derived features (velocity, volume ratio, spread momentum/volatility, order flow imbalance), temporal split, PyTorch Forecasting format, CLI pipeline

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
**Plans**: 5 plans

Plans:
- [x] 05-01-PLAN.md -- Sequence utilities (windowing, early stopping, seed, device, scaler helpers) with TDD tests
- [x] 05-02-PLAN.md -- GRUPredictor (hidden=64) with BasePredictor contract, warm-up stitching, TDD tests
- [x] 05-03-PLAN.md -- LSTMPredictor (hidden=32) with BasePredictor contract, warm-up stitching, TDD tests
- [x] 05-04-PLAN.md -- Tier 2 experiment harness (--tier flag, 3 seeds, tier2 JSONs, combined comparison table)
- [x] 05-05-PLAN.md -- TFT (MOD-07) deferral documentation + phase-level SUMMARY.md

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
**Plans**: 5 plans

Plans:
- [ ] 06-01-PLAN.md -- SpreadTradingEnv (Gymnasium Env): state (198,), Discrete(3) actions, dense reward with tx cost, TDD
- [ ] 06-02-PLAN.md -- AnomalyDetectorAutoencoder: 31->16->8->4->8->16->31 with 95th percentile threshold, TDD
- [ ] 06-03-PLAN.md -- PPORawPredictor (BasePredictor): SB3 PPO on raw features, action-to-prediction mapping, TDD
- [ ] 06-04-PLAN.md -- PPOFilteredPredictor (BasePredictor): PPO with autoencoder reward masking, TDD
- [ ] 06-05-PLAN.md -- Tier 3 harness (--tier 3, --tier all), __init__.py exports, train + produce tier3 JSONs

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
**Plans**: 5 plans

Plans:
- [ ] 07-01-PLAN.md -- Experiment 1: cross-tier comparison (formalize 8-model results into summary JSON, RMSE bar chart, P&L equity curves, LaTeX table)
- [ ] 07-02-PLAN.md -- Experiment 2: lookback window ablation (GRU+LSTM at lookback={2,6,12,18}, RMSE/P&L plots)
- [ ] 07-03-PLAN.md -- Experiment 3: spread threshold ablation (all 8 models at threshold={0.0,0.02,0.05,0.10}, heatmap/bar chart)
- [ ] 07-04-PLAN.md -- SHAP analysis (TreeSHAP on XGBoost) + transaction cost sensitivity (0-10pp, break-even per model)
- [ ] 07-05-PLAN.md -- Bootstrap confidence intervals (1000 resamples, 95% CI on RMSE/MAE/P&L/Sharpe, forest plot)

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
| 2.1. Trade-Based Data Reconstruction | 1/2 | Executing | - |
| 3. Feature Engineering | 0/1 | Planning complete | - |
| 4. Regression Baselines and Evaluation Framework | 1/2 | In Progress | - |
| 5. Time Series Models | 5/5 | Complete | 2026-04-06 |
| 6. RL and Autoencoder | 3/5 | In Progress|  |
| 7. Experiments and Interpretability | 0/5 | Planning complete | - |
| 8. Paper and Presentation | 0/0 | Not started | - |

---
*Roadmap created: 2026-04-01*
*Last updated: 2026-04-06*
