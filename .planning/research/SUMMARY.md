# Project Research Summary

**Project:** Kalshi vs. Polymarket Price Discrepancies (DS340 Final)
**Domain:** Cross-platform prediction market arbitrage with ML/RL complexity analysis
**Researched:** 2026-04-01
**Confidence:** HIGH

## Executive Summary

This is a financial ML academic project that builds an end-to-end pipeline for detecting and exploiting price discrepancies between two prediction market platforms (Kalshi and Polymarket). The core research question is whether increasing model complexity (from linear regression through recurrent networks to RL) improves arbitrage detection on cross-platform spread data. Expert practitioners build systems like this as a strict linear pipeline: ingest raw market data, match equivalent contracts across platforms via NLP, engineer microstructure features from synchronized time series, train a tiered set of models, and evaluate all models on identical metrics including simulated trading P&L. The critical insight from research is that the data engineering (ingestion, matching, alignment) is both the hardest and most important work -- every model depends on the quality of the matched-pairs dataset, and no model can compensate for bad matching or misaligned timestamps.

The recommended approach is to build the pipeline strictly in dependency order, front-loading all data work and validating with simple baselines before attempting complex models. The stack is fully verified and installed: PyTorch as the deep learning backbone, Stable Baselines 3 for PPO, PyTorch Forecasting for TFT, sentence-transformers for market matching, XGBoost and scikit-learn for baselines. All libraries are confirmed working in the project's virtual environment including MPS acceleration on Apple Silicon. The architecture follows a platform adapter pattern that isolates API-specific complexity, feeds into a registry-based matching pipeline, produces standardized feature matrices, and branches into three model tiers that all output a common prediction format for fair evaluation.

The top risks are: (1) insufficient matched pairs to achieve statistical significance, which could force scope reduction; (2) cross-platform time alignment errors that create phantom spreads and invalidate all downstream results; (3) market matching false positives that inject noise into training data; and (4) PPO reward shaping that produces degenerate policies. Mitigation requires running the matching pipeline in the first few days to establish dataset size, rigorous timestamp normalization with visual validation, manual review of every matched pair, and explicit reward function design with action distribution monitoring. The April 4 TA check-in (3 days away) is an immovable constraint that demands a working pipeline with at least one baseline result.

## Key Findings

### Recommended Stack

The entire stack has been verified installed and import-tested in `.venv/` on Python 3.12.12 with Apple Silicon MPS acceleration. No installation work is needed except for `pytest` (required for TDD workflow). See `.planning/research/STACK.md` for full version details.

**Core technologies:**
- **PyTorch 2.10.0:** Deep learning backbone for GRU, LSTM, autoencoder. Required by SB3, PyTorch Forecasting, and sentence-transformers. MPS-accelerated for training.
- **Stable Baselines 3 2.7.1:** Battle-tested PPO implementation. Use CPU (not MPS) -- SB3 explicitly warns against GPU for MLP policies.
- **PyTorch Forecasting 1.6.1:** Provides TFT with built-in attention interpretability. Requires data formatted as `TimeSeriesDataSet` with specific column structure (`time_idx`, `group_ids`, `target`).
- **XGBoost 3.2.0:** Primary tree-based baseline. Native SHAP integration via `TreeExplainer`. Preferred over LightGBM for this project.
- **sentence-transformers 5.3.0:** Market matching via `all-MiniLM-L6-v2` embeddings. Fast, sufficient for short contract titles.
- **requests 2.32.5:** Simple REST calls for batch API ingestion. Async is unnecessary.
- **SHAP 0.51.0:** `TreeExplainer` for XGBoost (exact), `DeepExplainer` for neural models (approximate), TFT uses built-in attention instead.

**Critical version note:** `pytest` needs installation (`pip install pytest pytest-cov`). Everything else is ready.

### Expected Features

See `.planning/research/FEATURES.md` for full feature landscape with dependency graph.

**Must have (table stakes):**
- Dual-platform API ingestion with caching and rate limiting
- NLP-based market matching pipeline (keyword + semantic similarity + manual curation)
- Time-aligned matched-pairs dataset with liquidity filtering
- Per-hour microstructure feature vectors (spread, volume, bid-ask, velocity, rolling stats)
- Naive baselines (spread-always-closes, volume-platform-correct)
- Tier 1 baselines: Linear Regression, XGBoost
- Tier 2 time series: GRU (primary recurrent), LSTM
- Tier 3 RL: PPO on raw features, PPO + autoencoder signal filter
- Evaluation framework: RMSE, MAE, directional accuracy, simulated P&L, Sharpe ratio
- Experiment 1: cross-tier architecture comparison (the centerpiece)
- EDA notebook and final paper

**Should have (differentiators):**
- TFT with variable selection and attention interpretability
- PPO-without-autoencoder ablation (isolates RL value, addresses professor's critique)
- Dual evaluation paradigm (regression metrics AND trading simulation)
- Settlement divergence documentation
- SHAP interpretability analysis
- Experiments 2 and 3: window length and spread threshold ablations

**Defer (only if time permits):**
- TFT (implement last -- highest complexity, likely to overfit on small data)
- LightGBM as secondary tree baseline (adds complexity, minimal analytical value)
- Extensive hyperparameter tuning (reasonable defaults are sufficient)

### Architecture Approach

The system is a six-component linear pipeline with a branching model layer. Data flows from platform-specific API adapters through a matching registry, into feature engineering that produces standardized matrices, branching into three model tiers that all output a common `PredictionResult` format for unified evaluation. Key patterns: platform adapter pattern (isolates API mess), registry pattern for matched pairs (decouples matching from feature engineering), experiment configuration as data (YAML-driven, reproducible), and common evaluation interface (every model implements the same `predict()` signature). See `.planning/research/ARCHITECTURE.md` for full component design and code examples.

**Major components:**
1. **Data Ingestion Layer** -- Platform-specific adapters that normalize raw API data into common schema. Handles Kalshi live/historical split, Polymarket's three-API structure, rate limiting, and disk caching.
2. **Market Matching Pipeline** -- Two-stage (keyword then semantic similarity) matching with human curation. Outputs a versioned `matched_pairs.json` registry.
3. **Feature Engineering** -- Time-aligns price series, computes microstructure features, produces flat and windowed datasets for all model tiers from the same underlying data.
4. **Model Layer (Tiers 1-3)** -- All models behind a `BaseModel` interface. Tier 1 (regression) takes flat vectors. Tier 2 (time series) takes windowed sequences. Tier 3 (RL) uses a custom Gym environment.
5. **Autoencoder** -- Trained separately as upstream filter. Reconstruction error fed to PPO as continuous feature or binary gate. Trained on training data only; threshold set on validation split.
6. **Evaluation and Simulation** -- Unified scoring of all models, profit simulation engine, SHAP analysis, experiment runners with YAML configs.

### Critical Pitfalls

See `.planning/research/PITFALLS.md` for all 16 pitfalls with detailed prevention strategies.

1. **Look-ahead bias in train/test split** -- Use strict temporal cutoffs. Never shuffle time series. Add assertion: `train_timestamps.max() < test_timestamps.min()`. This invalidates ALL results if violated.
2. **Cross-platform time alignment errors** -- Normalize all timestamps to UTC. Bucket Polymarket trades to match Kalshi candlestick boundaries. Visually validate alignment on known events. Misalignment creates phantom spreads that models learn to "predict."
3. **Market matching false positives** -- Semantic similarity cannot distinguish settlement criteria differences. Manually review EVERY matched pair. Log settlement source, criteria, and timezone for each. False matches poison all downstream training.
4. **Insufficient data for statistical significance** -- Dataset size is unknown until matching runs. If <30 pairs, reduce scope (drop TFT, simplify ablations). Use bootstrap confidence intervals. Frame findings as directional evidence.
5. **PPO degenerate policies** -- Monitor action distributions during training. If >90% same action, policy has collapsed. Compare against random trading baseline. Accept "PPO learns not to trade" as a valid finding.

## Implications for Roadmap

Based on combined research, the project decomposes into 8 phases driven by strict data-flow dependencies and the April 4 TA check-in constraint.

### Phase 1: Project Setup and Data Ingestion
**Rationale:** Everything depends on raw data. API connectivity is verified but ingestion scripts need building. This is the longest-pole dependency and must start immediately. Rate-limited API calls make this time-consuming (Pitfall 12).
**Delivers:** Kalshi and Polymarket adapter classes, raw OHLCV parquet files in `data/raw/`, caching infrastructure, rate limiting.
**Addresses:** Dual-platform API ingestion (table stakes), data quality foundation.
**Avoids:** Pitfall 12 (API rate limiting delays), Pitfall 14 (null OHLC handling), Pitfall 3 (timestamp alignment -- normalize to UTC from the start).
**Stack:** requests, pandas, numpy. Platform adapter pattern from ARCHITECTURE.md.

### Phase 2: Market Matching Pipeline
**Rationale:** Cannot build features without matched pairs. This is the highest-risk phase -- false matches poison everything downstream (Pitfall 4). Also determines dataset size, which gates scope decisions (Pitfall 6).
**Delivers:** `matched_pairs.json` registry with manually verified pair mappings, match quality scores, settlement criteria documentation.
**Addresses:** Market matching pipeline (table stakes), settlement divergence documentation (differentiator).
**Avoids:** Pitfall 4 (false positive matches), Pitfall 2 (survivorship bias -- document exclusions), Pitfall 6 (insufficient data -- get pair count early).
**Stack:** sentence-transformers (all-MiniLM-L6-v2), pandas.

### Phase 3: Feature Engineering and Dataset Construction
**Rationale:** All models consume feature matrices. Time alignment is critical (Pitfall 3) and feature design affects every downstream result (Pitfall 8). Must include `time_idx` and `group_ids` columns from the start for TFT compatibility.
**Delivers:** `data/processed/features.parquet` with per-hour microstructure vectors, temporal train/test split, flat and windowed dataset classes, feature normalization pipeline.
**Addresses:** Time-aligned dataset, microstructure features, lookback windows, target variable construction, liquidity filtering (all table stakes).
**Avoids:** Pitfall 1 (look-ahead bias -- enforce temporal split), Pitfall 3 (time alignment errors), Pitfall 8 (feature leakage -- predict changes, not levels), Pitfall 5 in ARCHITECTURE.md (comparing models on different feature sets).

### Phase 4: Tier 1 Baselines and Evaluation Framework
**Rationale:** Simple models validate the feature matrix and establish lower bounds. The evaluation framework must exist before complex models are built -- it enforces the common `PredictionResult` interface. This phase produces the TA check-in deliverable.
**Delivers:** Linear Regression and XGBoost trained and evaluated, naive baselines (spread-closes, volume-correct), profit simulation engine, RMSE/MAE/directional accuracy/P&L/Sharpe metrics, EDA notebook.
**Addresses:** All Tier 1 models, evaluation framework, profit simulation, naive baselines (table stakes). Dual evaluation paradigm (differentiator).
**Avoids:** Pitfall 15 (confusing prediction vs. trading performance -- report both), Pitfall 10 (Sharpe on small samples -- include bootstrap CIs), Pitfall 16 (ignoring transaction costs -- add rough estimates).
**Stack:** scikit-learn, XGBoost, matplotlib, seaborn.

### Phase 5: Tier 2 Time Series Models
**Rationale:** GRU/LSTM are the natural next step after regression baselines. They test whether temporal structure in the data adds predictive value. TFT is attempted last within this phase and only if dataset is large enough.
**Delivers:** GRU model (primary recurrent baseline), LSTM model (comparison), TFT model (if viable). All evaluated through the existing evaluation framework.
**Addresses:** GRU, LSTM, TFT (table stakes). TFT attention interpretability (differentiator).
**Avoids:** Pitfall 11 (TFT overkill -- implement last, time-box to 1 day, skip if <100 pairs).
**Stack:** PyTorch (nn.GRU, nn.LSTM), PyTorch Forecasting (TFT), PyTorch Lightning. Train on MPS.

### Phase 6: Tier 3 RL and Autoencoder
**Rationale:** The RL components are the most complex and most likely to underperform. Building them after Tiers 1-2 ensures the project has strong results even if PPO fails. The autoencoder must be trained before the PPO+AE variant.
**Delivers:** Custom Gym environment for spread trading, autoencoder trained on normal spread behavior, PPO on raw features (addresses professor's question), PPO + autoencoder signal filter (the proposed novel architecture).
**Addresses:** PPO variants, autoencoder anomaly detection (table stakes). PPO ablation isolating RL value (differentiator).
**Avoids:** Pitfall 5 (degenerate PPO policies -- monitor action distributions), Pitfall 9 (arbitrary anomaly threshold -- report multiple thresholds, consider continuous input). Anti-pattern 4 from ARCHITECTURE.md (training RL on full dataset).
**Stack:** Stable Baselines 3, Gymnasium, PyTorch (autoencoder). Train PPO on CPU.

### Phase 7: Experiments and Interpretability
**Rationale:** All models must exist before the centerpiece comparison (Experiment 1). Ablation experiments and SHAP analysis run on top of trained models and can be parallelized.
**Delivers:** Experiment 1 (cross-tier architecture comparison), Experiment 2 (window length ablation), Experiment 3 (spread threshold ablation), SHAP feature importance on XGBoost and neural models, comparison tables and figures.
**Addresses:** All three experiments (table stakes). SHAP analysis, comprehensive ablation story (differentiators).
**Avoids:** Pitfall 13 (SHAP on time series -- use TreeSHAP on XGBoost, attention for TFT), Pitfall 10 (Sharpe on small backtests -- bootstrap CIs).
**Stack:** SHAP, matplotlib, seaborn, experiment YAML configs.

### Phase 8: Paper and Presentation
**Rationale:** Depends on experiment results. Framing should be outlined early (by April 14) but final writing happens last. The narrative is "complexity vs. performance" -- PPO underperformance is a finding, not a failure.
**Delivers:** Final paper (standard academic structure), lightning talk slides (5-10 slides).
**Addresses:** Paper, slides (table stakes). Intellectual honesty framing (differentiator).
**Avoids:** Pitfall C (framing PPO underperformance as failure), Pitfall A (scope creep -- prioritize ruthlessly).

### Phase Ordering Rationale

- **Phases 1-3 are strictly sequential** -- each phase's output is the next phase's input. No parallelization possible. This is the critical path.
- **Phase 4 must follow Phase 3** -- baselines validate the feature matrix. But evaluation code can be written in parallel with Phase 3 (just needs the `PredictionResult` interface defined).
- **Phases 5 and 6 can run in parallel** -- two team members can split (Ian on Tier 2, Alvin on Tier 3, or vice versa). Both depend only on Phase 3 output and Phase 4's evaluation framework.
- **Phase 7 depends on all model phases** -- experiments need all models trained.
- **Phase 8 can begin outlining during Phase 7** -- paper framing should not wait for final results.
- **The April 4 TA check-in maps to Phase 4 completion** -- working pipeline + baseline results.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2 (Market Matching):** Highest-risk phase. Matching quality determines project viability. Needs research into specific Kalshi/Polymarket category overlap, settlement criteria documentation, and optimal similarity thresholds. Consider `/gsd:research-phase` before execution.
- **Phase 6 (RL + Autoencoder):** Reward function design, Gym environment structure, autoencoder architecture, and threshold strategy all need careful design. Brainstorming session (Superpowers) strongly recommended before coding.
- **Phase 3 (Feature Engineering):** TFT's `TimeSeriesDataSet` format requirements need research -- getting the column structure wrong causes painful refactoring later.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Data Ingestion):** REST API calls with caching. Well-documented pattern, API endpoints verified.
- **Phase 4 (Tier 1 Baselines):** scikit-learn LinearRegression, XGBoost regressor, standard metrics. No research needed.
- **Phase 5 (GRU/LSTM):** Standard PyTorch training loops. TFT may need light research for `TimeSeriesDataSet` config, but patterns exist in PyTorch Forecasting docs.
- **Phase 7 (Experiments):** Config-driven experiment runners. Standard pattern.
- **Phase 8 (Paper):** Standard academic writing.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All packages verified installed and import-tested. API connectivity confirmed. MPS and device strategy validated empirically. |
| Features | HIGH | Well-scoped by detailed project proposal. Feature landscape maps directly to established financial ML patterns. Grading priorities are clear. |
| Architecture | HIGH | Standard ML pipeline patterns (adapters, registries, common interfaces). All libraries have mature, stable APIs. |
| Pitfalls | MEDIUM-HIGH | Core pitfalls (look-ahead bias, degenerate RL, survivorship bias) are extremely well-established. API-specific pitfalls based on project documentation. No web verification available this session. |

**Overall confidence:** HIGH

### Gaps to Address

- **Actual dataset size is unknown.** The number of matchable Kalshi-Polymarket pairs will only be determined when the matching pipeline runs. This is the single biggest uncertainty. If <30 pairs, TFT should be dropped and ablation scope reduced. Run matching pipeline ASAP (Phase 2) to resolve.
- **Polymarket CLOB and Data API endpoints not connectivity-tested.** Kalshi and Gamma API are verified working, but CLOB and Data API responses need validation during Phase 1 implementation.
- **Autoencoder effectiveness in this domain is unproven.** The pattern (autoencoder anomaly detection as RL filter) is established in manufacturing and equities but has not been validated for prediction market spreads specifically. Confidence in the approach is MEDIUM.
- **Web search was unavailable for all research sessions.** All findings are based on training data and project documentation. Stack versions and API behaviors should be spot-checked against current docs during implementation, though all versions were verified via direct import.
- **Transaction cost estimates are rough.** Fee structures for Kalshi and Polymarket may have changed. The break-even analysis approach sidesteps this by reporting threshold rather than exact cost.

## Sources

### Primary (HIGH confidence)
- All stack versions verified via direct import in `.venv/` on 2026-04-01
- API connectivity verified via live HTTP requests (Kalshi, Polymarket Gamma)
- MPS compatibility and SB3 device warnings verified empirically
- Project proposal PDF (DS340 Project Proposal, Spring 2026)
- CLAUDE.md project instructions and technical context
- PROJECT.md validated requirements and constraints

### Secondary (MEDIUM-HIGH confidence)
- Financial ML best practices (de Prado, "Advances in Financial Machine Learning," 2018) -- backtesting pitfalls, temporal splitting
- RL for trading survey literature -- degenerate policy patterns, reward shaping
- Established ML pipeline patterns (scikit-learn, PyTorch, Gymnasium, PyTorch Forecasting)

### Tertiary (MEDIUM confidence)
- Prediction market-specific API behaviors -- based on project documentation, not live testing of all endpoints
- Autoencoder anomaly detection effectiveness for this specific domain -- pattern is established elsewhere, untested here

---
*Research completed: 2026-04-01*
*Ready for roadmap: yes*
