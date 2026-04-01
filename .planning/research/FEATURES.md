# Feature Landscape

**Domain:** Cross-platform prediction market arbitrage (Kalshi vs. Polymarket) with ML/RL complexity analysis
**Researched:** 2026-04-01
**Overall confidence:** HIGH (well-scoped academic project with detailed proposal; domain is established financial ML)

## Table Stakes

Features users (graders/TA/professor) expect. Missing any of these = project feels incomplete or earns a significantly lower grade.

### Data Pipeline

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Dual-platform API ingestion (Kalshi + Polymarket) | Core data source; without both platforms there is no cross-platform analysis | Medium | Polymarket has 3 separate APIs (Gamma, CLOB, Data API) with different rate limits. Kalshi has live/historical endpoint split. Both require careful pagination and error handling. |
| Market matching pipeline (keyword + semantic similarity) | The entire project depends on identifying the same real-world event across platforms with different naming conventions | High | No shared identifiers exist. Sentence-transformers for candidate generation, then manual curation for settlement criteria alignment. This is the hardest data engineering task. |
| Time-aligned matched-pairs dataset | Models need synchronized time series from both platforms to compute spreads | Medium | Must handle timezone alignment, missing data periods (no trades), and different granularity between platforms. Hourly aggregation is the target. |
| Liquidity filtering | Low-liquidity markets produce meaningless price data (Kalshi hourly contracts with <10 trades) | Low | Simple volume threshold. But choosing the threshold affects dataset size significantly. |
| Data quality validation and exploratory analysis | TA check-in (April 4) requires demonstrating dataset size, distributions, and sanity | Low | Notebook-level work: histograms of spread distributions, volume distributions, number of matched pairs, temporal coverage. |

### Feature Engineering

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Per-hour microstructure feature vectors | Models need structured inputs; raw price histories are insufficient for tabular and RL models | Medium | Spread (absolute and percentage), volume per platform, bid-ask spread, price velocity (rate of change), volume imbalance. These are the core signal. |
| Lookback window features | Time series context for non-sequential models (Linear Regression, XGBoost) | Low | Rolling means, rolling std, min/max over configurable windows (6h, 24h, 72h, 7d). Simple pandas operations. |
| Target variable construction | Models need a clear prediction target | Low | For regression: future spread value at T+N hours. For classification-like trading: binary "spread converges within T hours." For RL: reward is realized P&L. |

### Model Suite (Tiered Complexity Comparison)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Naive baselines (spread-always-closes, volume-platform-correct) | Lower bound; if ML models cannot beat these, the arbitrage signal may not exist | Low | Two simple heuristics: (1) predict spread always goes to zero by resolution, (2) higher-volume platform price is always correct. These must be implemented and evaluated. |
| Tier 1: Linear Regression baseline | Professor KG explicitly asked if this could "just be regression"; this must be strong | Low | scikit-learn LinearRegression on feature vectors. Must include profit simulation, not just RMSE. |
| Tier 1: XGBoost baseline | Standard strong tabular baseline in any ML comparison study | Low | XGBoost regressor with basic hyperparameter tuning. Feature importance comes for free. |
| Tier 2: GRU and/or LSTM | Recurrent models are the natural choice for sequential spread data | Medium | PyTorch implementation. GRU preferred (fewer parameters, faster training). LSTM as comparison. Both need proper sequence batching, train/val/test splits respecting temporal ordering. |
| Tier 2: TFT (Temporal Fusion Transformer) | Strongest time series model in the suite; handles mixed static/temporal features natively | High | Via PyTorch Forecasting library. Configuration-heavy (quantile loss, attention interpretability, variable selection). Worth the effort for the interpretability output it provides. |
| Tier 3: PPO agent (on raw features) | Per professor feedback -- isolates whether RL adds value independent of the autoencoder | High | Custom PPO implementation in PyTorch. State = feature vector, actions = {buy Kalshi, buy Polymarket, hold, exit}, reward = realized P&L. Requires environment simulation, reward shaping, and careful training loop. |
| Tier 3: PPO + Autoencoder signal filter | The leading/novel architecture from the proposal | High | Autoencoder trained on "normal" spread behavior, flags anomalies via high reconstruction error. PPO acts only on flagged windows. Two-stage training pipeline. |

### Evaluation Framework

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Regression metrics (RMSE, MAE, directional accuracy) | Standard ML evaluation; expected in any academic paper | Low | scikit-learn metrics. Directional accuracy (did we predict the sign of spread change correctly?) is more informative than RMSE for trading. |
| Profit simulation (P&L, win rate, Sharpe ratio) | Transforms regression results into actionable trading outcomes; this is what makes the project concrete | Medium | Must simulate: entry when model signals opportunity, exit when spread converges or timeout, track cumulative P&L per trade. Need to define position sizing, entry/exit thresholds. |
| Temporal train/test split | Avoids look-ahead bias; any financial ML paper requires this | Low | Train on earlier data, test on more recent data. No random splitting of time series. |
| Cross-model comparison table and figures | The centerpiece experiment demands a clear summary visualization | Low | Table with all models x all metrics. Bar charts or radar plots for visual comparison. This IS the main result of the paper. |

### Experiments

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Experiment 1: Complexity-vs-performance (all tiers, same eval) | This is the research question. Without it, there is no paper. | Medium | Run all models on same dataset/split/eval protocol. Compare across tiers. The result (likely: simpler models win) directly answers the research question. |
| Experiment 2: Window length ablation (6h, 24h, 72h, 7d) | Shows sensitivity to lookback horizon; standard ablation in time series work | Low | Re-run best model(s) with different lookback windows. Plot performance vs. window length. |
| Experiment 3: Spread threshold ablation (none, >2pp, >5pp, >10pp) | Tests whether filtering for larger spreads improves signal quality at the cost of fewer samples | Low | Re-filter dataset at each threshold, retrain, evaluate. Documents the sample-size vs. signal-quality tradeoff. |

### Deliverables

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Final paper | Course requirement. Must document methodology, results, analysis. | Medium | Standard academic paper structure: intro, related work, methods, experiments, results, discussion, conclusion. |
| Lightning talk slides | Course requirement for presentation. | Low | 5-10 slides summarizing the key finding (complexity vs. performance). |
| SHAP interpretability analysis | Listed in requirements; explains which features drive predictions | Low | SHAP on XGBoost (tree-based, fast) and optionally on neural models. Shows which microstructure features matter most. |

## Differentiators

Features that set this project apart from a "standard" DS340 project. Not required to pass, but valued for grade and intellectual contribution.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Autoencoder anomaly detection as RL signal filter | Novel architecture combination for this domain. No published work (to our knowledge) applies autoencoder-filtered PPO to prediction market arbitrage. | High | The autoencoder reframes the problem: instead of predicting price, detect "abnormal" spread behavior. This is genuinely novel framing for this application domain. Even if PPO underperforms, the anomaly detection analysis is independently interesting. |
| PPO-without-autoencoder ablation (isolating RL value) | Directly addresses professor's critique ("Could RL act directly without the autoencoder?"). Shows intellectual rigor. | Medium | This is the difference between "we built a complex thing" and "we systematically tested whether each component adds value." The ablation story is what makes this a strong project. |
| Dual evaluation paradigm (regression metrics + trading simulation) | Most academic ML projects only report regression metrics. Adding simulated trading outcomes grounds the analysis in real-world relevance. | Medium | The gap between "low RMSE" and "profitable trading strategy" is often large. Showing both tells a richer story. Sharpe ratio is particularly impressive to include. |
| Cross-platform market matching pipeline | Building the matched-pairs dataset from scratch (not using an existing dataset) demonstrates real data engineering skill. | High | This is unglamorous but impressive. NLP-based entity resolution across platforms with different naming conventions, settlement criteria, and data formats. Few academic projects do their own entity resolution. |
| Settlement divergence documentation | Acknowledging that the same "event" may settle differently on Kalshi vs. Polymarket shows domain expertise and intellectual honesty. | Low | Some spreads may not be true arbitrage if settlement criteria differ (e.g., different time cutoffs, different definitions of "above $X"). Documenting this is a mark of rigor. |
| Pure microstructure feature set (deliberate exclusion of external signals) | Isolates whether the arbitrage signal is self-contained within market behavior, answering a real research question. | Low | Most financial ML papers throw in every available signal. Deliberately constraining to microstructure-only and justifying why makes a cleaner experimental design. |
| TFT variable selection and attention interpretability | TFT produces attention weights and variable importance as a side effect of training, offering built-in interpretability without post-hoc methods. | Low (if TFT is already built) | Free interpretability beyond SHAP. Shows which time steps and which features the model attends to. Good figure for the paper. |
| Comprehensive ablation story (3 experiments) | Having three distinct experiments (architecture, window length, threshold) shows systematic thinking rather than "we built a model and reported accuracy." | Medium | Each experiment answers a different question. Together they tell a complete story about what matters for prediction market arbitrage. |

## Anti-Features

Features to explicitly NOT build. These are tempting but would hurt the project by adding scope, reducing focus, or introducing problems that cannot be solved in the timeline.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Live trading / real-money execution | Massively out of scope. Requires brokerage integration, position management, real-time risk controls. Also ethically questionable for an academic project. | Historical backtesting with simulated P&L. Acknowledge in paper that live execution would require additional infrastructure. |
| Transaction cost modeling | Kalshi fees and Polymarket gas costs are complex, variable, and would dominate the spread signal for small discrepancies. Modeling them requires assumptions that introduce more uncertainty than value. | Acknowledge transaction costs in the paper's limitations section. Note that real arbitrage profit would be lower than simulated. This is honest and sufficient. |
| External signal features (news, sentiment, social media) | Breaks the experimental design. The research question is whether microstructure alone contains the arbitrage signal. Adding external signals muddies the answer. | Keep features pure microstructure. If the project had a second phase, external signals would be a natural extension -- mention this in "Future Work." |
| Pretrained financial models (FinBERT, pretrained LLMs) | Professor KG would likely view this as "letting someone else's model do the work." Training from scratch demonstrates understanding. | All models trained from scratch on the matched-pairs dataset. Sentence-transformers for matching is the one exception, justified because matching is a data pipeline step, not the model under evaluation. |
| Real-time streaming pipeline | Massively overengineered for an academic project. WebSocket connections, real-time feature computation, and streaming inference add engineering complexity without analytical value. | Batch historical data pipeline. Process all data offline, generate feature matrices, train and evaluate models. |
| Web dashboard or visualization app | Not a software product project. Time spent on Streamlit/Dash/React is time not spent on modeling and analysis. | Jupyter notebooks for exploration, matplotlib/seaborn figures for the paper. Scripts for reproducibility. |
| Order book depth features | Polymarket and Kalshi order book snapshots are not reliably available historically. Attempting to use them would create data sparsity issues. | Stick to trade-derived features: price, volume, bid-ask spread (from candlestick data where available). Mention order book features as future work. |
| Multi-platform expansion (PredictIt, Metaculus, etc.) | More platforms = more matching complexity, more API quirks, more edge cases. Two platforms is already hard enough. | Kalshi and Polymarket only. Mention extensibility in future work. |
| Ensemble of all models | Tempting to combine predictions, but it obscures the complexity-vs-performance finding. If the ensemble wins, you cannot attribute performance to a specific tier. | Evaluate each model independently. The comparison IS the contribution. An ensemble would undermine the research question. |
| Hyperparameter optimization (Bayesian, grid search) | Time-consuming and distracts from the architecture comparison. With a small dataset, HPO overfits easily. | Reasonable defaults with light manual tuning. Document hyperparameter choices and justify them. If time permits at the end, do one round of sensitivity analysis. |

## Feature Dependencies

```
Dual-platform API ingestion
  |
  v
Market matching pipeline (keyword + semantic similarity)
  |
  v
Time-aligned matched-pairs dataset
  |
  +---> Liquidity filtering
  |       |
  |       v
  +---> Per-hour microstructure feature vectors
          |
          +---> Lookback window features
          |       |
          |       v
          +---> Target variable construction
                  |
                  +---> Naive baselines (spread-always-closes, volume-correct)
                  |
                  +---> Tier 1: Linear Regression, XGBoost
                  |       |
                  |       +---> SHAP interpretability
                  |
                  +---> Tier 2: GRU, LSTM, TFT
                  |       |
                  |       +---> TFT attention interpretability
                  |
                  +---> Autoencoder (trained on normal spread behavior)
                  |       |
                  |       v
                  +---> Tier 3: PPO on raw features (no autoencoder dependency)
                  |
                  +---> Tier 3: PPO + Autoencoder signal filter
                          |
                          v
                  All models feed into:
                  +---> Evaluation framework (regression metrics)
                  +---> Profit simulation (P&L, Sharpe)
                  +---> Experiment 1: cross-tier comparison
                          |
                          v
                  +---> Experiment 2: window length ablation
                  +---> Experiment 3: spread threshold ablation
                          |
                          v
                  +---> Final paper + lightning talk
```

**Critical path:** API ingestion --> Market matching --> Matched-pairs dataset --> Feature engineering --> Tier 1 baselines --> Evaluation framework --> Tier 2 models --> Autoencoder --> PPO --> Experiments --> Paper.

**Parallelizable work:**
- After feature engineering: Tier 1 models and autoencoder training can proceed in parallel.
- After evaluation framework exists: All models can be evaluated independently.
- Experiments 2 and 3 are independent and can run in parallel.
- Paper writing can begin once Experiment 1 results are available, even before Experiments 2 and 3 complete.

## MVP Recommendation

For the TA check-in on April 4, the minimum viable deliverable is:

**Must have by April 4:**
1. Dual-platform API ingestion (working scripts that pull Kalshi and Polymarket data)
2. Market matching pipeline (at least keyword-based, with semantic similarity in progress)
3. Initial matched-pairs dataset (even if small, with manual verification of a subset)
4. Per-hour feature vectors computed for matched pairs
5. Linear Regression and XGBoost trained and evaluated (RMSE, MAE, directional accuracy)
6. Basic profit simulation on at least one baseline
7. EDA notebook showing dataset size, spread distributions, volume distributions

**Defer to after April 4:**
- TFT implementation (highest complexity of time series models, PyTorch Forecasting is configuration-heavy)
- PPO agent (both variants) -- this is the most complex model and most likely to underperform
- Autoencoder training -- depends on having enough data to define "normal" behavior
- SHAP analysis -- requires trained models, easy to add at the end
- Experiments 2 and 3 -- ablations run after the primary comparison is done
- Paper and slides -- last week (April 20-27)

**Risk-ordered priorities after April 4:**
1. GRU/LSTM (Medium complexity, likely strong performers, fill out Tier 2)
2. Autoencoder (needed before PPO+autoencoder variant)
3. PPO on raw features (answers KG's question directly)
4. PPO + Autoencoder (the "leading" architecture)
5. TFT (impressive if included but not fatal if dropped)
6. Experiment 1 (the centerpiece -- run as soon as all models exist)
7. Experiments 2 and 3 (ablations -- nice to have)
8. SHAP analysis (quick once models are trained)
9. Paper and slides

## Grading Considerations

For a DS340 final project, the grade-driving features are (in rough order of importance):

1. **Working end-to-end pipeline** -- data in, predictions out, evaluation complete. A project with simpler models that works beats a project with complex models that does not run.
2. **Clear research question answered with evidence** -- "Does complexity help?" answered with comparative tables. The answer "no" is perfectly valid if supported.
3. **Proper evaluation methodology** -- temporal splits (no leakage), multiple metrics, naive baselines as lower bounds. These show ML maturity.
4. **Intellectual honesty** -- acknowledging when PPO underperforms, documenting settlement divergence, noting limitations. This is what separates an A from a B+.
5. **Novel or interesting framing** -- the autoencoder as anomaly detector (not predictor) is a genuinely interesting framing. The cross-platform matching pipeline is real data engineering. These differentiate from "we ran XGBoost on a Kaggle dataset."
6. **Code quality and reproducibility** -- clean code, tests, documented pipelines. The TA can re-run your experiments.

## Sources

- Project proposal PDF (DS340 Project Proposal (2).pdf) -- primary source for project scope, methods, and timeline
- PROJECT.md -- validated requirements and constraints
- CLAUDE.md -- technical context, API gotchas, model architecture details
- Training data knowledge of: Kalshi/Polymarket APIs, prediction market mechanics, financial ML arbitrage systems, PPO/autoencoder architectures, academic ML project evaluation criteria (HIGH confidence -- well-established domains)
- Note: WebSearch and WebFetch were unavailable for this research session. Findings rely on project documents and training data. Confidence remains HIGH because the domain (financial ML, prediction markets, academic project scoping) is well-covered in training data and the project documents are exceptionally detailed.
