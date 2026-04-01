# Domain Pitfalls

**Domain:** Cross-platform prediction market arbitrage with ML/RL (academic project)
**Researched:** 2026-04-01
**Overall confidence:** MEDIUM-HIGH (based on established financial ML literature, prediction market mechanics, and RL training dynamics; no web verification available this session)

---

## Critical Pitfalls

Mistakes that cause rewrites, invalidate results, or sink the project timeline.

---

### Pitfall 1: Look-Ahead Bias in Time-Series Train/Test Split

**What goes wrong:** Using random train/test splits (or k-fold cross-validation) on time-series data allows the model to train on future information and test on past data. Results look great in backtest but are meaningless. In this project, if you split the matched-pairs dataset randomly rather than temporally, every model will appear to predict spread convergence well because it has already "seen" nearby time points from the same contract.

**Why it happens:** Scikit-learn defaults (train_test_split, cross_val_score) use random shuffling. It is the path of least resistance. With matched pairs across two platforms, it is also tempting to split by market pair rather than by time, which still leaks temporal information if pairs overlap in time.

**Consequences:** All reported metrics (RMSE, MAE, Sharpe, P&L) are inflated and unreproducible. A reviewer or TA who asks "what happens on truly unseen future data?" will get a blank stare. The entire experiment section of the paper becomes invalid.

**Prevention:**
1. Use a strict temporal cutoff: train on data before date T, validate on T to T+delta, test on T+delta onward. Never shuffle.
2. For the window-length ablation, ensure the test set is always in the future relative to all training windows.
3. Implement a `TimeSeriesSplit`-style validator if doing any hyperparameter tuning. Do not use `sklearn.model_selection.KFold`.
4. Add an assertion in the data loader: `assert train_timestamps.max() < test_timestamps.min()`.

**Detection:** Before running any experiment, print the min/max timestamps of train, val, and test sets. If they overlap, stop.

**Phase relevance:** Feature engineering / dataset construction phase and experiment phase. The split must be decided and enforced before any model is trained.

**Confidence:** HIGH -- this is the single most well-documented pitfall in financial ML literature (de Prado 2018, Bailey et al. 2017).

---

### Pitfall 2: Survivorship / Availability Bias in Market Selection

**What goes wrong:** You only collect data from markets that are currently visible on both platforms, or markets that resolved successfully. This biases the dataset toward markets with clean outcomes and high liquidity. Markets that were delisted, had settlement disputes, or resolved ambiguously are excluded, which inflates apparent arbitrage profitability.

**Why it happens:** Both Kalshi and Polymarket APIs make it easier to discover active/recently-resolved markets than to find historical markets that were delisted or had edge-case resolutions. Polymarket's `/prices-history` returning empty for resolved markets (documented in CLAUDE.md) is already a symptom -- you are forced into a reconstruction path that may silently drop markets with no trade records.

**Consequences:** The dataset overrepresents "clean" arbitrage opportunities and underrepresents messy real-world cases. Reported profitability is optimistic. The paper's external validity claims are weakened.

**Prevention:**
1. Document your market universe explicitly: how many markets existed on each platform during the study period, how many you matched, how many you excluded and why.
2. Log every exclusion with a reason code (no trades, delisted, settlement dispute, low liquidity, unmatched).
3. Report results on both the filtered dataset AND the full dataset (including markets with sparse data) as a sensitivity analysis.
4. In the paper, have a "Limitations" paragraph specifically about survivorship bias.

**Detection:** If your matched-pairs dataset has a suspiciously high hit rate (>80% of spreads converge profitably), survivorship bias is likely at play.

**Phase relevance:** Data pipeline phase. Must be addressed during market discovery and matching.

**Confidence:** HIGH -- well-established concept in quantitative finance.

---

### Pitfall 3: Cross-Platform Time Alignment Errors

**What goes wrong:** Kalshi and Polymarket report timestamps differently (timezone handling, trade time vs. settlement time, candlestick period boundaries). If you join prices across platforms on "the same hour" but the hour boundaries are offset, or one platform uses trade execution time while the other uses order placement time, your spread calculations are systematically wrong. A 15-minute misalignment on an hourly candlestick can create phantom spreads.

**Why it happens:** Neither platform documents their timestamp semantics in great detail. Kalshi uses hourly candlesticks with `period_interval=60`, but the exact boundary alignment (top-of-hour? rolling?) is not always obvious. Polymarket trade records have individual timestamps but reconstructing OHLC bars requires choosing bucketing boundaries. If the two platforms are bucketed with different offsets, you get misaligned snapshots.

**Consequences:** The model learns to "predict" phantom spreads that are actually measurement artifacts. If spreads appear large due to misalignment and then "converge" when properly aligned data arrives, you get a model that looks good in backtest but is predicting noise. This is a subtle form of data leakage.

**Prevention:**
1. Normalize all timestamps to UTC with explicit timezone handling. Never rely on naive datetimes.
2. For Polymarket trade reconstruction, bucket trades into hourly bars aligned to the same boundary as Kalshi candlesticks. Verify alignment by inspecting a few known events where both platforms should show identical price movements.
3. Add a sanity check: for markets near resolution (final few hours), both platforms should converge toward the same settlement price. If they do not, your alignment is wrong.
4. Include a data validation step that plots both platform prices for a few well-known markets and visually confirms they track each other.

**Detection:** Plot raw price series from both platforms for 3-5 manually verified market pairs. If the series are consistently offset by a fixed lag, you have an alignment error.

**Phase relevance:** Data pipeline phase, specifically the time-alignment step after raw ingestion.

**Confidence:** HIGH -- timestamp misalignment is one of the most common bugs in cross-venue financial systems.

---

### Pitfall 4: Market Matching False Positives Poison the Dataset

**What goes wrong:** The sentence-transformer matching pipeline finds contracts that look similar (e.g., "Will BTC exceed $100K by March 2026?" on Kalshi vs. "Bitcoin price above $100K on March 31, 2026" on Polymarket) but have subtly different settlement criteria (one resolves on close price, the other on any intraday touch; one uses CoinGecko, the other uses Binance spot). These false matches create artificial spreads that reflect genuine settlement disagreement, not arbitrage opportunities.

**Why it happens:** Semantic similarity captures meaning overlap but cannot distinguish precise legal terms. Prediction markets are quasi-legal instruments where exact settlement definitions matter enormously. A cosine similarity of 0.95 between two contract titles does not mean the contracts are equivalent.

**Consequences:** False-matched pairs inject noise into the training data. The model trains on "spreads" that will never converge because the contracts measure different things. Worse, some of these false matches may appear to converge by coincidence, training the model on spurious patterns.

**Prevention:**
1. The automatic matching step (sentence-transformer similarity) must be followed by manual human review of EVERY matched pair. With an expected dataset of dozens to low hundreds of pairs, this is feasible.
2. During manual review, check: (a) identical underlying event, (b) identical resolution date/time, (c) identical settlement source (or at least highly correlated sources), (d) identical settlement criteria (touch vs. close, exact threshold).
3. Create a match quality score (exact match, close match, weak match) and run experiments both with all matches and with only exact matches.
4. Document settlement criteria differences for each pair in the dataset metadata.

**Detection:** If a matched pair shows a persistent spread that does not converge even near resolution, it is likely a false match or has divergent settlement criteria. Flag pairs where the final spread at resolution exceeds 5 percentage points.

**Phase relevance:** Market matching phase. This is the highest-risk phase of the entire project because all downstream work depends on match quality.

**Confidence:** HIGH -- this is specific to cross-platform prediction market work and well-understood in practice.

---

### Pitfall 5: PPO Reward Shaping That Creates Degenerate Policies

**What goes wrong:** The PPO agent learns to exploit the reward function rather than learn a genuine trading strategy. Common degenerate policies: (a) never trade (earns 0 reward, avoids all losses -- optimal if penalties for inaction are weak), (b) always trade in one direction (exploits an asymmetry in the reward function or dataset), (c) overfit to a single market pair that had high returns in training.

**Why it happens:** Reward shaping for financial RL is notoriously difficult. If reward = P&L, the agent optimizes for high-variance moonshot trades. If reward = Sharpe ratio over an episode, the agent may learn to make one safe trade and then stop. If reward includes a penalty for not trading, the agent may overtrade. The small dataset makes this worse because there are few enough episodes that degenerate strategies can achieve high training reward by memorizing.

**Consequences:** The PPO results section of the paper either shows a trivial policy ("the agent learned not to trade") or shows overfitted results that do not generalize. Either way, the RL contribution looks weak -- but for the wrong reasons (bad reward shaping, not inherent limitations of RL).

**Prevention:**
1. Design the reward function explicitly and justify each component in the paper: base P&L reward, risk-adjusted component, transaction cost proxy, and inaction penalty (if any).
2. Log the agent's action distribution during training. If >90% of actions are the same (e.g., "hold"), the policy has collapsed.
3. Compare against a random trading baseline (randomly enter/exit positions) to ensure PPO learns above-chance behavior.
4. Use episode-level metrics (total P&L, number of trades, win rate) rather than just mean reward to evaluate policy quality.
5. Accept that a "PPO learns not to trade" result is itself informative and document it honestly rather than hacking the reward to force trading.

**Detection:** Action distribution monitoring. If the policy entropy drops to near zero early in training, the agent has converged to a degenerate strategy.

**Phase relevance:** RL model phase. Reward design should be finalized BEFORE training begins, ideally with a brainstorming/design review session.

**Confidence:** HIGH -- degenerate RL policies are the most common failure mode in financial RL research.

---

### Pitfall 6: Insufficient Data Destroys Statistical Significance

**What goes wrong:** The number of matchable market pairs is unknown but likely small (PROJECT.md says "could be dozens or hundreds"). If you end up with 30-50 matched pairs, even the regression models will have marginal statistical significance. The RL models will almost certainly overfit. Your paper's results section has p-values >0.05 or confidence intervals so wide they are meaningless.

**Why it happens:** Prediction markets are a young industry with limited contract overlap between platforms. The Kalshi-Polymarket intersection on economics/finance and crypto categories is an empirical question that will only be answered when you run the matching pipeline.

**Consequences:** The centerpiece experiment (complexity-vs-performance comparison) lacks statistical power to distinguish between model tiers. You cannot claim "XGBoost outperforms PPO" if the difference is within noise. The paper's contribution is weakened.

**Prevention:**
1. Treat dataset size as a first-class concern. Run the matching pipeline EARLY (first week) to get a realistic count of matched pairs and total data points.
2. If the dataset is very small (<50 matched pairs), adjust scope: drop TFT (needs more data), simplify the ablation experiments, and frame the paper around the data pipeline and matching methodology rather than model comparison.
3. Use bootstrap confidence intervals on all metrics rather than point estimates. Report whether model differences are statistically significant.
4. Consider augmenting the temporal dimension: even if you have only 30 pairs, each pair may have hundreds of hourly observations. Frame the regression as predicting hourly spread changes (many data points) rather than per-market outcomes (few data points).
5. In the paper, be honest about sample size and frame appropriately: "We observe a trend suggesting X, though the sample size limits our ability to make strong claims."

**Detection:** After matching, count your pairs. If <30, trigger contingency planning immediately.

**Phase relevance:** This is a project-level risk that should be evaluated during the data pipeline phase (before model work begins).

**Confidence:** HIGH -- small sample size is the biggest practical risk for this specific project.

---

## Moderate Pitfalls

Mistakes that cause significant rework or weakened results, but are recoverable.

---

### Pitfall 7: Polymarket Price Reconstruction Introduces Systematic Bias

**What goes wrong:** Since Polymarket `/prices-history` returns empty for resolved markets, you must reconstruct OHLC bars from raw trade records. But trade records are noisy: a single large trade can dominate a bar's close price, gaps between trades create ambiguity (does price stay at last trade? or is it undefined?), and reconstruction methodology choices (VWAP vs. last-trade vs. time-weighted) produce materially different price series.

**Prevention:**
1. Document your reconstruction methodology explicitly.
2. Compare multiple reconstruction methods on a few markets where `/prices-history` still works (active markets) to validate that your reconstruction matches the "ground truth."
3. Use VWAP (volume-weighted average price) for the bar's representative price rather than last-trade, as it is more robust to individual outlier trades.
4. When a bar has zero trades, carry forward the last known price but flag it as "stale." Do not interpolate.
5. Report sensitivity: does using last-trade vs. VWAP change your results materially?

**Detection:** Compare your reconstructed prices to any available snapshots. If reconstruction shows 10+ basis points of systematic deviation, investigate.

**Phase relevance:** Data pipeline phase.

**Confidence:** MEDIUM-HIGH -- this is project-specific based on the documented Polymarket API limitation.

---

### Pitfall 8: Feature Leakage Through Spread Calculation

**What goes wrong:** The target variable (future spread or spread convergence) is computed from the same price data used to construct features. If features include "current spread" and the target is "spread at T+1," the model can trivially predict by learning that spreads are autocorrelated. This is not a bug per se -- spreads ARE autocorrelated -- but it means the model's apparent accuracy is dominated by persistence rather than genuine predictive signal.

**Prevention:**
1. Separate features cleanly from targets. Features should include information ABOUT the spread (magnitude, velocity, acceleration, volume imbalance) but the model should be evaluated on its ability to predict CHANGE, not level.
2. Frame the regression target as spread change (delta) or directional movement rather than absolute spread level. A naive "spread stays the same" baseline immediately exposes whether your model adds value beyond persistence.
3. Include the naive baseline (spread stays the same) and the volume baseline (higher-volume platform is correct) as formal baselines. If your ML models do not beat these, the added complexity is not justified -- and that is a valid finding.

**Detection:** If your linear regression achieves R-squared >0.9 on spread prediction, you are almost certainly predicting persistence rather than genuine convergence signal. Check by comparing against a lag-1 naive forecast.

**Phase relevance:** Feature engineering phase and experiment design phase.

**Confidence:** HIGH -- autocorrelation in financial time series is a well-known issue.

---

### Pitfall 9: Autoencoder Anomaly Threshold Selection is Arbitrary

**What goes wrong:** The autoencoder's anomaly detection depends on a reconstruction error threshold to flag "anomalous" spread patterns for the PPO agent. This threshold is a hyperparameter that dramatically changes the system's behavior. Set it too low: everything is anomalous, the filter does nothing. Set it too high: nothing passes through, the PPO agent never trades. There is no principled way to set it without a validation set, and with small data, you cannot afford a separate validation set for threshold tuning.

**Prevention:**
1. Report results across multiple threshold values (percentiles: 90th, 95th, 99th of training reconstruction error). Do not cherry-pick the one that gives the best backtest.
2. Use the threshold as an explicit ablation dimension in Experiment 3 (or add it to Experiment 1).
3. Compare "PPO + autoencoder at threshold X" against "PPO without autoencoder" to isolate whether the filtering helps at each threshold level.
4. Consider using the reconstruction error as a continuous feature input to PPO rather than a binary filter. This avoids the threshold problem entirely and gives the RL agent more information.

**Detection:** If your results are highly sensitive to the threshold (performance swings >30% across adjacent threshold values), the autoencoder is not providing a robust signal.

**Phase relevance:** RL model phase, specifically the autoencoder integration step.

**Confidence:** MEDIUM-HIGH -- threshold sensitivity is a known issue in anomaly detection systems.

---

### Pitfall 10: Sharpe Ratio is Meaningless on Small Backtests

**What goes wrong:** You compute Sharpe ratios on 20-30 trades and report them in the paper. With so few trades, the standard error of the Sharpe estimate is enormous. A reported Sharpe of 2.0 might have a 95% confidence interval of [-1.0, 5.0], making it statistically indistinguishable from zero.

**Prevention:**
1. Report the standard error of the Sharpe ratio alongside the point estimate. The asymptotic standard error is approximately `sqrt((1 + 0.5 * sharpe^2) / N)` where N is the number of return observations.
2. Use bootstrap resampling (with block bootstrap to preserve autocorrelation) to construct confidence intervals on all trading metrics.
3. If the number of trades is <50, explicitly note in the paper that trading metrics have wide confidence intervals and should be interpreted as directional evidence, not precise estimates.
4. Complement Sharpe with simpler metrics that are more robust in small samples: win rate, average profit per trade, maximum drawdown.

**Detection:** Compute the confidence interval. If it spans zero, acknowledge this.

**Phase relevance:** Evaluation / experiment phase.

**Confidence:** HIGH -- widely discussed in quantitative finance (Bailey and de Prado, 2012).

---

### Pitfall 11: TFT Overkill for This Dataset Scale

**What goes wrong:** PyTorch Forecasting's Temporal Fusion Transformer requires substantial data to learn its attention mechanisms effectively. On a dataset of dozens of market pairs with hundreds of hourly observations each, TFT is likely to overfit badly and be extremely slow to train compared to GRU. You spend days debugging TFT configuration and hyperparameters only to get results worse than XGBoost.

**Prevention:**
1. Implement TFT last, after GRU and LSTM. If GRU does not beat XGBoost, TFT almost certainly will not either.
2. Set a time box: spend no more than 1 day on TFT implementation and tuning. If it does not converge or requires extensive hyperparameter search, report "TFT did not converge on this dataset size" as a finding.
3. Use PyTorch Forecasting's built-in hyperparameter suggestions and do not do extensive custom tuning.
4. Consider dropping TFT from the paper if dataset size is <100 pairs and replacing it with a simpler variant or just noting "transformer-based approaches were not viable at this scale."

**Detection:** If TFT training loss plateaus above GRU's within 20 epochs, it is not going to catch up.

**Phase relevance:** Tier 2 model training phase. Gating decision: only attempt TFT if dataset is large enough.

**Confidence:** MEDIUM -- depends on actual dataset size, which is unknown.

---

### Pitfall 12: API Rate Limiting and Data Collection Takes Longer Than Expected

**What goes wrong:** The data collection phase is budgeted for a few hours but actually takes days. Polymarket's per-10-seconds rate limits, Kalshi's historical endpoint pagination, failed requests, retries, and the need to reconstruct Polymarket price history from individual trades all add up. You lose 3-4 days of the timeline on data collection alone.

**Prevention:**
1. Implement caching and checkpointing from the start. Save raw API responses to disk so you never need to re-fetch.
2. Build the data pipeline with resume capability: if it crashes at market pair 47 of 200, it picks up at 47 on restart.
3. Use exponential backoff for rate limit errors (429 responses). Do not just retry immediately.
4. Start data collection on Day 1, not Day 3. It is the longest pole in the tent.
5. Log progress: "Fetched X of Y markets, Y of Z trade records, estimated time remaining: T."

**Detection:** If after 2 hours of running the pipeline you have <10% of the expected data, re-estimate the timeline and adjust scope.

**Phase relevance:** Data pipeline phase. This is timeline-critical given the April 4 TA check-in.

**Confidence:** HIGH -- API data collection almost always takes longer than expected.

---

## Minor Pitfalls

Mistakes that cause annoyance, minor rework, or suboptimal results.

---

### Pitfall 13: SHAP on Time Series Models is Misleading

**What goes wrong:** SHAP values assume feature independence. For time-series features (lagged spread, lagged volume, price velocity), features are highly correlated across time steps. SHAP attributions in this setting can be unstable and misleading, assigning importance to arbitrary lag values rather than the underlying signal.

**Prevention:**
1. Use SHAP primarily on XGBoost (where it is well-understood and fast via TreeSHAP) rather than on GRU/LSTM.
2. For recurrent models, use integrated gradients or attention weights (for TFT) instead of SHAP.
3. Report SHAP feature importances as "suggestive" rather than definitive for any model with temporal features.

**Phase relevance:** Interpretability / SHAP analysis phase.

**Confidence:** MEDIUM -- SHAP limitations with correlated features are documented but the severity depends on the specific dataset.

---

### Pitfall 14: Null OHLC Fields in Kalshi Candlesticks Create Silent NaN Propagation

**What goes wrong:** Kalshi candlestick OHLC fields can be null when no trades occurred in a period (documented in CLAUDE.md). If you load this into a pandas DataFrame without explicit null handling, NaN values propagate silently through feature calculations (spread = Kalshi_price - Polymarket_price = NaN if either is null). Downstream models either crash or silently train on partial data.

**Prevention:**
1. After loading raw Kalshi data, immediately check for and count null OHLC values. Log the count.
2. Forward-fill (carry forward last known price) for short gaps (1-2 hours). For longer gaps, mark the entire period as "no data" rather than imputing.
3. Never drop rows silently. Always log how many rows were affected by null handling and what method was used.
4. Add a validation step: `assert df['spread'].notna().mean() > 0.9` (at least 90% of data is non-null after cleaning).

**Phase relevance:** Data pipeline phase, cleaning step.

**Confidence:** HIGH -- this is a project-specific gotcha explicitly documented in the project context.

---

### Pitfall 15: Confusing Prediction Performance with Trading Performance

**What goes wrong:** A model with the best RMSE on spread prediction may not be the best trading model. RMSE penalizes all errors equally, but trading cares about directional accuracy and magnitude of large moves. A model that accurately predicts small spread changes but misses large opportunities will look good on RMSE but perform poorly in simulated P&L.

**Prevention:**
1. Always report both regression metrics (RMSE, MAE) AND trading metrics (P&L, Sharpe, win rate, directional accuracy) for every model.
2. Include directional accuracy as a first-class metric: what fraction of the time does the model correctly predict whether the spread will widen or narrow?
3. In the paper discussion, explicitly address whether RMSE ranking agrees with P&L ranking. If they disagree, that is an interesting finding worth discussing.

**Phase relevance:** Evaluation phase.

**Confidence:** HIGH -- this disconnect is well-documented in quantitative finance literature.

---

### Pitfall 16: Ignoring Transaction Costs Inflates P&L Unrealistically

**What goes wrong:** Even though transaction costs are out of scope for detailed modeling, completely ignoring them when computing simulated P&L creates unrealistic results. Kalshi charges fees per contract, and Polymarket has gas costs on Polygon. If your average predicted spread is 3 percentage points and round-trip costs are 2 percentage points, your apparent 3pp profit becomes 1pp -- a 67% reduction.

**Prevention:**
1. Compute P&L both with and without a rough transaction cost estimate.
2. Use a simple flat cost per trade: 1% round-trip for Kalshi (fee structure), 0.5% for Polymarket (gas + taker fee). These are rough but better than zero.
3. Report the "break-even" transaction cost: "Our best model is profitable if round-trip costs are below X basis points."
4. In the paper, explicitly state that transaction costs are not modeled in detail but provide sensitivity analysis.

**Phase relevance:** Evaluation phase, profit simulation step.

**Confidence:** MEDIUM-HIGH -- the exact fee structures may have changed; the principle is solid.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation | Severity |
|-------------|---------------|------------|----------|
| Data pipeline (Kalshi) | Null OHLC fields, historical cutoff confusion | Validate nulls, call `/historical/cutoff` first | Moderate |
| Data pipeline (Polymarket) | Price reconstruction from trades, empty history endpoint | Validate against active markets, use VWAP | High |
| Data pipeline (time alignment) | Timezone/boundary misalignment creating phantom spreads | UTC normalization, visual validation | Critical |
| Market matching | False positive matches with different settlement criteria | Manual review of EVERY match, settlement criteria logging | Critical |
| Feature engineering | Autocorrelation masking as prediction, feature leakage | Predict changes not levels, include naive baselines | High |
| Dataset construction | Insufficient matched pairs for statistical significance | Run matching pipeline early, have scope contingency plan | Critical |
| Tier 1 models (regression) | Underinvesting in baselines (treating as afterthoughts) | Treat regression baselines as first-class; tune them properly | Moderate |
| Tier 2 models (time series) | TFT failing on small data, wasting time | Implement TFT last, time-box to 1 day | Moderate |
| Tier 3 models (RL) | Degenerate PPO policies, bad reward shaping | Monitor action distributions, design reward explicitly | High |
| Tier 3 models (autoencoder) | Arbitrary anomaly threshold, sensitivity | Report multiple thresholds, consider continuous input | Moderate |
| Evaluation | Sharpe on small samples, RMSE vs P&L disagreement | Bootstrap CIs, report both metric types | Moderate |
| Evaluation | Ignoring transaction costs | Rough cost estimates, break-even analysis | Low-Moderate |
| Interpretability | SHAP on correlated time-series features | Use TreeSHAP on XGBoost, attention for TFT | Low |
| Paper writing | Overclaiming from statistically weak results | Report CIs, frame as directional findings | Moderate |

---

## Academic-Specific Pitfalls

These pitfalls are specific to this being a course project with a deadline and TA review.

### Pitfall A: Spending Too Long on Data, Not Enough on Models

**What goes wrong:** The data pipeline (API ingestion, matching, time alignment, feature engineering) is the hardest engineering work but produces nothing visible to a TA or professor. If data work takes 2.5 of 4 weeks, you have 1.5 weeks for 7+ models, 3 experiments, SHAP analysis, and a paper.

**Prevention:** Hard deadline: data pipeline and matching must be complete by April 7 (3 days after TA check-in). If matching is incomplete by April 7, reduce scope (fewer model tiers, fewer ablations) rather than push data work later.

### Pitfall B: Not Having a Working Demo for TA Check-in (April 4)

**What goes wrong:** The TA check-in is in 3 days. If you show up with code that does not run, or a pipeline that is half-built, you lose credibility and potentially points.

**Prevention:** For April 4, have at minimum: (1) a working data pipeline that fetches data from both APIs, (2) a preliminary list of matched market pairs, (3) one baseline model (linear regression) producing a number on a small subset. This does not need to be polished or final.

### Pitfall C: Framing PPO Underperformance as Failure Instead of Finding

**What goes wrong:** PPO underperforms XGBoost (as expected), and the paper presents this as "our RL approach failed" rather than "our systematic comparison demonstrates that model complexity is not justified at this data scale." The former is a weak paper; the latter is the intended contribution.

**Prevention:** Write the paper's framing EARLY (outline by April 14). The narrative is: "We systematically tested whether increasing model complexity improves cross-platform prediction market arbitrage detection. Our results show [finding], suggesting that [insight]." PPO underperformance is the EXPECTED finding and supports the research question.

---

## Meta-Pitfall: Scope Creep Under Time Pressure

The project has many model tiers (7+ models), 3 experiments, SHAP analysis, and a paper due in 26 days. The biggest pitfall of all is trying to do everything at mediocre quality rather than prioritizing.

**Recommended priority if time is short:**
1. Data pipeline + matching (non-negotiable foundation)
2. Linear Regression + XGBoost + naive baselines (strong baselines)
3. Experiment 1: complexity comparison with whatever models are ready
4. Paper with honest findings
5. GRU (most likely to add value over XGBoost)
6. PPO variants (the RL exploration)
7. LSTM, TFT, SHAP, ablation experiments (nice to have)

If you can only do items 1-4 well, that is a better paper than items 1-7 done poorly.

---

## Sources

- Training data knowledge of financial ML best practices (de Prado, "Advances in Financial Machine Learning," 2018) -- HIGH confidence on backtesting pitfalls
- Training data knowledge of RL for trading (survey literature through 2024) -- HIGH confidence on degenerate policy pitfalls
- Project-specific API gotchas from CLAUDE.md and PROJECT.md -- HIGH confidence (project documentation)
- Prediction market mechanics from general domain knowledge -- MEDIUM-HIGH confidence
- Note: Web search was unavailable during this research session. All findings are based on training data and project documentation. Core pitfalls (look-ahead bias, survivorship bias, reward shaping) are extremely well-established in the literature and do not require web verification. API-specific pitfalls are based on the project's own documented findings.
