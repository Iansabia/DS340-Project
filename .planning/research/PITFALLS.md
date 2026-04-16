# Domain Pitfalls — v1.1 "Extended Evidence & Submission"

**Domain:** Cross-platform prediction-market arbitrage ML + live paper-trading system (academic project, DS340)
**Researched:** 2026-04-16
**Overall confidence:** HIGH on well-established categories (TFT overfitting, reproducibility, backtest-vs-live, LLM disclosure); MEDIUM-HIGH on project-specific integration pitfalls
**Relationship to v1.0:** This file EXTENDS, not replaces, the v1.0 pitfalls (look-ahead bias, survivorship, time alignment, matching false positives, RL degenerate policies, Sharpe on small samples, transaction costs). Those remain active. The v1.1 pitfalls below are *new failure modes* introduced by TFT, live-vs-backtest reconciliation, feature ablation, ensembling, paper credibility, reproducibility, and live-system safety.

---

## Category Map

| # | Category | Top Pitfalls | Guard Phase |
|---|----------|--------------|-------------|
| 1 | TFT on small data | Parameter bloat, attention collapse, val-loss disconnect | Phase 11 |
| 2 | Live vs backtest reconciliation | Fee-model mismatch, timestamp misalignment, resolved-mid-trade, position-sizing drift, mid-vs-last-trade prices | Phase 9 |
| 3 | Feature ablation | Correlated-feature double-attribution, ablation p-hacking, test-set contamination, seasonal masking | Phase 12 |
| 4 | Ensemble | Concordance-filter denominator trap, correlated base models, "best ensemble" selection bias, stale-member drag | Phase 13 |
| 5 | Paper credibility | Sharpe without per-pair independence, cherry-picked walk-forward window, scale-curve optimism at cap, survivorship silence, deflated-Sharpe omission | Phase 10, 14 |
| 6 | Reproducibility | Missing seeds in Tier-2 models, cached-result staleness, undocumented data-pipeline version, Python env drift, matplotlib backend | Phase 8, 14 |
| 7 | Live-system safety during v1.1 | Retrain-mid-experiment, schema changes breaking `positions.db`, features not in live bars, model artifact version skew | Phase 8, 13 |
| 8 | Academic integrity & LLM disclosure | ICLR-style disclosure requirement, plagiarism via uncited patterns, code-comment attribution, responsibility for fabricated claims | Phase 14 |
| 9 | Data leakage (general audit) | Scaler fit on full data, lag features created before split, target encoding leakage, across-pair leakage | All phases |

---

## Critical Pitfalls

Mistakes that sink the paper's credibility or break the live system. Each maps to a specific v1.1 phase (Phase 8–14 per the milestone plan).

---

### Pitfall 1: TFT Overfits Silently on 6,802 Rows

**Category:** TFT training
**Phase to address:** Phase 11 (TFT implementation)
**Severity:** HIGH

**What goes wrong:** PyTorch Forecasting's Temporal Fusion Transformer has roughly 10–100× more parameters than our GRU/LSTM (64 hidden units, 1 layer). On 6,802 training rows with ~47-bar sequences per pair, TFT has enough capacity to memorize the training set while generalizing poorly. The model reports low training MSE, the attention weights look "interpretable" to casual inspection, and the backtest P&L ends up within striking distance of XGBoost — but this is memorization wearing the costume of learning. When tested on any genuinely out-of-sample window (walk-forward window 11, new live bars), TFT's P&L collapses while XGBoost's holds.

**Why it happens:** TFT's variable-selection network and multi-head attention mechanisms were designed for large datasets (the original paper used 500k+ observations on electricity load forecasting). On small data, the variable-selection network will assign high weight to one or two features by chance in training and never re-explore, effectively becoming a nonlinear single-feature regressor with extra steps. Academic red flag: if your TFT attention weights concentrate >80% on one variable across all time steps, the model has collapsed to a degenerate solution.

**Consequences:** The paper reports "TFT achieves +$195 P&L" and the TA/reviewer asks "can this be reproduced on window-12 data?" and it can't. The deferred Tier-2 finding becomes embarrassing rather than impressive. Worse, if TFT accidentally *wins* the single-split by memorization, you may publish a claim that contradicts your own walk-forward evidence.

**Warning signs (check during training):**
- Training loss drops below 0.5× GRU's training loss but validation loss is flat or worse than GRU.
- Attention weight concentration: if `entropy(attention_weights) < 0.5 × log(n_features)` across >90% of time steps, the network has collapsed.
- Variable-selection weights are effectively one-hot (one feature gets 0.95+ weight).
- Validation RMSE plateaus above GRU's within 15 epochs and never catches up.
- Training takes more than 30× longer than GRU and still under-performs it.
- Walk-forward window-11 TFT P&L is negative while windows 1–10 are positive (pure overfitting fingerprint).

**Prevention strategy (concrete):**
1. **Parameter budget:** Cap TFT hidden_size at 16, attention_heads at 1, and hidden_continuous_size at 8. Document that this is smaller than the PyTorch Forecasting defaults because of dataset size. In the paper: "Hyperparameters were downsized relative to defaults to match our training-set size."
2. **Dropout floor:** Set `dropout ≥ 0.3` and `attention_dropout ≥ 0.2`. Do not reduce these for "cleaner" loss curves.
3. **Early-stopping on walk-forward, not single-split:** Use window-level walk-forward validation P&L as the stopping criterion, not in-sample loss. If walk-forward median P&L stops improving for 3 epochs, stop.
4. **Time-box:** Hard limit of 1 day on TFT. If TFT has not cleared GRU by then, report "TFT did not converge on this dataset size (6,802 rows, 47 bars/pair)" as the finding. This is a valid academic result — it directly supports the simplicity-wins thesis.
5. **Attention-weight audit:** Before reporting any TFT result, plot the attention heatmap across windows. If attention is concentrated on one variable, note this in the results section.
6. **Gate on live-compatible features only:** TFT must be trained on the same 51-column feature set the live system uses — no extra features that exist in historical bars but not live bars (see Pitfall 21).

**Detection code pattern:**
```python
# Run after each epoch
attn_entropy = -(attention_weights * attention_weights.log()).sum(dim=-1).mean()
max_feature_weight = variable_selection_weights.max(dim=-1).values.mean()
if attn_entropy < 0.5 * np.log(n_features) or max_feature_weight > 0.8:
    logger.warning(f"TFT attention collapse detected: entropy={attn_entropy:.3f}, max_vw={max_feature_weight:.3f}")
```

**Confidence:** HIGH — TFT overfitting on small data is documented in the original Lim et al. (2020) paper (Sec 5.3), in Darts/PyTorch Forecasting issues, and corroborated by our own Finding 2 (GRU already underperforms XGBoost at this scale; TFT is strictly more parameterized).

**Sources:** [Temporal Fusion Transformers for interpretable multi-horizon time series forecasting (Lim et al., ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S0169207021000637), [TFT darts documentation](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tft_model.html), [PyTorch Forecasting Stallion tutorial](https://pytorch-forecasting.readthedocs.io/en/v1.4.0/tutorials/stallion.html)

---

### Pitfall 2: Live-vs-Backtest P&L Gap Is Treated as Noise Instead of Bias

**Category:** Live vs backtest reconciliation
**Phase to address:** Phase 9 (Live-vs-backtest reconciliation)
**Severity:** CRITICAL for paper credibility

**What goes wrong:** The live paper-trading system on SCC has been accumulating trades since April 9. The backtest reports +$208 P&L on 1,673 test rows. When you compare live realized P&L to backtest-predicted P&L on the same pairs over the same days, they disagree by more than 20%. The default instinct is to shrug ("small sample, noisy"). But a systematic 20%+ gap is almost always a *bug*, not noise. Common culprits (all of which we are already exposed to):

1. **Fee-model mismatch:** backtest uses flat 2pp; live applies 2pp but some live trades route through different fee paths depending on maker/taker status — backtest doesn't model that.
2. **Timestamp alignment drift:** live uses `datetime.utcnow()` at decision time; backtest uses the bar's close timestamp. A 7-minute clock skew on a 15-minute cycle is a 47% boundary error.
3. **Look-ahead in "simulated" predictions:** the live system computes features from bar[t] and trades against price at bar[t]; the backtest computes features from bar[t] and trades against price at bar[t] — identical, but the live "price at bar[t]" is the *last trade*, while backtest uses the *bar close* (VWAP). Systematic bias in one direction.
4. **Position-sizing drift:** backtest assumes $1 per trade; live caps at $10 per pair + $200 cash floor, so live cannot open some positions backtest would have opened.
5. **Resolved-mid-trade contracts:** the live system caught a contract that resolved during the 15-minute cycle and the exit code booked the resolution price; the backtest ignored that mechanism because historical bars don't include intra-bar resolution.
6. **Different pair universe:** backtest trades on 144 filtered pairs; live trades on ~1,000 pairs (after discovery and quality filter). The 144-pair subset is a survivorship-biased slice of the live universe.

Industry evidence: ~90% of academic trading strategies fail when deployed live; a significant fraction of that failure is reconciliation bugs, not model decay.

**Why it happens:** The reconciliation exercise is usually skipped because it's tedious and reveals uncomfortable gaps. Authors and engineers prefer "backtest looked good, live looks good, shipping" over "let's quantify the delta exhaustively." The v1.0 paper draft currently does not reconcile these numbers rigorously — §5 reports backtest P&L and §4.4 describes the live system, but the two are not put side-by-side trade-for-trade.

**Consequences:** The paper's strongest headline claim — "the system works in production" — is unverifiable. A reviewer asks "live P&L was $X, backtest P&L was $Y on the same trades, why do these differ by 20%?" and you have no answer. This kills credibility.

**Warning signs:**
- Live hit rate deviates from backtest hit rate by >5 percentage points on n>200 trades.
- Per-trade P&L distribution is visibly different (e.g., live has a fatter left tail).
- Trade count on the same pair-day differs between live and backtest by >10%.
- Live trades show categories the backtest pair universe doesn't contain.

**Prevention strategy (concrete):**
1. **Build a reconciliation notebook** (Phase 9 deliverable): for each live trade, look up the corresponding backtest trade on the same (pair, timestamp) and compute the delta. Report median absolute delta, worst-case delta, and a histogram.
2. **Run the backtest simulator on live pair universe:** rerun the XGBoost backtest against the same ~1,000 pairs the live system monitors, not just the 144 filtered pairs. Report both numbers in the paper.
3. **Fee-model alignment:** have live and backtest use *exactly the same* fee function, imported from a single module. Write a unit test: `assert backtest_fee(trade) == live_fee(trade)` on a representative sample.
4. **Timestamp discipline:** all timestamps in all systems are UTC, stored as Unix seconds or pandas Timestamp with explicit tz. Add an assertion at system boundaries.
5. **Price-source alignment:** both backtest and live must use the same price representation. If backtest uses VWAP and live uses mid, one must migrate to the other. Document which was chosen and why.
6. **Resolution-event handling:** create a shared `is_resolved_at(contract, timestamp)` function. Both systems must use it. Backtest must simulate intra-bar resolution if the live system handles it.
7. **Report the gap in the paper:** §5.X "Live vs. backtest reconciliation". Numbers like "live P&L was 108% of backtest P&L on the same 87 overlap trades, within the 95% bootstrap CI of [0.92, 1.24]." This *increases* credibility — readers see you did the work.

**Detection code pattern:**
```python
# Join live trades to backtest trades on (pair_id, entry_ts_bucket)
merged = live_trades.merge(backtest_trades, on=['pair_id', 'entry_bucket'],
                           suffixes=('_live', '_bt'), how='outer', indicator=True)
only_live = (merged['_merge'] == 'left_only').sum()
only_bt = (merged['_merge'] == 'right_only').sum()
both = (merged['_merge'] == 'both').sum()
logger.info(f"Live only: {only_live}, Backtest only: {only_bt}, Both: {both}")
assert only_live + only_bt < 0.2 * both, f"Reconciliation gap too large: {only_live + only_bt}/{both}"
```

**Confidence:** HIGH — widely documented in quant finance literature, and we have direct exposure (already know our live pair universe differs from our backtest universe).

**Sources:** [QuantConnect Reconciliation Documentation](https://www.quantconnect.com/docs/v2/cloud-platform/live-trading/reconciliation), [Backtesting Limitations: Slippage and Liquidity — LuxAlgo](https://www.luxalgo.com/blog/backtesting-limitations-slippage-and-liquidity-explained/), [Backtesting AI Crypto Trading Strategies Safely](https://www.blockchain-council.org/cryptocurrency/backtesting-ai-crypto-trading-strategies-avoiding-overfitting-lookahead-bias-data-leakage/)

---

### Pitfall 3: Feature Ablation Becomes P-Hacking via Selection

**Category:** Feature ablation
**Phase to address:** Phase 12 (Feature ablation)
**Severity:** HIGH (paper-credibility risk)

**What goes wrong:** You ablate the 59-feature set to find the "minimum viable feature set." The natural approach is to remove features one-at-a-time, measure test-set P&L, and keep removing the feature whose removal hurts least. After 30+ ablation runs, you report "we can drop 40 features and keep 95% of P&L." This is **p-hacking by ablation**:

1. **Test-set contamination:** every ablation run uses the *same* test set. If you run 30 ablations, some combinations will look good on this test set by chance alone. The reported "minimum feature set" is optimized for THIS test set, not for unseen data.
2. **Correlated-feature double-attribution:** `spread`, `spread_ma_6`, `spread_ma_12`, `spread_zscore` all carry nearly identical information. Dropping `spread_zscore` causes zero P&L change because `spread_ma_6` absorbs the signal — not because `spread_zscore` is useless. Reporting "z-score is unnecessary" is a correlation error, not a feature-importance result.
3. **Seasonal masking:** if most of the test set is from a period where oil contracts dominate, ablating `near_expiry_indicator` won't hurt. But in a different regime, that feature is load-bearing. Single-split ablation masks regime dependence.
4. **Ablation direction bias:** removing features sequentially based on "least harm on test set" is greedy and finds a *local* minimum, not the true minimum feature set. Restarting with different drop orders can give wildly different "minimum sets."
5. **Tree-based importance vs. linear importance disagreement:** XGBoost SHAP and LR coefficient importance disagree on which features matter. Reporting "XGBoost says dollar_volume_ratio is #3 important, LR says it's #12 important, so we drop it" hides the disagreement.

**Why it happens:** Ablation feels rigorous and quantitative, but ablation without a held-out-from-ablation test set is textbook multiple-testing bias. Most ML papers don't distinguish "validation" and "ablation holdout" sets cleanly.

**Consequences:** The paper claims "we found a parsimonious 15-feature set achieving 98% of full-feature P&L." A reviewer reruns the ablation on a different test slice and finds the claim doesn't hold. Or worse — a later live-trading regime changes and the 15-feature model falls apart because load-bearing features were dropped based on regime-specific ablation.

**Warning signs:**
- Ablation results swing >20% in reported P&L when you change random seed or test-window cutoff.
- "Dropping feature X has zero effect" while "dropping feature Y has 30% effect" but X and Y are correlated >0.7.
- Your "minimum feature set" differs by >30% of features when you change ablation direction (forward vs. backward selection).
- Per-category P&L on the minimum set changes the category winners from the full set (suggests regime dependence).

**Prevention strategy (concrete):**
1. **Three-way split:** train / ablation-holdout / final-test. Use train for fitting each ablation run, ablation-holdout to *select* the minimum set, final-test to *report*. Final-test is touched exactly once.
2. **Correlation-aware grouping:** before ablation, cluster features by correlation at |r| > 0.85 and ablate *groups*, not individual features. Report "we can drop this correlated cluster" rather than "we can drop feature X."
3. **Permutation importance instead of drop-one:** use `sklearn.inspection.permutation_importance` with 10+ shuffles per feature. This gives a distribution, not a point estimate, and p-values for significance.
4. **Cross-validated ablation:** run each ablation through walk-forward validation (11 windows), not single-split. Report median ± IQR of the feature's importance across windows.
5. **Report all ablations, not just winners:** include every ablation run with full metrics in the supplementary material. Reviewers can check.
6. **Pre-registration:** write down the ablation protocol *before* running it (which features, which order, which success metric). Publish the pre-registered protocol alongside results. This is the strongest defense against p-hacking.
7. **Seasonal robustness check:** run the minimum-feature ablation separately on each category (oil vs. inflation vs. crypto). Report "minimum feature set varies by category" as a finding if it does.

**Detection code pattern:**
```python
# Check correlation-aware grouping before ablation
from scipy.cluster.hierarchy import fcluster, linkage
corr = df[features].corr().abs()
dist = 1 - corr
clusters = fcluster(linkage(squareform(dist), method='average'), t=0.15, criterion='distance')
# Features with same cluster_id are correlated >0.85 and should ablate together
```

**Confidence:** HIGH — this is the standard multiple-testing / p-hacking critique applied to ablation. Deflated Sharpe and CSCV literature (Bailey et al.) directly applies.

**Sources:** [Statistical Overfitting and Backtest Performance (Bailey et al., SSRN)](https://sdm.lbl.gov/oapapers/ssrn-id2507040-bailey.pdf), [Understanding Ablation Studies — Oreate AI](https://www.oreateai.com/blog/understanding-ablation-studies-a-key-tool-in-machine-learning/81214cd949a32c1df0d6172fda39efaf), [ABGEN: Evaluating LLMs in Ablation (ACL 2025)](https://aclanthology.org/2025.acl-long.611.pdf)

---

### Pitfall 4: Concordance Filter Shrinks the Denominator and Inflates Sharpe

**Category:** Ensemble
**Phase to address:** Phase 13 (Ensemble)
**Severity:** CRITICAL for paper credibility (we're already using this filter)

**What goes wrong:** The live system uses a "concordance filter" — skip the trade if LR and XGBoost disagree on sign. This is sensible risk management, but it has a nasty statistical side-effect: it systematically *removes the hardest trades*, keeping only the ones where two weakly-correlated models agree. The surviving subset has artificially high hit rate and Sharpe. When you report "ensemble achieved 51% win rate, per-trade Sharpe 0.44," that's on a subset of trades pre-filtered to be easy.

Worse, the concordance filter's selection criterion is computed using the same models being evaluated. This creates a subtle form of look-ahead: the filter "knows" which trades would be disagreements, i.e. harder, i.e. lower-expected-value. Filtering them out makes the *remaining* expected value look higher than it would be for a randomly-sampled trade.

If we also add a third model (GRU, LSTM) to the concordance filter, the denominator shrinks further and the remaining trades are even more pre-selected. This can create a misleading story: "our 4-model ensemble has Sharpe 1.2!" when really you're reporting performance on 20% of trades pre-filtered to be obvious.

**Why it happens:** The concordance filter looks like a "safety feature" — only trade when models agree. In reality it's a sample-selection bias. The reporter doesn't realize the filter is doing the same thing as "only show me trades where the models would have been right anyway, so the story looks better."

**Consequences:** Paper reports inflated Sharpe on a filtered subset. Reviewer asks "what's your Sharpe on the *unfiltered* trade stream — i.e., if a practitioner didn't have your ensemble?" and the number drops 30–50%.

**Warning signs:**
- Concordance filter rejects >40% of potential trades.
- Filtered subset has hit rate >20 percentage points higher than unfiltered.
- As you add more models to the concordance vote, Sharpe monotonically increases and trade count monotonically decreases.
- The filter rejects more trades in one category than another (implicit category-selection bias).

**Prevention strategy (concrete):**
1. **Always report both numbers.** For every ensemble result, report:
   - "Ensemble with concordance filter: $X P&L, N trades, Sharpe S."
   - "Base model without filter: $Y P&L, M trades, Sharpe T."
   - "Filter rejected K trades; their mean P&L was Z."
2. **Simulate the rejected trades:** compute P&L on the rejected trades using the base model's recommendation. If those trades were profitable on their own, the filter is *hurting* real P&L while inflating paper Sharpe. This is a red flag.
3. **Report filter-level details in paper.** Never say "our ensemble Sharpe is 0.6" without specifying the filter threshold.
4. **Deflated Sharpe:** compute Sharpe that corrects for the multiple-testing nature of trying several filter thresholds. Use the deflated Sharpe formula from Bailey & de Prado (2014).
5. **Benchmark against the "accept all trades" base case:** always have a line in the results table for "no filter" as the reference.
6. **Cross-category robustness:** report filtered performance per category. If the filter only works in one category, say so.

**Detection code pattern:**
```python
def concordance_audit(predictions_dict, actual_delta_spread, fees=0.02):
    # predictions_dict: {'lr': ndarray, 'xgb': ndarray, 'gru': ndarray, ...}
    signs = {k: np.sign(v) for k, v in predictions_dict.items()}
    all_agree = np.all(np.stack(list(signs.values())) == signs['lr'], axis=0)
    accepted_pnl = np.where(all_agree, predictions_dict['lr'] * actual_delta_spread - fees, 0).sum()
    rejected_pnl = np.where(~all_agree, predictions_dict['lr'] * actual_delta_spread - fees, 0).sum()
    print(f"Accepted: {all_agree.sum()} trades, P&L ${accepted_pnl:.2f}")
    print(f"Rejected: {(~all_agree).sum()} trades, hypothetical P&L ${rejected_pnl:.2f}")
    if rejected_pnl > 0:
        print("WARNING: Filter rejecting profitable trades — may be inflating Sharpe via subset bias")
```

**Confidence:** HIGH — this is the Bailey & de Prado "deflated Sharpe" / "probability of backtest overfitting" framework applied to our specific filter.

**Sources:** [The Deflated Sharpe Ratio (Bailey & Lopez de Prado)](https://www.researchgate.net/publication/286121118_The_Deflated_Sharpe_Ratio_Correcting_for_Selection_Bias_Backtest_Overfitting_and_Non-Normality), [Backtest overfitting in the ML era (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110), [A Unified Theory of Diversity in Ensemble Learning (JMLR)](https://jmlr.org/papers/volume24/23-0041/23-0041.pdf)

---

### Pitfall 5: Reporting Per-Trade Sharpe of 0.44 Without Per-Pair Independence Disclaimer

**Category:** Paper credibility
**Phase to address:** Phase 10 (Paper finalization), Phase 14 (Submission)
**Severity:** CRITICAL — this is the single biggest paper-credibility risk

**What goes wrong:** You report "XGBoost achieves per-trade Sharpe of 0.44" and then annualize naively to get a headline Sharpe of 40+. Any reviewer with quant-finance familiarity will immediately notice this is absurd — hedge funds celebrate Sharpe 2, elite funds achieve 3, and anything above 4 is a red flag for overfitting, survivorship, or unit-of-independence errors. Your paper looks either naïve or dishonest.

The correct statistical unit of independence here is **per-pair**, not per-trade. Trades on the same pair within 15-minute windows are heavily auto-correlated. Our historical analysis (Finding 17) showed:
- Per-trade Sharpe: 0.59
- Daily Sharpe × √252: 53 (absurd)
- **Per-pair annualized: 4.28** (correct)

Even 4.28 is likely inflated by the 2-week test window. An honest headline number is probably 2.5–3.5 after correcting for window length and applying realistic slippage.

**Why it happens:** Beginner-to-intermediate quant practitioners default to per-trade Sharpe because it's what scikit-learn and `backtrader` report. The annualization step uses √252 (trading days) without thinking about whether trades are independent. If you have 90 trades/day on correlated pairs, treating them as 90 independent returns is double-counting by an order of magnitude.

**Consequences:** If we publish the headline Sharpe without per-pair correction:
- TA/reviewer flags it as "implausibly high, did you check this?"
- Paper loses credibility on every *other* claim, because one obviously-wrong number casts doubt on all numbers.
- Worst case: TA concludes the project has a deeper methodological error (data leakage) and grades accordingly.

The v1.0 paper draft already addresses this in §5.8 ("Honest Sharpe-Ratio Accounting") — good. **The risk is that v1.1 experiments (TFT, ensemble, ablation) re-introduce this mistake by default** because the evaluation scripts report per-trade Sharpe and someone will copy-paste the headline number into the v1.1 extensions.

**Warning signs:**
- Any headline Sharpe number above 4.0 in the paper without explicit per-pair correction.
- Reported annualized Sharpe computed as `per_trade_sharpe × sqrt(252)` (obvious error).
- Sharpe computed without block-bootstrap confidence intervals.
- No disclaimer about test-window length.

**Prevention strategy (concrete):**
1. **Single source of truth for Sharpe.** Create `src/evaluation/sharpe.py` with four named functions:
   - `per_trade_sharpe(returns)` — documented as "over-estimate, includes correlation"
   - `per_pair_sharpe(returns, pair_ids)` — correct, aggregates within pair first
   - `annualized_sharpe(daily_returns)` — correct if returns really are daily
   - `bootstrap_sharpe_ci(returns, method='per_pair', n=10000)` — returns (sharpe, lower, upper)
2. **Paper rule:** every Sharpe reported in the paper MUST use one of these four functions and cite which. No ad-hoc Sharpe computations.
3. **Headline number is per-pair with CI.** For every model in the results table, the headline Sharpe is per-pair annualized with 95% bootstrap CI. Per-trade Sharpe is secondary, in parentheses, with a footnote explaining why it's higher.
4. **Mandatory disclaimer paragraph in §5.8 (already in v1.0 draft).** Keep this and extend for each new model (TFT, ensemble).
5. **Include a "what would be suspicious" paragraph.** "Sharpe above 4.0 in quantitative finance typically indicates overfitting, survivorship bias, or unit-of-independence errors. Our per-pair Sharpe of 3.2 sits just below this threshold and is likely inflated by the 2-week test window; a longer out-of-sample period would likely contract it toward 2.0–2.5." This signals statistical sophistication.

**Confidence:** HIGH — this is *the* most-commonly-flagged issue in quant papers reviewed by practitioners. de Prado and Bailey have written extensively on it.

**Sources:** [The Deflated Sharpe Ratio (Bailey & Lopez de Prado)](https://www.researchgate.net/publication/286121118_The_Deflated_Sharpe_Ratio_Correcting_for_Selection_Bias_Backtest_Overfitting_and_Non-Normality), [Reproducibility in machine-learning-based research (AI Magazine 2025)](https://onlinelibrary.wiley.com/doi/10.1002/aaai.70002)

---

### Pitfall 6: Cherry-Picked Walk-Forward Window for Headline Numbers

**Category:** Paper credibility
**Phase to address:** Phase 10 (Paper finalization)
**Severity:** HIGH

**What goes wrong:** The v1.0 walk-forward has 11 windows. The median per-trade Sharpe across windows is 0.424 (XGBoost). Window 10 is 0.540 — the best window. If you're tempted to write "our latest walk-forward window achieves Sharpe 0.54" in the abstract, that's cherry-picking. The reader thinks this is the paper's representative performance, but it's the best window out of 11.

Window 11 is 0.35 — the worst among meaningful windows. If we omit it from the reported range (because "it only has 110 test rows, too small"), that's also a cherry-pick: we should either include all windows or define the inclusion rule *before* looking at results.

**Why it happens:** Under deadline pressure, authors write the abstract late in the process, after having seen all the results, and naturally gravitate toward the best numbers. The decision "let's lead with window 10" feels like picking the most recent data, but it's also picking the highest-Sharpe window.

**Consequences:** Abstract says "our system achieves Sharpe 0.54 on the most recent walk-forward window" → reader thinks this is representative → reader computes their own walk-forward on a different window and gets 0.30–0.40 → reader concludes the paper is over-sold.

**Warning signs:**
- Abstract or conclusion cites a single window's result without context.
- Reported Sharpe in abstract differs from median Sharpe in Table 3b by >15%.
- Window selection criterion was not pre-specified.
- The "reason" for excluding a window was decided after seeing that window's results.

**Prevention strategy (concrete):**
1. **Pre-register window inclusion criteria.** Before writing §5.2, decide: "Windows with test count ≥ 200 are reported; window 11 (n=110) is excluded from summary statistics but shown in the full table." Document this BEFORE running the analysis and include the reason.
2. **Abstract reports central tendency + range.** "Per-trade Sharpe ranges from 0.31 to 0.54 across 11 windows, median 0.42."
3. **Always report median and mean.** If they disagree by >0.05, investigate (likely a window outlier).
4. **Show all windows.** Don't hide windows in appendix. Table 3 and 3b should include every window with full metrics.
5. **Per-category walk-forward for robustness.** Run walk-forward on each category separately. If oil's Sharpe is 0.8 and politics' is 0.1, that's a finding — disclose it rather than pooling.

**Confidence:** HIGH — textbook cherry-picking bias, standard in academic-integrity training for quant papers.

**Sources:** [Interpretable Hypothesis-Driven Trading: A Rigorous Walk-Forward Validation Framework (arXiv 2025)](https://arxiv.org/html/2512.12924v1), [Statistical Overfitting and Backtest Performance (Bailey et al.)](https://sdm.lbl.gov/oapapers/ssrn-id2507040-bailey.pdf)

---

### Pitfall 7: Data-Scaling Curve Plateau Hides Real Scaling Behavior

**Category:** Paper credibility
**Phase to address:** Phase 10 (Paper finalization)
**Severity:** MEDIUM-HIGH

**What goes wrong:** Your scaling curve (Table 5 in the current draft) shows XGBoost at +$210 for all bars/pair ≥ 100. You explain that the training-set is capped at 6,802 rows, so the scaling curve is flat beyond bar count 100. Fine — but then your paper claim "simpler models win at all scales tested" is only tested up to 6,802 training rows. Readers will naturally extrapolate: "at 100,000 rows, TFT would still lose." That extrapolation is unsupported.

Your 250-bar checkpoint is pending. When it fires, if the training set expands to, say, 12,000 rows, XGBoost might still dominate — OR GRU/LSTM might close the gap. Reporting the current scaling curve without noting "training-set cap reached" leaves the reader with a false impression.

**Why it happens:** Once the training-set cap is hit, additional bars/pair don't add training data (they add test data). The effect looks like a plateau, but it's actually a measurement artifact of a fixed pair universe. Without explaining this, the curve looks like proof of universal simplicity-wins.

**Consequences:** Paper's scale-invariance claim is fragile. When live data accumulates and the training set grows, the 2026 paper's conclusions may invert. This is fine if disclosed — expected if not.

**Warning signs:**
- Scaling curve plateau at exactly the training-set cap.
- No x-axis label indicating "training rows" vs. "bars per pair" (these are different).
- No annotation on the curve where the cap is hit.

**Prevention strategy (concrete):**
1. **Two-axis scaling plot.** X-axis top: "bars/pair." X-axis bottom: "training rows (capped at N)." Readers see the distinction.
2. **Annotation marking the cap:** "Dashed line at 100 bars/pair = training-set cap (6,802 rows). Beyond this, the plateau is an artifact."
3. **Explicitly disclose the limit in text.** "Our scaling experiment tests up to 6,802 training rows due to the fixed pair universe at the time of writing. The live system is accumulating data; if the training set grows beyond this, sequence models may close the gap with XGBoost."
4. **Frame the finding honestly.** "Within the range we can test (50–6,802 rows), XGBoost dominates. Whether this holds at 10× larger training sets is an open empirical question we cannot answer with the current dataset."

**Confidence:** HIGH — standard scaling-analysis critique.

---

### Pitfall 8: Survivorship Bias in the 144-Pair Universe

**Category:** Paper credibility
**Phase to address:** Phase 10 (Paper finalization)
**Severity:** HIGH

**What goes wrong:** The paper reports results on 144 matched pairs. These 144 pairs survived:
1. Kalshi + Polymarket API discovery (misses delisted markets).
2. Sentence-transformer semantic matching (misses non-obvious synonymy).
3. 10-rule quality filter (excludes 22.8% of naive matches).
4. Loosened-quality-filter improvements that only added pairs meeting new criteria.
5. Live tombstone TTL of 7 days (pairs that disappear and don't return are dropped).

Every filter is a survivorship event. The 144 pairs are the *survivors* of five selection stages. Their reported profitability is conditional on surviving all five. If a practitioner (or reviewer replicating) were to start fresh without these filters, results would differ.

This is already documented in v1.0 Pitfall 2 and §5/Limitations of the paper draft — partially. The v1.1 risk is that *as we add new experiments* (ablation, ensemble, TFT), each run uses the same 144-pair universe and inherits the same survivorship bias. The paper needs to state this bias *once* explicitly and note that all experiments share it.

**Why it happens:** It's impossible to analyze markets you don't have data for. The temptation is to treat the surviving universe as representative, especially when the filter logic has good per-filter justification.

**Consequences:** Paper's external validity claim is weakened. A reviewer asks "what's the performance on markets that failed the quality filter? How do we know you haven't filtered out all the losing trades?" and you must have an answer.

**Warning signs:**
- Paper abstract doesn't mention the 22.8% quality-filter rejection rate.
- "Limitations" paragraph in paper doesn't name survivorship bias explicitly.
- No sensitivity analysis showing how results change under looser filters.

**Prevention strategy (concrete):**
1. **Mandatory Limitations paragraph** naming survivorship bias, the 22.8% rejection rate, the 7-day tombstone TTL, and the 5-stage selection pipeline. One paragraph, §6.
2. **Run one "no quality filter" experiment** as a sensitivity analysis. Report the P&L difference. Expected: P&L decreases (quality filter was doing work), but by how much? If P&L goes to zero without the filter, that's important to disclose.
3. **Track rejected pairs' counterfactual P&L.** For pairs that were quality-filter-rejected, hypothetically simulate what would have happened if we'd traded them. Report the delta.
4. **Document the matching-funnel funnel-chart.** "Kalshi markets: 20,000. Polymarket markets: 30,000. Semantic matches: X. Post-quality-filter: Y. Post-live-tombstone: 144." Reader sees survivorship visually.
5. **Include an "unresolved markets" section.** How many of the 144 pairs resolved in our test window vs. are still open? Open pairs can't contribute P&L, so reported P&L is on the *resolved subset*.

**Confidence:** HIGH — classical survivorship bias, well-understood.

---

### Pitfall 9: Stochastic Training Without Seeds Kills Reproducibility

**Category:** Reproducibility
**Phase to address:** Phase 8 (Priority-1 cleanups), Phase 14 (Submission)
**Severity:** CRITICAL for submission credibility

**What goes wrong:** GRU, LSTM, TFT, PPO all use PyTorch, which is stochastic by default: random weight initialization, shuffled DataLoaders, dropout masks, CUDA non-determinism. If you don't explicitly seed all of these, running the same script twice gives different numbers. Even with perfect seeding, results may not be reproducible across PyTorch versions or between CPU/GPU.

Specific failures we're exposed to:
- `torch.manual_seed(42)` set but `numpy.random.seed()` not set — numpy operations in feature engineering produce different results.
- DataLoader seeded but `shuffle=True` without `generator` argument — still non-deterministic.
- CUDNN backends non-deterministic even when seeds are set (need `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`).
- Multi-worker DataLoaders have independent RNGs per worker that need per-worker seeding.
- `stable-baselines3` PPO has its own internal RNG that may not respect your global seed.

Consequences: we report "LSTM P&L +$182.72" but a reproducer gets +$175 or +$190. The Table 2 numbers differ from a rerun by 3–5%. The TA/reviewer tries to reproduce and fails. Paper credibility tanks.

**Why it happens:** Default PyTorch is stochastic. Reproducibility requires *every* RNG to be seeded, *and* non-deterministic kernels to be disabled. Most tutorials show `torch.manual_seed(42)` and imply that's enough — it isn't.

**Consequences:** Irreproducible numbers in the paper. Follow-up work can't extend ours because they can't reproduce the baseline. In a submission context (DS340 final), this is a credibility issue.

**Warning signs:**
- Running the training script twice in succession gives different test P&L.
- `torch.backends.cudnn.benchmark` is `True` (default).
- No `random_state` set in scikit-learn train_test_split calls.
- No seed for `numpy.random`, Python's `random` module.
- Results change when you change the number of DataLoader workers.

**Prevention strategy (concrete):**
1. **Create `src/utils/seed.py`** with a single function `set_all_seeds(seed=42)` that seeds numpy, torch, torch.cuda, python random, and sets cudnn deterministic flags. Call it at the top of every training script.
2. **Multi-seed reporting for Tier 2 & 3 models.** Run LSTM/GRU/TFT/PPO with seeds {42, 43, 44, 45, 46}. Report median ± IQR P&L, not single-seed P&L. This is the gold standard for stochastic ML reporting.
3. **Deterministic DataLoader pattern:**
   ```python
   g = torch.Generator()
   g.manual_seed(seed)
   DataLoader(dataset, shuffle=True, generator=g, num_workers=0)  # num_workers=0 for determinism
   ```
4. **Document environment:** freeze `requirements.txt` with exact versions. Include Python, PyTorch, XGBoost, scikit-learn versions in §4 of the paper.
5. **Reproducibility section in appendix:** "All experiments run with seed 42; Tier-2/3 models additionally tested with seeds {43-46} for variance. Environment: Python 3.12.3, PyTorch 2.3.0, XGBoost 2.0.3, on Linux 5.15 (SCC scc1.bu.edu) and macOS 14 (Apple M1)."
6. **Reproducibility smoke test:** commit a script `scripts/reproducibility_check.sh` that runs the pipeline twice and asserts outputs match bit-for-bit.

**Detection code pattern:**
```python
def set_all_seeds(seed=42):
    import random, numpy as np, torch, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
```

**Confidence:** HIGH — this is the canonical ML reproducibility issue, documented in PyTorch's own docs.

**Sources:** [Reproducibility — PyTorch documentation](https://docs.pytorch.org/docs/stable/notes/randomness.html), [The Challenge of Reproducible ML (arXiv)](https://arxiv.org/pdf/2109.03991), [Ensuring Training Reproducibility in PyTorch (LearnOpenCV)](https://learnopencv.com/ensuring-training-reproducibility-in-pytorch/), [Reproducibility in ML-based research (AI Magazine 2025)](https://onlinelibrary.wiley.com/doi/10.1002/aaai.70002)

---

### Pitfall 10: Cached Result Files Ship With Stale Numbers (We've Already Hit This)

**Category:** Reproducibility
**Phase to address:** Phase 8 (Priority-1 cleanups), Phase 14 (Submission)
**Severity:** CRITICAL (already happened once — v1 paper had wrong XGBoost numbers)

**What goes wrong:** `experiments/results/tier1/xgboost.json` contains cached P&L from a previous run. When the paper script is regenerated, it reads this cached JSON. But the cache is stale: the most recent 144-pair dataset expansion produced $201 XGBoost P&L, while the JSON still says $210 from a previous 120-pair run. Paper ships with $210; reader reruns; gets $201; wonders what's wrong. Our v1.0 paper had *exactly this bug* — the git status above even shows `M experiments/results/tier1/xgboost.json` as modified.

**Why it happens:** Result caching is a natural optimization — avoid rerunning expensive experiments. But caches must be invalidated when inputs change. Our experiment scripts don't check input-hash before using cached outputs, so stale caches persist across data-pipeline updates.

**Consequences:** Paper headline numbers disagree with code output. Reviewer runs the code and finds discrepancies. Worse, we may commit a paper version that disagrees with our own git history.

**Warning signs:**
- `experiments/results/**/*.json` files have timestamps older than the most recent `data/processed/train.parquet`.
- `git status` shows modified result files that weren't intended to change.
- Paper numbers differ from the last `experiments/results/*.json`.

**Prevention strategy (concrete):**
1. **Input-hash in result files:** every experiment result JSON includes a `data_hash` field (SHA-256 of the input parquet file). Before using a cached result, compare current data hash to stored hash. Mismatch → rerun.
   ```python
   import hashlib
   data_hash = hashlib.sha256(open('data/processed/train.parquet', 'rb').read()).hexdigest()[:16]
   if cached_result.get('data_hash') != data_hash:
       result = rerun_experiment()
       result['data_hash'] = data_hash
       save(result)
   ```
2. **Pre-submission freshness audit:** one day before paper submission, delete all cached results and rerun end-to-end. Commit the new results. Confirm numbers match paper.
3. **Single-command reproduction:** `make reproduce` must regenerate every table and figure from raw data with no cache reliance. Include this in the README.
4. **Git hook:** pre-commit hook that warns if result files are modified but data hash hasn't been updated in their metadata.
5. **Single source of truth for numbers:** paper numbers come from a single JSON file `paper_numbers.json` that is regenerated by one script from the fresh result files. No manual edits.
6. **Timestamped result logging:** every result file has a `computed_at` ISO timestamp. The paper build script checks that all referenced result files have timestamps within 7 days of the build time.

**Detection code pattern:**
```python
# In scripts/audit_result_freshness.py (run before paper submission)
from pathlib import Path
from datetime import datetime, timedelta
import json

data_mtime = Path('data/processed/train.parquet').stat().st_mtime
stale_results = []
for result_file in Path('experiments/results').rglob('*.json'):
    r = json.load(open(result_file))
    if 'computed_at' not in r:
        stale_results.append((result_file, 'no timestamp'))
    else:
        computed = datetime.fromisoformat(r['computed_at']).timestamp()
        if computed < data_mtime:
            stale_results.append((result_file, 'older than data'))

assert not stale_results, f"Stale results: {stale_results}"
```

**Confidence:** HIGH — this is a known bug, already incurred, and preventable.

**Sources:** Finding 10 in FINDINGS.md, git status in the session context.

---

### Pitfall 11: Retraining Mid-Experiment Invalidates Backtest Comparisons

**Category:** Live-system safety during v1.1
**Phase to address:** Phase 8, 13 (all phases with live system interaction)
**Severity:** CRITICAL

**What goes wrong:** You start the TFT experiment (Phase 11). You train TFT on the Apr 17 snapshot of `train.parquet` and report P&L on the Apr 17 test set. Two days later, the auto-retrain on SCC runs (every 6h), adding new bars to the training set. You then run a "comparison" between TFT (trained on Apr 17 data) and XGBoost (retrained by the cron at Apr 19). These are no longer comparable — different training sets, different test sets. The "TFT underperforms XGBoost" headline is an artifact of training on stale data.

Similarly, the ablation experiment (Phase 12) takes ~8 hours to run across 30 feature subsets. If auto-retrain fires during those 8 hours, mid-experiment feature subsets train on different data than earlier subsets. Ablation results are no longer comparable.

**Why it happens:** The live system is on a 6-hour retrain cron. Our v1.1 experiments can take hours. Overlap is almost certain unless we explicitly prevent it.

**Consequences:** All v1.1 model comparisons have a hidden confound: some models trained on the Apr 17 snapshot, others on Apr 19 or Apr 21 snapshots. The paper's "XGBoost vs TFT" comparison is really "XGBoost on Apr 19 data vs TFT on Apr 17 data," which is meaningless.

**Warning signs:**
- Different models in the same experiment table report different `train_rows` or different `data_hash`.
- Rerunning XGBoost gives different numbers than the previous run, but you haven't changed the code.
- The cron log shows a retrain during your experiment's wall-clock window.

**Prevention strategy (concrete):**
1. **Experiment data freeze.** At the start of each v1.1 experiment, copy `data/processed/train.parquet` to `experiments/snapshots/v1.1_{experiment_name}_{timestamp}.parquet`. Every model in the experiment trains on the frozen snapshot, not the live-updating file.
2. **Pause auto-retrain during paper-finalization week.** From Apr 20 onward, disable the SCC auto-retrain cron. Document this in the paper: "The auto-retrain cron was paused from 2026-04-20 for paper finalization to ensure comparability; it resumed 2026-04-28."
3. **Log training data hash per model run.** Every model's result JSON includes `train_data_hash`. Assert across all models in a comparison table that `train_data_hash` matches.
4. **Experiment orchestrator:** `scripts/run_v1_1_experiment.py` that:
   - Freezes data to a snapshot.
   - Runs all models in the experiment against the same snapshot.
   - Saves results with metadata linking them to the snapshot.
5. **Pre-experiment checklist:** before starting any v1.1 experiment, run `scripts/check_cron_quiet.sh` that verifies no auto-retrain is scheduled within the next 12 hours.

**Detection code pattern:**
```python
# At start of comparison script:
results_to_compare = [load('xgboost.json'), load('tft.json'), load('gru.json'), ...]
hashes = {r['train_data_hash'] for r in results_to_compare}
assert len(hashes) == 1, f"Models trained on different data snapshots: {hashes}"
```

**Confidence:** HIGH — direct exposure; the live system is active and auto-retrains.

---

### Pitfall 12: Adding Features That Don't Exist in Live Bars

**Category:** Live-system safety
**Phase to address:** Phase 12 (Ablation), Phase 13 (Ensemble)
**Severity:** HIGH

**What goes wrong:** The historical dataset has 59 features. The live-bar pipeline only computes 51 (some features require multi-hour context not available in real-time). If the ablation (Phase 12) decides the "minimum feature set" includes one of the 8 historical-only features, the live system can't implement it. You've published a model that looks great in backtest but can't be deployed.

Worse, if TFT (Phase 11) trains on the 59-feature set but the live ensemble (Phase 13) has to use the 51-feature set, TFT's reported P&L isn't what a deployable TFT would achieve.

**Why it happens:** The 8 missing features are computable in principle but weren't implemented in the live pipeline. Historical vs. live feature parity is a constant maintenance burden and tends to drift.

**Consequences:** Paper's "TFT achieves X" or "minimum feature set contains Y" claims are non-deployable. Live vs. backtest reconciliation (Pitfall 2) becomes impossible because live can't even compute the features.

**Warning signs:**
- `train.parquet` has columns that the live feature-engineering script doesn't produce.
- Live trades show NaN or zeros for features that are non-NaN in the backtest.
- Ablation concludes a feature matters, but the feature is not in live bars.

**Prevention strategy (concrete):**
1. **Feature parity audit script:** `scripts/feature_parity.py` that loads one historical row and one live row, computes diff of their columns, and asserts zero difference.
2. **Backtest uses only live-compatible features by default.** All Tier-1/2/3 model training scripts import `LIVE_FEATURES` from a shared constants module. Extra features require an explicit flag.
3. **"Deployable" tag on features:** each feature in `src/features/` has a docstring line `deployable: true/false`. Non-deployable features are excluded from ablation space.
4. **Two-tier feature set in the paper.** If we have features that only exist in backtest, report results both ways: "with full 59 features: $210; with live-compatible 51 features: $208." Disclose the gap.
5. **Ablation pre-filter:** before starting feature ablation, drop all non-deployable features from the search space. Document this in the ablation methodology.

**Confidence:** HIGH — direct exposure; we already know the split between historical-only and live-computable features.

---

## Moderate Pitfalls

Mistakes that hurt the paper or system but are recoverable.

---

### Pitfall 13: Double-Counting Ensemble Diversity When Base Models Are Correlated

**Category:** Ensemble
**Phase:** Phase 13
**Severity:** MEDIUM-HIGH

**What goes wrong:** LR and XGBoost are both linear-ish on this feature set (XGBoost depth-3 is essentially "weighted sum of short splits"). Their correlation on test-set predictions is ~0.95. Averaging them as an ensemble does almost nothing — the gain from "diversity" is mostly noise. Reporting "our ensemble adds value over base models" on numbers within measurement error is misleading. Similarly, GRU and LSTM are architectural siblings — their predictions correlate >0.9.

If we build a 4-model ensemble (LR, XGBoost, GRU, LSTM) and each pair is correlated >0.85, the *effective* ensemble size is roughly 2, not 4. Sharpe only improves by √2 under perfect independence; under the actual correlation, it improves by <1.2× of base.

**Prevention strategy:**
1. **Report pairwise correlation of predictions** before claiming ensemble benefits. If correlations are all >0.85, note that ensemble diversity is low.
2. **Include a diversity metric:** e.g., Q-statistic, disagreement rate, or prediction-variance ratio. Report in the ensemble section.
3. **Test the "no ensemble" null hypothesis:** bootstrap-CI on ensemble-vs-best-single-model P&L difference. If CI includes 0, the ensemble isn't measurably better.
4. **Consider non-ML members for real diversity:** a "trade every pair" baseline, a "short-spread-always" baseline — non-ML models add true diversity.

**Sources:** [A Unified Theory of Diversity in Ensemble Learning (JMLR)](https://jmlr.org/papers/volume24/23-0041/23-0041.pdf), [Ensembles in Machine Learning (dida blog)](https://dida.do/blog/ensembles-in-machine-learning)

---

### Pitfall 14: Schema Changes Break `positions.db` or Persisted State

**Category:** Live-system safety
**Phase:** All phases with live system interaction
**Severity:** MEDIUM-HIGH (direct exposure — we've hit this)

**What goes wrong:** v1.1 phases add new features or change fee models. If the new code changes the schema of `positions.db` or `trade_log.jsonl`, existing persisted positions become un-readable or misinterpreted. Example: adding a `fee_model_version` column to trades — old trades don't have it; code crashes on `KeyError`.

**Prevention:**
1. **Schema version field** in every persistent store. New code handles old versions gracefully (backward-compat reads, forward-only writes).
2. **Migration scripts:** any schema change ships with `scripts/migrate_v1_1_to_v1_2.py`. Test migration on a copy of prod data.
3. **Dry-run new schemas:** before deploying to SCC live, run the new code against a copy of `positions.db` and verify no errors.
4. **Content-addressed IDs everywhere:** we've already moved pair_ids to content-addressed (Finding 9). Ensure position IDs, trade IDs, and any new IDs follow the same pattern.

---

### Pitfall 15: Matplotlib Backend Differences Create Figure Variance

**Category:** Reproducibility
**Phase:** Phase 10, 14
**Severity:** MEDIUM

**What goes wrong:** Figure `walk_forward_pnl.png` looks slightly different on SCC (backend: Agg) vs. local Mac (backend: MacOSX). Font rendering, anti-aliasing, line smoothness differ. The paper uses figures from one environment; reviewer regenerates in another and sees visually-different figures.

**Prevention:**
1. **Explicit backend:** every plot script starts with `matplotlib.use('Agg')` before importing pyplot. Same backend everywhere.
2. **Fixed font:** `rcParams['font.family'] = 'DejaVu Sans'` (available everywhere). Avoid system fonts.
3. **PDF not PNG for paper:** PDF is vector, eliminates rendering artifacts. Paper submission uses PDF figures.
4. **Figure-generation script:** `scripts/regenerate_all_figures.sh` runs once from raw data, produces canonical figures. No manual matplotlib tweaks in notebooks.

---

### Pitfall 16: Python Environment Drift Between Authors/Machines

**Category:** Reproducibility
**Phase:** Phase 8, 14
**Severity:** MEDIUM

**What goes wrong:** Ian's Mac runs PyTorch 2.3.0, Alvin's Mac runs 2.1.0. SCC runs 2.2.0. Same code gives slightly different numbers because some PyTorch ops changed between versions. Same failure mode for XGBoost, scikit-learn, numpy.

**Prevention:**
1. **Pinned requirements.txt** with exact versions, committed.
2. **`.python-version`** file with `3.12.3` (not `3.12`).
3. **Dockerfile or lockfile:** `requirements.lock` from `pip-tools` / `uv`. Re-deriving the environment from a lockfile is deterministic.
4. **Environment hash in result JSON:** `"env_hash": "sha256(requirements.lock)"` — results from different environments don't compare.
5. **CI check:** GitHub Actions workflow that runs the experiment scripts on the pinned environment and checks numbers match checked-in results.

---

### Pitfall 17: Feature Correlation Inflates Ablation Importance

**Category:** Feature ablation
**Phase:** Phase 12
**Severity:** MEDIUM

(Covered as sub-point in Pitfall 3.) Specifically: Amihud illiquidity on Kalshi and Amihud on Polymarket correlate ~0.7. Dropping one alone has small effect; dropping both has large effect. Reporting "Amihud-Kalshi is unnecessary" is misleading. Report correlation-cluster ablations.

**Prevention:** See Pitfall 3 (correlation-aware grouping).

---

### Pitfall 18: Scaler Fit on Full Data = Classic Leakage

**Category:** Data leakage
**Phase:** All phases
**Severity:** MEDIUM-HIGH

**What goes wrong:** StandardScaler is fit on the *full* dataset, then the same scaler is applied to train and test. This leaks test-set statistics (mean, std) into training. The effect is small in practice but is a textbook leakage pattern reviewers look for.

**Prevention:**
1. **Fit scaler on train only.**
2. **Use a sklearn Pipeline** so the scaler is automatically fit-on-train only within each cross-validation fold:
   ```python
   from sklearn.pipeline import Pipeline
   pipe = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
   pipe.fit(X_train, y_train)  # scaler fits on train only
   ```
3. **Audit: grep for `StandardScaler` in the codebase** and verify every call fits on train-only.

---

### Pitfall 19: Target Leakage via Lag Feature Construction

**Category:** Data leakage
**Phase:** All phases
**Severity:** HIGH if it occurs

**What goes wrong:** You construct `spread_lag_1 = df['spread'].shift(1)` and then split train/test. But if the shift happens across the split boundary, the first test row has a lag feature derived from the last train row. This is a subtle leakage at the split boundary.

**Prevention:**
1. **Shift before split, with explicit handling:** drop the first row of test set if it has a lag from train.
2. **Or: compute lags per-pair, then split per-pair.** The lag never crosses pairs or train/test boundaries.
3. **Sanity check:** `assert df_test['spread_lag_1'].iloc[0]` is not equal to `df_train['spread'].iloc[-1]`.

---

### Pitfall 20: Seasonal Regime Change Masks Feature Importance

**Category:** Feature ablation
**Phase:** Phase 12
**Severity:** MEDIUM

**What goes wrong:** The test set is Apr-only data. Oil contracts dominate April (WTI expiration). Ablating `near_expiry_indicator` hurts a lot in April data. But if the test set were February (no oil expiration), the same feature would be less important. Single-split ablation makes seasonal features look universally important (or universally unimportant).

**Prevention:**
1. **Walk-forward ablation:** redo ablation on each walk-forward window and report per-window feature importance.
2. **Per-category ablation:** oil vs. inflation vs. crypto separately.
3. **Disclose seasonality in paper:** "Feature X is most important in the oil-expiration regime (our April test set); its importance in other regimes is untested."

---

## Academic Integrity & LLM Disclosure

### Pitfall 21: AI-Assisted Code Without Disclosure (ICLR 2026 Policy Parallel)

**Category:** Academic integrity
**Phase to address:** Phase 14 (Final submission)
**Severity:** CRITICAL (this is DS340 final paper)

**What goes wrong:** Ian and Alvin have used Claude Code extensively throughout the project — for data pipeline implementation, matching logic, TFT setup, paper drafting. The 2026 academic norm (ICLR 2026, ICML 2026, ACM CCS 2026) is that **authors must disclose LLM usage** and take full responsibility for the content.

DS340 is a course project, not an ICLR submission, but the same principles apply:
1. **Disclose usage explicitly** in the paper's methods section or acknowledgments.
2. **Authors are responsible** for all content — fabricated citations, hallucinated results, or plagiarism produced by the AI is on the authors.
3. **Verify AI-generated code and claims** independently.

The risk: submitting a paper that extensively used Claude without disclosure, and then the TA/professor realizes this during evaluation. Even if not strictly forbidden, undisclosed AI assistance looks deceptive and may be treated as an academic-integrity issue.

Specific 2026 CS conference policies we should follow:
- **ICLR 2026:** "Authors are asked to disclose in their submission any usage of LLMs ... Papers that make extensive usage of LLMs and do not disclose this usage will be desk-rejected."
- **ACM CCS 2026:** "If AI tools were used to generate or substantially rewrite substantive content, authors must include a dedicated 'Generative AI Usage' section that names the tools used, describes which parts were generated or heavily assisted, and explains how authors validated the AI-generated content."
- **NSF (research funding):** requires disclosure of AI use in project proposals.

**Why it happens:** Authors worry that disclosing AI usage will be penalized. The opposite is true — non-disclosure is now the penalty. Norms have shifted rapidly since 2024.

**Consequences for DS340:** If the professor/TA discovers extensive undisclosed AI usage, the paper's credibility is damaged. Worst case: academic-integrity complaint.

**Prevention strategy (concrete):**
1. **Include an "AI Assistance Disclosure" paragraph** in the paper's methodology section (or acknowledgments, or as a standalone appendix). Template:
   > "This project used Anthropic's Claude (Claude Opus 4.6) extensively as a coding assistant via the Claude Code CLI. Specifically, Claude assisted with: (1) drafting boilerplate code for API ingestion (Kalshi, Polymarket), data pipelines, and evaluation scripts; (2) debugging infrastructure bugs (e.g., the `condition_ids` vs. `condition_id` discovery); (3) drafting sections of this paper including initial prose for results tables and related-work summaries. All research contributions — model architecture choices, experimental design, interpretation of results, final numerical claims — were validated by the human authors (Sabia, Jang). All code was reviewed before deployment. All reported numbers were regenerated from source data by the authors prior to submission. Any errors are the authors' responsibility."
2. **Maintain an `AI_USAGE.md` log** in the repo: which files were substantially AI-authored vs. primarily human-authored. Commit this alongside the code.
3. **Do not use AI-generated citations without verification.** Every citation in the paper must be verified by a human following the URL/DOI — LLMs hallucinate citations regularly.
4. **Re-run every experiment personally.** Before submission, both authors run the pipeline end-to-end and confirm numbers match. This is the "validation" step ICLR 2026 requires.
5. **Commit history transparency:** `Co-Authored-By: Claude` trailers on commits that were substantially AI-assisted. This is already our practice per the session context.
6. **Author responsibility statement.** Paper must contain an explicit sentence: "The authors take full responsibility for the contents of this paper, including any content developed with AI assistance."

**Warning signs the paper has AI-integrity issues:**
- Citations whose URLs 404 or point to unrelated papers (hallucinated).
- Numerical claims in the paper that don't match any file in `experiments/results/`.
- Prose that sounds like Claude's default style (hedging, "it's important to note that...") but no disclosure.

**Confidence:** HIGH — 2025–2026 academic norms are clear; ICLR 2026, ICML 2026, ACM CCS 2026 have explicit policies; DS340 will likely inherit these conventions or already has them implicitly.

**Sources:** [Policies on Large Language Model Usage at ICLR 2026 — ICLR Blog](https://blog.iclr.cc/2025/08/26/policies-on-large-language-model-usage-at-iclr-2026/), [ICLR 2026 Response to LLM-Generated Papers and Reviews](https://blog.iclr.cc/2025/11/19/iclr-2026-response-to-llm-generated-papers-and-reviews/), [ICML 2026 Intro LLM Policy](https://icml.cc/Conferences/2026/Intro-LLM-Policy), [ACM CCS 2026 Call for Papers](https://www.sigsac.org/ccs/CCS2026/call-for/call-for-papers.html), [Responsible Use of LLMs in Manuscript Preparation (Current Protocols 2026)](https://currentprotocols.onlinelibrary.wiley.com/doi/10.1002/cpz1.70300), [When and how to disclose AI use in academic publishing: AMEE Guide 192](https://www.tandfonline.com/doi/full/10.1080/0142159X.2025.2607513), [Template: AI Use Disclosure Statement for Academic Papers](https://hastewire.com/blog/template-ai-use-disclosure-statement-for-academic-papers)

---

### Pitfall 22: Hallucinated Citations in Related-Work Section

**Category:** Academic integrity
**Phase to address:** Phase 14
**Severity:** CRITICAL

**What goes wrong:** LLMs are notorious for generating plausible-sounding citations that don't exist — "Smith & Jones (2023), 'Cross-platform prediction market arbitrage via deep learning,' NeurIPS 2023." Sounds plausible; might be entirely fabricated. If the paper's Related Work section includes any AI-assisted citations, every one must be verified.

Our paper cites: Manski (2006), Wolfers & Zitzewitz (2004), Burgi/Tuccella/Zitzewitz (2026), Amihud (2002), Corwin & Schultz (2012), Kyle (1985), Roll (1984), Grinsztajn/Oyallon/Varoquaux (NeurIPS 2022), and an "arXiv 2601.07131" reference. We need to verify every one.

**Prevention strategy:**
1. **Every citation gets a verified URL in `CITATIONS.md`.** Click the link; confirm the paper exists; confirm authors/title/year match.
2. **ArXiv ID sanity:** `2601.07131` — arXiv IDs starting with 2601 would be January 2026, but cross-check. The paper we cite must actually exist at that URL.
3. **Page-level verification:** open each PDF and find the claim we're citing. If we can't find it, the citation is wrong.
4. **BibTeX from Google Scholar / DOI.org** — always use authoritative sources, never LLM-generated BibTeX.
5. **Dead-URL audit script:** `scripts/check_citations.py` that HTTP-gets each URL in `CITATIONS.md` and reports 404s.

---

### Pitfall 23: Plagiarism via Uncited Pattern Reuse

**Category:** Academic integrity
**Phase:** Phase 14
**Severity:** MEDIUM-HIGH

**What goes wrong:** LLMs trained on public research code may reproduce specific implementations verbatim. If our autoencoder code closely mirrors a published implementation (even in variable names), failing to cite is plagiarism. This is hard to detect manually but easy to surface with code similarity tools.

**Prevention:**
1. **Cite inspiration for non-trivial algorithms.** If our PPO uses stable-baselines3, cite it. If our autoencoder is inspired by a blog post, cite it.
2. **Run plagiarism check** on text we didn't write ourselves — Turnitin or iThenticate if the course provides it.
3. **Acknowledge tutorials/StackOverflow** in an acknowledgments section if specific answers influenced code.

---

## Data Leakage Audit (Pitfall 24)

**Category:** Data leakage (v1.1 new experiments risk re-introducing leakage)
**Phase to address:** All phases; explicit audit in Phase 14
**Severity:** HIGH

**What goes wrong:** Every new experiment (ablation, TFT, ensemble) is another chance to accidentally leak test info. Specific patterns to audit:

### 24a. StandardScaler fit on full data
See Pitfall 18.

### 24b. Hyperparameter search on test set
If the XGBoost hyperparameter sweep (depth ∈ {3,5,7,9} × lr × n_est = 48 configs) is scored on the test set rather than validation set, we've selected hyperparameters via test-set peeking. Our current approach reports the best single-split P&L across 48 configs — this IS test-set selection. Mitigation: either (a) use walk-forward median P&L for selection (not single-split), or (b) disclose clearly that reported P&L is in-sample-selected and compute a deflated metric.

### 24c. Feature selection via test-set peek
If the ablation process is scored on the same test set used to report final numbers, the "minimum feature set" is overfit to that test set. See Pitfall 3.

### 24d. Ensemble weights learned on test set
If ensemble weights (how much to weight LR vs. XGBoost) are optimized on the test set, that's leakage. Weights must be learned on training or validation, then evaluated on test.

### 24e. Stop-loss / take-profit thresholds tuned on test
Live-system thresholds (TIME_STOP at X minutes, TAKE_PROFIT at Y%, STOP_LOSS at Z%) tuned on backtest-test set leak info. If these were optimized post-hoc on observed trade outcomes, every paper P&L number is biased upward.

### 24f. Category-aware entry filter thresholds
The "3× threshold for non-commodity" rule in the live system — was this tuned on historical category P&L? If so, it's fit to the same data we're reporting on.

### 24g. Quality filter rules written after observing bad matches
The 10-rule quality filter was iteratively developed by inspecting specific false-match examples. If any of those examples are in the test set, the filter has "seen" the test set. Mitigation: audit which false matches were the basis for each rule; drop test-set-derived rules from the reported pipeline.

**Prevention strategy:**
1. **Train / validation / test triple split.** Train for fitting, validation for selection (hyperparameters, ablation, ensemble weights, filters), test for reporting. Test set is touched exactly once per final model.
2. **Leakage audit document:** before submission, walk through every step of the pipeline and answer "did this use test-set data?" for each. Commit the audit to `LEAKAGE_AUDIT.md`.
3. **Time-ordered split discipline.** Validation is a chronological slice between train and test: `[train][val][test]` on the time axis, never shuffled.
4. **Hold-out test set physically separate.** In code, test data lives in a separate file that is only loaded in the final evaluation script. Anything that imports test data is flagged.
5. **Disclosure when leakage found:** if audit reveals a leakage (e.g., stop-loss thresholds were tuned on test data), redo the analysis on a fresh split or disclose the leakage explicitly.

**Sources:** [Data Leakage, Lookahead Bias, and Causality in Time Series (Kyle Jones, Medium)](https://medium.com/@kyle-t-jones/data-leakage-lookahead-bias-and-causality-in-time-series-analytics-76e271ba2f6b), [Avoiding Data Leakage in Timeseries 101 (Towards Data Science)](https://towardsdatascience.com/avoiding-data-leakage-in-timeseries-101-25ea13fcb15f/), [Purged cross-validation (Wikipedia)](https://en.wikipedia.org/wiki/Purged_cross-validation), [Hidden Leaks in Time Series Forecasting (arXiv 2025)](https://arxiv.org/html/2512.06932v1), [Data Preparation Without Data Leakage (MachineLearningMastery)](https://machinelearningmastery.com/data-preparation-without-data-leakage/)

---

## Technical Debt Patterns (v1.1 Specific)

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Skip TFT attention-weight audit | Faster Phase 11 | Published TFT result may be collapse, not learning | Never — attention audit takes 5 minutes |
| Cache `experiments/results/*.json` without data_hash | Faster re-runs | Stale results in paper (already happened v1.0) | Never — input-hash is a one-line fix |
| Run TFT with default hyperparameters | Phase 11 finishes on time | Overfit result that doesn't generalize | Only if explicitly disclosed as "out-of-the-box defaults" |
| Skip live-vs-backtest reconciliation | Phase 9 finishes on time | Strongest paper claim is unverified | Never — this is the centerpiece v1.1 contribution |
| Report per-trade Sharpe as headline | Paper looks stronger | Reviewers catch it, credibility tanks | Never — always use per-pair for headline |
| Ablate on test set, report "minimum set" | Ablation looks rigorous | P-hacking via selection bias | Never — use validation set for selection |
| Use same seed for all stochastic models | Simpler to run | Can't report variance, single-seed is a point estimate | Never — use multi-seed for Tier 2/3 |
| Skip LLM disclosure paragraph | Paper is shorter | 2026-era academic norm violation | Never — 1 paragraph is cheap insurance |
| Cherry-pick best walk-forward window for abstract | Better-looking headline | Detected as cherry-pick by any reviewer | Never — report median + range |

---

## Integration Gotchas (v1.1)

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| SCC auto-retrain cron during v1.1 experiments | Overlapping training runs corrupt comparison | Pause cron; freeze data snapshot; run experiment; resume cron |
| Backtest vs. live feature parity | Ablation proposes feature X that live doesn't compute | Pre-filter ablation space to live-deployable features |
| PPO from stable-baselines3 | Internal RNG ignores global seed | Pass `seed` to PPO constructor explicitly; rerun with multi-seed |
| PyTorch Forecasting TFT | Dataset object pre-scales internally | Disable internal scaler OR ensure single scaling pass |
| `matplotlib` on SCC (headless) | `plt.show()` crashes | `matplotlib.use('Agg')` before first import |
| JSON serialization of numpy types | `np.float32` not JSON-serializable | Always cast to `float()` before `json.dump` |
| `stable-baselines3` version pinning | SB3 3.x has different PPO API than 2.x | Pin to exact version in `requirements.txt` |
| Pandas `to_parquet` with index | Index lost on reload | `df.reset_index(drop=True).to_parquet(...)` or `read_parquet(... columns=...)` explicit |

---

## Performance Traps (v1.1 Specific)

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| TFT training time > budget | 1 epoch takes > 30 min on CPU | Use hidden_size=16, batch_size=64, max_epochs=30 | At any meaningful dataset size without GPU |
| Ablation runtime > 8 hours | 30 subsets × 30 min each | Cache intermediate models; run in parallel | 30+ subsets on sequential CPU |
| Ensemble prediction latency on live | Each bar requires 4 model inferences | Cache model objects; batch predictions | When number of pairs × models > 1000 inferences/minute |
| Reconciliation JSON is huge | `live_trades.json` reaches GB | Daily snapshot + rolling 30-day window only | Live system running > 30 days |

---

## "Looks Done But Isn't" Checklist (v1.1 Submission Gate)

Before paper submission on April 27, verify:

### Reproducibility
- [ ] **All seeds set:** `set_all_seeds(42)` called at top of every training script
- [ ] **Multi-seed results:** Tier-2/3 models reported with median ± IQR over 5 seeds
- [ ] **Cache freshness:** every `experiments/results/*.json` has matching `data_hash`
- [ ] **Environment lockfile:** `requirements.lock` committed; CI passes
- [ ] **Single-command reproduction:** `make reproduce` regenerates all tables/figures

### TFT (Phase 11)
- [ ] **Attention-weight audit:** entropy across variables > 0.5 × log(n_features)
- [ ] **Walk-forward validation:** TFT tested on 11 windows, not just single split
- [ ] **Downsized hyperparameters disclosed:** hidden_size, dropout reported in paper
- [ ] **Time-box respected:** TFT took ≤ 1 day; if it didn't, result is "TFT did not converge at this scale"
- [ ] **Feature parity:** TFT trained on live-deployable features only

### Live vs Backtest (Phase 9)
- [ ] **Trade-level reconciliation notebook exists** and is referenced in paper
- [ ] **Fee models aligned:** unit test `assert backtest_fee(t) == live_fee(t)` passes
- [ ] **Timestamp discipline:** all timestamps UTC, stored with tz-info
- [ ] **Price source aligned:** backtest and live both use VWAP (or both use mid)
- [ ] **Live-pair-universe backtest run** and reported alongside filtered-144-pair result

### Feature Ablation (Phase 12)
- [ ] **Three-way split** (train/val/test) used; test touched once
- [ ] **Correlation-clustering** applied before ablation; groups ablated together
- [ ] **Walk-forward ablation** (not single-split) used for selection
- [ ] **Per-category results** reported for seasonal robustness
- [ ] **Pre-registered protocol** exists; deviations explained

### Ensemble (Phase 13)
- [ ] **Base model correlations** reported; diversity metric computed
- [ ] **Accepted vs. rejected** trades P&L both reported
- [ ] **No-filter baseline** in results table
- [ ] **Deflated Sharpe** computed for ensemble
- [ ] **Ensemble weights learned on validation**, not test

### Paper Credibility (Phase 10, 14)
- [ ] **Per-pair Sharpe** is the headline; per-trade in parentheses
- [ ] **Walk-forward median + range** in abstract (not single window)
- [ ] **Survivorship-bias disclosure paragraph** in Limitations
- [ ] **Transaction-cost sensitivity** table (already in v1.0 draft) updated for v1.1 models
- [ ] **Scale-curve plateau** explicitly disclosed with training-cap annotation
- [ ] **No-filter, full-universe result** shown alongside filtered results

### Academic Integrity (Phase 14)
- [ ] **AI-assistance disclosure paragraph** in methods or acknowledgments
- [ ] **`AI_USAGE.md`** committed with file-level attribution
- [ ] **Every citation verified** by URL/DOI; `CITATIONS.md` updated
- [ ] **Authors' responsibility statement** present in paper
- [ ] **Plagiarism check run** if course provides one

### Live System Safety (Phase 8, 13)
- [ ] **Auto-retrain cron paused** during paper-finalization week (Apr 20–27)
- [ ] **Data snapshots frozen** for each v1.1 experiment
- [ ] **`positions.db` schema** backward-compatible with v1.0 persisted state
- [ ] **All features in ablation space** are live-deployable

### Data Leakage (Pitfall 24)
- [ ] **`LEAKAGE_AUDIT.md`** exists and is committed
- [ ] **Scaler fits on train only** (code audit)
- [ ] **Hyperparameters selected on validation**, not test
- [ ] **Stop-loss/take-profit thresholds** not tuned on reported-test data
- [ ] **Quality-filter rules** that were derived from test-set examples disclosed or removed

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Stale `experiments/results/*.json` (Pitfall 10) | LOW | Delete cache, rerun end-to-end, commit fresh results; 4–6 hours |
| TFT attention collapse (Pitfall 1) | LOW | Report "TFT did not converge at this scale" as finding; no rerun needed |
| Live-vs-backtest gap > 20% (Pitfall 2) | MEDIUM | Debug by trade-level reconciliation; likely a fee or timestamp bug; 1–2 days |
| P-hacked ablation (Pitfall 3) | MEDIUM-HIGH | Redo with three-way split and correlation-clustering; 1 day |
| Inflated ensemble Sharpe (Pitfall 4) | LOW | Report with/without concordance filter; 2 hours; reframe paper section |
| Per-trade Sharpe published as headline (Pitfall 5) | LOW | Recompute per-pair; edit paper; 2 hours |
| Missing seeds (Pitfall 9) | MEDIUM | Add seeds; rerun stochastic models with multi-seed; 4–8 hours |
| Undisclosed AI assistance (Pitfall 21) | LOW | Add disclosure paragraph; 30 minutes |
| Hallucinated citation discovered (Pitfall 22) | LOW | Remove or correct citation; 15 minutes per |
| Data leakage discovered (Pitfall 24) | HIGH | Redo analysis on fresh split; may take days; worst case: restructure paper around smaller-scope claim |

---

## Pitfall-to-Phase Mapping (for Roadmap Consumer)

| Pitfall | Prevention Phase | Verification Step |
|---------|------------------|-------------------|
| 1. TFT overfits | Phase 11 | Attention entropy > threshold; walk-forward window-11 P&L positive |
| 2. Live-vs-backtest gap | Phase 9 | Reconciliation notebook shows < 10% gap on overlap trades |
| 3. Ablation p-hacking | Phase 12 | Three-way split used; correlation-cluster ablation; pre-registered protocol |
| 4. Concordance filter inflates Sharpe | Phase 13 | Accept/reject trades both reported; deflated Sharpe computed |
| 5. Per-trade Sharpe headline | Phase 10, 14 | Every Sharpe in paper uses `src/evaluation/sharpe.py` functions with named method |
| 6. Cherry-picked window | Phase 10 | Abstract reports median + range, not single window |
| 7. Scale-curve plateau | Phase 10 | Annotation on Fig 2 marking training-cap; text explicitly discloses |
| 8. Survivorship bias silence | Phase 10 | Limitations paragraph names it; matching-funnel diagram |
| 9. Missing seeds | Phase 8, 14 | `set_all_seeds` imported and called in every training script; multi-seed results for Tier 2/3 |
| 10. Stale result cache | Phase 8, 14 | `data_hash` in every result JSON; pre-submission freshness audit passes |
| 11. Retrain during experiment | Phase 8, 13 | Data snapshot frozen per experiment; auto-retrain paused Apr 20–27 |
| 12. Live-incompatible features | Phase 12, 13 | Ablation space pre-filtered to `LIVE_FEATURES`; parity audit script passes |
| 13. Correlated ensemble members | Phase 13 | Pairwise correlation table included; diversity metric reported |
| 14. Schema breakage | All | Schema version field; migration scripts tested |
| 15. Matplotlib backend | Phase 10, 14 | Figures generated with `Agg`; PDF output |
| 16. Python env drift | Phase 8, 14 | `requirements.lock` committed; CI passes |
| 17. Correlated ablation attribution | Phase 12 | Correlation clustering applied (covered in 3) |
| 18. Scaler fit on full data | All | Pipeline pattern; code audit |
| 19. Lag feature leakage | All | Lag before split, handle boundary; per-pair splits |
| 20. Seasonal masking in ablation | Phase 12 | Per-category ablation; walk-forward ablation |
| 21. AI-assistance non-disclosure | Phase 14 | Disclosure paragraph; `AI_USAGE.md` |
| 22. Hallucinated citations | Phase 14 | `CITATIONS.md` with verified URLs; dead-URL script passes |
| 23. Pattern-reuse plagiarism | Phase 14 | Inspiration cited; acknowledgments complete |
| 24. Data leakage audit | All; formal in Phase 14 | `LEAKAGE_AUDIT.md` committed |

---

## Sources

**Financial ML / Backtesting:**
- [Statistical Overfitting and Backtest Performance (Bailey et al., SSRN)](https://sdm.lbl.gov/oapapers/ssrn-id2507040-bailey.pdf) — HIGH confidence on selection bias, deflated Sharpe, CSCV
- [The Deflated Sharpe Ratio (Bailey & Lopez de Prado)](https://www.researchgate.net/publication/286121118_The_Deflated_Sharpe_Ratio_Correcting_for_Selection_Bias_Backtest_Overfitting_and_Non-Normality)
- [Backtest overfitting in the ML era (ScienceDirect 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110)
- [Interpretable Hypothesis-Driven Trading: A Rigorous Walk-Forward Validation Framework (arXiv 2512.12924)](https://arxiv.org/html/2512.12924v1)
- [QuantConnect Live-Backtest Reconciliation Docs](https://www.quantconnect.com/docs/v2/cloud-platform/live-trading/reconciliation)
- [LuxAlgo: Backtesting Limitations — Slippage and Liquidity](https://www.luxalgo.com/blog/backtesting-limitations-slippage-and-liquidity-explained/)

**Reproducibility:**
- [PyTorch Reproducibility Documentation](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [Reproducibility in machine-learning-based research (Semmelrock et al., AI Magazine 2025)](https://onlinelibrary.wiley.com/doi/10.1002/aaai.70002)
- [The Challenge of Reproducible ML (arXiv 2109.03991)](https://arxiv.org/pdf/2109.03991)
- [Ensuring Training Reproducibility in PyTorch (LearnOpenCV)](https://learnopencv.com/ensuring-training-reproducibility-in-pytorch/)

**TFT / Deep Learning on Small Data:**
- [Temporal Fusion Transformers for interpretable multi-horizon time series forecasting (Lim et al., IJF)](https://www.sciencedirect.com/science/article/pii/S0169207021000637)
- [TFT — darts documentation](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tft_model.html)
- [PyTorch Forecasting Stallion Tutorial](https://pytorch-forecasting.readthedocs.io/en/v1.4.0/tutorials/stallion.html)

**Data Leakage:**
- [Data Leakage, Lookahead Bias, and Causality in Time Series (Kyle Jones)](https://medium.com/@kyle-t-jones/data-leakage-lookahead-bias-and-causality-in-time-series-analytics-76e271ba2f6b)
- [Avoiding Data Leakage in Timeseries 101 (Towards Data Science)](https://towardsdatascience.com/avoiding-data-leakage-in-timeseries-101-25ea13fcb15f/)
- [Hidden Leaks in Time Series Forecasting (arXiv 2512.06932)](https://arxiv.org/html/2512.06932v1)
- [Purged cross-validation (Wikipedia)](https://en.wikipedia.org/wiki/Purged_cross-validation)

**Academic Integrity / LLM Disclosure:**
- [Policies on Large Language Model Usage at ICLR 2026](https://blog.iclr.cc/2025/08/26/policies-on-large-language-model-usage-at-iclr-2026/) — authoritative for 2026 norms
- [ICLR 2026 Response to LLM-Generated Papers and Reviews](https://blog.iclr.cc/2025/11/19/iclr-2026-response-to-llm-generated-papers-and-reviews/)
- [ICML 2026 Intro LLM Policy](https://icml.cc/Conferences/2026/Intro-LLM-Policy)
- [ACM CCS 2026 Call for Papers](https://www.sigsac.org/ccs/CCS2026/call-for/call-for-papers.html)
- [Responsible Use of LLMs in Manuscript Preparation (Mijatović, Current Protocols 2026)](https://currentprotocols.onlinelibrary.wiley.com/doi/10.1002/cpz1.70300)
- [When and how to disclose AI use in academic publishing: AMEE Guide 192](https://www.tandfonline.com/doi/full/10.1080/0142159X.2025.2607513)

**Ensemble / Diversity:**
- [A Unified Theory of Diversity in Ensemble Learning (JMLR 2023)](https://jmlr.org/papers/volume24/23-0041/23-0041.pdf)
- [Diverse Models, United Goal — Survey of Ensemble Learning (CAAI 2025)](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cit2.70030)

**Project-specific:**
- `FINDINGS.md` (internal log) — HIGH confidence on project exposure to Pitfalls 2, 10, 11
- `.planning/research/PITFALLS.md` (v1.0) — HIGH confidence on baseline pitfalls this file extends
- `PAPER_DRAFT.md` v1.0 — identifies where credibility risks live in current prose

---
*Pitfalls research for: Kalshi × Polymarket cross-platform arbitrage, v1.1 milestone*
*Researched: 2026-04-16*
*Author: Research subagent for GSD `/gsd:new-milestone` Phase 6 (v1.1)*
