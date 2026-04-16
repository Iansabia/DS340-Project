# Feature Research — v1.1 "Extended Evidence & Submission"

**Domain:** Cross-platform prediction market arbitrage (Kalshi vs. Polymarket) — v1.1 paper-finalization milestone
**Researched:** 2026-04-16
**Confidence:** HIGH (strong literature anchors for all six capabilities; existing system provides internal calibration for "what's realistic")
**Scope note:** This document covers ONLY the six v1.1 target capabilities. The v1.0 `FEATURES.md` (retained as history) covers the foundational system that is already built.

---

## 0. How to Read This Document

Each of the six v1.1 capabilities has three subsections:

- **Table stakes** — must-have for the capability to be credible in a DS340 paper aimed at submission. Missing any of these would be flagged by a reviewer.
- **Differentiators** — elevate the paper beyond a routine DS340 project. Each is a concrete "paper-ready finding."
- **Anti-features** — common mistakes or tempting-but-wrong approaches. These go into the out-of-scope list in REQUIREMENTS.md.

Each capability also has a **dependencies** note flagging upstream prerequisites (e.g., TFT must be working before TFT can be in an ensemble).

---

## 1. Capability: TFT (Temporal Fusion Transformer) on Small Data (~7k rows, ~47-bar sequences)

**Why this capability exists:** TFT was the deferred Tier-2 transformer from v1.0. Training it now (a) completes the originally-promised model suite in the proposal, (b) gives the paper a transformer data point alongside GRU/LSTM, and (c) provides attention-based interpretability that SHAP cannot.

### 1.1 Table Stakes

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Shrunk architecture for small-data regime** (hidden_size=8-16, attention_head_size=1-2, hidden_continuous_size=8) | Default TFT (hidden=160, 4 heads) will badly overfit on 6,802 training rows. PyTorch Forecasting docs explicitly recommend hidden_size=8 for small datasets. | MEDIUM | Start at hidden_size=16, attention_head_size=2, lstm_layers=1, hidden_continuous_size=8. If validation loss diverges, shrink further to hidden_size=8. |
| **Aggressive regularization** (dropout=0.3-0.5, gradient_clip_val=0.1, weight_decay=0.01) | Literature on transformer overfitting on small data (AttentionDrop 2024, DropHead 2020) confirms 0.3-0.5 dropout for fine-tuning on small data. Our GRU/LSTM already use dropout=0.2; TFT should go higher. | LOW | gradient_clip_val=0.1 is a pytorch-forecasting standard. Early stopping on validation loss with patience=5-10. |
| **Quantile loss with quantiles=[0.1, 0.5, 0.9]** | TFT's designed loss function — NOT MSE. Gives prediction intervals "for free" which becomes a figure in §5. Using MSE would squander TFT's core feature. | LOW | `QuantileLoss(quantiles=[0.1, 0.5, 0.9])`. Use median (0.5) for the point forecast in P&L simulation. |
| **Identical evaluation protocol to GRU/LSTM** | Same train/test split, same target (Δs(t)), same features, same fee assumption (2pp), same metrics (RMSE, MAE, dir. accuracy, P&L, Sharpe). Without protocol parity, comparisons are meaningless. | LOW | Drop into existing evaluation harness. Add one row to Table 2. |
| **Lookback window matched to GRU/LSTM** (24 bars initially, optionally 12 for ablation) | Prevents conflating "TFT underperforms" with "TFT trained on different input." | LOW | TFT handles static + known-future + observed-past covariates; if we only use observed-past (no future covariates available), the input spec is equivalent to GRU/LSTM. |
| **GroupNormalizer on target per-pair** | PyTorch Forecasting requires per-group (per-pair) target normalization to handle scale differences across pairs. Skipping this leads to the model memorizing pair identity instead of learning dynamics. | LOW | `TimeSeriesDataSet(..., target_normalizer=GroupNormalizer(groups=["pair_id"], transformation="softplus"))`. |
| **TimeSeriesDataSet with correct time indexing** | Off-by-one errors in time_idx silently break TFT (it'll train fine, produce plausible predictions, all wrong). Must match the chronological ordering used by GRU/LSTM. | MEDIUM | Verify with a single-pair smoke test before running the full training. |
| **Training time accounted for in paper** | TFT typically takes 10-30x longer than GRU per epoch. Must decide: single run with fixed hyperparams, or skip hyperparameter search. | LOW | One run, hyperparameters justified from PyTorch Forecasting docs (citable default). Explicit "no hyperparameter search due to compute budget; default-initialized" statement in §4.3. |

### 1.2 Differentiators

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Variable Selection Network (VSN) attention plot** — native to TFT | Shows which of the 59 features TFT decides are informative PER time step. Free interpretability output that SHAP cannot produce. | LOW (if TFT trains successfully) | `model.interpret_output(raw_predictions)` returns attention weights over features and time. Plot as a heatmap. Compare to SHAP rankings — if they agree, that's Finding 23-style evidence. |
| **Temporal attention heatmap** — TFT's decoder attention over input timesteps | Answers "what lookback horizon does TFT actually use?" — directly tests the Finding 4 claim ("short lookback beats long lookback"). | LOW | `model.interpret_output` returns temporal attention. Plot as (pair, timestep, attention_weight) heatmap. If TFT attends only to last 6-12 timesteps, that confirms Finding 4 on a different architecture. |
| **Prediction intervals in P&L simulation** | Use the 10%/90% quantiles to construct confidence-gated entry filter: only trade when 10th percentile prediction still has the same sign as the 50th. This is a risk-aware entry filter that regression cannot provide. | MEDIUM | Adds a new column to Table 2: "P&L with interval-gated entry." Directly demonstrates TFT's quantile output adds value. |
| **Honest null finding framing** | If TFT underperforms GRU/LSTM (likely at our data scale), the paper gains "transformer does not help at 47-bar sequences" — a new empirical data point consistent with Finding 2 and the 2026 ArXiv 2603.16886 controlled comparison. | LOW | One-sentence finding in §6.1. The null result extends the "complexity is not an edge" narrative to a third architecture family. |
| **Literature-grounded framing that cites 2024-2026 TFT critiques** | Cite the 2026 ArXiv study showing TFT ranks 6-8th of 10 architectures on financial data (ModernTCN and PatchTST win); cite PatchTST as "what we'd try with more data." | LOW | Add 2-3 sentences in §2 (Related Work) and §7 (Future Work). Differentiates the paper from projects that "just ran TFT." |

### 1.3 Anti-Features

| Anti-Feature | Why Avoid | Alternative |
|--------------|-----------|-------------|
| **Running TFT with default hyperparameters (hidden=160, 4 heads)** | Guaranteed overfit on 6,802 rows. Will show spurious "TFT fails" when the real failure is configuration. | Use small-data hyperparameters from §1.1. Cite PyTorch Forecasting docs as source. |
| **Hyperparameter optimization via Optuna** | 30+ minutes per trial × 50+ trials = days of compute we don't have. PyTorch Forecasting's `optimize_hyperparameters` is known to over-tune at this data scale. | Fixed hyperparameters from small-data literature. State explicitly in §4.3. |
| **Adding PatchTST, Autoformer, or TimesNet implementations** | Each is a full sub-project. Out of scope for v1.1 given the 11-day deadline (Apr 16 → Apr 27). | Mention PatchTST/ModernTCN in §7 Future Work with the 2026 ArXiv citation. |
| **Training TFT with MSE loss to "match" GRU/LSTM** | Defeats the purpose of using TFT. MSE doesn't use the quantile output. | Use QuantileLoss. For direct comparison to GRU/LSTM's point forecasts, report the 0.5-quantile as the point forecast. |
| **Using TFT without the Variable Selection Network interpretability** | Skipping this loses the one differentiator TFT has over GRU/LSTM. | Always plot the VSN output (1 figure). It takes 20 lines of code. |
| **Trying to get TFT to beat XGBoost** | Prior scaling curve (Finding 22 pending) already shows this is unlikely at our data scale. Pressuring the model with HPO tricks will just produce overfit. | Report whatever honest result TFT produces. Frame as "null result extends simplicity thesis." |
| **Retraining TFT as part of the live system** | TFT training takes 10+ min; our 6-hour retrain cycle is already tight. Live system must stay on LR+XGBoost. | TFT is research-only. Train once, report once. Live system is unchanged. |

### 1.4 Dependencies

- **Upstream:** v1.0 dataset (complete), v1.0 evaluation harness (complete), GRU/LSTM baselines trained (complete).
- **Downstream:** Ensemble exploration (Capability 4) benefits if TFT is available as a 5th model in the ensemble, but does NOT strictly require it.

---

## 2. Capability: Live vs Backtest Reconciliation (10+ days of real autonomous trading)

**Why this capability exists:** Since April 9, the system has been making real paper-trading decisions on SCC. Almost no student paper has live data. This is the single most unique asset in the paper. But it's only credible if the reconciliation is rigorous — a hand-wavy "our live results look similar" section undersells the contribution.

### 2.1 Table Stakes

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Defined reconciliation period** (explicit start date, end date, # trading cycles, # trades) | Reviewers need to know what window the comparison covers. Handwavy "since April" fails. | LOW | State explicitly: "Live trading period: April 9-25, 2026 (17 calendar days, ~1,632 15-min cycles, N executed trades)." |
| **Paired live-vs-simulation P&L at the trade level** | Can't attribute divergence without trade-by-trade pairing. Need a joined table: live_trade_id → simulation_prediction at same timestamp → outcome. | MEDIUM | Requires joining `positions.db` + `trade_log.jsonl` with a shadow-simulation that replays the same features through the trained model offline. |
| **Summary comparison table** (live vs backtest: P&L, win rate, avg trade, Sharpe, # trades) | Headline table in §5.9 of the paper. Must include both values side-by-side with difference. | LOW | Single table, 6-8 rows, 3 columns (metric, backtest, live). |
| **Tracking error** (live_pnl - backtest_pnl time series, with std dev and correlation) | Standard portfolio-management metric. Shows how closely live follows backtest. TE < 1pp/trade is tight; TE > 5pp is concerning. | LOW | `tracking_error = np.std(live_pnl_per_trade - backtest_pnl_per_trade)`. Also report correlation between the two series. |
| **Exit-reason attribution** (share of realized P&L by TAKE_PROFIT / TIME_STOP / STOP_LOSS / RESOLUTION_EXIT / MOMENTUM) | Finding 14 already showed 72% of profit came from 1.4% of trades (TAKE_PROFIT). Live data refreshes this. Tells the story of WHY the live P&L differs from backtest. | LOW | Already have this telemetry from Finding 14. Just update numbers and add to §5.9. |
| **Category-level divergence** (live vs backtest P&L per category: oil, crypto, politics, etc.) | Finding 6 and Finding 21 both established category effects. Must test whether the category ranking holds in live data. | LOW | Stratify live trades by category, compare per-category live P&L to backtest per-category P&L from Table 4. |
| **Honest acknowledgment of paper-trading caveats** | Paper trading has zero slippage, zero adverse selection, instant fills at the simulated mid. Live paper P&L is NOT a prediction of real-money P&L. | LOW | 1-paragraph "Limitations of Paper Trading" subsection in §5.9. Cite Quantitative Brokers / slippage literature. |

### 2.2 Differentiators

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **P&L attribution decomposition** (Clean P&L vs Model P&L vs Unexplained) | Adapt FRTB-style P&L attribution framework. Decompose: (a) price-change P&L (predicted by features), (b) model-explained P&L (predicted by model given features), (c) unexplained P&L (surprises). Standard quant-risk methodology. | MEDIUM | 3-panel figure showing the three components over time. Cite P&L attribution literature (FRTB, BPI). |
| **Slippage decomposition — delay vs impact** | Even in paper trading, there's "delay slippage" — the price changes between when we fetch features and when the simulated trade fills. Quantify this. Market impact is zero in paper trading, so this one is a clean decomposition (vs real trading where attribution is impossible). | LOW | Compute delay from `cycle_start_time` to `trade_execution_time` in the cycle log. Report mean/median/p95 delay and resulting P&L cost. |
| **Walk-forward-style live evaluation** — split live period into early/mid/late sub-windows | Mirrors the 11-window backtest structure. Tests whether live edge is stable WITHIN the live period (not just vs backtest). | LOW | Split 17 days into 3 sub-windows of ~6 days each. Compute per-sub-window Sharpe. Add a mini-table. |
| **Live-only oil-vs-non-oil breakdown** | Finding 6 claimed oil is the edge. Live data either confirms or refutes. A clean "oil = +X%, non-oil = -Y%" result in live is stronger than the backtest version. | LOW | Already have the oil category tag. Just stratify live trades and report. |
| **Pair identity verification across live and backtest** | After Finding 9 (pair_id schema bug), any reconciliation MUST verify that live_pair_N matches backtest_pair_N by content-addressed identifier. This ablates the bug as a cause of divergence. | LOW | State the verification in the methodology section. Include a count: "N pairs appeared in both live and backtest; M appeared only in live; K only in backtest." |
| **Figure: live vs backtest cumulative P&L on the same axis** | Single figure that tells the whole story. If they track each other, that's the result. If they diverge, the figure shows the structural break. | LOW | Two lines on one axis. Color-blind-safe palette from seaborn-colorblind. |

### 2.3 Anti-Features

| Anti-Feature | Why Avoid | Alternative |
|--------------|-----------|-------------|
| **Claiming live paper trading validates real-money returns** | False and reviewers will catch it. Paper trading has no market impact, no queue dynamics, no actual fills. | Frame as "live simulation with real market data fetches" — emphasize the same model, same features, same data path, but simulated execution. |
| **Hiding the live P&L if it's negative** | Academic dishonesty. The point of the reconciliation IS the honest comparison. | Report it. If live underperforms backtest, discuss why (small sample, regime shift, pair identity changes). A negative result with diagnosis is a stronger contribution than a suppressed result. |
| **Aggregating all live trades into a single number and comparing to backtest** | Loses the attribution story. Reviewers will ask "what drove the divergence?" | Trade-level paired comparison. At minimum, per-category breakdown. |
| **Re-running the backtest with live data mixed in** | Data leakage. Backtest must remain purely historical, live must remain purely forward. | Keep them strictly disjoint. The whole value is that live is genuinely out-of-sample. |
| **Claiming 10 days is sufficient statistical power for Sharpe claims** | Not enough trades/time for tight CIs. Finding 17 already addressed Sharpe inflation from short windows. | Report live Sharpe with CI, note the CI is wide, avoid strong magnitude claims. |
| **Using live data to retune model hyperparameters for the paper** | Subtle look-ahead bias. The model used in live must be the one trained on pre-live data. | Document which model checkpoint produced which live trade. No retroactive tuning. |
| **Treating live-backtest agreement as proof of model quality** | Could just mean neither has encountered a real stress event. | Explicitly state: "Agreement between live and backtest confirms no infrastructure bugs and consistent feature computation — it does NOT validate the model's predictive quality, which is established by §5.1-5.6." |

### 2.4 Dependencies

- **Upstream:** Live system running continuously (currently running), content-addressed pair IDs (complete per Finding 9).
- **Downstream:** Nothing — this is a leaf capability that produces a self-contained paper section (§5.9).

---

## 3. Capability: Feature Ablation Study

**Why this capability exists:** We currently claim 59 features matter but have no evidence which ones. Finding 10 (adding rolling features was neutral) and Finding 12 (microstructure features are neutral at 47 bars) hint at the answer, but there's no systematic study. An ablation gives the paper a "minimum feature set" finding — a parsimony result consistent with the "complexity is not an edge" thesis.

### 3.1 Table Stakes

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Leave-One-Group-Out (LOGO) ablation over 5 feature groups** | Paper has 5 clearly-defined groups (Raw aligned=18, Cross-platform basic=6, Rolling/momentum=9, Classical microstructure=13, Prediction-market-specific=13). LOGO removes one group at a time and retrains. | MEDIUM | 5 retrains of LR + XGBoost = 10 total runs. Each ~15 sec. Roughly 3 min total. |
| **Baseline "all features" as reference** | Need the no-ablation number in the top row of the ablation table to compute deltas. | LOW | Already have: LR +$201.69, XGBoost +$208.85. |
| **Metric consistency with main tables** | Report the same metrics in the ablation table as in Table 2 (RMSE, Dir. Accuracy, P&L@2pp, Sharpe). | LOW | Reuse evaluation harness. |
| **Ablation table with deltas** | Every row shows the P&L change vs baseline, making it easy to read ("dropping microstructure cost -$X" or "dropping microstructure GAINED +$X"). | LOW | Table column: "ΔP&L vs full". Sort by magnitude of delta. |
| **Two models (LR + XGBoost) to rule out model-specific artifacts** | A feature group that's critical for XGBoost but neutral for LR tells a different story than one critical for both. | LOW | Add both models as separate columns in the ablation table. |
| **Explicit handling of NaN/zero-variance columns** | The 8 columns excluded from the 59 (per §5.1) must be documented. Ablation shouldn't reintroduce them. | LOW | One-sentence note in the ablation methodology. |

### 3.2 Differentiators

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Statistical significance via bootstrap CIs** | Finding 17 already uses bootstrap for Sharpe. Extend to ablation deltas: "Dropping rolling features causes a ΔP&L of -$X with 95% CI [a, b]." | MEDIUM | Bootstrap 1000 resamples of the test set per ablation variant. Compute CI of the delta. Significant if CI excludes 0. |
| **"Minimum sufficient feature set" derivation** — greedy forward selection | Start with 0 features, greedily add the feature that most improves P&L until improvement plateaus. Report the minimum set needed to match full-feature P&L within 5%. | MEDIUM | Paper-worthy finding: "A 12-feature subset achieves 95% of full-feature performance." This is a direct parsimony result. |
| **Alignment with SHAP rankings from v1.0** | We already have SHAP rankings (polymarket_vwap dominates). If the greedy selection's top features match SHAP's top features, that's dual evidence from two methods. | LOW | Already have SHAP data (Finding 5). Just add a side-by-side table. |
| **Grouped feature importance plot** | Adapt the "grouped feature importance" framework (Springer 2022) — bar chart with mean importance per group, error bars from LOGO deltas. | LOW | One figure. Matches the academic framework for mixed categorical/numerical tabular features. |
| **Counter-to-intuition framing if applicable** | Finding 12 hinted microstructure features are neutral at 47 bars. If ablation confirms "dropping 13 academic microstructure features costs <$5 in P&L," that's a publishable finding consistent with the data-scaling hypothesis. | LOW | 1-paragraph discussion in §5.x and §6.x. |
| **"Why more features don't help" section** grounded in bias-variance | Our regime is variance-dominated (small data). More features = more variance. Ablation results quantify this. Cite January 2026 "Matched Filter" paper (Finding 11) as validation. | LOW | 2-3 sentences in §6. |

### 3.3 Anti-Features

| Anti-Feature | Why Avoid | Alternative |
|--------------|-----------|-------------|
| **Single-feature leave-one-out** (LOFO on all 59 features) | With 59 features × 2 models = 118 runs. Individual feature ablations are noisy (many features are correlated) and produce false "this feature doesn't matter" conclusions when the feature is redundant with another. | LOGO (5 groups). If time permits, LOFO on the top-5 SHAP features as a supplement. |
| **Ablating features and claiming "feature X doesn't matter"** | Absence of evidence for a single feature is not evidence of absence. Correlated features can substitute for each other. | Frame results as "group X can be removed with negligible loss," not "feature X is useless." |
| **Retraining neural models (GRU/LSTM/TFT) for every ablation variant** | 5 ablations × 3 models × 3 min/epoch × 15 epochs = 3+ hours of compute. Not worth it given the main finding is about Tier 1. | Ablate on LR and XGBoost only. Add one sentence in methodology: "Ablation performed on Tier 1 models; Tier 2 qualitatively expected to follow." |
| **Using test-set performance to guide greedy selection** | Data leakage. Features would get selected because they happen to fit the test set. | Greedy selection on validation set (20% of training = 1,360 rows). Report final numbers on test set. |
| **Claiming "removing X features saves compute"** | Our models train in seconds. Feature-count parsimony doesn't save meaningful compute for LR/XGBoost. | Frame parsimony claims around statistical efficiency (lower variance) and interpretability, NOT compute. |
| **Dropping `spread` feature to prove it's critical** | `spread` is the target's input-time analogue (target = Δspread). Dropping it is a trivial "look how much it matters" result. | Keep `spread` always. Ablate only the engineered features ON TOP of the basic spread/mid. |
| **Ablating and NOT retraining** | Masking feature values at test time (e.g., setting to mean) and predicting — this is a different ablation protocol (BASED-XAI-style) and has known biases. | Retrain from scratch for each ablation. Takes 15 sec × 10 runs = trivially fast. |

### 3.4 Dependencies

- **Upstream:** Full-feature LR + XGBoost baselines (complete), SHAP rankings (complete).
- **Downstream:** Nothing. Self-contained.

---

## 4. Capability: Ensemble Exploration

**Why this capability exists:** In live deployment, we already use an LR+XGBoost ensemble with category-aware entry filter and concordance filter. The paper has not yet documented this systematically. Ensemble exploration (a) validates the production choice, (b) tests whether adding LSTM or TFT improves it, and (c) gives the paper an "evidence-based production ensemble" finding.

**IMPORTANT:** The v1.0 FEATURES.md listed "ensemble of all models" as an anti-feature (to preserve the complexity-vs-performance story). That remains true: the ensemble is NOT the centerpiece. It's a supplement that (a) matches the live deployment and (b) tests whether model diversity helps at the margin. The anti-feature framing is still correct FOR THE MAIN CLAIM, but the ensemble is valuable for the deployment narrative.

### 4.1 Table Stakes

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Four canonical ensemble variants compared** | Paper should compare: (a) LR alone, (b) LR + XGBoost equal-weighted average, (c) LR + LSTM equal-weighted, (d) LR + XGBoost + LSTM majority vote (sign consensus). These are the academic-standard ensemble strategies. | LOW | 4 ensemble variants × 1 evaluation each = 4 rows in a new table. |
| **Equal-weighted averaging as the baseline combiner** | Sklearn VotingRegressor default. Academic literature (Dong-Keon 2024, RoyalSociety 2024) starts here before claiming stacking adds value. | LOW | `np.mean([pred_lr, pred_xgb], axis=0)`. |
| **Majority-vote on sign** (trade only if all ensemble members agree on direction) | This IS the live deployment's "concordance filter." Must match what's in production. | LOW | `if sign(pred_lr) == sign(pred_xgb): trade(avg_pred)`. |
| **Evaluation at the same 2pp fee and same test set** | Protocol parity. Ensemble isn't evaluated under different rules than individual models. | LOW | Single ensemble table appended to Table 2. |
| **Explicit statement that ensemble matches live deployment** | Connects §5 (backtest results) to §4.4 (live system architecture). Tells reviewers "this is what actually runs." | LOW | 1 sentence at the top of the ensemble subsection. |
| **At least one ensemble column showing it's marginal improvement** | Honest result is "ensemble wins by $X-Y range in absolute P&L," not "ensemble is dramatically better." If it's dramatic, something's wrong. | LOW | Expected: 0-3% P&L improvement over best single model. |

### 4.2 Differentiators

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Stacking regression with held-out meta-model** | The academic-stronger alternative to equal-weighting. Train a Ridge meta-model on out-of-fold predictions from LR, XGBoost, LSTM. Cite MDPI 2025 heterogeneous stacked ensembles. | MEDIUM | ~30 lines of sklearn `StackingRegressor`. Requires cross-validation folds that respect temporal order. |
| **Disagreement-gated ensemble** — only trade when base models disagree on MAGNITUDE | Novel: the concordance filter already handles sign. Add a magnitude-disagreement filter: only trade when base models' magnitudes span > X. High disagreement = high uncertainty = lower conviction = skip trade. Counterintuitive, but aligns with ensemble uncertainty estimation literature. | MEDIUM | Add one extra column to ensemble table. Empirically test whether disagreement helps. |
| **Per-category ensemble winner** | Extend Finding 21 (LR wins 5/7, XGB wins crypto): which ensemble wins each category? If LR+XGB ensemble wins crypto (where XGB alone wins) and inflation (where LR alone wins), that's the "diversification wins" finding. | LOW | Rerun per-category breakdown with ensemble predictions. |
| **Ensemble weight sensitivity** — sweep LR-weight from 0.0 to 1.0 in steps of 0.1, plot P&L | Shows the ensemble is not cherry-picked. If the 0.5/0.5 weighting is near-optimal, that's evidence for equal-weighting. If a skewed weight wins, report honestly and explain. | LOW | 11 evaluations, each ~2 sec. Single plot: weight on x-axis, P&L on y-axis. |
| **Rank correlation between ensemble members** (Spearman ρ of predictions) | Ensembles help most when members are accurate AND decorrelated. If LR and XGBoost have ρ = 0.95, ensembling won't help much — that empirically explains why the ensemble gain is small. | LOW | Scalar in the methodology. |
| **Comparison to adding TFT to the ensemble** (if TFT is available) | "LR + XGBoost + LSTM + TFT majority vote" as a 4-model ensemble. Tests whether architectural diversity (tree + RNN + transformer) helps. | LOW (if TFT done) | Depends on Capability 1 being complete. |

### 4.3 Anti-Features

| Anti-Feature | Why Avoid | Alternative |
|--------------|-----------|-------------|
| **Framing ensemble as the main contribution** | v1.0 FEATURES.md already said this muddies the complexity-vs-performance story. If an ensemble wins, you can't attribute performance to a specific tier. | Frame ensemble as "what our deployment uses, and here's the margin vs best single model." Keep Tier-based comparison (Table 2) as the centerpiece. |
| **Optimizing ensemble weights on the test set** | Look-ahead. Ensemble weights are hyperparameters. | Optimize on validation set (last 20% of training data). Report on test set. |
| **Training a deep meta-model (neural stacker)** | 4 base predictions → neural network → single output. Tiny input, many parameters, guaranteed overfit on 6,802 rows. | Ridge regression as meta-model (1-2 hyperparameters). |
| **Including PPO in the ensemble** | PPO's backtest is -$7,724. Adding it will destroy the ensemble. | Exclude PPO explicitly. State in methodology: "Tier 3 models excluded due to catastrophic backtest performance (Table 2)." |
| **Retraining the ensemble members on different splits** | Confounds ensemble benefits with training-data variance. | All ensemble members train on the same training set and predict on the same test set. Differences come from architecture, not data. |
| **Majority-vote on magnitude instead of sign** | Majority vote is a classifier concept. For regression magnitudes, averaging is the right aggregation, not voting. | Sign-vote for entry gating (concordance filter); average for the actual trade size. |
| **Retroactively "tuning" the live concordance filter based on backtest** | Subtle look-ahead. Live filter was deployed in v1.0 before v1.1 analysis. | Document the live filter exactly as deployed. Any v1.1 analysis is post-hoc characterization, not tuning. |

### 4.4 Dependencies

- **Upstream:** LR + XGBoost + LSTM baselines (complete). TFT (Capability 1) if adding to 4-model ensemble.
- **Downstream:** Nothing. Ensemble is an additional paper section, not a precondition for anything else.

---

## 5. Capability: 250-Bar Data-Scaling Checkpoint

**Why this capability exists:** Finding 22 flagged this as "pending." The scaling curve in Table 5 already has 50, 100, 250, 500, 1000, 2000 bars/pair — but only 50 and 100 are populated with GRU/LSTM numbers; 250+ are plateau because the training cap is reached. The actual "new data point" here is filling in the 250-bar GRU/LSTM row (and possibly a TFT row) so the paper has a legitimate THIRD scale point rather than a plateau.

**Key clarification from PAPER_DRAFT.md:** The current Table 5 already shows LR and XGBoost plateauing at 100+ bars. The value of the 250-bar checkpoint is specifically for GRU/LSTM (and optionally TFT), because those are the models that SHOULD benefit from more sequence data.

### 5.1 Table Stakes

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **GRU and LSTM trained on 250-bar/pair data** | Current Table 5 shows GRU at +$186.67 and LSTM at +$182.76 at 100 bars. The 250-bar checkpoint is the first real test of whether sequence models benefit from more data. | MEDIUM | Wait for auto-retrain to cross 250-bar threshold. ETA 12-24h per FINDINGS.md. |
| **Same train/test split protocol across all three scale points** | Checkpoint system (§4.4) already ensures this: train on first N bars/pair, test on the last 20%. | LOW | Already implemented. |
| **Ranking comparison at each scale point** (Does XGBoost > LR > LSTM > GRU hold at 50, 100, 250?) | This is the core claim: ranking invariance across scale. Table 5 in PAPER_DRAFT.md already states this for 50/100; 250 adds a third point. | LOW | Add to existing Table 5. |
| **Update Finding 22 to reflect actual 250-bar data** | Currently marked "pending" in FINDINGS.md. After the checkpoint fires, update to actual findings. | LOW | Automated (already implemented per PAPER_DRAFT.md §4.4). |
| **Honest treatment of the plateau** | If LR/XGBoost are flat at 250 (per Table 5 trend), that's the honest finding. Don't hide it. | LOW | One paragraph in §5.4. |

### 5.2 Differentiators

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Log-log scaling plot** (training rows on x-axis, P&L or RMSE on y-axis) | Standard neural scaling-law presentation. Kaplan et al. 2020 and Hoffmann et al. 2022 established log-log plots as the canonical view. Even at our small scale, the format is correct. | LOW | `ax.set_xscale('log'); ax.set_yscale('log')`. Matplotlib one-liner. |
| **Error bars on scaling points** | Bootstrap 1000 resamples per scale point to compute 95% CIs. Error bars let reviewers see if the ranking differences are statistically significant. | MEDIUM | Already doing bootstrap for Finding 17 (Sharpe CIs). Extend to per-scale P&L. |
| **Extrapolation to the "if 500 bars" regime** | Fit a power law to the 3 scale points, extrapolate. Explicitly state "this is extrapolation, not data." Paper §6.2 already does this informally; make it quantitative. | LOW | `scipy.optimize.curve_fit` with `y = a * x^b`. Cite Kaplan 2020. |
| **Ranking invariance statement with statistical backing** | "Across all 3 scale points, XGBoost > LR > LSTM > GRU, with ranking confidence p < 0.05 by Spearman rank correlation test." | MEDIUM | Spearman rank test on the per-model rankings across scales. |
| **Per-pair bars histogram** | Show the distribution of bars/pair in the live dataset. Useful context: "At the 250-bar checkpoint, only X% of pairs have enough data to be in training." | LOW | One histogram figure. |
| **Connection to Finding 6.2 (model headroom predictions)** | The paper already has detailed per-model headroom predictions (LR: +$260-300, XGB: +$300-380, RNN: likely passes XGB). The 250-bar checkpoint is a falsifiable test of these predictions. | LOW | 2-3 sentences comparing predicted vs actual at the 250-bar checkpoint. |

### 5.3 Anti-Features

| Anti-Feature | Why Avoid | Alternative |
|--------------|-----------|-------------|
| **Waiting for 500, 1000, 2000 bar checkpoints before paper submission** | Deadline is Apr 27. 250 bars ETA 12-24h; 500+ is weeks away. | 250-bar checkpoint is the goal. Mention 500+ as future work. |
| **Claiming "ranking would flip at X bars" without evidence** | Our scaling curve plateaus because training-set cap is reached, not because of a rank-flip. Claims beyond the data range are extrapolation at best. | Report observed ranking. Note in §6 that the plateau is due to training-cap saturation, not a fundamental ranking. |
| **Presenting scaling curves without error bars** | Reviewers can't distinguish signal from noise. At 6,802 rows × 1,549 test rows, a $5 P&L difference has bootstrap CI width of roughly ±$30. | Error bars on every point. |
| **Adding a 4th point just for aesthetics** | A 4th point at 500 bars that's identical to 250 is visually redundant and signals "we tried to pad." | 3 honest points: 50, 100, 250. |
| **Fitting a scaling law to 3 points and claiming statistical validity** | 3 points can't rigorously fit a 2-parameter power law. Any fit is descriptive only. | Fit is for visual extrapolation. State explicitly: "Visual extrapolation only — 3 data points is below threshold for robust scaling-law fitting." Cite Hoffmann 2022 which uses dozens of points. |
| **Claiming our scaling curve refutes Kaplan/Chinchilla scaling laws** | Scaling laws are about parameters/data/compute in neural networks. Our XGBoost/LR results aren't in scope. | Cite scaling laws as conceptual framework. Don't claim refutation. |

### 5.4 Dependencies

- **Upstream:** Auto-retrain system firing (running), checkpoint threshold hit (pending).
- **Downstream:** Paper Table 5 (append row), §5.4 (update).

---

## 6. Capability: Paper Finalization (Publication-Quality Figures, Citations, Abstract Rewrite)

**Why this capability exists:** The paper draft is v2 with 8 tables and 9 figures. v1.1 is the final polish: color-blind figures, proper matplotlib style, consistent citations, tight abstract. This is the difference between "DS340 paper" and "submission-ready paper."

### 6.1 Table Stakes

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Color-blind-safe palette on all figures** | ~5% of reviewers have color vision deficiencies. Default matplotlib `tab10` is known problematic for deuteranopia. | LOW | One-line fix: `sns.set_theme(context='paper', style='ticks', palette='colorblind')` OR `plt.style.use('seaborn-v0_8-colorblind')`. Re-run figure scripts. |
| **Variable line styles + markers** (solid/dashed/dotted + circle/triangle/square) in addition to color | Shape redundancy ensures figures are legible in grayscale (printing) and for monochrome color deficiencies. | LOW | One `linestyles=[...]` and `markers=[...]` param. Apply globally via rcParams. |
| **Font size consistency** (12pt labels, 10pt ticks, 14pt titles — or whatever the submission spec says) | Mismatched font sizes signal sloppiness. | LOW | `plt.rcParams.update({'font.size': 12, 'axes.labelsize': 12, 'xtick.labelsize': 10})`. |
| **Figure DPI ≥ 300 for print** (savefig dpi=300 or vector PDF) | Low-res figures are a red flag in submissions. | LOW | `plt.savefig('fig.pdf')` (vector) or `plt.savefig('fig.png', dpi=300)`. |
| **All 9 figures referenced in text** | Every figure must be introduced and discussed. Unreferenced figures get cut. | LOW | Audit pass: Ctrl+F "Fig. N" in PAPER_DRAFT.md for N=1..9. |
| **Consistent citation format** (all citations in one style — e.g., Chicago or IEEE, not mixed) | Mixed citation styles in a single paper signal hasty writing. | LOW | Pick one style, apply throughout. The current draft mixes "Amihud (2002)" and "Burgi et al (2026)" — pick one. |
| **Abstract under 250 words** (the current draft's abstract is 412 words — too long) | Standard conference/journal abstract limit. DS340 might not require this, but submission-ready papers do. | LOW | Tighten to 200-250 words. Keep the 5-regime framing, headline numbers, and negative result. Cut the detailed breakdowns. |
| **Axis labels with units** (every axis has a label, units specified when relevant) | Unlabeled axes are the #1 figure mistake. | LOW | Audit pass on all 9 figures. |
| **No unreadable legends** (legends placed to avoid data overlap, font not too small) | Occluded legends frustrate readers. | LOW | `ax.legend(loc='best')` or explicit positioning. |

### 6.2 Differentiators

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Uniform style via one `figurestyle.py` module** | All figures import the same style functions. Guarantees consistency and lets readers recognize "this is Ian & Alvin's paper." | LOW | 50-line module that wraps `sns.set_theme` and common `rcParams`. Every figure imports it. |
| **LaTeX font rendering in figures** (`rcParams['text.usetex'] = True`) | Math symbols and labels render exactly as in the PDF body. Conference papers that use matplotlib default fonts look mismatched. | MEDIUM | Requires LaTeX installed; can fall back to `mathtext.fontset = 'cm'`. |
| **Subplot layouts with shared axes for comparison figures** | Walk-forward P&L (11 windows) across models → 4-panel subplot, same x-axis, same y-scale. Easier to compare than 4 separate figures. | LOW | `fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)`. |
| **Captions that stand alone** (reader can understand the figure from caption alone, without body text) | Best-practice for journal papers. Reviewers often skim figures + captions first. | LOW | Rewrite each caption to be 2-4 sentences of standalone explanation. |
| **Abstract structured as PROBLEM / METHOD / RESULT / CONTRIBUTION** | Most-cited abstract structure in ML papers. | LOW | 4-sentence skeleton then fill in. |
| **Named figure style in paper methodology** | "All figures use the seaborn-colorblind palette (Waskom 2021) with variable line styles for accessibility." One sentence, significant polish signal. | LOW | Add citation to Waskom 2021 (seaborn). |
| **Consistency check across all tables** | All 8 tables use the same number format (e.g., `+$XXX.XX` for P&L, `X.XXX` for Sharpe to 3dp). | LOW | Table-by-table audit. |
| **pypubfigs or similar publication-quality palette library** | Academic palette pack with citations (Steenwyk 2020). Signals attention to figure quality. | LOW | `pip install pypubfigs`. Set once. |

### 6.3 Anti-Features

| Anti-Feature | Why Avoid | Alternative |
|--------------|-----------|-------------|
| **Overly dense figures** (12 lines on one plot, tiny markers, overlapping) | Unreadable. Reviewers will skip them. | Max 4-6 series per plot. Use subplots for more. |
| **Rainbow palettes** (jet, rainbow, Set3) | Perceptually non-uniform, known accessibility problem. jet is explicitly deprecated in matplotlib. | viridis (sequential), colorblind (categorical). |
| **Using color alone to distinguish categories** | Fails for ~5% of reviewers and fails in grayscale print. | Color + linestyle + marker (triple encoding). |
| **3D plots for 2D data** | Decorative, not informative. 3D distorts perception. | 2D with color/size encoding for the third dimension. |
| **Screenshots of terminal output as figures** | Low-res, unstyled, signals hasty preparation. | Regenerate as matplotlib figure or proper table. |
| **Inconsistent model name abbreviations** (LR vs Linear Regression vs LR baseline) | Reviewers tracking the story get confused. | Pick one (probably "LR") and use throughout. |
| **Pasting raw pandas DataFrame output as tables** | Unstyled, no borders, variable spacing. | Use `df.to_latex()` (or markdown with proper alignment) and polish by hand. |
| **Abstract that buries the lede** ("We study prediction markets...") | Review triage often reads first 3 sentences. If the headline finding isn't there, the paper gets deprioritized. | First sentence: problem. Second: method. Third: headline result ("complexity is a liability"). |
| **Citations without DOIs or URLs** for recent (post-2020) papers | Harder for reviewers to verify. | Include DOI/arXiv ID for every post-2020 citation. |
| **Forgetting to anonymize if required** | Submission violation. | Check submission requirements. DS340 likely doesn't require anonymization, but confirm. |
| **Leaving TODO/TBD markers in the final draft** | Amateur. | Final pass with Ctrl+F "TODO", "TBD", "???", "XXX". |
| **Not proofreading the final PDF output** | LaTeX/markdown rendering sometimes breaks figures or citations silently. | Print to PDF, read cover-to-cover as a reader would. |

### 6.4 Dependencies

- **Upstream:** All results finalized (Capabilities 1-5). Can work on style early but final figure values depend on content.
- **Downstream:** Submission to Prof. KG and TA. End of project.

---

## Feature Dependencies (Cross-Capability)

```
Capability 1: TFT
    |
    +--> (optional) feeds into Capability 4: Ensemble (as 4th/5th member)
    |
Capability 2: Live vs Backtest Reconciliation
    (no dependencies — leaf)
    |
Capability 3: Feature Ablation
    (no dependencies — leaf)
    |
Capability 4: Ensemble Exploration
    (requires baseline LR/XGBoost/LSTM — all complete from v1.0)
    (benefits from TFT if Capability 1 is complete)
    |
Capability 5: 250-Bar Checkpoint
    (depends on auto-retrain hitting threshold — passive wait)
    |
Capability 6: Paper Finalization
    (depends on ALL other capabilities being complete to have final numbers)
    (style/layout work can start in parallel)
```

### Parallelization Strategy

- **Week of Apr 16 (now):** Start Capabilities 1 (TFT) + 3 (ablation) + 2 (live reconciliation) in parallel. Wait for Capability 5 (250-bar checkpoint) passively.
- **Week of Apr 20:** Capability 4 (ensemble) once TFT is done. Capability 6 (paper polish) once main numbers are frozen.
- **Week of Apr 27:** Final submission.

### Critical Path

The critical path for submission is `Capability 6 → final PDF`. Everything else feeds into it. The longest parallel chain is `Capability 1 (TFT) → Capability 4 (ensemble w/ TFT) → Capability 6 (paper)`, so that's the chain to start ASAP.

---

## MVP Definition for v1.1

### Ship with (must-complete before Apr 27)

- [ ] **TFT** trained with small-data hyperparameters, evaluated identically to GRU/LSTM, one row added to Table 2, VSN attention figure added (Capability 1).
- [ ] **Live vs backtest reconciliation** — summary comparison table, tracking error, exit-reason attribution, category-level divergence, honest caveats (Capability 2).
- [ ] **Feature ablation** via LOGO on 5 groups, delta-P&L table with bootstrap CIs, minimum-sufficient-subset discussion (Capability 3).
- [ ] **Ensemble baseline** — LR+XGBoost equal-weighted and majority-vote variants documented to match the live deployment, 1 new table (Capability 4).
- [ ] **250-bar scaling checkpoint** filled in to Table 5 for GRU/LSTM (passive wait on auto-retrain) (Capability 5).
- [ ] **Paper polish** — color-blind palette, abstract rewrite to <250 words, citation style consistency, caption standalone-ness (Capability 6).

### Defer if time-constrained

- [ ] TFT in 4-model ensemble (Capability 4 differentiator) — only if Capability 1 is clean.
- [ ] Stacking regression meta-model (Capability 4 differentiator).
- [ ] P&L attribution decomposition FRTB-style (Capability 2 differentiator).
- [ ] Greedy forward feature selection (Capability 3 differentiator).

### Out of scope for v1.1

- Adding PatchTST, Autoformer, TimesNet (mentioned in Future Work only).
- LOFO ablation on all 59 features (LOGO on 5 groups is enough).
- Real-money trading.
- External (news, sentiment) features.
- Additional platforms beyond Kalshi + Polymarket.

---

## Priority Matrix

| Capability | Table-stakes User Value | Implementation Cost | Priority | Rationale |
|-----------|------------------------|---------------------|----------|-----------|
| C1: TFT | HIGH | MEDIUM | P1 | Completes proposal suite, adds 3rd architecture family |
| C2: Live reconciliation | VERY HIGH | MEDIUM | P1 | Unique paper asset; almost no student paper has live data |
| C3: Feature ablation | HIGH | LOW | P1 | Easy win, directly tests the "features carry signal" claim from §6.1 |
| C4: Ensemble | MEDIUM | LOW | P2 | Documents live deployment; not the main claim |
| C5: 250-bar checkpoint | HIGH | LOW (passive) | P1 | Passive wait; zero cost to keep in scope |
| C6: Paper polish | VERY HIGH | LOW-MEDIUM | P1 | Submission-ready vs student-draft is massive grade delta |

**Priority key:**
- P1: Must have for submission Apr 27.
- P2: Should have if time permits.
- P3: Nice to have — moved to future-work in the paper.

---

## Confidence Notes

- **HIGH confidence** on Capabilities 1, 3, 4, 5, 6 — these are well-covered in academic literature (2024-2026 papers on TFT small-data, ensemble methods, scaling curves, figure conventions) and the existing v1.0 codebase already has most of the infrastructure.
- **MEDIUM-HIGH confidence** on Capability 2 — live vs backtest reconciliation methodology is well-established in quantitative finance (FRTB P&L attribution, QuantConnect reconciliation docs), but our specific application to prediction markets is a first for the project. The methodology is borrowed cleanly; the application is novel-but-straightforward.
- **LOW risk** on TFT training — the main risk is training-time budget (TFT is slow), not correctness. PyTorch Forecasting is well-documented.
- **LOW risk** on feature ablation — LR/XGBoost retrain in seconds; 10 variants trivially affordable.

---

## Sources

### Academic literature (2024-2026)
- A Controlled Comparison of Deep Learning Architectures for Multi-Horizon Financial Forecasting — Evidence from 918 Experiments ([arXiv 2603.16886](https://arxiv.org/html/2603.16886)) — confirms TFT ranks below ModernTCN/PatchTST on financial data, but also confirms all transformer models outperform naive baselines in financial forecasting.
- Neural Scaling Laws for Deep Regression ([arXiv 2509.10000](https://arxiv.org/html/2509.10000v1)) — framework for scaling curve presentation.
- Ensemble Learning: Comparative Analysis of Hybrid Voting and Ensemble Stacking ([arXiv 2509.02826](https://arxiv.org/abs/2509.02826)) — 2025 survey of voting vs stacking.
- Stock Price Prediction Using a Stacked Heterogeneous Ensemble ([MDPI 2025](https://www.mdpi.com/2227-7072/13/4/201)) — heterogeneous stacking with LR+LSTM+XGBoost+Transformer.
- An ensemble approach integrating LSTM and ARIMA models ([Royal Society Open Science 2024](https://royalsocietypublishing.org/doi/10.1098/rsos.240699)) — ensemble baseline methodology.
- AttentionDrop: A Novel Regularization Method for Transformer Models ([arXiv 2504.12088](https://arxiv.org/html/2504.12088)) — overfitting mitigation for TFT.
- Grouped feature importance and combined features effect plot ([Springer 2022](https://link.springer.com/article/10.1007/s10618-022-00840-5)) — LOGO framework.
- Explainability and importance estimate of time series classifier ([Nature Scientific Reports 2025](https://www.nature.com/articles/s41598-025-17703-w)) — LOGO for time series.
- Iterative feature exclusion ranking for deep tabular learning ([Springer 2025](https://link.springer.com/article/10.1007/s10115-025-02616-x)) — iterative LOFO methodology.
- Explaining neural scaling laws ([PNAS](https://www.pnas.org/doi/10.1073/pnas.2311878121)) — scaling-law plot conventions.
- Revisiting Scaling Laws for Language Models ([ACL 2025](https://aclanthology.org/2025.acl-long.1163.pdf)) — log-log plotting conventions.

### Documentation and practitioner guides
- PyTorch Forecasting TemporalFusionTransformer docs ([v1.4.0 tutorial](https://pytorch-forecasting.readthedocs.io/en/v1.4.0/tutorials/stallion.html)) — small-data hyperparameter defaults.
- PyTorch Forecasting TFT API ([docs](https://pytorch-forecasting.readthedocs.io/en/v1.0.0/api/pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer.html)) — parameter ranges.
- QuantConnect Live Trading Reconciliation ([docs](https://www.quantconnect.com/docs/v2/writing-algorithms/live-trading/reconciliation)) — industry-standard reconciliation methodology.
- Portfolio Optimization Backtesting ([slides](https://portfoliooptimizationbook.com/slides/slides-backtesting.pdf)) — academic backtesting framework.
- FRTB P&L Attribution ([Mazars](https://financialservices.forvismazars.com/attribution-tests-frtb-framework/)) — regulatory-grade P&L attribution framework.
- KX P&L Attribution glossary ([KX](https://kx.com/glossary/pl-attribution-analysis-in-finance/)) — clean P&L / model P&L / unexplained P&L decomposition.
- scikit-learn ensemble documentation ([sklearn 1.8.0](https://scikit-learn.org/stable/modules/ensemble.html)) — VotingRegressor, StackingRegressor canonical usage.
- LOFO Importance ([GitHub](https://github.com/aerdem4/lofo-importance)) — reference implementation of leave-one-feature-out.
- Interpretable ML Book §24 LOFO ([Molnar](https://christophm.github.io/interpretable-ml-book/lofo.html)) — theoretical framework for LOFO vs LOGO.
- seaborn color palettes ([docs](https://seaborn.pydata.org/tutorial/color_palettes.html)) — colorblind-safe categorical palettes.
- seaborn-colorblind palette ([docs](https://viscid-hub.github.io/Viscid-docs/docs/dev/styles/seaborn-colorblind.html)) — built-in accessibility palette.
- pypubfigs ([GitHub](https://github.com/JLSteenwyk/pypubfigs)) — publication-quality color palettes.
- Publication-Quality Plots in Python with Matplotlib ([Schuch 2025](https://www.fschuch.com/en/blog/2025/07/05/publication-quality-plots-in-python-with-matplotlib/)) — 2025 best practices.
- POS SISSA color-blind publication guidelines ([PoS](https://pos.sissa.it/guidelines.pdf)) — official conference guidelines for color-blind-safe figures.

### Internal references (v1.0 history)
- `.planning/research/FEATURES.md` (v1.0) — foundational features, retained as history.
- `FINDINGS.md` — 22 chronological findings that anchor what "already known" means for v1.1.
- `PAPER_DRAFT.md` — v2 paper structure (8 tables, 9 figures) defines where v1.1 outputs land.

---

*Feature research for: v1.1 "Extended Evidence & Submission" milestone*
*Researched: 2026-04-16*
*Next consumer: REQUIREMENTS.md author (turn table-stakes/differentiators into REQ-IDs) and gsd-roadmapper (map each capability to a phase).*
