# Technology Stack -- v1.1 Additions

**Project:** Kalshi vs. Polymarket Price Discrepancies (DS340 Final)
**Milestone:** v1.1 (Extended Evidence & Submission)
**Researched:** 2026-04-16
**Overall Confidence:** HIGH on PyPI versions (verified via PyPI in April 2026); MEDIUM on configuration recommendations (from docs + community posts, no empirical benchmark on this exact dataset).

> **This document complements, not replaces, the v1.0 STACK.md.**
> v1.0 documented the full baseline stack (Python 3.12 + PyTorch 2.10 + pytorch-forecasting 1.6.1 + sklearn 1.8 + SB3 + SHAP + matplotlib). That stack is validated and working. This document is scoped to the **new v1.1 capabilities**: TFT training on small data, feature-ablation tooling, live-vs-backtest reconciliation, publication figures, and ensemble exploration. Do NOT re-research anything in v1.0 STACK.md.

---

## Environment Drift Since v1.0 (VERIFY BEFORE PROCEEDING)

The currently active venv at `.venv/` has shifted away from the v1.0 snapshot. Before v1.1 work begins, phase planning must reconcile these facts:

| v1.0 STACK.md said | Current venv reality (2026-04-16) | Action |
|---|---|---|
| Python 3.12.12 | **Python 3.14.3** (`.venv/bin/python`) | Pin one; 3.14 changes string handling and thread model. Verify torch/pytorch-forecasting wheels exist for 3.14. |
| torch 2.10.0 | **torch 2.11.0** | Minor version, likely fine. Re-run MPS segfault test documented in Phase 5 decisions. |
| pytorch-forecasting 1.6.1 installed | **Not installed** | Must re-install for TFT work. |
| pytorch-lightning 2.6.1 installed | **Not installed** (transitively pulled by pytorch-forecasting on install) | Will return via pytorch-forecasting dep. |
| stable-baselines3 2.7.1 installed | **Not installed** | Only needed if re-running PPO for comparison; otherwise deferrable. |
| xgboost 3.2.0 | **xgboost 3.2.0** | Unchanged, good. |
| scikit-learn 1.8.0 | **scikit-learn 1.8.0** | Unchanged, good. |

**Risk:** Python 3.14 is newer than pytorch-forecasting's declared support window (PyPI metadata says `Python <3.15, >=3.10` -- so technically 3.14 is covered, but wheels may not exist for every transitive dep). Before Phase-8 planning, do `.venv/bin/pip install pytorch-forecasting==1.7.0 --dry-run` and confirm resolver succeeds. If it fails, pin Python 3.12 via a fresh venv.

---

## Summary Verdict

**Add these five libraries, nothing more:**

1. `pytorch-forecasting==1.7.0` + `pytorch-lightning` (transitive) -- for TFT (already validated in v1.0 research, needs re-install)
2. `quantstats==0.0.81` -- for live-vs-backtest tearsheet (one function does exactly what we need)
3. `SciencePlots==2.2.1` -- for publication figures (IEEE/Nature styles, single line to apply)
4. `empyrical-reloaded==0.5.12` -- only if quantstats internals prove inadequate (fallback, probably skip)
5. `pip install -e .` your own `src/` as an editable package so ablation scripts can import without PYTHONPATH hacks

**Do NOT add:**
- `neuralforecast` / `darts` / `pytorch-tcn` / `PatchTST` (standalone) -- over-engineers the TFT story, more libs to debug than paper pages to write
- `pyfolio-reloaded` -- pinned to old matplotlib APIs, quantstats does 90% of the same thing with cleaner output
- `tueplots` -- overlaps with SciencePlots; pick one, and SciencePlots has more inertia + IEEE/Nature styles
- `lofo-importance` -- sklearn's `permutation_importance` + a small leave-one-group-out loop is 30 lines and auditable; the extra dep isn't worth it for a class paper
- `mlforecast` -- its training API is opinionated in a way that would force refactoring our existing `BasePredictor` interface; ensembles are easier as custom code
- `optuna` -- hyperparameter tuning is premature for v1.1 given timeline

---

## Stack Additions (Detailed)

### 1. TFT Training on Small Data (6,802 rows, ~47-bar avg sequences)

**Primary library:** `pytorch-forecasting==1.7.0` (re-install; was 1.6.1 in v1.0)

**Why this and not an alternative:**
- `pytorch-forecasting` already has our `TimeSeriesDataSet` plumbing in the Phase-3 data pipeline. Switching to `neuralforecast` (Nixtla) or `darts` would require re-formatting data into their native schemas -- a week of work we don't have before April 27.
- The TFT implementation in `pytorch-forecasting` 1.7.0 is mature (the new v2 TFT module added in 1.5 is available alongside the legacy class, so we can use whichever has better small-data behavior).
- `neuralforecast` is a strong library and actively maintained (v3.1.7 as of April 2026), but the integration cost is the problem, not the library.

**Configuration for our data (6,802 train rows, 978-row test, 47-bar avg, 31 features):**

| Parameter | Recommended value | Rationale |
|---|---|---|
| `hidden_size` | 8-16 (start at 8) | The 1.7.0 default is 16 for legacy TFT, 64 for TFT v2. For small data, use 8. A known GitHub issue (#1322) documents that hidden_size=160 on a 40-60k row dataset required 512GB RAM. We are an order of magnitude smaller; stay under 32. |
| `hidden_continuous_size` | 8 (and <= `hidden_size`) | Standard recommendation in the pytorch-forecasting API reference. |
| `attention_head_size` | 1 | Smaller datasets cannot afford multi-head attention without overfitting. |
| `lstm_layers` | 1 | More layers = more params; not justified at 6.8k rows. |
| `dropout` | 0.1-0.3 | Higher than regression baselines (0.0). This is the primary regularization. |
| `learning_rate` | 1e-3 (Adam) | Standard for TFT; use `LearningRateFinder` from Lightning if unsure. |
| `max_encoder_length` | 12-24 bars | Matches our GRU/LSTM lookback (Phase 5 found 2 bars marginally beat 6; TFT should test 12 and 24 to cover its "gets better with more context" nature without drowning in padding). |
| `max_prediction_length` | 1 bar | Single-step forecast, matches all other models for fair comparison. |
| Loss | `QuantileLoss` | Returns prediction intervals, useful for the trading layer (fee threshold = conservative trade only if even the 10th percentile predicts profitable). Alternatively `RMSE` for strict apples-to-apples with Tier-1/2. Start with `RMSE` for the comparison table, add a `QuantileLoss` variant as a paper-section bonus if time permits. |
| Batch size | 32-64 | MPS memory on our laptops can handle this. Lightning Trainer accumulates if needed. |
| Epochs | 30-50 with `EarlyStopping(patience=5)` | Monitor val_loss. |

**Param-to-sample math:**
- TFT with `hidden_size=16`, 1 LSTM layer, 31 features: approx. 4,000-8,000 params (rough order of magnitude).
- 6,802 training rows / 4,000 params = **1.7 samples per parameter**.
- Rule of thumb from deep-learning literature: want >= 10 samples/param for reliable training; minimum for neural nets is often cited as >= 50 samples per 1 feature.
- **This is precisely why TFT was deferred in v1.0 (param-to-sample ratio 1.9 documented in Phase-5 decisions).**
- The honest path: train TFT with `hidden_size=8`, 0.3 dropout, early stopping, expect it to tie GRU/LSTM or lose slightly, and write up the result as "at this dataset scale, transformers do not beat RNNs -- consistent with the complexity-vs-performance thesis."

**Alternatives considered and rejected:**

| Alternative | Why not for v1.1 |
|---|---|
| `neuralforecast` PatchTST | Would require re-plumbing data. PatchTST is designed for long-horizon forecasting; our task is next-bar prediction. Mismatch. |
| `darts` TFTModel | Double-implementation risk; our code is already pytorch-forecasting native. |
| `pytorch-tcn` (TCN model) | Adds another model tier (Tier 2b?) that isn't in PROJECT.md requirements. Scope creep. |
| Custom transformer from scratch | Re-implementing attention correctly on 11 days to submission is malpractice. |

**Key decision point for the plan-phase step:**
If TFT fails to train stably (val_loss diverges within 10 epochs even at hidden_size=8), drop it and write up the negative result. Do not spend >2 days debugging TFT on a dataset this small. The paper's complexity-vs-performance framing already supports "TFT attempted, failed to beat RNNs due to sample size" as a valid finding.

**Confidence:** HIGH on library choice (already using it). MEDIUM on specific hyperparameters (community recommendations, not empirical on our data).

---

### 2. Feature Ablation Study

**Primary tool:** `sklearn.inspection.permutation_importance` (already installed in scikit-learn 1.8.0) + a custom 30-line leave-one-group-out loop.

**Why not a dedicated library:**
- `lofo-importance` (Leave-One-Feature-Out) is the closest third-party alternative, but it's a thin wrapper over a cross-validated fit loop. Rolling our own keeps the logic auditable in the paper's methodology section.
- `eli5` has a `PermutationImportance` class, but scikit-learn's native implementation is the industry default as of 1.8 and works model-agnostically (tree, linear, neural -- just need a `predict` method).

**Recommended approach for v1.1 feature ablation:**

```
Feature groups (defined in src/features/):
- Kalshi microstructure (5-7 features): kalshi_vwap, kalshi_volume, kalshi_volatility, etc.
- Polymarket microstructure (5-7 features): polymarket_vwap, polymarket_volume, etc.
- Spread-derived (5-7 features): spread_mean, spread_volatility, spread_velocity, etc.
- Bid-ask features (3-5 features)
- Time-based (2-3 features)

Two parallel studies:
1. Permutation importance (sklearn.inspection.permutation_importance, n_repeats=10):
   - Model-agnostic, handles correlated features weakly but fine for paper.
   - Apply to XGBoost (fastest), optionally GRU/LSTM via sklearn-compatible wrapper.
   - Time: ~minutes per model.

2. Leave-one-group-out (custom ~30 lines):
   - For each feature group G:
     - Drop columns in G from train+test.
     - Retrain XGBoost (and optionally LR + GRU).
     - Record RMSE delta vs. full-feature baseline.
   - Produces the clean "minimum feature set" chart for the paper.
   - Time: ~hours for Tier 2 retraining; XGBoost only is fine for v1.1.
```

**Per v1.0 Phase-07 decisions: `polymarket_vwap` already dominates SHAP importance (0.138 vs. 0.016 next).** The ablation should confirm this structurally: dropping the Polymarket microstructure group should degrade performance far more than dropping any other group. If not, that's a surprising and paper-worthy finding.

**Existing stack suffices for this.** No new library needed. SHAP 0.51.0 is already installed; permutation_importance is in sklearn 1.8.

**Confidence:** HIGH.

---

### 3. Live-vs-Backtest Reconciliation

**Primary library:** `quantstats==0.0.81` (released 2026-01-13, actively maintained).

**Why quantstats:**
- Single function `qs.reports.html(returns, benchmark)` generates a tearsheet comparing two return series -- exactly the backtest-vs-live comparison we need.
- Takes a pandas Series indexed by datetime, which is trivial to produce from `positions.db` and `trade_log.jsonl` + the Phase-07.1 walk-forward backtester output.
- Modern maintenance (2026 release), Python 3.10+ supported.
- Produces publication-usable tables (Sharpe, Sortino, max drawdown, win rate, monthly returns heatmap).

**Why not `pyfolio-reloaded`:**
- `pyfolio-reloaded` 0.9.9 (June 2025) is maintained but has a heavier dependency tree (requires `ipython`, stuck on older matplotlib patterns).
- Its killer feature for our use case is `create_full_tear_sheet(..., live_start_date=...)` which splits returns into "simulated" and "live" sections -- useful but overlaps with what quantstats does.
- **If quantstats struggles with our specific reconciliation need (comparing per-pair prediction error, not just return series), fall back to pyfolio-reloaded.** Otherwise skip it.

**Why not `empyrical-reloaded` alone:**
- `empyrical-reloaded` 0.5.12 is a metrics library (Sharpe, Sortino, alpha, beta, calmar) with no tearsheet generation.
- Both quantstats and pyfolio-reloaded internally call empyrical. Installing empyrical directly only makes sense if we write our own custom tearsheet, which is more work than using quantstats.

**Our reconciliation workflow (to build in Phase 8 or equivalent):**

```
Input:
- positions.db (SQLite) -- actual live paper trades since Phase 07.3
- trade_log.jsonl        -- per-trade entry/exit log (redundant with positions.db but JSON is easier to diff)
- walk_forward_results.parquet (from Phase 07.1) -- what the backtester would have predicted for the same days

Output:
- Dataframe with columns [date, pair_id, predicted_return_backtest, actual_return_live, diff]
- quantstats HTML tearsheet comparing two aggregate return series
- A scatter plot of predicted vs. actual per-pair returns
- Summary stats: per-model implementation shortfall, per-pair bias, systematic deviation indicators

Libraries required:
- quantstats (new)
- pandas, sqlite3, json (stdlib or already installed)
- matplotlib (already installed)
```

**Important:** The MEMORY context flags a `pair_id schema bug` (3 code paths disagree on `live_NNNN` meaning) and notes that live P&L since April 9 is suspect. **The reconciliation exercise must first resolve this schema bug -- otherwise we're reconciling garbage against garbage.** Budget 1-2 days for schema audit before running quantstats.

**Confidence:** HIGH on library choice (quantstats is the canonical Python tool for this). MEDIUM on applicability to prediction-market data (quantstats was designed for equity strategies; per-pair contract returns may need a custom aggregation wrapper before passing to `qs.reports`).

---

### 4. Publication-Quality Figures

**Primary library:** `SciencePlots==2.2.1` (released 2026-02-25, actively maintained).

**Why SciencePlots:**
- One-liner usage: `import scienceplots; plt.style.use(['science', 'ieee'])` -- all figures in the paper become IEEE-compliant in a single session.
- Includes `ieee`, `nature`, `grid`, `high-vis`, `bright`, and `no-latex` variant styles.
- Active 2026 release, mature (2.x series).

**LaTeX question:** SciencePlots recommends LaTeX but offers a `no-latex` style variant. **For a DS340 submission we do not need LaTeX rendering.** Rationale:
- Pros of LaTeX: Math expressions in labels look identical to the paper body.
- Cons: Adds a MacTeX/TeXLive install requirement for teammates (Alvin), triples plot generation time, can break CI on SCC if LaTeX isn't installed there.
- **Use `plt.style.use(['science', 'ieee', 'no-latex'])`.** The visual difference is imperceptible in the final paper; the workflow cost difference is significant.

**Why not `tueplots`:**
- `tueplots` 0.2.4 (March 2026) is a clean design -- "no internal state, no aesthetic opinions" -- but it only provides sizing helpers. Styling (colors, line widths, grid behavior) is still our problem.
- SciencePlots handles both sizing AND styling.
- Since they overlap in purpose, pick one. SciencePlots has broader adoption, more journal-specific styles, and more examples online.

**Workflow for paper figures:**

```python
import scienceplots
import matplotlib.pyplot as plt

# At the top of each figure-generating script
plt.style.use(['science', 'ieee', 'no-latex'])

# Or for a more conservative default that works across venues:
plt.style.use(['science', 'no-latex', 'grid'])
```

**Additionally (no install needed, pure matplotlib):**
- Set `plt.rcParams['figure.dpi'] = 300` for paper-quality raster exports.
- Use `fig.savefig('name.pdf', bbox_inches='tight')` for PDF-native figures in the LaTeX paper.
- Explicit font sizes: `plt.rcParams['font.size'] = 9` (IEEE column width).

**Confidence:** HIGH. SciencePlots is the de facto standard; the `no-latex` option eliminates the only real friction.

---

### 5. Ensemble Exploration

**Primary tool:** Custom 40-line ensemble class wrapping our existing `BasePredictor` interface. **Do NOT use sklearn's `VotingRegressor` or `StackingRegressor` in the critical path.**

**Why custom:**
- Our Phase-04 `BasePredictor` abstract class has `fit`, `predict`, `evaluate` methods and a pickle save/load contract that the deployment pipeline relies on.
- sklearn's `VotingRegressor` expects sklearn-style estimators with `fit(X, y)` taking 2D arrays -- but our Tier-2 models (GRU, LSTM) take 3D sequence tensors. Shoehorning them into VotingRegressor means writing an adapter anyway, at which point we might as well write the ensemble class directly.
- `StackingRegressor` has the same issue, and its cross-validated meta-learner is overkill for a class paper with 978 test rows.

**What to try for v1.1 ensemble finding:**

```python
class EnsemblePredictor(BasePredictor):
    """Simple mean-of-predictions ensemble. Can be extended to weighted mean."""
    def __init__(self, predictors, weights=None):
        self.predictors = predictors          # list of BasePredictor instances
        self.weights = weights or [1.0/len(predictors)] * len(predictors)

    def predict(self, X):
        preds = np.column_stack([p.predict(X) for p in self.predictors])
        return (preds * self.weights).sum(axis=1)

    # fit/save/load delegate to constituent predictors
```

**Three ensemble variants to evaluate (each a row in the paper's final comparison table):**

1. **LR + LR-seed:** Two Linear Regression models trained with different feature normalization strategies. Sanity check that ensembling helps at all. (Likely ties single LR.)
2. **LR + XGBoost (already deployed):** Our Oracle VM deployment uses `LR + XGBoost` ensemble averaging per Phase-07.3 decisions. The v1.1 task is to verify this combination is optimal vs. alternatives.
3. **LR + LSTM or LR + GRU:** Mix a Tier-1 and Tier-2 model. Linear + non-linear diversity.
4. **(Optional) Majority-vote on direction:** Predict the sign of the spread change by majority vote across 3-5 models, use magnitude from the best-single-model. Trading-metric only, not RMSE.

**Meta-learner question (sklearn StackingRegressor alternative):**
- A gradient-boosted meta-learner (XGBoost predicting on LR + GRU outputs) can in principle beat weighted averaging.
- But: meta-learner trained on 6,802 rows with 2-4 input features will overfit wildly without careful cross-validation.
- **Skip for v1.1.** Simple weighted averaging is the appropriate complexity tier for our dataset size.

**Why not `mlforecast`:**
- Nixtla's `mlforecast` supports sklearn-style regressors in an ensemble API, but the training loop is opinionated -- it assumes time-series cross-validation with its `cross_validation` method. Our walk-forward backtester (Phase 07.1) already does this our way. Using mlforecast would mean maintaining two forecasting frameworks.

**Confidence:** HIGH. Existing stack (numpy, pandas, our `BasePredictor`) is sufficient.

---

## Revised Installation Plan

Run in `.venv/` before starting v1.1 phase execution:

```bash
# Step 1: Verify Python 3.14 wheel availability (may need fresh venv on 3.12 if wheels missing)
.venv/bin/pip install pytorch-forecasting==1.7.0 --dry-run

# Step 2: If dry-run succeeds, install
.venv/bin/pip install pytorch-forecasting==1.7.0

# Step 3: Add v1.1-specific libraries
.venv/bin/pip install quantstats==0.0.81
.venv/bin/pip install SciencePlots==2.2.1

# Step 4 (optional, only if quantstats proves inadequate):
# .venv/bin/pip install empyrical-reloaded==0.5.12
# .venv/bin/pip install pyfolio-reloaded==0.9.9

# Step 5: Re-freeze requirements.txt so SCC and GHA pick up the changes
.venv/bin/pip freeze > requirements.txt

# Step 6 (if re-running RL comparison):
# .venv/bin/pip install stable-baselines3==2.7.1 gymnasium==1.2.3
```

---

## Version Compatibility Notes

| Library | Python | torch | Notes |
|---|---|---|---|
| pytorch-forecasting 1.7.0 | `>=3.10, <3.15` | requires torch + pytorch-lightning | Should work on current 3.14 venv, but verify with dry-run first. |
| quantstats 0.0.81 | >=3.10 (approx.) | n/a | Pure Python + pandas; no torch dependency. |
| SciencePlots 2.2.1 | >=3.7 | n/a | Matplotlib styles only; trivial install. |
| empyrical-reloaded 0.5.12 | >=3.10 | n/a | Requires pandas >= 2.2.2 (we have 2.3.3, fine). |
| pyfolio-reloaded 0.9.9 | >=3.9 | n/a | Not recommended as first choice. |

**Potential Python 3.14 compatibility risks (verify):**
- pytorch-forecasting's transitive deps (scikit-base, tensorboard) may lag wheel releases for 3.14.
- If any dep fails to install, downgrade to Python 3.12 for v1.1 work and keep 3.14 for other projects.

---

## Cross-Reference to v1.0 STACK.md

The following v1.0 items are **unchanged** in v1.1 and should not be re-researched:
- Python choice (except for version drift documented above)
- PyTorch, sentence-transformers, sklearn, XGBoost choices
- requests, pandas, numpy (data pipeline)
- SHAP (interpretability)
- matplotlib + seaborn (base plotting; SciencePlots supplements, does not replace)
- stable-baselines3 (PPO, when re-running for comparison)
- gymnasium (env interface)

---

## Sources

- **PyPI verification (2026-04-16):**
  - [pytorch-forecasting 1.7.0](https://pypi.org/project/pytorch-forecasting/) (Apr 5, 2026)
  - [quantstats 0.0.81](https://pypi.org/project/quantstats/) (Jan 13, 2026)
  - [SciencePlots 2.2.1](https://pypi.org/project/SciencePlots/) (Feb 25, 2026)
  - [tueplots 0.2.4](https://pypi.org/project/tueplots/) (Mar 24, 2026)
  - [empyrical-reloaded 0.5.12](https://pypi.org/project/empyrical-reloaded/) (maintained 2025-2026)
  - [pyfolio-reloaded 0.9.9](https://pypi.org/project/pyfolio-reloaded/) (Jun 2, 2025)
  - [neuralforecast 3.1.7](https://pypi.org/project/neuralforecast/) (Apr 10, 2026) -- evaluated, rejected for scope reasons

- **TFT hyperparameter guidance:**
  - [pytorch-forecasting TFT API ref](https://pytorch-forecasting.readthedocs.io/en/latest/api/pytorch_forecasting.models.temporal_fusion_transformer._tft_v2.TFT.html)
  - [GitHub Issue #1322 -- TFT memory consumption on small datasets](https://github.com/sktime/pytorch-forecasting/issues/1322) (documents hidden_size=160 failures)
  - Default values verified: legacy TFT hidden_size=16, TFT v2 hidden_size=64, hidden_size_range=(8,128)

- **Feature ablation references:**
  - [sklearn 1.8 permutation_importance docs](https://scikit-learn.org/stable/modules/permutation_importance.html) -- model-agnostic, n_repeats parameter
  - [lofo-importance on GitHub](https://github.com/aerdem4/lofo-importance) -- evaluated, rejected (rolling our own is auditable)

- **Live-vs-backtest reconciliation references:**
  - [quantstats GitHub](https://github.com/ranaroussi/quantstats) -- `qs.reports.html` + Monte Carlo additions
  - [pyfolio `create_full_tear_sheet(live_start_date=...)`](https://quantopian.github.io/pyfolio/) -- canonical pattern for backtest-to-live comparison

- **Publication figures references:**
  - [SciencePlots GitHub](https://github.com/garrettj403/SciencePlots) -- IEEE/Nature styles, no-latex variant
  - [tueplots GitHub](https://github.com/pnkraemer/tueplots) -- evaluated, overlaps with SciencePlots

- **Ensemble references:**
  - [sklearn VotingRegressor 1.8.0](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html)
  - [sklearn StackingRegressor 1.8.0](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html)
  - Rejected due to sklearn estimator API mismatch with our BasePredictor for Tier-2 models

- **Current venv inspection (2026-04-16):** `.venv/bin/pip list` shows Python 3.14.3, torch 2.11.0, xgboost 3.2.0, sklearn 1.8.0; pytorch-forecasting and stable-baselines3 NOT currently installed (drift from v1.0 STACK.md).

---

*v1.1 stack additions researched: 2026-04-16*
*Complements v1.0 STACK.md dated 2026-04-01*
