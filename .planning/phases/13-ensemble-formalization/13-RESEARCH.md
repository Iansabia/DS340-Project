# Phase 13: Ensemble Formalization - Research

**Researched:** 2026-04-22
**Domain:** Weighted-average ensemble design, concordance filter audit, BasePredictor extension
**Confidence:** HIGH — grounded directly in the shipped codebase; every answer traces to an existing file

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ENSM-01 | New `src/models/ensemble.py` implementing `EnsemblePredictor(BasePredictor)`. Picklable via `BasePredictor.save/load`. | Pickle-based save/load in `base.py` lines 90-124; architecture design in section "EnsemblePredictor Architecture" below |
| ENSM-02 | Four ensemble variants evaluated: (a) LR alone, (b) LR + XGBoost equal-weight, (c) LR + LSTM, (d) majority-vote LR + XGBoost + LSTM | Variant specs in "Four Variants Spec" section; "majority-vote" = sign-agreement gate on the average, defined precisely below |
| ENSM-03 | Concordance filter audit with BOTH filtered and unfiltered P&L. Includes rejection rate and P&L on rejected trades. Flag if rejected trades are profitable. Guards P4. | P4 concordance-filter denominator trap documented in PITFALLS.md §4; audit table schema in "Concordance Audit" section |
| ENSM-04 | Weight sensitivity sweep: LR-weight from 0.0 to 1.0 in steps of 0.1 (11 data points). One plot. | Sweep design in "Weight Sweep" section; uses verify_headline.py `simulate_pnl()` internally |
| ENSM-05 | `EnsemblePredictor` NOT wired into `src/live/strategy.py` during v1.1. Git diff of that file in Phase 13 commits must be empty. | `strategy.py` read end-to-end — concordance filter is lines 427-429; must remain untouched |
| ENSM-06 | New `experiments/run_ensemble_sweep.py` (~100 LOC). | Pattern analysis in "Runner Script" section; mirrors verify_headline.py structure |
| ENSM-07 | Paper section 4.4 rewritten with evidence-based ensemble justification; new ensemble table added to section 5. | Paper-writing notes in "Paper Output" section |
</phase_requirements>

---

## Summary

Phase 13 has a clear technical structure: one new model class (`EnsemblePredictor`), one experiment runner, a concordance audit table, an 11-point weight sweep, and paper prose. The codebase is already complete and working; this phase adds a formalization layer over what the live strategy already does.

The most critical analytical question is the P4 concordance pitfall: does the concordance filter (skip if sign(LR) != sign(XGB)) inflate Sharpe by eliminating the hardest trades? The audit must show both filtered and unfiltered P&L and compute counterfactual P&L on rejected trades. If rejected trades are net profitable, the filter is selectively killing real edge while making the reported Sharpe look better.

Since LR (+$201.69, Sharpe 0.434) and XGBoost (+$201.63, Sharpe 0.436) are essentially tied, the weight sensitivity sweep is expected to show near-flat P&L across the 0.0-to-1.0 range. The honest paper finding will be "ensemble weight choice is immaterial; the concordance filter is the main discriminator." This is a publishable result, not a failure.

**Primary recommendation:** Build `EnsemblePredictor` as a `BasePredictor` subclass taking `(list_of_predictors, weights, concordance_mode)`. Train members fresh inside `fit()`. Run all 11 weight steps through the existing `simulate_pnl()` function from `verify_headline.py`. Produce the concordance audit table as an explicit two-row comparison (filtered vs unfiltered). Do not touch `strategy.py`.

---

## Standard Stack

### Core — Already Installed

| Library | Purpose | Relevant Usage |
|---------|---------|----------------|
| `numpy` | Weighted average, sign check, stack predictions | `np.stack`, `np.sign`, `np.where` |
| `pandas` | Feature DataFrames passed to child predictors | Same columns as all other models |
| `pickle` (stdlib) | Serialization via `BasePredictor.save/load` | Lines 90-124 of `src/models/base.py` |

**No new libraries needed.** `sklearn.ensemble.VotingRegressor` is explicitly rejected (incompatible with 3D tensor inputs from GRU/LSTM — confirmed by SUMMARY.md §Top Stack Additions).

### Existing Code Re-used

| File | What to Reuse |
|------|---------------|
| `src/models/base.py` | `BasePredictor` ABC — extend, do not modify |
| `experiments/verify_headline.py` | `simulate_pnl()` function — copy logic verbatim or import |
| `src/evaluation/results_store.py` | `save_results(model_name, metrics, output_dir, extra)` |
| `src/features/engineering.py` | `compute_derived_features()` — already used by all models |
| Tests pattern from `tests/models/test_linear_regression.py` | Mirror for `test_ensemble.py` |

---

## Architecture Patterns

### EnsemblePredictor Architecture

**Recommended constructor signature:**

```python
class EnsemblePredictor(BasePredictor):
    def __init__(
        self,
        members: list[BasePredictor],
        weights: list[float] | None = None,
        concordance_mode: str = "none",  # "none" | "strict"
    ) -> None:
```

**Why `members` as pre-constructed predictors, not lazy names:**
- The caller controls member construction (hyperparameters, seeds) before passing in.
- `EnsemblePredictor.fit()` calls `m.fit(X, y)` for each member — training happens inside `fit()`.
- No lazy construction needed: the experiment script builds `LR()`, `XGB()`, `LSTM()` instances and passes them in.
- This is consistent with how `LinearRegressionPredictor()` and `XGBoostPredictor()` are instantiated everywhere in the codebase.

**Why train members inside `fit()` rather than accept pre-trained:**
- Reproducibility: one `fit(X_train, y_train)` call trains everything on the same data.
- The paper's evaluation protocol already trains and evaluates in a single pass per script; pre-trained members would require loading pickles, which adds complexity and coupling.
- Academic rule: all ensemble members must train on the same training set and predict on the same test set. If we accept pre-trained members, there is no enforcement of this.

**Sequence-model `group_id` constraint:**
- LSTM and GRU require `group_id` column in X.
- When an ensemble contains a sequence model, the experiment script must pass `X_with_group_id`.
- `EnsemblePredictor.predict()` passes the full X to each child; LR and XGB ignore the extra column (they use `.values` on specific columns they saw during fit, or all columns — check `linear_regression.py` line 34: `X_train.values` — so group_id column passes through harmlessly as a numeric category code).

**Concordance mode implementation:**

The concordance filter in `strategy.py` (lines 427-429) is:
```python
if np.sign(lr_pred) != np.sign(xgb_pred):
    continue
```

This means: if both predictions are the same sign, proceed; else skip (return `0.0` in backtest context). When prediction is `0.0`, `simulate_pnl` applies `abs(p) > fee` threshold check, so `0.0` correctly triggers "no trade."

```python
def predict(self, X: pd.DataFrame) -> np.ndarray:
    preds = np.stack([m.predict(X) for m in self._members])  # shape (n_members, n_rows)
    weights = self._weights / self._weights.sum()             # normalized
    weighted = (weights[:, None] * preds).sum(axis=0)        # shape (n_rows,)
    if self._concordance_mode == "strict":
        # All members must agree on sign; else emit 0.0
        signs = np.sign(preds)
        agree = np.all(signs == signs[0], axis=0)            # bool (n_rows,)
        return np.where(agree, weighted, 0.0)
    return weighted
```

**Picklability:** `EnsemblePredictor` pickles cleanly because `BasePredictor.save()` does `pickle.dump(self, f)` — the entire object including its `_members` list. Each child predictor (LR, XGB, LSTM) is itself picklable. LSTM/GRU contain PyTorch `nn.Module` objects which are picklable by default. **No manifest file or separate member pickle files are needed** — one `ensemble.pkl` contains everything.

**Verification:** The `BasePredictor.load()` method checks `isinstance(model, BasePredictor)` — `EnsemblePredictor` inherits from `BasePredictor`, so this check passes automatically.

### Four Ensemble Variants Specification

| Variant | Members | Weights | Concordance | Notes |
|---------|---------|---------|-------------|-------|
| (a) LR alone | `[LR]` | `[1.0]` | none | Baseline — identical to standalone LR eval |
| (b) LR + XGB equal-weight | `[LR, XGB]` | `[0.5, 0.5]` | strict | Matches live `strategy.py` exactly |
| (c) LR + LSTM equal-weight | `[LR, LSTM]` | `[0.5, 0.5]` | strict | Tests whether RNN adds diversity |
| (d) Majority-vote LR + XGB + LSTM | `[LR, XGB, LSTM]` | `[1/3, 1/3, 1/3]` | strict | All three must agree on sign |

**"Majority-vote" definition for regression models:**
- For 3 regression models, majority-vote = "at least 2 of 3 agree on sign."
- Concretely: use `concordance_mode="strict"` with 3 members — this means ALL 3 must agree (stricter than majority). This matches the REQUIREMENTS.md wording "majority-vote" which in this context means sign-consensus gating, not a true 2-of-3 vote.
- If true 2-of-3 is desired, implement a separate `"majority"` concordance mode: `agree = np.sum(np.sign(preds) == np.sign(preds[0]), axis=0) >= (n_members // 2 + 1)`.
- **Recommendation:** implement both `"strict"` (all must agree) and `"majority"` (half+1 must agree). Use `"strict"` for variants (b), (c), (d) in ENSM-02 to match live behavior. Report the difference in the concordance audit.

**TFT is excluded** because Phase 11 confirmed: TFT did NOT converge (RMSE 0.3262 vs GRU 0.2928). This was locked in STATE.md: "4-variant ensemble for Phase 13 (no TFT variant)."

### Concordance Audit Table Schema

This is the most important analytical output of Phase 13 (guards P4).

**Required columns (ENSM-03):**

| Column | Description |
|--------|-------------|
| Variant | Variant name (a), (b), (c), (d) |
| Mode | "filtered" or "unfiltered" |
| # trades | N rows where `abs(pred) > fee` (plus concordance gate for filtered) |
| Rejection rate | `(unfiltered_trades - filtered_trades) / unfiltered_trades` |
| P&L (filtered) | Sum of P&L on accepted trades only |
| P&L (unfiltered) | Sum of P&L if concordance gate removed |
| P&L on rejected | Counterfactual: P&L if rejected trades had been taken |
| Flag | "WARNING: rejected trades profitable" if P&L_rejected > 0 |

**Implementation pattern — compute both in one pass:**

```python
# Source: verify_headline.py simulate_pnl() pattern + concordance extension
def concordance_audit(lr_preds, xgb_preds, actuals, fee=0.02):
    avg = (lr_preds + xgb_preds) / 2.0
    agree_mask = np.sign(lr_preds) == np.sign(xgb_preds)
    
    # Unfiltered: use avg prediction, no concordance gate
    unfiltered = simulate_pnl(avg, actuals, fee)
    
    # Filtered: only trades where concordance passes
    filtered_preds = np.where(agree_mask, avg, 0.0)
    filtered = simulate_pnl(filtered_preds, actuals, fee)
    
    # P&L on rejected trades specifically
    rejected_preds = np.where(~agree_mask, avg, 0.0)
    rejected = simulate_pnl(rejected_preds, actuals, fee)
    
    rejection_rate = 1.0 - filtered["num_trades"] / max(unfiltered["num_trades"], 1)
    flag = rejected["pnl"] > 0
    return {..., "rejection_rate": rejection_rate, "flag": flag}
```

### Weight Sensitivity Sweep

11 evaluations at LR-weight = 0.0, 0.1, 0.2, ..., 1.0:
- weight=0.0: XGB alone (LR gets zero weight)
- weight=0.5: live default (equal weight)
- weight=1.0: LR alone

For each weight step:
1. Instantiate `EnsemblePredictor([LR, XGB], weights=[w, 1-w], concordance_mode="strict")`
2. `fit(X_train, y_train)` — trains both members fresh
3. `predict(X_test)` with concordance gate active
4. Run `simulate_pnl()` to get P&L and Sharpe

Plot: x = LR weight, y = P&L, two lines: "filtered" (concordance active) and "unfiltered" (concordance disabled).

**Expected result:** Near-flat curve across weight values, since LR (+$201.69) and XGB (+$201.63) are nearly identical. If true, the paper finding is: "Ensemble weight choice is not material; the concordance filter is the primary source of P&L discrimination."

### Runner Script Pattern

`experiments/run_ensemble_sweep.py` (~100 LOC) should follow `verify_headline.py`:

```
1. set_all_seeds(42)
2. Load + build train/test DataFrames (same build() helper)
3. Extract feature columns
4. For each variant (a-d): fit, predict, simulate_pnl, concordance_audit
5. For weight sweep: 11 iterations of LR-weight 0.0->1.0
6. save_results() to experiments/results/ensemble/*.json
7. Print concordance audit table
8. Save weight sweep data to JSON (for matplotlib plot)
```

**Should NOT extend verify_headline.py** — separate script is cleaner. The ensemble sweep has distinct output structure (concordance audit table, weight sweep plot) that would bloat verify_headline.py. Keep separation per the "one script per experiment type" pattern in the codebase.

### Paper Output (ENSM-07)

Section 4.4 rewrite needs:
1. Statement that live deployment uses LR + XGB equal-weight with concordance filter
2. Evidence: "Concordance filter rejects X% of potential trades; accepted trades have filtered P&L vs unfiltered P&L of..."
3. If rejected trades are profitable: honest statement "concordance filter trades P&L for lower variance (higher Sharpe on smaller N)"

New ensemble table in section 5 columns:
- Variant | # Trades | P&L (filtered) | P&L (unfiltered) | Rejection Rate | Sharpe (filtered) | P&L on Rejected | Flag

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead |
|---------|-------------|-------------|
| Ensemble aggregation | Custom averaging loop per experiment | `EnsemblePredictor.predict()` shared class |
| P&L simulation | Second implementation of fee logic | Import `simulate_pnl` from `verify_headline.py` or use `src/evaluation/profit_sim.simulate_profit` |
| Sharpe calculation | Per-trade Sharpe ad hoc | Existing `sharpe_per_trade` from `verify_headline.py` simulate_pnl — report per-pair-corrected via `src/evaluation/sharpe.py` for paper |
| Weight normalization | Manual division | `weights / weights.sum()` is 1 line; do it inside `predict()` not every call site |

---

## Common Pitfalls

### Pitfall 1: Concordance Filter Inflating Sharpe (P4 — CRITICAL)

**What goes wrong:** The concordance filter removes trades where models disagree — systematically eliminating the hardest trades. This inflates both hit rate and Sharpe on the remaining (easy) trades. Reporting "ensemble Sharpe 0.55" without showing unfiltered Sharpe misrepresents the filter's effect.

**Why it happens:** The filter looks like a risk control, not a selection bias. Authors report filtered performance without computing the counterfactual.

**How to avoid:** Always report both filtered and unfiltered P&L in the same table. Compute counterfactual P&L on rejected trades. Flag if rejected P&L > 0.

**Warning signs from PITFALLS.md §P4:**
- Concordance filter rejects >40% of potential trades
- Filtered subset has hit rate >20pp higher than unfiltered
- As more models are added to vote, Sharpe monotonically increases and trade count monotonically decreases

### Pitfall 2: group_id Column Silently Corrupts LR/XGB Predictions

**What goes wrong:** When passing `X_with_group_id` to ensemble containing both sequence models (LSTM) and flat models (LR, XGB), the `group_id` column is a numeric category code. LR fits on `X_train.values` (all columns), which includes `group_id`. If `group_id` codes change between train and test (new pairs added to test), LR predictions become wrong silently.

**How to avoid:** In `EnsemblePredictor.fit()`, for flat models (LR, XGB), pass `X.drop(columns=["group_id"], errors="ignore")`. For sequence models (GRU, LSTM), pass the full X including `group_id`. Track which child models need group_id.

**Implementation:**
```python
def predict(self, X):
    preds = []
    for m, w in zip(self._members, self._weights):
        if hasattr(m, "_needs_group_id") and m._needs_group_id:
            preds.append(m.predict(X))
        else:
            preds.append(m.predict(X.drop(columns=["group_id"], errors="ignore")))
```

Alternatively: check existing behavior — `LinearRegressionPredictor.predict()` calls `X.values` on all columns after training on all columns. If `group_id` was in training X, it must be in test X. The safest approach: always pass `X_with_group_id` to all members during both `fit()` and `predict()`. LR/XGB use `.values` and will include group_id as a feature — which is the same behavior as the existing `run_baselines.py` does NOT include group_id in the feature set (NON_FEATURE_COLUMNS excludes `group_id`).

**Confirmed safe approach:** Separate the feature set used for fit/predict by model type, identical to how `verify_headline.py` does it (lines 85-116: `seq_cols` for GRU/LSTM, plain `feats` for LR/XGB).

### Pitfall 3: Members Re-trained with Different Seeds Across Variants

**What goes wrong:** Variant (b) uses LR+XGB trained at `seed=42`. Variant (c) uses LR+LSTM. If LR in variant (c) trains with a different seed or different X (because group_id is included), the LR predictions differ from standalone LR — making the comparison unfair.

**How to avoid:** In `run_ensemble_sweep.py`, train a single canonical LR and XGB instance once, then construct ensemble variants by passing the same trained instances OR by ensuring all calls use `set_all_seeds(42)` before each `fit()`.

**Recommendation:** Call `set_all_seeds(42)` inside `EnsemblePredictor.fit()` before calling each member's `fit()`. Document this in the class docstring.

### Pitfall 4: `strategy.py` Modified Accidentally

**What goes wrong:** Developer refactors imports or style in `strategy.py` while working on `ensemble.py` in the same session.

**How to verify:** Run `git diff src/live/strategy.py` before any Phase 13 commit. If the diff is non-empty, stage the ensemble files only (`git add src/models/ensemble.py experiments/run_ensemble_sweep.py`).

**Acceptance gate for ENSM-05:** The Phase 13 verification step should explicitly check `git log --follow -p src/live/strategy.py` covers no Phase 13 commits.

### Pitfall 5: Majority-Vote Semantics Ambiguity

**What goes wrong:** REQUIREMENTS.md says "majority-vote LR + XGBoost + LSTM" but majority-vote is a classifier concept. For 3 regression models, it is ambiguous whether this means:
- (A) All 3 must agree on sign (strict)
- (B) At least 2 of 3 must agree on sign

**Clarification from research:** The live strategy concordance filter requires ALL members to agree (strict). For consistency with live behavior, use strict concordance for variant (d). The paper should state "sign-consensus gate (all members must agree on direction)" rather than "majority vote."

---

## Code Examples

### EnsemblePredictor Skeleton

```python
# src/models/ensemble.py
# Source: ARCHITECTURE.md §4, strategy.py lines 427-429, base.py lines 42-84
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import BasePredictor
from src.utils.seed import set_all_seeds


class EnsemblePredictor(BasePredictor):
    """Weighted average ensemble of BasePredictor instances.

    Args:
        members: List of BasePredictor instances. Trained during fit().
        weights: Per-member weights (need not sum to 1; normalized in predict).
                 If None, equal weights are assigned.
        concordance_mode: 'none' (no gate) or 'strict' (all members must agree
                          on sign, else emit 0.0 — matching live strategy.py).
        seed: Random seed applied before each member's fit() call.
    """

    def __init__(
        self,
        members: list[BasePredictor],
        weights: list[float] | None = None,
        concordance_mode: str = "none",
        seed: int = 42,
    ) -> None:
        if not members:
            raise ValueError("EnsemblePredictor requires at least one member")
        self._members = members
        self._weights = np.array(
            weights if weights is not None else [1.0] * len(members), dtype=float
        )
        if self._weights.sum() == 0:
            raise ValueError("Weights must not all be zero")
        self._concordance_mode = concordance_mode
        self._seed = seed
        self._fitted = False

    @property
    def name(self) -> str:
        member_names = "+".join(m.name for m in self._members)
        return f"Ensemble({member_names}, {self._concordance_mode})"

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> "EnsemblePredictor":
        set_all_seeds(self._seed)
        for m in self._members:
            m.fit(X_train, y_train)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("EnsemblePredictor must be fit before predict")
        preds = np.stack([m.predict(X) for m in self._members])  # (n_members, n_rows)
        weights = self._weights / self._weights.sum()
        weighted = (weights[:, None] * preds).sum(axis=0)          # (n_rows,)
        if self._concordance_mode == "strict":
            signs = np.sign(preds)
            agree = np.all(signs == signs[[0], :], axis=0)         # (n_rows,)
            return np.where(agree, weighted, 0.0)
        return weighted
```

### Concordance Audit Function

```python
# experiments/run_ensemble_sweep.py (partial)
# Source: PITFALLS.md P4 detection code + verify_headline.py simulate_pnl()
def concordance_audit(member_preds: dict[str, np.ndarray],
                      actuals: np.ndarray,
                      fee: float = 0.02) -> dict:
    """Compute filtered vs unfiltered P&L and P&L on rejected trades.

    Args:
        member_preds: {'lr': ndarray, 'xgb': ndarray, ...}
        actuals: Ground-truth spread changes
        fee: Trading fee threshold

    Returns:
        dict with keys: filtered, unfiltered, rejected, rejection_rate, flag
    """
    stacked = np.stack(list(member_preds.values()))   # (n_members, n_rows)
    weights = np.ones(len(member_preds)) / len(member_preds)
    avg_pred = (weights[:, None] * stacked).sum(axis=0)
    agree_mask = np.all(np.sign(stacked) == np.sign(stacked[[0]]), axis=0)

    unfiltered = simulate_pnl(avg_pred, actuals, fee)
    filtered = simulate_pnl(np.where(agree_mask, avg_pred, 0.0), actuals, fee)
    rejected = simulate_pnl(np.where(~agree_mask, avg_pred, 0.0), actuals, fee)

    rejection_rate = 0.0
    if unfiltered["num_trades"] > 0:
        rejection_rate = 1.0 - filtered["num_trades"] / unfiltered["num_trades"]

    flag = rejected["pnl"] > 0
    if flag:
        print("WARNING: concordance filter is rejecting profitable trades "
              f"(rejected P&L = ${rejected['pnl']:+.2f}) — P4 concordance trap active")

    return {
        "filtered": filtered,
        "unfiltered": unfiltered,
        "rejected": rejected,
        "rejection_rate": round(rejection_rate, 4),
        "flag_rejected_profitable": flag,
    }
```

### Feature Routing for Mixed Ensemble

```python
# How to handle flat vs sequence model feature routing in run_ensemble_sweep.py
# Source: verify_headline.py lines 82-116 pattern

# Build feature sets once
feats = feature_cols(train)           # flat feature list (excludes group_id)
nonzero = [c for c in feats if train[c].std() > 1e-10]
seq_cols = nonzero + ["group_id"]     # for LSTM

# For ensemble variants containing LSTM:
# Pass X_with_group_id to EnsemblePredictor — LR/XGB need to be
# explicitly constructed to only use feats (not seq_cols).
# Cleanest approach: pass X[feats] to flat-only ensembles,
# X[seq_cols] to mixed ensembles, and document in the LSTM child
# that group_id is required.
```

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (already installed, see `tests/models/test_linear_regression.py`) |
| Config file | None detected (inferred from `tests/conftest.py`) |
| Quick run command | `python -m pytest tests/models/test_ensemble.py -x -q` |
| Full suite command | `python -m pytest tests/ -q` |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ENSM-01 | EnsemblePredictor fit/predict contract | unit | `pytest tests/models/test_ensemble.py -x` | No — Wave 0 |
| ENSM-01 | EnsemblePredictor picklable via save/load | unit | `pytest tests/models/test_ensemble.py::test_save_load -x` | No — Wave 0 |
| ENSM-01 | Weight normalization sums to 1 | unit | `pytest tests/models/test_ensemble.py::test_weight_normalization -x` | No — Wave 0 |
| ENSM-01 | concordance_mode=strict emits 0.0 on disagreement | unit | `pytest tests/models/test_ensemble.py::test_concordance_strict -x` | No — Wave 0 |
| ENSM-02 | All 4 variants produce non-zero predictions | smoke | `python -m experiments.run_ensemble_sweep --dry-run` | No — Wave 0 |
| ENSM-03 | Concordance audit flag fires when rejected trades profitable | unit | `pytest tests/models/test_ensemble.py::test_concordance_audit_flag -x` | No — Wave 0 |
| ENSM-04 | Weight sweep produces 11 data points | smoke | `python -m experiments.run_ensemble_sweep --sweep-only` | No — Wave 0 |
| ENSM-05 | strategy.py not modified | manual | `git diff src/live/strategy.py` (must be empty) | n/a |

### Sampling Rate

- Per task commit: `python -m pytest tests/models/test_ensemble.py -x -q`
- Per wave merge: `python -m pytest tests/ -q`
- Phase gate: Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/models/test_ensemble.py` — covers ENSM-01 (fit/predict, save/load, concordance modes, weight normalization, NaN propagation, group_id routing)
- [ ] `experiments/run_ensemble_sweep.py` — covers ENSM-02, ENSM-03, ENSM-04, ENSM-06

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| `EnsemblePredictor` as experiment-level glue | Dedicated `BasePredictor` subclass | Deployable, testable, composable in walk-forward |
| Hard-coded `(lr_pred + xgb_pred) / 2.0` in strategy.py | Parameterized weights + concordance_mode | Auditable, sweepable |
| "Model ensemble" listed as anti-feature in v1.0 requirements | Ensemble as deployment documentation (not main claim) | Honest framing: documents live system, not the centerpiece |
| TFT planned as 5th ensemble variant | 4-variant ensemble (TFT excluded) | Phase 11 negative result confirmed; locked in STATE.md |

**Deprecated / Out of Scope:**
- `sklearn.VotingRegressor` — incompatible with 3D tensor inputs from GRU/LSTM
- Stacking regression meta-model — explicitly out of scope in v1.1 Non-Goals
- Deep meta-model neural stacker — guaranteed overfit at N=6,802
- Optuna weight optimization — out of scope; equal-weight is the justified default

---

## Open Questions

1. **Feature routing for mixed ensemble (LSTM + LR/XGB)**
   - What we know: LSTM needs `group_id`; LR/XGB trained with `X.values` include all columns
   - What's unclear: Will LR trained on `seq_cols` (which include `group_id`) perform differently than LR trained on `feats` alone? Yes — `group_id` is a category code that LR can memorize.
   - Recommendation: For ensemble variants (a), (b), (c where LSTM is NOT a member): use `X[feats]` for all members. For variant (c) LR+LSTM and (d) LR+XGB+LSTM: construct LR and XGB instances that fit on `X[feats]` and LSTM on `X[seq_cols]`. Simplest implementation: `EnsemblePredictor` stores per-member feature masks.

2. **What to do if rejected trades are net profitable (P4 flag fires)**
   - What we know: PITFALLS.md says flag it; P4 requires explicit mention in paper
   - What's unclear: How to frame this honestly without making the live system look broken
   - Recommendation: Frame as "concordance filter converts real P&L to Sharpe improvement by rejecting low-confidence trades — not all of which are losing trades. The filter is a risk control with a P&L cost." Include exact numbers in the table. This is the honest academic finding.

3. **Variant (a) LR alone is just the standalone LR baseline**
   - What we know: verify_headline.py already computes LR +$201.69
   - What's unclear: Should variant (a) re-run or reuse the existing number?
   - Recommendation: Re-run inside `run_ensemble_sweep.py` using `EnsemblePredictor([LR], weights=[1.0], concordance_mode="none")` for methodological consistency. This ensures same data build pipeline and same seed. Cross-check against verify_headline.py number — should be identical within <1% (ENV-04 tolerance).

---

## Sources

### Primary (HIGH confidence — code read directly)

- `src/models/base.py` — `BasePredictor` ABC; `save()` uses `pickle.dump(self, f)`; `load()` checks `isinstance(model, BasePredictor)`
- `src/live/strategy.py` lines 413-429 — exact concordance filter implementation: `np.sign(lr_pred) != np.sign(xgb_pred)` triggers `continue`
- `experiments/verify_headline.py` — `simulate_pnl()` function; build/feature_cols/main() pattern to reuse
- `src/models/linear_regression.py` — `predict()` uses `X.values` (all columns); `fit()` uses `X_train.values`
- `src/models/xgboost_model.py` — same pattern as LR
- `.planning/research/ARCHITECTURE.md` §4 — EnsemblePredictor design, code sketch, concordance mode spec
- `.planning/research/PITFALLS.md` §P4 — concordance filter denominator trap, `concordance_audit()` code pattern
- `.planning/research/FEATURES.md` §C4 — ensemble table stakes, anti-features, expected outcome
- `.planning/REQUIREMENTS.md` ENSM-01..07 — formal acceptance criteria

### Secondary (MEDIUM confidence — synthesized research)

- `.planning/research/SUMMARY.md` — Stack recommendation: "sklearn VotingRegressor is incompatible with our Tier-2 models' 3D tensor inputs"
- PITFALLS.md §P4 — Bailey & de Prado (2014) Deflated Sharpe framework applied to concordance filter

### Tertiary (LOW confidence — not re-verified for this phase)

- FEATURES.md §C4 §4.2 — "Rank correlation between ensemble members (Spearman ρ of predictions)" as differentiator — not required by ENSM-02..06 but worth adding if time permits

---

## Metadata

**Confidence breakdown:**
- EnsemblePredictor design: HIGH — traced directly to `base.py`, `strategy.py`, `linear_regression.py`
- Concordance audit: HIGH — PITFALLS.md P4 provides working code pattern; strategy.py confirms exact live semantics
- Weight sweep: HIGH — 11 iterations of existing `simulate_pnl()`, trivial implementation
- Feature routing for mixed ensemble: MEDIUM — ambiguity exists between `feats` vs `seq_cols` for LR when LSTM is a co-member; open question #1 above
- Paper section: MEDIUM — depends on empirical numbers from the sweep

**Research date:** 2026-04-22
**Valid until:** 2026-04-27 (paper submission deadline; no external dependencies to decay)
