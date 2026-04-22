# Phase 12 Research — Feature Ablation Study

**Researched:** 2026-04-22 (inline orchestrator research; subagent rate-limited)
**Confidence:** HIGH — grounded in live codebase inspection plus v1.1 research foundation

---

## Phase Summary

Run a pre-registered Leave-One-Group-Out (LOGO) ablation across 5 feature groups on both LR and XGBoost. Identify the minimum sufficient feature set for profitable trading. Report all runs including unfavorable ones. Guard against P3 p-hacking via three-way temporal split.

---

## 1. Exact Feature Group Definitions (inspected from live codebase)

Total features used by models: **51** (from 59 engineered, 8 excluded for NaN/zero-variance at `NON_FEATURE_COLUMNS`).

The 5 pre-registered groups (atomic, non-overlapping):

### Group A — Raw aligned OHLCV (15 features)
```
kalshi_vwap, kalshi_open, kalshi_high, kalshi_low, kalshi_close,
kalshi_volume, kalshi_trade_count,
polymarket_vwap, polymarket_open, polymarket_high, polymarket_low, polymarket_close,
polymarket_volume, polymarket_trade_count, polymarket_dollar_volume
```

### Group B — Cross-platform basic (10 features)
```
spread, mid_price, volume_ratio, dollar_volume_ratio, price_divergence_pct,
spread_range, trade_count_ratio, price_velocity, boundary_distance,
kalshi_dollar_volume
```

### Group C — Rolling / momentum (6 features)
```
spread_momentum, spread_momentum_6, spread_momentum_12,
spread_volatility, spread_volatility_6, spread_zscore
```

### Group D — Classical microstructure (13 features)
```
polymarket_realized_spread, kalshi_amihud, polymarket_amihud,
kalshi_kyle_lambda, polymarket_kyle_lambda,
kalshi_roll_spread, polymarket_roll_spread,
kalshi_cs_spread, polymarket_cs_spread,
kalshi_hl_vol, polymarket_hl_vol,
polymarket_order_flow_imbalance, ofi_differential
```

### Group E — Prediction-market / trade-dynamics (7 features)
```
longshot_score,
kalshi_max_trade_size, polymarket_max_trade_size,
kalshi_hours_since_last_trade, polymarket_hours_since_last_trade,
polymarket_buy_volume, polymarket_sell_volume
```

**Total:** 15 + 10 + 6 + 13 + 7 = 51 ✓ (matches verified headline count)

---

## 2. Baseline to Beat (All 51 Features)

From fresh Apr 22 retrain (`experiments/results/tier1/*.json`) — **this is the all-features baseline every LOGO run compares against**:

| Model | P&L @2pp | RMSE | Dir. Acc. |
|---|---|---|---|
| Linear Regression | +$232.67 | 0.3063 | 66.7% |
| XGBoost (depth-3) | +$232.83 | 0.2899 | 66.4% |

LOGO delta-P&L = `P&L(51 - group) - P&L(51 features)`. Negative deltas indicate the dropped group is load-bearing; near-zero deltas indicate droppable.

---

## 3. Three-Way Split Design (ABLA-03 — P3 guard)

**Problem:** Running 10 ablation configurations (5 groups × 2 models) on the same test set and picking the best set risks p-hacking the test-set ordering.

**Solution — temporal three-way split:**

Current single-split uses ~80/20 train/test on time-sorted data (6,802 train / 1,673 test rows).

**Phase 12 split (preserves chronological ordering):**
```
Original train (6,802 rows) → [train_proper | ablation_holdout]   — 85/15 temporal split
Original test  (1,673 rows) → final_test                          — UNTOUCHED
```

Sizes:
- `train_proper`: ~5,780 rows (earliest 85% of old train)
- `ablation_holdout`: ~1,020 rows (latest 15% of old train) — used ONLY for LOGO selection
- `final_test`: 1,673 rows (original test set, FROZEN until after selection)

**Discipline:**
1. All 10 LOGO configurations trained on `train_proper`, evaluated on `ablation_holdout`
2. Best-performing feature set selected on `ablation_holdout` scores
3. Selected set re-trained on `train_proper + ablation_holdout` (full original train), evaluated ONCE on `final_test`
4. Final-test number is the reportable P&L; ablation-holdout numbers go in the ablation table but are noted as selection metric, not generalization metric

**Alternative considered and rejected:** Splitting the test set. Rejected because our test set is only 1,673 rows and further splitting would produce noisy estimates.

---

## 4. Bootstrap CI Methodology (ABLA-04)

**Procedure:**
1. For each (model, dropped_group) configuration, compute per-trade P&L series on `ablation_holdout`
2. Bootstrap resample with replacement: 1,000 iterations
3. For each bootstrap sample, compute total P&L
4. Report 95% CI as [2.5th percentile, 97.5th percentile]

**Why 1,000 resamples:** Standard for trading P&L CIs; gives smooth percentile estimates without excessive compute (~10s per config).

**Delta CI:**
For the delta vs baseline, use the paired-bootstrap:
1. Pair each trade with its baseline P&L (same trade index, different model config)
2. Bootstrap sample trade indices
3. Compute delta = sum(config_pnl - baseline_pnl)
4. 95% CI from 1,000 resamples

**Critical:** If the 95% CI of a group's delta straddles zero, the group is "not conclusively load-bearing" — call this out explicitly rather than claiming any effect.

---

## 5. Pre-Registration Protocol Content (ABLA-01)

`.planning/ablation_protocol.md` must be committed **BEFORE** `run_feature_ablation.py` executes. Per P3 pitfall guard, minimum contents:

1. **Hypothesis:** Which groups are expected load-bearing vs droppable. Be specific — claims like "classical microstructure is droppable at N=6,802 because rolling windows are data-starved (Finding 12)" are stronger than "we'll see what happens."

2. **Feature groups (copy from Section 1 above):** All 51 features assigned to exactly one group. No overlaps.

3. **Split design (copy from Section 3):** Exact row counts, temporal ordering rule.

4. **Models tested:** LR (default hyperparams) and XGBoost (depth=3, n_est=100, lr=0.01 per Finding 13).

5. **Primary metric:** P&L @ 2pp fees on `ablation_holdout` (selection metric) and `final_test` (reportable metric).

6. **Success criteria for each group's "load-bearing" claim:** 95% CI of delta-P&L entirely below zero AND magnitude > $10 AND directional accuracy drops > 2 percentage points.

7. **Reporting pre-commitment:** "ALL 10 LOGO configurations will be reported in the final table regardless of outcome. No configurations will be retrospectively added or removed."

8. **Final-test discipline:** "Final-test split is not touched until after ablation-holdout selection is frozen and the protocol is signed off."

---

## 6. Integration Architecture

### New file: `experiments/run_feature_ablation.py` (~200 LOC)

**Structure:**
- Reuses `verify_headline.py` helpers (`build`, `feature_cols`, `simulate_pnl`) — import, don't duplicate
- Parses `.planning/ablation_protocol.md` feature groups from a structured `FEATURE_GROUPS` dict at top of file (source of truth is the protocol doc, code mirrors it)
- Runs 10 configs: {LR, XGBoost} × {baseline, drop_A, drop_B, drop_C, drop_D, drop_E}
- For each config: fit on `train_proper`, predict on `ablation_holdout`, compute P&L + RMSE + DA, bootstrap CIs
- Writes: `experiments/results/ablation/summary.json`, `per_config.csv`, `bootstrap_distributions.npz`
- Writes human-readable `experiments/results/ablation/report.md` with the pre-committed table layout

### Data flow

```
train.parquet  →  build()  →  train_proper (85%) + ablation_holdout (15%)
test.parquet   →  build()  →  final_test (untouched until Task 4)

for (model, dropped_group) in 10 configs:
  X_train = train_proper[feats - FEATURE_GROUPS[dropped_group]]
  X_hold  = ablation_holdout[same cols]
  fit(X_train, y) → predict(X_hold) → metrics + bootstrap CI
  
# Then after selection is frozen:
retrain best config on (train_proper + ablation_holdout) → evaluate on final_test → REPORT
```

### CLI: `python -m experiments.run_feature_ablation [--dry-run]`
- `--dry-run` validates feature-group sums to 51 and prints split sizes without training

---

## 7. Expected Outcome

Based on prior findings:

| Group | Prediction | Rationale |
|---|---|---|
| A — Raw OHLCV | Load-bearing | SHAP (Finding 5): polymarket_vwap dominates; TFT VSN top-5 includes polymarket_high |
| B — Cross-platform basic | **Load-bearing** | `spread` and `price_divergence_pct` are the target-defining features |
| C — Rolling/momentum | Neutral-to-droppable | Finding 10: spread_momentum etc. were neutral at 47 bars/pair |
| D — Classical microstructure | **Droppable** | Finding 12: Amihud/Kyle/Roll neutral at current data scale |
| E — Prediction-market | Droppable | longshot_score is one feature; boundary_distance questionable value |

**Predicted minimum sufficient set:** Groups A + B = 25 features (49% of 51). If this holds, it's a publishable parsimony finding: "25 features carry the full edge; 26 features are dead weight at N=6,802."

**What would be surprising:** If any of {C, D, E} are load-bearing. That would invalidate the Finding 10/12 narrative and be an even stronger paper result.

---

## 8. Reporting Discipline (ABLA-05)

**Mandatory reporting format (the ablation table):**

| Model | Dropped Group | # Features | P&L @ 2pp | ΔP&L | 95% CI of ΔP&L | RMSE | Dir. Acc. |
|---|---|---|---|---|---|---|---|
| LR | — (baseline) | 51 | +$X | 0 | — | 0.306 | 66.7% |
| LR | A (raw OHLCV) | 36 | +$Y | -$Z | [a, b] | 0.XX | X.X% |
| LR | B (cross-platform) | 41 | ... | ... | ... | ... | ... |
| LR | C (rolling) | 45 | ... | ... | ... | ... | ... |
| LR | D (microstructure) | 38 | ... | ... | ... | ... | ... |
| LR | E (prediction-market) | 44 | ... | ... | ... | ... | ... |
| XGBoost | — (baseline) | 51 | +$X | 0 | — | ... | ... |
| ... 5 more rows for XGBoost ... |

All 12 rows (6 LR + 6 XGBoost) reported. No "we only tried the best-looking ones" language allowed.

**Minimum-sufficient-set subtable:** After ablation-holdout selection, report final-test number for the selected set vs full-51 baseline on final-test. One row of honest comparison.

---

## 9. Paper Integration (ABLA-08)

New section `§5.10 Feature Ablation` (after §5.9 Live vs Backtest Reconciliation).

Structure:
- 1-paragraph intro: motivation (parsimony, P3 p-hacking guard via pre-registration)
- Protocol summary (point readers to committed `.planning/ablation_protocol.md`)
- Table (12 rows above) with caption noting 95% CIs
- 1-paragraph findings + implications for §6 Discussion
- 1-paragraph future-work hook: "at 250+ bars/pair, classical microstructure likely becomes load-bearing; ablation should be re-run"

Length: ~1/2 page. This is a subsection, not a centerpiece.

---

## Validation Architecture

| Requirement | Automated verification |
|---|---|
| ABLA-01 (pre-registration) | `git log --oneline -- .planning/ablation_protocol.md` shows commit BEFORE `experiments/run_feature_ablation.py` |
| ABLA-02 (LOGO on LR and XGBoost) | `summary.json` has 12 config entries (2 models × 6 configs including baseline) |
| ABLA-03 (three-way split) | `summary.json` contains `train_proper_rows`, `ablation_holdout_rows`, `final_test_rows` with expected sizes |
| ABLA-04 (bootstrap CIs) | Each config entry has `ci_lower` and `ci_upper` fields |
| ABLA-05 (all runs reported) | Table in `report.md` and `§5.10` has exactly 12 rows including baselines |
| ABLA-06 (LR vs XGB separately) | Per-model report sections exist |
| ABLA-07 (run_feature_ablation.py ~200 LOC) | File exists; `wc -l` shows < 300 |
| ABLA-08 (paper §5.10) | `grep "### 5.10 Feature Ablation" PAPER_DRAFT.md` returns a match |

---

## Open Questions

1. **Should we run ablation only on `ablation_holdout` + final-test once, or also run on full `test.parquet` for the ablation-holdout numbers?** Recommendation: ablation-holdout only during selection; final-test used only for one-shot confirmation of selected set.

2. **What if a group's 95% CI straddles zero with |mean| > $10?** Recommendation: classify as "inconclusive, pending more data" rather than load-bearing. The paper language should reflect this ambiguity honestly.

3. **Do we need a separate run with per-feature (not per-group) ablation?** Recommendation: NO. LOGO over 5 groups is what was pre-registered in research. Per-feature LOFO at N=6,802 produces noisy single-feature results and was explicitly rejected as anti-feature in v1.1 SUMMARY.md.

---

## Ready for Planning

All 8 ABLA requirements have concrete implementation guidance. Feature group definitions are complete and atomic. Split design protects against P3. Bootstrap methodology is specified. Integration plan identifies exactly one new 200-LOC script with no changes to existing files.

**Estimated effort:** 2 tasks, 1 wave.
- Task 1: Write protocol + run ablation (~60 min: mostly compute, script dev ~30 min)
- Task 2: Write §5.10 + update FINDINGS
