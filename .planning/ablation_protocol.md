# LOGO Feature Ablation — Pre-Registration Protocol

**Pre-registered:** 2026-04-22
**Authors:** Ian Sabia (U33871576), Alvin Jang (U64760665)
**Status:** Locked — this document is committed to git before any experiment runs

---

## 1. Hypothesis

We predict that not all 51 features are load-bearing for profitable trading. Specifically:

- **Group A (Raw OHLCV, 15 features): Load-bearing.**
  SHAP analysis (Finding 5) shows `polymarket_vwap` dominates feature importance rankings. TFT VSN attention top-5 includes `polymarket_high`. These raw price series encode the fundamental cross-platform price relationship.

- **Group B (Cross-platform basic, 10 features): Load-bearing.**
  `spread` and `price_divergence_pct` are the target-defining features — the spread IS what we are predicting the change of. Dropping Group B should collapse both RMSE and P&L since the model loses direct access to the quantity it must forecast.

- **Group C (Rolling/momentum, 6 features): Neutral-to-droppable.**
  Finding 10 showed spread_momentum metrics were neutral at 47 bars/pair. Rolling windows require sufficient history per pair; at N=6,802 with ~47 bars/pair, these features likely add noise rather than signal.

- **Group D (Classical microstructure, 13 features): Droppable.**
  Finding 12 confirmed Amihud, Kyle lambda, Roll spread, and CS spread measures were neutral at current data scale. Prediction markets operate differently from equity markets; Amihud illiquidity requires many trades to stabilize. At ~47 bars/pair these are Nyquist-starved.

- **Group E (Prediction-market/trade-dynamics, 7 features): Droppable.**
  `longshot_score` is a single feature based on probabilistic calibration that has not shown consistent predictive value. `boundary_distance` has questionable value at N=6,802. Trade size features (max_trade_size, hours_since_last_trade) are sparse and noisy.

**Predicted minimum sufficient set:** Groups A + B = 25 features (49% of 51). If confirmed, this is a parsimony finding: "25 features carry the full edge; 26 features are dead weight at N=6,802."

---

## 2. Feature Groups (complete, non-overlapping)

All 51 features are assigned to exactly one group. No feature appears in more than one group.

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

**Total:** 15 + 10 + 6 + 13 + 7 = 51 ✓

---

## 3. Three-Way Split Design

**Problem:** Running 10 ablation configurations (5 groups × 2 models) on the same test set risks p-hacking the test-set ordering. We must protect the final test set from selection bias.

**Solution — temporal three-way split:**

```
Original train (6,802 rows) → [train_proper | ablation_holdout]   — 85/15 temporal split
Original test  (1,673 rows) → final_test                          — UNTOUCHED
```

Expected sizes:
- `train_proper`: ~5,780 rows (earliest 85% of train.parquet) — training only
- `ablation_holdout`: ~1,020 rows (latest 15% of train.parquet) — LOGO selection metric ONLY
- `final_test`: 1,673 rows (test.parquet) — FROZEN until after selection is complete

**Discipline:**
1. All LOGO configurations are trained on `train_proper`, evaluated on `ablation_holdout`
2. Best-performing feature set is selected based on `ablation_holdout` scores alone
3. Selected set is re-trained on `train_proper + ablation_holdout` (full original train), then evaluated ONCE on `final_test`
4. Final-test number is the reportable P&L; ablation-holdout numbers populate the ablation table but are noted as selection metric (not generalization metric)

**Temporal ordering:** Globally sorted by `time_idx` before split; no shuffling at any stage.

---

## 4. Models Tested

| Model | Hyperparameters | Rationale |
|---|---|---|
| Linear Regression | Default scikit-learn (no regularization) | Interpretable baseline; matches headline table |
| XGBoost | depth=3, n_estimators=100, learning_rate=0.01 | Finding 13 optimal hyperparameters |

No other models are tested in this ablation to avoid multiple comparisons explosion.

---

## 5. Primary Metric

**Selection metric (ablation_holdout):** P&L @ 2pp transaction fees

Formula: for each prediction where `|pred| > 0.02`, take the trade. P&L = sum of (|actual| - 0.02) if correct direction, else -(|actual| + 0.02).

**Secondary metrics (reported but not used for selection):**
- RMSE on ablation_holdout
- Directional accuracy on ablation_holdout

**Reportable generalization metric:** Final-test P&L for the selected minimum sufficient set (one number, evaluated once).

---

## 6. Success Criteria for "Load-Bearing" Classification

A group is classified as **load-bearing** if AND ONLY IF ALL THREE conditions hold on `ablation_holdout`:

1. 95% CI of delta-P&L is entirely below zero (delta < 0 means dropping hurts)
2. |mean delta-P&L| > $10
3. Directional accuracy drops > 2 percentage points

A group is classified as **inconclusive** if the 95% CI of delta-P&L straddles zero with |mean| > $10. This is reported honestly rather than claimed as droppable.

A group is classified as **droppable** if the 95% CI of delta-P&L straddles zero with |mean| ≤ $10.

---

## 7. Reporting Pre-Commitment

> "ALL 12 LOGO configurations will be reported in the final ablation table regardless of outcome. No configurations will be retrospectively added or removed. The table will include both LR and XGBoost baselines with their respective delta-P&L and 95% CIs. Unfavorable results (groups predicted to be droppable but found to be load-bearing) will be reported without hedging."

The 12 configurations are:
- LR × {baseline (all 51), drop_A, drop_B, drop_C, drop_D, drop_E} = 6 configs
- XGBoost × {baseline (all 51), drop_A, drop_B, drop_C, drop_D, drop_E} = 6 configs

---

## 8. Final-Test Discipline

> "The final_test split (test.parquet, 1,673 rows) is not touched until after ablation-holdout selection is frozen. The selected feature set is re-trained on train_proper + ablation_holdout (original full train), then evaluated ONCE on final_test. This number is reported in §5.10 of the paper as the generalization result. No further tuning occurs after this evaluation."

---

## 9. Bootstrap CI Methodology

**Procedure:**
1. For baseline config: compute per-trade P&L series on `ablation_holdout`
2. For each drop-group config: compute per-trade P&L series on `ablation_holdout`
3. Paired bootstrap: resample trade indices with replacement, 1,000 iterations
4. Per iteration: compute delta = P&L(drop_config[idx]) - P&L(baseline[idx])
5. Report 95% CI as [2.5th percentile, 97.5th percentile] of the 1,000 delta values

**Why 1,000 resamples:** Standard for trading P&L CIs; gives smooth percentile estimates.

**Baseline configs** have delta_pnl = 0.0 and ci = "—" (no delta defined for baseline vs itself).

---

## 10. Date and Signatories

**Pre-registered:** 2026-04-22
**Authors:** Ian Sabia, Alvin Jang

This document is committed to git BEFORE `experiments/run_feature_ablation.py` is created. The commit timestamp on this file is the evidence for ABLA-01 (pre-registration guard).
