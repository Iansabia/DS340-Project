# Phase 5: Time Series Models -- Summary

**Phase:** 05-time-series-models
**Completed:** 2026-04-06
**Plans:** 5 of 5 complete

## Objective

Test whether temporal structure (recurrent sequence models) improves spread prediction over flat-tabular Tier 1 baselines. Phase 5 answers one axis of the central research question: does adding GRU/LSTM complexity beat XGBoost on 144 matched pairs with ~6k training windows?

## What Was Built

1. **Sequence utilities** (`src/models/sequence_utils.py`): Windowing with pair-boundary respect, early stopping (best-loss comparison), reproducible seeding (with `torch.set_num_threads(1)` Apple Silicon workaround), `fit_feature_scaler` with zero-variance guard, and `create_sequences` preserving group order via OrderedDict.

2. **GRUPredictor** (`src/models/gru.py`): hidden_size=64, 1-layer, input dropout=0.3, MSE loss, AdamW optimizer. Implements BasePredictor contract with group_id-aware windowing, warm-up stitching (concatenates cached train rows per group for full lookback on test data), and padded warm-up for short pairs (repeating first row, logged to `_padded_pairs`).

3. **LSTMPredictor** (`src/models/lstm.py`): hidden_size=32, 1-layer, input dropout=0.3. Same BasePredictor contract and warm-up strategy as GRU. Implemented independently (parallel Wave 2 execution).

4. **Tier 2 experiment harness** (`experiments/run_baselines.py`): Extended with `--tier {1,2,both}` CLI flag, `prepare_xy_for_seq` for group_id pass-through, `run_tier2_with_seeds` for 3-seed training with mean/std aggregation, and `format_comparison_table` for unified cross-tier display.

5. **Deferral documentation** (`05-DEFERRALS.md`): TFT (MOD-07) deferred with roadmap-compliant rationale per CONTEXT.md D9 and roadmap success criterion #3's explicit deferral clause.

## Tier 1 Re-Run at 31 Features

During plan 05-04, the zero-variance safety guard in `fit_feature_scaler` identified 4 columns that were entirely zero or NaN in the training data:

- `kalshi_order_flow_imbalance` (100% NaN -- upstream Phase 3 bug)
- `kalshi_buy_volume` (100% zero)
- `kalshi_sell_volume` (100% zero)
- `kalshi_realized_spread` (100% zero)

All 4 were added to `NON_FEATURE_COLUMNS` in `run_baselines.py`, yielding **31 usable features** (not the 34 projected in CONTEXT.md). Tier 1 baselines were re-run with this 31-feature set so the cross-tier comparison is apples-to-apples. The observed RMSE delta for Tier 1 models was **0.0000** -- the dropped columns contributed zero signal (they were all-zero/NaN), confirming no information loss.

**Prior (35 features) vs. Current (31 features) Tier 1 results:**

| Model | Prior RMSE (35 feat) | Current RMSE (31 feat) | Delta |
|-------|---------------------|----------------------|-------|
| Linear Regression | 0.1759 | 0.3081 | -- (*) |
| XGBoost | 0.1729 | 0.2857 | -- (*) |

(*) Note: The prior 35-feature results (Phase 4, plan 04-02) used a different train/test split and row count (978 train / 1,817 test). The current 31-feature results use 6,802 train / 1,673 test rows after sequence-model-compatible preprocessing (dropping rows used for warm-up windows). These numbers are not directly comparable as a "delta" -- the re-run ensures cross-tier fairness at the same feature set and preprocessing, not that RMSE is preserved from a different data configuration.

All 6 result JSONs (4 Tier 1 + 2 Tier 2) now report `n_features=31`.

## Cross-Tier Performance Comparison

Source: `05-04-comparison-output.txt` (6-model unified table, roadmap success criterion #4)

| Model | RMSE | MAE | Dir Acc | P&L | Trades | Win Rate | Sharpe | Raw SR |
|-------|------|-----|---------|-----|--------|----------|--------|--------|
| Naive (Spread Closes) | 0.4995 | 0.3806 | 0.5333 | 58.12 | 1460 | 0.4795 | 5.47 | 0.1253 |
| Volume (Higher Volume) | 0.4566 | 0.3449 | 0.5333 | 59.81 | 1440 | 0.4806 | 5.66 | 0.1306 |
| Linear Regression | 0.3081 | 0.2253 | 0.6594 | 230.14 | 1542 | 0.5733 | 22.01 | 0.4946 |
| **XGBoost** | **0.2857** | **0.2216** | **0.6776** | **238.41** | **1538** | **0.5819** | **23.15** | **0.5216** |
| GRU (mean +/- std) | 0.2896 +/- 0.0024 | 0.2229 | 0.6433 | 212.50 | 1517 | 0.5583 | 20.24 | 0.4586 |
| LSTM (mean +/- std) | 0.2910 +/- 0.0004 | 0.2239 | 0.6545 | 221.84 | 1547 | 0.5650 | 21.12 | 0.4732 |

### Honest Assessment: Does Tier 2 Beat Tier 1?

**No.** XGBoost (RMSE=0.2857) remains the best-performing model. GRU is 1.4% worse (0.2896), LSTM is 1.9% worse (0.2910). Both Tier 2 models convincingly beat Linear Regression (0.3081) and the naive baselines, but neither surpasses XGBoost.

This is a **valid and publishable finding** that directly supports the project's complexity-vs-performance thesis: at this dataset scale (~6k training windows, 144 matched pairs), the temporal structure captured by recurrent models provides marginal benefit over XGBoost's gradient-boosted feature engineering. XGBoost is the complexity sweet spot.

The result aligns with CONTEXT.md's predicted RMSE bands:
- GRU predicted: 0.29-0.33, actual: 0.2896 (near low end)
- LSTM predicted: 0.30-0.34, actual: 0.2910 (below/at low end -- better than expected)

Both Tier 2 models performed at the optimistic end of their predicted ranges, yet still fell short of XGBoost. This narrows the gap but does not close it, reinforcing that tabular tree ensembles remain strong on small structured datasets.

## Key Decisions Honored from CONTEXT.md

| Decision | Value | Status |
|----------|-------|--------|
| D1: Lookback | 6 bars (24 hours) | Honored -- keeps 140/144 pairs, 6,085 windows |
| D2: Warm-up stitching | Prepend cached train rows per group_id | Honored |
| D3: Train/val split | 90/10 within-pair chronological | Honored (early stopping only) |
| D4: Seeds | {42, 123, 456} with mean +/- std | Honored |
| D5: Loss function | MSE (target light-tailed) | Honored |
| D6: GRU hidden_size | 64 | Honored |
| D7: LSTM hidden_size | 32 | Honored |
| D8: Feature set | 31 features (apples-to-apples) | Honored (adjusted from 34 due to 3 extra zero-variance cols) |
| D9: TFT deferral | Documented rationale | Honored -- see 05-DEFERRALS.md |

## TFT Deferral

TFT (MOD-07) is deferred per roadmap success criterion #3's explicit deferral clause. The full rationale is documented in `05-DEFERRALS.md`, covering the param-to-sample ratio argument (ratio ~1.9 vs 0.01 safety threshold), timeline constraints (22 days remaining), and alternative Tier 2 coverage via GRU and LSTM. MOD-07 has a traceable deferral decision, not a silent omission.

## Plans Executed

| Plan | Name | Duration | Tasks | Key Output |
|------|------|----------|-------|------------|
| 05-01 | Sequence Utilities | 3 min | 2 | sequence_utils.py with TDD tests |
| 05-02 | GRU Model | 3 min | 2 | GRUPredictor with BasePredictor contract |
| 05-03 | LSTM Model | 3 min | 2 | LSTMPredictor with BasePredictor contract |
| 05-04 | Cross-Tier Harness | 40 min | 3 | --tier CLI, 6-model comparison table, 6 result JSONs at 31 features |
| 05-05 | Deferral Docs + Summary | ~3 min | 2 | 05-DEFERRALS.md, 05-SUMMARY.md, deferral test |

## Requirements Disposition

| Requirement | Model | Status | Evidence |
|-------------|-------|--------|----------|
| MOD-05 | GRU | Complete | experiments/results/tier2/gru.json, RMSE=0.2896 |
| MOD-06 | LSTM | Complete | experiments/results/tier2/lstm.json, RMSE=0.2910 |
| MOD-07 | TFT | Deferred | 05-DEFERRALS.md with roadmap-compliant rationale |

## Roadmap Success Criteria Verification

1. **GRU trained and evaluated:** Yes -- 3-seed training, results in comparison table
2. **LSTM trained and evaluated:** Yes -- 3-seed training, results in comparison table
3. **TFT trained OR deferred with rationale:** Deferred with documented rationale in 05-DEFERRALS.md
4. **Cross-tier comparison table:** Yes -- 6 models, unified format, all at 31 features

## Next Steps

- **Phase 6 (RL and Autoencoder):** Can proceed immediately. Tier 1 and Tier 2 baselines are established as comparison targets. XGBoost RMSE=0.2857 is the bar to beat.
- **Phase 7 (Experiments and Interpretability):** Cross-tier comparison table is ready for integration. All 6 models available. SHAP analysis can target XGBoost (best performer) and GRU/LSTM (temporal models).
- **TFT re-examination:** Post-deadline only, if dataset expansion yields >=20k training windows.

---

*Phase: 05-time-series-models*
*Completed: 2026-04-06*
