---
phase: 07-experiments-and-interpretability
plan: 04
subsystem: experiments
tags: [shap, treeshap, xgboost, transaction-costs, interpretability, sensitivity-analysis]

# Dependency graph
requires:
  - phase: 07-01
    provides: Cross-tier comparison results (all 8 models evaluated with metrics JSON)
  - phase: 04-evaluation-framework
    provides: XGBoostPredictor, results_store, profit simulation
provides:
  - TreeSHAP feature importance analysis for XGBoost (summary plot, bar plot, CSV)
  - Transaction cost sensitivity analysis (break-even per model, sensitivity plot, JSON)
affects: [08-paper, 07-05]

# Tech tracking
tech-stack:
  added: [shap 0.51.0 (TreeExplainer)]
  patterns: [matplotlib Agg backend for headless plotting, load_all_results for cross-tier aggregation]

key-files:
  created:
    - experiments/run_shap_analysis.py
    - experiments/run_transaction_costs.py
    - experiments/results/shap/xgboost_feature_importance.csv
    - experiments/figures/shap_summary_plot.png
    - experiments/figures/shap_bar_plot.png
    - experiments/results/transaction_costs/sensitivity_results.json
    - experiments/figures/transaction_cost_sensitivity.png
  modified: []

key-decisions:
  - "polymarket_vwap dominates SHAP importance (0.138 vs next 0.016) -- indicates Polymarket price level is the primary driver of XGBoost spread-change predictions"
  - "At realistic Kalshi fees (5-7pp), all Tier 1/2 models remain profitable; naive baselines and PPO-Filtered go negative"
  - "XGBoost has highest break-even at 15.5pp, confirming it as the most fee-resilient model"

patterns-established:
  - "SHAP analysis pattern: train XGBoost with same hyperparams, TreeExplainer on test set, save beeswarm + bar + CSV"
  - "Transaction cost sensitivity pattern: load_all_results across tiers, vary cost 0-10pp, compute break-even"

requirements-completed: [EXP-04, EVAL-03]

# Metrics
duration: 3min
completed: 2026-04-06
---

# Phase 7 Plan 4: SHAP and Transaction Cost Analysis Summary

**TreeSHAP reveals polymarket_vwap as dominant XGBoost feature (0.138 mean |SHAP|); transaction cost sensitivity shows Tier 1/2 models survive 5-7pp Kalshi fees while naive baselines and PPO-Filtered go negative**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-06T18:10:01Z
- **Completed:** 2026-04-06T18:12:42Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- TreeSHAP computed for XGBoost across 31 features on 1673 test samples; top feature polymarket_vwap (0.138) dominates, followed by price_velocity (0.016) and spread_volatility (0.013)
- Transaction cost sensitivity analysis across all 8 models: XGBoost break-even at 15.5pp, Linear Regression at 14.9pp, both well above realistic fee levels
- At 5pp fees: XGBoost nets +161.51, Linear Regression +153.04, GRU +136.65, LSTM +144.49; Naive and Volume go negative; PPO-Filtered deeply negative (-40.34)

## Task Commits

Each task was committed atomically:

1. **Task 1: TreeSHAP analysis for XGBoost** - `15a88cb9` (feat)
2. **Task 2: Transaction cost sensitivity** - `2a171f53` (feat)

## Files Created/Modified
- `experiments/run_shap_analysis.py` - TreeSHAP analysis: trains XGBoost, computes SHAP values, saves plots and CSV
- `experiments/run_transaction_costs.py` - Varies transaction cost 0-10pp, computes break-even per model, produces sensitivity plot
- `experiments/results/shap/xgboost_feature_importance.csv` - 31 features sorted by mean |SHAP| value
- `experiments/figures/shap_summary_plot.png` - Beeswarm summary plot (198KB)
- `experiments/figures/shap_bar_plot.png` - Bar plot of mean |SHAP| per feature (100KB)
- `experiments/results/transaction_costs/sensitivity_results.json` - Per-model P&L at 21 cost levels + break-even costs
- `experiments/figures/transaction_cost_sensitivity.png` - Line plot of net P&L vs transaction cost (210KB)

## Decisions Made
- polymarket_vwap dominating SHAP importance (10x larger than next feature) suggests XGBoost relies heavily on the Polymarket price level as a proxy for spread dynamics
- Break-even analysis validates that Tier 1/2 models (14-16pp break-even) are robust to transaction costs, while PPO-Filtered (0.5pp break-even) has essentially no trading edge
- Used same XGBoost hyperparameters as run_baselines (n_estimators=200, max_depth=4, lr=0.05) for consistent SHAP interpretation

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- SHAP plots and transaction cost sensitivity plot are ready for the paper (Phase 8)
- Feature importance CSV can be referenced in the interpretability section
- Transaction cost results provide the economic viability argument: simpler models survive real fees, RL complexity is not justified
- One more plan (07-05) remains in Phase 7

## Self-Check: PASSED

All 7 output files verified present. Both task commits (15a88cb9, 2a171f53) confirmed in git log.

---
*Phase: 07-experiments-and-interpretability*
*Completed: 2026-04-06*
