---
phase: 07-experiments-and-interpretability
plan: 01
subsystem: experiments
tags: [matplotlib, latex, visualization, cross-tier-comparison]

# Dependency graph
requires:
  - phase: 04-evaluation-framework
    provides: Tier 1 baseline result JSONs
  - phase: 05-time-series-models
    provides: Tier 2 GRU/LSTM result JSONs with seed stats
  - phase: 06-rl-and-autoencoder
    provides: Tier 3 PPO-Raw/PPO-Filtered result JSONs with seed stats
provides:
  - Cross-tier summary JSON consolidating all 8 models
  - Publication-quality RMSE bar chart and P&L equity curves
  - LaTeX table ready for Phase 8 paper
affects: [08-paper-and-presentation]

# Tech tracking
tech-stack:
  added: [matplotlib]
  patterns: [post-hoc result synthesis from JSON, tier-based color coding]

key-files:
  created:
    - experiments/run_experiment1_comparison.py
    - experiments/results/experiment1/cross_tier_summary.json
    - experiments/figures/experiment1_rmse_bar.png
    - experiments/figures/experiment1_pnl_curves.png
    - experiments/figures/experiment1_comparison_table.tex
  modified: []

key-decisions:
  - "Horizontal bar chart for RMSE (easier to read model names than vertical)"
  - "Distinct line styles per tier (solid/dashed/dotted) in equity curves for grayscale readability"
  - "Bold best-in-column values in LaTeX table for quick scanning"

patterns-established:
  - "Experiment script pattern: load JSONs -> produce summary + figures + LaTeX, no retraining"

requirements-completed: [EXP-01]

# Metrics
duration: 3min
completed: 2026-04-06
---

# Phase 7 Plan 01: Experiment 1 Cross-Tier Comparison Summary

**Cross-tier comparison formalized with RMSE bar chart, P&L equity curves, LaTeX table, and consolidated JSON for all 8 models across 3 tiers**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-06T14:27:01Z
- **Completed:** 2026-04-06T14:30:03Z
- **Tasks:** 1
- **Files created:** 5

## Accomplishments
- Consolidated all 8 model results (4 Tier 1 + 2 Tier 2 + 2 Tier 3) into a single summary JSON with RMSE ranking and tier labels
- Generated RMSE bar chart (72KB) color-coded by tier with error bars for seeded Tier 2/3 models
- Generated overlaid P&L equity curves (223KB) showing all 8 models on same axes with tier-distinct line styles
- Produced LaTeX table with bold best-in-column values and horizontal rules between tiers, ready for paper
- XGBoost (RMSE=0.2857) confirmed as best overall; PPO-Filtered worst (RMSE=0.3268, P&L=4.6) -- validates complexity thesis

## Task Commits

Each task was committed atomically:

1. **Task 1: Create Experiment 1 cross-tier comparison script** - `e00dd766` (feat)

## Files Created/Modified
- `experiments/run_experiment1_comparison.py` - Main script: loads tier JSONs, produces all outputs
- `experiments/results/experiment1/cross_tier_summary.json` - Consolidated 8-model summary with tier labels and RMSE ranking
- `experiments/figures/experiment1_rmse_bar.png` - Horizontal bar chart of RMSE by model, color-coded by tier
- `experiments/figures/experiment1_pnl_curves.png` - Overlaid equity curves for all 8 models
- `experiments/figures/experiment1_comparison_table.tex` - LaTeX table with bold best values and tier separators

## Decisions Made
- Used horizontal bar chart for RMSE (model names are long, horizontal labels more readable)
- Added tier-distinct line styles (solid/dashed/dotted) to equity curves for grayscale print readability
- Error bars shown only for Tier 2/3 models (Tier 1 has deterministic single-run results)
- LaTeX table uses `\midrule` between tier groups for visual separation

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- LaTeX table and figures ready for Phase 8 paper integration
- Summary JSON available for any further cross-tier analysis
- Same pattern (load JSON -> produce figures) applies to Experiment 2 (lookback) and Experiment 3 (threshold)

---
*Phase: 07-experiments-and-interpretability*
*Completed: 2026-04-06*
