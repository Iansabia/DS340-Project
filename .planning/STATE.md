---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: Extended Evidence & Submission
status: in-progress
stopped_at: Completed 10-01-PLAN.md (250-bar scaling checkpoint) — all 3 tasks done
last_updated: "2026-04-22T21:27:45Z"
last_activity: 2026-04-22 -- Phase 10 plan 01 completed
progress:
  total_phases: 7
  completed_phases: 3
  total_plans: 5
  completed_plans: 5
  percent: 71
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-17)

**Core value:** Empirically answer whether model complexity improves cross-platform prediction market arbitrage detection
**Current focus:** v1.1 -- Extended Evidence & Submission (Phase 8 ready to plan)

## Current Position

Phase: 10 of 14 (250-Bar Scaling Checkpoint)
Plan: 1 of 1 in current phase (complete)
Status: Phase 10 complete
Last activity: 2026-04-22 -- Phase 10 plan 01 executed

Progress: [███████░░░] 71%

## Performance Metrics

**Velocity (from v1.0):**
- Total plans completed: 33
- Average duration: 5.5 min
- Total execution time: ~3 hours

**v1.1 Velocity:**
- Total plans completed: 5
- Average duration: ~3 min
- Total execution time: ~15 min

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [v1.1 Roadmap]: Phase 8 gates all downstream; venv may need Python 3.12 rebuild
- [v1.1 Roadmap]: TFT time-box 1 day; negative result is paper-worthy (Option B -- always proceed)
- [v1.1 Roadmap]: Reconciliation window Apr 11-25 (post pair_id fix); schema bug confirmed resolved
- [v1.1 Roadmap]: EnsemblePredictor NOT wired into live strategy.py during v1.1
- [v1.1 Roadmap]: Phases 9/10/11/12 parallelize after Phase 8; Phase 13 waits on 11; Phase 14 terminal
- [Phase 08-environment-and-baseline-verification]: Python 3.14.3 venv compatible with pytorch-forecasting 1.7.0, quantstats 0.0.81, SciencePlots 2.2.1; no Python 3.12 rebuild required
- [Phase 08-02]: run_baselines.py now calls compute_derived_features() + select_dtypes(['number']) to align with 51-feature pipeline used in verify_headline.py
- [Phase 08-02]: PPO imports in run_baselines.py are lazy (deferred) to allow Tier 1 to run without stable_baselines3 installed
- [Phase 08-02]: set_all_seeds(42) is the standard entry point seed call for all experiment scripts; P&L reconciliation skipped in ENV-05 check (different profit sim implementations)
- [Phase 09-live-vs-backtest-reconciliation]: profit_sim.simulate_profit is canonical fee function for reconciliation (threshold-only model)
- [Phase 09-live-vs-backtest-reconciliation]: Shadow simulation tracking error of $12.06 is a paper finding: model directional anti-correlation with live entry logic
- [Phase 09-live-vs-backtest-reconciliation]: CLI wrapper delegates 100% of analysis to src/analysis/reconciliation.py — no inline logic
- [Phase 09-live-vs-backtest-reconciliation]: Section 5.9 numbers sourced exclusively from experiments/results/reconciliation/summary.json — no fabrication
- [2026-04-22]: Phase 9 re-run on 8-day snapshot (10,154 trades, 577 pairs); §5.9 and Finding 23 updated; crypto regime flip and TAKE_PROFIT tail (n=88) documented
- [Phase 10-250-bar-scaling-checkpoint]: GRU and LSTM trained successfully at 250 bars (torch env working); XGBoost > LR > GRU > LSTM ranking invariant across 5x data growth (50/100/250 bars)
- [Phase 10-250-bar-scaling-checkpoint]: Auto-trigger root cause: run_data_scaling.py --auto reads only train.parquet (max 141 bars/pair); manual --bars-per-pair 250 required
- [Phase 10-250-bar-scaling-checkpoint]: PYTHONPATH must be set to project root when running scripts from venv

### Pending Todos

None yet.

### Blockers/Concerns

- [Risk] Python 3.14 may lack pytorch-forecasting wheels; Phase 8 dry-run gates everything
- [Risk] TFT at 6,802 rows is below safe training thresholds; 1-day time-box enforced
- [Deadline] April 27 submission -- 10 days from roadmap creation

## Session Continuity

Last session: 2026-04-22T21:27:45Z
Stopped at: Completed 10-01-PLAN.md (250-bar scaling checkpoint) — all tasks done
Resume file: None
Next action: Phase 11 planning
