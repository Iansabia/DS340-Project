---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: Extended Evidence & Submission
status: completed
stopped_at: Completed 14-03-PLAN.md — README + Marp slides shipped; two fix cycles closed residual paper-integrity gaps; user approved at human-verify checkpoint. Phase 14 complete; submission-ready.
last_updated: "2026-04-23T21:52:35.466Z"
last_activity: 2026-04-23 -- Phase 14 plan 03 executed (README.md reproduction table, slides.md + slides.pdf, §5.12/§5.13 writeups, Appendix B placeholder repair, abstract oil qualifier, §6.4 item 9, 56-file AI attribution)
progress:
  total_phases: 7
  completed_phases: 7
  total_plans: 15
  completed_plans: 15
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-17)

**Core value:** Empirically answer whether model complexity improves cross-platform prediction market arbitrage detection
**Current focus:** v1.1 -- Extended Evidence & Submission (Phase 14 paper finalization in progress)

## Current Position

Phase: 14 of 14 (Paper Finalization + Presentation) — COMPLETE
Plan: 3 of 3 complete (IEEE figures + abstract scrub done; paper integrity pass done; README + slides + final polish done)
Status: Phase 14 COMPLETE; v1.1 milestone COMPLETE. Repository submission-ready — README reproduction table committed, slides.md/slides.pdf rendered for 4-minute lightning talk, §5.12/§5.13 writeups added, Appendix B placeholders repaired, abstract oil claim narrowed, §6.4 item 9 documents commodity-matching limitation, 56 src/*.py files carry AI-attribution headers (POL-09 substantive).
Last activity: 2026-04-23 -- Phase 14 plan 03 executed (README.md reproduction table, slides.md + slides.pdf, §5.12/§5.13 writeups, Appendix B placeholder repair, abstract oil qualifier, §6.4 item 9, 56-file AI attribution)

Progress: [██████████] 100%

## Performance Metrics

**Velocity (from v1.0):**
- Total plans completed: 33
- Average duration: 5.5 min
- Total execution time: ~3 hours

**v1.1 Velocity:**
- Total plans completed: 7
- Average duration: ~4 min
- Total execution time: ~23 min

| Phase                     | Plan | Duration | Tasks | Files |
| ------------------------- | ---- | -------- | ----- | ----- |
| 13-ensemble-formalization | 02   | 2min     | 2     | 7     |

*Updated after each plan completion*
| Phase 13-ensemble-formalization P03 | 48min | 3 tasks | 3 files |
| Phase 14-paper-finalization-presentation P01 | 20min | 3 tasks | 17 files |
| Phase 14-paper-finalization-presentation P02 | 5min | 3 tasks | 2 files |
| Phase 14-paper-finalization-presentation P03 | 90min | 3 tasks | 60 files |

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
- [Phase 11-01-tft-training]: TFT does NOT beat GRU at N=6802 (RMSE 0.3262 vs 0.2928) — documented negative result per Option B; extends simplicity-wins thesis to transformers
- [Phase 11-01-tft-training]: Use lightning.pytorch Trainer (NOT pytorch_lightning) with pytorch_forecasting 1.7.0 — different LightningModule base class causes TypeError
- [Phase 11-01-tft-training]: Round-based batch predict for TFT: K rounds x all_groups together (~11 rounds) instead of 1673 per-row calls; 150x speedup
- [Phase 11-01-tft-training]: compute_derived_features drops group_id (not in OUTPUT_COLUMNS); must re-attach from raw parquet before prepare_xy_for_seq
- [Phase 11-01-tft-training]: GroupNormalizer(transformation=None) confirmed correct for signed spread-change targets; softplus causes degenerate predictions
- [Phase 11]: TFT negative result (Branch B): converged=False — 4-variant ensemble for Phase 13 (no TFT variant)
- [Phase 11]: VSN attention is healthy (entropy=2.656, not degenerate) even when predictive performance is weak — data-volume bottleneck, not architecture failure
- [Phase 12-01-feature-ablation]: Pre-registration via git commit ordering: ablation_protocol.md committed at b15534b before run_feature_ablation.py at 46b253a (ABLA-01)
- [Phase 12-01-feature-ablation]: All 51-feature groups classified as droppable on ablation_holdout (1,021 rows) — CIs all straddle zero, |delta| < $10; insufficient statistical power at this holdout size
- [Phase 12-01-feature-ablation]: Ablation holdout P&L (+$56.54 LR, +$54.00 XGB) differs from full train-test headline (+$232) due to smaller evaluation window (1,021 vs 6,800 rows)
- [Phase 12-01-feature-ablation]: final_test (test.parquet, 1,673 rows) untouched — frozen for one-shot evaluation in §5.10 paper section
- [Phase 12-02-feature-ablation-paper]: §5.10 written with honest power-limitation framing — N=1,021 ablation holdout insufficient to detect effects < $10; all 51 features retained per pre-registered protocol
- [Phase 12-02-feature-ablation-paper]: Finding 25 documents pre-registered null result; ablation should be re-run at 250+ bars/pair for tighter inference (§7 item 8 added)
- [Phase 13-01-ensemble-formalization]: EnsemblePredictor uses all-members-agree concordance semantics ('strict' mode: np.sign(preds) constant across member axis) — generalizes strategy.py's binary LR/XGB check to N members without pairwise-voting complexity
- [Phase 13-01-ensemble-formalization]: Weight normalization inside predict(), not __init__() — lets sweep runner mutate _weights in-place without re-instantiating (simplifies Plan 13-03)
- [Phase 13-01-ensemble-formalization]: set_all_seeds(seed) called inside fit(), not constructor — constructor stays a pure metadata builder
- [Phase 13-01-ensemble-formalization]: No custom pickle hooks — BasePredictor.save/load handles ensemble serialization natively because all supported member types are already picklable
- [Phase 13-01-ensemble-formalization]: ENSM-05 guard held — src/live/strategy.py untouched; ensemble wiring deferred to post-v1.1 per roadmap decision
- [Phase 13-02-ensemble-formalization]: Per-member feature routing lives at the experiment-script level (fit_mixed_ensemble / predict_mixed_members helpers) — keeps EnsemblePredictor's single-X contract intact and avoids invalidating the 13 tests from Plan 13-01
- [Phase 13-02-ensemble-formalization]: Variant (a) LR-solo P&L = $+201.69 exactly equals variant (c) LR-member P&L (sanity cross-check verified group_id does not leak into LR via variant (c))
- [Phase 13-02-ensemble-formalization]: P4 concordance trap fires on 3/4 filtered variants (rejected P&L +$1.95, +$9.52, +$13.08 for LR+XGB, LR+LSTM, LR+XGB+LSTM) — documented empirically for §5.11
- [Phase 13-02-ensemble-formalization]: Weight sweep P&L spread $4.68 across 11 LR-weight points (0.0→1.0) — weight choice is immaterial; concordance filter is the discriminator
- [Phase 13-02-ensemble-formalization]: ENSM-05 guard re-verified — git diff src/live/strategy.py empty; strategy.py lines 427-429 still contain live concordance check
- [Phase 13-03-ensemble-formalization]: §5.11 reports per-trade Sharpe (0.437 unfiltered vs 0.455 filtered) to quantify concordance filter's Sharpe boost — directly exposes P4 trap magnitude numerically
- [Phase 13-03-ensemble-formalization]: §5.11 framing: weight immateriality first, then concordance filter as true discriminator, then P4 flag with dollar cost — prevents filter being described as pure risk control
- [Phase 13-03-ensemble-formalization]: §4.4 updated via one-paragraph addition (not rewrite); preserves existing architecture narrative; explicitly references ENSM-05 guard (strategy.py untouched)
- [Phase 13-03-ensemble-formalization]: Finding 26 mirrors Finding 25 format exactly; keeps FINDINGS.md scanable rather than a prose collection
- [Phase 14-01-paper-finalization]: Consolidated figure regeneration via scripts/regenerate_figures.py rather than editing 11 separate experiment scripts — keeps IEEE styling in one place and makes regeneration idempotent
- [Phase 14-01-paper-finalization]: TFT VSN weights persisted to experiments/results/tft/vsn_importance.json (via patched extract_tft_heatmap.py) so downstream figure regenerations skip the ~8-minute TFT retrain
- [Phase 14-01-paper-finalization]: Abstract landed at 244 words (6 words of headroom under 250 cap) leaving room for Plan 14-02 adjustments
- [Phase 14-01-paper-finalization]: Figure 2 SCAL-03/POL-08 cap annotation embedded on-figure via red dotted axvline + text ("plateau at N=6,802, fixed pair universe") — readable without body text
- [Phase 14-01-paper-finalization]: Stale "per-trade Sharpe 0.59 / 4.3" claim on §1.4 item 7 + dead §4.6 cross-reference both removed; per-pair ≈ 3.2 (Table 8) is now the consistent headline across abstract, §1.4, §8
- [Phase 14-02-paper-integrity]: [Anonymous] arXiv entry placed as reference #1 (sorts before alphabetical authors); Cont, Kukanov & Stoikov (2014) inserted as #4 to resolve stale in-text citation at §6.2.1 line 536
- [Phase 14-02-paper-integrity]: Rule 1 bug fix during Task 1: §1.4 item 6 stale (Table 6) → (Table 7) for transaction-cost analysis (unrelated to 5.10/5.11 renumber, caught via full grep sweep)
- [Phase 14-02-paper-integrity]: check_paper.sh uses sed -n '/^## X$/,/^## Y$/p' for section ranges not awk '/^## X/,/^## /' — awk range collapses when start pattern also matches the end
- [Phase 14-02-paper-integrity]: Appendix B rewritten in in-text appearance order (11 figures Fig 1-Fig 11); Fig 10 resolves 'Figure 2b' ghost; Fig 11 resolves '[Insert Figure]' placeholder
- [Phase 14-02-paper-integrity]: §6.4 now has 8 items: preserved original 1-5 (POL-08 regression check); added live-cohort truncation (6), category-tagging gaps (7), crypto regime flip (8) from Finding 23
- [Phase 14-03-readme-slides]: README.md is the canonical reproduction map; PAPER_DRAFT.md Appendix A deprecated to a single pointer — prevents drift between paper commands and code
- [Phase 14-03-readme-slides]: Marp CLI (Node toolchain) chosen over Beamer/LaTeX for lightning-talk slides — faster iteration, PDF committed alongside source
- [Phase 14-03-readme-slides]: Two Rule 1/2 fix cycles at human-verify checkpoint — added §5.12 (lookback) and §5.13 (threshold) body writeups to close orphan-table references; repaired Appendix B Fig 10/Fig 11 placeholders missed in 14-02 renumber
- [Phase 14-03-readme-slides]: Abstract oil claim narrowed to "where the matched cohort is representable" — honest scope given ~30% cross-platform matching rate; §6.4 item 9 documents commodity-matching cohort as generalization limit
- [Phase 14-03-readme-slides]: POL-09 upgraded from Acknowledgments-only disclosure to per-file AI-attribution headers on 56 src/**/*.py modules — survivorship-proof evidence of AI involvement

### Pending Todos

None yet.

### Blockers/Concerns

- [Risk] Python 3.14 may lack pytorch-forecasting wheels; Phase 8 dry-run gates everything
- [Risk] TFT at 6,802 rows is below safe training thresholds; 1-day time-box enforced
- [Deadline] April 27 submission -- 10 days from roadmap creation

## Session Continuity

Last session: 2026-04-23T21:52:35.466Z
Stopped at: Completed 14-03-PLAN.md — Phase 14 complete, v1.1 milestone 100%, repository submission-ready (paper + README + slides)
Resume file: None
Next action: Run `/gsd:verify-work` on Phase 14; Alvin readability pass on §1 + §8; render final PDF via pandoc for April 27 submission
