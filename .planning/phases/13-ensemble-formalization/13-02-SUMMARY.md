---
phase: 13-ensemble-formalization
plan: 02
subsystem: experiments
tags: [ensemble, concordance-audit, weight-sweep, p4-guard, feature-routing, matplotlib]

# Dependency graph
requires:
  - phase: 13-ensemble-formalization
    provides: EnsemblePredictor(BasePredictor) class (Plan 13-01)
  - phase: 08-environment-and-baseline-verification
    provides: verify_headline.py build()/feature_cols()/simulate_pnl() pipeline
  - phase: 08-environment-and-baseline-verification
    provides: src/utils/seed.set_all_seeds() reproducibility entry point
provides:
  - Empirical P&L numbers for 4 ensemble variants (a)-(d) with concordance audit
  - 11-point LR-weight sensitivity sweep (filtered + unfiltered) on LR+XGB
  - ensemble_weight_sweep.png — publication-ready figure for §5.11
  - Per-member feature routing pattern (flat vs sequence views) as a helper-function recipe callers use at the experiment-script level instead of inside EnsemblePredictor
  - summary.json — machine-readable input for Plan 13-03's paper section
affects:
  - 13-03 (paper §5.11 consumes summary.json verbatim; no re-run needed)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Experiment-script-level per-member feature routing (_needs_seq helper dispatches X_flat vs X_seq by predictor class name) — keeps EnsemblePredictor API single-X while supporting mixed flat/sequence ensembles"
    - "concordance_audit() function computes filtered/unfiltered/rejected P&L in one pass — P4 guard surfaces rejected P&L > 0 as stdout WARNING and summary.json flag"
    - "Weight sweep mutates fresh EnsemblePredictor instances per w (not in-place) to guarantee train-time isolation; set_all_seeds(42) between every fit"

key-files:
  created:
    - experiments/run_ensemble_sweep.py
    - experiments/results/ensemble/summary.json
    - experiments/results/ensemble/a_lr_alone.json
    - experiments/results/ensemble/b_lr_xgb_equal_weight.json
    - experiments/results/ensemble/c_lr_lstm_equal_weight.json
    - experiments/results/ensemble/d_lr_xgb_lstm_strict.json
    - experiments/figures/ensemble_weight_sweep.png
  modified: []

key-decisions:
  - "Per-member feature routing lives in the experiment script (fit_mixed_ensemble + predict_mixed_members helpers), NOT inside EnsemblePredictor — keeps the ensemble class's single-X API intact and avoids breaking any of the 13 tests locked in Plan 13-01"
  - "Variant (a) LR-alone uses a raw LinearRegressionPredictor call (not EnsemblePredictor wrapper) to keep the sanity cross-check with variant (c)'s LR member mathematically trivial — both now use identical fit(X_flat, y) calls"
  - "Sanity cross-check asserts |variant_a.pnl - variant_c.member_lr_pnl| < $2.00 — fires if LR silently gets group_id in variant (c); observed delta was 0.00 (exact match at $+201.69)"
  - "Weight sweep uses FRESH EnsemblePredictor instances per w step (not _weights mutation) — isolates training side effects and matches how a production sweep runner would behave"
  - "P4 flag raised for all 3 filtered variants (b), (c), (d) with rejected P&L of +$1.95, +$9.52, +$13.08 — honest finding: concordance filter converts real P&L to variance reduction"
  - "ENSM-05 guard verified post-hoc: git diff src/live/strategy.py is empty; most recent strategy.py commit (e312e18) predates all Phase 13 commits"

patterns-established:
  - "Mixed-feature-routing helper pair (fit_mixed_ensemble + predict_mixed_members) — copy this pattern into any future experiment that needs an ensemble mixing flat and sequence predictors"
  - "Concordance audit as a pure function of {member_preds dict, actuals, fee} — no class state, no side effects beyond the P4 stdout WARNING"
  - "Weight-sweep figure style: scienceplots IEEE if available, plain matplotlib fallback, dashed zero line, two-line P&L plot at 300 DPI"

requirements-completed: [ENSM-02, ENSM-03, ENSM-04, ENSM-05, ENSM-06]

# Metrics
duration: 2min
completed: 2026-04-23
---

# Phase 13 Plan 02: Ensemble Sweep — 4 Variants, Concordance Audit, 11-Point Weight Sweep

**Empirical ensemble evidence for §5.11: all 4 variants evaluated with filtered/unfiltered/rejected P&L, P4 flag fires on 3/4 filtered variants, weight sweep confirms near-flat P&L across LR weight 0.0–1.0 — honest finding that the concordance filter is the real discriminator, not weight choice.**

## Performance

- **Duration:** ~2 min wall clock (117s)
- **Started:** 2026-04-23T15:54:31Z
- **Completed:** 2026-04-23T15:56:28Z
- **Tasks:** 2 (1 implementation + 1 ENSM-05 verification)
- **Files created:** 7 (runner + 5 result JSONs + 1 figure)
- **Files modified:** 0 (ENSM-05 guard: strategy.py untouched)

## Accomplishments

- `experiments/run_ensemble_sweep.py` (~430 LOC) implements the full ENSM-02/03/04/06 deliverable in a single standalone script that mirrors `verify_headline.py` structure
- 4-variant evaluation with per-member feature routing: LR/XGB consume flat features (50 cols), LSTM consumes sequence features (51 cols including `group_id`). Variant (c) LR-member P&L = $+201.69 exactly matches variant (a) LR-solo P&L — the group-id-leakage guard (RESEARCH.md Pitfall 2) holds
- Concordance audit surfaces the P4 trap empirically: (b) LR+XGB rejects 4.80% of trades with $+1.95 P&L; (c) LR+LSTM rejects 7.03% with $+9.52; (d) LR+XGB+LSTM rejects 11.65% with $+13.08. All three filtered variants fire the "WARNING: concordance filter is rejecting profitable trades" stdout line and carry `"flag_rejected_profitable": true` in summary.json
- 11-point LR-weight sweep (0.0 → 1.0, step 0.1) produces a near-flat P&L curve ($199.54 – $204.22 filtered, $201.63 – $207.93 unfiltered) confirming the RESEARCH.md prediction that weight choice is not material
- `ensemble_weight_sweep.png` saved at 300 DPI with IEEE scienceplots style, dashed zero line, filtered vs unfiltered lines
- ENSM-05 safety guard verified: `git diff src/live/strategy.py` is empty; lines 427-429 still contain `if np.sign(lr_pred) != np.sign(xgb_pred): continue`; last commit touching strategy.py (`e312e18`) predates all Phase 13 commits
- All 13 ensemble unit tests from Plan 13-01 still pass after this plan (no regressions introduced)

## Concordance Audit Table (stdout + summary.json)

| Variant                    | # trades (filtered) | # trades (unfiltered) | Rejection rate | P&L (filtered) | P&L (unfiltered) | P&L (rejected) | P4 flag |
| -------------------------- | ------------------- | --------------------- | -------------- | -------------- | ---------------- | -------------- | ------- |
| (a) LR alone               | 1549                | 1549                  | 0.00%          | $+201.69       | $+201.69         | $+0.00         | ok      |
| (b) LR + XGB equal-weight  | 1489                | 1564                  | 4.80%          | $+202.14       | $+204.09         | $+1.95         | WARN    |
| (c) LR + LSTM equal-weight | 1441                | 1550                  | 7.03%          | $+191.79       | $+201.31         | $+9.52         | WARN    |
| (d) LR + XGB + LSTM strict | 1373                | 1554                  | 11.65%         | $+194.86       | $+207.93         | $+13.08        | WARN    |

Key reading: (d)'s concordance filter rejects 11.65% of potential trades and those rejected trades would have been **net profitable** ($+13.08). Filtered P&L is $13.07 LOWER than unfiltered. The filter trades real P&L for variance reduction — the P4 pitfall, now documented empirically.

## Weight Sweep (stdout + summary.json)

| LR weight | P&L filtered | P&L unfiltered |
| --------- | ------------ | -------------- |
| 0.0       | $+199.54     | $+201.63       |
| 0.1       | $+200.99     | $+204.50       |
| 0.2       | $+202.60     | $+207.21       |
| 0.3       | $+202.61     | $+205.77       |
| 0.4       | $+202.52     | $+206.50       |
| 0.5       | $+202.14     | $+204.09       |
| 0.6       | $+202.64     | $+204.09       |
| 0.7       | $+202.48     | $+203.38       |
| 0.8       | $+202.60     | $+203.11       |
| 0.9       | $+202.80     | $+202.09       |
| 1.0       | $+204.22     | $+201.69       |

Spread of filtered P&L: $4.68 across the entire 0.0 → 1.0 range. Spread of unfiltered: $6.30. Weight choice is effectively noise — the concordance filter is the dominant signal, matching RESEARCH.md's prior expectation.

## Task Commits

1. **Task 1: Write run_ensemble_sweep.py + artifacts** — `3d04ded` (feat)
2. **Task 2: ENSM-05 safety guard verification** — verification-only (no files changed, no commit required; `git diff src/live/strategy.py` empty)

**Plan metadata:** pending (this summary + STATE.md + ROADMAP.md update in the final commit)

## Files Created/Modified

- `experiments/run_ensemble_sweep.py` — 426 LOC. Main runner: `main()` builds train/test, runs variants (a)-(d) via `_run_variant_[a-d]`, runs 11-point weight sweep via `_run_weight_sweep`, prints concordance audit, saves figure and summary.json. Helpers: `fit_mixed_ensemble`, `predict_mixed_members`, `combine_predictions`, `concordance_audit`, `_variant_record`, `_print_audit_table`, `_save_weight_sweep_figure`, `_needs_seq`.
- `experiments/results/ensemble/summary.json` — 4 variants + 11 weight-sweep points. Primary consumer: Plan 13-03 paper §5.11.
- `experiments/results/ensemble/{a,b,c,d}_*.json` — per-variant results files written via `save_results()` for comparison-table compatibility with the existing `results_store` pattern.
- `experiments/figures/ensemble_weight_sweep.png` — 65 KB, 300 DPI, IEEE scienceplots style.

## Decisions Made

- **Per-member feature routing at the script level, not inside EnsemblePredictor.** The ensemble class's single-X contract is test-locked by 13 tests in Plan 13-01. Extending `__init__` to accept a per-member `feature_cols` mapping was an option but would have broken at least 3 of those tests. Instead, the script defines `fit_mixed_ensemble()` and `predict_mixed_members()` helpers that dispatch `X_flat` vs `X_seq` based on `type(member).__name__`. Variants (c) and (d) use this path; variants (a) and (b) use the ensemble class's native `fit(X)/predict(X)` directly.
- **Variant (a) LR-alone bypasses the EnsemblePredictor wrapper.** Used a raw `LinearRegressionPredictor` for variant (a) so the sanity cross-check with variant (c)'s LR member is mathematically identical (both are `LinearRegressionPredictor().fit(X_flat, y_train).predict(X_flat_test)`). This produced the exact match $+201.69 = $+201.69 observed at runtime.
- **Fresh EnsemblePredictor per weight-sweep step, not in-place weight mutation.** RESEARCH.md noted `_weights` mutation is possible (Plan 13-01 decision), but constructing a fresh instance per w guarantees no training-time contamination from a previous w. Also matches how a production sweep runner would behave.
- **Flag all three filtered variants as P4 WARN.** Rejected P&L > 0 triggers stdout WARNING and `flag_rejected_profitable: true` in JSON. This is the honest finding for §5.11.
- **scienceplots import wrapped in try/except.** Keeps the script runnable on systems without scienceplots while preferring IEEE style when available.

## Deviations from Plan

None — plan executed exactly as written.

The plan specified ~100-130 LOC; final script is 426 LOC because helper functions (`_variant_record`, `_print_audit_table`, `_save_weight_sweep_figure`, per-variant runners) were factored out for readability and to keep `main()` linear. Zero behavioral deviations from the plan's verification, acceptance, or done criteria.

## Issues Encountered

- **LSTM warm-up padding for 3 pairs (group_id 17, 21, 25).** During variants (c) and (d), LSTM predict printed `WARN [LSTM]: padding applied for group_id=N, n_rows_available=5, lookback=6` for 3 test pairs where the stitched train+test sequence was shorter than the 6-bar lookback. This is expected behavior of the LSTM's warm-up-stitching path and was present in Phase 10/11 runs as well — not a Phase 13 regression. Logged for transparency; no action taken.
- **LSTM retraining in variant (d) duplicates epochs from variant (c).** Each variant constructs fresh member instances (required for reproducibility and for matching the RESEARCH.md "Pitfall 3: members re-trained with different seeds" mitigation by calling `set_all_seeds(42)` before each variant). This adds ~30 s to total runtime but is a correctness requirement. Documented, not optimized.

## Deferred Issues

None from this plan. Pre-existing full-suite failures (17 items) remain deferred per `deferred-items.md` from Plan 13-01.

## User Setup Required

None — no external services or credentials. Pure in-process numpy/pandas/matplotlib work.

## Self-Check: PASSED

- `experiments/run_ensemble_sweep.py` present — FOUND (426 LOC)
- `experiments/results/ensemble/summary.json` present — FOUND
- `experiments/figures/ensemble_weight_sweep.png` present — FOUND (65 KB)
- Task 1 commit `3d04ded` in git log — FOUND
- `jq '.variants | length' experiments/results/ensemble/summary.json` returns 4 — VERIFIED
- `jq '.weight_sweep | length' experiments/results/ensemble/summary.json` returns 11 — VERIFIED
- `git diff src/live/strategy.py` empty (ENSM-05) — VERIFIED
- strategy.py lines 427-429 still contain `if np.sign(lr_pred) != np.sign(xgb_pred): continue` — VERIFIED
- Sanity cross-check variant (a) == variant (c) LR-member: $+201.69 == $+201.69 — VERIFIED (exact match)
- `pytest tests/models/test_ensemble.py -q` returns 13/13 passing — VERIFIED (no regressions from Plan 13-01)
- P4 flag printed for variants (b), (c), (d) on stdout — VERIFIED

## Next Phase Readiness

- Plan 13-03 (paper §5.11) can consume `experiments/results/ensemble/summary.json` verbatim for table population and narrative framing. No additional runs needed.
- Honest §5.11 framing is already empirically supported:
  - "Ensemble weight choice is not material" — weight sweep spread $4.68 across 11 points
  - "Concordance filter is the primary discriminator" — filtered vs unfiltered differ systematically
  - "Filter converts real P&L to variance reduction (P4 trap)" — rejected trades net profitable in 3/4 filtered variants
- The weight-sweep figure is publication-ready (300 DPI, IEEE style, dashed zero line, two labeled lines).

---
*Phase: 13-ensemble-formalization*
*Completed: 2026-04-23*
