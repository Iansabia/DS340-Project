---
phase: 13-ensemble-formalization
plan: 03
subsystem: paper
tags: [paper, finding, ensemble, concordance-audit, weight-sweep, p4-guard, evidence-based]

# Dependency graph
requires:
  - phase: 13-ensemble-formalization
    provides: experiments/results/ensemble/summary.json — 4 variants + 11-point weight sweep (Plan 13-02)
  - phase: 13-ensemble-formalization
    provides: experiments/figures/ensemble_weight_sweep.png (Plan 13-02)
  - phase: 13-ensemble-formalization
    provides: EnsemblePredictor class reference (Plan 13-01)
provides:
  - PAPER_DRAFT.md §5.11 Ensemble Formalization — 4-variant comparison table with filtered/unfiltered/rejected P&L columns and weight-sweep figure reference
  - PAPER_DRAFT.md §4.4 updated — evidence-based ensemble justification with concordance audit numbers
  - FINDINGS.md Finding 26 — ensemble weight immateriality ($4.68 sweep span) + concordance filter P4 audit (rejected trades net profitable on 3/4 filtered variants)
  - Empirical closure of the ensemble evidence loop for the paper submission
affects:
  - Phase 14 (paper finalization consumes §5.11 and Finding 26 for Table refinement and abstract)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Paper section numbers sourced exclusively from results/ensemble/summary.json — zero fabrication, zero placeholders"
    - "P4 concordance-trap empirical disclosure: §5.11 reports BOTH filtered and unfiltered P&L in the same table and explicitly states rejected trades were profitable in 3/4 variants"
    - "§4.4 cites §5.11 numbers directly (weight sweep span $4.68, rejection rate 4.80%, per-trade Sharpe filtered 0.455 vs unfiltered 0.437) — not qualitative claims"

key-files:
  created:
    - .planning/phases/13-ensemble-formalization/13-03-SUMMARY.md
  modified:
    - PAPER_DRAFT.md
    - FINDINGS.md

key-decisions:
  - "§5.11 reports per-trade Sharpe (0.437 unfiltered vs 0.455 filtered) to quantify the concordance filter's Sharpe boost — directly exposes the P4 trap magnitude numerically rather than just flagging it"
  - "§5.11 framing follows the RESEARCH.md honest-finding template: state the weight immateriality first, then the concordance filter as the true discriminator, then the P4 flag with dollar cost — prevents the filter from being described as pure risk control"
  - "§4.4 update kept to one paragraph addition (not a full rewrite) — preserves existing live-architecture narrative while adding the evidence-based justification sentence; live strategy.py reference explicitly calls out ENSM-05 guard"
  - "Finding 26 uses the exact Phase 12 Finding 25 formatting (title/core-finding/implications structure) rather than a novel format — keeps FINDINGS.md style consistent and scanable for the paper writer"
  - "Did not add a new figure — §5.11 references the existing experiments/figures/ensemble_weight_sweep.png via caption per ENSM-04; no new matplotlib code needed"

patterns-established:
  - "Paper-prose-from-JSON pattern: load summary.json, populate a mental template, never hand-derive numbers — Phase 14 polish work can mechanically cross-check every §5.11 number against the JSON"
  - "Human-verify checkpoint as the final gate before docs commit: reviewer reads the prose before SUMMARY.md is written, so any correction happens in the same plan rather than a new one"

requirements-completed: [ENSM-07]

# Metrics
duration: 48min
completed: 2026-04-23
---

# Phase 13 Plan 03: Paper §5.11 Ensemble Formalization + §4.4 Update + Finding 26 Summary

**PAPER_DRAFT.md §5.11 added with 4-variant ensemble table (filtered/unfiltered/rejected P&L + rejection rate + P4 flag), §4.4 updated with evidence-based ensemble justification citing the $4.68 weight-sweep span, and FINDINGS.md Finding 26 documenting ensemble weight immateriality and the empirically-flagged concordance P4 trap — closing Phase 13's paper evidence loop.**

## Performance

- **Duration:** ~48 min (includes human-verify checkpoint review window)
- **Started:** 2026-04-23T16:01:16Z (Task 1 commit)
- **Completed:** 2026-04-23T16:49:25Z (final docs commit)
- **Tasks:** 3 (2 auto + 1 human-verify checkpoint)
- **Files created:** 1 (this SUMMARY)
- **Files modified:** 2 (PAPER_DRAFT.md, FINDINGS.md)

## Accomplishments

- PAPER_DRAFT.md §5.11 Ensemble Formalization written with the full 4-variant comparison table: each row populated directly from `experiments/results/ensemble/summary.json` (variant name, members, concordance mode, # trades filtered/unfiltered, P&L filtered/unfiltered, rejection rate, rejected P&L, P4 flag). No placeholder values; no approximations.
- §5.11 explicitly cites the 11-point weight sweep: "filtered P&L spans only $4.68 across LR-weight 0.0 → 1.0 ($+199.54 to $+204.22)" — matches summary.json to the cent.
- §5.11 references `experiments/figures/ensemble_weight_sweep.png` with a full caption connecting the near-flat curve to the immateriality claim.
- §5.11 honestly reports the P4 flag: variants (b), (c), (d) all have profitable rejected trades ($+1.95, $+9.52, $+13.08) and the section explicitly states the filter "trades real P&L for variance reduction" rather than burying the cost.
- PAPER_DRAFT.md §4.4 updated with a new evidence-based paragraph: cites weight sweep span ($4.68), per-trade Sharpe filtered vs unfiltered (0.455 vs 0.437), rejection rate (4.80%), and rejected P&L ($+1.95). Includes the ENSM-05 guard statement: "src/live/strategy.py untouched throughout Phase 13."
- FINDINGS.md Finding 26 appended with the full honest assessment: weight immateriality ($4.68 span), concordance audit numbers per variant, P4 flag status, and the implication that the LR+XGB concordance filter is evidence-based risk control, not a cherry-picked configuration.
- Human-verify checkpoint passed: user reviewed the prose and approved ("approved"); no corrections needed.
- ENSM-05 guard verified one last time: `git diff src/live/strategy.py` empty throughout Phase 13.

## Task Commits

Each task was committed atomically:

1. **Task 1: Write PAPER_DRAFT.md §5.11 and update §4.4** — `09345d8` (docs)
2. **Task 2: Add Finding 26 to FINDINGS.md** — `7b1b602` (docs)
3. **Task 3: Human-verify checkpoint** — APPROVED (verification-only, no commit)

**Plan metadata:** final commit on this plan (this SUMMARY.md + STATE.md + ROADMAP.md + REQUIREMENTS.md update).

## Files Created/Modified

- `PAPER_DRAFT.md` — §5.11 Ensemble Formalization section added after §5.10 Feature Ablation with 4-variant table, weight-sweep figure reference, P4 flag disclosure, and interpretation paragraph. §4.4 Live System Architecture updated with a paragraph citing concordance audit numbers and the ENSM-05 guard.
- `FINDINGS.md` — Finding 26 appended (after Finding 25) documenting ensemble weight immateriality, concordance audit, and live-system implications.
- `.planning/phases/13-ensemble-formalization/13-03-SUMMARY.md` — this file.

## Decisions Made

- **Report per-trade Sharpe numerically in §4.4 (0.437 vs 0.455), not just qualitatively.** The P4 trap's magnitude is most visible as a Sharpe delta, so we exposed that number explicitly rather than only showing dollar P&L. Lets reviewers see that the filter's Sharpe improvement is small.
- **Frame §5.11 as "weight is noise, filter is the discriminator, filter has a cost."** This three-beat structure follows the RESEARCH.md honest-finding template and prevents the concordance filter from being described as pure risk control without acknowledging its real-P&L cost.
- **§4.4 gets a one-paragraph addition, not a rewrite.** Preserves the existing live-architecture narrative (volume threshold, pair routing, etc.) while adding the evidence-based ensemble justification as a standalone paragraph that can be moved or extended later without touching surrounding text.
- **Finding 26 mirrors Finding 25's format exactly.** Same title-with-date convention, same core-finding → implications structure. Keeps FINDINGS.md a scanable reference document rather than a prose collection.
- **No new figure generated.** The weight-sweep figure from Plan 13-02 (`experiments/figures/ensemble_weight_sweep.png`) is already publication-ready; §5.11 references it via caption per ENSM-04 without re-rendering.

## Deviations from Plan

None — plan executed exactly as written.

The plan specified the exact §5.11 table structure, the required §4.4 update content, and the required Finding 26 fields. All three deliverables landed with numbers traceable to `experiments/results/ensemble/summary.json`. The human-verify checkpoint was approved without correction requests.

## Issues Encountered

- **None.** Prose was written directly from summary.json, checkpoint approved on first review, and all verification commands passed.

## Deferred Issues

None from this plan. The 17 pre-existing full-suite test failures logged in `deferred-items.md` (from Plan 13-01) remain out of scope for v1.1 and are recommended for a post-submission tech-debt plan.

## User Setup Required

None — no external services, no credentials, no environment variables. Pure documentation work.

## Self-Check: PASSED

- `PAPER_DRAFT.md` contains §5.11 — VERIFIED (`grep -n "5.11 Ensemble" PAPER_DRAFT.md` returns line 475)
- `PAPER_DRAFT.md` §4.4 references concordance audit numbers — VERIFIED (line 193 contains "concordance filter," "$4.68," "4.80%")
- `FINDINGS.md` contains Finding 26 — VERIFIED (line 563 within Finding 26 body references implication for live system)
- Task 1 commit `09345d8` in git log — FOUND
- Task 2 commit `7b1b602` in git log — FOUND
- `git diff src/live/strategy.py` empty (ENSM-05 guard) — VERIFIED
- `grep -c "5.11\|Ensemble" PAPER_DRAFT.md` returns 6 (non-zero) — VERIFIED
- summary.json has 4 variants and 11 weight-sweep points — VERIFIED
- No "[X]" or "[N]" placeholders remain in §5.11 or §4.4 — VERIFIED (visual check + checkpoint approval)
- Human-verify checkpoint approved — VERIFIED (user response "approved")

## Next Phase Readiness

- **Phase 13 is now COMPLETE.** All 7 requirements (ENSM-01 through ENSM-07) closed. Ensemble evidence loop is sealed: class + tests (13-01) → empirical sweep (13-02) → paper prose + Finding 26 (13-03).
- **Phase 14 (Paper Finalization + Presentation) unblocked.** Phase 14's polish work (SciencePlots styling, abstract trim, Sharpe correction, limitations disclaimer) can begin immediately. §5.11 numbers are already final; no Phase 13 re-runs should be needed during Phase 14.
- **Deadline status:** Apr 27 final submission is 4 days away. Project is at 12 of 12 v1.1 plans complete (100%) pending this final commit; only Phase 14's 12 POL requirements remain.
- **ENSM-05 guard held end-to-end.** Every Phase 13 commit (13-01 RED, 13-01 GREEN, 13-02 sweep, 13-03 paper) was verified to leave `src/live/strategy.py` untouched. EnsemblePredictor → live deployment is deliberately deferred to post-v1.1 per roadmap decision.

---
*Phase: 13-ensemble-formalization*
*Completed: 2026-04-23*
