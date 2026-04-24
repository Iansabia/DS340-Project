---
phase: 15-live-commodity-matching-engineering-fixes
plan: 01
subsystem: matching
tags: [kalshi-api, commodity-discovery, root-cause-diagnostic, active-matches]

requires:
  - phase: 14-paper-finalization-presentation
    provides: "§6.4 item 9 (commodity-matching cohort limitation) + canonical false-match reference"

provides:
  - "Root-cause diagnostic (DIAGNOSTIC.md) naming exact pipeline stage that drops daily WTI / Brent commodity markets"
  - "Live Kalshi /series evidence (Probe 1-3) enumerating 47 Commodities-category series"
  - "On-disk active_matches.json inventory (Probe 4) with canonical 113,287 count + 836 oil-adjacent breakdown"
  - "Gap set (Probe 5): 6 oil series + ~125 open markets visible on Kalshi but absent from pipeline"
  - "5 hypotheses evaluated (H1 CONFIRMED, H2 PARTIAL, H3 REJECTED, H4 REJECTED, H5 PARTIAL)"
  - "Fix checklist handed to Plan 15-03 with exact file paths and line numbers"

affects:
  - 15-02 (asset-class guardrail — receives H4-complement fix recommendation)
  - 15-03 (discovery category + classifier rules — receives Fix 1, Fix 2, Fix 3)
  - 15-04 (metrics / monitoring)
  - 15-05 (permanent record in paper §6.4 supplement)

tech-stack:
  added: []
  patterns:
    - "Evidence-first root-cause diagnostics (probe live API before proposing code changes)"
    - "Hypothesis ranking with explicit VERDICT labels (CONFIRMED / REJECTED / PARTIAL)"

key-files:
  created:
    - ".planning/phases/15-live-commodity-matching-engineering-fixes/15-01-DIAGNOSTIC.md"
  modified: []

key-decisions:
  - "H1 CONFIRMED as the blocker: KALSHI_DISCOVERY_CATEGORIES omits 'Commodities' — Kalshi migrated oil/brent/grain/metal series into a dedicated category after the taxonomy change"
  - "H3 REJECTED on Polymarket-absence grounds: the existing KXWTIMAX-26DEC31-T130 ↔ Bitcoin-130k false match proves Polymarket has candidates; the matcher is picking the wrong ones because the correct oil counterparts never enter the candidate pool"
  - "H4 REJECTED for primary gap: filter can't over-reject markets that never reach it (discovery drops them upstream)"
  - "H5 PARTIAL: src/features/category.py _RULES missing KXBRENT* + KXWTIH/E/EU/MINM/MIN variants — these would be classified as 'other' and get 3x threshold penalty after H1 lands"
  - "Fix prioritisation: single-line tuple edit (Fix 1) unblocks 7 series + ~125 markets; classifier extension (Fix 2) prevents downstream threshold penalty; quality-filter guardrail (Fix 4) deferred to Plan 15-02 as belt-and-braces"

patterns-established:
  - "Diagnostic-first workflow: write the evidence file BEFORE the fix, commit it, then hand it to the implementation plan — mirrors TDD's RED-before-GREEN rule for research work"
  - "Two-part diagnostic structure: ## Evidence from <source> + ## Root-Cause Hypotheses (with VERDICT table) — readable, grep-friendly, handoff-ready"

requirements-completed:
  - COM-01

duration: 4min
completed: 2026-04-23
---

# Phase 15 Plan 01: Commodity Discovery Gap Diagnostic Summary

**Kalshi discovery loop omits the 'Commodities' category (47 live series, ~125 open oil markets), blocking daily WTI / Brent from reaching active_matches.json — root cause pinned to src/live/market_discovery.py:249 for Plan 15-03 single-line fix.**

## Performance

- **Duration:** 4 min
- **Started:** 2026-04-24T00:08:47Z
- **Completed:** 2026-04-24T00:13:11Z
- **Tasks:** 2
- **Files created:** 1 (15-01-DIAGNOSTIC.md)

## Accomplishments

- Live Kalshi `/series` probe confirmed `Commodities` category exists with 47 series (including all gap-set tickers: KXWTI, KXWTIW, KXBRENTMON, KXBRENTW, KXWTIMINM, KXWTIMIN).
- Cross-reference against `data/live/active_matches.json` (113,287 entries, 836 oil-adjacent) produced the concrete gap set: **6 oil series + ~125 open markets visible on Kalshi but zero reaching the pipeline**.
- Five hypotheses evaluated: **H1 CONFIRMED** (discovery category gap), H2 PARTIAL (/events throttle risk), H3 REJECTED (Polymarket has counterparts), H4 REJECTED for primary gap (filter unreachable from upstream drop), H5 PARTIAL (classifier missing Brent family + WTI variants).
- Fix checklist handed to Plan 15-03: Fix 1 = one-line tuple edit at `src/live/market_discovery.py:249`; Fix 2 = extend `_RULES` in `src/features/category.py`; Fix 3 = /events throttle monitor; Fix 4 (asset-class guardrail) handed separately to Plan 15-02.

## Task Commits

Each task was committed atomically:

1. **Task 1: Enumerate live Kalshi commodity series via /series API** — `fc2fcbd` (docs)
2. **Task 2: Rank root-cause hypotheses with verdict + fix recommendations** — `e01cea4` (docs)

_TDD not applicable — research / diagnostic work, not code changes._

## Files Created/Modified

- `.planning/phases/15-live-commodity-matching-engineering-fixes/15-01-DIAGNOSTIC.md` — 7 KB root-cause diagnostic with Probe 1-5 evidence tables, hypothesis rankings, fix checklist, and handoff table to Plan 15-03 / 15-02.

## Decisions Made

- **Kept plan-specified canonical numbers verbatim** (113,287 total entries, 395 oil-adjacent 380/15 split) even though the live snapshot has drifted to 836 oil-adjacent 602/234 split since the 2026-04-11 paper citation. Documented both the paper-snapshot numbers (as required by acceptance criteria) and the current refresh (for Plan 15-03 planning).
- **Framed H4 as REJECTED for primary gap, not rejected outright** — the existing KXAAAGAS evictions show the filter IS operating on commodity tickers, but for the *daily WTI / Brent* primary-gap case, the filter never sees those pairs (discovery drops them first). Keeping the H4 analysis prevents Plan 15-03 from wasting effort searching for phantom filter bugs.
- **Deferred gas_prices-to-commodity tuple fix in strategy.py** (`is_commodity` list at `src/live/strategy.py:405-407`) as out of scope. That is a policy decision (should AAA retail gas get the commodity-low-threshold treatment?), not a correctness bug.

## Deviations from Plan

None — plan executed exactly as written. Two tasks, one file, verification criteria (automated grep checks) all passed on first attempt.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- **Plan 15-02** can proceed independently (already in-flight per `15-02-SUMMARY.md` on disk; asset-class guardrail landed as Rule 10).
- **Plan 15-03** has an unambiguous fix checklist: start with Fix 1 (one-line tuple edit), add Fix 2 (classifier rules extension) in the same plan, monitor Fix 3 during verification.
- **No blockers.** The diagnostic is evidence-backed, grep-searchable, and leaves no ambiguity about which file and function Plan 15-03 must modify.

---
*Phase: 15-live-commodity-matching-engineering-fixes*
*Completed: 2026-04-23*

## Self-Check: PASSED

- `.planning/phases/15-live-commodity-matching-engineering-fixes/15-01-DIAGNOSTIC.md` → FOUND on disk
- Commit `fc2fcbd` (Task 1) → FOUND in git log
- Commit `e01cea4` (Task 2) → FOUND in git log
- Both required sections (`## Evidence from Kalshi /series`, `## Root-Cause Hypotheses`) present in diagnostic
- At least one VERDICT CONFIRMED (H1)
- Fix Recommendations section names concrete file paths for Plan 15-03 handoff
