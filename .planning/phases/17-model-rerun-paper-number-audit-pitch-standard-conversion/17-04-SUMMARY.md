---
phase: 17-model-rerun-paper-number-audit-pitch-standard-conversion
plan: 04
subsystem: project-management
tags: [phase-closure, state-md, roadmap, requirements-traceability, canonical-results-convention, pitch-standard]

# Dependency graph
requires:
  - phase: 17-model-rerun-paper-number-audit-pitch-standard-conversion
    plan: 01
    provides: experiments/results/canonical/headline.json (single source of truth) + 17-02-PPO-DIAGNOSTIC.md (units-mismatch root cause)
  - phase: 17-model-rerun-paper-number-audit-pitch-standard-conversion
    plan: 02
    provides: scripts/audit_paper_numbers.py + PAPER_DRAFT.md pitch-standard conversion
  - phase: 17-model-rerun-paper-number-audit-pitch-standard-conversion
    plan: 03
    provides: slides_deck.html pitch-standard rebuild + scripts/check_paper.sh REPL-06 regression checks (19/19 OK)
provides:
  - STATE.md decision-log entries documenting PPO root cause (verbatim from 17-02-PPO-DIAGNOSTIC.md), canonical-results JSON convention, pitch-standard adoption as house style
  - ROADMAP.md Phase 17 entry marked 4/4 Complete with Resolution summary; v1.1 milestone checklist Phases 15/16/17 added; phase-tracking table updated through Phase 17
  - REQUIREMENTS.md traceability: REPL-01 through REPL-07 all Complete (7/7); v1.1 milestone fully shipped 100%
affects: [post-Phase-17 maintenance, future-phase context-loading, paper submission April 27]

# Tech tracking
tech-stack:
  added: []  # No new code
  patterns:
    - "Phase-closure workflow: STATE.md frontmatter (status complete + percent 100) + Current Position update + Decisions append → ROADMAP.md milestone checklist + tracking-table row + Plans 4/4 with commit hashes + Resolution summary block → REQUIREMENTS.md checkbox flip + traceability table flip → atomic commits"
    - "Canonical-results JSON convention codified in STATE.md as a project-wide rule: every paper / slide numeric claim derives from experiments/results/canonical/headline.json; no metric may cite a JSON file outside that directory"
    - "Pitch-standard format codified as house style for all future result presentations: per-trade Sharpe + per-trade alpha (bps) lead; cumulative dollars in tables only"

key-files:
  created: []
  modified:
    - .planning/STATE.md (frontmatter to 100%/complete + Current Position + 6 new Phase 17 closure decisions + Session Continuity)
    - .planning/ROADMAP.md (v1.1 milestone checklist Phases 15/16/17 + phase-tracking table Phases 16/17 rows + Phase 17 Plans block 4/4 with hashes + Resolution summary + footer)
    - .planning/REQUIREMENTS.md (REPL-07 checkbox + traceability row + footer)

key-decisions:
  - "STATE.md decisions section uses an APPEND pattern (preserves all prior entries verbatim) rather than rewrite — keeps the chronological project history intact for future-phase context loading"
  - "PPO root-cause text in STATE.md is the verbatim diagnostic paragraph from 17-02-PPO-DIAGNOSTIC.md Root Cause section, not a paraphrase — ensures STATE.md and the diagnostic file cannot drift"
  - "ROADMAP.md v1.1 milestone checklist gained Phases 15, 16, 17 simultaneously because none had been added (the milestone block was last touched at Phase 14 completion). Phase 17-04 closes this gap as part of project-state hygiene rather than leaving it for a future hot-fix"
  - "Phase 17 Plans block in ROADMAP.md uses two-line format per plan: '[x] {plan-name} ({REPL ids}) — {date}, commits {hash list}'. Hashes drawn from git log: db6fb4a/5212e14/9fa60a0 (17-01), 232b47b/62c3368/8c34f6d (17-02), af04c3c/18e79ba/c9567fe (17-03), and 17-04's own commits (a20b40b for STATE.md, d450ff0 for ROADMAP+REQUIREMENTS)"
  - "Resolution summary in ROADMAP.md leads with the PPO 600× root cause (units mismatch + mid_price-driven contract-count inflation) rather than the canonical-results convention or pitch-standard adoption — root cause was the original triggering bug, the conventions are downstream remediation"

patterns-established:
  - "Phase-closure plan = STATE.md + ROADMAP.md + REQUIREMENTS.md atomic edits in two commits (one per file group); per-plan-of-prior-plans commit hashes embedded in ROADMAP.md so future maintenance can locate any prior plan's work in one grep"
  - "Resolution summary block on ROADMAP.md phase entries: 1-paragraph summary citing root cause + conventions established. Differentiates 'phase complete' (checkbox state) from 'phase resolved' (substantive narrative for future reviewers)"

requirements-completed: [REPL-07]

# Metrics
duration: 3min
completed: 2026-04-25
---

# Phase 17 Plan 04: Phase Closure Summary

**STATE.md frontmatter advanced to 100% complete; ROADMAP.md Phase 17 entry marked 4/4 Complete with Resolution summary citing the PPO units-mismatch root cause and the canonical-results JSON convention; REQUIREMENTS.md REPL-01 through REPL-07 all flipped to Complete (7/7). v1.1 Extended Evidence & Submission milestone fully shipped — April 27 paper submission is locked.**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-04-25T18:12:08Z
- **Completed:** 2026-04-25T18:15:18Z
- **Tasks:** 2 / 2
- **Files modified:** 3 (STATE.md, ROADMAP.md, REQUIREMENTS.md)

## Final STATE.md Frontmatter Values

| Field | Pre-17-04 | Post-17-04 |
|---|---|---|
| `total_phases` | 10 | **10** (unchanged — Phase 17 already counted) |
| `completed_phases` | 8 | **10** |
| `total_plans` | 25 | **25** (unchanged) |
| `completed_plans` | 23 | **25** |
| `percent` | 84 | **100** |
| `status` | completed | **complete** |
| `last_updated` | 2026-04-25T18:10:55.638Z | **2026-04-25T18:12:08Z** |
| `last_activity` | 2026-04-25 | **"2026-04-25 -- Plan 17-04 executed (Phase 17 closure: canonical results convention adopted, pitch-standard headlines locked)"** |
| `stopped_at` | "Completed 17-03-PLAN.md..." | **"Phase 17 complete — paper has canonical numerics in pitch-standard format; April 27 submission ready"** |

## STATE.md Decisions Section

**6 new bullets appended** to `## Accumulated Context > ### Decisions` (preserving all prior entries verbatim):

1. `[Phase 17-01-canonical-rerun]` — names experiments/results/canonical/headline.json as the single source of truth; documents protocol (seed=42, position=$100, threshold=0.02, train=6802/test=1673); lists the 13 metric fields per model; codifies "no paper or slide may cite a JSON file outside experiments/results/canonical/"
2. `[Phase 17-02-PPO-diagnostic]` — full verbatim PPO root cause from 17-02-PPO-DIAGNOSTIC.md (units mismatch between profit_sim and WalkForwardBacktester, ~200× position scaling × ~3× mid_price-driven contract-count inflation = 609× ratio); `−$7,724` paper claim diagnosed as transcription typo of `−$87,724`; archive paths named
3. `[Phase 17-02-paper-audit]` — names scripts/audit_paper_numbers.py as the automated cross-reference tool with section-aware filtering, per-number proximity model resolution, tolerance-based comparison
4. `[Phase 17-02-pitch-standard]` — pitch-standard headline format codified: per-trade Sharpe + per-trade alpha (bps) lead; per-trade alpha formula = (total_pnl / num_trades / position_size) × 10,000; worked examples for LR (15.0 bps) and PPO+AE (0.5 bps); abstract / Tables 2 + 8 / §5.1 / §5.8 / §6.3 / §8 all updated
5. `[Phase 17-03-slide-validator]` — slides_deck.html Results slide leads with bps + Sharpe; PPO+AE bar added to main chart (visually negligible — tier-3 collapse self-evident); scripts/check_paper.sh extended with 3 REPL-06 regression checks; total 19/19 OK
6. `[Phase 17-04-closure]` — pitch-standard adopted as project's house style; canonical results JSON pattern codified as the convention going forward; v1.1 milestone fully shipped 100%

## Verbatim PPO Root Cause (as written in STATE.md)

> PPO 600× magnitude divergence root cause: units mismatch between two valid simulators with different units, compounded by mid_price-driven contract-count inflation when position_size is held fixed at $100. profit_sim.simulate_profit (canonical) returns raw probability-point spread P&L with no fees and no position sizing; WalkForwardBacktester.run (legacy) returns dollar P&L for a $100-notional fixed-size strategy that computes num_contracts = $100 / mid_price ≈ 200 per trade (because contracts trade at ~$0.50 mid), then gross_pnl = num_contracts × actual_change × direction minus 5pp round-trip fees. The 609× legacy/canonical ratio decomposes as ~200× position scaling × ~3× mid_price-driven contract-count inflation on low-priced commodity contracts that trade at $0.10–$0.30. Both simulators are correct for the question they ask; the paper accidentally cited a legacy figure (in dollars) when its surrounding text used canonical figures (in spread units). It is not a code bug, a training mismatch, or an episode-horizon issue. See .planning/phases/17-model-rerun-paper-number-audit-pitch-standard-conversion/17-02-PPO-DIAGNOSTIC.md for the full diagnostic.

## ROADMAP.md Changes

**v1.1 milestone checklist** — added 3 entries (Phases 15, 16, 17) that had been missing:

```markdown
- [x] **Phase 15: Live Commodity-Matching Engineering Fixes** ... (completed 2026-04-24)
- [x] **Phase 16: Paper Revision — Phase 15 Integration + Final Review** ... (completed 2026-04-24)
- [x] **Phase 17: Model Rerun + Paper Number Audit + Pitch-Standard Conversion** ... (completed 2026-04-25)
```

**Phase tracking table** — added 2 rows (Phases 16, 17 had been missing — table previously stopped at Phase 15):

```
| 16. Paper Revision — Phase 15 Integration + Final Review | 3/3 | Complete | 2026-04-24 | - |
| 17. Model Rerun + Paper Number Audit + Pitch-Standard | 4/4 | Complete | 2026-04-25 | - |
```

**Phase 17 Plans block** — converted from `Plans: TBD` placeholder to 4/4 complete with commit hashes, plus a Resolution summary block:

- Plans 17-01 through 17-04 each marked `[x]` with date and commit-hash list
- Resolution summary cites: PPO units-mismatch root cause; canonical results JSON convention at `experiments/results/canonical/headline.json`; pitch-standard format adopted as house style; paper / slide / validator (`scripts/check_paper.sh` 19/19 OK) all reconcile

## REQUIREMENTS.md Changes

- REPL-07 checkbox flipped from `[ ]` to `[x]` in the v1.1 Phase 17 requirements list
- Traceability table row `| REPL-07 | Phase 17 | Pending |` flipped to `| REPL-07 | Phase 17 | Complete |`
- Footer updated: `*Last updated: 2026-04-25 (Phase 17 closed: REPL-01 through REPL-07 all Complete; v1.1 milestone 100%)*`

**Final REPL traceability:** 7/7 Complete (REPL-01 ✓, REPL-02 ✓, REPL-03 ✓, REPL-04 ✓, REPL-05 ✓, REPL-06 ✓, REPL-07 ✓). Zero Pending REPL entries remain.

## Task Commits

1. **Task 1: Append Phase 17 closure decisions to STATE.md + advance frontmatter to 100%** — `a20b40b` (docs)
2. **Task 2: Mark Phase 17 Complete in ROADMAP.md + flip REPL-07 to Complete in REQUIREMENTS.md** — `d450ff0` (docs)

## Files Created/Modified

- `.planning/STATE.md` (modified) — Frontmatter advanced to 100%/complete; Current Position rewritten for Phase 17 closure; 6 new Phase 17 decision entries appended (preserving all prior entries verbatim); Session Continuity stopped_at updated
- `.planning/ROADMAP.md` (modified) — v1.1 milestone checklist Phases 15/16/17 entries added; phase-tracking table Phases 16/17 rows added; Phase 17 entry: Plans block converted from TBD to 4/4 Complete with commit hashes per plan; Resolution summary block appended; footer updated
- `.planning/REQUIREMENTS.md` (modified) — REPL-07 checkbox `[ ]` → `[x]`; traceability row Pending → Complete; footer updated

## Decisions Made

See key-decisions in frontmatter. Highlights:

- **APPEND pattern for STATE.md decisions section** — preserves chronological project history; never rewrite or delete prior entries
- **Verbatim PPO root cause text** — copied straight from 17-02-PPO-DIAGNOSTIC.md, not paraphrased; ensures STATE.md and the diagnostic file cannot drift
- **v1.1 milestone checklist gap** (Phases 15/16/17 had not been added since Phase 14 completion) — closed as part of Phase 17-04 hygiene rather than deferred
- **Resolution summary leads with root cause** — the PPO units-mismatch is the original bug; canonical-results convention and pitch-standard adoption are downstream remediation. ROADMAP.md should reflect this causal ordering.

## Deviations from Plan

**1. [Rule 1 - Bug] STATE.md frontmatter `status` was `completed` (non-canonical) — flipped to `complete` per plan spec.**
- **Found during:** Task 1 frontmatter edit
- **Issue:** Pre-Phase-17-04 STATE.md had `status: completed` (past-tense participle, non-canonical for the gsd-state-version 1.0 schema). Plan spec calls for `status: complete`. Likely a stale value from a prior phase closure.
- **Fix:** Set `status: complete` (matches plan spec)
- **Files modified:** .planning/STATE.md
- **Commit:** a20b40b

**2. [Rule 2 - Missing critical] ROADMAP.md v1.1 milestone checklist had been missing Phases 15, 16, 17 entries since Phase 14 closure.**
- **Found during:** Task 2 v1.1 milestone block edit
- **Issue:** The plan only required adding Phase 17 to the v1.1 checklist, but Phase 15 and 16 [x] entries were also absent (the checklist stopped at Phase 14). Leaving 15 and 16 missing while adding 17 would have been inconsistent.
- **Fix:** Added Phase 15, 16, 17 entries together as a 3-line block. Each cites completion date and a one-line accomplishment summary.
- **Files modified:** .planning/ROADMAP.md
- **Commit:** d450ff0

**3. [Rule 2 - Missing critical] ROADMAP.md phase-tracking table had been missing Phase 16 and 17 rows.**
- **Found during:** Task 2 phase-tracking-table edit
- **Issue:** Same root cause as deviation #2 — table stopped at Phase 15. Plan spec only required adding the Phase 17 row, but adding it without Phase 16 would have left a hole.
- **Fix:** Added both Phase 16 (3/3 Complete 2026-04-24) and Phase 17 (4/4 Complete 2026-04-25) rows in correct table position
- **Files modified:** .planning/ROADMAP.md
- **Commit:** d450ff0

---

**Total deviations:** 3 auto-fixed (1 Rule 1 bug, 2 Rule 2 missing critical project-state hygiene)
**Impact on plan:** All deviations strengthen the plan. (1) brings STATE.md to canonical schema; (2) and (3) close pre-existing v1.1 milestone documentation gaps that would have been re-discovered in any future phase. No scope creep — these are hygiene fixes within the closure mandate.

## Issues Encountered

None blocking. Pre-existing `placeholder` / `{Model}` / `XXX` mentions in STATE.md decision history (e.g., line 127 references "Fig 11 resolves '[Insert Figure]' placeholder" from Phase 14-02; line 149 contains the literal string `{Model}` inside a quoted pitch-standard format template) match a naive `grep -cE '\{|placeholder'` regex but are NOT unfilled placeholders. Verified manually that no Task 1 / Task 2 edits introduced any unfilled placeholder text.

## Next Phase Readiness

**v1.1 Extended Evidence & Submission milestone is fully shipped (100%).** April 27, 2026 paper submission is locked.

Optional future work (not blocking submission):
- Paper §6.4 revision citing the 1,224-position live commodity cohort (Phase 15) as empirical follow-up to the acknowledged limitation
- Co-author readability review on §1 Introduction and §8 Conclusions (REV-06, currently Pending)
- Cover-to-cover final read by Ian (REV-07, currently Pending)

Note: REV-06 and REV-07 remain Pending in REQUIREMENTS.md but are NOT blocking — the paper has been substantively complete since Phase 16-03 and all Phase 17 numeric reconciliation is done. They are quality-gate review items, not blockers for the submission deadline.

## Self-Check: PASSED

All claimed files exist on disk:
- `.planning/STATE.md` ✓ (modified)
- `.planning/ROADMAP.md` ✓ (modified)
- `.planning/REQUIREMENTS.md` ✓ (modified)
- `.planning/phases/17-model-rerun-paper-number-audit-pitch-standard-conversion/17-04-SUMMARY.md` ✓ (this file)

All claimed commits exist in git history:
- `a20b40b` (Task 1: docs(17-04) close Phase 17 in STATE.md) ✓
- `d450ff0` (Task 2: docs(17-04) close Phase 17 in ROADMAP + REQUIREMENTS) ✓

All plan-level acceptance criteria validated:
- STATE.md percent: 100 ✓
- STATE.md status: complete ✓
- STATE.md total_phases: 10 ✓
- STATE.md completed_phases: 10 ✓
- STATE.md total_plans / completed_plans: 25 ✓
- STATE.md 6 new Phase 17 decision entries (5 required) ✓
- STATE.md canonical/headline.json mentioned 3× ✓
- ROADMAP.md TBD lines: 0 ✓
- ROADMAP.md Phase 17 row in tracking table ✓
- ROADMAP.md Plans 4/4 complete ✓
- ROADMAP.md Resolution summary block present ✓
- ROADMAP.md v1.1 Phase 17 [x] checkbox ✓
- REQUIREMENTS.md REPL-01..07 all Complete: 7/7 ✓
- REQUIREMENTS.md REPL Pending: 0 ✓
- REQUIREMENTS.md REPL [x] checkboxes: 7 ✓
- Cross-file unfilled placeholders: 0 (the 1 STATE.md match is the literal string `{Model}` inside a quoted pitch-standard template — not an unfilled placeholder) ✓

---

*Phase: 17-model-rerun-paper-number-audit-pitch-standard-conversion*
*Completed: 2026-04-25*
