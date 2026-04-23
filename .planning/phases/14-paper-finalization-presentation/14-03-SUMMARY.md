---
phase: 14-paper-finalization-presentation
plan: 03
subsystem: paper
tags: [readme, marp, slides, reproduction, ieee, submission]

# Dependency graph
requires:
  - phase: 14-paper-finalization-presentation
    provides: IEEE-styled figures, trimmed abstract, Table 9/10 renumber, check_paper.sh validator, §6.4 expansion
provides:
  - README.md at repo root with reproduction commands for every paper table and figure
  - slides.md (Marp source) + slides.pdf for 4-minute lightning talk
  - PAPER_DRAFT.md Appendix A converted to README pointer
  - §5.12 lookback sweep and §5.13 threshold heatmap body sections (proposal-experiment writeups)
  - Appendix B placeholders repaired (Fig 10, Fig 11)
  - Abstract oil-category qualifier added ("where matched cohort is representable")
  - §6.4 item 9 commodity-matching limitation documented
  - AI-attribution headers on 56 src/**/*.py files (POL-09 code-level coverage)
affects:
  - Post-submission: future reproduction attempts will enter through README.md
  - Future work: Phase 15+ can key off README table for automated reproduction CI

# Tech tracking
tech-stack:
  added:
    - "@marp-team/marp-cli (slide rendering to PDF)"
  patterns:
    - "Repo-root README with paper-object reproduction table (one row per table/figure)"
    - "Marp markdown → PDF pipeline for academic lightning talks"
    - "AI-attribution header block as docstring on every src/ module"

key-files:
  created:
    - README.md
    - slides.md
    - slides.pdf
    - .planning/phases/14-paper-finalization-presentation/14-03-SUMMARY.md
  modified:
    - PAPER_DRAFT.md (Appendix A pointer, §5.12, §5.13, Appendix B fixes, abstract qualifier, §6.4 item 9)
    - src/**/*.py (56 files — AI attribution headers)

key-decisions:
  - "Appendix A deprecated to a single README pointer — single source of truth for reproduction"
  - "Slides rendered via Marp (Node toolchain already available) rather than LaTeX Beamer"
  - "User feedback during checkpoint triggered two fix cycles — accepted as Rule 1/2 deviations; §5.12 / §5.13 added as post-hoc body sections for proposal experiments that were otherwise orphan tables"
  - "AI attribution applied at code-module granularity to satisfy POL-09 substantively (not just acknowledgments paragraph)"
  - "Abstract oil claim narrowed to 'where the matched cohort is representable' after KG feedback — avoids overstating universal oil edge"

patterns-established:
  - "Repo-root README.md as canonical reproduction map"
  - "slides.md + slides.pdf committed together (source + artifact)"
  - "Post-hoc body writeups for proposal-experiment tables to avoid orphan references"

requirements-completed:
  - POL-11
  - POL-12

# Metrics
duration: 90min
completed: 2026-04-23
---

# Phase 14 Plan 03: README + Slides + Final Polish Summary

**Submission-ready repository: README reproduction table, Marp lightning-talk slides, and two rounds of checkpoint fixes closing residual paper issues.**

## Performance

- **Duration:** ~90 min (including two human-verify fix cycles)
- **Started:** 2026-04-23T20:00:00Z (approx)
- **Completed:** 2026-04-23T21:51:20Z
- **Tasks:** 3 (2 auto + 1 human-verify checkpoint with 2 fix iterations)
- **Files modified:** 60+ (README, slides.md, slides.pdf, PAPER_DRAFT.md, 56 src/*.py)

## Accomplishments

- README.md at repo root with a 16-row reproduction table covering every paper Table (2–10) and Figure (1–11), every script path verified to exist on disk
- PAPER_DRAFT.md Appendix A shortened to a canonical pointer at the README (removed 18-line command list that would have drifted from code)
- slides.md (Marp source) + slides.pdf (rendered 7 pages, 4-minute timed) committed together
- §5.12 (lookback sweep) and §5.13 (threshold heatmap) body sections added — previously these were orphan tables referenced only in proposal experiments
- Appendix B placeholders for Fig 10 and Fig 11 repaired to match final numbering
- Abstract oil claim qualified: "where the matched cohort is representable" — addresses KG concern that the commodity win-rate framing overstates generality
- §6.4 Limitations item 9 added: commodity-matching cohort limitation (only ~30% of contracts match cross-platform; oil edge conditional on that cohort)
- AI-attribution headers added to 56 `src/**/*.py` files satisfying POL-09 at code-module level (not just in Acknowledgments)
- User approved at final cover-to-cover human-verify checkpoint

## Task Commits

Each task was committed atomically:

1. **Task 1 — README + Appendix A pointer (POL-11):** `2a5ab9f` (docs)
2. **Task 2 — Marp slides.md + slides.pdf (POL-12):** `1e4a3e5` (feat)
3. **Task 3 fix cycle 1 — §5.12 + §5.13 + Appendix B placeholders:** `c5e5142` (docs)
4. **Task 3 fix cycle 2 — abstract oil qualifier + §6.4 item 9 + 56-file AI attribution:** `1b06036` (docs)

**Plan metadata commit:** (this commit) `docs(14-03): complete README + slides + final polish plan`

_Note: Task 3 is a human-verify checkpoint; user's "approve" response triggered the two fix cycles before final sign-off._

## Files Created/Modified

- `README.md` — Repo-root reproduction map (created)
- `slides.md` — Marp source, 7 sections (created)
- `slides.pdf` — Rendered 7-page PDF, ≤4:00 read-through (created)
- `PAPER_DRAFT.md` — Appendix A pointer, §5.12 lookback, §5.13 threshold, Appendix B repair, abstract qualifier, §6.4 item 9 (modified)
- `src/**/*.py` — 56 files received AI-attribution docstring headers (modified)

## Decisions Made

- **Appendix A → pointer:** Keeping command lists in the paper would drift from code; README is canonical, Appendix A now references it.
- **Marp over Beamer:** Node/marp toolchain already present; faster iteration than LaTeX for a 4-minute talk.
- **Post-hoc §5.12/§5.13 writeups:** Proposal experiments had tables without body sections; rather than drop the tables, added compact writeups to close the cross-reference loop.
- **Code-level AI attribution:** POL-09 acknowledgments paragraph was present but weak evidence of AI involvement; per-file headers make disclosure unambiguous and survivorship-proof.
- **Oil claim narrowing:** "Where matched cohort is representable" is the honest scope statement — the 76.5% oil win-rate holds on the ~30% of contracts that match cross-platform, not on the full oil market.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 — Bug] §5.12 / §5.13 orphan references**
- **Found during:** Task 3 (human-verify checkpoint, fix cycle 1)
- **Issue:** Tables from proposal experiments (lookback sweep, threshold heatmap) were referenced in cross-ref prose but had no body section; reader hits dead-end.
- **Fix:** Added compact §5.12 and §5.13 body sections summarizing each experiment's result, keyed to the existing tables/figures.
- **Files modified:** `PAPER_DRAFT.md`
- **Verification:** `check_paper.sh` still green (16/16); cross-refs now resolve.
- **Committed in:** `c5e5142`

**2. [Rule 1 — Bug] Appendix B Fig 10 / Fig 11 placeholders**
- **Found during:** Task 3 (fix cycle 1, same commit)
- **Issue:** Appendix B had residual placeholder captions for Fig 10 (TFT VSN) and Fig 11 (ensemble sweep) not updated during 14-02 renumber.
- **Fix:** Replaced placeholders with final captions matching in-text numbering.
- **Files modified:** `PAPER_DRAFT.md` Appendix B
- **Verification:** Appendix B now 11/11 in-text order; grep for "[Insert Figure]" returns 0.
- **Committed in:** `c5e5142`

**3. [Rule 2 — Missing Critical] Abstract oil claim overstatement**
- **Found during:** Task 3 (fix cycle 2)
- **Issue:** Abstract implied oil category edge is universal; KG-style critique would catch this — the edge holds only on the matched cross-platform cohort (~30% of contracts).
- **Fix:** Added qualifier "where the matched cohort is representable" to the headline oil sentence.
- **Files modified:** `PAPER_DRAFT.md` abstract
- **Verification:** Abstract still ≤ 250 words; claim now matches §5.3 and §6.4 scope.
- **Committed in:** `1b06036`

**4. [Rule 2 — Missing Critical] §6.4 item 9 commodity-matching limitation**
- **Found during:** Task 3 (fix cycle 2)
- **Issue:** Limitations section did not name the commodity-matching cohort as a generalization limit, even though it's the biggest caveat on the oil result.
- **Fix:** Added §6.4 item 9 documenting that cross-platform matching covers only a fraction of the commodity universe; oil edge is conditional on that cohort.
- **Files modified:** `PAPER_DRAFT.md` §6.4
- **Verification:** §6.4 now has 9 items; `check_paper.sh` green.
- **Committed in:** `1b06036`

**5. [Rule 2 — Missing Critical] POL-09 AI attribution at code level**
- **Found during:** Task 3 (fix cycle 2)
- **Issue:** Acknowledgments mentioned AI pair-programming, but no per-file evidence — could be read as boilerplate disclosure.
- **Fix:** Added AI-attribution docstring header block to 56 `src/**/*.py` files.
- **Files modified:** 56 files under `src/`
- **Verification:** `grep -l "AI-assistant" src/**/*.py | wc -l` → 56.
- **Committed in:** `1b06036`

---

**Total deviations:** 5 auto-fixed (2 bug, 3 missing critical) — all triggered by user feedback at the human-verify checkpoint. No scope creep; all fixes closed residual paper-integrity gaps that the original Task 1/2 scope did not anticipate.

**Impact on plan:** Two fix cycles extended Task 3 duration ~60 min beyond the original ~30 min checkpoint budget. Final paper is materially stronger (narrower claims, complete cross-refs, substantive AI disclosure).

## Issues Encountered

- **Marp rendering:** required `--allow-local-files` flag for walk-forward PNG embed on Results slide — documented in commit message for future renders.
- **Slides timing:** dry-run measured ~3:45 against a 4:00 budget; no cuts required.
- **User "approve" sequence:** after first fix cycle user flagged additional issues; pattern suggests future checkpoints should pre-ask "any remaining concerns?" before accepting approval.

## User Setup Required

None — no external service configuration.

## Next Phase Readiness

- **Phase 14 complete** after this plan — all three phase-14 plans shipped.
- **Submission-ready:** paper + README + slides + check_paper.sh green. Ready for April 27 submission.
- **Remaining before submission:** Alvin's final readability pass on §1 + §8 (scheduled post-plan); render final PDF via pandoc.
- **Post-submission:** consider Phase 15 for CI-enforced reproduction (automate README table against actual script runs).

## Self-Check: PASSED

Verified:
- `README.md` exists at repo root ✓
- `slides.md` + `slides.pdf` exist ✓
- Commit `2a5ab9f` in git log ✓
- Commit `1e4a3e5` in git log ✓
- Commit `c5e5142` in git log ✓
- Commit `1b06036` in git log ✓
- PAPER_DRAFT.md Appendix A is a short pointer ✓
- `check_paper.sh` green at time of last fix cycle ✓

---
*Phase: 14-paper-finalization-presentation*
*Completed: 2026-04-23*
