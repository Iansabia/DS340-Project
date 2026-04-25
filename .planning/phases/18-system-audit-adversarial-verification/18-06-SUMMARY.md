---
phase: 18-system-audit-adversarial-verification
plan: 06
subsystem: testing
tags: [audit, paper-trace, regex-extraction, csv, regression-checks, bash, python]

# Dependency graph
requires:
  - phase: 18-01
    provides: experiments/audit/ scaffolding + tests/audit/ infrastructure
  - phase: 18-02
    provides: experiments/results/audit/sharpe_audit.json (Wave 1 Tier 1 verdict for the ≈3.2 abstract claim)
  - phase: 17-02
    provides: scripts/audit_paper_numbers.py regex bank (DOLLAR_RE / SHARPE_RE / BPS_RE / PCT_RE) reused verbatim
provides:
  - experiments/audit/build_paper_numbers_csv.py (Tier 5 generator, 192 lines)
  - experiments/results/audit/paper_numbers.csv (86 traceable numeric claims, 8-col schema)
  - 7 new audit_-prefixed regression checks in scripts/check_paper.sh
  - First MISMATCH flag: per-pair annualized Sharpe ≈ 3.2 (paper) vs 18.60/7.04 (Plan 18-02 reproduction)
affects: [Plan 18-07 AUDIT_REPORT.md, PAPER_DRAFT.md headline updates if MISMATCH stays]

# Tech tracking
tech-stack:
  added: []  # zero new deps; uses stdlib csv/json/re only
  patterns:
    - "Headline-section restriction (Phase 17 lesson): only audit Abstract / §5.1 / §5.8 / §6.3 / §8 Conclusions to keep false-positive rate at zero"
    - "Two-pass dedup-by-end-position when stacking multiple regexes (SHARPE_RE + SHARPE_LONG_RE) on the same line"
    - "MISMATCH-as-finding: if a paper number fails to trace, the trace itself becomes the audit finding — do not 'fix' the paper in the trace plan"

key-files:
  created:
    - experiments/audit/build_paper_numbers_csv.py
    - experiments/results/audit/paper_numbers.csv
    - .planning/phases/18-system-audit-adversarial-verification/18-06-SUMMARY.md
  modified:
    - scripts/check_paper.sh

key-decisions:
  - "MISMATCH for per-pair Sharpe ≈ 3.2 surfaced inline in paper_numbers.csv with pre-filled note pointing to sharpe_audit.json — Plan 18-07 inherits a head start instead of re-discovering the discrepancy"
  - "SHARPE_LONG_RE added (200-char window) to capture long-form 'Sharpe...is ≈ 3.2' claims that SHARPE_RE's 30-char window misses; deduplicated with end-position keys to avoid double-counting the short-form Sharpe claims"
  - "audit_per_pair_sharpe_3_2_in_abstract enforces consistency of whatever value the paper claims; it does NOT validate 3.2 against canonical (that's a paper number, not a JSON number) — Plan 18-07 will resolve the value question in PAPER_DRAFT.md"
  - "Source-file column defaults to canonical/headline.json + run_canonical.py; only the special-case 3.2 sharpe value points to sharpe_audit.json + audit_sharpe.py because that is the only headline number whose authoritative source is an audit script, not the canonical run"

patterns-established:
  - "Tier 5 traceability map (paper_numbers.csv) is the input contract for AUDIT_REPORT.md (Plan 18-07)"
  - "scripts/check_paper.sh is now the runtime guardrail for paper-vs-canonical drift; any future canonical retrain must keep these 7 audit_ checks green"

requirements-completed:
  - AUDIT-05

# Metrics
duration: 4min
completed: 2026-04-25
---

# Phase 18 Plan 06: Tier 5 Paper-Number Trace Summary

**86 traceable numeric claims emitted to paper_numbers.csv across 5 headline sections; 7 audit_ regression checks added to check_paper.sh; first MISMATCH (per-pair Sharpe ≈ 3.2 vs Plan 18-02 reproduction 18.60/7.04) flagged for Plan 18-07.**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-04-25T19:54:10Z
- **Completed:** 2026-04-25T19:58:00Z
- **Tasks:** 2
- **Files modified:** 3 (2 created, 1 modified)

## Accomplishments

- `experiments/audit/build_paper_numbers_csv.py` (192 lines) walks PAPER_DRAFT.md headline sections and emits CSV rows.
- `experiments/results/audit/paper_numbers.csv` ships **86 claims** in a stable 8-column schema (`claim_text, claim_value, kind, paper_section, line_number, source_file, source_command, match_status`).
- 7 new `audit_`-prefixed regression checks in `scripts/check_paper.sh` (≥5 required) — all passing; total checks 19 → 26.
- Two MISMATCH rows pre-flagged for Plan 18-07 (the abstract + §8 Conclusions instances of the per-pair annualized Sharpe ≈ 3.2 claim, both pointing to `sharpe_audit.json` since Plan 18-02 reproduced 18.60 naive / 7.04 BLdP-corrected from the canonical trade ledger).

### CSV breakdown by kind (86 total)

| kind   | count |
| ------ | ----- |
| bps    | 28    |
| dollar | 25    |
| pct    | 22    |
| sharpe | 11    |

### CSV breakdown by paper section (86 total)

| paper_section                          | count |
| -------------------------------------- | ----- |
| ### 5.1 Headline Model Comparison      | 42    |
| ## 8. Conclusions                      | 21    |
| ### 5.8 Honest Sharpe-Ratio Accounting | 10    |
| ## Abstract                            | 8     |
| ### 6.3 The Negative Result on PPO     | 5     |

### New audit_ checks added (7 total)

| name                                       | metric checked                              | grep result |
| ------------------------------------------ | ------------------------------------------- | ----------- |
| audit_lr_per_trade_sharpe_in_paper         | LR Sharpe 0.501                             | 5 hits      |
| audit_lr_alpha_bps_in_paper                | LR 15.0 bps                                 | 5 hits      |
| audit_xgb_per_trade_sharpe_in_paper        | XGB Sharpe 0.499                            | 7 hits      |
| audit_ppo_filtered_alpha_bps_in_paper      | PPO+autoencoder 0.5 bps                     | 5 hits      |
| audit_per_pair_sharpe_3_2_in_abstract      | abstract cites ≈3.2 (Plan 18-07 will resolve) | 1 hit       |
| audit_walk_forward_11_windows_in_paper     | 11-window walk-forward                      | 6 hits      |
| audit_test_rows_1673_in_paper              | 1,673 test rows                             | 10 hits     |

### check_paper.sh check count

**Before:** 19 / **After:** 26 / **Delta:** +7 audit_-prefixed checks. All 26 OK.

## Task Commits

1. **Task 1: build_paper_numbers_csv.py + paper_numbers.csv** - `cddfe76` (feat)
2. **Task 2: extend check_paper.sh with 7 audit_ checks** - `20dbee4` (feat)

## Files Created/Modified

- `experiments/audit/build_paper_numbers_csv.py` (created) - Tier 5 CSV generator; walks PAPER_DRAFT.md headline sections only, emits one CSV row per numeric claim, deduplicates SHARPE_RE+SHARPE_LONG_RE collisions, pre-flags the per-pair Sharpe 3.2 MISMATCH inline.
- `experiments/results/audit/paper_numbers.csv` (created) - 86 claims with 8-column schema; consumed by Plan 18-07 AUDIT_REPORT.md.
- `scripts/check_paper.sh` (modified) - +51 lines under new "AUDIT-05: Phase 18 number-by-number regression checks" block, before final summary so checks are counted.

## Decisions Made

- **MISMATCH-as-finding pattern:** Plan 18-06's job is to TRACE numbers; if a number doesn't trace, the trace itself becomes the audit finding. We did NOT fix PAPER_DRAFT.md — that is Plan 18-07's job. We only enumerated and flagged.
- **SHARPE_LONG_RE addition:** The verbatim RESEARCH.md regex bank uses `[Ss]harpe[^0-9]{1,30}([0-9]\.[0-9]+)` which misses the abstract's "Sharpe...is ≈ 3.2" because the intervening text is 71 chars > 30-char window. Added a complementary `[Ss]harpe[^.\n]{0,200}?(?:is|≈|approximately|of)\s*≈?\s*([0-9]\.[0-9]+)` regex with end-position dedup so we capture the long form without double-counting the short form. Justified under deviation Rule 2 (missing critical functionality — the headline ≈ 3.2 claim is exactly the number Phase 18 was created to audit; missing it would defeat Tier 5).
- **Special-case source attribution:** Default source_file = canonical/headline.json. Special-case = 3.2 sharpe → sharpe_audit.json (because Plan 18-02 is the authoritative source for that derived metric, not the per-trade-canonical headline).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added SHARPE_LONG_RE to capture abstract's ≈ 3.2 claim**
- **Found during:** Task 1 (initial run after copying RESEARCH.md skeleton verbatim)
- **Issue:** SHARPE_RE's 30-char window between "Sharpe" and the numeric group missed the abstract's headline "per-pair annualized Sharpe, treating each of 144 pairs as one independent bet, is ≈ 3.2" claim — the intervening text is 71 chars. Without this fix the abstract's ≈ 3.2 claim was invisible in paper_numbers.csv, and the whole Phase 18 audit was created precisely to interrogate this number.
- **Fix:** Added `SHARPE_LONG_RE = re.compile(r"[Ss]harpe[^.\n]{0,200}?(?:is|≈|approximately|of)\s*≈?\s*([0-9]\.[0-9]+)")` with a `seen_on_line` dedup set keyed on `(kind, value, end-position-of-numeric-group)` to prevent double-counting the short-form `Sharpe of 0.501` claims.
- **Files modified:** experiments/audit/build_paper_numbers_csv.py
- **Verification:** Re-ran builder; abstract line 12 now produces a row `"Sharpe, treating each of 144 pairs as one independent bet, is ≈ 3.2",3.2,sharpe,## Abstract,12,...,MISMATCH:...`. Total CSV rows went 85 → 86 (one new row, no duplicates).
- **Committed in:** cddfe76 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 missing critical)
**Impact on plan:** Necessary for Tier 5 to interrogate its load-bearing claim. No scope creep.

## Issues Encountered

None — both tasks completed cleanly. Pre-existing baseline `bash scripts/check_paper.sh` (19 checks) ran green before and after; pytest tests/audit (15 tests) ran green before and after.

## User Setup Required

None.

## Next Phase Readiness

**Ready for Plan 18-07** (Wave 3 — AUDIT_REPORT.md aggregation):
- `experiments/results/audit/paper_numbers.csv` is the canonical input for the report's number-by-number traceability table.
- 2 MISMATCH rows pre-flagged require resolution in PAPER_DRAFT.md OR in §6.4 Limitations:
  - Abstract line 12: per-pair annualized Sharpe ≈ 3.2 → trace points to `experiments/results/audit/sharpe_audit.json` which reproduced naive=18.60, BLdP-corrected=7.04. The "3.2" claim does not appear in any audit JSON; it is likely an outdated copy from a prior backtest run with a different annualization formula. Plan 18-07 must either (a) revise the abstract to cite the corrected 7.04 value (with documented assumption stack), or (b) document the 3.2 derivation explicitly (e.g., "treating sqrt(N) with 95% CI bound" — the paper currently does not show the math).
  - §8 Conclusions line 710: same claim, same resolution path.
- `scripts/check_paper.sh audit_per_pair_sharpe_3_2_in_abstract` will need updating in Plan 18-07 if the headline value changes — the canonical `canon()` helper does not currently extract this derived metric (it is not in `headline.json["models"]`).

**Wave 2 status:** Plans 18-04 (Tier 3 cost — pending), 18-05 (Tier 4 survivorship — pending), 18-06 (Tier 5 paper trace — DONE) form Wave 2. 18-04 and 18-05 still need to run before Plan 18-07 (Wave 3) aggregates findings.

## Self-Check: PASSED

- [x] `experiments/audit/build_paper_numbers_csv.py` exists (192 lines >= 80 minimum)
- [x] `experiments/results/audit/paper_numbers.csv` exists (87 lines = header + 86 data rows >= 21 minimum)
- [x] CSV header contains all 8 required columns
- [x] All 86 CSV rows have paper_section restricted to headline sections
- [x] All 86 CSV rows have kind ∈ {sharpe, bps, dollar, pct}
- [x] `scripts/check_paper.sh` has 7 `audit_`-prefixed checks (>= 5 required)
- [x] `bash scripts/check_paper.sh` exits 0 with all 26 checks OK (19 existing + 7 new)
- [x] `PYTHONPATH=. python3 -m pytest tests/audit -q` exits 0 (15/15 passed)
- [x] Commit `cddfe76` exists in git log
- [x] Commit `20dbee4` exists in git log

---
*Phase: 18-system-audit-adversarial-verification*
*Completed: 2026-04-25*
