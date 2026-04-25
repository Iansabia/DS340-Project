---
phase: 18-system-audit-adversarial-verification
plan: 05
subsystem: testing
tags: [audit, survivorship, selection-bias, tier-4, pair-universe, content-addressed-pair-id]

# Dependency graph
requires:
  - phase: 18-01
    provides: tests/audit/ scaffolding + conftest.py fixtures (Tier 4 retroactive_drop_pair_history fixture)
  - phase: 15
    provides: src/live/pair_ids.py make_pair_id() — content-addressed scheme for synthesizing pair_ids from active_matches.json
provides:
  - experiments/audit/audit_survivorship.py (Tier 4 candidate-vs-realized universe diff + 10-pair random sample)
  - experiments/results/audit/survivorship_audit.json (n_candidates=148238, n_realized=144, n_dropped=148094, drop_rate=0.999, verdict=PASS)
  - tests/audit/test_audit_survivorship.py (3 fixture tests passing)
  - 10 pair_ids marked human_classification=pending_human_review for Plan 18-07 Wave 3 checkpoint
affects: [18-07-AUDIT_REPORT, paper-section-6.4-survivorship, finding-3]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Tier 4 audit Pattern 2 contract: {audit, tier, verdict, ran_at, ...findings, assumptions[]} JSON output (verdict ∈ {PASS, REVIEW_REQUIRED, FAILED})"
    - "Content-addressed pair_id synthesis in audits: when active_matches.json lacks top-level pair_id key, use src.live.pair_ids.make_pair_id(kalshi_ticker, poly_id) — keeps audit meaningful for the post-2026-04-11 schema"
    - "Manual-review handoff: random_sample entries marked human_classification='pending_human_review' for Wave 3 checkpoint (no LLM self-classification)"

key-files:
  created:
    - experiments/audit/audit_survivorship.py
    - tests/audit/test_audit_survivorship.py
    - experiments/results/audit/survivorship_audit.json
  modified:
    - .planning/phases/18-system-audit-adversarial-verification/deferred-items.md

key-decisions:
  - "Extended candidate universe via make_pair_id(kalshi_ticker, poly_id) instead of skipping active_matches.json — graceful schema handling that produces a meaningful (not vacuous) drop_rate signal"
  - "Heuristic classifier marks all pairs with < 20 bars as low_overlap (structural); pairs from active_matches.json that never reached the offline pipeline classify as low_overlap_n_bars=0, which is correct by construction"
  - "FAILED verdict reserved for explicit post-resolution outcome metadata (post_hoc, negative_return, loss tokens in drop_reason); benign heuristic fallthrough triggers REVIEW_REQUIRED, not FAILED"
  - "All 10 random-sample entries marked pending_human_review per <additional_context> directive — Plan 18-07 (Wave 3) opens the checkpoint for Ian to manually classify, not the executor"

patterns-established:
  - "Tier 4 verdict logic: PASS if heuristics covered all sample entries (n_review=0); REVIEW_REQUIRED if any fell through; FAILED only on explicit retroactive evidence"
  - "10-pair random sample is determinism-locked: random.Random(SEED=42).sample(sorted(dropped), 10)"
  - "Audit JSON includes a stable assumption stack so Plan 18-07's AUDIT_REPORT.md can mechanically aggregate findings"

requirements-completed: [AUDIT-04]

# Metrics
duration: 5min
completed: 2026-04-25
---

# Phase 18 Plan 05: Tier 4 Survivorship/Selection Audit Summary

**Quantified survivorship rate of 99.9% (148,094 of 148,238 candidate pairs dropped before reaching canonical test universe) with deterministic 10-pair random sample — all classified `low_overlap_n_bars=0` (structural, live-only candidates that never entered offline pipeline) — verdict=PASS pending Ian's manual review per Plan 18-07.**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-04-25T19:52Z
- **Completed:** 2026-04-25T19:57Z
- **Tasks:** 2 (both auto, both TDD)
- **Files modified:** 4 (3 created + 1 deferred-items append)

## Accomplishments

- `experiments/audit/audit_survivorship.py` (223 lines) computes candidate ∪ active_matches − realized = dropped, with deterministic seed=42 random-sample of 10 dropped pairs and heuristic drop-reason classification
- `experiments/results/audit/survivorship_audit.json` produced: n_candidate_pairs=148,238, n_realized_pairs=144, n_dropped_pairs=148,094, drop_rate=0.9990, verdict=PASS
- `tests/audit/test_audit_survivorship.py` 3 tests passing (low_overlap, pre_test_window, post_hoc_drop alias)
- 10 pair_ids handed off to Plan 18-07 Wave 3 checkpoint for Ian's manual classification
- Tier 4's load-bearing evidence (per RESEARCH.md): "drop rate of 99.9% with structural reasons in all 10 random samples" — strongest possible heuristic finding before manual review

## Audit Findings (Verbatim from JSON)

| Field                          | Value                                                                          |
| ------------------------------ | ------------------------------------------------------------------------------ |
| `verdict`                      | **PASS** (heuristics covered all 10 sample entries; n_requiring_manual_review=0) |
| `n_candidate_pairs`            | 148,238 (144 from aligned_pairs.parquet + 148,094 synthesized from active_matches.json via make_pair_id) |
| `n_realized_pairs`             | 144 (test.parquet pair_id unique)                                              |
| `n_dropped_pairs`              | 148,094 (148,238 − 144)                                                        |
| `drop_rate`                    | 0.9990 (essentially all live-discovery candidates never reached offline pipeline) |
| `random_sample_size`           | 10                                                                             |
| `n_requiring_manual_review`    | 0 (all 10 classified as `structural` by heuristic)                             |

### 10-Pair Random Sample (seed=42) — All `pending_human_review` for Plan 18-07

| # | pair_id                                  | drop_reason_inferred           | manual_classification |
| - | ---------------------------------------- | ------------------------------ | --------------------- |
| 1 | `kxbtc26apr1500t61200-0x58fb4378`        | low_overlap_n_bars=0           | structural            |
| 2 | `kxbnb26apr1822b467-0x1b3b0ec3`          | low_overlap_n_bars=0           | structural            |
| 3 | `kxcpicore26aprt01-0x42df8b1f`           | low_overlap_n_bars=0           | structural            |
| 4 | `kxbtcd26apr2217t8099999-0x23fb92bb`     | low_overlap_n_bars=0           | structural            |
| 5 | `kxbtcd26apr2005t7229999-0x63743613`     | low_overlap_n_bars=0           | structural            |
| 6 | `kxbtc26apr2017b73125-0x63743613`        | low_overlap_n_bars=0           | structural            |
| 7 | `kxbtc26apr1400b64050-0x1c2f06de`        | low_overlap_n_bars=0           | structural            |
| 8 | `kxsole26apr2010b69-0xedfc3d87`          | low_overlap_n_bars=0           | structural            |
| 9 | `kxbnbd26apr2407t63499-0x1b3b0ec3`       | low_overlap_n_bars=0           | structural            |
| 10| `kxethd26apr2122t152999-0x4c608ba8`      | low_overlap_n_bars=0           | structural            |

**Pattern in sample:** Every entry is a Kalshi crypto/CPI sub-hourly contract from late April 2026 (after the test.parquet snapshot of 2026-04-09). These are live-discovery candidates that postdate the offline pipeline build — they never reached `aligned_pairs.parquet` because they didn't exist yet on 2026-04-09. This is a structural temporal-cohort mismatch, not retroactive dropping. Ian to confirm in Plan 18-07.

**Verdict caveat:** PASS is heuristic-only. Per VALIDATION.md manual-only verifications row, the final verdict requires Ian's sign-off after reviewing each of the 10 pair_ids in the JSON. Plan 18-07 opens that checkpoint.

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement audit_survivorship.py** — `efd6807` (feat)
2. **Task 2: Fixture tests for classify_drop_reason** — `33f7b7b` (test)

**Plan metadata commit:** `52d942d` (docs: complete plan)

## Files Created/Modified

- `experiments/audit/audit_survivorship.py` — 223-line Tier 4 audit script (functions: `candidate_pair_universe`, `realized_pair_universe`, `classify_drop_reason`, `_coerce_active_match_pair_id`, `main`)
- `tests/audit/test_audit_survivorship.py` — 3 fixture tests proving classifier distinguishes low_overlap (5 bars) vs pre_test_window (50 bars), plus VALIDATION.md naming alias
- `experiments/results/audit/survivorship_audit.json` — Tier 4 audit output, consumed by Plan 18-07
- `.planning/phases/18-system-audit-adversarial-verification/deferred-items.md` — re-logged pre-existing `test_excludes_pair_with_insufficient_trades` failure (out-of-scope, already documented from Plan 18-02)

## Decisions Made

1. **Extended candidate universe via `make_pair_id` synthesis** — RESEARCH.md skeleton uses `m.get("pair_id")` which returns None for the current 148k-entry active_matches.json (schema stores `kalshi_ticker` + `poly_id` separately, not a top-level `pair_id` key). Without synthesis, the candidate set would equal the realized set (144 == 144), yielding n_dropped=0 and a vacuous audit. With synthesis via `src.live.pair_ids.make_pair_id`, we recover 148,094 dropped candidates and a meaningful drop_rate=0.999. This is consistent with the plan's "If `data/live/active_matches.json` does not exist or has different schema, gracefully skip that source" — schema is *present*, just shaped differently, so we adapt rather than skip.

2. **All sample entries marked `human_classification: pending_human_review`** — per `<additional_context>` directive, the executor must NOT classify the pairs; Plan 18-07 (Wave 3) opens the manual-review checkpoint for Ian. Heuristic `manual_classification_required` field is the LLM's best guess; `human_classification` is the actual decision field.

3. **FAILED reserved for retroactive evidence only** — verdict=FAILED triggers iff any sample entry's `drop_reason_inferred` contains `post_hoc`, `negative_return`, or `loss` tokens. Heuristic fallthrough (`unknown_structural` cases) triggers REVIEW_REQUIRED. This matches the Pattern 2 contract from Plan 18-03 (Suspicious findings ≠ FAILED).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Active matches schema synthesis via `make_pair_id`**

- **Found during:** Task 1 (initial implementation)
- **Issue:** RESEARCH.md skeleton's `m.get("pair_id")` returns None for every entry in the current 148k-row active_matches.json — its schema uses `kalshi_ticker` + `poly_id` keys instead of a top-level `pair_id`. Pure verbatim copy would have produced n_dropped=0 (candidates == realized, both 144) and a vacuous Tier 4 audit with no random sample to classify.
- **Fix:** Added `_coerce_active_match_pair_id()` helper that tries (1) direct `pair_id` key, (2) `make_pair_id(kalshi_ticker, poly_id)` content-addressed synthesis using the post-2026-04-11 scheme from `src/live/pair_ids.py`, (3) skip entry. This recovers 148,094 dropped candidates and produces a meaningful drop_rate=0.999.
- **Files modified:** `experiments/audit/audit_survivorship.py` (lines 56-79, helper function)
- **Verification:** `n_candidate_pairs=148238` in JSON output (vs. 144 if active_matches were skipped); 10 sample pair_ids in JSON match the content-addressed format used in `aligned_pairs.parquet` (`kxXXX-0xYYYYYYYY`).
- **Committed in:** `efd6807` (Task 1)

**2. [Rule 2 - Missing Critical] Added `human_classification: pending_human_review` field to each random_sample entry**

- **Found during:** Task 1 (per `<additional_context>` directive)
- **Issue:** RESEARCH.md skeleton only emits `manual_classification_required` (the LLM's heuristic verdict). Plan 18-07 needs an explicit "human has not yet reviewed this" sentinel so the Wave 3 checkpoint can mechanically detect which pairs are pending vs. confirmed.
- **Fix:** Added `human_classification: "pending_human_review"` to each entry. Distinguishes heuristic prediction (`manual_classification_required`) from human decision (`human_classification`). When Ian reviews, he replaces `pending_human_review` with `structural` or `retroactive` and reruns to lock the verdict.
- **Files modified:** `experiments/audit/audit_survivorship.py` (lines 145-152)
- **Verification:** `jq '.random_sample[].human_classification' ... | sort -u` returns `"pending_human_review"` for all 10 entries.
- **Committed in:** `efd6807` (Task 1)

**3. [Rule 2 - Missing Critical] Added FAILED-detection logic for explicit retroactive evidence**

- **Found during:** Task 1 (per `<interfaces>` schema spec at lines 105-106)
- **Issue:** RESEARCH.md skeleton always returns PASS or REVIEW_REQUIRED; never emits FAILED. The `<interfaces>` block explicitly states FAILED is reserved for entries whose `drop_reason` mentions post-resolution outcome metadata (e.g., `realized_return < 0`).
- **Fix:** Added `n_failed` counter that scans for `post_hoc`, `negative_return`, or `loss` tokens in any `drop_reason_inferred`. If any present → verdict=FAILED. Otherwise standard PASS/REVIEW_REQUIRED logic.
- **Files modified:** `experiments/audit/audit_survivorship.py` (lines 158-167)
- **Verification:** Current run produces verdict=PASS (no failed tokens in heuristic classifications); FAILED branch reachable if a future audit run encounters retroactive evidence.
- **Committed in:** `efd6807` (Task 1)

---

**Total deviations:** 3 auto-fixed (all Rule 2 — missing critical functionality from skeleton)
**Impact on plan:** All three deviations are necessary to produce a meaningful audit (vs. vacuous PASS on empty sample) and to satisfy the Pattern 2 contract documented in the plan's `<interfaces>` block. No scope creep — all fixes stay inside the audit_survivorship.py module and don't touch unrelated code.

## Issues Encountered

- **Pre-existing test failure** in `tests/data/test_aligner.py::TestAlignAllPairs::test_excludes_pair_with_insufficient_trades` re-confirmed during regression sweep (already documented from Plan 18-02). OUT OF SCOPE — Plan 18-05 touched only `experiments/audit/` and `tests/audit/`. Re-logged in `deferred-items.md`.

## Next Phase Readiness

- **Plan 18-06 (Tier 5 paper-number trace)** can proceed in parallel with 18-04/05 — no dependency on this audit's verdict.
- **Plan 18-07 (Wave 3 AUDIT_REPORT.md + checkpoint)** is the consumer of this audit:
  1. Mechanically aggregate `experiments/results/audit/survivorship_audit.json` into AUDIT_REPORT.md row "Tier 4 — Selection / survivorship: PASS (pending Ian's manual review of 10 pair_ids)".
  2. Open checkpoint for Ian to classify each of the 10 pair_ids and rerun the audit to lock the final verdict.
  3. If all 10 confirm structural → audit passes; if any classify retroactive → FAILED, paper §6.4 item 3 must be rewritten.
- **Tier 4 evidence quality:** The 99.9% drop_rate is dominated by temporal-cohort mismatch (live-discovery candidates from late April 2026 that postdate the test.parquet snapshot of 2026-04-09), not retroactive selection. This is the "structural" finding Phase 14 §6.4 item 3 already disclosed; this audit *quantifies* it.

## Self-Check: PASSED

Verified:
- `experiments/audit/audit_survivorship.py` exists (223 lines, ≥ 130) ✓
- `tests/audit/test_audit_survivorship.py` exists (39 lines, ≥ 25) ✓
- `experiments/results/audit/survivorship_audit.json` exists with all required keys ✓
- Commit `efd6807` exists in `git log` ✓
- Commit `33f7b7b` exists in `git log` ✓
- 7/7 audit tests pass (3 new + 4 fixtures) ✓
- `PYTHONPATH=. python3 experiments/audit/audit_survivorship.py` exits 0 ✓
- jq invariants: random_sample length ≤ 10, drop_rate ∈ [0,1], verdict ∈ {PASS, REVIEW_REQUIRED, FAILED} ✓

---
*Phase: 18-system-audit-adversarial-verification*
*Plan: 05 (Wave 2 Tier 4)*
*Completed: 2026-04-25*
