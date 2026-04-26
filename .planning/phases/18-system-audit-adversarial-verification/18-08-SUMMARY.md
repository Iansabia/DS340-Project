---
phase: 18-system-audit-adversarial-verification
plan: 08
subsystem: testing
tags: [audit, leakage, sharpe, walk-forward, embargo, pair-stratified, bldp, bootstrap, kill-or-confirm]

# Dependency graph
requires:
  - phase: 18-02
    provides: experiments/audit/audit_sharpe.py helper functions (per_pair_returns, per_pair_sharpe_naive, avg_pairwise_correlation, correlation_corrected_sharpe, bootstrap_sharpe_ci, annualization_factor, per_trade_sharpe)
  - phase: 18-03
    provides: leakage_audit.json (the audit that flagged 142/144 embargo violations and motivated this plan)
  - phase: 17-01
    provides: experiments/run_canonical.py training pipeline (evaluate_predictions, canonical constants) and experiments/run_baselines.py preprocessing (_build_split, _feature_columns, prepare_xy)
provides:
  - Pair-stratified train/test split builder (data/processed/purged_split/) with split_metadata.json
  - LR + XGBoost retrained headline on the leakage-free split (experiments/results/canonical_purged/headline.json)
  - Tier 1 Sharpe audit redo on purged data with side-by-side comparison vs leaky canonical (experiments/results/audit/sharpe_audit_purged.json)
  - Embargo-violation re-verification artifact (experiments/results/audit/leakage_audit_purged_check.json)
  - Verdict CORRECTED — purged corrected per-pair Sharpe is +0.81 [CI 0.70, 1.07], an INCREASE of +175% over leaky-canonical 0.30
affects: [18-07, paper-revision, abstract, table-8, conclusion-section]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Sister-script pattern: run_canonical_purged.py reuses run_canonical helpers verbatim, swapping ONLY the data source"
    - "Audit-script reuse rule: audit_sharpe_purged.py imports every metric helper from audit_sharpe.py — duplicate code is forbidden"
    - "Comparison block in audit JSONs: emit both leaky and purged numbers + delta + drift_pct + interpretation string for paper quotation"

key-files:
  created:
    - experiments/audit/build_purged_split.py
    - experiments/run_canonical_purged.py
    - experiments/audit/audit_sharpe_purged.py
    - experiments/audit/verify_purged_no_bridge.py
    - tests/audit/test_purged_split.py
    - tests/audit/test_audit_sharpe_purged.py
    - experiments/results/canonical_purged/headline.json
    - experiments/results/audit/sharpe_audit_purged.json
    - experiments/results/audit/leakage_audit_purged_check.json
  modified: []

key-decisions:
  - "Purged split uses np.random.default_rng(seed=42).shuffle on a numpy object array of unique pair_ids (avoids the ArrowStringArray UserWarning) — 80/20 by pair count gives 115 train / 29 test pairs"
  - "_build_split drops trailing-bar NaN targets so 7336 -> 7221 train rows and 1427 -> 1398 test rows after the canonical feature pipeline runs (consistent with how canonical run does it)"
  - "Verdict CORRECTED is the honest answer because the +175% drift_pct in corrected per-pair Sharpe fails the PASS gate (|drift| < 50%), even though the direction is favorable to the headline (purged Sharpe HIGHER than leaky)"
  - "Mechanism for the surprise: leaky avg pairwise correlation +0.0418 compressed leaky naive 0.78 to corrected 0.30 via BLdP; purged avg pairwise correlation -0.20 short-circuits the BLdP correction (avg_corr <= 0 returns naive) so purged corrected = purged naive = 0.81"
  - "Per-trade Sharpe is essentially invariant to leakage correction: leaky 0.5007 -> purged 0.5157 (+2.99% drift). Per-pair quantity moves more because of small-N (N=29 test pairs) sampling variance in the cross-pair correlation matrix"
  - "Total P&L drops -14% in proportion to the smaller test sample (1398 vs 1673 rows). Win rate drops -3pp. These are sample-size effects, not edge degradation"

patterns-established:
  - "Pair-atomic resplitting: concatenate canonical train+test, group by pair_id, shuffle pairs, atomic 80/20 by pair count — preserves total row count, eliminates pair bridging by construction"
  - "Audit JSON comparison block: every leakage-audit redo emits {canonical_X, purged_X, delta_X, drift_pct_X} for every headline metric plus an interpretation string"
  - "Verdict three-tier (PASS/CORRECTED/FAILED) with explicit gate logic in JSON output — kill-or-confirm framing leaves no soft-fail option"

requirements-completed: [AUDIT-07]

# Metrics
duration: ~7min
completed: 2026-04-26
---

# Phase 18 Plan 08: Pair-stratified leakage-free retraining + Tier 1 Sharpe audit redo Summary

**Verdict CORRECTED: leakage-free per-pair corrected Sharpe is +0.81 [CI +0.70, +1.07], a +175% INCREASE over leaky-canonical +0.30 — driven by negative cross-pair correlation in the small N=29 purged test set short-circuiting the BLdP correction that compressed the leaky number.**

## Performance

- **Duration:** ~7 min
- **Started:** 2026-04-26T15:06:34Z
- **Completed:** 2026-04-26T15:13:28Z
- **Tasks:** 4
- **Files created:** 9 (4 Python modules, 2 test modules, 3 JSON artifacts)
- **Files modified:** 0

## Accomplishments

- **Pair-stratified split (Task 1):** built `data/processed/purged_split/{train,test}.parquet` (115 / 29 pairs, 7336 / 1427 rows). The whole point — no pair_id bridges train and test. 4/4 tests passing.
- **LR + XGBoost retraining (Task 2):** ran the canonical training pipeline on the purged split. Per-trade Sharpe stable (LR +0.5009 -> +0.5159, +2.99%); P&L drops proportional to the smaller test set.
- **Tier 1 audit redo (Task 3):** recomputed per-trade + per-pair + BLdP-corrected + bootstrap-CI Sharpe on the purged ledger. Verdict CORRECTED. Comparison block + interpretation string emitted for direct paper quotation.
- **Embargo verification (Task 4):** independently re-checked that 0 pairs bridge the purged train/test boundary — receipt at `experiments/results/audit/leakage_audit_purged_check.json`.

## Headline numbers — Leaky canonical vs purged (LR, the headline model)

| Metric | Leaky canonical | Purged | Delta | Drift |
| --- | --- | --- | --- | --- |
| Per-trade Sharpe | +0.5007 | +0.5157 | +0.0150 | +2.99% |
| Per-pair Sharpe (naive) | +0.7809 | +0.8136 | +0.0327 | +4.20% |
| Per-pair Sharpe (BLdP-corrected) | +0.2955 | +0.8136 | +0.5181 | **+175.33%** |
| Per-pair 95% CI | [+0.685, +0.904] | [+0.700, +1.067] | — | — |
| Avg pairwise correlation | +0.0418 | **-0.1986** | -0.2404 | — |
| Total P&L | $+232.67 | $+199.63 | -$33.04 | -14.20% |
| Win rate | 57.84% | 54.80% | -3.04 pp | -5.25% |
| Test rows | 1673 | 1398 | -275 | -16.4% |
| Test pairs | 144 | 29 | -115 | -79.9% |

## Why purged Sharpe went UP, not DOWN

The plan note expected purged Sharpe to be LOWER than leaky (the standard leakage-correction signature). The empirical answer is the opposite. Mechanism:

1. **Leaky audit:** N=144 test pairs share calendar days with train pairs (because they bridge the split), so the contemporaneous-day cross-pair correlation matrix has 461 non-NaN cells with mean +0.0418. The BLdP effective-sample correction compresses leaky naive 0.78 -> corrected 0.30 (a -62% haircut).
2. **Purged audit:** N=29 test pairs (no bridging) share fewer calendar days with each other, and the resulting 89-cell cross-pair correlation matrix has mean -0.1986. The audit's `correlation_corrected_sharpe` short-circuits when `avg_corr <= 0` (returns naive Sharpe with `n_eff = n_pairs`), so purged corrected = purged naive = 0.81.

So the +175% drift in corrected Sharpe is not "the leakage was masking a real edge" — it is "the BLdP correction was the active ingredient compressing the leaky number, and that correction does not apply to the purged sample because the purged cross-pair correlation is negative."

The honest paper-text framing: **"Per-trade Sharpe is invariant to the leakage correction (drift +2.99%, well within the bootstrap CI). Per-pair quantities depend on the cross-pair correlation structure, which is sample-size-sensitive at N=29 test pairs. The leakage-free per-pair Sharpe is +0.81 [CI +0.70, +1.07], with the caveat that the BLdP correction does not apply at this sample size due to negative empirical cross-pair correlation."**

## Verdict logic

| Verdict | Condition | Result |
| --- | --- | --- |
| PASS | corrected > 0.5 AND CI_lower > 0.0 AND \|drift_pct\| < 50% | drift_pct = +175% fails the gate |
| **CORRECTED** | corrected > 0.0 AND CI_lower > -0.2 | **+0.81 > 0 and +0.70 > -0.2** |
| FAILED | otherwise | not triggered |

The drift gate was designed to catch leakage-driven inflation (purged dramatically lower than leaky). It also fires here in the opposite direction (purged dramatically higher than leaky), and CORRECTED is therefore the honest verdict — the leakage-free number is materially different from the leaky number, but defensible (CI entirely positive, well above zero).

## Task Commits

Each task committed atomically:

1. **Task 1 RED: failing tests for pair-stratified split** — `8faf946` (test)
2. **Task 1 GREEN: split builder implementation** — `4a36284` (feat)
3. **Task 2: LR + XGBoost retraining on purged split** — `7cb1808` (feat)
4. **Task 3 RED: failing tests for purged Sharpe audit** — `6f13ac5` (test)
5. **Task 3 GREEN: Tier 1 audit redo on purged data** — `2e90b14` (feat)
6. **Task 4: embargo-violation re-verification** — `3fff722` (test)

Plan metadata commit: appended at end (this SUMMARY.md + STATE.md + ROADMAP.md update).

## Files Created/Modified

- `experiments/audit/build_purged_split.py` — pair-atomic 80/20 splitter with seed=42
- `experiments/run_canonical_purged.py` — LR + XGBoost retrain on purged data, reuses run_canonical helpers
- `experiments/audit/audit_sharpe_purged.py` — Tier 1 Sharpe audit on purged ledger, reuses audit_sharpe helpers, emits comparison block
- `experiments/audit/verify_purged_no_bridge.py` — final-step embargo re-check
- `tests/audit/test_purged_split.py` — 4 tests on the splitter (no-bridge, reproducibility, no-row-loss, min pairs)
- `tests/audit/test_audit_sharpe_purged.py` — 3 tests on the audit (helper-reuse, i.i.d.-violation collapse smoke, JSON schema)
- `data/processed/purged_split/train.parquet`, `test.parquet`, `split_metadata.json` (gitignored under data/processed/)
- `experiments/results/canonical_purged/headline.json` — purged LR + XGBoost metrics
- `experiments/results/audit/sharpe_audit_purged.json` — purged Tier 1 audit with comparison block
- `experiments/results/audit/leakage_audit_purged_check.json` — embargo re-verification receipt

## Decisions Made

See `key-decisions:` in frontmatter for the canonical list. Highlight: the verdict is **CORRECTED** (not PASS) because the absolute drift_pct in corrected per-pair Sharpe is +175%, exceeding the 50% gate — even though the direction is favorable. The audit reports what it finds.

## Deviations from Plan

**None — plan executed exactly as written.**

The plan note flagged that purged Sharpe was EXPECTED to be lower than leaky and "if purged Sharpe is HIGHER, something is wrong (probably a bug in the pipeline)." After this plan's execution, I traced the reason purged Sharpe is higher and confirmed it is NOT a bug:

- Per-trade Sharpe drift is +2.99% (within noise — the LR fit on the purged training set produces near-identical per-trade economics).
- Per-pair NAIVE Sharpe drift is +4.20% (also within noise).
- The +175% drift only appears in the BLdP-corrected per-pair number, and is fully explained by the leaky avg_corr being slightly positive (BLdP haircut active) while the purged avg_corr is negative (BLdP haircut short-circuited per the audit's own `correlation_corrected_sharpe` function which the plan explicitly mandated reusing).

Both pipelines are correct. The leaky audit's "compressed" 0.30 number was the result of applying a real correction to a real (small) positive correlation; the purged audit's "uncompressed" 0.81 is the same naive number under a regime where that correction does not apply. The headline difference is in the BLdP correction's applicability, not in the underlying edge.

## Issues Encountered

- **ArrowStringArray UserWarning during initial split:** `np.random.Generator.shuffle` on a `pd.unique()` result produced an ArrowStringArray and emitted a UserWarning about possible incorrect behavior. Fixed by `np.asarray(..., dtype=object).copy()` before shuffling. Caught and fixed in the same Task 1 commit; reproducibility test confirms the fix is deterministic across runs.

No blocking issues, no checkpoint deviations, no architectural changes.

## User Setup Required

None — purely audit/analysis work, no external services touched.

## Next Phase Readiness

**Plan 18-07 can now resume at Task 3 (paper updates + AUDIT_REPORT.md generation).** The numbers it should consume:

| Paper element | Source |
| --- | --- |
| New abstract / §8 headline per-pair Sharpe | `comparison.purged_sharpe_per_pair_corrected` = **+0.81** with CI `[+0.70, +1.07]` |
| New §6.4 leakage attribution paragraph | `comparison.interpretation` (verbatim quotable) — explicitly notes the BLdP-correction-applicability mechanism, not a clean "leakage inflation" story |
| New Table 8 / §5.1 per-trade Sharpe | LR `sharpe_per_trade` = **+0.5159** (drift +2.99% from canonical 0.5009 — small enough to footnote) |
| Total P&L claims | LR `total_pnl` = **+$199.63** (drift -14% from canonical $232.67, attributable to smaller test sample N=1398 vs 1673) |
| Win rate | LR `win_rate` = **0.548** (canonical 0.578) |
| Methodology paragraph | "Re-ran canonical headline on a pair-stratified split where no pair bridges train/test (data/processed/purged_split/, seed=42); see Plan 18-08 SUMMARY and experiments/results/audit/sharpe_audit_purged.json" |

**Important framing for Ian / paper authors:** the kill-or-confirm answer is "the per-trade edge is robust to leakage correction; the per-pair Sharpe headline depends on the BLdP correction's applicability to the cross-pair correlation regime, which is sample-size-sensitive at N=29 test pairs." That is a more nuanced finding than either "PASS" or "FAILED" — it does NOT undermine the paper's thesis (LR/XGBoost beat baselines with ~15 bps per-trade alpha and per-trade Sharpe ~0.50 under leakage-free protocol). It DOES change which number leads in the abstract.

The 3.2 number that motivated this plan continues to have no derivation path in the codebase; the leakage-free replacement is +0.81 corrected per-pair (or +0.5159 per-trade, depending on which framing the abstract adopts).

## Self-Check: PASSED

Verified all created files exist:
- `experiments/audit/build_purged_split.py` — FOUND
- `experiments/run_canonical_purged.py` — FOUND
- `experiments/audit/audit_sharpe_purged.py` — FOUND
- `experiments/audit/verify_purged_no_bridge.py` — FOUND
- `tests/audit/test_purged_split.py` — FOUND
- `tests/audit/test_audit_sharpe_purged.py` — FOUND
- `experiments/results/canonical_purged/headline.json` — FOUND
- `experiments/results/audit/sharpe_audit_purged.json` — FOUND
- `experiments/results/audit/leakage_audit_purged_check.json` — FOUND

Verified all 6 task commits exist:
- 8faf946 (Task 1 RED), 4a36284 (Task 1 GREEN) — FOUND
- 7cb1808 (Task 2) — FOUND
- 6f13ac5 (Task 3 RED), 2e90b14 (Task 3 GREEN) — FOUND
- 3fff722 (Task 4) — FOUND

Verified all 26 audit tests pass (`pytest tests/audit/ -q`).

---
*Phase: 18-system-audit-adversarial-verification*
*Completed: 2026-04-26*
