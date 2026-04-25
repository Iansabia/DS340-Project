---
phase: 18-system-audit-adversarial-verification
plan: 02
subsystem: testing
tags: [audit, sharpe, bootstrap, bailey-lopez-de-prado, correlation-correction, tier-1, kill-or-confirm]

# Dependency graph
requires:
  - phase: 18-01
    provides: tests/audit/conftest.py fixtures (perfectly_correlated_pair_returns) + experiments/audit/__init__.py + experiments/results/audit/ scaffolding
  - phase: 17-01
    provides: experiments/results/canonical/headline.json (the LR sharpe_per_trade=0.5009 number being audited)
  - phase: 08-02
    provides: experiments/run_baselines.py (_build_split, _feature_columns, load_train_test, prepare_xy, NON_FEATURE_COLUMNS, TARGET_COLUMN imports)
provides:
  - "experiments/audit/audit_sharpe.py: Tier 1 Sharpe recompute with Bailey-Lopez de Prado correlation correction + 10,000-resample bootstrap CI"
  - "experiments/results/audit/sharpe_audit.json: machine-readable Tier 1 verdict (PASS) with full assumption stack"
  - "tests/audit/test_audit_sharpe.py: fixture test proving correlation correction collapses Sharpe under perfect pair correlation"
  - "Empirical finding: avg_pair_corr=0.042 -> n_eff=20.6 -> per_pair_sharpe drops from naive 0.781 to corr_corrected 0.296 (62% reduction)"
affects:
  - 18-03 (Tier 2 leakage audit, can run in parallel)
  - 18-07 (AUDIT_REPORT.md generator consumes sharpe_audit.json)
  - paper-numerics (per_pair_sharpe_corr_corrected may justify abstract caveat in §6.4 Phase 18 follow-up)

# Tech tracking
tech-stack:
  added: []  # zero new dependencies; uses numpy/pandas already pinned
  patterns:
    - "Audit script pattern: recompute headline metric from raw ledger, write JSON verdict to experiments/results/audit/, fixture-test the failure mode the audit is designed to catch"
    - "Verdict trichotomy: PASS / CORRECTED / FAILED — no soft-fail, no REVIEW state"
    - "Bailey-Lopez de Prado n_eff correction codified as reusable function for downstream Sharpe audits"

key-files:
  created:
    - experiments/audit/audit_sharpe.py
    - experiments/results/audit/sharpe_audit.json
    - tests/audit/test_audit_sharpe.py
    - .planning/phases/18-system-audit-adversarial-verification/deferred-items.md
  modified: []

key-decisions:
  - "Audit verdict is PASS because avg_pair_corr=0.042 falls below the 0.10 verdict-trigger threshold per plan spec — but the corrected per-pair Sharpe (0.296) is 62% below the naive (0.781), and that material reduction must surface in the Plan 18-07 AUDIT_REPORT.md narrative even though it does not flip the verdict"
  - "Annualization derived from ledger time span (test_span_days=92.67, pairs_per_year=567, factor=23.8) rather than hard-coded — paper §5.8 implicit factor must be cross-checked against this in Plan 18-07"
  - "Bootstrap is plain percentile (10,000 resamples, seed=42) — does NOT correct for autocorrelation; assumption stack flags Politis-Romano stationary bootstrap as the upgrade path if AR(1) coef is large"
  - "Pre-existing tests/data/test_aligner.py failure (Phase 14 provenance) deferred to deferred-items.md per scope-boundary rule — not blocking AUDIT-01"

patterns-established:
  - "Pattern: Audit script reads canonical/headline.json as single source of truth (Pattern 1 from 18-RESEARCH.md), recomputes from raw data, emits JSON verdict to experiments/results/audit/"
  - "Pattern: Each audit script paired with one fixture test in tests/audit/ that injects the failure mode the audit must catch (TDD applied at the audit-correctness layer, not just the production-code layer)"
  - "Pattern: Verdict logic centralized in main() with explicit thresholds (drift>0.01 -> FAILED; avg_corr>0.10 AND corrected<0.5*naive -> CORRECTED; else PASS) — no implicit verdicts"

requirements-completed:
  - AUDIT-01

# Metrics
duration: 4min
completed: 2026-04-25
---

# Phase 18 Plan 02: Tier 1 Sharpe 3.2 Audit Summary

**Recomputed LR per-trade Sharpe (0.5007) within 0.00016 of canonical 0.5009; per-pair Sharpe naive=0.781 collapses to 0.296 (-62%) under Bailey-Lopez de Prado correlation correction at avg_corr=0.042; verdict=PASS; bootstrap 95% CI [0.685, 0.904] on naive Sharpe.**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-04-25T19:46:16Z
- **Completed:** 2026-04-25T19:49:58Z
- **Tasks:** 2 (both committed atomically)
- **Files modified:** 4 created (audit script + JSON output + test + deferred-items log)

## Accomplishments

- **Tier 1 audit script live and reproducible.** `experiments/audit/audit_sharpe.py` (268 lines, 9 functions) reproduces the canonical LR per-trade Sharpe within 0.0002 of the canonical `headline.json` value — far below the 0.01 drift tolerance.
- **Bailey-Lopez de Prado correlation correction applied.** `n_eff = N / (1 + (N-1) * avg_corr)` with `avg_pair_corr = 0.042` collapses the effective sample from 144 to 20.6 pairs, dropping per-pair Sharpe from naive 0.781 to corrected 0.296 (a 62% material reduction even though the verdict-trigger threshold is set at 0.5).
- **Bootstrap 95% CI computed reproducibly.** 10,000 percentile-method resamples (seed=42) produce CI=[0.685, 0.904] on the naive per-pair Sharpe, providing the confidence interval Plan 17-02 audit promised the paper would carry going forward.
- **Annualization formula made explicit.** Audit derives `pairs_per_year = 567.19` from `test_span_days = 92.67` and 144 pairs, yielding `annualization_factor = 23.8`. This is the multiplication chain the paper §5.8 leaves implicit; Plan 18-07 will reconcile it against the abstract's "≈3.2" claim.
- **Audit-correctness fixture test passes.** `tests/audit/test_audit_sharpe.py` synthesizes a 144-pair × 30-day perfectly-correlated ledger (`avg_corr ≈ 1.0`) and asserts the correction collapses the corrected Sharpe to <5% of naive — proving the audit catches the failure mode it is designed to catch.

## Verbatim Audit Numbers (from `experiments/results/audit/sharpe_audit.json`)

| Field                                    | Value                                |
| ---------------------------------------- | ------------------------------------ |
| `verdict`                                | `PASS`                               |
| `per_trade_sharpe_recomputed`            | `0.500716001685626`                  |
| `per_trade_sharpe_canonical`             | `0.5008777055495818`                 |
| `per_trade_sharpe_drift`                 | `0.0001617038639557533` (< 0.01 ✓)   |
| `per_pair_sharpe_naive`                  | `0.780852099060539`                  |
| `per_pair_sharpe_naive_ci_95`            | `[0.685, 0.904]`                     |
| `avg_pairwise_corr`                      | `0.041830805130603534`               |
| `n_pairs_compared`                       | `461` (cells in upper-triangle ex-NaN) |
| `n_eff`                                  | `20.62503854560834`                  |
| `per_pair_sharpe_corr_corrected`         | `0.29551866792107984`                |
| `annualization.test_span_days`           | `92.67`                              |
| `annualization.pairs_per_year`           | `567.19`                             |
| `annualization.annualization_factor`     | `23.8158`                            |
| `per_pair_sharpe_annualized_naive`       | `18.60`                              |
| `per_pair_sharpe_annualized_corrected`   | `7.04`                               |
| `n_bootstrap`                            | `10000`                              |
| `assumptions` (length)                   | `4`                                  |

## Paper-Correction Trigger Status

**The verdict is PASS, so Plan 18-07 does NOT need to update the paper headline number based on Tier 1 alone.** However, Plan 18-07 SHOULD surface the following in `AUDIT_REPORT.md`:

1. **avg_pair_corr is non-trivial (0.042) and the corrected Sharpe is 62% below naive** — this is documented even though it doesn't flip the verdict, because the *reader* will ask the question.
2. **The annualized "3.2" in the paper does not match this audit's annualization factor** — paper-numerics audit (Tier 5, Plan 18-06) must trace the implicit factor back to its source script. The audit's empirical factor of 23.8 (per-pair × √567) yields a much higher annualized number (18.6 naive, 7.0 corrected) than the paper's 3.2, suggesting the paper used a different (smaller) annualization multiplier — likely `sqrt(trades_per_year / trades_per_pair)` rather than `sqrt(pairs_per_year)`. **This is a Tier 5 finding, not a Tier 1 verdict change.**
3. **Bootstrap CI on corrected Sharpe is NOT computed** — the script only bootstraps the naive Sharpe vector. If Plan 18-07 wants a CI on the corrected number, it can compose `bootstrap_sharpe_ci × sqrt(n_eff/N)` ratio at the percentile level.

## Pytest Output Proving the Fixture Catches the Failure Mode

```
tests/audit/test_audit_sharpe.py::test_audit_sharpe_catches_inflated_independence PASSED [ 20%]
tests/audit/test_fixtures.py::test_perfectly_correlated_returns_has_avg_corr_near_one PASSED [ 40%]
tests/audit/test_fixtures.py::test_synthetic_lookahead_src_contains_negative_shift PASSED [ 60%]
tests/audit/test_fixtures.py::test_zero_fee_kwargs_match_audit_target PASSED    [ 80%]
tests/audit/test_fixtures.py::test_retroactive_drop_marker_set PASSED    [100%]

============================== 5 passed in 2.39s ===============================
```

The fixture test asserts (verbatim from RESEARCH.md):
- `avg_corr > 0.99` (synthetic perfect-correlation ledger)
- `n_eff < 5` (effective sample collapses from 144 to ~1)
- `|s_corr| < 0.05 * |s_naive|` (correction reduces Sharpe by >95%)

## Task Commits

Each task was committed atomically:

1. **Task 1: experiments/audit/audit_sharpe.py + sharpe_audit.json** — `0b4dff3` (feat)
2. **Task 2: tests/audit/test_audit_sharpe.py** — `c1214db` (test)

**Plan metadata:** _to be committed at end of execute-plan flow_

## Files Created/Modified

- `experiments/audit/audit_sharpe.py` (268 lines) — Tier 1 Sharpe recompute, correlation correction, bootstrap CI, JSON verdict writer.
- `experiments/results/audit/sharpe_audit.json` — Verdict PASS, per_trade_drift=0.00016, per_pair_corr_corrected=0.296, full 4-item assumption stack.
- `tests/audit/test_audit_sharpe.py` (55 lines) — Fixture test proving correlation correction collapses Sharpe at avg_corr=1.
- `.planning/phases/18-system-audit-adversarial-verification/deferred-items.md` — Logs pre-existing `tests/data/test_aligner.py` failure as out-of-scope per scope-boundary rule.

## Decisions Made

- **Verdict thresholds taken verbatim from plan spec.** `avg_corr > 0.10 AND corrected < 0.5 * naive` triggers CORRECTED. The empirical `avg_corr = 0.042` is below 0.10, so the verdict is PASS even though `corrected (0.296) < 0.5 * naive (0.391)`. The plan author chose these thresholds knowing both must fire — the test for "is Sharpe inflated" requires both meaningful pair correlation AND large fractional reduction. PASS here means "raw recompute reproduces canonical, AND pair correlation is small enough that the i.i.d. assumption is approximately defensible".
- **Annualization derived from ledger, not hard-coded.** The audit measures `test_span_days` empirically from `entry_ts.max() - entry_ts.min()` and computes `pairs_per_year = n_pairs * (365 / test_span_days)`. This is the correct primitive; the paper's implicit annualization is what Plan 18-06 (Tier 5 number trace) is supposed to reconcile.
- **Bootstrap is plain percentile, not stationary-bootstrap.** Per plan spec: 10,000 resamples with replacement, percentile method. The assumption stack explicitly flags this and points to Politis-Romano stationary bootstrap as the AR(1)-correction upgrade if a follow-up audit needs it. Out of scope for AUDIT-01.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Timestamp normalization for int64 epoch-seconds parquet**

- **Found during:** Task 1 (audit_sharpe.py first execution)
- **Issue:** RESEARCH.md skeleton (lines 240-478, copied verbatim per plan instructions) contained `test_df["timestamp"].astype("int64").values // 10**9`. This works only if `timestamp` is a pandas `datetime64[ns]` (nanoseconds-since-epoch). The canonical `data/processed/*.parquet` actually stores `timestamp` as `int64` epoch SECONDS. Dividing already-epoch-seconds by 1e9 collapsed every entry to 1.7 (essentially zero), making `entry_day` constant across the whole test set. Visible-in-output symptoms: `test_span_days=0.0`, `pairs_per_year=52560` (default-floor extrapolation), `avg_pairwise_corr=0.0` (only one day, no cross-day variation), `n_pairs_compared=0`.
- **Fix:** Replaced the single-line conversion with a dtype-aware block: if `timestamp` is `datetime64`, divide by 1e9; otherwise treat as int64 and apply a heuristic (`max(ts) > 1e12` ⇒ nanoseconds, else epoch-seconds). Real epoch-seconds in 2024–2026 are ~1.7e9; nanoseconds are ~1.7e18. Heuristic is robust against both representations and falls through cleanly for either.
- **Files modified:** `experiments/audit/audit_sharpe.py` (lines 73–88, replacing the ~5-line block from RESEARCH.md)
- **Verification:** Re-run produced sane numbers: `test_span_days=92.67`, `pairs_per_year=567`, `avg_pairwise_corr=0.042`, `n_pairs_compared=461`. The recomputed per-trade Sharpe (0.5007) still matches canonical (0.5009) within 0.0002 — confirming the fix is local to ledger-time-axis construction and does not touch the per-trade Sharpe pathway.
- **Committed in:** `0b4dff3` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug)
**Impact on plan:** The fix was correctness-required — without it, every downstream cross-sectional metric (`avg_pair_corr`, `n_eff`, `per_pair_sharpe_corr_corrected`, annualization) would have been silently wrong. The per-trade Sharpe (the primary "kill-or-confirm" check) was unaffected, so the verdict logic still correctly defends `PASS`. No scope creep.

## Issues Encountered

- **Pre-existing test failure in `tests/data/test_aligner.py`** discovered during the verification regression sweep. Phase 14 provenance, completely outside Phase 18 audit scope. Logged to `deferred-items.md`. Not a blocker for AUDIT-01.

## User Setup Required

None - no external service configuration required. The audit script is fully reproducible from `data/processed/` + `experiments/results/canonical/headline.json` with `seed=42`.

## Self-Check: PASSED

- `experiments/audit/audit_sharpe.py` exists ✓ (268 lines, ≥180 required)
- `experiments/results/audit/sharpe_audit.json` exists with all required keys ✓
- `tests/audit/test_audit_sharpe.py` exists and passes ✓
- Commit `0b4dff3` (Task 1) present in `git log` ✓
- Commit `c1214db` (Task 2) present in `git log` ✓
- Self-check verifications run before drafting this section.

## Next Phase Readiness

- **Plan 18-03 (Tier 2 leakage audit) unblocked.** It runs in parallel with this plan and consumes the same `tests/audit/conftest.py` fixtures from Plan 18-01.
- **Plan 18-07 (AUDIT_REPORT.md generator) has its first input JSON** to ingest at `experiments/results/audit/sharpe_audit.json`. The schema is the one specified in 18-PLAN.md `<interfaces>` Pattern 2.
- **Open follow-up for Plan 18-06 (Tier 5 paper number trace):** the audit's annualized per-pair Sharpe (18.6 naive / 7.0 corrected) does NOT match the paper's headline ≈3.2. Plan 18-06 must trace which annualization factor produced 3.2 and decide whether the audit's empirical factor (23.8) or the paper's implicit factor is the canonical one. This is the natural trigger point for any abstract / §5 / Table 8 footnote correction.

---
*Phase: 18-system-audit-adversarial-verification*
*Completed: 2026-04-25*
