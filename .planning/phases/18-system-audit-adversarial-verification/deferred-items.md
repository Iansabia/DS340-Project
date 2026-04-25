# Phase 18 Deferred Items

Out-of-scope discoveries logged during Phase 18 plan execution. Per the GSD
scope-boundary rule, these are NOT auto-fixed during the audit phase — they
predate the work in this phase and live outside the audit subsystem.

## From Plan 18-02 (Tier 1 Sharpe audit)

### Pre-existing test failure: `tests/data/test_aligner.py::TestAlignAllPairs::test_excludes_pair_with_insufficient_trades`

- **Discovered:** 2026-04-25 during full `pytest tests/ --ignore=tests/audit` regression check at end of Plan 18-02.
- **Failure:** `assert report["exclusion_reasons"]["insufficient_trades"] == 1` got `0`.
- **Last touched:** `src/data/aligner.py` and `tests/data/test_aligner.py` were last modified in Phase 14 (commit `c5e7158`, "docs(14-03): qualify abstract oil claim..."). Plan 18-02 made zero changes to `src/data/` or `tests/data/`.
- **Disposition:** OUT OF SCOPE for Phase 18. Phase 18 is an adversarial audit of the headline numerics; the data-aligner subsystem is unrelated to Tier 1 Sharpe verification. The matched-pairs dataset feeding the audit is the canonical `data/processed/` parquet, which was generated under the older (working) aligner state — the regression in this test suggests an exclusion-reason classification change, but the produced data is what `experiments/run_canonical.py` audited.
- **Recommendation:** Triage in a future maintenance phase (or as part of Phase 18 Plan 18-04 if cost realism work touches the aligner). Not a blocker for AUDIT-01 sign-off.

## From Plan 18-05 (Tier 4 survivorship audit)

### Re-confirmed pre-existing aligner test failure

- **Discovered:** 2026-04-25 during regression sweep at end of Plan 18-05.
- **Same failure as Plan 18-02 entry above** — `tests/data/test_aligner.py::TestAlignAllPairs::test_excludes_pair_with_insufficient_trades` still asserts `0 == 1`.
- **Verified pre-existing:** Reproduced at HEAD without Plan 18-05 changes. Plan 18-05 touched only `experiments/audit/` and `tests/audit/`; zero `src/data/` modifications.
- **Disposition:** OUT OF SCOPE. Already logged from Plan 18-02. No new action.
