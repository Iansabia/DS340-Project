---
phase: 15-live-commodity-matching-engineering-fixes
plan: 02
subsystem: matching
tags: [quality-filter, asset-class, regression-test, tdd, commodity, crypto, COM-02]

# Dependency graph
requires:
  - phase: 15-live-commodity-matching-engineering-fixes
    provides: Plan 15-01 diagnostic context (discovery gap vs matching bug split; this plan handles the matching side)
provides:
  - Rule 10 (asset-class consistency) added to src/matching/quality_filter.py::filter_active_match
  - Regression test pinning the KXWTIMAX-26DEC31-T130 vs Bitcoin-$130K false match
  - Token vocabularies (COMMODITY_ASSET_TOKENS, CRYPTO_ASSET_TOKENS, KALSHI_COMMODITY_PREFIXES, KALSHI_CRYPTO_PREFIXES) for future extension
  - _detect_asset_class() helper — reusable asset-class inference from ticker + title
affects: [live-trading, active-matches-pipeline, collector, discovery]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "TDD (RED-then-GREEN) enforced as two separate atomic git commits"
    - "Asymmetric confidence rule: rejection requires BOTH sides to produce confident signals; ambiguous evidence passes through to other rules"
    - "Ticker-prefix authoritative over title tokens when both produce a signal"

key-files:
  created:
    - tests/matching/test_rule_10_asset_class.py
  modified:
    - src/matching/quality_filter.py

key-decisions:
  - "Rule 10 only fires when k_class AND p_class are both non-None and differ — ambiguous asset-class cases fall through to existing rules (avoids false positives on non-financial markets)"
  - "Kalshi ticker prefix wins over title tokens because tickers are authoritative and titles can contain stray numeric strings"
  - "Token vocabularies exposed as module-level constants so future plans can extend without touching filter_active_match internals"
  - "Canonical hex id 0x885a6abefad122348b4fbd503473d7fd1f9035d0438cf988a7591620f316a859 pinned verbatim in the test file — survives as evidence the specific real-world bug is regression-covered"

patterns-established:
  - "Asset-class taxonomy (commodity/crypto) — extensible pattern for future forex, metals, equities categories"
  - "Four-test coverage shape for asymmetric rules: reject direction A→B, reject direction B→A, accept symmetric A→A, accept symmetric B→B"

requirements-completed: [COM-02]

# Metrics
duration: 12min
completed: 2026-04-23
---

# Phase 15 Plan 02: Rule 10 Asset-Class Mismatch Regression Summary

**Rule 10 rejects numerically-coincident cross-asset strikes (commodity vs crypto) in `filter_active_match`, closing the KXWTIMAX-26DEC31-T130 vs Bitcoin-$130K false-match bug caught in live paper trading.**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-04-24T (session start)
- **Completed:** 2026-04-24T (both commits landed)
- **Tasks:** 2 (RED + GREEN)
- **Files modified:** 2 (1 new test, 1 quality_filter.py)

## Accomplishments

- Four-test regression coverage committed in a standalone RED commit that FAILED on pre-fix code
- Rule 10 added to `filter_active_match` with module-level token vocabularies and a reusable `_detect_asset_class` helper
- Canonical false match `(KXWTIMAX-26DEC31-T130, 0x885a6abe...a859)` now returns `(False, "asset_class_mismatch (kalshi=commodity, poly=crypto)")` — verified by automated test
- Full matching suite remains green (104 passed) with no regression on existing rules

## Task Commits

Each task was committed atomically per the TDD discipline:

1. **Task 1 (RED): Failing regression test** — `6297ac8` (`test(matching)`)
   - Created `tests/matching/test_rule_10_asset_class.py` with 4 tests
   - 2 rejection tests failed on pre-fix code (no Rule 10 existed)
   - 2 symmetric-accept tests passed on pre-fix code (confirming base behavior)
2. **Task 2 (GREEN): Rule 10 fix** — `071b7db` (`feat(matching)`)
   - Added `COMMODITY_ASSET_TOKENS`, `CRYPTO_ASSET_TOKENS`, `KALSHI_COMMODITY_PREFIXES`, `KALSHI_CRYPTO_PREFIXES` module-level tuples
   - Added `_detect_asset_class(ticker_u, title)` helper with ticker-prefix-wins-over-title semantics
   - Added Rule 10 block immediately before the final `return True, None` in `filter_active_match`
   - All 4 regression tests PASS after fix

**Plan metadata commit:** pending (this SUMMARY.md + STATE.md + ROADMAP.md update)

## Files Created/Modified

- `tests/matching/test_rule_10_asset_class.py` (NEW, 106 lines) — 4 regression tests: rejection forward, rejection inverse, symmetric-commodity accept, symmetric-crypto accept
- `src/matching/quality_filter.py` (MODIFIED, +67 lines) — token vocabularies after line 22; `_detect_asset_class` helper above `filter_active_match`; Rule 10 block inside `filter_active_match` before final `return True, None`

## Token Vocabularies (Exact Values Committed)

```python
COMMODITY_ASSET_TOKENS = (
    "WTI", "CRUDE", "OIL", "DIESEL", "GAS", "GASOLINE",
    "HEATINGOIL", "HEATING OIL", "BRENT",
)
CRYPTO_ASSET_TOKENS = (
    "BITCOIN", "BTC", "ETHEREUM", "ETH", "DOGECOIN", "DOGE",
    "SOLANA", "SOL", "CARDANO", "ADA", "XRP", "LITECOIN", "LTC",
)
KALSHI_COMMODITY_PREFIXES = (
    "KXWTI", "KXWTID", "KXWTIW", "KXWTIMAX",
    "KXAAAGASD", "KXAAAGASW", "KXAAAGASM",
    "KXBRENTMON", "KXDIESEL", "KXHEATINGOIL", "KXCRUDE", "KXGASOLINE",
)
KALSHI_CRYPTO_PREFIXES = (
    "KXBTC", "KXETH", "KXBITCOIN", "KXETHEREUM",
    "KXDOGE", "KXSOL", "KXADA", "KXXRP", "KXLTC",
)
```

## Decisions Made

- **Asymmetric confidence semantics:** Rule 10 fires only when BOTH sides produce a confident but mismatched asset-class signal. This was deliberate — ambiguous cases (neither side has an asset-class token) must fall through to existing rules so non-financial markets aren't disturbed.
- **Ticker prefix wins over title tokens:** Kalshi tickers are authoritative; titles occasionally embed stray numeric strings or analogies. The helper checks prefixes first and only inspects tokens if prefix lookup returns nothing.
- **Polymarket has no ticker prefix:** `_detect_asset_class("", p_title)` is called with empty ticker for the Polymarket side — the function correctly skips prefix checks and inspects the title alone.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

- **Pre-existing environment issue in matching test suite:** Running `pytest tests/matching/ -q` triggered `ModuleNotFoundError: sentence_transformers` in collection for `test_pipeline.py`, `test_scorer.py`, `test_semantic_matcher.py`. This is **not caused by our changes** — confirmed by running `python -c "import sentence_transformers"` (same error) and by the fact that those files never import from `quality_filter`. Per the scope-boundary rule, this is out-of-scope for 15-02 and logged here as a pre-existing deferred item. The 104 tests that DO collect all pass (including our new 4).

## User Setup Required

None.

## Verification Evidence

```
$ pytest tests/matching/test_rule_10_asset_class.py -q
============================= 4 passed in 0.01s ==============================

$ pytest tests/matching/ -q --ignore=tests/matching/test_pipeline.py \
  --ignore=tests/matching/test_scorer.py \
  --ignore=tests/matching/test_semantic_matcher.py
============================= 104 passed in 0.12s ==============================

$ git log --oneline --grep="15-02"
071b7db feat(matching): add Rule 10 asset-class mismatch (COM-02, 15-02 GREEN)
6297ac8 test(matching): add failing Rule 10 asset-class mismatch regression (15-02 RED)
```

## Next Phase Readiness

- COM-02 closed — regression test + fix committed as separate atomic commits
- Plan 15-03 can now apply the discovery fix (COM-03, COM-04, COM-05) knowing the matching layer is protected against the specific Bitcoin-vs-oil false-match class
- Future extension: additional asset classes (forex, metals, equities) can be added to the token vocabulary without touching `filter_active_match`

## Self-Check: PASSED

- `tests/matching/test_rule_10_asset_class.py` — FOUND
- `src/matching/quality_filter.py` Rule 10 block — FOUND (grep for `Rule 10` returns 2)
- Commit `6297ac8` (RED) — FOUND via `git log --oneline --grep="15-02"`
- Commit `071b7db` (GREEN) — FOUND via `git log --oneline --grep="15-02"`
- All 9 acceptance-criteria grep checks passed with expected counts

---
*Phase: 15-live-commodity-matching-engineering-fixes*
*Plan: 02*
*Completed: 2026-04-23*
