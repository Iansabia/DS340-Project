---
phase: 01-data-ingestion
plan: 02
subsystem: data
tags: [kalshi, api, candlestick, historical-cutoff, parquet, ingestion]

# Dependency graph
requires:
  - phase: 01-data-ingestion/01-01
    provides: MarketDataAdapter ABC, ResilientClient, schemas, conftest fixtures
provides:
  - KalshiAdapter implementing MarketDataAdapter with historical/live endpoint routing
  - Kalshi CLI ingestion script with metadata export
  - 16 unit tests covering cutoff, routing, parsing, caching
affects: [01-data-ingestion/01-03, 02-market-matching]

# Tech tracking
tech-stack:
  added: []
  patterns: [historical-cutoff-routing, time-window-chunking, bid-ask-midpoint-fallback]

key-files:
  created:
    - src/data/kalshi.py
    - src/data/ingest_kalshi.py
    - tests/data/test_kalshi.py
  modified: []

key-decisions:
  - "Rate limit set to 18 req/s (buffer below Kalshi's 20 req/s Basic tier)"
  - "Lexicographic ISO8601 comparison for cutoff routing (avoids datetime parsing overhead)"
  - "numpy bool comparison uses == instead of is in tests (pandas DataFrame values are numpy types)"

patterns-established:
  - "Cutoff routing: call /historical/cutoff once, cache, route per-market based on close_time"
  - "Time-window chunking: MAX_CANDLES_PER_REQUEST * PERIOD_INTERVAL * 60 seconds per chunk"
  - "Null OHLC fallback: bid-ask midpoint when trade price is null, None when both are absent"
  - "Metadata export: _metadata.json in output dir for downstream matching pipeline"

requirements-completed: [DATA-01, DATA-04, DATA-05, DATA-06]

# Metrics
duration: 27min
completed: 2026-04-02
---

# Phase 1 Plan 2: Kalshi Adapter Summary

**KalshiAdapter with dynamic /historical/cutoff routing, minute-level candlestick parsing with null OHLC bid-ask midpoint fallback, and CLI ingestion script**

## Performance

- **Duration:** 27 min
- **Started:** 2026-04-02T18:06:00Z
- **Completed:** 2026-04-02T18:32:33Z
- **Tasks:** 2
- **Files created:** 3

## Accomplishments
- KalshiAdapter implementing MarketDataAdapter with dynamic historical cutoff routing via /historical/cutoff (cached, not hardcoded)
- Market discovery via /series -> /events pagination chain across Economics, Crypto, and Financials categories
- Minute-level candlestick parsing with null OHLC handling (bid-ask midpoint fallback) and time-window chunking (4500 candles/request)
- CLI ingestion script that discovers markets, exports metadata JSON, and fetches candlesticks idempotently

## Task Commits

Each task was committed atomically:

1. **Task 1: KalshiAdapter implementation with TDD** - `408c49b6` (feat)
2. **Task 2: Kalshi CLI ingestion script** - `fd33cdab` (feat)

_Note: Task 1 followed TDD (RED-GREEN-REFACTOR). Test file written first, confirmed import failure, then implementation written to pass all 16 tests._

## Files Created/Modified
- `src/data/kalshi.py` - KalshiAdapter with cutoff routing, market discovery, candlestick parsing (339 lines)
- `src/data/ingest_kalshi.py` - CLI entry point with --categories and --output-dir args, metadata export (84 lines)
- `tests/data/test_kalshi.py` - 16 unit tests across 5 test classes (439 lines)

## Decisions Made
- Rate limit set to 18 req/s to leave buffer below Kalshi's 20 req/s Basic tier limit
- Used lexicographic ISO8601 comparison for cutoff routing rather than parsing to datetime objects (simpler, faster, ISO8601 sorts correctly as strings)
- Tests use `==` instead of `is` for boolean comparison with DataFrame values (pandas returns numpy bool types where `np.True_ is True` evaluates to False)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed numpy bool comparison in tests**
- **Found during:** Task 1 (GREEN phase)
- **Issue:** Tests used `is True`/`is False` for has_trades assertions, but pandas DataFrame values are numpy bools where `np.True_ is True` is False
- **Fix:** Changed to `== True`/`== False` with noqa comments
- **Files modified:** tests/data/test_kalshi.py
- **Verification:** All 16 tests pass
- **Committed in:** 408c49b6 (part of Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug fix)
**Impact on plan:** Minor test assertion fix. No scope creep.

## Issues Encountered
None - implementation followed plan specification cleanly.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- KalshiAdapter is complete and tested, ready for use in Phase 2 (Market Matching)
- CLI script can be run with `.venv/bin/python -m src.data.ingest_kalshi` when ready to fetch real data
- Plan 01-03 (Polymarket adapter) can proceed independently; it follows the same MarketDataAdapter interface

## Self-Check: PASSED

All files exist, all commits verified.

---
*Phase: 01-data-ingestion*
*Completed: 2026-04-02*
