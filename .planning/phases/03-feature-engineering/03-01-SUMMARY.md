---
phase: 03-feature-engineering
plan: 01
subsystem: features
tags: [pandas, parquet, feature-engineering, time-series-alignment, microstructure]

# Dependency graph
requires:
  - phase: 01-data-ingestion
    provides: Raw Kalshi and Polymarket parquet candlestick files in data/raw/
  - phase: 02-market-matching
    provides: accepted_pairs.json with matched pair metadata
provides:
  - Feature matrix parquet with hourly aligned spread and microstructure features
  - Build report JSON documenting filtering decisions for all pairs
  - Reusable alignment and feature computation modules
affects: [04-evaluation-framework, 05-baseline-models, 06-time-series-models]

# Tech tracking
tech-stack:
  added: []
  patterns: [bid-ask-midpoint-fallback, string-none-to-nan-conversion, hourly-floor-alignment, forward-fill-with-gap-limit]

key-files:
  created:
    - src/features/__init__.py
    - src/features/schemas.py
    - src/features/pair_loader.py
    - src/features/alignment.py
    - src/features/engineering.py
    - src/features/build_features.py
    - tests/features/__init__.py
    - tests/features/conftest.py
    - tests/features/test_pair_loader.py
    - tests/features/test_alignment.py
    - tests/features/test_engineering.py
  modified: []

key-decisions:
  - "String None to NaN conversion required for Kalshi parquet files (object dtype stores Python None, not NaN)"
  - "Bid-ask midpoint fallback critical but only 1/77 paired Kalshi markets have actual trade data"
  - "Forward-fill limited to 6-hour gap maximum to prevent stale price propagation"
  - "76/77 pairs excluded at alignment stage due to Kalshi having zero usable price data (no trades, no bid/ask)"

patterns-established:
  - "Alignment pattern: floor timestamps to hour, dedup by last-per-hour, inner-merge, ffill with gap limit, drop nulls"
  - "Feature schema contract: FEATURE_COLUMNS list in schemas.py defines output parquet column order"
  - "Build report pattern: JSON file alongside parquet documenting pipeline filtering decisions"

requirements-completed: [FEAT-01, FEAT-03, FEAT-05]

# Metrics
duration: 5min
completed: 2026-04-04
---

# Phase 3 Plan 1: Feature Engineering Pipeline Summary

**Hourly-aligned feature matrix with spread, bid-ask spread, and price velocity from matched Kalshi-Polymarket pairs (3,227 rows, 1 viable pair)**

## Performance

- **Duration:** 5 min
- **Started:** 2026-04-04T00:37:41Z
- **Completed:** 2026-04-04T00:42:55Z
- **Tasks:** 2
- **Files modified:** 11

## Accomplishments
- Built complete feature engineering pipeline: pair loading, hourly alignment, feature computation, liquidity filtering, and parquet output
- Handles real data edge case where 76/77 Kalshi paired markets have zero trade data and zero bid/ask data
- Produces feature_matrix.parquet with 3,227 hourly feature rows for 1 viable pair (kxlargecut25-0x735a2a98)
- Build report documents exclusion reasons for all 76 filtered pairs

## Task Commits

Each task was committed atomically:

1. **Task 1: Feature schemas, pair loader, and test infrastructure** - `e5d9cece` (feat)
2. **Task 2: Hourly alignment, feature computation, liquidity filter, and build CLI** - `40b913b9` (feat)

_Both tasks followed TDD RED-GREEN: 24 total tests._

## Files Created/Modified
- `src/features/schemas.py` - FEATURE_COLUMNS (10 cols) and MIN_HOURS_THRESHOLD=10
- `src/features/pair_loader.py` - load_valid_pairs: filters to pairs with parquet on both platforms
- `src/features/alignment.py` - align_pair_hourly: timestamp merge, bid-ask midpoint fallback, ffill
- `src/features/engineering.py` - compute_features + filter_low_liquidity
- `src/features/build_features.py` - CLI entry point running full pipeline
- `tests/features/conftest.py` - Reusable fixtures: sample_kalshi_df, sample_polymarket_df, sample_kalshi_all_null_df
- `tests/features/test_pair_loader.py` - 6 tests for schemas and pair loading
- `tests/features/test_alignment.py` - 6 tests for hourly alignment including bid-ask fallback
- `tests/features/test_engineering.py` - 12 tests for features, filtering, and build pipeline integration

## Decisions Made
- **String None to NaN conversion**: Kalshi parquet files store None as Python None in object-typed columns. Used `pd.to_numeric(errors='coerce')` to handle this cleanly.
- **Forward-fill gap limit of 6 hours**: Prevents stale prices from propagating across long gaps (e.g., overnight), which would create artificial spread stability.
- **All 77 pairs have parquet files on both platforms** (plan estimated only 12). However, 76/77 Kalshi markets have ALL null data (no trades, no bid/ask), so only 1 pair produces usable features.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Kalshi parquet stores None as Python None in object columns, not NaN**
- **Found during:** Task 2 (alignment implementation)
- **Issue:** Plan assumed Kalshi null values would be NaN/null. Real data uses Python None in object-dtype columns, requiring explicit type conversion.
- **Fix:** Added `_to_numeric_or_nan()` helper using `pd.to_numeric(errors='coerce')` in alignment.py
- **Files modified:** src/features/alignment.py
- **Verification:** test_bid_ask_midpoint_fallback_for_null_ohlc passes; pipeline runs on real data
- **Committed in:** 40b913b9

**2. [Rule 1 - Bug] Plan's pair count estimate was wrong (12 vs actual 77 with both parquets)**
- **Found during:** Task 2 (pipeline run on real data)
- **Issue:** Plan stated only 12 pairs have parquet on both platforms. All 77 do, but 76 have completely null Kalshi data.
- **Fix:** No code change needed -- pipeline handles this correctly via alignment null-dropping. Build report documents the reality.
- **Verification:** Pipeline produces 1 pair with 3,227 rows; 76 excluded in build_report.json

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both necessary for correctness on real data. No scope creep.

## Issues Encountered
- The single viable pair (kxlargecut25-0x735a2a98) is the only Kalshi market with actual trade data among matched pairs. This severely limits the feature matrix for downstream modeling. Phase 03-02 (data augmentation/enrichment) may need to address this.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Feature matrix at `data/processed/feature_matrix.parquet` is ready for model consumption
- Only 1 pair with 3,227 hourly rows -- downstream phases need to account for limited data
- Build report at `data/processed/build_report.json` documents the filtering rationale
- Phase 03-02 should consider whether Polymarket-only data or alternative proxy strategies can expand the dataset

## Self-Check: PASSED

All 13 created files verified present. Both task commits (e5d9cece, 40b913b9) verified in git log. 24 tests passing.

---
*Phase: 03-feature-engineering*
*Completed: 2026-04-04*
