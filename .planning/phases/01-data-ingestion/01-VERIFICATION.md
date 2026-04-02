---
phase: 01-data-ingestion
verified: 2026-04-01T00:00:00Z
status: passed
score: 15/15 must-haves verified
gaps: []
human_verification:
  - test: "Run .venv/bin/python -m src.data.ingest_kalshi against live Kalshi API"
    expected: "Parquet files appear in data/raw/kalshi/, metadata JSON written, no errors on rate-limit or retry paths"
    why_human: "End-to-end live API call cannot be verified programmatically without making real network requests and incurring actual rate-limiting"
  - test: "Run .venv/bin/python -m src.data.ingest_polymarket against live Polymarket APIs"
    expected: "Parquet files appear in data/raw/polymarket/, metadata JSON with clobTokenIds written, CLOB-first fallback path exercised"
    why_human: "Three-API Polymarket integration requires live calls to confirm Gamma keyword matching returns real results and CLOB/Data fallback triggers correctly"
---

# Phase 1: Data Ingestion Verification Report

**Phase Goal:** Raw historical market data from both platforms is reliably available on disk for downstream processing
**Verified:** 2026-04-01
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | pytest runs and discovers tests in tests/data/ | VERIFIED | 45 tests collected and pass in 0.86s |
| 2 | pyarrow is installed and can read/write parquet | VERIFIED | pyarrow 23.0.1 in requirements.txt; parquet write/read exercised in caching tests |
| 3 | ResilientClient retries on HTTP 429/500/502/503/504 with exponential backoff | VERIFIED | `status_forcelist=[429, 500, 502, 503, 504]` with urllib3 Retry; test_retries_on_500 and test_raises_on_404 pass |
| 4 | ResilientClient enforces rate limiting between requests | VERIFIED | `time.sleep(sleep_time)` enforcing `min_interval`; test_enforces_minimum_interval passes |
| 5 | MarketDataAdapter ABC defines the contract for platform adapters | VERIFIED | `class MarketDataAdapter(ABC)` with abstract `list_markets` and `get_candlesticks` plus concrete `get_or_fetch_candlesticks` and `ingest_all` |
| 6 | Parquet schema is defined with typed columns for candlestick data and market metadata | VERIFIED | `CANDLESTICK_COLUMNS`, `OPTIONAL_COLUMNS`, `METADATA_COLUMNS`, `MarketMetadata` dataclass in schemas.py |
| 7 | KalshiAdapter discovers all resolved markets in Economics, Crypto, and Financials via /series endpoint | VERIFIED | `list_markets` chains /series -> /events with cursor pagination; test_discovers_markets_via_series_events, test_handles_cursor_pagination pass |
| 8 | KalshiAdapter fetches minute-level candlesticks with automatic time-window chunking (max 4500 per request) | VERIFIED | `chunk_seconds = MAX_CANDLES_PER_REQUEST * PERIOD_INTERVAL * 60`; test_time_window_chunking passes (3+ chunks for 600000s span) |
| 9 | KalshiAdapter handles null OHLC fields by computing bid-ask midpoint as fallback price | VERIFIED | `close = (float(bid_close) + float(ask_close)) / 2`; test_null_ohlc_uses_bid_ask_midpoint passes |
| 10 | KalshiAdapter calls /historical/cutoff at start of ingestion and routes each market to historical or live endpoint | VERIFIED | `_get_historical_cutoff()` called first in `ingest_all`; test_ingest_all_calls_cutoff_first and routing tests pass |
| 11 | Running the Kalshi ingestion script produces parquet files and skips cached markets | VERIFIED | `ingest_all` with `get_or_fetch_candlesticks` checks cache before fetching; test_cache_miss_writes_parquet and test_cache_hit_skips_fetch pass |
| 12 | PolymarketAdapter discovers resolved markets via Gamma API /events with keyword-based filtering | VERIFIED | CRYPTO_KEYWORDS and FINANCE_KEYWORDS match on `f"{title} {description}"`; 7 list_markets tests pass |
| 13 | PolymarketAdapter extracts clobTokenIds from Gamma market metadata | VERIFIED | `json.loads(raw_token_ids)` parsing stringified JSON; test_extracts_clob_token_ids passes |
| 14 | PolymarketAdapter fetches price history from CLOB /prices-history using startTs/endTs chunking with Data API fallback | VERIFIED | `_fetch_clob_prices` with 14-day chunks; `_fetch_trades` fallback at MAX_TRADE_OFFSET=3000; test_falls_back_to_trades and test_trades_offset_limit pass |
| 15 | Running the Polymarket ingestion script produces parquet files and skips already-cached markets | VERIFIED | test_cache_miss_writes_parquet and test_cache_hit_skips_fetch pass; CLI script imports cleanly |

**Score:** 15/15 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/data/base.py` | MarketDataAdapter ABC with list_markets and get_candlesticks methods | VERIFIED | 105 lines; `class MarketDataAdapter(ABC)` with 2 abstract methods + 2 concrete methods; imports `validate_candlestick_df` from schemas |
| `src/data/client.py` | ResilientClient with retry, rate limiting, logging | VERIFIED | 63 lines; `class ResilientClient`; `status_forcelist=[429, 500, 502, 503, 504]`; `time.sleep(sleep_time)` for rate limiting |
| `src/data/schemas.py` | MarketMetadata dataclass, candlestick column definitions | VERIFIED | 59 lines; `class MarketMetadata` (7 fields); `CANDLESTICK_COLUMNS` (6 cols); `validate_candlestick_df` function |
| `tests/data/conftest.py` | Shared test fixtures for mock API responses | VERIFIED | Contains `mock_kalshi_candlestick_response`, `mock_polymarket_clob_prices_response`, `mock_polymarket_trades_response`, and 6 additional fixtures |
| `src/data/kalshi.py` | KalshiAdapter implementing MarketDataAdapter | VERIFIED | 339 lines; `class KalshiAdapter(MarketDataAdapter)`; `_get_historical_cutoff`, `_is_historical`, `list_markets`, `get_candlesticks`, `_parse_candlesticks`, `ingest_all` |
| `src/data/ingest_kalshi.py` | CLI entry point for Kalshi ingestion | VERIFIED | 84 lines; `def main()` with argparse --categories and --output-dir; `_metadata.json` export; `adapter.ingest_all` |
| `tests/data/test_kalshi.py` | Unit tests for Kalshi adapter (min 80 lines) | VERIFIED | 439 lines; 16 tests across 5 classes covering cutoff, routing, discovery, parsing, caching |
| `src/data/polymarket.py` | PolymarketAdapter implementing MarketDataAdapter | VERIFIED | 350 lines; `class PolymarketAdapter(MarketDataAdapter)`; GAMMA_BASE, CLOB_BASE, DATA_BASE; `prices-history`; `startTs`/`endTs`; MAX_TRADE_OFFSET=3000; `_trades_to_ohlcv` |
| `src/data/ingest_polymarket.py` | CLI entry point for Polymarket ingestion | VERIFIED | 80 lines; `def main()` with argparse; `_metadata.json` with `clob_token_ids`; `adapter.ingest_all` |
| `tests/data/test_polymarket.py` | Unit tests for Polymarket adapter (min 100 lines) | VERIFIED | 370 lines; 18 tests across 5 classes covering discovery, CLOB fetch, dual-token, trade fallback, offset limit, OHLCV aggregation, caching |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/data/client.py` | `urllib3.util.retry.Retry` | HTTPAdapter with retry strategy | VERIFIED | `Retry(total=max_retries, backoff_factor=backoff_factor, status_forcelist=[429, 500, 502, 503, 504], ...)` at lines 32-37 |
| `src/data/base.py` | `src/data/schemas.py` | import of MarketMetadata and schema constants | VERIFIED | `from src.data.schemas import MarketMetadata, validate_candlestick_df` at line 6 |
| `src/data/kalshi.py` | `src/data/base.py` | inherits MarketDataAdapter | VERIFIED | `class KalshiAdapter(MarketDataAdapter)` at line 21 |
| `src/data/kalshi.py` | `src/data/client.py` | uses ResilientClient for all HTTP | VERIFIED | `from src.data.client import ResilientClient`; used in `__init__` and all API calls via `self.client` |
| `src/data/kalshi.py` | `src/data/schemas.py` | returns MarketMetadata and validated DataFrames | VERIFIED | `from src.data.schemas import MarketMetadata`; `MarketMetadata(...)` constructed in `list_markets` |
| `src/data/kalshi.py` | `https://api.elections.kalshi.com/trade-api/v2/historical/cutoff` | `_get_historical_cutoff` method called at start of `ingest_all` | VERIFIED | `self.client.get("historical/cutoff")` in `_get_historical_cutoff`; method called first in `ingest_all` |
| `src/data/polymarket.py` | `src/data/base.py` | inherits MarketDataAdapter | VERIFIED | `class PolymarketAdapter(MarketDataAdapter)` at line 21 |
| `src/data/polymarket.py` | `src/data/client.py` | uses ResilientClient for all three APIs | VERIFIED | Three separate `ResilientClient` instances in `__init__` for gamma, clob, data clients |
| `src/data/polymarket.py` | `src/data/schemas.py` | returns MarketMetadata and validated DataFrames | VERIFIED | `from src.data.schemas import MarketMetadata, CANDLESTICK_COLUMNS`; `MarketMetadata(...)` in `list_markets` |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| DATA-01 | 01-02-PLAN | Kalshi API adapter handles live/historical endpoint split via /historical/cutoff | SATISFIED | `_get_historical_cutoff` + `_is_historical` dynamically route per-market; test_routes_pre_cutoff and test_routes_post_cutoff pass |
| DATA-02 | 01-03-PLAN | Polymarket adapter queries Gamma for metadata, CLOB for prices, Data API for trade records | SATISFIED | Three separate ResilientClient instances; `list_markets` uses Gamma; `_fetch_clob_prices` uses CLOB; `_fetch_trades` uses Data API |
| DATA-03 | 01-03-PLAN | Polymarket price history reconstruction from trade records for resolved markets | SATISFIED | `_trades_to_ohlcv` aggregates raw trades into hourly OHLCV bars; test_falls_back_to_trades and test_aggregates_trades_to_hourly pass |
| DATA-04 | 01-01-PLAN, 01-02-PLAN, 01-03-PLAN | Rate limiting and local caching for both platform APIs | SATISFIED | `ResilientClient.min_interval` enforces rate limiting; `get_or_fetch_candlesticks` provides file-based parquet caching; all cache tests pass |
| DATA-05 | 01-01-PLAN, 01-02-PLAN, 01-03-PLAN | Automated retry with exponential backoff for transient API failures | SATISFIED | urllib3 `Retry` with `backoff_factor` on `status_forcelist=[429,500,502,503,504]`; test_retries_on_500 passes |
| DATA-06 | 01-01-PLAN, 01-02-PLAN, 01-03-PLAN | Raw data storage with timestamps in data/raw/ | SATISFIED | `data/raw/kalshi/` and `data/raw/polymarket/` directories exist; parquet files include `timestamp` column; validated by `validate_candlestick_df` |

All 6 phase requirements (DATA-01 through DATA-06) are satisfied. No orphaned requirements detected.

---

### Anti-Patterns Found

No anti-patterns found.

Scanned `src/data/client.py`, `src/data/base.py`, `src/data/schemas.py`, `src/data/kalshi.py`, `src/data/polymarket.py`, `src/data/ingest_kalshi.py`, `src/data/ingest_polymarket.py` for:
- TODO/FIXME/HACK/PLACEHOLDER comments: none found
- Stub returns (return null, return {}, return []): one `return []` at polymarket.py:98 is a legitimate guard clause for empty keyword map, not a stub
- Empty handlers: none found
- Hardcoded cutoff dates: none found (DATA-01 explicitly verified dynamic)

---

### Human Verification Required

#### 1. Kalshi live ingestion end-to-end

**Test:** Run `.venv/bin/python -m src.data.ingest_kalshi --categories Crypto --output-dir /tmp/kalshi_test` against the live API
**Expected:** At least one .parquet file in /tmp/kalshi_test/, a _metadata.json file, and log lines confirming "Historical cutoff: markets settled before..." and "Ingestion complete: {stats}"
**Why human:** End-to-end live API call requires real network I/O. Cannot verify without making actual HTTP requests to Kalshi. Tests cover unit behavior with mocks.

#### 2. Polymarket live ingestion end-to-end

**Test:** Run `.venv/bin/python -m src.data.ingest_polymarket --categories crypto --output-dir /tmp/poly_test` against the live API
**Expected:** At least one .parquet file in /tmp/poly_test/, _metadata.json with non-empty clob_token_ids, log confirming whether CLOB or trade fallback was used
**Why human:** Three-API integration requires live calls to confirm Gamma keyword matching, CLOB price history responses, and Data API trade fallback all function against the real APIs as documented in CLAUDE.md gotchas.

---

### Gaps Summary

No gaps found. All automated checks passed.

The phase goal is fully achieved: the codebase provides a complete, tested, and wired data ingestion layer for both platforms. All 6 requirements are satisfied, all 45 tests pass, all artifacts are substantive (no stubs), and all key links are wired. The only outstanding items are live API integration tests that require human execution.

---

_Verified: 2026-04-01_
_Verifier: Claude (gsd-verifier)_
