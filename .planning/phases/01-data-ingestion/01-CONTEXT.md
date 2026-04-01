# Phase 1: Data Ingestion - Context

**Gathered:** 2026-04-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Build API adapters for Kalshi and Polymarket that ingest all available historical resolved market data in economics/finance and cryptocurrency categories. Output is parquet files in `data/raw/` with caching, rate limiting, and retry logic.

</domain>

<decisions>
## Implementation Decisions

### Market Scope
- Start with **economics/finance** and **crypto** categories only
- Pull from both platforms within these categories
- If matched pair count is too low after Phase 2, expand to additional categories later

### Time Range
- Pull **all available historical data** (Polymarket back to ~2020, Kalshi back to ~2021)
- More data = more potential matched pairs and supports the window length experiment (6h to 7d)
- Can always filter down during feature engineering

### Market Status
- Pull **resolved/settled markets only** — active markets can't be used for backtesting
- This means using Kalshi's `/historical/` endpoints for settled markets (check cutoff first)
- For Polymarket, must reconstruct price history from trade records since `/prices-history` returns empty for resolved markets

### Data Storage
- **Parquet format** for all raw data — columnar, fast, type-safe
- Store in `data/raw/kalshi/` and `data/raw/polymarket/`
- One parquet file per market or per batch (implementation detail left to planner)

### Kalshi Granularity
- Pull **minute-level candlesticks** (`period_interval=1`) — maximum resolution available
- Gives more data points per market; can aggregate to hourly during feature engineering
- OHLC fields can be null when no trades occurred — handle gracefully

### Polymarket Granularity
- Reconstruct from **individual trade records** via Data API — this is the finest granularity available
- Also pull market metadata from Gamma API (clobTokenIds, descriptions, categories, resolution criteria)
- Rate limits are per-10-seconds (generous), not per-hour

### Retry and Resilience
- Exponential backoff on transient failures (HTTP 429, 500, 502, 503)
- Respect rate limits proactively (sleep between batches if needed)
- Idempotent re-runs: skip already-cached data based on local file existence

### Claude's Discretion
- Internal code structure (class hierarchy, function signatures)
- Exact parquet schema and partitioning strategy
- Logging and progress reporting approach
- Pagination handling details

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### API Documentation
- `CLAUDE.md` — Known data pipeline gotchas (Polymarket token IDs, Kalshi historical cutoff, null OHLC)
- `.planning/research/STACK.md` — Verified installed packages and API connectivity
- `.planning/research/PITFALLS.md` — Data pipeline pitfalls (time alignment, look-ahead bias)

### Architecture
- `.planning/research/ARCHITECTURE.md` — Platform adapter pattern, component boundaries, data flow

</canonical_refs>

<specifics>
## Specific Ideas

- Kalshi base URL: `https://api.elections.kalshi.com/trade-api/v2`
- Kalshi historical cutoff endpoint: `GET /historical/cutoff`
- Kalshi candlestick endpoint: `GET /series/{series_ticker}/markets/{ticker}/candlesticks` (live) or `GET /historical/markets/{ticker}/candlesticks` (historical)
- Polymarket Gamma API: `https://gamma-api.polymarket.com` (metadata, clobTokenIds)
- Polymarket Data API: `https://data-api.polymarket.com` (trade records for price reconstruction)
- Polymarket CLOB API: `https://clob.polymarket.com` (orderbook, current prices — may not be needed for historical data)
- Install `pytest` before coding starts (missing from venv per Stack research)
- Generate `requirements.txt` for team reproducibility

</specifics>

<deferred>
## Deferred Ideas

- Expanding to additional categories (politics, sports) — deferred to after Phase 2 scope assessment
- Real-time streaming / WebSocket connections — out of scope entirely
- Polymarket CLOB endpoint testing — verify in Phase 1 but may not be needed for historical data

</deferred>

---

*Phase: 01-data-ingestion*
*Context gathered: 2026-04-01 via discuss-phase*
