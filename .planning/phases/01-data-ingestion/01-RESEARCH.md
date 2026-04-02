# Phase 1: Data Ingestion - Research

**Researched:** 2026-04-01
**Domain:** API-based data ingestion from Kalshi and Polymarket prediction market platforms
**Confidence:** HIGH (all endpoints live-tested, response schemas verified, rate limits documented)

## Summary

This phase builds two API adapters (Kalshi and Polymarket) that ingest historical resolved market data in economics/finance and crypto categories, saving the results as parquet files in `data/raw/`. The research revealed several critical findings that differ from initial assumptions documented in CLAUDE.md:

1. **Kalshi historical candlesticks** have a 5,000-candlestick-per-request limit. Minute-level data (`period_interval=1`) requires chunking into ~3.5-day windows; hourly data allows ~208-day windows. The historical endpoint response schema differs slightly from the live endpoint (field naming: `close` vs `close_dollars`). Category discovery works via the `/series` endpoint with a `category` parameter (verified: 459 Economics series, 226 Crypto series, 218 Financials series).

2. **Polymarket price history** is best obtained via the CLOB API `/prices-history` endpoint with explicit `startTs`/`endTs` parameters (NOT the `interval` parameter, which returns empty for resolved markets). This provides pre-aggregated hourly price data and is far more efficient than reconstructing from individual trades. The Data API `/trades` endpoint is a fallback but has a hard 3,000-offset pagination limit (~3,500 trades max per market). Polymarket's tag system is too sparse for category discovery; keyword-based filtering on market questions is necessary.

3. **Missing dependencies:** `pyarrow` is NOT installed (parquet read/write fails without it) and `pytest` is NOT installed. Both must be installed before any implementation begins.

**Primary recommendation:** Use a dual-strategy approach: Kalshi via historical candlestick endpoint with time-window chunking, Polymarket via CLOB `/prices-history` with `startTs`/`endTs` chunking (falling back to Data API trades for markets where CLOB returns no data). Install pyarrow and pytest as Wave 0 setup.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Market Scope:** Economics/finance and crypto categories only. Expand later if matched pair count is too low.
- **Time Range:** All available historical data (Polymarket ~2020, Kalshi ~2021). Can filter down during feature engineering.
- **Market Status:** Resolved/settled markets only. Active markets excluded.
- **Data Storage:** Parquet format in `data/raw/kalshi/` and `data/raw/polymarket/`.
- **Kalshi Granularity:** Minute-level candlesticks (`period_interval=1`). Aggregate to hourly during feature engineering.
- **Polymarket Granularity:** Reconstruct from individual trade records via Data API. Also pull metadata from Gamma API.
- **Retry and Resilience:** Exponential backoff on transient failures (429, 500, 502, 503). Rate limit proactively. Idempotent re-runs.

### Claude's Discretion
- Internal code structure (class hierarchy, function signatures)
- Exact parquet schema and partitioning strategy
- Logging and progress reporting approach
- Pagination handling details

### Deferred Ideas (OUT OF SCOPE)
- Expanding to additional categories (politics, sports) -- deferred to after Phase 2 scope assessment
- Real-time streaming / WebSocket connections -- out of scope entirely
- Polymarket CLOB endpoint testing -- verify in Phase 1 but may not be needed for historical data
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DATA-01 | Kalshi API adapter that handles live/historical endpoint split via `/historical/cutoff` | Cutoff endpoint verified: returns `market_settled_ts` (currently 2026-01-01). Historical markets use `GET /historical/markets/{ticker}/candlesticks`. Live markets use `GET /series/{series_ticker}/markets/{ticker}/candlesticks`. Category discovery via `GET /series?category=Economics`. |
| DATA-02 | Polymarket API adapter that queries Gamma for metadata, CLOB for prices, Data API for trade records | Gamma API verified for market discovery (`/events`, `/markets` with `closed=true`). CLOB `/prices-history` with `startTs`/`endTs` is the primary price source. Data API `/trades` is the fallback with 3000-offset limit. |
| DATA-03 | Polymarket price history reconstruction from trade records for resolved markets | CLOB `/prices-history` with explicit timestamps works for resolved markets (verified: 336 hourly points returned for a resolved NBA market). Data API `/trades` available as fallback with pagination constraints. |
| DATA-04 | Rate limiting and local caching for both platform APIs | Kalshi: 20 req/sec (Basic tier, read). Polymarket: no documented rate limit but practical ~10 req/sec recommended. Caching via file-existence check on parquet files per market. |
| DATA-05 | Automated retry with exponential backoff for transient API failures | urllib3 2.6.3 Retry class with HTTPAdapter provides built-in exponential backoff. Configure for status codes 429, 500, 502, 503, 504. |
| DATA-06 | Raw data storage with timestamps in `data/raw/` | Parquet via pyarrow (must install). One file per market ticker. Timestamps stored as UTC int64 (Unix seconds). |
</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| requests | 2.32.5 | HTTP client for all API calls | Already installed. Synchronous is appropriate for batch ingestion. urllib3 Retry handles backoff. |
| pandas | 2.3.3 | Data manipulation and parquet I/O | Already installed. `pd.DataFrame.to_parquet()` and `pd.read_parquet()` for storage. |
| pyarrow | 23.0.1 | Parquet engine for pandas | **NOT INSTALLED -- must install.** Required backend for `df.to_parquet()`. Latest version compatible with pandas 2.3. |
| urllib3 | 2.6.3 | Retry/backoff via `urllib3.util.retry.Retry` | Already installed (dependency of requests). Provides `Retry` class with `backoff_factor` and `status_forcelist`. |
| pytest | 9.0.2 | Test runner for TDD | **NOT INSTALLED -- must install.** Required by AGENTS.md TDD mandate. |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| logging (stdlib) | N/A | Structured logging for ingestion progress | Always -- log API calls, retries, cache hits, errors |
| time (stdlib) | N/A | Rate limiting via `time.sleep()` | Between API request batches to respect rate limits |
| json (stdlib) | N/A | Parse API responses, serialize metadata | Parsing Gamma API responses, clobTokenIds |
| pathlib (stdlib) | N/A | File path management | Cache checking, parquet file paths |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| requests + urllib3 Retry | tenacity | tenacity is more flexible but adds a dependency. urllib3 Retry is already installed and sufficient for HTTP-level retries. |
| requests (sync) | httpx (async) | Async would speed up parallel fetches but adds complexity. Batch ingestion is a one-time operation; 2-3 hours is acceptable. |
| pyarrow | fastparquet | pyarrow is the pandas default engine and more actively maintained. pyarrow is also needed by pytorch-forecasting later. |
| One parquet per market | Partitioned parquet dataset | Partitioning adds complexity for small datasets (<1000 markets). One file per market is simpler for cache invalidation. |

**Installation (Wave 0):**
```bash
.venv/bin/pip install pyarrow pytest pytest-cov
.venv/bin/pip freeze > requirements.txt
```

## Architecture Patterns

### Recommended Project Structure
```
src/
    __init__.py
    data/
        __init__.py
        base.py              # MarketDataAdapter ABC, common schemas
        kalshi.py             # KalshiAdapter
        polymarket.py         # PolymarketAdapter
        client.py             # Resilient HTTP client (retry, rate limit)
        schemas.py            # Parquet schemas, dataclass definitions
        ingest.py             # CLI entry point / orchestrator
data/
    raw/
        kalshi/              # One parquet per market ticker
        polymarket/          # One parquet per conditionId
tests/
    __init__.py
    data/
        __init__.py
        test_client.py       # HTTP retry, rate limit tests
        test_kalshi.py       # Kalshi adapter tests
        test_polymarket.py   # Polymarket adapter tests
        test_schemas.py      # Schema validation tests
        conftest.py          # Shared fixtures (mock responses)
```

### Pattern 1: Platform Adapter with Common Interface
**What:** Abstract base class enforcing a common interface for both platforms. Each adapter normalizes platform-specific quirks into a standard output schema.
**When to use:** Always. This is the foundational pattern for this phase.
**Example:**
```python
# src/data/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd

@dataclass
class MarketMetadata:
    """Platform-agnostic market metadata."""
    market_id: str           # Platform-specific ID (ticker or conditionId)
    question: str            # Human-readable market question
    category: str            # economics, crypto, financials
    platform: str            # 'kalshi' or 'polymarket'
    resolution_date: str     # ISO8601 UTC
    result: str | None       # 'yes', 'no', or None if not yet resolved
    outcomes: list[str]      # ['Yes', 'No'] or platform-specific

class MarketDataAdapter(ABC):
    """Common interface for platform-specific data fetching."""
    
    @abstractmethod
    def list_markets(self, categories: list[str]) -> list[MarketMetadata]:
        """Return metadata for all resolved markets in given categories."""
        ...
    
    @abstractmethod
    def get_candlesticks(self, market_id: str) -> pd.DataFrame:
        """Return candlestick/price data for a single market.
        
        Returns DataFrame with columns:
            timestamp (int): Unix seconds UTC
            open (float|None): Opening price
            high (float|None): High price  
            low (float|None): Low price
            close (float|None): Closing price
            volume (float): Trade volume
        """
        ...
```

### Pattern 2: Resilient HTTP Client
**What:** Centralized HTTP client wrapping `requests.Session` with retry, rate limiting, and logging. Both adapters use this client rather than making raw `requests.get()` calls.
**When to use:** For every API call.
**Example:**
```python
# src/data/client.py
import time
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

class ResilientClient:
    """HTTP client with retry, rate limiting, and logging."""
    
    def __init__(
        self,
        base_url: str,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        requests_per_second: float = 10.0,
    ):
        self.base_url = base_url.rstrip('/')
        self.min_interval = 1.0 / requests_per_second
        self._last_request_time = 0.0
        
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=True,
        )
        self.session = requests.Session()
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
    
    def get(self, path: str, params: dict | None = None) -> dict:
        """Rate-limited GET request with retry."""
        # Enforce rate limit
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        
        url = f"{self.base_url}/{path.lstrip('/')}"
        logger.debug(f"GET {url} params={params}")
        self._last_request_time = time.time()
        
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
```

### Pattern 3: File-Based Caching (Idempotent Re-runs)
**What:** Before fetching any market's data, check if the parquet file already exists on disk. If it does, skip the fetch. This makes the ingestion script idempotent and resumable.
**When to use:** Every market fetch.
**Example:**
```python
# In the adapter
from pathlib import Path

def get_or_fetch_candlesticks(self, market_id: str, output_dir: Path) -> pd.DataFrame:
    """Fetch candlesticks, using cache if available."""
    cache_path = output_dir / f"{market_id}.parquet"
    if cache_path.exists():
        logger.info(f"Cache hit: {cache_path}")
        return pd.read_parquet(cache_path)
    
    df = self.get_candlesticks(market_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info(f"Cached: {cache_path} ({len(df)} rows)")
    return df
```

### Pattern 4: Time-Window Chunking for Large Requests
**What:** Both APIs have per-request limits (Kalshi: 5,000 candlesticks; Polymarket CLOB: variable). Break large time ranges into smaller windows, fetch each, and concatenate.
**When to use:** Any market with a long history (months to years).
**Example:**
```python
def fetch_candlesticks_chunked(
    self, 
    ticker: str, 
    start_ts: int, 
    end_ts: int,
    period_interval: int = 1,  # minutes
    max_candles_per_request: int = 5000,
) -> pd.DataFrame:
    """Fetch candlesticks in time-window chunks."""
    chunk_seconds = max_candles_per_request * period_interval * 60
    frames = []
    current_start = start_ts
    
    while current_start < end_ts:
        current_end = min(current_start + chunk_seconds, end_ts)
        data = self.client.get(
            f"historical/markets/{ticker}/candlesticks",
            params={
                "start_ts": current_start,
                "end_ts": current_end,
                "period_interval": period_interval,
            }
        )
        if data.get("candlesticks"):
            frames.append(self._parse_candlesticks(data["candlesticks"]))
        current_start = current_end
    
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
```

### Anti-Patterns to Avoid

- **Fetching all data in one request:** Both APIs have per-request limits. Always chunk.
- **Hardcoding the historical cutoff date:** The cutoff moves over time. Always call `/historical/cutoff` at the start of each run.
- **Using Polymarket `interval` parameter for resolved markets:** Returns empty. Always use `startTs`/`endTs`.
- **Relying solely on Polymarket Data API `/trades` for price history:** Limited to 3,500 trades per market. CLOB `/prices-history` is the preferred source.
- **Storing raw API JSON instead of normalized parquet:** Wastes disk, makes downstream consumption harder. Normalize to common schema and store as parquet immediately.
- **Silent null handling:** Kalshi OHLC fields are frequently null. Never drop these rows silently; log the count and forward-fill or mark.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| HTTP retry with backoff | Custom retry loop with sleep | `urllib3.util.retry.Retry` + `HTTPAdapter` | Handles 429 Retry-After headers, exponential backoff, configurable status codes. Battle-tested. |
| Parquet read/write | Custom CSV/JSON serialization | `pandas.DataFrame.to_parquet()` with pyarrow | Type-safe columnar storage, compression, fast reads. Industry standard. |
| Rate limiting | Manual timestamp tracking | Simple `time.sleep()` in centralized client | Good enough for batch ingestion. No need for token bucket or leaky bucket at 10-20 req/sec. |
| Progress reporting | Print statements | `logging` module with structured formatters | Filterable by level, redirectable to file, includes timestamps. |

**Key insight:** The ingestion phase is a batch ETL job, not a real-time system. Keep it simple. The complexity is in handling API quirks (null fields, truncated pagination, inconsistent endpoint behavior), not in infrastructure.

## Common Pitfalls

### Pitfall 1: Kalshi 5,000 Candlestick Limit
**What goes wrong:** Requesting a full year of minute-level data returns HTTP 400: "requested time range with candlesticks: X, max candlesticks: 5000."
**Why it happens:** Minute-level data for 1 year = 525,600 candles, far exceeding the 5,000 limit.
**How to avoid:** Calculate chunk size: `5000 * period_interval_minutes * 60` seconds per chunk. For minute-level: ~3.47 days per chunk. For hourly: ~208 days per chunk.
**Warning signs:** HTTP 400 responses from candlestick endpoints.
**Confidence:** HIGH -- verified via live API call.

### Pitfall 2: Polymarket CLOB `interval` vs `startTs/endTs`
**What goes wrong:** Using `interval=max` on `/prices-history` for resolved markets returns an empty history array.
**Why it happens:** The CLOB API prunes aggregated data for resolved markets when using the `interval` parameter. The `startTs`/`endTs` path accesses a different data store.
**How to avoid:** Always use explicit `startTs` and `endTs` parameters. Never use `interval` for resolved markets. Chunk into ~15-day windows.
**Warning signs:** Empty `history` array in response despite the market having volume.
**Confidence:** HIGH -- verified via live API calls and confirmed by GitHub issue #216.

### Pitfall 3: Polymarket Data API 3,000 Offset Limit
**What goes wrong:** Paginating beyond offset=3000 on `/trades` returns HTTP 400: "max historical activity offset of 3000 exceeded."
**Why it happens:** Polymarket enforces a hard pagination limit on historical trade queries. With `limit=500`, you can access at most 3,500 trades per market.
**How to avoid:** Use CLOB `/prices-history` as the primary data source (provides pre-aggregated hourly prices with no offset limit). Use Data API `/trades` only as a fallback for markets where CLOB returns no data.
**Warning signs:** HTTP 400 at offset 3001+.
**Confidence:** HIGH -- verified via live API call.

### Pitfall 4: Kalshi Historical vs Live Endpoint Field Names
**What goes wrong:** The historical candlestick endpoint uses different field names than the live endpoint. Historical uses `price.close`, `price.open`, etc. Live uses `price.close_dollars`, `price.open_dollars`, etc. Writing a single parser for both breaks.
**Why it happens:** The historical and live endpoints were built at different times with different schemas.
**How to avoid:** Write separate response parsers for historical and live endpoints, both normalizing to the same output DataFrame schema. The adapter handles the split transparently.
**Warning signs:** KeyError or None values when parsing candlestick responses.
**Confidence:** HIGH -- verified by comparing live endpoint docs with historical endpoint live response.

### Pitfall 5: Kalshi Null OHLC Fields Are Extremely Common
**What goes wrong:** The majority of minute-level candlesticks have null price fields because no trades occurred. In our test, 100% of candlesticks in a month-long window had null OHLC prices. Only `yes_bid`, `yes_ask`, and `volume` were populated.
**Why it happens:** Many Kalshi markets have sparse trading activity. Minutes or hours pass between trades.
**How to avoid:** Use `yes_bid` and `yes_ask` midpoint as the price when `price.close` is null. The bid-ask midpoint is always available (it reflects the order book) even when no trades occurred. Log the percentage of null-trade candlesticks per market as a data quality metric.
**Warning signs:** A market where >80% of candlesticks have null prices is likely low-liquidity and should be flagged.
**Confidence:** HIGH -- verified via live API call (4/4 candlesticks had null prices in test).

### Pitfall 6: Polymarket Category Discovery is Weak
**What goes wrong:** Polymarket's tag system is poorly organized. There are only ~100 tags, mostly ad-hoc labels like "Costello," "fartcoin," and "wemby." The tag IDs for "Crypto" and "Finance" exist (21 and 120) but return very few events (5 each in our test). There is no reliable category taxonomy.
**Why it happens:** Polymarket tags are community/admin-created and not systematically maintained.
**How to avoid:** Use keyword-based filtering on market `question` and `description` fields rather than relying on tag_id filtering. Keywords: `btc`, `bitcoin`, `eth`, `ethereum`, `crypto`, `solana`, `fed`, `inflation`, `cpi`, `rate`, `gdp`, `recession`, `s&p`, `nasdaq`, `unemployment`. Combine with series slugs where available (e.g., `eth-weeklies`, `btc-weeklies`).
**Warning signs:** Tag-filtered queries returning fewer than 20 events.
**Confidence:** HIGH -- verified via live API calls.

### Pitfall 7: pyarrow Not Installed
**What goes wrong:** `df.to_parquet()` raises `ImportError: Missing optional dependency 'pyarrow'`.
**Why it happens:** pyarrow was never installed in the venv despite parquet being the planned storage format.
**How to avoid:** Install pyarrow in Wave 0 before any implementation.
**Confidence:** HIGH -- verified by attempting `import pyarrow` in the venv.

## Code Examples

### Kalshi: Market Discovery via Series

```python
# Source: Verified via live API call 2026-04-01
# Kalshi organizes markets into Series (recurring events) with category labels.
# Category filter works on the /series endpoint, not /markets.

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
CATEGORIES = ["Economics", "Crypto", "Financials"]

def discover_kalshi_markets(client, categories):
    """Discover all resolved Kalshi markets in target categories."""
    all_tickers = []
    
    # Step 1: Get historical cutoff
    cutoff = client.get("historical/cutoff")
    market_settled_ts = cutoff["market_settled_ts"]  # ISO8601 string
    
    # Step 2: Get series by category
    for category in categories:
        series_list = client.get("series", params={"category": category})
        for series in series_list["series"]:
            series_ticker = series["ticker"]
            
            # Step 3: Get events for each series
            cursor = ""
            while True:
                params = {
                    "series_ticker": series_ticker,
                    "status": "settled",
                    "with_nested_markets": True,
                    "limit": 200,
                }
                if cursor:
                    params["cursor"] = cursor
                events = client.get("events", params=params)
                
                for event in events.get("events", []):
                    for market in event.get("markets", []):
                        all_tickers.append({
                            "ticker": market["ticker"],
                            "event_ticker": market["event_ticker"],
                            "series_ticker": series_ticker,
                            "category": category,
                            "title": market.get("title", ""),
                            "status": market.get("status"),
                            "result": market.get("result"),
                            "close_time": market.get("close_time"),
                        })
                
                cursor = events.get("cursor", "")
                if not cursor:
                    break
    
    return all_tickers
```

### Kalshi: Historical Candlestick Fetch with Chunking

```python
# Source: Verified via live API call 2026-04-01
# Historical endpoint: GET /historical/markets/{ticker}/candlesticks
# Max 5000 candlesticks per request. Minute-level: chunk into ~3.47 day windows.

def fetch_kalshi_candlesticks(client, ticker, start_ts, end_ts, period_interval=1):
    """Fetch candlesticks with automatic time-window chunking.
    
    Args:
        period_interval: 1 (minute), 60 (hour), or 1440 (day)
    """
    MAX_CANDLES = 4500  # Leave buffer below 5000 limit
    chunk_seconds = MAX_CANDLES * period_interval * 60
    
    all_candles = []
    current_start = start_ts
    
    while current_start < end_ts:
        current_end = min(current_start + chunk_seconds, end_ts)
        data = client.get(
            f"historical/markets/{ticker}/candlesticks",
            params={
                "start_ts": current_start,
                "end_ts": current_end,
                "period_interval": period_interval,
            }
        )
        candles = data.get("candlesticks", [])
        all_candles.extend(candles)
        current_start = current_end
    
    return parse_kalshi_candles(all_candles)

def parse_kalshi_candles(candles):
    """Parse Kalshi historical candlestick response into DataFrame.
    
    IMPORTANT: Historical endpoint uses 'close', 'open', etc.
    Live endpoint uses 'close_dollars', 'open_dollars', etc.
    """
    rows = []
    for c in candles:
        price = c.get("price", {})
        bid = c.get("yes_bid", {})
        ask = c.get("yes_ask", {})
        
        # Use trade price if available, otherwise bid-ask midpoint
        trade_close = price.get("close")  # Can be None
        bid_close = bid.get("close")
        ask_close = ask.get("close")
        
        if trade_close is not None:
            close_price = float(trade_close)
        elif bid_close is not None and ask_close is not None:
            close_price = (float(bid_close) + float(ask_close)) / 2
        else:
            close_price = None
        
        rows.append({
            "timestamp": c["end_period_ts"],
            "open": float(price["open"]) if price.get("open") else None,
            "high": float(price["high"]) if price.get("high") else None,
            "low": float(price["low"]) if price.get("low") else None,
            "close": close_price,
            "volume": float(c.get("volume", "0")),
            "open_interest": float(c.get("open_interest", "0")),
            "yes_bid_close": float(bid_close) if bid_close else None,
            "yes_ask_close": float(ask_close) if ask_close else None,
            "has_trades": price.get("close") is not None,
        })
    
    return pd.DataFrame(rows)
```

### Polymarket: Market Discovery via Gamma API

```python
# Source: Verified via live API call 2026-04-01
# Gamma API uses offset-based pagination. Events contain nested markets.
# Tag-based filtering is unreliable; use keyword filtering instead.

GAMMA_BASE = "https://gamma-api.polymarket.com"

CRYPTO_KEYWORDS = [
    "btc", "bitcoin", "eth", "ethereum", "solana", "sol ",
    "crypto", "defi", "token", "altcoin", "nft",
]
FINANCE_KEYWORDS = [
    "fed", "inflation", "cpi", "gdp", "recession", "unemployment",
    "rate cut", "rate hike", "interest rate", "s&p", "nasdaq",
    "dow", "treasury", "bond", "yield", "fomc",
]

def discover_polymarket_markets(client, keywords):
    """Discover resolved Polymarket markets matching keywords."""
    all_markets = []
    offset = 0
    
    while True:
        events = client.get("events", params={
            "closed": "true",
            "limit": 100,
            "offset": offset,
        })
        if not events:
            break
        
        for event in events:
            title = event.get("title", "").lower()
            description = event.get("description", "").lower()
            text = f"{title} {description}"
            
            if any(kw in text for kw in keywords):
                for market in event.get("markets", []):
                    clob_ids = json.loads(market.get("clobTokenIds", "[]"))
                    all_markets.append({
                        "condition_id": market["conditionId"],
                        "clob_token_ids": clob_ids,
                        "question": market.get("question", ""),
                        "slug": market.get("slug", ""),
                        "end_date": market.get("endDate", ""),
                        "volume": market.get("volumeNum", 0),
                        "outcomes": market.get("outcomes", []),
                        "description": market.get("description", ""),
                        "resolution_source": market.get("resolutionSource", ""),
                    })
        
        offset += 100
        if len(events) < 100:
            break
    
    return all_markets
```

### Polymarket: Price History via CLOB with Chunking

```python
# Source: Verified via live API call 2026-04-01
# CRITICAL: Use startTs/endTs, NOT interval parameter (returns empty for resolved markets)
# Chunk into ~14-day windows for reliable data retrieval

CLOB_BASE = "https://clob.polymarket.com"
CHUNK_DAYS = 14

def fetch_polymarket_prices(client, token_id, start_ts, end_ts, fidelity=60):
    """Fetch price history from CLOB API with time-window chunking.
    
    Args:
        token_id: CLOB token ID (long numeric string from clobTokenIds)
        fidelity: Minutes per data point (1, 5, 15, 60)
    
    Returns:
        DataFrame with columns: timestamp, price
    """
    chunk_seconds = CHUNK_DAYS * 86400
    all_points = []
    current_start = start_ts
    
    while current_start < end_ts:
        current_end = min(current_start + chunk_seconds, end_ts)
        data = client.get("prices-history", params={
            "market": token_id,
            "startTs": current_start,
            "endTs": current_end,
            "fidelity": fidelity,
        })
        history = data.get("history", [])
        all_points.extend(history)
        current_start = current_end
    
    if not all_points:
        return pd.DataFrame(columns=["timestamp", "price"])
    
    df = pd.DataFrame(all_points)
    df = df.rename(columns={"t": "timestamp", "p": "price"})
    df["price"] = df["price"].astype(float)
    return df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
```

### Polymarket: Trade Fallback via Data API

```python
# Source: Verified via live API call 2026-04-01
# Hard offset limit: 3000 (error at 3001+)
# Max trades per market: ~3500 (limit=500, offset=0..3000)
# Use as fallback when CLOB prices-history returns no data

def fetch_polymarket_trades(client, condition_id, max_offset=3000):
    """Fetch all accessible trades from Data API.
    
    Returns at most ~3500 trades due to API offset limit.
    """
    all_trades = []
    offset = 0
    
    while offset <= max_offset:
        trades = client.get("trades", params={
            "market": condition_id,
            "limit": 500,
            "offset": offset,
        })
        if not trades:
            break
        all_trades.extend(trades)
        if len(trades) < 500:
            break
        offset += 500
    
    return all_trades

def trades_to_ohlcv(trades, freq_minutes=60):
    """Aggregate raw trades into OHLCV bars.
    
    Uses volume-weighted average price (VWAP) for representative price.
    """
    df = pd.DataFrame(trades)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["price"] = df["price"].astype(float)
    df["size"] = df["size"].astype(float)
    
    df = df.set_index("timestamp").sort_index()
    
    ohlcv = df.resample(f"{freq_minutes}min").agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("size", "sum"),
        vwap=("price", lambda x: np.average(x, weights=df.loc[x.index, "size"]) if len(x) > 0 else np.nan),
        trade_count=("price", "count"),
    )
    
    return ohlcv.reset_index()
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Polymarket `/prices-history` with `interval=max` | Use `startTs`/`endTs` params instead | Discovered via GH issue #216 (2024) | Returns data for resolved markets that `interval` cannot |
| Kalshi live endpoints for all data | Historical endpoints required for settled markets | March 2026 cutoff | Markets settled before `market_settled_ts` only accessible via `/historical/` |
| Polymarket single API | Three APIs (Gamma + CLOB + Data) | Architecture evolution | Must query correct API for each data type |
| Kalshi `period_interval=60` (hourly) | `period_interval=1` (minute-level) now available | 2025 | More granular data but requires chunking due to 5K limit |

**Deprecated/outdated:**
- Kalshi live endpoints for settled markets: Removed from live endpoints as of March 6, 2026. Must use `/historical/` prefix.
- Polymarket `interval` parameter for resolved markets: Technically not deprecated but functionally broken for resolved markets. Use `startTs`/`endTs` instead.

## Open Questions

1. **Polymarket CLOB data availability for old markets**
   - What we know: CLOB `/prices-history` returned data for a 2025 NBA market but may not have data for very old (2020-2022) resolved markets
   - What's unclear: Is there a data retention cutoff for CLOB price history? Do all resolved markets have CLOB data, or only recent ones?
   - Recommendation: Implement dual strategy -- try CLOB first, fall back to Data API trades. Log which markets had CLOB data vs. needed trade reconstruction. This will empirically answer the question.

2. **Kalshi category mapping completeness**
   - What we know: "Economics" (459 series), "Crypto" (226 series), "Financials" (218 series) are available via `/series?category=X`
   - What's unclear: Are there economics/finance markets under other categories (e.g., "Companies" for stock-related markets)?
   - Recommendation: Start with the three listed categories. If matched pair count is low after Phase 2, review "Companies" and other categories for additional candidates.

3. **Kalshi `series_ticker` requirement for live candlesticks**
   - What we know: The live candlestick endpoint requires both `series_ticker` and `ticker` in the path. The historical endpoint only requires `ticker`.
   - What's unclear: Can we always derive `series_ticker` from the market's `event_ticker`?
   - Recommendation: Since all target markets are settled (before the January 2026 cutoff), use only the historical endpoint. No need for the live endpoint.

4. **Polymarket CLOB prices-history consistency**
   - What we know: In testing, one clobTokenId (Yes token) returned 0 points for the same query that the other token (No token) returned 336 points. Subsequent calls were consistent.
   - What's unclear: Is this a data issue or a caching issue? Does it affect other markets?
   - Recommendation: Fetch price history for BOTH clobTokenIds (Yes and No). Use whichever returns data. If both return data, use the Yes token price (standard convention).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2 (to install) |
| Config file | None -- create `pytest.ini` or `pyproject.toml` [tool.pytest] in Wave 0 |
| Quick run command | `.venv/bin/python -m pytest tests/data/ -x -q` |
| Full suite command | `.venv/bin/python -m pytest tests/ -v --tb=short` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-01 | Kalshi adapter routes to historical/live endpoint based on cutoff | unit (mock HTTP) | `.venv/bin/python -m pytest tests/data/test_kalshi.py::test_endpoint_routing -x` | No -- Wave 0 |
| DATA-01 | Kalshi adapter parses candlestick response (incl. null OHLC) | unit (mock HTTP) | `.venv/bin/python -m pytest tests/data/test_kalshi.py::test_parse_candlesticks -x` | No -- Wave 0 |
| DATA-02 | Polymarket adapter queries Gamma for metadata | unit (mock HTTP) | `.venv/bin/python -m pytest tests/data/test_polymarket.py::test_gamma_market_discovery -x` | No -- Wave 0 |
| DATA-02 | Polymarket adapter fetches CLOB prices-history | unit (mock HTTP) | `.venv/bin/python -m pytest tests/data/test_polymarket.py::test_clob_price_fetch -x` | No -- Wave 0 |
| DATA-03 | Polymarket trades-to-OHLCV aggregation produces correct bars | unit (pure function) | `.venv/bin/python -m pytest tests/data/test_polymarket.py::test_trade_aggregation -x` | No -- Wave 0 |
| DATA-04 | Rate limiter enforces minimum interval between requests | unit | `.venv/bin/python -m pytest tests/data/test_client.py::test_rate_limiting -x` | No -- Wave 0 |
| DATA-04 | File cache skips re-fetch for existing parquet files | unit | `.venv/bin/python -m pytest tests/data/test_kalshi.py::test_cache_hit -x` | No -- Wave 0 |
| DATA-05 | Retry logic retries on 429/500/502/503 with backoff | unit (mock HTTP) | `.venv/bin/python -m pytest tests/data/test_client.py::test_retry_on_server_error -x` | No -- Wave 0 |
| DATA-06 | Parquet files written to correct paths with expected schema | integration | `.venv/bin/python -m pytest tests/data/test_schemas.py::test_parquet_schema -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `.venv/bin/python -m pytest tests/data/ -x -q`
- **Per wave merge:** `.venv/bin/python -m pytest tests/ -v --tb=short`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/__init__.py` -- package init
- [ ] `tests/data/__init__.py` -- package init
- [ ] `tests/data/conftest.py` -- shared fixtures (mock API responses for Kalshi and Polymarket)
- [ ] `tests/data/test_client.py` -- ResilientClient tests (retry, rate limit)
- [ ] `tests/data/test_kalshi.py` -- Kalshi adapter tests
- [ ] `tests/data/test_polymarket.py` -- Polymarket adapter tests
- [ ] `tests/data/test_schemas.py` -- Parquet schema validation
- [ ] Framework install: `.venv/bin/pip install pytest pytest-cov pyarrow`
- [ ] Config: `pytest.ini` or `[tool.pytest.ini_options]` in `pyproject.toml`
- [ ] `src/__init__.py`, `src/data/__init__.py` -- package init files

## API Reference (Verified)

### Kalshi Endpoints

| Endpoint | Method | Purpose | Key Params | Limits |
|----------|--------|---------|------------|--------|
| `/historical/cutoff` | GET | Get cutoff timestamps | None | N/A |
| `/series` | GET | List series by category | `category`, `tags`, `include_volume` | No pagination (returns all) |
| `/events` | GET | List events by series | `series_ticker`, `status`, `with_nested_markets`, `limit` (max 200), `cursor` | Cursor-based pagination |
| `/historical/markets` | GET | List historical markets | `event_ticker`, `tickers`, `limit` (max 1000), `cursor` | Cursor-based pagination |
| `/historical/markets/{ticker}/candlesticks` | GET | Historical candlestick data | `start_ts`, `end_ts`, `period_interval` (1/60/1440) | **Max 5000 candlesticks per request** |
| `/markets/candlesticks` | GET | Batch candlesticks (live) | `market_tickers` (CSV, max 100), `start_ts`, `end_ts`, `period_interval` | Max 10,000 candlesticks total |

**Base URL:** `https://api.elections.kalshi.com/trade-api/v2`
**Auth:** No auth required for public market data endpoints
**Rate limit:** 20 req/sec (Basic tier, read operations)

### Polymarket Endpoints

| Endpoint | API | Method | Purpose | Key Params | Limits |
|----------|-----|--------|---------|------------|--------|
| `/events` | Gamma | GET | List events | `closed`, `tag_id`, `limit` (max 100), `offset` | Offset pagination |
| `/markets` | Gamma | GET | List markets | `closed`, `limit`, `offset`, `end_date_min`, `end_date_max`, `slug` | Offset pagination |
| `/tags` | Gamma | GET | List all tags | None | Returns all (~100) |
| `/prices-history` | CLOB | GET | Token price history | `market` (token ID), `startTs`, `endTs`, `fidelity` | **Use startTs/endTs, NOT interval** |
| `/trades` | Data | GET | Trade records | `market` (conditionId), `limit` (max 500), `offset` | **Max offset: 3000** |

**Base URLs:**
- Gamma: `https://gamma-api.polymarket.com`
- CLOB: `https://clob.polymarket.com`
- Data: `https://data-api.polymarket.com`

**Auth:** No auth required for read endpoints
**Rate limit:** Not officially documented; practical limit ~100 req/10sec

### Kalshi Historical Cutoff (Current)
```json
{
  "market_settled_ts": "2026-01-01T00:00:00Z",
  "orders_updated_ts": "2026-01-01T00:00:00Z",
  "trades_created_ts": "2026-01-01T00:00:00Z"
}
```
Since we only pull resolved/settled markets, and all target data predates this cutoff, **use only the `/historical/` endpoints**.

### Kalshi Categories Available
| Category | Series Count | Relevance |
|----------|-------------|-----------|
| Economics | 459 | Primary target |
| Crypto | 226 | Primary target |
| Financials | 218 | Primary target |
| Companies | 321 | Potential expansion if needed |
| All others | 6,130 | Out of scope |

### Polymarket Tag IDs (Verified)
| Tag | ID | Closed Events | Notes |
|-----|----|---------------|-------|
| Crypto | 21 | ~5 | Very sparse; use keyword filtering instead |
| Finance | 120 | ~5 | Very sparse; use keyword filtering instead |
| Macro | 102973 | ~1 | Extremely sparse |

**Recommendation:** Do NOT rely on Polymarket tag_id filtering. Use keyword search on event/market `question` and `description` fields.

## Sources

### Primary (HIGH confidence)
- Kalshi API docs: `https://docs.kalshi.com/api-reference/` -- historical candlesticks, cutoff, markets, series endpoints
- Polymarket API docs: `https://docs.polymarket.com/` -- Gamma markets, CLOB pricing, Data trades
- Live API testing from project environment (2026-04-01) -- all endpoints verified with actual HTTP requests

### Secondary (MEDIUM confidence)
- GitHub issue #216 (Polymarket/py-clob-client) -- `/prices-history` empty data for resolved markets, workaround via startTs/endTs
- GitHub issue #189 (Polymarket/py-clob-client) -- CLOB historical pricing blank responses
- Polymarket Data API docs gist: `https://gist.github.com/shaunlebron/0dd3338f7dea06b8e9f8724981bb13bf` -- endpoint specifications

### Tertiary (LOW confidence)
- Polymarket rate limits: Not officially documented. Practical limit inferred from community reports (~100 req/10sec). Flag for validation during implementation.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all packages verified installed or version-checked via pip
- Architecture: HIGH -- patterns derived from verified API response schemas and tested endpoints
- API specifics: HIGH -- all endpoints live-tested with actual HTTP requests
- Pitfalls: HIGH -- all documented pitfalls verified via live API testing
- Polymarket rate limits: LOW -- not officially documented, inferred from community

**Research date:** 2026-04-01
**Valid until:** 2026-04-15 (API endpoints are stable; cutoff dates may shift)
