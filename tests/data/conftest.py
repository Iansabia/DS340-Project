"""Shared test fixtures for data ingestion tests."""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock
import tempfile


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary directory mimicking data/raw/ structure."""
    kalshi_dir = tmp_path / "raw" / "kalshi"
    kalshi_dir.mkdir(parents=True)
    poly_dir = tmp_path / "raw" / "polymarket"
    poly_dir.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def mock_kalshi_candlestick_response():
    """Sample Kalshi historical candlestick API response."""
    return {
        "candlesticks": [
            {
                "end_period_ts": 1704067200,
                "period_interval": 1,
                "price": {"open": "0.55", "high": "0.60", "low": "0.50", "close": "0.58"},
                "yes_bid": {"close": "0.56"},
                "yes_ask": {"close": "0.60"},
                "volume": "150",
                "open_interest": "1000",
            },
            {
                "end_period_ts": 1704067260,
                "period_interval": 1,
                "price": {"open": None, "high": None, "low": None, "close": None},
                "yes_bid": {"close": "0.57"},
                "yes_ask": {"close": "0.61"},
                "volume": "0",
                "open_interest": "1000",
            },
        ]
    }


@pytest.fixture
def mock_kalshi_null_ohlc_response():
    """Kalshi response where ALL candlesticks have null trade prices (common for low-liquidity)."""
    return {
        "candlesticks": [
            {
                "end_period_ts": 1704067200 + i * 60,
                "period_interval": 1,
                "price": {"open": None, "high": None, "low": None, "close": None},
                "yes_bid": {"close": str(0.50 + i * 0.01)},
                "yes_ask": {"close": str(0.55 + i * 0.01)},
                "volume": "0",
                "open_interest": "500",
            }
            for i in range(5)
        ]
    }


@pytest.fixture
def mock_kalshi_series_response():
    """Sample Kalshi /series API response."""
    return {
        "series": [
            {
                "ticker": "KXBTC",
                "title": "Bitcoin Price",
                "category": "Crypto",
            },
            {
                "ticker": "KXCPI",
                "title": "CPI Report",
                "category": "Economics",
            },
        ]
    }


@pytest.fixture
def mock_kalshi_events_response():
    """Sample Kalshi /events API response with nested markets."""
    return {
        "events": [
            {
                "event_ticker": "KXBTC-25DEC31",
                "markets": [
                    {
                        "ticker": "KXBTC-25DEC31-T50000",
                        "event_ticker": "KXBTC-25DEC31",
                        "title": "Will Bitcoin exceed $50,000 by Dec 31?",
                        "status": "settled",
                        "result": "yes",
                        "close_time": "2025-12-31T23:59:59Z",
                    }
                ],
            }
        ],
        "cursor": "",
    }


@pytest.fixture
def mock_kalshi_cutoff_response():
    """Sample Kalshi /historical/cutoff response."""
    return {
        "market_settled_ts": "2026-01-01T00:00:00Z",
        "orders_updated_ts": "2026-01-01T00:00:00Z",
        "trades_created_ts": "2026-01-01T00:00:00Z",
    }


@pytest.fixture
def mock_polymarket_gamma_events_response():
    """Sample Polymarket Gamma API /events response."""
    return [
        {
            "id": "12345",
            "title": "Will Bitcoin reach $100k by March 2026?",
            "description": "This market resolves Yes if BTC price exceeds $100,000.",
            "closed": True,
            "markets": [
                {
                    "conditionId": "0xabc123",
                    "question": "Will Bitcoin reach $100k by March 2026?",
                    "slug": "will-bitcoin-reach-100k-march-2026",
                    "clobTokenIds": json.dumps(["71321095738111", "71321095738222"]),
                    "endDate": "2026-03-31",
                    "volumeNum": 500000,
                    "outcomes": '["Yes", "No"]',
                    "description": "Resolves based on Coinbase BTC/USD price.",
                    "resolutionSource": "Coinbase",
                }
            ],
        }
    ]


@pytest.fixture
def mock_polymarket_clob_prices_response():
    """Sample Polymarket CLOB /prices-history response."""
    return {
        "history": [
            {"t": 1704067200, "p": "0.65"},
            {"t": 1704070800, "p": "0.68"},
            {"t": 1704074400, "p": "0.62"},
            {"t": 1704078000, "p": "0.70"},
        ]
    }


@pytest.fixture
def mock_polymarket_trades_response():
    """Sample Polymarket Data API /trades response."""
    return [
        {"id": "1", "price": "0.65", "size": "100", "timestamp": "1704067200", "side": "BUY"},
        {"id": "2", "price": "0.66", "size": "50", "timestamp": "1704067500", "side": "SELL"},
        {"id": "3", "price": "0.68", "size": "200", "timestamp": "1704070900", "side": "BUY"},
        {"id": "4", "price": "0.62", "size": "75", "timestamp": "1704074500", "side": "SELL"},
    ]
