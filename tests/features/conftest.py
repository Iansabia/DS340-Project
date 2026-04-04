"""Shared test fixtures for feature engineering tests."""
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


# Fixed epoch for reproducible test data: 2024-01-01 00:00:00 UTC
BASE_TIMESTAMP = 1704067200
HOUR = 3600


@pytest.fixture
def sample_pairs():
    """Three sample pair dicts mimicking accepted_pairs.json structure."""
    return [
        {
            "kalshi_market_id": "KXUE-JPN26FEB-2.1",
            "polymarket_market_id": "0x4f0e5c99cb52770413a218edf0219d8a20ad128ff3804a2582191d58ab5f882a",
            "kalshi_question": "Japan unemployment rate in Feb 2026?",
            "polymarket_question": "Will Japan's February 2026 unemployment rate be >=3.0%?",
            "category": "Economics",
            "kalshi_resolution_date": "2026-03-30T23:33:33Z",
            "polymarket_resolution_date": "2026-03-31T00:00:00Z",
            "confidence_score": 0.83,
            "pair_id": "kxuejpn26feb2.1-0x4f0e5c99",
            "status": "accepted",
        },
        {
            "kalshi_market_id": "KXU3-26FEB-T4.3",
            "polymarket_market_id": "0x89655f46c06c6a7e50610771a7917477fd36ceae3eba8b32f32a2606c9100473",
            "kalshi_question": "Will the unemployment rate (U-3) be above 4.3% in February?",
            "polymarket_question": "Will the February 2026 unemployment rate be 4.3%?",
            "category": "Economics",
            "kalshi_resolution_date": "2026-03-07T18:11:11Z",
            "polymarket_resolution_date": "2026-03-07T18:00:00Z",
            "confidence_score": 0.79,
            "pair_id": "kxu326febt4.3-0x89655f46",
            "status": "accepted",
        },
        {
            "kalshi_market_id": "KXCPI-26FEB-T0.1",
            "polymarket_market_id": "0x0e5014c8f329a364111a1f60bf8adc728f1570412e3624da96ef233cc9e76b01",
            "kalshi_question": "Will CPI month-over-month be above 0.1% in February?",
            "polymarket_question": "February 2026 CPI MoM >= 0.1%?",
            "category": "Economics",
            "kalshi_resolution_date": "2026-03-12T12:30:00Z",
            "polymarket_resolution_date": "2026-03-12T13:00:00Z",
            "confidence_score": 0.85,
            "pair_id": "kxcpi26febt0.1-0x0e5014c8",
            "status": "accepted",
        },
    ]


@pytest.fixture
def tmp_feature_dir(tmp_path):
    """Temporary directory with data/raw/kalshi/ and data/raw/polymarket/ subdirs."""
    kalshi_dir = tmp_path / "raw" / "kalshi"
    kalshi_dir.mkdir(parents=True)
    poly_dir = tmp_path / "raw" / "polymarket"
    poly_dir.mkdir(parents=True)
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def sample_kalshi_df():
    """Kalshi DataFrame: 48 hourly rows, some with trades, some with null OHLC + bid/ask.

    Mimics real data where Kalshi stores None as string 'None' in object columns.
    First 24 rows: have trades (numeric close, bid, ask).
    Last 24 rows: no trades (all None).
    """
    timestamps = [BASE_TIMESTAMP + i * HOUR for i in range(48)]

    # First 24 hours: have trades with numeric data
    close_vals = [0.55 + 0.01 * (i % 5) for i in range(24)]
    bid_vals = [c - 0.02 for c in close_vals]
    ask_vals = [c + 0.02 for c in close_vals]
    volumes = [100.0 + i * 10 for i in range(24)]
    has_trades_first = [True] * 24

    # Last 24 hours: no trades (all None as strings, matching real data format)
    close_none = [None] * 24
    bid_none = [None] * 24
    ask_none = [None] * 24
    volumes_none = [0.0] * 24
    has_trades_last = [False] * 24

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": [str(c) if c is not None else None for c in close_vals] + close_none,
        "high": [str(c + 0.01) if c is not None else None for c in close_vals] + close_none,
        "low": [str(c - 0.01) if c is not None else None for c in close_vals] + close_none,
        "close": [str(c) if c is not None else None for c in close_vals] + close_none,
        "volume": volumes + volumes_none,
        "open_interest": [500.0] * 48,
        "yes_bid_close": [str(b) if b is not None else None for b in bid_vals] + bid_none,
        "yes_ask_close": [str(a) if a is not None else None for a in ask_vals] + ask_none,
        "has_trades": has_trades_first + has_trades_last,
    })


@pytest.fixture
def sample_kalshi_all_null_df():
    """Kalshi DataFrame where ALL rows have null OHLC and bid/ask (common for paired markets).

    This matches the real data where 76/77 paired Kalshi markets have zero trades.
    """
    timestamps = [BASE_TIMESTAMP + i * HOUR for i in range(48)]

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": [None] * 48,
        "high": [None] * 48,
        "low": [None] * 48,
        "close": [None] * 48,
        "volume": [0.0] * 48,
        "open_interest": [500.0] * 48,
        "yes_bid_close": [None] * 48,
        "yes_ask_close": [None] * 48,
        "has_trades": [False] * 48,
    })


@pytest.fixture
def sample_polymarket_df():
    """Polymarket DataFrame: 48 hourly rows, all with close prices, volume=0.

    Uses same hourly timestamp grid as sample_kalshi_df for alignment testing.
    """
    timestamps = [BASE_TIMESTAMP + i * HOUR for i in range(48)]
    close_vals = [0.60 + 0.005 * (i % 8) for i in range(48)]

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": close_vals,
        "high": [c + 0.005 for c in close_vals],
        "low": [c - 0.005 for c in close_vals],
        "close": close_vals,
        "volume": [0] * 48,
    })
