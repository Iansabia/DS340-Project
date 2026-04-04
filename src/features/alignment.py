"""Hourly timestamp alignment for matched market pairs.

Aligns Kalshi and Polymarket candlestick data to a common hourly grid,
handling null OHLC data via bid-ask midpoint fallback and forward-fill.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Maximum gap (in rows) for forward-fill of null close prices.
# At hourly resolution, 6 rows = 6 hours.
MAX_FFILL_LIMIT = 6

# Output columns from alignment (before feature engineering).
ALIGNED_COLUMNS = [
    "timestamp",
    "pair_id",
    "kalshi_close",
    "polymarket_close",
    "kalshi_volume",
    "polymarket_volume",
    "kalshi_bid_close",
    "kalshi_ask_close",
]


def _to_numeric_or_nan(series: pd.Series) -> pd.Series:
    """Convert a series that may contain string 'None' or actual None to float.

    Kalshi parquet files store None as Python None in object-typed columns.
    This converts them to proper NaN floats.
    """
    return pd.to_numeric(series, errors="coerce")


def _prepare_kalshi(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare Kalshi DataFrame: convert types, apply bid-ask midpoint fallback.

    If the 'close' column is entirely null but 'yes_bid_close' and 'yes_ask_close'
    exist, computes close as (bid + ask) / 2.
    """
    result = df.copy()

    # Convert string-typed price columns to numeric
    for col in ["open", "high", "low", "close"]:
        if col in result.columns:
            result[col] = _to_numeric_or_nan(result[col])

    for col in ["yes_bid_close", "yes_ask_close"]:
        if col in result.columns:
            result[col] = _to_numeric_or_nan(result[col])

    # Bid-ask midpoint fallback: if close is null but bid/ask exist, use midpoint
    if "yes_bid_close" in result.columns and "yes_ask_close" in result.columns:
        null_close = result["close"].isna()
        has_bid_ask = result["yes_bid_close"].notna() & result["yes_ask_close"].notna()
        fallback_mask = null_close & has_bid_ask

        if fallback_mask.any():
            midpoint = (result["yes_bid_close"] + result["yes_ask_close"]) / 2
            result.loc[fallback_mask, "close"] = midpoint[fallback_mask]
            logger.debug(
                "Applied bid-ask midpoint fallback for %d/%d rows",
                fallback_mask.sum(),
                len(result),
            )

    return result


def align_pair_hourly(
    kalshi_df: pd.DataFrame,
    polymarket_df: pd.DataFrame,
    pair_id: str,
) -> pd.DataFrame:
    """Align Kalshi and Polymarket data to a common hourly timestamp grid.

    Steps:
    1. Floor timestamps to hour boundaries.
    2. Apply bid-ask midpoint fallback for null Kalshi OHLC.
    3. Keep last row per hour per platform (dedup).
    4. Inner-merge on timestamp (only hours in both platforms).
    5. Forward-fill remaining null close prices (up to 6-hour gap).
    6. Drop rows where either platform still has null close.

    Args:
        kalshi_df: Raw Kalshi candlestick DataFrame.
        polymarket_df: Raw Polymarket candlestick DataFrame.
        pair_id: Unique identifier for this matched pair.

    Returns:
        Aligned DataFrame with columns from ALIGNED_COLUMNS.
        Empty DataFrame (with correct columns) if no overlapping data.
    """
    empty_result = pd.DataFrame(columns=ALIGNED_COLUMNS)

    if len(kalshi_df) == 0 or len(polymarket_df) == 0:
        return empty_result

    # Step 1: Prepare Kalshi data (type conversion + bid-ask fallback)
    k = _prepare_kalshi(kalshi_df)

    # Step 2: Prepare Polymarket data (ensure numeric types)
    p = polymarket_df.copy()
    for col in ["open", "high", "low", "close"]:
        if col in p.columns:
            p[col] = _to_numeric_or_nan(p[col])

    # Step 3: Floor timestamps to hour boundaries
    k["timestamp"] = k["timestamp"] // 3600 * 3600
    p["timestamp"] = p["timestamp"] // 3600 * 3600

    # Step 4: Keep last row per hour (dedup)
    k = k.sort_values("timestamp").groupby("timestamp").last().reset_index()
    p = p.sort_values("timestamp").groupby("timestamp").last().reset_index()

    # Step 5: Inner merge on timestamp
    merged = pd.merge(
        k[["timestamp", "close", "volume"]
          + ([c for c in ["yes_bid_close", "yes_ask_close"] if c in k.columns])],
        p[["timestamp", "close", "volume"]],
        on="timestamp",
        suffixes=("_kalshi", "_poly"),
    )

    if len(merged) == 0:
        return empty_result

    # Rename columns
    rename_map = {
        "close_kalshi": "kalshi_close",
        "close_poly": "polymarket_close",
        "volume_kalshi": "kalshi_volume",
        "volume_poly": "polymarket_volume",
    }
    merged = merged.rename(columns=rename_map)

    # Rename bid/ask if present
    if "yes_bid_close" in merged.columns:
        merged = merged.rename(columns={
            "yes_bid_close": "kalshi_bid_close",
            "yes_ask_close": "kalshi_ask_close",
        })
    else:
        merged["kalshi_bid_close"] = np.nan
        merged["kalshi_ask_close"] = np.nan

    # Step 6: Forward-fill null close prices (up to MAX_FFILL_LIMIT gap)
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    merged["kalshi_close"] = merged["kalshi_close"].ffill(limit=MAX_FFILL_LIMIT)
    merged["polymarket_close"] = merged["polymarket_close"].ffill(limit=MAX_FFILL_LIMIT)

    # Step 7: Drop rows where either platform still has null close
    merged = merged.dropna(subset=["kalshi_close", "polymarket_close"])

    if len(merged) == 0:
        return empty_result

    # Add pair_id
    merged["pair_id"] = pair_id

    # Ensure all output columns present, fill missing with NaN
    for col in ALIGNED_COLUMNS:
        if col not in merged.columns:
            merged[col] = np.nan

    # Select and order output columns
    result = merged[ALIGNED_COLUMNS].copy()
    result = result.reset_index(drop=True)

    logger.debug(
        "Aligned pair %s: %d hours from %d Kalshi x %d Polymarket rows",
        pair_id,
        len(result),
        len(kalshi_df),
        len(polymarket_df),
    )

    return result
