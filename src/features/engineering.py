"""Feature computation and liquidity filtering for matched market pairs.

Computes spread, bid-ask spread, and price velocity features from aligned
hourly data. Filters out low-liquidity pairs.
"""
import logging

import numpy as np
import pandas as pd

from src.features.schemas import FEATURE_COLUMNS, MIN_HOURS_THRESHOLD

logger = logging.getLogger(__name__)


def compute_features(aligned_df: pd.DataFrame) -> pd.DataFrame:
    """Compute microstructure features from aligned hourly data.

    Features computed:
    - spread: kalshi_close - polymarket_close
    - bid_ask_spread: kalshi_ask_close - kalshi_bid_close (NaN if unavailable)
    - kalshi_velocity: close[t] - close[t-1] (NaN for first row per pair)
    - polymarket_velocity: close[t] - close[t-1] (NaN for first row per pair)

    Args:
        aligned_df: Output of align_pair_hourly with columns including
            kalshi_close, polymarket_close, kalshi_bid_close, kalshi_ask_close.

    Returns:
        DataFrame with columns matching FEATURE_COLUMNS from schemas.py.
    """
    df = aligned_df.copy()

    # Spread: price difference between platforms
    df["spread"] = df["kalshi_close"] - df["polymarket_close"]

    # Bid-ask spread (NaN propagates naturally if bid/ask are NaN)
    if "kalshi_bid_close" in df.columns and "kalshi_ask_close" in df.columns:
        df["bid_ask_spread"] = df["kalshi_ask_close"] - df["kalshi_bid_close"]
    else:
        df["bid_ask_spread"] = np.nan

    # Price velocity: first difference of close prices
    # Compute per pair_id to avoid cross-pair contamination
    df["kalshi_velocity"] = df.groupby("pair_id")["kalshi_close"].diff()
    df["polymarket_velocity"] = df.groupby("pair_id")["polymarket_close"].diff()

    # Select and order output columns
    result = df[FEATURE_COLUMNS].copy()
    return result


def filter_low_liquidity(
    pairs_data: list[tuple[str, pd.DataFrame]],
    min_hours: int = MIN_HOURS_THRESHOLD,
) -> tuple[list[tuple[str, pd.DataFrame]], list[dict]]:
    """Filter out pairs with insufficient non-null close data.

    A pair is excluded if EITHER platform has fewer than `min_hours` hours
    of non-null close price data.

    Args:
        pairs_data: List of (pair_id, aligned_df) tuples.
        min_hours: Minimum non-null close hours required per platform.

    Returns:
        Tuple of (kept_pairs, filter_report) where:
        - kept_pairs: List of (pair_id, aligned_df) that passed the filter.
        - filter_report: List of dicts documenting excluded pairs with
          pair_id, kalshi_hours, poly_hours, and reason.
    """
    kept = []
    excluded = []

    for pair_id, df in pairs_data:
        kalshi_hours = int(df["kalshi_close"].notna().sum()) if len(df) > 0 else 0
        poly_hours = int(df["polymarket_close"].notna().sum()) if len(df) > 0 else 0

        if kalshi_hours >= min_hours and poly_hours >= min_hours:
            kept.append((pair_id, df))
        else:
            reasons = []
            if kalshi_hours < min_hours:
                reasons.append(f"Kalshi has {kalshi_hours} hours (need {min_hours})")
            if poly_hours < min_hours:
                reasons.append(f"Polymarket has {poly_hours} hours (need {min_hours})")

            excluded.append({
                "pair_id": pair_id,
                "kalshi_hours": kalshi_hours,
                "poly_hours": poly_hours,
                "reason": "; ".join(reasons),
            })

    logger.info(
        "Liquidity filter: kept %d/%d pairs (excluded %d)",
        len(kept),
        len(pairs_data),
        len(excluded),
    )

    return kept, excluded
