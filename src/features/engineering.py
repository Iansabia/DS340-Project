"""Derived feature computation from aligned_pairs data.

Computes microstructure features from Phase 2.1's aligned_pairs.parquet:
spread velocity, volume ratio, spread momentum/volatility, and order flow
imbalance for each platform.

All rolling/diff operations use groupby("pair_id") to prevent cross-pair
contamination.

Low-liquidity filtering (FEAT-03) is enforced upstream by
src/data/aligner.py (MIN_TRADES_PER_PLATFORM=20). No additional
filtering is needed here.
"""
import numpy as np
import pandas as pd

from src.features.schemas import OUTPUT_COLUMNS


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to aligned_pairs DataFrame.

    All rolling/diff operations are per pair_id to prevent cross-pair
    contamination. NaN inputs produce NaN outputs (no imputation here).

    Args:
        df: aligned_pairs.parquet loaded as DataFrame (31 columns).

    Returns:
        DataFrame with 31 + 6 = 37 columns.
    """
    result = df.copy()

    # 1. price_velocity: spread[t] - spread[t-1] per pair
    result["price_velocity"] = result.groupby("pair_id")["spread"].diff()

    # 2. volume_ratio: kalshi_volume / polymarket_volume
    #    Replace inf/-inf with NaN for division-by-zero safety
    result["volume_ratio"] = (
        result["kalshi_volume"] / result["polymarket_volume"]
    ).replace([np.inf, -np.inf], np.nan)

    # 3. spread_momentum: spread.rolling(3).mean() per pair
    result["spread_momentum"] = result.groupby("pair_id")["spread"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    # 4. spread_volatility: spread.rolling(3).std() per pair
    result["spread_volatility"] = result.groupby("pair_id")["spread"].transform(
        lambda x: x.rolling(3, min_periods=2).std()
    )

    # 5. kalshi_order_flow_imbalance: (buy - sell) / (buy + sell)
    k_denom = result["kalshi_buy_volume"] + result["kalshi_sell_volume"]
    result["kalshi_order_flow_imbalance"] = (
        (result["kalshi_buy_volume"] - result["kalshi_sell_volume"]) / k_denom
    ).where(k_denom != 0, np.nan)

    # 6. polymarket_order_flow_imbalance: same formula for polymarket
    p_denom = result["polymarket_buy_volume"] + result["polymarket_sell_volume"]
    result["polymarket_order_flow_imbalance"] = (
        (result["polymarket_buy_volume"] - result["polymarket_sell_volume"]) / p_denom
    ).where(p_denom != 0, np.nan)

    # Ensure column order matches schema
    return result[OUTPUT_COLUMNS]
