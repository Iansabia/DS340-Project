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

    # --- Richer signal features (2026-04-12) ---
    # Multi-window spread momentum: captures short/medium/long trends.
    # The 3-bar window above catches immediate moves; 6 and 12 catch
    # the trend the models need to distinguish real convergence from noise.
    result["spread_momentum_6"] = result.groupby("pair_id")["spread"].transform(
        lambda x: x.rolling(6, min_periods=1).mean()
    )
    result["spread_momentum_12"] = result.groupby("pair_id")["spread"].transform(
        lambda x: x.rolling(12, min_periods=1).mean()
    )

    # Medium-window volatility (6-bar) — complements the 3-bar version.
    result["spread_volatility_6"] = result.groupby("pair_id")["spread"].transform(
        lambda x: x.rolling(6, min_periods=2).std()
    )

    # Spread z-score: how many standard deviations the current spread is
    # from its recent mean. High absolute z-score = anomalous spread =
    # potential entry signal. Uses 6-bar window for stability.
    result["spread_zscore"] = (
        (result["spread"] - result["spread_momentum_6"])
        / result["spread_volatility_6"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    # Spread range: max - min over recent 6-bar window. Wide range =
    # volatile/uncertain pair; narrow range = stable convergence.
    result["spread_range"] = result.groupby("pair_id")["spread"].transform(
        lambda x: x.rolling(6, min_periods=1).max() - x.rolling(6, min_periods=1).min()
    )

    # Liquidity ratio features: asymmetry between platforms signals
    # which side is driving price discovery.
    result["dollar_volume_ratio"] = (
        result["kalshi_dollar_volume"] / result["polymarket_dollar_volume"]
    ).replace([np.inf, -np.inf], np.nan)

    result["trade_count_ratio"] = (
        result["kalshi_trade_count"] / result["polymarket_trade_count"]
    ).replace([np.inf, -np.inf], np.nan)

    # Mid-price: where both platforms roughly agree. Used as a
    # denominator for relative divergence.
    result["mid_price"] = (result["kalshi_close"] + result["polymarket_close"]) / 2.0

    # Price divergence as a percentage of mid-price. A 0.05 spread at
    # mid=0.50 (10%) is more meaningful than 0.05 at mid=0.90 (5.6%).
    result["price_divergence_pct"] = (
        result["spread"].abs() / result["mid_price"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    # Ensure column order matches schema
    return result[OUTPUT_COLUMNS]
