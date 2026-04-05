"""Feature engineering schemas and column contracts.

Defines the column contract between the aligned_pairs.parquet input
(from Phase 2.1 aligner) and the feature engineering output.
"""

# Columns inherited from aligned_pairs.parquet (31 columns from Phase 2.1)
ALIGNED_COLUMNS = [
    "timestamp",
    "kalshi_vwap",
    "kalshi_open",
    "kalshi_high",
    "kalshi_low",
    "kalshi_close",
    "kalshi_volume",
    "kalshi_trade_count",
    "kalshi_dollar_volume",
    "kalshi_buy_volume",
    "kalshi_sell_volume",
    "kalshi_realized_spread",
    "kalshi_max_trade_size",
    "kalshi_has_trade",
    "kalshi_hours_since_last_trade",
    "polymarket_vwap",
    "polymarket_open",
    "polymarket_high",
    "polymarket_low",
    "polymarket_close",
    "polymarket_volume",
    "polymarket_trade_count",
    "polymarket_dollar_volume",
    "polymarket_buy_volume",
    "polymarket_sell_volume",
    "polymarket_realized_spread",
    "polymarket_max_trade_size",
    "polymarket_has_trade",
    "polymarket_hours_since_last_trade",
    "spread",
    "pair_id",
]

# New derived feature columns computed by engineering.py
DERIVED_FEATURE_COLUMNS = [
    "price_velocity",                    # spread[t] - spread[t-1] per pair
    "volume_ratio",                      # kalshi_volume / polymarket_volume
    "spread_momentum",                   # spread.rolling(3).mean() per pair
    "spread_volatility",                 # spread.rolling(3).std() per pair
    "kalshi_order_flow_imbalance",       # (buy_vol - sell_vol) / (buy_vol + sell_vol)
    "polymarket_order_flow_imbalance",   # same for polymarket
]

# Full output columns = ALIGNED_COLUMNS + DERIVED_FEATURE_COLUMNS
OUTPUT_COLUMNS = ALIGNED_COLUMNS + DERIVED_FEATURE_COLUMNS
