"""Feature engineering schemas and constants.

Defines the output contract for the feature engineering pipeline:
column names, data types, and filtering thresholds.
"""

# Minimum number of hours with non-null close price required on EACH platform
# for a pair to pass the low-liquidity filter.
MIN_HOURS_THRESHOLD = 10

# Output columns for the feature matrix parquet file.
# Each row represents one pair at one hourly timestamp.
FEATURE_COLUMNS = [
    "timestamp",              # int: Unix seconds UTC, floored to hour
    "pair_id",                # str: unique matched-pair identifier
    "kalshi_close",           # float: Kalshi close price (trade or bid-ask midpoint)
    "polymarket_close",       # float: Polymarket close price
    "spread",                 # float: kalshi_close - polymarket_close
    "kalshi_volume",          # float: Kalshi trade volume (0 if no trades)
    "polymarket_volume",      # float: Polymarket trade volume (0 if no trades)
    "bid_ask_spread",         # float: Kalshi yes_ask_close - yes_bid_close (NaN if unavailable)
    "kalshi_velocity",        # float: kalshi_close[t] - kalshi_close[t-1] (NaN for first row)
    "polymarket_velocity",    # float: polymarket_close[t] - polymarket_close[t-1] (NaN for first row)
]
