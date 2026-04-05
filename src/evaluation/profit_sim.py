"""Trading profit simulation for spread-prediction strategies.

Given predicted spread changes and realized spread changes, this module
simulates a simple threshold-based trading strategy and reports P&L,
trade count, win rate, Sharpe ratio, and a cumulative P&L series for
equity-curve plotting.

The same simulator is used for regression baselines, time-series models,
and RL policies so results are directly comparable.

IMPORTANT: Sharpe ratio is computed as a TIME-SERIES metric over all bars
(including non-trading bars as zero return), not per-trade. This prevents
the inflation that comes from computing Sharpe only over trades.

Annualization uses 4-hour bars: 6 bars/day * 365 days/year = 2190 bars/year.
Prediction markets trade 24/7, so no trading-day adjustment.
"""
from __future__ import annotations

import math

import numpy as np


# Annualization factor for 4-hour bars, 24/7 markets.
# 6 bars per day * 365 days per year = 2190 bars per year.
BARS_PER_YEAR = 6 * 365  # = 2190


def simulate_profit(
    predictions: np.ndarray,
    actuals: np.ndarray,
    threshold: float = 0.02,
) -> dict:
    """Simulate threshold-based directional trading.

    Strategy:
        For each timestep i:
          - If ``|predictions[i]| > threshold``, enter a trade.
          - Position direction = ``sign(predictions[i])``.
          - Profit = ``actuals[i] * sign(predictions[i])``.
        Otherwise, no trade (zero return for that bar).

    Args:
        predictions: Predicted spread changes, shape (n,).
        actuals: Realized spread changes, shape (n,).
        threshold: Minimum absolute predicted spread required to trade.

    Returns:
        Dict with keys:
          - ``total_pnl`` (float): sum of per-trade P&L.
          - ``num_trades`` (int): number of trades taken.
          - ``win_rate`` (float): fraction of trades with positive P&L.
            0.0 if no trades.
          - ``sharpe_ratio`` (float): annualized time-series Sharpe over
            per-bar returns (including non-trading bars as 0). Uses
            sqrt(2190) for 4-hour bars, 24/7. 0.0 if std is 0.
          - ``sharpe_per_trade`` (float): UNANNUALIZED Sharpe of per-trade
            returns. Raw mean/std ratio. Useful for comparing trade quality.
          - ``pnl_series`` (list[float]): cumulative P&L across all bars
            (for equity-curve plotting).
    """
    predictions = np.asarray(predictions, dtype=float)
    actuals = np.asarray(actuals, dtype=float)

    if predictions.shape != actuals.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} vs actuals {actuals.shape}"
        )

    n = len(predictions)
    trade_mask = np.abs(predictions) > threshold
    num_trades = int(trade_mask.sum())

    if num_trades == 0:
        return {
            "total_pnl": 0.0,
            "num_trades": 0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
            "sharpe_per_trade": 0.0,
            "pnl_series": [],
        }

    # Per-bar position: +1, -1, or 0 (no trade)
    positions = np.zeros(n)
    positions[trade_mask] = np.sign(predictions[trade_mask])

    # Per-bar returns (0 when not trading)
    bar_returns = positions * actuals

    # Per-trade returns (only for bars where we traded)
    trade_returns = bar_returns[trade_mask]

    total_pnl = float(trade_returns.sum())
    win_rate = float((trade_returns > 0.0).mean())

    # Cumulative P&L series across all bars (including zeros)
    pnl_series = np.cumsum(bar_returns).tolist()

    # Time-series Sharpe (CORRECT): computed over ALL bars
    bar_std = float(bar_returns.std())
    if bar_std == 0.0 or n < 2:
        sharpe_ratio = 0.0
    else:
        bar_mean = float(bar_returns.mean())
        sharpe_ratio = bar_mean / bar_std * math.sqrt(BARS_PER_YEAR)

    # Per-trade Sharpe (unannualized, for reference)
    if num_trades < 2:
        sharpe_per_trade = 0.0
    else:
        trade_std = float(trade_returns.std())
        if trade_std == 0.0:
            sharpe_per_trade = 0.0
        else:
            trade_mean = float(trade_returns.mean())
            sharpe_per_trade = trade_mean / trade_std

    return {
        "total_pnl": total_pnl,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "sharpe_ratio": sharpe_ratio,
        "sharpe_per_trade": sharpe_per_trade,
        "pnl_series": pnl_series,
    }
