"""Trading profit simulation for spread-prediction strategies.

Given predicted spread changes and realized spread changes, this module
simulates a simple threshold-based trading strategy and reports P&L,
trade count, win rate, Sharpe ratio, and a cumulative P&L series for
equity-curve plotting.

The same simulator is used for regression baselines, time-series models,
and RL policies so results are directly comparable.
"""
from __future__ import annotations

import math

import numpy as np


# Trading-day annualization factor (approximate).
ANNUALIZATION_FACTOR = 252


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
        Otherwise, no trade.

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
          - ``sharpe_ratio`` (float): annualized Sharpe over trade returns,
            using a 252 trading-day factor. 0.0 if fewer than 2 trades or
            if return std is 0.
          - ``pnl_series`` (list[float]): cumulative P&L after each trade
            (for equity-curve plotting).
    """
    predictions = np.asarray(predictions, dtype=float)
    actuals = np.asarray(actuals, dtype=float)

    if predictions.shape != actuals.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} vs actuals {actuals.shape}"
        )

    trade_mask = np.abs(predictions) > threshold
    num_trades = int(trade_mask.sum())

    if num_trades == 0:
        return {
            "total_pnl": 0.0,
            "num_trades": 0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
            "pnl_series": [],
        }

    directions = np.sign(predictions[trade_mask])
    trade_returns = actuals[trade_mask] * directions

    total_pnl = float(trade_returns.sum())
    win_rate = float((trade_returns > 0.0).mean())
    pnl_series = np.cumsum(trade_returns).tolist()

    if num_trades < 2:
        sharpe_ratio = 0.0
    else:
        std = float(trade_returns.std())
        if std == 0.0:
            sharpe_ratio = 0.0
        else:
            mean = float(trade_returns.mean())
            sharpe_ratio = mean / std * math.sqrt(ANNUALIZATION_FACTOR)

    return {
        "total_pnl": total_pnl,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "sharpe_ratio": sharpe_ratio,
        "pnl_series": pnl_series,
    }
