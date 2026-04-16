"""Walk-forward backtest: retrain on rolling windows, test on the next window.

Tests whether the model edge is stable ACROSS TIME or just a lucky single
train/test split. The existing ``run_backtest.py`` uses a fixed train/test
split — this script does true walk-forward evaluation by retraining from
scratch at each window boundary.

Logic:
    1. Concatenate train.parquet + test.parquet for max time coverage
    2. Build target column (next-bar spread change)
    3. Sort globally by timestamp, split into N equal-time windows
    4. For each window i in [1 .. N-1]:
         - Train on all data from windows [0 .. i-1]  (expanding window)
         - Test on window i
         - Record per-window metrics (RMSE, DA, P&L@2pp, Sharpe)
    5. Save log + plot metric evolution over time

Run:
    python -m experiments.run_walk_forward                 # 6 windows, Tier 1 only
    python -m experiments.run_walk_forward --windows 4     # 4 windows
    python -m experiments.run_walk_forward --threshold 0.02
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Matplotlib must use non-interactive backend for headless runs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.features.engineering import compute_derived_features  # noqa: E402
from src.models.linear_regression import LinearRegressionPredictor  # noqa: E402
from src.models.naive import NaivePredictor  # noqa: E402
from src.models.volume import VolumePredictor  # noqa: E402
from src.models.xgboost_model import XGBoostPredictor  # noqa: E402

logger = logging.getLogger(__name__)

# Must match run_baselines.NON_FEATURE_COLUMNS to use same feature set
NON_FEATURE_COLUMNS = {
    "timestamp",
    "pair_id",
    "time_idx",
    "group_id",
    "spread_change_target",
    "kalshi_order_flow_imbalance",  # 100% NaN upstream
    "kalshi_buy_volume",            # zero upstream
    "kalshi_sell_volume",           # zero upstream
    "kalshi_realized_spread",       # zero upstream
}

TARGET_COLUMN = "spread_change_target"


def _load_combined_data(data_dir: Path) -> pd.DataFrame:
    """Load train + test, apply feature engineering, build target."""
    train = pd.read_parquet(data_dir / "train.parquet")
    test = pd.read_parquet(data_dir / "test.parquet")
    df = pd.concat([train, test], ignore_index=True)

    df = compute_derived_features(df)
    df = df.fillna(0.0)

    # Sort per-pair first so shift(-1) respects pair boundaries
    sort_cols = [c for c in ["pair_id", "time_idx", "timestamp"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    df[TARGET_COLUMN] = df.groupby("pair_id")["spread"].shift(-1) - df["spread"]
    df = df.dropna(subset=["spread", TARGET_COLUMN]).reset_index(drop=True)
    df = df.fillna(0.0)

    logger.info(
        "Loaded %d rows across %d pairs, timestamps [%d .. %d]",
        len(df), df["pair_id"].nunique(), df["timestamp"].min(), df["timestamp"].max(),
    )
    return df


def _feature_columns(df: pd.DataFrame) -> list[str]:
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    return [c for c in numeric if c not in NON_FEATURE_COLUMNS]


def _make_windows(df: pd.DataFrame, n_windows: int) -> list[tuple[int, int]]:
    """Split df (already sorted by timestamp) into N equal-time windows.

    Returns list of (timestamp_start, timestamp_end) inclusive/exclusive tuples.
    """
    ts = df["timestamp"].sort_values().to_numpy()
    t_min, t_max = int(ts[0]), int(ts[-1])
    step = (t_max - t_min) // n_windows
    windows = []
    for i in range(n_windows):
        start = t_min + i * step
        end = t_min + (i + 1) * step if i < n_windows - 1 else t_max + 1
        windows.append((start, end))
    return windows


def _simulate_pnl(preds: np.ndarray, actuals: np.ndarray, fee: float = 0.02) -> dict:
    """Same P&L simulation logic as run_baselines + profit_sim: trade only when
    |pred| > fee, pay fee on wins AND losses (symmetric).
    """
    trade_pnls = []
    for p, a in zip(preds, actuals):
        if abs(p) > fee:
            if np.sign(p) == np.sign(a):
                trade_pnls.append(abs(a) - fee)
            else:
                trade_pnls.append(-(abs(a) + fee))
    if not trade_pnls:
        return {
            "num_trades": 0,
            "pnl": 0.0,
            "win_rate": 0.0,
            "sharpe_per_trade": 0.0,
        }
    trade_pnls = np.array(trade_pnls)
    return {
        "num_trades": int(len(trade_pnls)),
        "pnl": float(np.sum(trade_pnls)),
        "win_rate": float(np.mean(trade_pnls > 0)),
        "sharpe_per_trade": float(
            np.mean(trade_pnls) / np.std(trade_pnls, ddof=1)
        ) if np.std(trade_pnls, ddof=1) > 0 else 0.0,
    }


def run_walk_forward(
    data_dir: Path = Path("data/processed"),
    n_windows: int = 6,
    threshold: float = 0.02,
    output_dir: Path = Path("experiments/results/walk_forward"),
) -> list[dict]:
    """Run walk-forward backtest over N time windows.

    Returns the per-window metrics list (also appended to log.jsonl).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_combined_data(data_dir)
    feature_cols = _feature_columns(df)
    logger.info("Features: %d", len(feature_cols))

    windows = _make_windows(df, n_windows)
    logger.info("Windows: %d spans of %d seconds each", n_windows, windows[0][1] - windows[0][0])

    # Model factories so we get a fresh model at each window.
    # Baselines (Naive, Volume) included to show the edge over naive strategies.
    # GRU/LSTM included if torch is available.
    model_factories: dict[str, callable] = {
        "naive": NaivePredictor,
        "volume": VolumePredictor,
        "linear_regression": LinearRegressionPredictor,
        "xgboost": XGBoostPredictor,
    }

    try:
        from src.models.gru import GRUPredictor  # noqa: WPS433
        from src.models.lstm import LSTMPredictor  # noqa: WPS433
        model_factories["gru"] = GRUPredictor
        model_factories["lstm"] = LSTMPredictor
        logger.info("torch available — including GRU + LSTM")
    except ImportError:
        logger.info("torch not available — skipping Tier 2")

    all_results = []

    # Start at window 1 — need window 0 as training baseline
    for i in range(1, n_windows):
        train_end = windows[i][0]      # everything before this window
        test_start = windows[i][0]
        test_end = windows[i][1]

        train_mask = df["timestamp"] < train_end
        test_mask = (df["timestamp"] >= test_start) & (df["timestamp"] < test_end)

        train_df = df[train_mask]
        test_df = df[test_mask]

        if len(test_df) == 0 or len(train_df) < 100:
            logger.warning(
                "Window %d: skipping (train=%d, test=%d)",
                i, len(train_df), len(test_df),
            )
            continue

        X_train = train_df[feature_cols]
        y_train = train_df[TARGET_COLUMN]
        X_test = test_df[feature_cols]
        y_test = test_df[TARGET_COLUMN].to_numpy()

        logger.info(
            "Window %d: train=%d rows, test=%d rows (time: %s .. %s)",
            i, len(train_df), len(test_df),
            datetime.fromtimestamp(test_start, tz=timezone.utc).strftime("%Y-%m-%d"),
            datetime.fromtimestamp(test_end, tz=timezone.utc).strftime("%Y-%m-%d"),
        )

        window_result = {
            "window_idx": i,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "test_start_ts": int(test_start),
            "test_end_ts": int(test_end),
            "test_start_iso": datetime.fromtimestamp(test_start, tz=timezone.utc).isoformat(),
            "test_end_iso": datetime.fromtimestamp(test_end, tz=timezone.utc).isoformat(),
            "n_features": len(feature_cols),
            "threshold": threshold,
            "models": {},
        }

        for name, factory in model_factories.items():
            model = factory()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
            da_mask = y_test != 0
            da = float(np.mean(np.sign(preds[da_mask]) == np.sign(y_test[da_mask]))) if da_mask.sum() else 0.0

            trade_metrics = _simulate_pnl(preds, y_test, fee=threshold)

            window_result["models"][name] = {
                "rmse": round(rmse, 5),
                "directional_accuracy": round(da, 4),
                **{k: round(v, 5) if isinstance(v, float) else v for k, v in trade_metrics.items()},
            }

            logger.info(
                "  %s: RMSE=%.4f DA=%.1f%% trades=%d P&L=$%+.2f Sharpe/trade=%.3f",
                name, rmse, da * 100, trade_metrics["num_trades"],
                trade_metrics["pnl"], trade_metrics["sharpe_per_trade"],
            )

        all_results.append(window_result)

    # Persist to JSONL
    log_path = output_dir / "log.jsonl"
    with log_path.open("w") as f:
        for row in all_results:
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote %d windows to %s", len(all_results), log_path)

    return all_results


def plot_walk_forward(
    results: list[dict],
    output_dir: Path = Path("experiments/figures"),
) -> None:
    """Produce per-metric line plots of each model across windows."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results:
        logger.warning("No results to plot")
        return

    model_names = list(results[0]["models"].keys())
    window_idxs = [r["window_idx"] for r in results]

    metrics_to_plot = [
        ("pnl", "P&L ($)", "walk_forward_pnl.png"),
        ("sharpe_per_trade", "Per-trade Sharpe", "walk_forward_sharpe.png"),
        ("win_rate", "Win Rate", "walk_forward_winrate.png"),
    ]

    for metric_key, ylabel, fname in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(9, 5))
        for name in model_names:
            vals = [r["models"][name][metric_key] for r in results]
            ax.plot(window_idxs, vals, marker="o", label=name.replace("_", " ").title())
        ax.set_xlabel("Window index (chronological)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Walk-Forward Backtest: {ylabel} per Window")
        ax.grid(True, alpha=0.3)
        ax.legend()
        if metric_key == "pnl":
            ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        fig.tight_layout()
        out_path = output_dir / fname
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        logger.info("Wrote %s", out_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Walk-forward backtest (Tier 1 models)")
    parser.add_argument("--windows", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    data_dir = Path(args.data_dir)
    results = run_walk_forward(
        data_dir=data_dir,
        n_windows=args.windows,
        threshold=args.threshold,
    )
    if not args.no_plot:
        plot_walk_forward(results)

    # Summary
    print("\n=== WALK-FORWARD SUMMARY ===")
    print(f"{'Win':>4s} {'Train':>7s} {'Test':>7s}  " + "  ".join(
        f"{name[:8]:>8s}_P&L {name[:8]:>8s}_WR" for name in results[0]["models"].keys()
    ) if results else "No results")
    for r in results:
        row = f"{r['window_idx']:>4d} {r['train_rows']:>7d} {r['test_rows']:>7d}"
        for name in r["models"].keys():
            m = r["models"][name]
            row += f"  ${m['pnl']:>+8.2f} {m['win_rate']:>8.1%}"
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
