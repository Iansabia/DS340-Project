"""Tier 1 baseline comparison: Naive / Volume / Linear Regression / XGBoost.

This is the April-4 TA check-in deliverable. It:

  1. Loads the Phase 3 matched-pairs parquet files (``train.parquet`` and
     ``test.parquet``) from ``data/processed/``.
  2. Computes the per-row spread-change target (``spread.shift(-1) - spread``
     within each ``pair_id``) and drops rows where the target or required
     features are missing.
  3. Trains and evaluates all four Tier 1 models on the same train/test
     split and the same feature matrix, using ``BasePredictor.evaluate`` so
     every model is scored on the same metrics.
  4. Saves each model's results to JSON in ``experiments/results/tier1/``
     via ``src.evaluation.results_store``.
  5. Prints a formatted comparison table showing RMSE, MAE, directional
     accuracy, P&L, trades, win rate, and Sharpe for every model.

Run:
    python -m experiments.run_baselines
    python -m experiments.run_baselines --threshold 0.05
    python -m experiments.run_baselines --data-dir data/processed --results-dir experiments/results/tier1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.results_store import load_all_results, save_results
from src.models.base import BasePredictor
from src.models.linear_regression import LinearRegressionPredictor
from src.models.naive import NaivePredictor
from src.models.volume import VolumePredictor
from src.models.xgboost_model import XGBoostPredictor


DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_RESULTS_DIR = Path("experiments/results/tier1")
DEFAULT_THRESHOLD = 0.02

# Columns that must never be fed to models as features.
NON_FEATURE_COLUMNS = {
    "timestamp",
    "pair_id",
    "time_idx",
    "group_id",
    "spread_change_target",
}

TARGET_COLUMN = "spread_change_target"


def _build_split(df: pd.DataFrame) -> pd.DataFrame:
    """Add the per-pair spread-change target and drop rows without signal.

    The target is next-bar spread minus current spread within each pair.
    Rows are dropped if the target is NaN (last bar of each pair), or if
    ``spread`` is NaN (missing bar, no price signal to predict from).
    """
    df = df.sort_values(["pair_id", "time_idx"]).reset_index(drop=True)
    df[TARGET_COLUMN] = (
        df.groupby("pair_id")["spread"].shift(-1) - df["spread"]
    )
    df = df.dropna(subset=["spread", TARGET_COLUMN]).reset_index(drop=True)
    # Models can't handle remaining NaNs in feature columns; fill with 0.
    df = df.fillna(0.0)
    return df


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of columns to feed into the models."""
    return [
        c
        for c in df.columns
        if c not in NON_FEATURE_COLUMNS
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def load_train_test(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Phase 3 ``train.parquet`` / ``test.parquet``.

    Raises:
        FileNotFoundError: With a helpful message if either file is missing.
    """
    train_path = data_dir / "train.parquet"
    test_path = data_dir / "test.parquet"

    missing = [p for p in (train_path, test_path) if not p.exists()]
    if missing:
        msg = (
            f"Matched-pairs feature data not found: "
            f"{', '.join(str(p) for p in missing)}. "
            "Run Phase 3 (Feature Engineering) first."
        )
        raise FileNotFoundError(msg)

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    return train, test


def prepare_xy(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Slice ``df`` into feature matrix + target array."""
    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].to_numpy(dtype=float)
    return X, y


def build_models() -> list[BasePredictor]:
    """Instantiate the four Tier 1 baseline models."""
    return [
        NaivePredictor(),
        VolumePredictor(),
        LinearRegressionPredictor(),
        XGBoostPredictor(n_estimators=200, max_depth=4, learning_rate=0.05),
    ]


def format_comparison_table(results: list[dict]) -> str:
    """Render a fixed-width comparison table from loaded result dicts."""
    model_width = max(
        (len(r["model_name"]) for r in results), default=0
    )
    model_width = max(model_width, len("Model"))
    header = (
        f"{'Model':<{model_width}} | {'RMSE':>7} | {'MAE':>7} "
        f"| {'Dir Acc':>7} | {'P&L':>8} | {'Trades':>6} "
        f"| {'Win Rate':>8} | {'Sharpe':>7}"
    )
    separator = (
        f"{'-' * model_width}-+-{'-' * 7}-+-{'-' * 7}-+-{'-' * 7}-+-"
        f"{'-' * 8}-+-{'-' * 6}-+-{'-' * 8}-+-{'-' * 7}"
    )
    lines = [
        "====== Tier 1 Baseline Results ======",
        "",
        header,
        separator,
    ]
    for r in results:
        m = r["metrics"]
        lines.append(
            f"{r['model_name']:<{model_width}} | "
            f"{m['rmse']:>7.4f} | "
            f"{m['mae']:>7.4f} | "
            f"{m['directional_accuracy']:>7.4f} | "
            f"{m['total_pnl']:>8.4f} | "
            f"{m['num_trades']:>6d} | "
            f"{m['win_rate']:>8.4f} | "
            f"{m['sharpe_ratio']:>7.4f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate all four Tier 1 baseline models on the "
            "Phase 3 matched-pairs dataset."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing train.parquet and test.parquet "
        f"(default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Alias for --data-dir (accepts a directory path).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Where to save JSON results (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Profit-sim trade threshold (default: {DEFAULT_THRESHOLD})",
    )
    args = parser.parse_args(argv)

    data_dir = args.data_path or args.data_dir

    try:
        train_raw, test_raw = load_train_test(data_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    train = _build_split(train_raw)
    test = _build_split(test_raw)
    if len(train) == 0 or len(test) == 0:
        print(
            "Error: no usable rows after target computation. "
            f"train={len(train)} test={len(test)}",
            file=sys.stderr,
        )
        return 1

    feature_cols = _feature_columns(train)
    X_train, y_train = prepare_xy(train, feature_cols)
    X_test, y_test = prepare_xy(test, feature_cols)

    print(
        f"Loaded {len(train)} train rows, {len(test)} test rows, "
        f"{len(feature_cols)} features."
    )
    print(f"Results will be saved to: {args.results_dir}")
    print()

    models = build_models()
    for model in models:
        model.fit(X_train, y_train)
        metrics = model.evaluate(X_test, y_test, threshold=args.threshold)
        # Strip the full pnl_series from the stored JSON's summary metrics
        # section; keep it available under a separate key for plotting.
        pnl_series = metrics.pop("pnl_series")
        extra = {
            "threshold": args.threshold,
            "n_train_rows": len(train),
            "n_test_rows": len(test),
            "n_features": len(feature_cols),
            "pnl_series": pnl_series,
        }
        save_results(model.name, metrics, args.results_dir, extra=extra)

    all_results = load_all_results(args.results_dir)
    print(format_comparison_table(all_results))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
