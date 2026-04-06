"""Cross-tier model comparison: Tier 1 baselines + Tier 2 time-series models.

Originally the April-4 TA check-in deliverable (Tier 1 only).  Extended in
Phase 5 Plan 04 to support ``--tier {1,2,both}`` for the unified cross-tier
comparison table that is Phase 5's centerpiece.

It:

  1. Loads the Phase 3 matched-pairs parquet files (``train.parquet`` and
     ``test.parquet``) from ``data/processed/``.
  2. Computes the per-row spread-change target (``spread.shift(-1) - spread``
     within each ``pair_id``) and drops rows where the target or required
     features are missing.
  3. Trains and evaluates models on the same train/test split and the same
     34-feature matrix (``kalshi_order_flow_imbalance`` excluded — 100% NaN
     upstream Phase 3 bug), using ``BasePredictor.evaluate`` so every model
     is scored on the same metrics.
  4. For Tier 2 models, runs 3 seeds {42, 123, 456} and reports mean +/- std.
  5. Saves each model's results to JSON via ``src.evaluation.results_store``.
  6. Prints a formatted comparison table showing RMSE, MAE, directional
     accuracy, P&L, trades, win rate, and Sharpe for every model.

Run:
    python -m experiments.run_baselines                        # default: Tier 1 only
    python -m experiments.run_baselines --tier 2               # Tier 2 only (GRU + LSTM, 3 seeds)
    python -m experiments.run_baselines --tier both            # Combined comparison table
    python -m experiments.run_baselines --tier both --threshold 0.05
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.results_store import load_all_results, save_results
from src.models.base import BasePredictor
from src.models.gru import GRUPredictor
from src.models.linear_regression import LinearRegressionPredictor
from src.models.lstm import LSTMPredictor
from src.models.naive import NaivePredictor
from src.models.volume import VolumePredictor
from src.models.xgboost_model import XGBoostPredictor


DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_RESULTS_DIR = Path("experiments/results/tier1")
DEFAULT_RESULTS_DIR_TIER2 = Path("experiments/results/tier2")
DEFAULT_THRESHOLD = 0.02

# Seeds for Tier 2 multi-seed aggregation (CONTEXT.md D8).
TIER2_SEEDS = [42, 123, 456]

# Columns that must never be fed to models as features.
NON_FEATURE_COLUMNS = {
    "timestamp",
    "pair_id",
    "time_idx",
    "group_id",
    "spread_change_target",
    "kalshi_order_flow_imbalance",  # 100% NaN -- upstream Phase 3 bug, drop at harness level
}

TARGET_COLUMN = "spread_change_target"

# Fixed display order for the combined comparison table.
_MODEL_ORDER = [
    "Naive (Spread Closes)",
    "Volume (Higher Volume Correct)",
    "Linear Regression",
    "XGBoost",
    "GRU",
    "LSTM",
]


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
    """Slice ``df`` into feature matrix + target array (Tier 1)."""
    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].to_numpy(dtype=float)
    return X, y


def prepare_xy_for_seq(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build (X_with_group, y) for Tier 2 sequence models.

    Tier 2 needs ``group_id`` alongside features for pair-boundary-respecting
    windowing.  Returns ``X = df[feature_cols + ['group_id']]``,
    ``y = df['spread_change_target'].to_numpy()``.
    """
    X = df[feature_cols + ["group_id"]].copy()
    y = df[TARGET_COLUMN].to_numpy(dtype=float)
    return X, y


def build_models(tier: str = "1") -> list[BasePredictor]:
    """Instantiate models for the requested tier.

    Args:
        tier: ``"1"`` for Tier 1 regression baselines, ``"2"`` for Tier 2
            time-series models, ``"both"`` for all models.

    Returns:
        List of ``BasePredictor`` instances.

    Raises:
        ValueError: If *tier* is not one of ``{"1", "2", "both"}``.
    """
    tier1 = [
        NaivePredictor(),
        VolumePredictor(),
        LinearRegressionPredictor(),
        XGBoostPredictor(n_estimators=200, max_depth=4, learning_rate=0.05),
    ]
    tier2 = [
        GRUPredictor(random_state=42),
        LSTMPredictor(random_state=42),
    ]
    if tier == "1":
        return tier1
    if tier == "2":
        return tier2
    if tier == "both":
        return tier1 + tier2
    raise ValueError(f"Unknown tier: {tier!r}")


def run_tier2_with_seeds(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
    threshold: float,
    results_dir: Path,
    n_train: int,
    n_test: int,
    n_features: int,
    seeds: list[int] | None = None,
) -> None:
    """Train each Tier 2 model with multiple seeds and save aggregated results.

    For each model class (GRU, LSTM): trains once per seed, collects per-seed
    RMSE, and saves the last seed's full result alongside seed-aggregated
    statistics in the ``extra`` field.
    """
    if seeds is None:
        seeds = TIER2_SEEDS

    X_train_seq, y_train = prepare_xy_for_seq(df_train, feature_cols)
    X_test_seq, y_test = prepare_xy_for_seq(df_test, feature_cols)

    model_classes: list[type[BasePredictor]] = [GRUPredictor, LSTMPredictor]

    for cls in model_classes:
        seed_rmses: list[float] = []
        last_metrics: dict | None = None
        last_pnl_series: list | None = None

        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"Training {cls.__name__} with seed={seed}")
            print(f"{'='*60}")
            model = cls(random_state=seed)  # type: ignore[call-arg]
            model.fit(X_train_seq, y_train)
            metrics = model.evaluate(X_test_seq, y_test, threshold=threshold)

            seed_rmses.append(metrics["rmse"])
            last_pnl_series = metrics.pop("pnl_series")
            last_metrics = metrics

        assert last_metrics is not None and last_pnl_series is not None

        mean_rmse = float(np.mean(seed_rmses))
        std_rmse = float(np.std(seed_rmses, ddof=0))

        extra = {
            "threshold": threshold,
            "n_train_rows": n_train,
            "n_test_rows": n_test,
            "n_features": n_features,
            "pnl_series": last_pnl_series,
            "seeds": seeds,
            "seed_rmses": [float(r) for r in seed_rmses],
            "mean_rmse": mean_rmse,
            "std_rmse": std_rmse,
        }

        model_name = cls.__name__.replace("Predictor", "")
        save_results(model_name, last_metrics, results_dir, extra=extra)
        print(
            f"\n[{model_name}] Saved: mean_rmse={mean_rmse:.4f} "
            f"+/- {std_rmse:.4f} over {len(seeds)} seeds"
        )


def format_comparison_table(
    results: list[dict],
    tier: str = "1",
) -> str:
    """Render a fixed-width comparison table from loaded result dicts.

    Args:
        results: List of result dicts (from ``load_all_results``).
        tier: ``"1"``, ``"2"``, or ``"both"`` -- controls the title and
            display order.

    Returns:
        Formatted multi-line string.
    """
    # Sort by fixed model order; unknown models go to the end
    def _sort_key(r: dict) -> int:
        name = r["model_name"]
        try:
            return _MODEL_ORDER.index(name)
        except ValueError:
            return len(_MODEL_ORDER)

    results = sorted(results, key=_sort_key)

    model_width = max(
        (len(r["model_name"]) for r in results), default=0
    )
    model_width = max(model_width, len("Model"))

    header = (
        f"{'Model':<{model_width}} | {'RMSE':>7} | {'MAE':>7} "
        f"| {'Dir Acc':>7} | {'P&L':>8} | {'Trades':>6} "
        f"| {'Win Rate':>8} | {'Sharpe':>7} | {'Raw SR':>7}"
    )
    separator = (
        f"{'-' * model_width}-+-{'-' * 7}-+-{'-' * 7}-+-{'-' * 7}-+-"
        f"{'-' * 8}-+-{'-' * 6}-+-{'-' * 8}-+-{'-' * 7}-+-{'-' * 7}"
    )

    if tier == "1":
        title = "====== Tier 1 Baseline Results ======"
    elif tier == "2":
        title = "====== Tier 2 Time Series Results ======"
    else:
        title = "====== Cross-Tier Comparison ======"

    lines = [
        title,
        "",
        "  Sharpe: annualized time-series Sharpe (4h bars, 24/7, sqrt(2190))",
        "  Raw SR: unannualized per-trade mean/std (raw trade quality)",
        "",
        header,
        separator,
    ]

    for r in results:
        m = r["metrics"]
        extra = r.get("extra", {})

        # Build RMSE display: append seed std for Tier 2 models
        rmse_val = m["rmse"]
        rmse_str = f"{rmse_val:>7.4f}"
        seed_note = ""
        if "mean_rmse" in extra and "std_rmse" in extra:
            seed_note = f" (+/-{extra['std_rmse']:.4f})"

        lines.append(
            f"{r['model_name']:<{model_width}} | "
            f"{rmse_str}{seed_note} | "
            f"{m['mae']:>7.4f} | "
            f"{m['directional_accuracy']:>7.4f} | "
            f"{m['total_pnl']:>8.4f} | "
            f"{m['num_trades']:>6d} | "
            f"{m['win_rate']:>8.4f} | "
            f"{m['sharpe_ratio']:>7.4f} | "
            f"{m.get('sharpe_per_trade', 0.0):>7.4f}"
        )

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate models on the Phase 3 matched-pairs dataset. "
            "Use --tier to select Tier 1 (regression baselines), Tier 2 "
            "(time-series: GRU, LSTM), or both for a combined comparison."
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
        default=None,
        help="Where to save JSON results. If omitted, routes by tier: "
        f"tier1 -> {DEFAULT_RESULTS_DIR}, tier2 -> {DEFAULT_RESULTS_DIR_TIER2}",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Profit-sim trade threshold (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--tier",
        type=str,
        choices=["1", "2", "both"],
        default="1",
        help="Which tier(s) to run: 1=regression baselines, "
        "2=time-series (GRU, LSTM), both=combined comparison",
    )
    args = parser.parse_args(argv)

    data_dir = args.data_path or args.data_dir
    tier = args.tier

    # Resolve results directories
    tier1_results_dir = args.results_dir or DEFAULT_RESULTS_DIR
    tier2_results_dir = args.results_dir or DEFAULT_RESULTS_DIR_TIER2
    # If user explicitly sets --results-dir and tier is 'both', both tiers
    # write to the same dir (user override). Otherwise use separate dirs.
    if args.results_dir and tier == "both":
        tier2_results_dir = args.results_dir

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
    n_train = len(train)
    n_test = len(test)
    n_features = len(feature_cols)

    print(
        f"Loaded {n_train} train rows, {n_test} test rows, "
        f"{n_features} features."
    )

    # ---- Tier 1 ----
    if tier in ("1", "both"):
        print(f"\nResults will be saved to: {tier1_results_dir}")
        print()
        X_train, y_train = prepare_xy(train, feature_cols)
        X_test, y_test = prepare_xy(test, feature_cols)

        models = build_models(tier="1")
        for model in models:
            model.fit(X_train, y_train)
            metrics = model.evaluate(X_test, y_test, threshold=args.threshold)
            pnl_series = metrics.pop("pnl_series")
            extra = {
                "threshold": args.threshold,
                "n_train_rows": n_train,
                "n_test_rows": n_test,
                "n_features": n_features,
                "pnl_series": pnl_series,
            }
            save_results(model.name, metrics, tier1_results_dir, extra=extra)

    # ---- Tier 2 ----
    if tier in ("2", "both"):
        print(f"\nTier 2 results will be saved to: {tier2_results_dir}")
        run_tier2_with_seeds(
            df_train=train,
            df_test=test,
            feature_cols=feature_cols,
            threshold=args.threshold,
            results_dir=tier2_results_dir,
            n_train=n_train,
            n_test=n_test,
            n_features=n_features,
        )

    # ---- Print comparison table ----
    if tier == "1":
        all_results = load_all_results(tier1_results_dir)
        print()
        print(format_comparison_table(all_results, tier="1"))
    elif tier == "2":
        all_results = load_all_results(tier2_results_dir)
        print()
        print(format_comparison_table(all_results, tier="2"))
    else:  # "both"
        tier1_results = load_all_results(tier1_results_dir)
        tier2_results = load_all_results(tier2_results_dir)
        combined = tier1_results + tier2_results
        print()
        print(format_comparison_table(combined, tier="both"))

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
