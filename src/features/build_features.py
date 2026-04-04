"""CLI entry point for the feature engineering pipeline.

Loads matched pairs, aligns hourly timestamps, computes features,
filters low-liquidity markets, and saves feature_matrix.parquet.
"""
import argparse
import json
import logging
import os

import pandas as pd

from src.features.alignment import align_pair_hourly
from src.features.engineering import compute_features, filter_low_liquidity
from src.features.pair_loader import load_valid_pairs

logger = logging.getLogger(__name__)


def build_feature_matrix(
    pairs_path: str,
    kalshi_dir: str,
    polymarket_dir: str,
    output_dir: str,
) -> pd.DataFrame:
    """Run the full feature engineering pipeline.

    Steps:
    1. Load valid pairs (both platforms have parquet data).
    2. For each pair, read parquet files and align hourly timestamps.
    3. Filter out low-liquidity pairs.
    4. Compute features for surviving pairs.
    5. Concatenate, sort, and save to parquet.
    6. Write build_report.json with pipeline statistics.

    Args:
        pairs_path: Path to accepted_pairs.json.
        kalshi_dir: Directory with Kalshi parquet files.
        polymarket_dir: Directory with Polymarket parquet files.
        output_dir: Directory to save feature_matrix.parquet and build_report.json.

    Returns:
        Concatenated feature DataFrame (empty if no pairs survive filtering).
    """
    # Step 1: Load valid pairs
    valid_pairs = load_valid_pairs(pairs_path, kalshi_dir, polymarket_dir)

    # Count total from the JSON file
    with open(pairs_path) as f:
        total_accepted = len(json.load(f))

    logger.info(
        "Pipeline start: %d accepted pairs, %d with data on both platforms",
        total_accepted,
        len(valid_pairs),
    )

    # Step 2: Align each pair
    aligned_pairs = []
    alignment_failures = []

    for pair in valid_pairs:
        pair_id = pair["pair_id"]
        kalshi_file = os.path.join(kalshi_dir, f"{pair['kalshi_market_id']}.parquet")
        poly_file = os.path.join(polymarket_dir, f"{pair['polymarket_market_id']}.parquet")

        try:
            kalshi_df = pd.read_parquet(kalshi_file)
            poly_df = pd.read_parquet(poly_file)

            aligned = align_pair_hourly(kalshi_df, poly_df, pair_id)

            if len(aligned) > 0:
                aligned_pairs.append((pair_id, aligned))
            else:
                alignment_failures.append({
                    "pair_id": pair_id,
                    "kalshi_hours": 0,
                    "poly_hours": 0,
                    "reason": "No overlapping hours with non-null close on both platforms",
                })
                logger.debug("Pair %s: no aligned hours, skipping", pair_id)

        except Exception as e:
            alignment_failures.append({
                "pair_id": pair_id,
                "kalshi_hours": 0,
                "poly_hours": 0,
                "reason": f"Alignment error: {e}",
            })
            logger.warning("Failed to align pair %s: %s", pair_id, e)

    logger.info(
        "Alignment: %d pairs produced data, %d failed/empty",
        len(aligned_pairs),
        len(alignment_failures),
    )

    # Step 3: Filter low-liquidity pairs
    kept_pairs, filter_report = filter_low_liquidity(aligned_pairs)

    # Combine all exclusion reasons
    all_excluded = alignment_failures + filter_report

    # Step 4: Compute features for surviving pairs
    feature_dfs = []
    for pair_id, aligned_df in kept_pairs:
        features = compute_features(aligned_df)
        feature_dfs.append(features)

    # Step 5: Concatenate and sort
    if feature_dfs:
        result = pd.concat(feature_dfs, ignore_index=True)
        result = result.sort_values(["pair_id", "timestamp"]).reset_index(drop=True)
    else:
        from src.features.schemas import FEATURE_COLUMNS
        result = pd.DataFrame(columns=FEATURE_COLUMNS)

    logger.info(
        "Feature matrix: %d rows, %d pairs",
        len(result),
        result["pair_id"].nunique() if len(result) > 0 else 0,
    )

    # Step 6: Save outputs
    os.makedirs(output_dir, exist_ok=True)

    parquet_path = os.path.join(output_dir, "feature_matrix.parquet")
    result.to_parquet(parquet_path, index=False)
    logger.info("Saved feature matrix to %s", parquet_path)

    report = {
        "total_accepted_pairs": total_accepted,
        "pairs_with_data": len(valid_pairs),
        "pairs_aligned": len(aligned_pairs),
        "pairs_after_liquidity_filter": len(kept_pairs),
        "total_feature_rows": len(result),
        "unique_pairs": int(result["pair_id"].nunique()) if len(result) > 0 else 0,
        "excluded_pairs": all_excluded,
    }

    report_path = os.path.join(output_dir, "build_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved build report to %s", report_path)

    return result


def main():
    """CLI entry point for feature engineering pipeline."""
    parser = argparse.ArgumentParser(
        description="Build feature matrix from matched market pairs."
    )
    parser.add_argument(
        "--pairs-path",
        default="data/processed/accepted_pairs.json",
        help="Path to accepted_pairs.json (default: data/processed/accepted_pairs.json)",
    )
    parser.add_argument(
        "--kalshi-dir",
        default="data/raw/kalshi",
        help="Directory with Kalshi parquet files (default: data/raw/kalshi)",
    )
    parser.add_argument(
        "--polymarket-dir",
        default="data/raw/polymarket",
        help="Directory with Polymarket parquet files (default: data/raw/polymarket)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory for feature_matrix.parquet (default: data/processed)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    result = build_feature_matrix(
        pairs_path=args.pairs_path,
        kalshi_dir=args.kalshi_dir,
        polymarket_dir=args.polymarket_dir,
        output_dir=args.output_dir,
    )

    print(f"Feature matrix: {len(result)} rows, {result['pair_id'].nunique() if len(result) > 0 else 0} pairs")
    print(f"Columns: {list(result.columns)}")


if __name__ == "__main__":
    main()
