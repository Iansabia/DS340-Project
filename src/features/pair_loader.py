"""Load matched pairs and filter to those with data on both platforms."""
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def load_valid_pairs(
    pairs_path: str,
    kalshi_dir: str,
    polymarket_dir: str,
) -> list[dict]:
    """Load accepted pairs and return only those with parquet files on both platforms.

    Args:
        pairs_path: Path to accepted_pairs.json.
        kalshi_dir: Directory containing Kalshi parquet files ({market_id}.parquet).
        polymarket_dir: Directory containing Polymarket parquet files ({token_id}.parquet).

    Returns:
        List of pair dicts for pairs where both platforms have parquet data.
    """
    with open(pairs_path) as f:
        all_pairs = json.load(f)

    logger.info("Loaded %d accepted pairs from %s", len(all_pairs), pairs_path)

    valid_pairs = []
    missing_kalshi = 0
    missing_polymarket = 0
    missing_both = 0

    for pair in all_pairs:
        kalshi_file = os.path.join(kalshi_dir, f"{pair['kalshi_market_id']}.parquet")
        poly_file = os.path.join(polymarket_dir, f"{pair['polymarket_market_id']}.parquet")

        has_kalshi = os.path.exists(kalshi_file)
        has_poly = os.path.exists(poly_file)

        if has_kalshi and has_poly:
            valid_pairs.append(pair)
        elif not has_kalshi and not has_poly:
            missing_both += 1
        elif not has_kalshi:
            missing_kalshi += 1
        else:
            missing_polymarket += 1

    skipped = len(all_pairs) - len(valid_pairs)
    if skipped > 0:
        reasons = []
        if missing_kalshi > 0:
            reasons.append(f"missing Kalshi data: {missing_kalshi}")
        if missing_polymarket > 0:
            reasons.append(f"missing Polymarket data: {missing_polymarket}")
        if missing_both > 0:
            reasons.append(f"missing both platforms: {missing_both}")
        logger.info("Skipped %d pairs: %s", skipped, "; ".join(reasons))

    logger.info("Found %d pairs with data on both platforms", len(valid_pairs))
    return valid_pairs
