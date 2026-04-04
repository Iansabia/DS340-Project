"""Tests for feature engineering pair loader and schemas."""
import json
import pandas as pd
import pytest
from pathlib import Path


class TestFeatureSchemas:
    """Tests for src/features/schemas.py constants."""

    def test_feature_columns_contains_expected_names(self):
        from src.features.schemas import FEATURE_COLUMNS

        expected = [
            "timestamp", "pair_id", "kalshi_close", "polymarket_close",
            "spread", "kalshi_volume", "polymarket_volume", "bid_ask_spread",
            "kalshi_velocity", "polymarket_velocity",
        ]
        assert FEATURE_COLUMNS == expected

    def test_feature_columns_has_10_entries(self):
        from src.features.schemas import FEATURE_COLUMNS

        assert len(FEATURE_COLUMNS) == 10

    def test_min_hours_threshold_is_10(self):
        from src.features.schemas import MIN_HOURS_THRESHOLD

        assert MIN_HOURS_THRESHOLD == 10


class TestLoadValidPairs:
    """Tests for src/features/pair_loader.load_valid_pairs."""

    def test_returns_only_pairs_with_both_parquet_files(
        self, sample_pairs, tmp_feature_dir, sample_kalshi_df, sample_polymarket_df
    ):
        """Only pairs where both kalshi and polymarket parquet files exist are returned."""
        from src.features.pair_loader import load_valid_pairs

        kalshi_dir = tmp_feature_dir / "raw" / "kalshi"
        poly_dir = tmp_feature_dir / "raw" / "polymarket"
        processed_dir = tmp_feature_dir / "processed"

        # Write parquet for first pair only (both platforms)
        sample_kalshi_df.to_parquet(kalshi_dir / f"{sample_pairs[0]['kalshi_market_id']}.parquet")
        sample_polymarket_df.to_parquet(poly_dir / f"{sample_pairs[0]['polymarket_market_id']}.parquet")

        # Write kalshi-only for second pair (no polymarket)
        sample_kalshi_df.to_parquet(kalshi_dir / f"{sample_pairs[1]['kalshi_market_id']}.parquet")

        # Write polymarket-only for third pair (no kalshi)
        sample_polymarket_df.to_parquet(poly_dir / f"{sample_pairs[2]['polymarket_market_id']}.parquet")

        # Write accepted_pairs.json
        pairs_path = processed_dir / "accepted_pairs.json"
        pairs_path.write_text(json.dumps(sample_pairs))

        result = load_valid_pairs(
            pairs_path=str(pairs_path),
            kalshi_dir=str(kalshi_dir),
            polymarket_dir=str(poly_dir),
        )

        assert len(result) == 1
        assert result[0]["pair_id"] == sample_pairs[0]["pair_id"]

    def test_returns_empty_list_when_no_parquet_files_exist(
        self, sample_pairs, tmp_feature_dir
    ):
        """Returns empty list when no parquet files exist on disk."""
        from src.features.pair_loader import load_valid_pairs

        kalshi_dir = tmp_feature_dir / "raw" / "kalshi"
        poly_dir = tmp_feature_dir / "raw" / "polymarket"
        processed_dir = tmp_feature_dir / "processed"

        pairs_path = processed_dir / "accepted_pairs.json"
        pairs_path.write_text(json.dumps(sample_pairs))

        result = load_valid_pairs(
            pairs_path=str(pairs_path),
            kalshi_dir=str(kalshi_dir),
            polymarket_dir=str(poly_dir),
        )

        assert result == []

    def test_returned_dicts_have_expected_keys(
        self, sample_pairs, tmp_feature_dir, sample_kalshi_df, sample_polymarket_df
    ):
        """Returned pair dicts contain required keys."""
        from src.features.pair_loader import load_valid_pairs

        kalshi_dir = tmp_feature_dir / "raw" / "kalshi"
        poly_dir = tmp_feature_dir / "raw" / "polymarket"
        processed_dir = tmp_feature_dir / "processed"

        # Write parquet for first pair on both platforms
        sample_kalshi_df.to_parquet(kalshi_dir / f"{sample_pairs[0]['kalshi_market_id']}.parquet")
        sample_polymarket_df.to_parquet(poly_dir / f"{sample_pairs[0]['polymarket_market_id']}.parquet")

        pairs_path = processed_dir / "accepted_pairs.json"
        pairs_path.write_text(json.dumps(sample_pairs))

        result = load_valid_pairs(
            pairs_path=str(pairs_path),
            kalshi_dir=str(kalshi_dir),
            polymarket_dir=str(poly_dir),
        )

        assert len(result) == 1
        pair = result[0]
        expected_keys = {
            "pair_id", "kalshi_market_id", "polymarket_market_id",
            "kalshi_resolution_date", "polymarket_resolution_date", "category",
        }
        assert expected_keys.issubset(set(pair.keys()))
