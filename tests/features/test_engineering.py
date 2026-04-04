"""Tests for feature computation, liquidity filter, and build pipeline."""
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from tests.features.conftest import BASE_TIMESTAMP, HOUR


class TestComputeFeatures:
    """Tests for src/features/engineering.compute_features."""

    def _make_aligned_df(self, n_rows=20):
        """Helper: create a sample aligned DataFrame."""
        return pd.DataFrame({
            "timestamp": [BASE_TIMESTAMP + i * HOUR for i in range(n_rows)],
            "pair_id": ["test-pair"] * n_rows,
            "kalshi_close": [0.55 + 0.01 * (i % 5) for i in range(n_rows)],
            "polymarket_close": [0.60 + 0.005 * (i % 8) for i in range(n_rows)],
            "kalshi_volume": [100.0] * n_rows,
            "polymarket_volume": [0.0] * n_rows,
            "kalshi_bid_close": [0.53 + 0.01 * (i % 5) for i in range(n_rows)],
            "kalshi_ask_close": [0.57 + 0.01 * (i % 5) for i in range(n_rows)],
        })

    def test_spread_is_kalshi_minus_polymarket(self):
        """Spread = kalshi_close - polymarket_close."""
        from src.features.engineering import compute_features

        df = self._make_aligned_df()
        result = compute_features(df)

        expected_spread = df["kalshi_close"] - df["polymarket_close"]
        pd.testing.assert_series_equal(
            result["spread"], expected_spread, check_names=False
        )

    def test_bid_ask_spread_computation(self):
        """bid_ask_spread = kalshi_ask_close - kalshi_bid_close."""
        from src.features.engineering import compute_features

        df = self._make_aligned_df()
        result = compute_features(df)

        expected_bas = df["kalshi_ask_close"] - df["kalshi_bid_close"]
        pd.testing.assert_series_equal(
            result["bid_ask_spread"], expected_bas, check_names=False
        )

    def test_bid_ask_spread_nan_when_unavailable(self):
        """bid_ask_spread is NaN when Kalshi bid/ask columns are NaN."""
        from src.features.engineering import compute_features

        df = self._make_aligned_df()
        df["kalshi_bid_close"] = np.nan
        df["kalshi_ask_close"] = np.nan

        result = compute_features(df)
        assert result["bid_ask_spread"].isna().all()

    def test_price_velocity_computation(self):
        """Velocity = close[t] - close[t-1] for each platform."""
        from src.features.engineering import compute_features

        df = self._make_aligned_df(5)
        result = compute_features(df)

        # First row should be NaN
        assert pd.isna(result["kalshi_velocity"].iloc[0])
        assert pd.isna(result["polymarket_velocity"].iloc[0])

        # Subsequent rows: diff
        for i in range(1, 5):
            expected_kv = df["kalshi_close"].iloc[i] - df["kalshi_close"].iloc[i - 1]
            expected_pv = df["polymarket_close"].iloc[i] - df["polymarket_close"].iloc[i - 1]
            assert np.isclose(result["kalshi_velocity"].iloc[i], expected_kv)
            assert np.isclose(result["polymarket_velocity"].iloc[i], expected_pv)

    def test_output_has_feature_columns(self):
        """Output DataFrame has all FEATURE_COLUMNS."""
        from src.features.engineering import compute_features
        from src.features.schemas import FEATURE_COLUMNS

        df = self._make_aligned_df()
        result = compute_features(df)

        assert list(result.columns) == FEATURE_COLUMNS


class TestFilterLowLiquidity:
    """Tests for src/features/engineering.filter_low_liquidity."""

    def _make_pair_data(self, pair_id, n_hours, n_null=0):
        """Helper: create (pair_id, aligned_df) with n_hours rows, n_null of which have null close."""
        df = pd.DataFrame({
            "timestamp": [BASE_TIMESTAMP + i * HOUR for i in range(n_hours)],
            "pair_id": [pair_id] * n_hours,
            "kalshi_close": [0.55] * n_hours,
            "polymarket_close": [0.60] * n_hours,
            "kalshi_volume": [100.0] * n_hours,
            "polymarket_volume": [0.0] * n_hours,
            "kalshi_bid_close": [0.53] * n_hours,
            "kalshi_ask_close": [0.57] * n_hours,
        })
        # Set some rows to null
        if n_null > 0:
            df.loc[df.index[:n_null], "kalshi_close"] = np.nan
        return (pair_id, df)

    def test_removes_pairs_below_threshold(self):
        """Pairs with fewer than MIN_HOURS_THRESHOLD non-null hours are excluded."""
        from src.features.engineering import filter_low_liquidity
        from src.features.schemas import MIN_HOURS_THRESHOLD

        # Pair with only 5 non-null hours (below threshold of 10)
        pair_data = [self._make_pair_data("low-liq", 5)]

        kept, report = filter_low_liquidity(pair_data)

        assert len(kept) == 0
        assert len(report) == 1
        assert report[0]["pair_id"] == "low-liq"

    def test_keeps_pairs_at_threshold(self):
        """Pairs with exactly MIN_HOURS_THRESHOLD non-null hours are kept."""
        from src.features.engineering import filter_low_liquidity
        from src.features.schemas import MIN_HOURS_THRESHOLD

        pair_data = [self._make_pair_data("at-threshold", MIN_HOURS_THRESHOLD)]

        kept, report = filter_low_liquidity(pair_data)

        assert len(kept) == 1
        assert kept[0][0] == "at-threshold"
        assert len(report) == 0

    def test_keeps_pairs_above_threshold(self):
        """Pairs above threshold are kept."""
        from src.features.engineering import filter_low_liquidity

        pair_data = [self._make_pair_data("high-liq", 50)]

        kept, report = filter_low_liquidity(pair_data)

        assert len(kept) == 1

    def test_mixed_pairs(self):
        """Mix of above/below threshold pairs filtered correctly."""
        from src.features.engineering import filter_low_liquidity

        pair_data = [
            self._make_pair_data("keep-1", 20),
            self._make_pair_data("drop-1", 5),
            self._make_pair_data("keep-2", 15),
            self._make_pair_data("drop-2", 8),
        ]

        kept, report = filter_low_liquidity(pair_data)

        kept_ids = [p[0] for p in kept]
        assert "keep-1" in kept_ids
        assert "keep-2" in kept_ids
        assert "drop-1" not in kept_ids
        assert "drop-2" not in kept_ids
        assert len(report) == 2


class TestBuildFeatureMatrix:
    """Integration tests for build_features.build_feature_matrix."""

    def test_produces_parquet_output(
        self, sample_pairs, tmp_feature_dir, sample_kalshi_df, sample_polymarket_df
    ):
        """Build pipeline produces a non-empty parquet file."""
        from src.features.build_features import build_feature_matrix

        kalshi_dir = tmp_feature_dir / "raw" / "kalshi"
        poly_dir = tmp_feature_dir / "raw" / "polymarket"
        processed_dir = tmp_feature_dir / "processed"

        # Write parquet for first pair
        sample_kalshi_df.to_parquet(
            kalshi_dir / f"{sample_pairs[0]['kalshi_market_id']}.parquet"
        )
        sample_polymarket_df.to_parquet(
            poly_dir / f"{sample_pairs[0]['polymarket_market_id']}.parquet"
        )

        # Write accepted_pairs.json with only first pair
        pairs_path = processed_dir / "accepted_pairs.json"
        pairs_path.write_text(json.dumps([sample_pairs[0]]))

        result = build_feature_matrix(
            pairs_path=str(pairs_path),
            kalshi_dir=str(kalshi_dir),
            polymarket_dir=str(poly_dir),
            output_dir=str(processed_dir),
        )

        assert result is not None
        assert len(result) > 0

        # Check parquet file exists
        parquet_path = processed_dir / "feature_matrix.parquet"
        assert parquet_path.exists()

        loaded = pd.read_parquet(parquet_path)
        assert len(loaded) == len(result)

    def test_writes_build_report(
        self, sample_pairs, tmp_feature_dir, sample_kalshi_df, sample_polymarket_df
    ):
        """Build pipeline writes build_report.json with expected keys."""
        from src.features.build_features import build_feature_matrix

        kalshi_dir = tmp_feature_dir / "raw" / "kalshi"
        poly_dir = tmp_feature_dir / "raw" / "polymarket"
        processed_dir = tmp_feature_dir / "processed"

        # Write parquet for first pair
        sample_kalshi_df.to_parquet(
            kalshi_dir / f"{sample_pairs[0]['kalshi_market_id']}.parquet"
        )
        sample_polymarket_df.to_parquet(
            poly_dir / f"{sample_pairs[0]['polymarket_market_id']}.parquet"
        )

        pairs_path = processed_dir / "accepted_pairs.json"
        pairs_path.write_text(json.dumps([sample_pairs[0]]))

        build_feature_matrix(
            pairs_path=str(pairs_path),
            kalshi_dir=str(kalshi_dir),
            polymarket_dir=str(poly_dir),
            output_dir=str(processed_dir),
        )

        report_path = processed_dir / "build_report.json"
        assert report_path.exists()

        with open(report_path) as f:
            report = json.load(f)

        expected_keys = {
            "total_accepted_pairs",
            "pairs_with_data",
            "pairs_after_liquidity_filter",
            "total_feature_rows",
            "excluded_pairs",
        }
        assert expected_keys.issubset(set(report.keys()))
        assert report["total_accepted_pairs"] == 1
        assert report["pairs_with_data"] == 1

    def test_handles_all_null_kalshi_pair(
        self, sample_pairs, tmp_feature_dir, sample_kalshi_all_null_df, sample_polymarket_df
    ):
        """Pipeline handles pairs where Kalshi has zero usable data gracefully."""
        from src.features.build_features import build_feature_matrix

        kalshi_dir = tmp_feature_dir / "raw" / "kalshi"
        poly_dir = tmp_feature_dir / "raw" / "polymarket"
        processed_dir = tmp_feature_dir / "processed"

        # Write all-null kalshi data
        sample_kalshi_all_null_df.to_parquet(
            kalshi_dir / f"{sample_pairs[0]['kalshi_market_id']}.parquet"
        )
        sample_polymarket_df.to_parquet(
            poly_dir / f"{sample_pairs[0]['polymarket_market_id']}.parquet"
        )

        pairs_path = processed_dir / "accepted_pairs.json"
        pairs_path.write_text(json.dumps([sample_pairs[0]]))

        result = build_feature_matrix(
            pairs_path=str(pairs_path),
            kalshi_dir=str(kalshi_dir),
            polymarket_dir=str(poly_dir),
            output_dir=str(processed_dir),
        )

        # With all-null Kalshi data, the pair should be filtered out
        # Result should be empty but pipeline should not crash
        assert result is not None
        assert len(result) == 0

        # Build report should document the exclusion
        with open(processed_dir / "build_report.json") as f:
            report = json.load(f)
        assert report["pairs_after_liquidity_filter"] == 0
