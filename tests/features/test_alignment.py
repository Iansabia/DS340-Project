"""Tests for hourly alignment of matched pair data."""
import numpy as np
import pandas as pd
import pytest

from tests.features.conftest import BASE_TIMESTAMP, HOUR


class TestAlignPairHourly:
    """Tests for src/features/alignment.align_pair_hourly."""

    def test_produces_hourly_timestamps_common_to_both(
        self, sample_kalshi_df, sample_polymarket_df
    ):
        """Aligned output contains only hours present in both platforms."""
        from src.features.alignment import align_pair_hourly

        result = align_pair_hourly(
            sample_kalshi_df, sample_polymarket_df, pair_id="test-pair"
        )

        assert len(result) > 0
        # All timestamps should be on hour boundaries
        assert (result["timestamp"] % HOUR == 0).all()
        # Timestamps should be a subset of both inputs
        kalshi_hours = set(sample_kalshi_df["timestamp"] // HOUR * HOUR)
        poly_hours = set(sample_polymarket_df["timestamp"] // HOUR * HOUR)
        result_hours = set(result["timestamp"])
        assert result_hours.issubset(kalshi_hours & poly_hours)

    def test_forward_fills_null_close_prices(self, sample_polymarket_df):
        """Null close prices are forward-filled within a 6-hour gap."""
        from src.features.alignment import align_pair_hourly

        # Create Kalshi data with some close values and gaps
        timestamps = [BASE_TIMESTAMP + i * HOUR for i in range(20)]
        kalshi_df = pd.DataFrame({
            "timestamp": timestamps,
            "open": [None] * 20,
            "high": [None] * 20,
            "low": [None] * 20,
            "close": [None] * 20,
            "volume": [0.0] * 20,
            "open_interest": [500.0] * 20,
            "yes_bid_close": [str(0.50)] * 5 + [None] * 3 + [str(0.55)] * 5 + [None] * 7,
            "yes_ask_close": [str(0.54)] * 5 + [None] * 3 + [str(0.59)] * 5 + [None] * 7,
            "has_trades": [False] * 20,
        })

        # Trim polymarket to same 20 hours
        poly_df = sample_polymarket_df.iloc[:20].copy()

        result = align_pair_hourly(kalshi_df, poly_df, pair_id="test-pair")

        # Should have at least the 10 rows with bid/ask data (5 + 3 ffill + 5)
        # but not the last 7 where gap > 6 hours
        assert len(result) > 0
        # No null kalshi_close in output (dropped or ffilled)
        assert result["kalshi_close"].notna().all()

    def test_drops_rows_where_both_null(self, sample_polymarket_df):
        """Rows where both platforms have null close are dropped."""
        from src.features.alignment import align_pair_hourly

        # Kalshi with all null
        timestamps = [BASE_TIMESTAMP + i * HOUR for i in range(10)]
        kalshi_df = pd.DataFrame({
            "timestamp": timestamps,
            "open": [None] * 10,
            "high": [None] * 10,
            "low": [None] * 10,
            "close": [None] * 10,
            "volume": [0.0] * 10,
            "open_interest": [500.0] * 10,
            "yes_bid_close": [None] * 10,
            "yes_ask_close": [None] * 10,
            "has_trades": [False] * 10,
        })

        # Polymarket with some null close
        poly_df = sample_polymarket_df.iloc[:10].copy()
        poly_df.loc[poly_df.index[:3], "close"] = np.nan

        result = align_pair_hourly(kalshi_df, poly_df, pair_id="test-pair")

        # Since Kalshi is all null and has no bid/ask either, all rows where
        # Kalshi has no data should be dropped. Only rows where Polymarket
        # has close AND Kalshi was forward-filled could survive -- but with
        # no Kalshi bid/ask, kalshi_close is all NaN, so all should be dropped.
        # The result should be empty since Kalshi has no usable data.
        assert len(result) == 0

    def test_returns_empty_when_no_overlapping_timestamps(
        self, sample_kalshi_df, sample_polymarket_df
    ):
        """Empty DataFrame returned when platforms have no common hours."""
        from src.features.alignment import align_pair_hourly

        # Shift polymarket timestamps to a non-overlapping range
        poly_shifted = sample_polymarket_df.copy()
        poly_shifted["timestamp"] = poly_shifted["timestamp"] + 1_000_000

        result = align_pair_hourly(
            sample_kalshi_df, poly_shifted, pair_id="test-pair"
        )

        assert len(result) == 0
        assert "kalshi_close" in result.columns
        assert "polymarket_close" in result.columns

    def test_bid_ask_midpoint_fallback_for_null_ohlc(self, sample_polymarket_df):
        """When Kalshi OHLC is null but bid/ask exists, uses midpoint as close."""
        from src.features.alignment import align_pair_hourly

        timestamps = [BASE_TIMESTAMP + i * HOUR for i in range(10)]
        kalshi_df = pd.DataFrame({
            "timestamp": timestamps,
            "open": [None] * 10,
            "high": [None] * 10,
            "low": [None] * 10,
            "close": [None] * 10,
            "volume": [0.0] * 10,
            "open_interest": [500.0] * 10,
            "yes_bid_close": [str(0.50)] * 10,
            "yes_ask_close": [str(0.60)] * 10,
            "has_trades": [False] * 10,
        })

        poly_df = sample_polymarket_df.iloc[:10].copy()

        result = align_pair_hourly(kalshi_df, poly_df, pair_id="test-pair")

        assert len(result) == 10
        # Kalshi close should be (0.50 + 0.60) / 2 = 0.55
        assert np.allclose(result["kalshi_close"], 0.55)

    def test_output_columns(self, sample_kalshi_df, sample_polymarket_df):
        """Aligned output has expected column set."""
        from src.features.alignment import align_pair_hourly

        result = align_pair_hourly(
            sample_kalshi_df, sample_polymarket_df, pair_id="test-pair"
        )

        expected_cols = {
            "timestamp", "pair_id", "kalshi_close", "polymarket_close",
            "kalshi_volume", "polymarket_volume",
            "kalshi_bid_close", "kalshi_ask_close",
        }
        assert expected_cols.issubset(set(result.columns))
