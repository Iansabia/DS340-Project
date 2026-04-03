"""Tests for quality_filter module."""
import pytest
from src.matching.quality_filter import (
    parse_date,
    extract_direction,
    extract_threshold,
    directions_compatible,
    thresholds_compatible,
    filter_candidates,
)


class TestParseDate:
    def test_iso_with_timezone(self):
        d = parse_date("2026-03-06T13:29:00Z")
        assert d is not None
        assert d.year == 2026 and d.month == 3 and d.day == 6

    def test_date_only(self):
        d = parse_date("2026-01-15")
        assert d is not None
        assert d.year == 2026

    def test_empty(self):
        assert parse_date("") is None
        assert parse_date(None) is None


class TestExtractDirection:
    def test_above(self):
        assert extract_direction("Will unemployment be above 4.3%?") == "above"

    def test_below(self):
        assert extract_direction("Will BTC drop below $60,000?") == "below"

    def test_exactly(self):
        assert extract_direction("Will unemployment be exactly 4.1%?") == "exactly"

    def test_between(self):
        assert extract_direction("Will BTC be between $80k and $85k?") == "between"

    def test_none(self):
        assert extract_direction("GDP growth in 2025?") is None


class TestExtractThreshold:
    def test_percentage(self):
        assert extract_threshold("Will unemployment be above 4.3%?") == 4.3

    def test_dollar(self):
        assert extract_threshold("Will BTC be above $80,000?") == 80000

    def test_no_threshold(self):
        assert extract_threshold("GDP growth in 2025?") is None


class TestDirectionsCompatible:
    def test_same(self):
        assert directions_compatible("above", "above") is True

    def test_opposite(self):
        assert directions_compatible("above", "below") is False

    def test_above_exactly(self):
        assert directions_compatible("above", "exactly") is False

    def test_none(self):
        assert directions_compatible(None, "above") is True


class TestThresholdsCompatible:
    def test_same(self):
        assert thresholds_compatible(4.3, 4.3) is True

    def test_close_percentage(self):
        assert thresholds_compatible(4.3, 4.1) is True  # within 0.5

    def test_different(self):
        assert thresholds_compatible(4.3, 3.4) is False

    def test_close_dollar(self):
        assert thresholds_compatible(80000, 81000) is True  # within 5%

    def test_far_dollar(self):
        assert thresholds_compatible(80000, 60000) is False

    def test_none(self):
        assert thresholds_compatible(None, 4.3) is True


class TestFilterCandidates:
    def test_filters_low_confidence(self):
        candidates = [{"confidence_score": 0.3, "kalshi_question": "test", "polymarket_question": "test"}]
        assert len(filter_candidates(candidates)) == 0

    def test_filters_temporal_mismatch(self):
        candidates = [{
            "confidence_score": 0.9,
            "kalshi_question": "Will unemployment be above 4.3%?",
            "polymarket_question": "Will unemployment be above 4.3%?",
            "kalshi_resolution_date": "2026-01-15T00:00:00Z",
            "polymarket_resolution_date": "2022-01-15T00:00:00Z",
        }]
        assert len(filter_candidates(candidates)) == 0

    def test_filters_direction_mismatch(self):
        candidates = [{
            "confidence_score": 0.9,
            "kalshi_question": "Will unemployment be above 4.3%?",
            "polymarket_question": "Will unemployment be below 4.3%?",
            "kalshi_resolution_date": "2026-01-15T00:00:00Z",
            "polymarket_resolution_date": "2026-01-20T00:00:00Z",
        }]
        assert len(filter_candidates(candidates)) == 0

    def test_passes_good_pair(self):
        candidates = [{
            "confidence_score": 0.9,
            "kalshi_question": "Will unemployment be above 4.3%?",
            "polymarket_question": "Will U-3 unemployment be above 4.3%?",
            "kalshi_resolution_date": "2026-01-15T00:00:00Z",
            "polymarket_resolution_date": "2026-01-10T00:00:00Z",
        }]
        result = filter_candidates(candidates)
        assert len(result) == 1
        assert result[0]["resolution_gap_days"] == 5
