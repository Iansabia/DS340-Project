"""Tests for the market_discovery upsert logic.

Network fetches (fetch_active_kalshi_markets, fetch_active_poly_markets)
and semantic matching (match_markets) are integration-heavy and not
tested here. The unit tests cover the pure upsert + dedup behaviour
that keeps active_matches.json stable across discovery runs.
"""
from __future__ import annotations

import pytest

from src.live.market_discovery import (
    _is_matchable_kalshi_market,
    _kalshi_mid,
    make_match_key,
    upsert_active_matches,
)


class TestKalshiMid:
    def test_yes_bid_ask_dollars_strings(self):
        m = {"yes_bid_dollars": "0.4500", "yes_ask_dollars": "0.4600"}
        assert _kalshi_mid(m) == pytest.approx(0.455)

    def test_numeric_yes_bid_ask(self):
        m = {"yes_bid": 0.30, "yes_ask": 0.40}
        assert _kalshi_mid(m) == pytest.approx(0.35)

    def test_last_price_fallback(self):
        m = {"last_price_dollars": "0.7500"}
        assert _kalshi_mid(m) == pytest.approx(0.75)

    def test_empty_returns_zero(self):
        assert _kalshi_mid({}) == 0.0

    def test_zero_bid_ask_falls_back(self):
        # Fresh market with no quotes but a last trade
        m = {
            "yes_bid_dollars": "0.0000",
            "yes_ask_dollars": "0.0000",
            "last_price_dollars": "0.65",
        }
        assert _kalshi_mid(m) == pytest.approx(0.65)


class TestIsMatchableKalshiMarket:
    def test_accepts_simple_title(self):
        m = {
            "ticker": "KXWTI-26APR08-T107.99",
            "title": "Will the WTI front-month settle price be above $107.99?",
        }
        assert _is_matchable_kalshi_market(m) is True

    def test_rejects_kxmve_parlay(self):
        m = {
            "ticker": "KXMVECROSSCATEGORY-S12345-ABCDE",
            "title": "yes Denver,no Over 240.5,yes Xander Schauffele",
        }
        assert _is_matchable_kalshi_market(m) is False

    def test_rejects_empty_title(self):
        m = {"ticker": "KXWTI-26APR08-T107.99", "title": ""}
        assert _is_matchable_kalshi_market(m) is False

    def test_rejects_multi_leg_comma_title(self):
        m = {
            "ticker": "KXFOO-BAR",
            "title": "yes A,no B,yes C",
        }
        assert _is_matchable_kalshi_market(m) is False


class TestMakeMatchKey:
    def test_stable_key_same_inputs(self):
        m = {"kalshi_ticker": "KXWTI-26APR08-T107.99", "poly_id": "123456"}
        assert make_match_key(m) == make_match_key(m)

    def test_different_tickers_produce_different_keys(self):
        a = {"kalshi_ticker": "KXWTI-26APR08-T107.99", "poly_id": "123456"}
        b = {"kalshi_ticker": "KXWTI-26APR09-T107.99", "poly_id": "123456"}
        assert make_match_key(a) != make_match_key(b)

    def test_different_poly_ids_produce_different_keys(self):
        a = {"kalshi_ticker": "KXWTI-26APR08-T107.99", "poly_id": "111"}
        b = {"kalshi_ticker": "KXWTI-26APR08-T107.99", "poly_id": "222"}
        assert make_match_key(a) != make_match_key(b)


class TestUpsertActiveMatches:
    """Upsert invariants:

    1. Existing pair_ids NEVER change (collectors rely on array position).
    2. New matches are appended at the end with fresh pair_ids.
    3. Duplicates (same kalshi_ticker + poly_id) are collapsed to one entry
       and take updated price/volume/similarity fields from the new entry.
    4. Each match tracks a `discovered_at` timestamp. Existing entries
       keep their original discovery time; new entries get the fresh one.
    5. Each match tracks a `last_seen` timestamp that updates on every
       upsert where the match appears in the new batch.
    """

    def test_append_new_matches_to_empty(self):
        existing = []
        new = [
            {"kalshi_ticker": "KXA-1", "poly_id": "A", "similarity": 0.9, "spread": 0.1},
            {"kalshi_ticker": "KXB-2", "poly_id": "B", "similarity": 0.8, "spread": 0.2},
        ]
        merged, stats = upsert_active_matches(existing, new, now_ts=1000)
        assert len(merged) == 2
        assert merged[0]["kalshi_ticker"] == "KXA-1"
        assert merged[0]["discovered_at"] == 1000
        assert merged[0]["last_seen"] == 1000
        assert stats["added"] == 2
        assert stats["updated"] == 0

    def test_existing_matches_preserve_index(self):
        """Stability: existing pairs at index N must stay at index N."""
        existing = [
            {"kalshi_ticker": "KXA-1", "poly_id": "A", "similarity": 0.9, "discovered_at": 500},
            {"kalshi_ticker": "KXB-2", "poly_id": "B", "similarity": 0.8, "discovered_at": 500},
        ]
        # Fresh discovery: both still found + one brand new.
        new = [
            {"kalshi_ticker": "KXA-1", "poly_id": "A", "similarity": 0.91, "spread": 0.11},
            {"kalshi_ticker": "KXB-2", "poly_id": "B", "similarity": 0.82, "spread": 0.22},
            {"kalshi_ticker": "KXC-3", "poly_id": "C", "similarity": 0.95, "spread": 0.3},
        ]
        merged, stats = upsert_active_matches(existing, new, now_ts=2000)
        assert len(merged) == 3
        # Order preserved
        assert merged[0]["kalshi_ticker"] == "KXA-1"
        assert merged[1]["kalshi_ticker"] == "KXB-2"
        assert merged[2]["kalshi_ticker"] == "KXC-3"
        # Old discovered_at preserved
        assert merged[0]["discovered_at"] == 500
        # New one gets current ts
        assert merged[2]["discovered_at"] == 2000
        # last_seen updated on all three
        assert merged[0]["last_seen"] == 2000
        assert merged[2]["last_seen"] == 2000
        assert stats["added"] == 1
        assert stats["updated"] == 2

    def test_updates_prices_on_existing(self):
        """When an existing pair is rediscovered, numeric fields refresh."""
        existing = [
            {"kalshi_ticker": "KXA-1", "poly_id": "A", "similarity": 0.9,
             "kalshi_mid": 0.50, "poly_price": 0.40, "spread": 0.10,
             "discovered_at": 500},
        ]
        new = [
            {"kalshi_ticker": "KXA-1", "poly_id": "A", "similarity": 0.92,
             "kalshi_mid": 0.60, "poly_price": 0.35, "spread": 0.25},
        ]
        merged, _ = upsert_active_matches(existing, new, now_ts=1000)
        assert len(merged) == 1
        assert merged[0]["kalshi_mid"] == 0.60
        assert merged[0]["poly_price"] == 0.35
        assert merged[0]["spread"] == 0.25
        assert merged[0]["similarity"] == 0.92
        # Preserved
        assert merged[0]["discovered_at"] == 500

    def test_missing_from_new_marks_as_not_seen(self):
        """Pairs not in the new batch keep their old last_seen, not the fresh one."""
        existing = [
            {"kalshi_ticker": "KXA-1", "poly_id": "A", "similarity": 0.9,
             "discovered_at": 500, "last_seen": 500},
            {"kalshi_ticker": "KXB-2", "poly_id": "B", "similarity": 0.8,
             "discovered_at": 500, "last_seen": 500},
        ]
        # Only KXA-1 was rediscovered this run
        new = [
            {"kalshi_ticker": "KXA-1", "poly_id": "A", "similarity": 0.9},
        ]
        merged, stats = upsert_active_matches(existing, new, now_ts=2000)
        assert len(merged) == 2
        # KXA-1 refreshed
        assert merged[0]["last_seen"] == 2000
        # KXB-2 stale — still at 500
        assert merged[1]["last_seen"] == 500
        assert stats["stale"] == 1

    def test_dedup_within_new_batch(self):
        """If the new batch has duplicates, only one entry is kept."""
        existing = []
        new = [
            {"kalshi_ticker": "KXA-1", "poly_id": "A", "similarity": 0.9},
            {"kalshi_ticker": "KXA-1", "poly_id": "A", "similarity": 0.85},  # dup
            {"kalshi_ticker": "KXB-2", "poly_id": "B", "similarity": 0.8},
        ]
        merged, stats = upsert_active_matches(existing, new, now_ts=1000)
        assert len(merged) == 2
        # Later duplicate wins (more recent data)
        assert merged[0]["similarity"] == 0.85
