"""TDD tests for PositionManager with SQLite persistence.

Tests position lifecycle (open/update/close), SQLite persistence across
restarts, P&L computation for both directions, spread history tracking,
and JSONL backup logging.

Exit rule tests (test_exit_rule_*) are in the second TDD batch.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import pytest

from src.live.position_manager import ExitReason, Position, PositionManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path(tmp_path):
    """Provide a temporary SQLite database path."""
    return tmp_path / "test_positions.db"


@pytest.fixture
def jsonl_path(tmp_path):
    """Provide a temporary JSONL history path."""
    return tmp_path / "test_history.jsonl"


@pytest.fixture
def pm(db_path, jsonl_path):
    """Create a fresh PositionManager."""
    return PositionManager(db_path=str(db_path), history_jsonl_path=str(jsonl_path))


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------

def test_create_tables(db_path, jsonl_path):
    """PositionManager creates both tables on init."""
    pm = PositionManager(db_path=str(db_path), history_jsonl_path=str(jsonl_path))
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    assert "closed_positions" in tables
    assert "positions" in tables


# ---------------------------------------------------------------------------
# Open position
# ---------------------------------------------------------------------------

def test_open_position(pm):
    """open_position inserts a row with all fields."""
    pm.open_position(
        pair_id="pair_001",
        kalshi_ticker="KXWTI-26APR08-T105.99",
        direction="short_spread",
        entry_spread=0.50,
        kalshi_price=0.80,
        poly_price=0.30,
        tier="DAILY",
        bar_interval_seconds=14400,
        resolution_date="2026-04-10T23:59:00Z",
    )
    positions = pm.get_open_positions()
    assert "pair_001" in positions
    pos = positions["pair_001"]
    assert pos.pair_id == "pair_001"
    assert pos.kalshi_ticker == "KXWTI-26APR08-T105.99"
    assert pos.direction == "short_spread"
    assert pos.entry_spread == 0.50
    assert pos.entry_kalshi_price == 0.80
    assert pos.entry_poly_price == 0.30
    assert pos.tier == "DAILY"
    assert pos.bar_interval_seconds == 14400
    assert pos.resolution_date == "2026-04-10T23:59:00Z"
    assert pos.bars_held == 0
    assert pos.current_spread == 0.50
    assert pos.unrealized_pnl == 0.0


# ---------------------------------------------------------------------------
# Update position
# ---------------------------------------------------------------------------

def test_update_position(pm):
    """update_position updates spread, increments bars_held, tracks min/max."""
    pm.open_position(
        pair_id="pair_002",
        kalshi_ticker="TICKER",
        direction="short_spread",
        entry_spread=0.50,
        kalshi_price=0.80,
        poly_price=0.30,
        tier="WEEKLY",
        bar_interval_seconds=3600,
    )
    pm.update_position("pair_002", current_spread=0.40)
    pos = pm.get_open_positions()["pair_002"]
    assert pos.bars_held == 1
    assert pos.current_spread == 0.40
    assert pos.min_spread_since_entry == 0.40
    assert pos.max_spread_since_entry == 0.50

    pm.update_position("pair_002", current_spread=0.60)
    pos = pm.get_open_positions()["pair_002"]
    assert pos.bars_held == 2
    assert pos.current_spread == 0.60
    assert pos.max_spread_since_entry == 0.60
    assert pos.min_spread_since_entry == 0.40


def test_update_spread_history(pm):
    """spread_history JSON array grows with each update."""
    pm.open_position(
        pair_id="pair_sh",
        kalshi_ticker="T",
        direction="short_spread",
        entry_spread=0.50,
        kalshi_price=0.80,
        poly_price=0.30,
        tier="DAILY",
        bar_interval_seconds=14400,
    )
    for spread in [0.48, 0.45, 0.42]:
        pm.update_position("pair_sh", current_spread=spread)

    pos = pm.get_open_positions()["pair_sh"]
    history = json.loads(pos.spread_history)
    assert history == [0.48, 0.45, 0.42]


# ---------------------------------------------------------------------------
# Unrealized P&L
# ---------------------------------------------------------------------------

def test_unrealized_pnl_short_spread(pm):
    """short_spread: unrealized_pnl = entry - current. entry=0.50, current=0.30 -> +0.20"""
    pm.open_position(
        pair_id="pnl_short",
        kalshi_ticker="T",
        direction="short_spread",
        entry_spread=0.50,
        kalshi_price=0.80,
        poly_price=0.30,
        tier="DAILY",
        bar_interval_seconds=14400,
    )
    pm.update_position("pnl_short", current_spread=0.30)
    pos = pm.get_open_positions()["pnl_short"]
    assert abs(pos.unrealized_pnl - 0.20) < 1e-9


def test_unrealized_pnl_long_spread(pm):
    """long_spread: unrealized_pnl = current - entry. entry=-0.20, current=-0.10 -> +0.10"""
    pm.open_position(
        pair_id="pnl_long",
        kalshi_ticker="T",
        direction="long_spread",
        entry_spread=-0.20,
        kalshi_price=0.30,
        poly_price=0.50,
        tier="DAILY",
        bar_interval_seconds=14400,
    )
    pm.update_position("pnl_long", current_spread=-0.10)
    pos = pm.get_open_positions()["pnl_long"]
    assert abs(pos.unrealized_pnl - 0.10) < 1e-9


# ---------------------------------------------------------------------------
# Persistence across restarts
# ---------------------------------------------------------------------------

def test_persistence(db_path, jsonl_path):
    """Position persists: new PositionManager on same DB sees existing positions."""
    pm1 = PositionManager(db_path=str(db_path), history_jsonl_path=str(jsonl_path))
    pm1.open_position(
        pair_id="persist_test",
        kalshi_ticker="T",
        direction="short_spread",
        entry_spread=0.50,
        kalshi_price=0.80,
        poly_price=0.30,
        tier="WEEKLY",
        bar_interval_seconds=3600,
    )

    # Create a new PositionManager on the same DB
    pm2 = PositionManager(db_path=str(db_path), history_jsonl_path=str(jsonl_path))
    positions = pm2.get_open_positions()
    assert "persist_test" in positions
    assert positions["persist_test"].entry_spread == 0.50


# ---------------------------------------------------------------------------
# Close position
# ---------------------------------------------------------------------------

def test_close_position(pm):
    """close_position removes from positions, inserts into closed_positions."""
    pm.open_position(
        pair_id="close_me",
        kalshi_ticker="T",
        direction="short_spread",
        entry_spread=0.50,
        kalshi_price=0.80,
        poly_price=0.30,
        tier="DAILY",
        bar_interval_seconds=14400,
    )
    pm.update_position("close_me", current_spread=0.30)
    pm.close_position(
        pair_id="close_me",
        reason=ExitReason.TAKE_PROFIT,
        exit_spread=0.30,
        exit_time="2026-04-09T12:00:00Z",
    )
    assert not pm.has_position("close_me")
    closed = pm.get_closed_positions()
    assert len(closed) == 1
    assert closed[0]["pair_id"] == "close_me"
    assert closed[0]["exit_reason"] == "TAKE_PROFIT"


def test_close_position_pnl_short(pm):
    """short_spread close: entry=0.50, exit=0.20 -> pnl=+0.30"""
    pm.open_position(
        pair_id="pnl_close_s",
        kalshi_ticker="T",
        direction="short_spread",
        entry_spread=0.50,
        kalshi_price=0.80,
        poly_price=0.30,
        tier="DAILY",
        bar_interval_seconds=14400,
    )
    pm.close_position(
        pair_id="pnl_close_s",
        reason=ExitReason.MANUAL,
        exit_spread=0.20,
        exit_time="2026-04-09T12:00:00Z",
    )
    closed = pm.get_closed_positions()
    assert abs(closed[0]["realized_pnl"] - 0.30) < 1e-9


def test_close_position_pnl_long(pm):
    """long_spread close: entry=-0.30, exit=-0.10 -> pnl=+0.20"""
    pm.open_position(
        pair_id="pnl_close_l",
        kalshi_ticker="T",
        direction="long_spread",
        entry_spread=-0.30,
        kalshi_price=0.30,
        poly_price=0.60,
        tier="DAILY",
        bar_interval_seconds=14400,
    )
    pm.close_position(
        pair_id="pnl_close_l",
        reason=ExitReason.MANUAL,
        exit_spread=-0.10,
        exit_time="2026-04-09T12:00:00Z",
    )
    closed = pm.get_closed_positions()
    assert abs(closed[0]["realized_pnl"] - 0.20) < 1e-9


# ---------------------------------------------------------------------------
# has_position
# ---------------------------------------------------------------------------

def test_has_position(pm):
    """has_position returns True when open, False after close."""
    pm.open_position(
        pair_id="has_test",
        kalshi_ticker="T",
        direction="short_spread",
        entry_spread=0.50,
        kalshi_price=0.80,
        poly_price=0.30,
        tier="DAILY",
        bar_interval_seconds=14400,
    )
    assert pm.has_position("has_test") is True
    pm.close_position(
        pair_id="has_test",
        reason=ExitReason.MANUAL,
        exit_spread=0.40,
        exit_time="2026-04-09T12:00:00Z",
    )
    assert pm.has_position("has_test") is False


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

def test_get_open_positions_empty(pm):
    """get_open_positions returns empty dict initially."""
    assert pm.get_open_positions() == {}


# ---------------------------------------------------------------------------
# JSONL backup
# ---------------------------------------------------------------------------

def test_jsonl_backup(pm, jsonl_path):
    """Closing a position appends to position_history.jsonl."""
    pm.open_position(
        pair_id="jsonl_test",
        kalshi_ticker="T",
        direction="short_spread",
        entry_spread=0.50,
        kalshi_price=0.80,
        poly_price=0.30,
        tier="DAILY",
        bar_interval_seconds=14400,
    )
    pm.close_position(
        pair_id="jsonl_test",
        reason=ExitReason.TAKE_PROFIT,
        exit_spread=0.20,
        exit_time="2026-04-09T12:00:00Z",
    )
    assert jsonl_path.exists()
    lines = jsonl_path.read_text().strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["pair_id"] == "jsonl_test"
    assert record["exit_reason"] == "TAKE_PROFIT"
    assert abs(record["realized_pnl"] - 0.30) < 1e-9
