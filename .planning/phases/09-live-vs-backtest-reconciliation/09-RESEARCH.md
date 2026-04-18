# Phase 9: Live vs Backtest Reconciliation - Research

**Researched:** 2026-04-16
**Domain:** Live paper-trading system reconciliation against historical backtest; SQLite position store, Parquet bar data, Python analysis pipeline
**Confidence:** HIGH — all findings derived from direct codebase inspection of the actual running system

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| RECON-01 | New `src/analysis/` subpackage with `reconciliation.py` module | Architecture section: new package pattern established; no src/analysis exists yet |
| RECON-02 | Reconciliation window April 11–25; exclude force_close_schema_fix | Confirmed: 0 pre-April-11 rows in DB; all 2530 rows are post-fix; exit_reason filter needed |
| RECON-03 | Trade-level pairing on `(pair_id, entry_ts_bucket)` | Pair IDs match between bars.parquet and closed_positions (263/263 overlap); timestamp alignment strategy documented |
| RECON-04 | Single shared fee function (`profit_sim.simulate_profit`) | CRITICAL FINDING: fee model mismatch documented — verify_headline.py uses a different fee function; must decide canonical |
| RECON-05 | Summary comparison table: live P&L vs simulated P&L | Data available: 2530 closed positions, all columns present |
| RECON-06 | Category-level breakdown (oil vs non-oil) | kalshi_ticker column in closed_positions; `derive_category_from_ticker` works; category breakdown documented |
| RECON-07 | Exit-reason attribution table | All exit reasons present in DB: TIME_STOP(1508), RESOLUTION_EXIT(821), MOMENTUM(190), STOP_LOSS(10), TAKE_PROFIT(1) |
| RECON-08 | Acceptance gate: (only_live + only_backtest) / matched_trades < 20% | Backtest uses test.parquet (144 pairs, ends 2026-04-01); live has 7037 pairs — universe mismatch is the #1 gap risk |
| RECON-09 | Paper section 5.9 with findings and paper-trading caveats | Content strategy documented |
| RECON-10 | `experiments/run_live_reconciliation.py` CLI wrapper | Pattern established by existing run_*.py scripts |
</phase_requirements>

---

## Summary

Phase 9 is in a better position than the v1.1 research anticipated. The pair_id schema bug that was flagged as "blocking" has been confirmed fixed: all 2530 closed positions in `positions.db` use content-addressed pair_ids (format: `kxdoge26apr1417b0092-0x5b919435`), zero zombies, zero pre-April-11 entries. The reconciliation window is clean from day one.

The live system has 2530 closed positions spanning April 14–16, 2026. The category breakdown shows crypto (261 trades, +$4.33 total P&L) and inflation (1010 trades, +$1.96) as the dominant profitable categories. Oil is absent from the live dataset because the commodity discovery gap (documented in MEMORY) was fixed on April 11 but WTI and similar contracts have since expired or not been discovered. The "oil is the edge" finding from the backtest cannot be tested in the live window — this must be explicitly called out in paper section 5.9.

The single most important architectural constraint for this phase: `verify_headline.py` uses a fee model where the 2pp fee is deducted from winning trades AND added to losing trades, while `profit_sim.simulate_profit` uses the 2pp only as an entry hurdle with no deduction from P&L. These two functions are NOT equivalent. RECON-04 requires a single shared fee function — the planner must decide which fee model is canonical and ensure the reconciliation module uses it exclusively. Using the wrong one will produce a systematic offset in the comparison table that would mislead readers.

**Primary recommendation:** Use `src.evaluation.profit_sim.simulate_profit` as the canonical fee function (as RECON-04 specifies), and document explicitly in the paper that the backtest P&L numbers in the comparison table may differ from Table 2 (which uses `verify_headline.py`'s fee model) by a predictable, documented amount.

---

## Standard Stack

### Core (already installed — no new installations needed)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `sqlite3` | stdlib | Read `positions.db` via `PositionManager` | Already the storage backend |
| `pandas` | 2.x | DataFrame operations, parquet I/O | Already in stack |
| `numpy` | 1.x | P&L calculations | Already in stack |
| `pytest` | 8.x | Unit tests for reconciliation.py | Already in stack |

### No new libraries needed

The reconciliation module requires: SQLite access (via `PositionManager`), Parquet reading (pandas), feature recomputation (existing `compute_derived_features`), model inference (existing `BasePredictor.load()`), P&L computation (existing `profit_sim.simulate_profit`). All are already installed.

**Installation:** No new packages.

---

## Architecture Patterns

### Recommended Project Structure

```
src/analysis/
├── __init__.py          # empty package marker
└── reconciliation.py    # pure analysis logic, testable without CLI

experiments/
└── run_live_reconciliation.py    # ~40 LOC CLI wrapper

experiments/results/
└── reconciliation/
    ├── summary.json
    ├── per_position.csv
    └── report.md

tests/
└── analysis/
    ├── __init__.py
    └── test_reconciliation.py
```

### Pattern 1: Pure-Logic Module in src/analysis/, CLI Wrapper in experiments/

**What:** The reconciliation logic lives in `src/analysis/reconciliation.py` as pure Python functions (no I/O, no CLI). The experiment script `experiments/run_live_reconciliation.py` orchestrates file I/O, calls the functions, and writes output artifacts.

**When to use:** Any time you need testable analysis logic that operates on two completed artifact sets. This is exactly the same pattern as `src/evaluation/profit_sim.py` (pure) vs `experiments/verify_headline.py` (CLI).

**Why not put it in src/evaluation/:** `src/evaluation/` owns in-loop utilities called thousands of times per training cycle. Reconciliation is a one-shot comparison of two finished artifact sets. Keeping them separate prevents the evaluation module from accumulating analytical debt.

### Pattern 2: PositionManager.get_closed_positions() for DB Access

**What:** Always read `positions.db` through `PositionManager.get_closed_positions()`, never via raw SQLite in analysis code.

**Why:** The `PositionManager` owns the schema contract. If the schema evolves, analysis code gets the fix for free. Direct SQLite access from `src/analysis/` creates a second schema dependency that will drift.

**Code path:**
```python
# Source: src/live/position_manager.py lines 372-377
def get_closed_positions(self) -> list[dict]:
    """Return all closed positions as a list of dicts, newest first."""
    rows = self._conn.execute(
        "SELECT * FROM closed_positions ORDER BY id DESC"
    ).fetchall()
    return [dict(row) for row in rows]
```

### Pattern 3: Backtest Counterfactual via Bars Lookup + Model Replay

**What:** For each closed live position, the reconciliation reconstructs what the backtest would have predicted by (1) reading the bar from `bars.parquet` at `(pair_id, entry_time)`, (2) running `compute_derived_features`, (3) running `BasePredictor.load(lr_path).predict(features)` and `BasePredictor.load(xgb_path).predict(features)`, and (4) computing simulated P&L using `simulate_profit`.

**Key constraint:** Use the same model artifacts that were live at the time of the trade. The deployed models are at `models/deployed/linear_regression.pkl` and `models/deployed/xgboost.pkl`. These are the only models used in the live system since April 11.

### Anti-Patterns to Avoid

- **Using verify_headline.py's `simulate_pnl` function for reconciliation.** That function has a different fee model (fee deducted from P&L, not just threshold). RECON-04 requires `profit_sim.simulate_profit` exclusively.
- **Opening positions.db directly with sqlite3 in analysis code.** Always use `PositionManager.get_closed_positions()`.
- **Comparing live P&L numbers directly to Table 2 numbers.** Table 2 was computed on the historical test.parquet dataset; live data covers a completely different pair universe and time window.
- **Using `derive_category_from_pair_id` for the live positions.** This function returns `"other"` for content-addressed live pair_ids that don't match the old format. Instead, use `derive_category_from_ticker(closed_position["kalshi_ticker"])` — the `kalshi_ticker` column is always populated in `closed_positions`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| DB access | Raw sqlite3 queries in reconciliation.py | `PositionManager.get_closed_positions()` | Schema drift, connection handling |
| P&L calculation | Custom spread × direction formula | `src.evaluation.profit_sim.simulate_profit` | RECON-04 requirement; single source of truth |
| Feature engineering | Manual column computation | `src.features.engineering.compute_derived_features` | 61-column pipeline already tested |
| Category lookup | Regex on pair_id | `src.features.category.derive_category_from_ticker(row["kalshi_ticker"])` | pair_id prefix parsing breaks on content-addressed ids |
| Model inference | Loading pkl files directly | `src.models.base.BasePredictor.load(path).predict(X)` | Correct scaler loading, dtype alignment |

---

## Common Pitfalls

### Pitfall 1: Universe Mismatch Between Live and Backtest
**What goes wrong:** The historical backtest (`test.parquet`) covers 144 pairs with data through April 1, 2026. The live system monitors 7,037 pairs. Most live trades have no matching backtest pair. An "only-live + only-backtest" gap of nearly 100% is guaranteed if you try to join on pair_id across the two systems directly.

**Why it happens:** The backtest universe is frozen at 144 quality-filtered pairs from the historical dataset. The live universe is discovered fresh each cycle with a different quality filter.

**How to avoid:** The reconciliation is NOT a comparison of the same pairs — it is a shadow simulation. For each closed live position, run the live bar through the deployed model and compute what the model would have predicted. This is the "backtest counterfactual" for that specific trade. You are not looking up the live pair_id in test.parquet; you are replaying the bar through the model.

**Acceptance gate (RECON-08):** Applies to the shadow simulation, not to a join against test.parquet. The gap metric should be: what fraction of live closed positions could NOT be replayed (because bars.parquet was missing the relevant bar at entry time)?

### Pitfall 2: Fee Model Mismatch
**What goes wrong:** `verify_headline.py` has its own `simulate_pnl` function that deducts the 2pp fee from winning trades and adds it to losing trades. `profit_sim.simulate_profit` uses the 2pp only as an entry threshold — no deduction from P&L. These produce different absolute P&L numbers for the same predictions.

**Concrete impact:**
- verify_headline.py for a winning trade with actual=0.05: P&L = 0.05 - 0.02 = 0.03
- profit_sim.simulate_profit for same trade: P&L = 0.05 (no deduction)

**How to avoid:** RECON-04 mandates `profit_sim.simulate_profit` for the reconciliation. Document this in the paper: "Reconciliation P&L uses the threshold-only fee model (no explicit cost deduction), which may differ from Table 2 P&L by approximately 2pp per trade."

### Pitfall 3: Wrong Category Function
**What goes wrong:** `derive_category_from_pair_id("kxdoge26apr1417b0092-0x5b919435")` returns `"crypto"` for crypto pairs but `derive_category_from_pair_id("kxhormuznorm26mar17b270101-0xdb7fa6f9")` returns `"other"` because `KXHORMUZNORM` is not in the category rules. `derive_category_from_pair_id` was designed for the training dataset pair_id format.

**Correct approach:** Use `derive_category_from_ticker(closed_position["kalshi_ticker"])` — the `kalshi_ticker` column stores the full Kalshi ticker (e.g., `"KXDOGE-26APR1417-B0.092"`) and `derive_category_from_ticker` handles the full prefix matching correctly.

**Verified result:** Running `derive_category_from_ticker("KXDOGE-26APR1417-B0.092")` returns `"crypto"`. Running `derive_category_from_ticker("KXARMOMINF-26APR14-T2.5")` returns `"inflation"`. The ticker-based function handles all 2530 live positions correctly.

### Pitfall 4: Timestamp Alignment Between ISO String and Unix Integer
**What goes wrong:** `positions.db` stores `entry_time` and `exit_time` as ISO 8601 strings (e.g., `"2026-04-14T13:21:45Z"`). `bars.parquet` stores `timestamp` as Unix epoch integers. Finding the right bar for a position requires converting between these formats.

**How to avoid:** Convert entry_time to Unix timestamp via `pd.Timestamp(entry_time).timestamp()` and then find the nearest bar using `bars[(bars["pair_id"] == pair_id) & (bars["timestamp"] <= entry_ts)].nlargest(1, "timestamp")` — the last bar at or before entry time.

### Pitfall 5: Missing Bars for Resolved Positions
**What goes wrong:** `bars.parquet` accumulates bars from April 7 onward. Some positions in `closed_positions` entered on April 14 on pairs that resolved quickly (RESOLUTION_EXIT, 821 trades). For very short-lived positions (bars_held=0 or 1), bars.parquet may have only one bar for the pair and the "spread at exit" may not be independently retrievable from bars.

**How to avoid:** For RESOLUTION_EXIT positions, the exit_spread is the resolution price. The backtest counterfactual for these positions uses entry bar features → model prediction → realized spread = (exit_spread - entry_spread). All required data is in closed_positions itself; bars.parquet lookup is only needed for the features at entry time.

### Pitfall 6: Oil Category Missing from Live Data
**What goes wrong:** RECON-06 requires oil vs non-oil breakdown. But the live data (April 14–16) has zero oil positions. The category breakdown shows: crypto(261), inflation(1010), gdp(192), other(1033), fed_rates(14), politics_policy(20). No oil, no WTI contracts.

**Why it happens:** The commodity discovery gap (Kalshi 429 + Polymarket shallow pagination) was fixed on April 11. WTI contracts discovered after that date either haven't entered positions yet or have expired. The "oil is the edge" Finding 6 cannot be reproduced on live data.

**How to handle:** RECON-06 should be written as "commodity-enabled categories (crypto, inflation) vs non-commodity" rather than "oil vs non-oil." State explicitly in paper §5.9: "WTI oil contracts were not present in the live trading window (April 14–16) due to the post-fix discovery window timing."

---

## Code Examples

### closed_positions Schema (verified 2026-04-16)
```python
# Columns in closed_positions table (from PRAGMA table_info):
# id INTEGER AUTOINCREMENT
# pair_id TEXT          -- content-addressed, e.g. "kxdoge26apr1417b0092-0x5b919435"
# kalshi_ticker TEXT    -- e.g. "KXDOGE-26APR1417-B0.092"
# direction TEXT        -- "short_spread" or "long_spread"
# entry_spread REAL     -- signed (positive if kalshi > poly)
# entry_time TEXT       -- ISO 8601 UTC, e.g. "2026-04-14T13:21:45Z"
# exit_time TEXT        -- ISO 8601 UTC
# entry_kalshi_price REAL
# entry_poly_price REAL
# exit_spread REAL
# bars_held INTEGER
# realized_pnl REAL     -- direction-aware: short=(entry-exit), long=(exit-entry)
# exit_reason TEXT      -- "TIME_STOP","RESOLUTION_EXIT","MOMENTUM","STOP_LOSS","TAKE_PROFIT"
# tier TEXT             -- "DAILY","WEEKLY","MONTHLY","QUARTERLY","UNKNOWN"
# max_spread REAL
# min_spread REAL
```

### Realized P&L Formula (from position_manager.py lines 271-275)
```python
# Direction-aware realized P&L
if direction == "short_spread":
    realized_pnl = entry_spread - exit_spread
else:  # long_spread
    realized_pnl = exit_spread - entry_spread
```

### Correct Category Lookup for Live Positions
```python
# WRONG: derive_category_from_pair_id returns "other" for most content-addressed IDs
# CORRECT: use kalshi_ticker column
from src.features.category import derive_category_from_ticker

for pos in pm.get_closed_positions():
    category = derive_category_from_ticker(pos["kalshi_ticker"])
```

### Bar Lookup at Position Entry Time
```python
import pandas as pd
from datetime import datetime, timezone

bars = pd.read_parquet("data/live/bars.parquet")

def get_entry_bar(pair_id: str, entry_time_iso: str) -> pd.Series | None:
    """Return the bar row for (pair_id) at or just before entry_time."""
    entry_ts = int(pd.Timestamp(entry_time_iso).timestamp())
    pair_bars = bars[(bars["pair_id"] == pair_id) & (bars["timestamp"] <= entry_ts)]
    if pair_bars.empty:
        return None
    return pair_bars.nlargest(1, "timestamp").iloc[0]
```

### Shadow Simulation Pattern
```python
from src.models.base import BasePredictor
from src.features.engineering import compute_derived_features
from src.evaluation.profit_sim import simulate_profit
import numpy as np

# Load deployed models (same artifacts live system uses)
lr_model = BasePredictor.load("models/deployed/linear_regression.pkl")
xgb_model = BasePredictor.load("models/deployed/xgboost.pkl")

import json
with open("models/deployed/feature_columns.json") as f:
    feature_columns = json.load(f)

def simulate_position(pos: dict, entry_bar: pd.Series) -> dict | None:
    """Compute what backtest would have predicted for a live position."""
    row_df = pd.DataFrame([entry_bar])
    row_df = compute_derived_features(row_df).fillna(0.0)
    # Align to deployed feature columns
    X = row_df[[c for c in feature_columns if c in row_df.columns]]
    if X.empty:
        return None
    lr_pred = float(lr_model.predict(X)[0])
    xgb_pred = float(xgb_model.predict(X)[0])
    avg_pred = (lr_pred + xgb_pred) / 2.0
    actual_spread_change = pos["exit_spread"] - pos["entry_spread"]
    # Use profit_sim as RECON-04 requires (threshold-only fee model)
    result = simulate_profit(
        np.array([avg_pred]),
        np.array([actual_spread_change]),
        threshold=0.02
    )
    return {
        "pair_id": pos["pair_id"],
        "live_pnl": pos["realized_pnl"],
        "sim_pnl": result["total_pnl"],
        "lr_pred": lr_pred,
        "xgb_pred": xgb_pred,
        "avg_pred": avg_pred,
        "actual_change": actual_spread_change,
        "exit_reason": pos["exit_reason"],
        "category": derive_category_from_ticker(pos["kalshi_ticker"]),
    }
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `live_NNNN` index-based pair IDs | Content-addressed `kalshi_norm-0xpoly_prefix` | 2026-04-11 (commit dee9205) | Phase 9 window is clean; no zombie positions |
| pair_mapping.json as canonical lookup | `make_pair_id()` function in `src/live/pair_ids.py` | 2026-04-11 | pair_mapping.json is now stale/obsolete; do not use for reconciliation |
| verify_headline.py fee model (deducts from P&L) | profit_sim.simulate_profit (threshold only) | v1.0 design | Two different fee models exist; RECON-04 picks profit_sim as canonical |

---

## Data Inventory (verified 2026-04-16)

### positions.db (data/live/positions.db)
- **Tables:** `positions` (open), `closed_positions`, `sqlite_sequence`
- **Total closed positions:** 2530
- **Date range:** 2026-04-14 to 2026-04-16 (3 days only — system was restarted or cleaned)
- **Exit reason distribution:** TIME_STOP=1508, RESOLUTION_EXIT=821, MOMENTUM=190, STOP_LOSS=10, TAKE_PROFIT=1
- **Total realized P&L:** +$6.03 (avg +$0.0024/trade)
- **Category breakdown:** crypto(261, +$4.33), inflation(1010, +$1.96), gdp(192, -$0.35), other(1033, +$0.10)
- **Pair_id format:** Content-addressed (post-fix), e.g. `kxdoge26apr1417b0092-0x5b919435`
- **Pair universe:** 263 unique pairs in closed_positions, all present in bars.parquet

### bars.parquet (data/live/bars.parquet)
- **Shape:** 88,671 rows x 61 columns
- **Date range:** 2026-04-07 to 2026-04-16
- **Pair universe:** 7,037 unique pairs
- **Overlap with closed_positions:** 263/263 (100%) — every traded pair has bar history
- **Feature columns:** 57 non-index columns (deployed model uses 54 of these)
- **Notable:** Does NOT have `spread_change_target` — must compute at reconciliation time

### paper_trades_*.jsonl (data/live/)
- **Available files:** paper_trades_2026-04-11 through 2026-04-14 (no 2026-04-15/16 files)
- **Format:** `{timestamp, collection_time, pair_id, model, prediction, direction, threshold, kalshi_price, polymarket_price, spread, trade}`
- **Critical issue:** paper_trades_2026-04-11 uses OLD `live_NNNN` pair_ids; paper_trades_2026-04-14 uses new content-addressed pair_ids
- **Decision:** Do NOT use paper_trades as the canonical source. Use closed_positions for actual trade data. paper_trades records every cycle's signals, not just executed trades; it's too large (87k–321k lines/day) and the pair_id format is inconsistent.
- **Canonical source:** `positions.db / closed_positions` table is the canonical trade record.

### models/deployed/
- `linear_regression.pkl` — deployed LR model
- `xgboost.pkl` — deployed XGBoost model
- `feature_columns.json` — 54 feature names used by deployed models
- **Feature alignment:** deployed model features are a subset of bars.parquet columns (bars has 57 non-index features; deployed uses 54; missing from bars: none — all 54 deployed features exist in bars)

### data/processed/test.parquet
- **Shape:** 1,817 rows x 39 columns
- **Date range:** 2025-12-29 to 2026-04-01 (historical backtest data only)
- **Pair count:** 144 pairs
- **Key point:** This is the HISTORICAL backtest dataset. It ends before the live window begins. There is zero temporal overlap with positions.db. Do not attempt to join these two datasets by pair_id.

---

## Open Questions

1. **Data gap: only 3 days of closed positions (April 14–16)**
   - What we know: positions.db has 2530 records all from April 14–16. The live system ran since April 11 but earlier positions are not in the DB.
   - What's unclear: Were positions from April 11–13 cleared from the DB? Were they in a different DB instance on SCC?
   - Recommendation: Check SCC for the live positions.db before writing the reconciliation. The April 14–16 data is enough for a valid analysis (2530 is substantial), but the paper should state the actual window clearly.

2. **RECON-08 acceptance gate needs reinterpretation**
   - What we know: The gate is "(only_live + only_backtest) / matched_trades < 20%". In this system, "backtest" means shadow simulation, not a lookup in test.parquet.
   - What's unclear: How do we count "only_backtest"? If we're running a shadow simulation for every closed live position, there are no "backtest-only" trades by construction.
   - Recommendation: Reinterpret RECON-08 as: "shadow simulation matched / total closed positions > 80%". A position fails to match if bars.parquet lacks the entry bar. Given 100% bar overlap confirmed above, this gate should be trivially met.

3. **Fee model for paper section 5.9**
   - What we know: Two fee models exist (verify_headline.py vs profit_sim.simulate_profit). RECON-04 mandates profit_sim. But Table 2 in the paper was produced by verify_headline.py.
   - What's unclear: Will the comparison table be confusing if it shows "live P&L: $X vs backtest P&L: $Y" where both are computed with the threshold-only model, but the reader is comparing to Table 2 which uses the deduction model?
   - Recommendation: In paper §5.9, explicitly note: "Shadow-simulation P&L uses the threshold-only fee model, consistent with `profit_sim.simulate_profit`. Table 2 P&L uses a 2pp transaction-cost deduction model. The two are not directly comparable in absolute terms; the reconciliation focuses on directional accuracy and tracking error."

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (already installed) |
| Config file | none (discovered from project root) |
| Quick run command | `.venv/bin/python3 -m pytest tests/analysis/ -x -q` |
| Full suite command | `.venv/bin/python3 -m pytest tests/ --ignore=tests/matching --ignore=tests/models/test_ppo_filtered.py --ignore=tests/models/test_ppo_raw.py --ignore=tests/models/test_trading_env.py -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| RECON-01 | `src/analysis/reconciliation.py` importable, functions present | unit | `.venv/bin/python3 -m pytest tests/analysis/test_reconciliation.py -x` | ❌ Wave 0 |
| RECON-02 | Filter excludes force_close_schema_fix and pre-April-11 | unit | `.venv/bin/python3 -m pytest tests/analysis/test_reconciliation.py::test_window_filter -x` | ❌ Wave 0 |
| RECON-03 | Trade-level pairing returns correct matched/only-live counts | unit | `.venv/bin/python3 -m pytest tests/analysis/test_reconciliation.py::test_pair_trades -x` | ❌ Wave 0 |
| RECON-04 | Shadow simulation uses profit_sim.simulate_profit, not verify_headline's function | unit | `.venv/bin/python3 -m pytest tests/analysis/test_reconciliation.py::test_fee_function_identity -x` | ❌ Wave 0 |
| RECON-05 | Summary comparison dict has required keys | unit | `.venv/bin/python3 -m pytest tests/analysis/test_reconciliation.py::test_summary_schema -x` | ❌ Wave 0 |
| RECON-06 | Category breakdown splits by kalshi_ticker, not pair_id | unit | `.venv/bin/python3 -m pytest tests/analysis/test_reconciliation.py::test_category_breakdown -x` | ❌ Wave 0 |
| RECON-07 | Exit-reason attribution groups all 5 exit reasons | unit | `.venv/bin/python3 -m pytest tests/analysis/test_reconciliation.py::test_exit_reason_attribution -x` | ❌ Wave 0 |
| RECON-08 | Acceptance gate assertion raises on gap >= 20% | unit | `.venv/bin/python3 -m pytest tests/analysis/test_reconciliation.py::test_acceptance_gate -x` | ❌ Wave 0 |
| RECON-09 | Paper section content — manual review | manual-only | n/a | n/a |
| RECON-10 | CLI wrapper runs to completion with --dry-run | smoke | `.venv/bin/python3 experiments/run_live_reconciliation.py --dry-run` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `.venv/bin/python3 -m pytest tests/analysis/ -x -q`
- **Per wave merge:** `.venv/bin/python3 -m pytest tests/ --ignore=tests/matching --ignore=tests/models/test_ppo_filtered.py --ignore=tests/models/test_ppo_raw.py --ignore=tests/models/test_trading_env.py -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/analysis/__init__.py` — package marker
- [ ] `tests/analysis/test_reconciliation.py` — covers RECON-01 through RECON-08
- [ ] `src/analysis/__init__.py` — package marker
- [ ] `src/analysis/reconciliation.py` — main module

---

## Sources

### Primary (HIGH confidence — direct codebase inspection)
- `data/live/positions.db` — schema verified via PRAGMA table_info; 2530 records counted and dated
- `src/live/position_manager.py` — P&L formula at lines 271-275; get_closed_positions at lines 372-377; ExitReason enum at lines 25-33
- `src/live/pair_ids.py` — make_pair_id logic; content-addressed format; documents the April 11 fix
- `src/evaluation/profit_sim.py` — fee model (threshold-only); full function verified lines 30-168
- `experiments/verify_headline.py` — alternative fee model (deduction-based); simulate_pnl at lines 47-63
- `src/features/category.py` — derive_category_from_ticker (correct); derive_category_from_pair_id (returns "other" for live IDs)
- `data/live/bars.parquet` — 88671 rows, 61 columns, 7037 pairs, date range April 7-16
- `data/processed/test.parquet` — 1817 rows, 144 pairs, ends April 1 (no overlap with live window)
- `models/deployed/feature_columns.json` — 54 deployed feature names; verified all present in bars.parquet

### Secondary (MEDIUM confidence — planning documents)
- `.planning/research/PITFALLS.md` — P2 (live-vs-backtest gap), detection code pattern
- `.planning/research/ARCHITECTURE.md` — reconciliation data-flow blueprint
- `.planning/research/FEATURES.md` — C2 table stakes and anti-features
- `.planning/REQUIREMENTS.md` — RECON-01 through RECON-10 acceptance criteria

---

## Metadata

**Confidence breakdown:**
- DB schema and data: HIGH — directly inspected
- Pair_id format: HIGH — confirmed fix; verified against bars.parquet (263/263 overlap)
- Fee model discrepancy: HIGH — read both functions; confirmed they differ
- Category handling: HIGH — ran derive_category_from_ticker against actual DB rows
- Data volume: HIGH — counted directly from DB
- Missing oil category: HIGH — verified category breakdown shows zero oil positions
- Wave 0 test gaps: HIGH — src/analysis does not exist; confirmed no tests/analysis/

**Research date:** 2026-04-16
**Valid until:** 2026-04-27 (submission date) — data is from the live running system which continues to accumulate

**Key date clarification:** The reconciliation window requirement (RECON-02) says April 11–25. The current DB only has data from April 14–16. The planner should note that the window will widen as the system continues running, and the implementation should parameterize the window start/end dates rather than hardcoding them.
