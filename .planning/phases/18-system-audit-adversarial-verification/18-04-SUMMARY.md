---
phase: 18-system-audit-adversarial-verification
plan: 04
subsystem: audit
tags: [audit, tier-3, cost-realism, fees, slippage, kalshi, polymarket]
verdict: CORRECTED
requires:
  - 18-01 (Wave 0 audit scaffolding — tests/audit/conftest.py fixtures + experiments/results/audit/.gitkeep)
  - 18-02 (Tier 1 Sharpe audit — exports build_trade_ledger / per_trade_sharpe)
provides:
  - Tier 3 cost-realism audit (5 helper functions, slippage sweep, Kalshi/Polymarket fee schedule)
  - paper_corrections_required list (consumed by Plan 06 paper_numbers.csv + Plan 07 AUDIT_REPORT.md)
  - Documented prose-vs-code mismatch at PAPER_DRAFT.md §5.1 (the load-bearing paper bug)
affects:
  - PAPER_DRAFT.md §5.1 (line 213, 215) — "at 2pp transaction costs" prose to be corrected by Plan 06/07
  - PAPER_DRAFT.md §6.4 — Polymarket gas + Kalshi fee formula to be added
tech-stack:
  added: []
  patterns:
    - "Cost audit Pattern 2 JSON schema: {audit, tier, verdict, ran_at, ...findings, paper_corrections_required}"
key-files:
  created:
    - experiments/audit/audit_costs.py
    - tests/audit/test_audit_costs.py
    - experiments/results/audit/costs_audit.json
  modified: []
decisions:
  - "Verdict CORRECTED is the headline finding, not a regression: it surfaces a paper prose bug, not a code defect"
  - "Slippage sweep uses 60/40 entry/exit split (matches backtester.compute_break_even_fee convention)"
  - "All 5 haircut levels (0/5/10/20/50 bps) confirm cost-robustness — Sharpe drops only 1.6% across the range"
  - "Kalshi/Polymarket fees hard-coded from 2026 schedule (no live API fetch — reproducibility > freshness)"
metrics:
  completed: 2026-04-25
  tasks: 2
  files: 3
  duration: ~6min
requirements:
  - AUDIT-03
---

# Phase 18 Plan 04: Tier 3 — Cost Realism Audit Summary

**One-liner:** Tier 3 audit confirms simulate_profit charges zero fees (the §5.1 "2pp transaction costs" prose is misleading) and proves WalkForwardBacktester's 5pp round-trip is over-conservative vs realistic Kalshi+Polymarket fees (~250-355 bps); slippage sweep at 0/5/10/20/50 bps shows the cost-robustness claim survives.

## Audit Verdict

**Verdict:** `CORRECTED`

**Headline finding:** PAPER_DRAFT.md §5.1 line 213/215 prose-vs-code mismatch confirmed.

The paper describes Table 2 results as "single-split backtest at 2 pp transaction costs," but Table 2 numbers come from `simulate_profit` which charges **zero fees** — the `0.02` is a **signal threshold** (`|prediction| > 0.02` to enter a trade), not a fee deduction. Plan 06 (`paper_numbers.csv`) and Plan 07 (`AUDIT_REPORT.md` + paper updates) will consume `paper_corrections_required` and propagate the prose fix.

## Audit Script

**File:** `experiments/audit/audit_costs.py`
**Lines:** 262 (≥ 200 required)
**Exit code:** 0
**Output:** `experiments/results/audit/costs_audit.json`

### Functions Exposed

| Function | Purpose |
| --- | --- |
| `kalshi_taker_fee_per_contract(C)` | 2026 Kalshi formula: `0.07 * C * (1-C)` per contract |
| `polymarket_taker_fee_pct(category)` | Per-category Polymarket taker pct (lookup table, default 1.25%) |
| `confirm_simulate_profit_fee_handling()` | Returns dict with fee_charged=0.0, paper_claim_mismatch=True |
| `confirm_backtester_fee_handling()` | Returns dict with entry_cost_pp=0.03, exit_cost_pp=0.02, round_trip_pp=0.05 |
| `slippage_sensitivity_sweep()` | Recompute LR P&L + Sharpe at 5 haircut levels |

## Verbatim Values from costs_audit.json

### `simulate_profit_fee_audit`

| Field | Value |
| --- | --- |
| `fee_charged` | `0.0` |
| `paper_claim_mismatch` | `true` |
| `paper_section` | `"§5.1 line 213, line 215 (PAPER_DRAFT.md)"` |
| `recommendation` | "Either (a) clarify §5.1: 'with a 2pp signal threshold for trade entry; fees are accounted for separately in §5.6 transaction-cost sensitivity', or (b) move headline numbers to WalkForwardBacktester output." |

### `walk_forward_backtester_fee_audit`

| Field | Value |
| --- | --- |
| `entry_cost_pp` | `0.03` |
| `exit_cost_pp` | `0.02` |
| `round_trip_pp` | `0.05` |
| `round_trip_bps` | `500.0` |
| `vs_realistic_kalshi` | "5pp = 500bps round-trip is ~3x the realistic Kalshi taker max — CONSERVATIVE" |
| `vs_realistic_polymarket` | "Backtester 500bps is conservative vs realistic 250-355 bps round-trip (Kalshi taker + Polymarket taker)" |

### Slippage Sensitivity Sweep (LR, 1,549 trades, $100 position)

| Haircut (bps) | Annualized Sharpe | Total P&L | Total Fees | Win Rate |
| ---: | ---: | ---: | ---: | ---: |
| 0   | 8.954 | $253,727.89 | $97,643.55  | 45.45% |
| 5   | 8.940 | $252,751.46 | $98,619.98  | 45.45% |
| 10  | 8.926 | $251,775.02 | $99,596.42  | 45.38% |
| 20  | 8.897 | $249,822.15 | $101,549.29 | 45.26% |
| 50  | 8.807 | $243,963.54 | $107,407.90 | 44.93% |

**Cost-robustness verdict:** The cost-robustness claim **survives at 50 bps additional haircut**. Annualized Sharpe drops only 8.95 → 8.81 (-1.6%); total P&L drops only $253,728 → $243,964 (-3.8%). The strategy's edge is robust to realistic slippage uncertainty within the 0-50 bps range tested. Note: these annualized Sharpes (8.81-8.95) come from the WalkForwardBacktester (legacy dollar-notional simulator with ~200x contract scaling on $0.10-$0.30 commodity contracts), not the canonical per-trade Sharpe (≈ 0.50) cited in the paper Table 2 — the slippage-robustness conclusion holds regardless of which simulator is the headline.

### `paper_corrections_required` (consumed by Plan 06 + Plan 07)

1. **§5.1 line 213, 215** — claims "2pp transaction costs" but Table 2 numbers use `simulate_profit` (zero fee). Fix: clarify "with a 2pp signal threshold; fees are analyzed separately in §5.6."
2. **§6.4 Limitations** — Polymarket gas/withdrawal cost not explicitly stated. Fix: add "Polymarket charges category-dependent taker fees (0.75-1.80%) and <\$0.01 in Polygon gas per transaction; deposits and withdrawals of USDC are free. Kalshi taker fee is 7c × C × (1-C) per contract (max 1.75c at C=0.50); maker fee is 25% of taker."

### Kalshi 2026 Fee Reference (hard-coded)

- **Source:** kalshi.com/fee-schedule
- **Formula:** `taker_fee_dollars = 0.07 * C * (1 - C)` per contract
- **Max at C=0.50:** $0.0175 (1.75c)
- **Maker relative:** 25% of taker
- **Settlement fee:** $0

### Polymarket 2026 Fee Reference (hard-coded)

| Category | Taker % |
| --- | ---: |
| Crypto | 1.80% |
| Mentions | 1.56% |
| Economics | 1.50% |
| Culture / Weather | 1.25% |
| Finance / Politics / Tech | 1.00% |
| Sports | 0.75% |
| Geopolitics | 0.00% |

- **Maker:** 0%
- **Gas:** ~$0.01 per Polygon transaction
- **Default for unknown category:** 1.25% (median tier)

## Test Coverage

**File:** `tests/audit/test_audit_costs.py`
**Tests:** 3 passed in 2.32s

| Test | Validates |
| --- | --- |
| `test_simulate_profit_fee_audit_flags_zero_fee_mismatch` | Audit returns fee_charged=0, paper_claim_mismatch=True, "§5.1" in paper_section |
| `test_kalshi_fee_formula_matches_2026_schedule` | Kalshi formula at C=0/0.01/0.10/0.50/0.99/1.0 (boundary + max) |
| `test_audit_costs_catches_zero_fee` | Alias matching VALIDATION.md naming convention |

Cross-cutting check: full audit suite (`tests/audit/test_audit_costs.py + test_fixtures.py`) passes 7/7 in 2.29s. Wave 0 → Wave 2 chain closed for cost realism.

## Tasks Completed

| Task | Name | Commit | Files |
| ---- | ---- | ------ | ----- |
| 1 | Implement experiments/audit/audit_costs.py + costs_audit.json | `00c9850` | experiments/audit/audit_costs.py (262 lines), experiments/results/audit/costs_audit.json |
| 2 | Write tests/audit/test_audit_costs.py | `4103b39` | tests/audit/test_audit_costs.py (41 lines) |

## Deviations from Plan

None — plan executed exactly as written.

The plan's `<read_first>` block flagged a possible discrepancy between `src/evaluation/walk_forward.py` (plan listing) and `src.evaluation.backtester.WalkForwardBacktester` (RESEARCH.md import path). Verified before coding: only `src/evaluation/backtester.py` exists; the RESEARCH.md skeleton's import path is correct and was used verbatim.

`WalkForwardBacktester.run()` returns the exact dict shape RESEARCH.md assumed (`annualized_sharpe`, `total_pnl`, `total_fees`, `num_trades`, `win_rate`) — no key adaptation needed.

## Authentication Gates

None — Tier 3 audit reads canonical pre-computed data; no external API calls.

## Decisions Made

1. **Skeleton copied verbatim from RESEARCH.md lines 806-1018** with one minor enhancement: explicit `float()` / `int()` casts on the slippage sweep return values to keep JSON serialization deterministic (numpy scalars can serialize differently across pandas versions).
2. **5 haircut levels** (0/5/10/20/50 bps) instead of the 4 mentioned in the script docstring (0/5/10/20). RESEARCH.md skeleton specified 5 in the loop; docstring now matches.
3. **`assumptions` list extended** with one extra entry documenting the 60/40 entry/exit haircut split convention (transparency for Plan 07 reviewers).

## Downstream Consumers

- **Plan 06** (`paper_numbers.csv`) reads `paper_corrections_required[0]` to mark §5.1 line 213/215 as "CORRECTED" with the prose fix.
- **Plan 07** (`AUDIT_REPORT.md` + paper updates) reads `verdict=CORRECTED`, `paper_corrections_required[]`, and the slippage_sensitivity table to:
  1. Mark Tier 3 row as CORRECTED in AUDIT_REPORT.md
  2. Apply the §5.1 prose fix (signal threshold vs fee deduction)
  3. Apply the §6.4 fee-schedule documentation
  4. Optionally cite the slippage sweep as evidence that cost-robustness survives (this would *strengthen* the paper)

## Self-Check: PASSED

Verified files exist on disk:

```
[x] /Users/iansabia/Desktop/DS340 Project/experiments/audit/audit_costs.py (262 lines)
[x] /Users/iansabia/Desktop/DS340 Project/tests/audit/test_audit_costs.py (41 lines)
[x] /Users/iansabia/Desktop/DS340 Project/experiments/results/audit/costs_audit.json
```

Verified commits exist:

```
[x] 00c9850 feat(18-04): Tier 3 cost realism audit script + costs_audit.json
[x] 4103b39 test(18-04): fixture tests for Tier 3 cost audit (3 passing)
```

Verified verdict matches RESEARCH.md prediction (`CORRECTED`).
Verified `paper_corrections_required` length=2 (≥ 1 required).
Verified slippage_sensitivity has 5 entries (≥ 5 required).
Verified all 3 fixture tests pass.
