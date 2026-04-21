---
phase: 09-live-vs-backtest-reconciliation
plan: "02"
subsystem: analysis
tags: [reconciliation, cli-wrapper, paper-section, live-deployment]
dependency_graph:
  requires:
    - src/analysis/reconciliation.py (all 6 public functions)
    - experiments/results/reconciliation/summary.json (canonical numbers)
    - experiments/results/reconciliation/report.md (structured findings)
  provides:
    - experiments/run_live_reconciliation.py (CLI wrapper)
    - PAPER_DRAFT.md §5.9 (live vs backtest reconciliation section)
  affects:
    - Paper section 5.9 (live deployment evidence — now written)
tech_stack:
  added: []
  patterns:
    - CLI wrapper pattern: argparse + --dry-run flag, delegates all logic to src/analysis/
    - No inline analysis logic in experiments/ scripts
key_files:
  created:
    - experiments/run_live_reconciliation.py
  modified:
    - PAPER_DRAFT.md (added §5.9)
decisions:
  - "CLI wrapper delegates 100% of analysis to src/analysis/reconciliation.py — no inline logic"
  - "Section 5.9 numbers sourced exclusively from experiments/results/reconciliation/summary.json"
  - "Fee model disclaimer mandatory in §5.9: threshold-only (reconciliation) vs 2pp deduction (Table 2)"
  - "Oil absence acknowledged in §5.9 as explicit limitation on live validation of Finding 6"
metrics:
  duration: "5 minutes"
  completed: "2026-04-21"
  tasks_completed: 2
  files_created: 1
  files_modified: 1
---

# Phase 09 Plan 02: CLI Wrapper and Paper §5.9 Summary

**One-liner:** CLI wrapper over reconciliation module (--dry-run confirmed) and paper §5.9 written with actual numbers from summary.json, including mandatory fee disclaimer and oil absence acknowledgment.

## What Was Built

- `experiments/run_live_reconciliation.py` — ~65 LOC CLI wrapper with --db, --bars, --models-dir, --window-start, --dry-run arguments. All reconciliation logic delegated to `src/analysis/reconciliation.py`. Dry-run mode confirmed: prints "Loaded 2530 positions in window. --dry-run: no simulation run." and exits 0.

- `PAPER_DRAFT.md §5.9` — Full reconciliation section written with numbers from `experiments/results/reconciliation/summary.json`:
  - Opening paragraph: live paper-trading deployment context
  - Data window and coverage table (2530/2530 matched, 100%, gate PASSED)
  - Fee model disclaimer (threshold-only vs 2pp deduction — not directly comparable)
  - Summary comparison table (live +$6.03, sim -$6.03, tracking error +$12.06)
  - Category breakdown table (6 categories, no oil row)
  - Exit-reason attribution table (5 exit reasons)
  - 3 bullet findings (directional anti-correlation, crypto/inflation dominance, resolution/time exits)
  - WTI oil absence acknowledgment (explicit limitation)
  - Paper-trading caveats (no slippage, no partial fills, notional capital)

## Key Findings Reported in §5.9

The section documents the directional anti-correlation finding: deployed models capture mean-reversion in spread space, while live strategy profitability comes from spread-magnitude entry thresholding, not directional model alignment. This is a transparency-of-method finding that contextualizes why the reconciliation P&L is inverted.

## Deviations from Plan

None — plan executed exactly as written.

## Checkpoint Status

Task 2 is marked `type="checkpoint:human-verify"`. The section has been written and committed. The checkpoint requires human review of §5.9 for accuracy, completeness, and number correctness before the plan is formally closed.

## Self-Check: PASSED

- `experiments/run_live_reconciliation.py` exists: confirmed
- `PAPER_DRAFT.md` contains "5.9": confirmed (line 375)
- `PAPER_DRAFT.md` contains "threshold-only": confirmed (line 388)
- `PAPER_DRAFT.md` contains "WTI oil contracts were not present": confirmed (line 428)
- Task 1 commit 86fc0a9: confirmed
- Task 2 commit 676c6b5: confirmed
