---
phase: 9
slug: live-vs-backtest-reconciliation
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-17
---

# Phase 9 — Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing) + inline assertions |
| **Config file** | existing pytest config |
| **Quick run command** | `.venv/bin/python -m experiments.run_live_reconciliation --dry-run` |
| **Full suite command** | `.venv/bin/python -m pytest tests/analysis/ -x -q` |
| **Estimated runtime** | ~30 seconds (DB queries + shadow sim, no model training) |

## Sampling Rate

- **After every task commit:** Quick run command
- **After every plan wave:** Full suite
- **Before verify-work:** Full suite green
- **Max feedback latency:** 30 seconds

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | Status |
|---------|------|------|-------------|-----------|-------------------|--------|
| 09-01-01 | 01 | 1 | RECON-01,03,04 | unit+integration | `pytest tests/analysis/test_reconciliation.py -x` | ⬜ pending |
| 09-01-02 | 01 | 1 | RECON-05,06,07,08 | integration | `.venv/bin/python -m experiments.run_live_reconciliation` | ⬜ pending |
| 09-02-01 | 02 | 2 | RECON-09,10 | manual+smoke | Check PAPER_DRAFT.md §5.9 exists with findings | ⬜ pending |

## Wave 0 Requirements

- [ ] `src/analysis/__init__.py` — create analysis package
- [ ] `tests/analysis/__init__.py` + `test_reconciliation.py` — test stubs

*Co-created with implementation tasks (Phase 9 is a leaf with no prior infrastructure).*

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| §5.9 paper quality | RECON-09 | Prose review | Read §5.9, check caveats present |

## Exceptions

- Wave 0 artifacts co-created with implementation (leaf phase, no prior test infra)
- Oil category absent from live data — paper must acknowledge, not a test failure

**Approval:** approved 2026-04-17
