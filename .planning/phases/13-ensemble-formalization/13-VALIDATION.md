---
phase: 13
slug: ensemble-formalization
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-23
---

# Phase 13 — Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest + inline assertions |
| **Config file** | existing pytest config |
| **Quick run command** | `.venv/bin/python -m pytest tests/models/test_ensemble.py -x -q` |
| **Full suite command** | `.venv/bin/python -m experiments.run_ensemble_sweep` |
| **Estimated runtime** | <30s for tests; ~10 min for full sweep |

## Sampling Rate

- **After every task commit:** Quick run
- **Before verify-work:** Full sweep produces summary.json with all 4 variants + 11 weight-sweep points
- **Max feedback latency:** <30s tests

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | Status |
|---------|------|------|-------------|-----------|-------------------|--------|
| 13-01-01 | 01 | 1 | ENSM-01,05 | unit+contract | `pytest tests/models/test_ensemble.py -x -q` | ⬜ pending |
| 13-02-01 | 02 | 2 | ENSM-02,03,04,06 | integration | `jq '.variants \| length == 4' experiments/results/ensemble/summary.json` | ⬜ pending |
| 13-02-02 | 02 | 2 | ENSM-05 | safety | `git diff src/live/strategy.py` returns empty in phase-13 commits | ⬜ pending |
| 13-03-01 | 03 | 3 | ENSM-07 | smoke | `grep -q "### 5.11 Ensemble" PAPER_DRAFT.md \|\| grep -q "Ensemble Formalization" PAPER_DRAFT.md` | ⬜ pending |

## Wave 0 Requirements

- [ ] `tests/models/test_ensemble.py` — TDD test suite created in Wave 1 (Plan 13-01 Task 1 is RED phase)

*Co-created with implementation per TDD discipline.*

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Paper ensemble section narrative | ENSM-07 | Prose review | Confirm concordance audit honestly reports whether rejected trades are profitable (P4 flag surfaced) |

## Exceptions

- **ENSM-05 "NOT wired into live" guard:** Enforced by git-diff check in Task 13-02-02 verify. If any Phase 13 commit touches `src/live/strategy.py`, the guard fails.

**Approval:** approved 2026-04-23
