---
phase: 10
slug: 250-bar-scaling-checkpoint
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-22
---

# Phase 10 — Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest + inline assertions |
| **Config file** | existing pytest config |
| **Quick run command** | `jq '.metrics_by_model | keys' experiments/results/data_scaling/log.jsonl | tail -5` |
| **Full suite command** | `.venv/bin/python -m pytest tests/ -x -q` |
| **Estimated runtime** | ~45 min (Tier 1+2 training at 250 bars) |

## Sampling Rate

- **After every task commit:** Quick run command (JSON inspection)
- **Before verify-work:** jsonl has 250-bar entry with all 6 models
- **Max feedback latency:** 60 seconds for validation checks (training exempt)

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | Status |
|---------|------|------|-------------|-----------|-------------------|--------|
| 10-01-01 | 01 | 1 | SCAL-01 | integration | `tail -1 experiments/results/data_scaling/log.jsonl \| jq '.bars_per_pair == 250'` | ⬜ pending |
| 10-01-02 | 01 | 1 | SCAL-02,03 | smoke | `test -f experiments/results/data_scaling/pnl_at_2pp_vs_data.png && grep -q "250" PAPER_DRAFT.md` | ⬜ pending |
| 10-01-03 | 01 | 1 | SCAL-04,05 | manual | Check Finding 22 + §5.4 text for ranking invariance statement | ⬜ pending |

## Wave 0 Requirements

*None — all scripts and infrastructure already exist (Phase 8 verified).*

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Paper narrative quality | SCAL-05 | Prose review | Read §5.4 update for ranking-invariance claim clarity |

## Exceptions

- Feedback latency for the actual scaling-experiment training run (~45 min) exceeds 30s threshold. Inherent to training time; accepted.
- Auto-trigger fix is out of scope for v1.1 (SCC bug fix is a future-work item, not a paper contribution).

**Approval:** approved 2026-04-22
