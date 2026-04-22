---
phase: 12
slug: feature-ablation
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-22
---

# Phase 12 — Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest + inline assertions |
| **Config file** | existing pytest config |
| **Quick run command** | `.venv/bin/python -m experiments.run_feature_ablation --dry-run` |
| **Full suite command** | `.venv/bin/python -m experiments.run_feature_ablation` (produces all 12 configs) |
| **Estimated runtime** | Dry-run <5s; full 12-config run ~15 min (LR fast, XGB with bootstrap) |

## Sampling Rate

- **After every task commit:** Dry-run
- **Before verify-work:** Full run produces `summary.json` with 12 configs + `report.md`
- **Max feedback latency:** <5s for dry-run; training exempted

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | Status |
|---------|------|------|-------------|-----------|-------------------|--------|
| 12-01-01 | 01 | 1 | ABLA-01 | smoke | `git log --oneline .planning/ablation_protocol.md \| head -1` | ⬜ pending |
| 12-01-02 | 01 | 1 | ABLA-02..07 | integration | `jq '.configs \| length == 12' experiments/results/ablation/summary.json` | ⬜ pending |
| 12-02-01 | 02 | 2 | ABLA-08 | smoke | `grep -q "### 5.10 Feature Ablation" PAPER_DRAFT.md` | ⬜ pending |

## Wave 0 Requirements

*None — all scripts, data files, and infrastructure already exist.*

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Paper §5.10 narrative quality | ABLA-08 | Prose review | Read §5.10 top-to-bottom, verify all 12 rows are present (no cherry-picking) |

## Exceptions

- ABLA-01 requires protocol commit BEFORE experiment run. This is enforced by commit ordering (Task 1's action creates + commits protocol, THEN runs experiment).
- Bootstrap CI computation adds ~3 min to total runtime; accepted.

**Approval:** approved 2026-04-22
