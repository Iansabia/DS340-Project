---
phase: 11
slug: tft-training
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-22
---

# Phase 11 — Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest + inline assertions |
| **Config file** | existing pytest config |
| **Quick run command** | `.venv/bin/python -c "from src.models.tft import TFTPredictor; print('OK')"` |
| **Full suite command** | `.venv/bin/python -m pytest tests/models/test_tft.py -x -q` |
| **Estimated runtime** | Fast smoke <5s; single-split training 30-60 min CPU |

## Sampling Rate

- **After every task commit:** Quick run (import check)
- **Before verify-work:** Full suite green OR documented negative-result artifact present
- **Max feedback latency:** 30s for tests; training exempted (inherent to model size)

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | Status |
|---------|------|------|-------------|-----------|-------------------|--------|
| 11-01-01 | 01 | 1 | TFT-01,02 | unit | `pytest tests/models/test_tft.py::test_basepredictor_contract -x` | ⬜ pending |
| 11-01-02 | 01 | 1 | TFT-03,04 | integration | `.venv/bin/python -m experiments.run_tft` (produces results or documented failure) | ⬜ pending |
| 11-02-01 | 02 | 2 | TFT-05,08 | smoke | `test -f experiments/figures/tft_variable_importance.png` | ⬜ pending |
| 11-02-02 | 02 | 2 | TFT-06,07 | integration | `grep -q "TFT" PAPER_DRAFT.md && grep -q "^| TFT " PAPER_DRAFT.md` | ⬜ pending |

## Wave 0 Requirements

- [ ] `tests/models/__init__.py` — exists (check if missing)
- [ ] `tests/models/test_tft.py` — stubs for BasePredictor contract

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Training outcome (success vs no-convergence) | TFT-04 | Requires reading val_loss trajectory and attention-entropy output | Inspect training log; confirm val_loss comparison to GRU baseline (0.2928 RMSE) is documented |

## Exceptions

- Training latency (~30-60 min CPU) inherent; accepted per VALIDATION pattern for Phase 8/10
- TFT may legitimately fail to converge — documented negative result is a valid completion per TFT-04 Option B gate

**Approval:** approved 2026-04-22
