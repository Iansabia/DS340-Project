---
phase: 8
slug: environment-and-baseline-verification
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-17
---

# Phase 8 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing) |
| **Config file** | `pyproject.toml` / `pytest.ini` if exists |
| **Quick run command** | `.venv/bin/python -m experiments.verify_headline` |
| **Full suite command** | `.venv/bin/python -m pytest tests/ -x -q` |
| **Estimated runtime** | ~120 seconds (includes GRU/LSTM training) |

---

## Sampling Rate

- **After every task commit:** Run `.venv/bin/python -m experiments.verify_headline`
- **After every plan wave:** Run `.venv/bin/python -m pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 08-01-01 | 01 | 1 | ENV-01 | smoke | `.venv/bin/pip install pytorch-forecasting==1.7.0 --dry-run` | ✅ | ⬜ pending |
| 08-01-02 | 01 | 1 | ENV-02 | smoke | `.venv/bin/python -c "import pytorch_forecasting, quantstats, scienceplots"` | ❌ W0 | ⬜ pending |
| 08-02-01 | 02 | 2 | ENV-03 | unit | `.venv/bin/python -c "from src.utils.seed import set_all_seeds; set_all_seeds(42)"` | ❌ W0 | ⬜ pending |
| 08-02-02 | 02 | 2 | ENV-04 | integration | `.venv/bin/python -m experiments.verify_headline` (run twice, compare) | ✅ | ⬜ pending |
| 08-02-03 | 02 | 2 | ENV-05 | integration | `diff <(jq .results experiments/results/verify_headline.json) <(jq .results experiments/results/tier1/xgboost.json)` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `src/utils/__init__.py` — create utils package
- [ ] `src/utils/seed.py` — shared seed utility (9 RNG sources)

*Existing test infrastructure covers all other requirements.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| (none) | — | — | — |

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have automated verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Exceptions:**
- Wave 0 artifacts (`src/utils/seed.py`, `check_reproducibility.py`) are co-created with implementation in 08-02. Accepted because Phase 8 is the gating phase — there is no prior infrastructure to build stubs against.
- Feedback latency for ENV-04 (~240s) exceeds 30s threshold. Inherent: reproducibility verification requires running full model training twice. Accepted.

**Approval:** approved 2026-04-17
