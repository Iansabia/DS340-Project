---
phase: 1
slug: data-ingestion
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-01
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (to be installed in Wave 0) |
| **Config file** | none — Wave 0 installs |
| **Quick run command** | `python -m pytest tests/data/ -x -q` |
| **Full suite command** | `python -m pytest tests/data/ -v` |
| **Estimated runtime** | ~30 seconds (API calls are cached/mocked in tests) |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/data/ -x -q`
- **After every plan wave:** Run `python -m pytest tests/data/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 01-00-01 | 00 | 0 | - | setup | `python -m pytest --version` | ❌ W0 | ⬜ pending |
| 01-01-01 | 01 | 1 | DATA-01 | unit+integration | `python -m pytest tests/data/test_kalshi.py -v` | ❌ W0 | ⬜ pending |
| 01-02-01 | 02 | 1 | DATA-02, DATA-03 | unit+integration | `python -m pytest tests/data/test_polymarket.py -v` | ❌ W0 | ⬜ pending |
| 01-01-02 | 01 | 1 | DATA-04, DATA-05 | unit | `python -m pytest tests/data/test_rate_limit.py -v` | ❌ W0 | ⬜ pending |
| 01-02-02 | 02 | 1 | DATA-06 | integration | `python -m pytest tests/data/test_storage.py -v` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `pip install pytest pyarrow` — missing test framework and parquet support
- [ ] `tests/__init__.py` — test package init
- [ ] `tests/data/__init__.py` — data test subpackage
- [ ] `tests/data/conftest.py` — shared fixtures (mock API responses, temp directories)
- [ ] `requirements.txt` — generated for team reproducibility

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Rate limits respected | DATA-04 | Requires real API calls to verify timing | Run full ingestion, check logs for rate limit sleeps |
| Retry on transient failure | DATA-05 | Requires simulating network failure | Kill network mid-run, verify retry in logs |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
