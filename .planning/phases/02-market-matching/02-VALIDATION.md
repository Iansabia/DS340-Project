---
phase: 2
slug: market-matching
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-02
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (installed in Phase 1) |
| **Config file** | pytest.ini |
| **Quick run command** | `python -m pytest tests/matching/ -x -q` |
| **Full suite command** | `python -m pytest tests/matching/ -v` |
| **Estimated runtime** | ~15 seconds (semantic model loads once via fixture) |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/matching/ -x -q`
- **After every plan wave:** Run `python -m pytest tests/matching/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | MATCH-01, MATCH-02, MATCH-04 | unit | `python -m pytest tests/matching/test_matcher.py -v` | ❌ W0 | ⬜ pending |
| 02-01-02 | 01 | 1 | MATCH-05 | unit | `python -m pytest tests/matching/test_settlement.py -v` | ❌ W0 | ⬜ pending |
| 02-02-01 | 02 | 2 | MATCH-03 | unit+integration | `python -m pytest tests/matching/test_curation.py -v` | ❌ W0 | ⬜ pending |
| 02-02-02 | 02 | 2 | MATCH-01-05 | integration | `python -m pytest tests/matching/test_pipeline.py -v` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/matching/__init__.py` — test subpackage init
- [ ] `tests/matching/conftest.py` — shared fixtures (sample MarketMetadata, mock embeddings)
- [ ] `src/matching/__init__.py` — matching module init

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Curation UI usability | MATCH-03 | Requires human interaction with CLI | Run `python -m src.matching.curate` on sample data, verify accept/reject flow |
| Match quality on real data | MATCH-01, MATCH-02 | Requires running ingestion + real API data | Run full pipeline after ingestion, inspect matched_pairs.json |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
