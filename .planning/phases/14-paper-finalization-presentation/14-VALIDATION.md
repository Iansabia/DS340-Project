---
phase: 14
slug: paper-finalization-presentation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-23
---

# Phase 14 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x (for any new ieee_style helper tests) + shell/grep for document checks |
| **Config file** | `pytest.ini` (existing) |
| **Quick run command** | `pytest tests/plotting/ -q` (for plotting helper); `bash scripts/check_paper.sh` (for document integrity, to be created in Wave 0 if needed) |
| **Full suite command** | `pytest -q && bash scripts/check_paper.sh` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run relevant quick check (plotting → pytest; prose → grep-based integrity check; figure regen → visual + pytest).
- **After every plan wave:** Run full paper-integrity check (`bash scripts/check_paper.sh` or equivalent grep battery).
- **Before `/gsd:verify-work`:** All POL-* grep checks must pass; `wc -w` on abstract must be ≤ 250; all figure PNGs must exist and be readable.
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

Fill in during planning. Every POL-* requirement must map to at least one automated check.

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 14-01-XX | 01 | 1 | POL-01 | grep | `grep -c "SciencePlots\|science.*ieee" src/plotting/ieee_style.py` returns ≥ 1 | ❌ W0 | ⬜ pending |
| 14-01-XX | 01 | 1 | POL-02 | file check | `test -f experiments/figures/*.png && file $* \| grep -c "300 dpi"` | ❌ W0 | ⬜ pending |
| 14-01-XX | 01 | 1 | POL-03 | grep | `grep -E "\\\\caption|caption:" PAPER_DRAFT.md \| wc -l` ≥ figure count | ✅ | ⬜ pending |
| 14-01-XX | 01 | 1 | POL-04 | word count | `sed -n '/^## Abstract/,/^## /p' PAPER_DRAFT.md \| wc -w` ≤ 250 | ✅ | ⬜ pending |
| 14-01-XX | 01 | 1 | POL-07 | grep | `grep -c "per-pair.*Sharpe.*3\\.2" PAPER_DRAFT.md` ≥ 1 AND `grep -c "0\\.59.*annualize\\|0\\.59.*4\\.3" PAPER_DRAFT.md` == 0 | ✅ | ⬜ pending |
| 14-02-XX | 02 | 2 | POL-05 | grep | no duplicate "Table 6" / "Table 7" assignments in section headings | ✅ | ⬜ pending |
| 14-02-XX | 02 | 2 | POL-06 | grep | every "Figure N" reference in text matches a unique caption | ✅ | ⬜ pending |
| 14-02-XX | 02 | 2 | POL-08 | grep | `grep -c "survivorship" PAPER_DRAFT.md` ≥ 1 in §6.4 | ✅ | ⬜ pending |
| 14-02-XX | 02 | 2 | POL-09 | grep | Fig 2 caption contains "N=1,021\|cap\|truncat" | ✅ | ⬜ pending |
| 14-02-XX | 02 | 2 | POL-10 | grep | `grep -c "AI assistant\\|Claude\\|AI-assisted" PAPER_DRAFT.md` ≥ 1 in Acknowledgments | ✅ | ⬜ pending |
| 14-03-XX | 03 | 3 | POL-11 | file check | `test -f README.md && grep -c "python.*experiments/" README.md` ≥ number of paper tables | ❌ | ⬜ pending |
| 14-03-XX | 03 | 3 | POL-12 | file check | `test -f slides/*.pdf` AND slide count estimates 4-min talk | ❌ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

Task IDs will be finalized during planning.

---

## Wave 0 Requirements

- [ ] `src/plotting/ieee_style.py` — SciencePlots wrapper with Okabe-Ito palette fallback (tested via `tests/plotting/test_ieee_style.py`)
- [ ] `tests/plotting/test_ieee_style.py` — unit test: verify `apply_ieee_style()` sets `rcParams['savefig.dpi'] == 300` and returns a valid palette list
- [ ] `scripts/check_paper.sh` — grep battery that runs all POL-* document-integrity checks in one shot (optional but highly recommended for the sampling loop)

*If Wave 0 is skipped, plans must embed the ieee_style helper creation as an in-line task within the figure-regeneration plan.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Figure readability in B&W print | POL-01 | Requires human visual check; line-style/marker differentiation not grep-verifiable | Print a selected subset (Figs 2, 5, 10) on B&W printer; confirm series remain distinguishable |
| Slide pacing for 4-min talk | POL-12 | Requires dry-run timing; slide count is heuristic, not authoritative | Run a dry-run of `slides/presentation.pdf` out loud against a timer; confirm ≤ 4:00 with 10s/slide average |
| Final PDF cover-to-cover review | POL-12 | Subjective prose + flow check | Read PDF front-to-back, mark any residual TODO/placeholder/stale number on a checklist |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (ieee_style helper, check_paper.sh)
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter after planner finalizes task IDs

**Approval:** pending
