---
phase: 18
slug: system-audit-adversarial-verification
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-25
---

# Phase 18 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x (existing) |
| **Config file** | `pytest.ini` (existing) |
| **Quick run command** | `pytest tests/audit/ -q` |
| **Full suite command** | `pytest tests/audit/ -q && bash scripts/check_paper.sh` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/audit/test_<tier>.py -q` (≤ 5s per tier)
- **After every plan wave:** Run `pytest tests/audit/ -q && bash scripts/check_paper.sh`
- **Before `/gsd:verify-work`:** Full suite must be green AND `AUDIT_REPORT.md` row count ≥ 6 (one per tier) AND every row has PASS / CORRECTED / FAILED verdict (no blanks)
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

Filled in during planning. Every AUDIT-* requirement must map to at least one automated check; "passing" the audit is itself the test.

| Task ID  | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|----------|------|------|-------------|-----------|-------------------|-------------|--------|
| 18-01-XX | 01   | 0    | scaffolding | file check | `test -d experiments/audit && test -d tests/audit && test -d experiments/results/audit` | ❌ W0 | ⬜ pending |
| 18-01-XX | 01   | 0    | fixture tests | unit | `pytest tests/audit/test_fixtures.py -q` (asserts each audit *would* catch its target failure) | ❌ W0 | ⬜ pending |
| 18-02-XX | 02   | 1    | AUDIT-01 | integration | `python experiments/audit/audit_sharpe.py && jq -e '.per_pair_sharpe_naive' experiments/results/audit/sharpe_audit.json` | ❌ W0 | ⬜ pending |
| 18-02-XX | 02   | 1    | AUDIT-01 | unit | `pytest tests/audit/test_audit_sharpe.py -q` | ❌ W0 | ⬜ pending |
| 18-03-XX | 03   | 1    | AUDIT-02 | integration | `python experiments/audit/audit_leakage.py && jq -e '.embargo_violations | length == 0' experiments/results/audit/leakage_audit.json` | ❌ W0 | ⬜ pending |
| 18-03-XX | 03   | 1    | AUDIT-02 | unit | `pytest tests/audit/test_audit_leakage.py -q` (synthetic look-ahead feature MUST be flagged) | ❌ W0 | ⬜ pending |
| 18-04-XX | 04   | 2    | AUDIT-03 | integration | `python experiments/audit/audit_costs.py && jq -e '.fee_audit_status' experiments/results/audit/costs_audit.json` | ❌ W0 | ⬜ pending |
| 18-04-XX | 04   | 2    | AUDIT-03 | unit | `pytest tests/audit/test_audit_costs.py -q` (zero-fee fixture MUST be flagged) | ❌ W0 | ⬜ pending |
| 18-05-XX | 05   | 2    | AUDIT-04 | integration | `python experiments/audit/audit_survivorship.py && jq -e '.random_sample | length == 10' experiments/results/audit/survivorship_audit.json` | ❌ W0 | ⬜ pending |
| 18-06-XX | 06   | 2    | AUDIT-05 | file check | `test -f experiments/results/audit/paper_numbers.csv && wc -l < experiments/results/audit/paper_numbers.csv` ≥ 20 | ❌ W0 | ⬜ pending |
| 18-06-XX | 06   | 2    | AUDIT-05 | grep | `grep -c "^audit_" scripts/check_paper.sh` ≥ 5 (at least 5 new regression checks) | ❌ W0 | ⬜ pending |
| 18-07-XX | 07   | 3    | AUDIT-06 | file check | `test -f AUDIT_REPORT.md && grep -cE "^\| (Tier [1-6]\|AUDIT-0[1-6])" AUDIT_REPORT.md` ≥ 6 | ❌ | ⬜ pending |
| 18-07-XX | 07   | 3    | AUDIT-06 | grep | `grep -cE "(PASS\|CORRECTED\|FAILED)" AUDIT_REPORT.md` ≥ 6 (no blank verdicts) | ❌ | ⬜ pending |
| 18-07-XX | 07   | 3    | AUDIT-06 | conditional | If any audit row is CORRECTED: `git diff --name-only HEAD~3..HEAD | grep -E "(PAPER_DRAFT.md\|slides_deck.html)"` MUST be non-empty | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky · W0 = Wave 0 dependency*

Task IDs will be finalized during planning.

---

## Wave 0 Requirements

- [ ] `experiments/audit/` directory created
- [ ] `experiments/results/audit/` directory created
- [ ] `tests/audit/` directory + `tests/audit/__init__.py`
- [ ] `tests/audit/conftest.py` — shared fixtures (synthetic perfectly-correlated pair returns, synthetic look-ahead feature, zero-fee fixture, retroactive-filter fixture)
- [ ] `tests/audit/test_fixtures.py` — sanity test that fixtures behave as specified (e.g., perfectly-correlated returns have avg_pair_corr ≈ 1.0; synthetic look-ahead feature uses `df.shift(-1)`)
- [ ] No new framework install — pytest 7.x already in `requirements.txt`
- [ ] No new dependency installs except `arch` (only if research recommends `StationaryBootstrap` — Wave 0 task adds to `requirements.txt` if needed; otherwise scipy bootstrap suffices)

*If Wave 0 fixture tests are skipped, audit scripts cannot prove they would catch their target failure modes — Tier-by-tier verification becomes "trust me" instead of "demonstrably correct." Wave 0 is mandatory.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Random pair-drop spot-check classification | AUDIT-04 | Each of the 10 randomly-selected dropped pairs must be eyeballed against its row in `data/processed/pairs/` history to confirm the drop reason is structural (insufficient overlap, low liquidity), not retroactive. Rule classification is a human judgment call. | Review the 10 rows in `experiments/results/audit/survivorship_audit.json`, mark each as `structural` or `retroactive` in the JSON, and re-run `audit_survivorship.py` to update the verdict |
| Final `AUDIT_REPORT.md` PASS/CORRECTED/FAILED verdicts | AUDIT-06 | Each verdict requires reading the corresponding JSON output AND deciding whether the assumption stack is defensible (a judgment call). The script generates the table; the human writes the verdict prose. | After all audit scripts run, manually edit `AUDIT_REPORT.md` to set each row's verdict + 1-2 sentence finding |
| Paper-prose corrections (e.g., the "2pp transaction costs" §5.1 wording bug already surfaced in research) | AUDIT-05 | Prose edits are not grep-verifiable beyond "the wrong phrase is gone"; reviewer must read the new text in context to confirm it accurately describes what `simulate_profit` does | Edit PAPER_DRAFT.md §5.1; re-run `bash scripts/check_paper.sh`; eyeball the new paragraph for accuracy |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (audit dirs, fixture conftest, fixture tests)
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter after planner finalizes task IDs

**Approval:** pending
