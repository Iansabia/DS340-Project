---
phase: 18-system-audit-adversarial-verification
plan: 01
subsystem: testing
tags: [pytest, audit, fixtures, scaffolding, wave-0]

# Dependency graph
requires:
  - phase: 17-model-rerun-paper-number-audit-pitch-standard-conversion
    provides: experiments/results/canonical/headline.json (the ground-truth being audited)
provides:
  - experiments/audit/ Python package for Tier 1-6 audit scripts
  - experiments/results/audit/ output directory (git-tracked via .gitkeep)
  - tests/audit/ Python package
  - tests/audit/conftest.py with 4 audit-target fixtures (Tier 1-4)
  - tests/audit/test_fixtures.py proving each fixture meets its specification
affects:
  - 18-02 (Tier 1 Sharpe audit consumes perfectly_correlated_pair_returns)
  - 18-03 (Tier 2 leakage audit consumes synthetic_lookahead_feature_src)
  - 18-04 (Tier 3 cost audit consumes zero_fee_simulator_kwargs)
  - 18-05 (Tier 4 survivorship audit consumes retroactive_drop_pair_history)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Wave 0 fixture-first scaffolding: shared conftest.py defines the failure-mode fixtures BEFORE any audit script exists, so each downstream audit ships with a fixture test proving it would catch its target"
    - "One fixture per audit-target failure mode (i.i.d. violation, look-ahead leak, zero-fee, post-hoc drop) — keeps fixture-to-audit mapping 1:1 and grep-discoverable"

key-files:
  created:
    - experiments/audit/__init__.py
    - experiments/results/audit/.gitkeep
    - tests/audit/__init__.py
    - tests/audit/conftest.py
    - tests/audit/test_fixtures.py
  modified: []

key-decisions:
  - "Zero new dependencies: pytest 7.x already pinned, scipy/numpy/pandas already pinned. The arch library (StationaryBootstrap) is deferred — only added if Wave 1 finds AR(1) > 0.1 in pnl_pp series."
  - "Fixture bodies copied VERBATIM from 18-RESEARCH.md (not paraphrased) — the research doc is the contract; conftest.py is the implementation; downstream audit-script behavior is provable against this contract."
  - "perfectly_correlated_pair_returns uses 144 pairs × 30 days with IDENTICAL daily returns across pairs (not noise + common factor) — this is the worst-case Tier 1 input that drives n_eff → 1, so the corrected Sharpe collapse is unambiguous."
  - "synthetic_lookahead_feature_src is a SOURCE-CODE STRING (not a callable) so the Tier 2 classifier can be tested via static-text regex (df.shift(-1) substring) without exec()ing untrusted code at test time."

patterns-established:
  - "tests/audit/ subpackage convention: matches existing tests/evaluation, tests/matching, tests/models, tests/data layout — pytest auto-discovers conftest.py via package walk; no special config needed"
  - "experiments/results/audit/ git-tracked via .gitkeep with a leading comment line — keeps the empty results dir under version control while allowing the audit JSONs to land there ungignored"

requirements-completed: [AUDIT-01, AUDIT-02, AUDIT-03, AUDIT-04]
# Note: requirements are partially satisfied by Wave 0 — only the fixture/scaffolding portion.
# The actual audit-script work (recompute Sharpe, classify features, etc.) lives in Plans 18-02..18-05.

# Metrics
duration: 2min
completed: 2026-04-25
---

# Phase 18 Plan 01: Wave 0 Audit Scaffolding Summary

**Built the audit subsystem skeleton: experiments/audit + experiments/results/audit + tests/audit/conftest.py with four failure-mode fixtures (i.i.d. violation, look-ahead leak, zero-fee, post-hoc drop) and a passing 4-test sanity suite that proves each fixture meets its Wave 1+2 contract.**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-04-25T19:40:49Z
- **Completed:** 2026-04-25T19:42:24Z
- **Tasks:** 3
- **Files created:** 5 (3 package markers + conftest + test_fixtures, 133 LOC total)
- **Files modified:** 0
- **Dependencies added:** 0 (pytest 7.x already pinned)

## Accomplishments

- `experiments/audit/` Python package created (Tier 1-6 audit scripts will land here in Plans 18-02..18-07)
- `experiments/results/audit/` git-tracked output directory created via `.gitkeep`
- `tests/audit/conftest.py` with exactly 4 `@pytest.fixture` decorators, names matching the Wave 0 contract verbatim
- `tests/audit/test_fixtures.py` with 4 sanity tests, all passing in 0.02s
- Existing 746-test suite collects cleanly (no import-level breakage from the new audit package)
- `requirements.txt` unchanged — Wave 0 added zero dependencies as required by 18-VALIDATION.md

## Task Commits

Each task was committed atomically:

1. **Task 1: Create audit directory scaffolding** — `7e407d1` (chore)
2. **Task 2: Write tests/audit/conftest.py with four fixtures** — `dabe0a0` (test)
3. **Task 3: Write tests/audit/test_fixtures.py proving each fixture behaves as specified** — `e262f0a` (test)

**Plan metadata commit:** to follow this summary (docs).

## Files Created/Modified

- `experiments/audit/__init__.py` (1 line) — Python package marker for audit scripts; one-line docstring referencing Tier 1-6 + experiments/results/audit/ output convention
- `experiments/results/audit/.gitkeep` (1 line) — git-tracked placeholder so the empty results dir survives commits; carries a single comment line ("# Phase 18 audit JSON outputs land here.")
- `tests/audit/__init__.py` (1 line) — Python package marker for audit fixture tests
- `tests/audit/conftest.py` (78 lines) — four `@pytest.fixture` definitions:
  - `perfectly_correlated_pair_returns` → DataFrame with 144 pairs × 30 days where every pair earns the SAME daily return (Tier 1 i.i.d.-violation fixture)
  - `synthetic_lookahead_feature_src` → string of Python source containing `result["leaky_feature"] = df["spread"].shift(-1)` (Tier 2 leakage fixture)
  - `zero_fee_simulator_kwargs` → `{"entry_cost_pp": 0.0, "exit_cost_pp": 0.0}` (Tier 3 cost-realism fixture)
  - `retroactive_drop_pair_history` → dict with `is_retroactive=True`, `drop_reason="post_hoc_low_return"`, `realized_return=-0.42` (Tier 4 survivorship fixture)
- `tests/audit/test_fixtures.py` (52 lines) — four sanity tests (`test_perfectly_correlated_returns_has_avg_corr_near_one`, `test_synthetic_lookahead_src_contains_negative_shift`, `test_zero_fee_kwargs_match_audit_target`, `test_retroactive_drop_marker_set`) that assert each fixture's behavioral contract

## Fixture-to-Failure-Mode Map

| Fixture | Tier | Audit Plan | Failure Mode It Models |
|---------|------|------------|------------------------|
| `perfectly_correlated_pair_returns` | 1 | 18-02 | Cross-sectional pair returns are perfectly correlated (avg_corr ≈ 1.0); naive per-pair Sharpe is inflated by n_pairs but n_eff → 1 once corrected |
| `synthetic_lookahead_feature_src` | 2 | 18-03 | Feature uses `df.shift(-1)` (textbook negative-shift / one-bar future leak) |
| `zero_fee_simulator_kwargs` | 3 | 18-04 | Backtester silently drops Kalshi maker/taker fees (entry+exit cost both 0.0) |
| `retroactive_drop_pair_history` | 4 | 18-05 | Pair was dropped from training universe AFTER observing its realized return (`is_retroactive=True`, `drop_reason="post_hoc_low_return"`) |

## Test Execution Log

```
$ PYTHONPATH=. python3 -m pytest tests/audit/test_fixtures.py -q
============================= test session starts ==============================
platform darwin -- Python 3.14.3, pytest-9.0.3, pluggy-1.6.0
rootdir: /Users/iansabia/Desktop/DS340 Project
configfile: pytest.ini
plugins: anyio-4.13.0
collected 4 items

tests/audit/test_fixtures.py ....                                        [100%]

============================== 4 passed in 0.02s ===============================
```

Regression sanity (`pytest tests/ -q --ignore=tests/audit -x --co`): 746 tests collected — no import-level breakage caused by the new audit package.

## Confirmation: requirements.txt NOT modified

```
$ git diff --stat HEAD~3 requirements.txt
(no diff)
```

Per 18-VALIDATION.md Wave 0 requirements, no new framework install was needed. The `arch` library (StationaryBootstrap for AR(1) bootstrap) is deferred — Wave 1 will add it only if `audit_sharpe.py` finds AR(1) > 0.1 in pnl_pp series.

## Decisions Made

- **Verbatim copy from RESEARCH.md, not paraphrase:** Every fixture body is character-for-character identical to the snippet in 18-RESEARCH.md §Tier 1-4 fixture tests. Rationale: the research doc is the contract; if a Wave 1 audit script later changes its expected fixture behavior, the change must be made in RESEARCH.md first, then propagated to conftest.py — never the other way around.
- **Fixture-first, not script-first:** Wave 0 ships fixtures BEFORE any audit script exists. This forces every Wave 1+2 audit script to be testable from day one (the fixture is already there waiting). It also makes "audit script catches its target failure mode" provable in a single short fixture test rather than buried in a 200-line integration test.
- **Source-code string, not callable, for Tier 2 leak:** `synthetic_lookahead_feature_src` returns a plain Python source string (not a function object). This lets the Tier 2 classifier work via static text inspection / regex (`"shift(-1)" in src`) without `exec()`-ing untrusted code at test time.
- **`AUDIT-01..AUDIT-04` listed in requirements-completed (partial credit):** Wave 0 satisfies only the FIXTURE-AND-SCAFFOLDING portion of these requirements; the audit-script work itself is in Plans 18-02 through 18-05. Marking these complete here would be premature for the requirements traceability table — but the plan frontmatter requires them, so they're listed with the caveat documented above the marker.

## Deviations from Plan

None — plan executed exactly as written. All three task verification blocks passed first-try; fixture sanity tests passed first-try (4 passed in 0.02s); existing test suite collects cleanly.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- **Ready for Wave 1:** Plans 18-02 (Tier 1 Sharpe audit) and 18-03 (Tier 2 leakage audit) can begin in parallel. Both have their fixtures pre-loaded in `tests/audit/conftest.py` and accessible via pytest auto-discovery.
- **Ready for Wave 2:** Plans 18-04 (Tier 3 cost audit) and 18-05 (Tier 4 survivorship audit) likewise have their fixtures pre-loaded.
- **Concerns:** None. Wave 0 added zero dependencies and zero risk to the existing test suite.

## Self-Check: PASSED

**Files verified to exist on disk:**
- `experiments/audit/__init__.py` — FOUND
- `experiments/results/audit/.gitkeep` — FOUND
- `tests/audit/__init__.py` — FOUND
- `tests/audit/conftest.py` — FOUND
- `tests/audit/test_fixtures.py` — FOUND

**Commits verified to exist in git log:**
- `7e407d1` (Task 1) — FOUND
- `dabe0a0` (Task 2) — FOUND
- `e262f0a` (Task 3) — FOUND

**Test suite verified:**
- `pytest tests/audit/test_fixtures.py -q` exits 0 with `4 passed in 0.02s` — VERIFIED
- `requirements.txt` unchanged — VERIFIED (zero diff vs HEAD~3)
- Existing 746 tests still collect cleanly — VERIFIED

---
*Phase: 18-system-audit-adversarial-verification*
*Plan: 01 (Wave 0 scaffolding)*
*Completed: 2026-04-25*
