---
phase: 15-live-commodity-matching-engineering-fixes
plan: 03
subsystem: live-trading
tags: [kalshi-discovery, commodity-matching, market_discovery, category-rules, collector-reservation, scc-validation, COM-03, COM-04, COM-05]

# Dependency graph
requires:
  - phase: 15-live-commodity-matching-engineering-fixes
    provides: "Plan 15-01 diagnostic (H1 CONFIRMED category gap + H5 PARTIAL classifier miscoding); Plan 15-02 Rule 10 asset-class filter as belt-and-braces against re-opening the KXWTIMAX↔Bitcoin false match"
provides:
  - "'Commodities' added to KALSHI_DISCOVERY_CATEGORIES (src/live/market_discovery.py) — daily/weekly WTI + Brent + grain + metal series now reach discovery"
  - "_RULES extension in src/features/category.py — Brent family + remaining WTI variants + KXCRUDE/KXDIESEL/KXHEATINGOIL/KXGASOLINE now classify as 'oil' instead of 'other' (avoids 3x threshold penalty)"
  - "200-slot commodity reservation in src/live/collector.py _load_live_pairs — prevents commodity pairs (max similarity ≈ 0.794) from being evicted by the MAX_LIVE_PAIRS=2000 similarity cap dominated by sports/politics pairs at ≥ 0.85"
  - "Regenerated data/live/active_matches.json (115,104 entries; 336 non-evicted non-KXWTIMAX commodity pairs) and data/live/pair_mapping.json (2,000 entries; 200 non-KXWTIMAX commodity pairs) as the post-fix live state"
  - "24-hour SCC validation window demonstrating 1,224 closed non-KXWTIMAX commodity positions — exceeds COM-04 target by 122x"
affects: [live-trading, matching-pipeline, paper-6.4, future-paper-revisions]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Category-specific slot reservation in MAX_LIVE_PAIRS similarity cap — prevents newly-onboarded asset classes from being silently evicted by established high-similarity domains"
    - "Diagnostic-first → TDD-test → discovery-fix → SCC-validation pipeline (15-01 → 15-02 → 15-03) — each plan commits before the next starts, mirrors RED-before-GREEN but across research and engineering layers"

key-files:
  created:
    - ".planning/phases/15-live-commodity-matching-engineering-fixes/15-03-SUMMARY.md"
  modified:
    - "src/live/market_discovery.py"
    - "src/features/category.py"
    - "src/live/collector.py"
    - "data/live/active_matches.json"
    - "data/live/pair_mapping.json"
    - "data/live/position_history.jsonl"
    - ".planning/STATE.md"
    - ".planning/ROADMAP.md"
    - ".planning/REQUIREMENTS.md"

key-decisions:
  - "H1 + H5 (both) applied in Task 1 — the H5 classifier extension is essential post-H1 even though the diagnostic marked H5 as PARTIAL, because without it newly-discovered Brent/daily-WTI tickers would be classified as 'other' and hit the 3x threshold penalty in strategy.py:437"
  - "Commodity slot reservation (200 slots out of 2000 in MAX_LIVE_PAIRS) added during Task 2 as a Rule 3 (blocking) fix — diagnostic did not predict this because the similarity-cap eviction only became visible after H1 flooded discovery with new pairs at similarity ≈ 0.794 (well below the sports/politics-dominated 0.85+ tail)"
  - "24h SCC validation window used the ≥ 10 closed non-KXWTIMAX commodity positions target from COM-04 — actual result exceeded target by 122x (1,224 closed), with daily KXWTI (409 positions) and weekly KXBRENTW (486 positions) being the two highest-volume series, directly filling the paper §6.4 item-9 gap"

patterns-established:
  - "Similarity-cap eviction safety pattern: when a new asset class is added to discovery, reserve a slot budget inside MAX_LIVE_PAIRS equal to the pair-count target — prevents silent eviction by domains with historically higher embedding similarity"
  - "Post-deployment validation cadence: push fix → wait 24h of SCC cycles (~96 runs at 15-min interval) → count closed positions by series — gives honest evidence the fix propagated end-to-end through discovery/matching/trading/exit"

requirements-completed:
  - COM-03
  - COM-04
  - COM-05

# Metrics
duration: ~33h (wall-clock, dominated by 24h SCC validation window)
completed: 2026-04-24
---

# Phase 15 Plan 03: Commodity Discovery Fix + 24h SCC Validation Summary

**Added "Commodities" to `KALSHI_DISCOVERY_CATEGORIES`, extended `_RULES` in `src/features/category.py` for the Brent + daily-WTI family, and reserved 200 slots for commodity pairs inside the `MAX_LIVE_PAIRS=2000` similarity cap — post-fix 24h SCC window closed 1,224 non-KXWTIMAX commodity positions (122x COM-04 target), directly filling the paper §6.4 item-9 live-cohort gap.**

## Performance

- **Duration:** ~33 hours wall-clock (dominated by the 24-hour SCC validation window; active coding for Tasks 1-2 was ~30 minutes)
- **Started:** 2026-04-23T00:15:00Z (Task 1 coding begins)
- **Completed:** 2026-04-24T13:22:19Z (post-window validation + this SUMMARY)
- **Tasks:** 3 (Task 1 + Task 2 + Task 3 human-verify checkpoint)
- **Files modified:** 9 (3 source, 3 live state, 3 planning)

## Accomplishments

- **H1 fix (CONFIRMED blocker from 15-01):** `src/live/market_discovery.py:249` now reads `("Economics", "Crypto", "Financials", "Politics", "Climate", "Commodities")`. Discovery sweep picks up 47 Commodities-category series (including KXWTI, KXWTIW, KXBRENTMON, KXBRENTW, KXBRENTD, KXWTIMIN, KXWTIMINM — the exact gap set enumerated in `15-01-DIAGNOSTIC.md` Probe 5).
- **H5 fix (PARTIAL → fully applied):** `src/features/category.py::_RULES` extended with Brent family (`KXBRENTMON`, `KXBRENTW`, `KXBRENTD`, `KXBRENT`) and remaining WTI variants (`KXWTIH`, `KXWTIEU`, `KXWTIE`, `KXWTID`, `KXWTIMINM`, `KXWTIMIN`) plus `KXCRUDE`, `KXDIESEL`, `KXHEATINGOIL`, `KXGASOLINE`. All route to category `"oil"`, which qualifies for the low-threshold commodity bucket in `strategy.py:405-407` — newly discovered oil tickers trade at 1x threshold instead of the 3x "other" penalty.
- **Similarity-cap reservation (unplanned Rule 3 fix):** `src/live/collector.py::_load_live_pairs` now reserves 200 slots out of the global `MAX_LIVE_PAIRS=2000` cap for commodity pairs. Without this reservation, every commodity pair would have been evicted because their max similarity (~0.794) falls below the sports/politics-dominated tail above 0.85. This was NOT predicted by the 15-01 diagnostic — it only became visible once H1 flooded the candidate pool with new low-similarity pairs.
- **Regenerated live state:** `data/live/active_matches.json` (115,104 entries, 336 active non-KXWTIMAX commodity pairs: KXAAAGAS 231 / KXWTI 45 / KXBRENTW 40 / KXBRENTMON 20) and `data/live/pair_mapping.json` (2,000 entries, 200 non-KXWTIMAX commodity pairs). Both targets (COM-03 ≥ 20, COM-04 ≥ 1) exceeded by orders of magnitude.
- **24-hour SCC validation window:** After `git push origin master` deployed the fix, the SCC trading cycle (15-min cadence, ~96 cycles/24h) closed **1,224 non-KXWTIMAX commodity positions** by 2026-04-24T13:00:11Z. COM-04's ≥ 10-closed-position target was exceeded by **122x**.
- **Rule 10 regression holding:** `tests/matching/test_rule_10_asset_class.py` still exits 0 — canonical KXWTIMAX-26DEC31-T130 ↔ Bitcoin-$130K pair still rejected with `asset_class_mismatch (kalshi=commodity, poly=crypto)`. No collateral damage from discovery-side changes.

## Task Commits

Each task was committed atomically per the CLAUDE.md / GSD discipline:

1. **Task 1: Apply discovery fix(es) per 15-01 Fix Recommendations** — `38d7970` (`fix(live-discovery): apply 15-01 fix recommendations for commodity gap (COM-03)`)
   - `src/live/market_discovery.py:249` — add `"Commodities"` to `KALSHI_DISCOVERY_CATEGORIES` (H1 fix).
   - `src/features/category.py::_RULES` — add Brent family + remaining WTI variants + KXCRUDE/KXDIESEL/KXHEATINGOIL/KXGASOLINE (H5 fix).
   - Verified via 4/4 PASS on `tests/matching/test_rule_10_asset_class.py`, 156 PASS on `tests/matching/ tests/live/` (minus the pre-existing sentence_transformers import error documented in 15-02), and canonical false-match regression still rejected.
2. **Task 2: Run post-fix discovery end-to-end and validate COM-03 / COM-04 pre-window metrics** — `d217ff1` (`data(live): regenerate active_matches + pair_mapping after Phase 15 discovery fix (COM-03, COM-04)`)
   - Included inline Rule 3 blocking fix to `src/live/collector.py` (200-slot commodity reservation inside `_load_live_pairs` MAX_LIVE_PAIRS=2000 cap).
   - Post-fix metrics: 336 active non-KXWTIMAX commodity pairs in `active_matches.json`, 200 in `pair_mapping.json` (both targets blown past).
3. **Task 3: Human-verify checkpoint — 24h SCC window + STATE/ROADMAP/REQUIREMENTS update** — *this commit* (`docs(15-03): complete discovery fix plan — 1,224 commodity positions closed in 24h validation window`)
   - User approved after the 24-hour window validated COM-04 at 1,224 closed positions.

**Plan metadata commit:** folds Task 3 into this SUMMARY + STATE.md + ROADMAP.md + REQUIREMENTS.md as one focused atomic commit.

## Files Created/Modified

- `src/live/market_discovery.py` (MODIFIED) — H1 fix: added `"Commodities"` to `KALSHI_DISCOVERY_CATEGORIES` tuple at line 249.
- `src/features/category.py` (MODIFIED) — H5 fix: extended `_RULES` with Brent family + daily/weekly WTI variants + KXCRUDE/KXDIESEL/KXHEATINGOIL/KXGASOLINE → "oil".
- `src/live/collector.py` (MODIFIED) — Rule 3 blocking fix: added commodity-pair slot reservation (200/2000) inside `_load_live_pairs` to prevent similarity-cap eviction.
- `data/live/active_matches.json` (REGENERATED) — 115,104 entries post-fix; 336 active non-KXWTIMAX commodity pairs.
- `data/live/pair_mapping.json` (REGENERATED) — 2,000 entries post-fix; 200 non-KXWTIMAX commodity pairs.
- `data/live/position_history.jsonl` (GROWN DURING 24H WINDOW) — 1,224 closed non-KXWTIMAX commodity positions added during the SCC validation run.
- `.planning/STATE.md` (MODIFIED) — Phase 15 diagnostic note appended under Decisions; progress and session fields updated.
- `.planning/ROADMAP.md` (MODIFIED) — Phase 15 marked complete in the status table; Plan 15-03 checkbox ticked.
- `.planning/REQUIREMENTS.md` (MODIFIED) — COM-03, COM-04, COM-05 flipped from Pending to Complete in the v1.1 traceability table.

## Post-Window Validation Numbers (Verbatim)

- **Total closed non-KXWTIMAX commodity positions:** **1,224** by 2026-04-24T13:00:11Z (1,159 post-push as of the orchestrator's earlier sample, with 65 additional closures in the last hour of the window)
- **Aggregate P&L:** **+$1.96** (paper trading, no slippage, no partial fills)
- **Win rate:** **36.0%** (441 / 1,224)
- **Last exit:** 2026-04-24T13:00:11Z
- **Per-series breakdown** (closed position counts):
  - `KXBRENTW` — **486** (weekly Brent range)
  - `KXWTI` — **409** (daily WTI on-day — the primary paper §6.4 gap; now the second-highest-volume series)
  - `KXWTIW` — **213** (weekly WTI range)
  - `KXBRENTMON` — **76** (monthly Brent)
  - `KXBRENTD` — **16** (daily Brent)
  - `KXAAAGASD` — **11** (daily AAA retail gasoline)
  - `KXAAAGASW` — **7** (weekly AAA retail gasoline)
  - `KXAAAGASM` — **6** (monthly AAA retail gasoline)

COM-04's ≥ 10-closed-position target exceeded by **122x**. The KXBRENTW + KXWTI + KXWTIW trio alone (1,108 closures) validates that the discovery fix reached the primary gap-set tickers identified in `15-01-DIAGNOSTIC.md` Probe 5.

## Decisions Made

- **Applied H5 at Task 1 time, not deferred:** 15-01 marked H5 as PARTIAL, but the classifier extension is a prerequisite for the H1 fix to actually trade — without it, newly discovered Brent/daily-WTI tickers fall through to `"other"` and hit the 3x threshold penalty. Shipping H1 alone would have added discovery coverage without trading coverage.
- **Similarity-cap reservation added without checkpoint:** The `_load_live_pairs` eviction was caught only when Task 2's post-discovery metrics showed 0 commodity pairs in `pair_mapping.json` despite 336 being present in `active_matches.json`. Root cause: the MAX_LIVE_PAIRS=2000 cap is sorted by similarity and commodity pairs cluster around 0.794 while the top 2,000 is dominated by sports/politics at ≥ 0.85. This is a Rule 3 (blocking) deviation — classified as auto-fix, not checkpoint, because without it the 24-hour SCC window would have closed zero commodity positions.
- **24h window approval threshold:** the plan said ≥ 10 closed non-KXWTIMAX commodity positions; actual result was 1,224 (122x). No ambiguity in approval — user "approved" after the orchestrator surfaced the numbers.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Commodity-pair slot reservation in `_load_live_pairs` similarity cap**
- **Found during:** Task 2 (post-fix discovery metrics validation)
- **Issue:** After Task 1's H1 fix, `data/live/active_matches.json` contained 336 active non-KXWTIMAX commodity pairs, but `data/live/pair_mapping.json` contained zero. Root cause: `MAX_LIVE_PAIRS=2000` similarity cap in `src/live/collector.py::_load_live_pairs` sorts pairs by embedding similarity descending and evicts everything past slot 2000. Commodity pairs cluster around similarity 0.794 (oil-to-oil titles share "oil" tokens but differ on horizon/strike), while sports/politics pairs dominate the ≥ 0.85 tail. Without intervention, every commodity pair would be silently evicted at the collector stage — Task 1's discovery fix would have propagated zero trades to the 24h window.
- **Fix:** Added a 200-slot reservation for commodity pairs inside `_load_live_pairs` — commodity pairs are sorted into a reserved pool before the global MAX_LIVE_PAIRS sort. 200 slots chosen as 10% of the global cap and 10x the COM-03 target (≥ 20 active non-evicted commodity pairs).
- **Files modified:** `src/live/collector.py` (inside `_load_live_pairs`)
- **Verification:** Post-fix `pair_mapping.json` contains 200 non-KXWTIMAX commodity pairs — matches the reservation budget exactly. 24h SCC window subsequently closed 1,224 positions drawn from the reserved slot pool.
- **Committed in:** `d217ff1` (Task 2 commit — bundled with `active_matches.json` + `pair_mapping.json` regeneration)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Essential for correctness. Without the similarity-cap reservation, Task 2's metrics would have passed (active_matches.json had the 336 commodity pairs) but the 24h SCC window would have closed zero positions because the collector's pair_mapping.json is the actual input to `trading_cycle.py`. The H1 + H5 fixes would have been a silent no-op. No scope creep — the reservation is strictly in-scope for COM-04 (post-fix validation must show ≥ 10 closed commodity positions, which requires the pairs to survive collector eviction).

## Issues Encountered

None — the 24-hour SCC window produced 1,224 closed commodity positions, 122x the ≥ 10 target. Pre-existing sentence_transformers import error in three matching test files is out-of-scope per scope-boundary rule (documented in 15-02-SUMMARY.md); unchanged by this plan.

## User Setup Required

None — discovery fix is pure code + deployment. SCC was already configured.

## Next Phase Readiness

- **Phase 15 complete.** COM-01 through COM-05 all closed with evidence committed.
- **Paper §6.4 item 9 (live-cohort commodity-matching gap) is now empirically addressed.** The 1,224 closed positions across KXWTI / KXBRENTW / KXWTIW / KXBRENTMON / KXBRENTD constitute a live-validation cohort that future paper revisions can cite as a follow-up to the v1.1 submission's acknowledged limitation. Aggregate P&L (+$1.96) and win rate (36.0%) are notable: the backtest Finding 6 suggested oil was the edge, and the live cohort now allows that claim to be re-checked against real deployment behavior rather than a limitation footnote.
- **No blockers.** Live trading system is running on post-fix code. Daily WTI / Brent series trade end-to-end through discovery → matching → collector → trading cycle → position close.
- **Residual limitations for future-work documentation (paper §6.4 revision):** (a) Rule 10 asymmetric-confidence semantics still require BOTH sides to produce confident asset-class signals — if Polymarket titles adopt non-standard commodity wording the filter may miss novel false-match variants; (b) MAX_LIVE_PAIRS=2000 is still the hard global cap — if additional asset classes are onboarded, the reservation pattern will need to be generalized rather than hard-coded per category.

---
*Phase: 15-live-commodity-matching-engineering-fixes*
*Plan: 03*
*Completed: 2026-04-24*

## Self-Check: PASSED

- `.planning/phases/15-live-commodity-matching-engineering-fixes/15-03-SUMMARY.md` → FOUND on disk
- `src/live/market_discovery.py` → FOUND (H1 fix site)
- `src/features/category.py` → FOUND (H5 fix site)
- `src/live/collector.py` → FOUND (Rule 3 reservation fix site)
- `data/live/active_matches.json` → FOUND (regenerated post-fix state)
- `data/live/pair_mapping.json` → FOUND (regenerated post-fix state)
- `data/live/position_history.jsonl` → FOUND (24h SCC window closures)
- Commit `38d7970` (Task 1 fix) → FOUND via `git log --oneline --all | grep 38d7970`
- Commit `d217ff1` (Task 2 data regeneration) → FOUND via `git log --oneline --all | grep d217ff1`
- Post-window count re-verified: 1,224 closed non-KXWTIMAX commodity positions (matches orchestrator's verified numbers verbatim)
