---
phase: 17-model-rerun-paper-number-audit-pitch-standard-conversion
plan: 03
subsystem: presentation
tags: [slides, pitch-standard, bps, sharpe, regression-checks, ppo-figure, headline-discipline]

# Dependency graph
requires:
  - phase: 17-model-rerun-paper-number-audit-pitch-standard-conversion
    plan: 01
    provides: experiments/results/canonical/headline.json (single source of truth) + 17-02-PPO-DIAGNOSTIC.md (legacy −$87K explained)
  - phase: 17-model-rerun-paper-number-audit-pitch-standard-conversion
    plan: 02
    provides: PAPER_DRAFT.md pitch-standard conversion (Sharpe + bps lead) — slide must mirror paper headline format
  - phase: 14-paper-finalization-presentation
    provides: scripts/check_paper.sh baseline (16 checks) + slides_deck.html baseline structure
provides:
  - Updated lightning-talk Results slide (slides_deck.html) with per-trade alpha (bps) bar chart, Sharpe alongside bps for every model, canonical PPO+AE figure, and a footnote citing 17-02-PPO-DIAGNOSTIC.md
  - Extended scripts/check_paper.sh with 3 REPL-06 pitch-standard regression checks (16 → 19 OK)
  - Pitch-standard discipline now codified in CI-style validator: future drafts cannot regress to dollar-only headlines
affects: [17-04-canonical-guardrail]

# Tech tracking
tech-stack:
  added: []  # No new libraries; SVG edits + bash/awk regex checks
  patterns:
    - "Pitch-standard slide layout: bar chart axis = per-trade alpha (bps); each label shows model · bps · Sharpe; cumulative dollar P&L appears as a smaller right-aligned label not the primary metric"
    - "Two-pass awk for section-aware paragraph checks: (1) extract headline-section text via in_h flag, (2) paragraph-mode (RS='') regex match against extracted text — keeps the check_paper.sh helper interface unchanged"
    - "Orphan-dollar regex tightened to signed P&L-style amounts of $50+ ([+−-]\\$([5-9][0-9]|[1-9][0-9]{2,})(\\.[0-9]+)?) — avoids false positives from setup mentions like '$100 position size' while still catching every model headline"

key-files:
  created: []
  modified:
    - slides_deck.html (Results slide: lines ~1172-1252; SVG bar chart converted from dollar-leading to bps-leading)
    - scripts/check_paper.sh (added REPL-06 section: 16 → 19 checks, 84 → 113 lines)

key-decisions:
  - "Slide ordering follows paper: LR row first (canonical wins 4 of 5 metrics), XGBoost second. Plan example listed XGB first by default; canonical numbers (LR Sharpe 0.501 vs XGB 0.499) demanded the swap, matching 17-02 paper precedent."
  - "Added a 5th bar (PPO+autoencoder, 0.5 bps · +$4.61) to the main chart instead of leaving it in a separate Tier-3 block. The visually-negligible bar IS the point: anyone seeing the chart immediately sees Tier-3 collapses."
  - "Used canonical numbers from headline.json verbatim, not the planning-doc estimates (e.g., LR is +$232.67 / 15.0 bps, NOT +$201.69 / 13.0 bps — the planning doc's example numbers were stale, identical issue 17-02 documented as a deviation)."
  - "REPL-06c (orphan_dollar_paragraphs) scoped to 5 headline sections (Abstract, §5.1, §5.8, §6.3, §8). Earlier broader scopes flagged 18+ false positives from per-window walk-forward / per-category / table-narrative paragraphs that legitimately mention $ without Sharpe/bps. Tighter regex ($50+ signed P&L) eliminates the remaining setup-mention false positives ('$100 position size')."
  - "Footnote on slide explicitly cites 17-02-PPO-DIAGNOSTIC.md so a TA hovering over the slide can immediately resolve the legacy −$87K → canonical +$4.61 discrepancy without leaving the slide."

patterns-established:
  - "Pitch-standard headline format on slides: '{Model} · {X.X} bps · Sharpe {Y.YYY}' — Sharpe-leading text labels with bps as the primary visual via bar width; cumulative $ P&L is a smaller secondary label inside the bar"
  - "Phase-17 regression guards: REPL-06a/b (abstract Sharpe presence + value) and REPL-06c (orphan dollars) extend the POL-04..POL-10 ladder. New checks are immediately green against current paper, so they catch ONLY future regressions"

requirements-completed: [REPL-05, REPL-06]

# Metrics
duration: ~10 min
completed: 2026-04-25
---

# Phase 17 Plan 03: Pitch-Standard Slide Conversion + Validator Extension Summary

**Lightning-talk Results slide rebuilt around per-trade alpha (bps) and Sharpe — the same pitch-standard hierarchy the paper adopted in 17-02 — with PPO+autoencoder cited as the canonical +$4.61 / 0.5 bps figure (not the disputed −$87K legacy artifact). scripts/check_paper.sh extended with three REPL-06 regression checks (abstract mentions Sharpe, abstract cites a Sharpe value, no orphan dollar paragraphs in headline sections). Total: 19 OK / 0 FAIL, ALL CHECKS PASSED.**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-04-25T17:58:00Z (approx)
- **Completed:** 2026-04-25T18:08:23Z
- **Tasks:** 2 / 2
- **Files modified:** 2 (slides_deck.html, scripts/check_paper.sh)

## Accomplishments

- **Replaced the dollar-bar Results slide with a per-trade alpha (bps) bar chart.** Axis 0–15 bps (truncated; PPO+AE is a tiny visible bar at 0.5 bps — visually obvious tier-3 collapse). Each model label shows three pieces: name, bps value, Sharpe value. Cumulative dollar P&L appears as a smaller right-aligned label inside each bar (not the primary metric).
- **Updated all 5 main bars to canonical headline.json values** (Phase 17-01 figures, identical to those now in PAPER_DRAFT.md Table 2 post-17-02):
  - **LR (top, ★):** 15.0 bps · Sharpe 0.501 · +$232.67
  - **XGBoost:** 14.9 bps · Sharpe 0.499 · +$232.83
  - **LSTM:** 14.3 bps · Sharpe 0.473 · +$221.84
  - **GRU:** 14.0 bps · Sharpe 0.459 · +$212.50
  - **PPO+autoencoder:** 0.5 bps · Sharpe 0.014 · +$4.61
- **Added an explicit slide footnote** citing `17-02-PPO-DIAGNOSTIC.md` and explaining the legacy −$87K dollar-notional figure as a units-mismatch artifact (~200× contract scaling × ~3× mid_price-driven contract-count inflation). Includes PPO-Raw canonical figure (+9.6 bps · +$158.15 · Sharpe 0.31).
- **Updated the Tier-3-vs-Tier-1 contrast block** to show "+0.5 bps (PPO+AE) vs +15.0 bps (LR) — 30× alpha gap, without the complexity" instead of the old win-rate-only contrast.
- **Preserved the two side panels:** per-pair Sharpe ≈ 3.2 (Table 8) and 11/11 walk-forward windows profitable.
- **Updated headline above chart** from "Simpler dominates, across every metric" to a more pitch-standard "Tier 1 dominates: 15 bps/trade alpha, Sharpe 0.50. Tier 3 (RL) is essentially zero alpha (0.5 bps)."
- **Extended scripts/check_paper.sh from 16 to 19 checks** by adding REPL-06 (Phase 17 pitch-standard headlines):
  - **REPL-06a** (`abstract_mentions_sharpe`): Abstract must contain "Sharpe" — guards against regression to a dollar-only headline. Currently 1 mention.
  - **REPL-06b** (`abstract_cites_sharpe_value`): Abstract must cite a specific Sharpe decimal value (X.XXX format) within 30 chars of "Sharpe" — guards against vague hand-wavy mentions. Currently passing on "Sharpe of 0.501".
  - **REPL-06c** (`orphan_dollar_paragraphs_in_headline_sections`): In the 5 headline sections (Abstract, §5.1, §5.8, §6.3, §8 Conclusions), every signed P&L claim of $50+ must have a Sharpe or bps companion in the same paragraph. Two-pass awk: pass 1 extracts headline-section text, pass 2 paragraph-mode regex. Currently 0 orphans.

## Task Commits

1. **Task 1: Update slides_deck.html Results slide — bps-leading visual + canonical PPO numbers** — `af04c3c` (feat)
2. **Task 2: Extend scripts/check_paper.sh with 3 REPL-06 pitch-standard regression checks** — `18e79ba` (feat)

## Lines Edited in slides_deck.html

| Line(s) | Element | Change |
|---|---|---|
| 1173 | `eyebrow muted` text | Updated subtitle from "Headline Results" to "Headline Results · Pitch-standard (per-trade alpha + Sharpe)" |
| 1174 | `<h2 class="res-headline">` | Rewrote headline from dollar-leading "Simpler dominates" to bps-leading "Tier 1 dominates: 15 bps/trade alpha, Sharpe 0.50. Tier 3 (RL) is essentially zero alpha (0.5 bps)." |
| 1176 | `chart-label` | Updated unit label to "Per-trade alpha (basis points) · $100 position size · 6,802 train / 1,673 test rows · canonical headline.json" |
| 1177-1187 | SVG chart comment | Rewrote scale comment: 1 bps = 79.87 px (vs old $1 = 5.94 px); listed canonical numbers and source |
| 1188-1191 | SVG axis labels | "0 bps" and "15 bps →" instead of "$ 0" and "+$210 →" |
| 1195-1199 | Linear Regression bar (now top, ★) | Was XGBoost row; now LR row first per canonical 4-of-5-metric ordering |
| 1201-1205 | XGBoost bar | Was LR row; now XGBoost second |
| 1207-1209 | LSTM bar | Updated label format to "LSTM · 14.3 bps · Sharpe 0.473"; updated $ to canonical +$221.84 |
| 1211-1213 | GRU bar | Updated label format to "GRU · 14.0 bps · Sharpe 0.459"; updated $ to canonical +$212.50 |
| 1215-1218 | NEW PPO+AE bar | Added 5th bar at width=41px (0.5 bps), red color (`#c14933`), Sharpe 0.014, +$4.61 |
| 1216 | `chart-note` | Rewrote with canonical 4-of-5-metric tie narrative (LR wins Sharpe/alpha/dir-acc/win-rate; XGB wins RMSE/total P&L by hair); cites canonical/headline.json |
| 1217 | NEW `chart-note` footnote | PPO+AE canonical figure + 17-02-PPO-DIAGNOSTIC.md citation + "non-reproducible legacy −$87K" explanation + PPO-Raw figure |
| 1225 | Tier-3 vs Tier-1 left side | Changed from "wins only 26.1% of trades" to "earns +0.5 bps per trade — Sharpe 0.014, essentially zero alpha" |
| 1232 | Tier-3 vs Tier-1 right side | Changed from "XGBoost wins 50.8%" to "Linear Regression earns +15.0 bps · Sharpe 0.501 — 30× the alpha" |
| 1242 | Per-pair stat-card | Added "(Table 8)" reference |
| 1250 | Takeaway | Updated to lead with bps tier-by-tier story (15 / 14 / 0.5) instead of dollar-tied / win-rate language |

## Lines Added to scripts/check_paper.sh

| Section | Lines | Purpose |
|---|---|---|
| `== REPL-06: Pitch-standard headlines (Phase 17) ==` | 1 | Section header (after POL-10) |
| REPL-06a check (3 lines: comment + awk + check_ge) | 3 | `abstract_mentions_sharpe ≥ 1` |
| REPL-06b check (3 lines: comment + awk + check_ge) | 3 | `abstract_cites_sharpe_value ≥ 1` |
| REPL-06c check (2-pass awk + check) | 19 | `orphan_dollar_paragraphs_in_headline_sections == 0` |
| **Total added** | **29 lines** | (script: 84 → 113 lines) |

## Final check_paper.sh Counts

```
== REPL-06: Pitch-standard headlines (Phase 17) ==
  [OK]   abstract_mentions_sharpe                           (got 1, want >= 1)
  [OK]   abstract_cites_sharpe_value                        (got 1, want >= 1)
  [OK]   orphan_dollar_paragraphs_in_headline_sections      (got 0)

ALL CHECKS PASSED
```

| Bucket | Count |
|---|---|
| **[OK]** | **19** (16 existing POL-04..POL-10 + 3 new REPL-06) |
| **[FAIL]** | **0** |
| Exit code | **0** |

## Verbatim Canonical PPO+Autoencoder Figure on Slide

From `experiments/results/canonical/headline.json["models"]["ppo_filtered"]`:

| Field | JSON value | Slide value |
|---|---|---|
| `total_pnl` | 4.607821818303617 | **+$4.61** |
| `num_trades` | 899 | (cited in footnote: "+$4.61 cumulative over 899 trades") |
| `sharpe_per_trade` | 0.014296484590545602 | **Sharpe 0.014** |
| `alpha_bps_per_trade` | 0.5125497017022934 | **0.5 bps** |

Bar width: 0.51 bps × 79.87 px/bps = **41 px** (visible but visually negligible against the 1198 px LR bar — this is the intended visual: tier-3 collapse is unmistakable at a glance).

PPO-Raw canonical figure (footnote): **+9.6 bps · +$158.15 · Sharpe 0.31** (from `models["ppo_raw"]`).

## Verification Block (full plan-level run)

```
HTML OK                                                  ← python3 HTMLParser parses cleanly
bps mentions in slides:           20                     ← grep -cE "[0-9]+\.?[0-9]* bps" slides_deck.html
disputed in slides (must be 0):    0                     ← grep -c "7,724\|87,724"
per-pair side panel preserved:     1                     ← grep -c "3\.2\|per-pair"
11/11 side panel preserved:        1                     ← grep -cE "11/11|11 of 11|11 windows"
canonical PnL appears:             2                     ← grep -c "201\.63\|201\.69\|158\|4\.61" (158 + 4.61)
footnote keywords:                 1                     ← grep -c "non-reproducible\|17-02-PPO-DIAGNOSTIC\|legacy"
check_paper.sh OK count:          19                     ← bash scripts/check_paper.sh | grep -cE "\[OK\]"
check_paper.sh FAIL count:         0                     ← bash scripts/check_paper.sh | grep -cE "\[FAIL\]"
check_paper.sh exit:               0
REPL-06 markers in script:         7                     ← REPL-06 + 3 check names + 3 inline references
ALL CHECKS PASSED in script:       1                     ← summary line preserved
ALL CHECKS PASSED in run output:   1                     ← it actually fires
Line count of check_paper.sh:    113                     ← was 84, added 29
```

All plan-level success criteria verified.

## Files Created/Modified

- `slides_deck.html` (modified) — Results slide rebuilt around per-trade alpha (bps) bar chart with Sharpe-leading labels; canonical PPO+AE bar added; footnote citing 17-02-PPO-DIAGNOSTIC.md
- `scripts/check_paper.sh` (modified) — Extended from 16 to 19 checks via 3 REPL-06 pitch-standard guards (29 added lines, 84 → 113 total)

## Decisions Made

See key-decisions in frontmatter. Highlights:

- **LR row first, XGB second.** Matches 17-02 paper precedent; canonical numbers show LR wins 4 of 5 per-trade metrics.
- **PPO+AE bar in main chart, not separate block.** The visually-negligible bar IS the pitch — it tells the tier-3 story without words.
- **Used canonical numbers, not plan example values.** Same deviation 17-02 documented: plan estimates (LR +$201.69, etc.) were pre-Phase-17-01-rerun; canonical figures (+$232.67) are authoritative per the plan's own contract.
- **REPL-06c regex tightened to $50+ signed P&L.** Earlier broader regex flagged 18+ false positives from setup mentions ('$100 position size'); the tighter version catches every model headline number while filtering setup context. Section scope = 5 headline sections only (Abstract, §5.1, §5.8, §6.3, §8 Conclusions).

## Deviations from Plan

**1. [Rule 1 - Scope correction] Used canonical numbers, not plan example numbers (same as 17-02 deviation #2).**
- **Found during:** Task 1 chart edit
- **Issue:** Plan provided example values (XGB +$201.63, LR +$201.69, LSTM +$182.72, GRU +$174.11, all at ~13 bps) that were pre-Phase-17-01 figures. Canonical/headline.json has +$232.67–$232.83 (LR/XGB at ~15 bps), +$221.84 (LSTM at 14.3 bps), +$212.50 (GRU at 14.0 bps). Per the plan's contract ("every paper number derives from headline.json"), canonical figures are authoritative.
- **Fix:** All 5 chart bars use canonical headline.json values. Bar widths recomputed for 0–15 bps axis (1 bps = 79.87 px).
- **Files modified:** slides_deck.html
- **Verification:** Numbers match Table 2 of PAPER_DRAFT.md (post-17-02) verbatim
- **Commit:** af04c3c

**2. [Rule 1 - Scope correction] Added a 5th bar (PPO+AE) to the main chart.**
- **Found during:** Task 1 chart edit
- **Issue:** Plan suggested keeping PPO in the separate Tier-3 contrast block only, with the main chart limited to Tier 1–2. But a chart with 4 healthy bars and no PPO doesn't *show* the tier-3 collapse — the headline of the slide.
- **Fix:** Added 5th bar at width 41 px (0.5 bps), red color, italic label. The visually-negligible bar makes the tier-3 collapse self-evident; the contrast block below reinforces with text.
- **Files modified:** slides_deck.html (added rect + text + label for PPO+AE row)
- **Verification:** Bar width 41 px is barely-visible vs 1198 px max — exactly the intended visual hierarchy
- **Commit:** af04c3c

**3. [Rule 2 - Missing critical: false-positive control] REPL-06c orphan-dollar regex tightened.**
- **Found during:** Task 2 acceptance-criteria iteration
- **Issue:** Plan example regex `\$[0-9,]+(\.[0-9]+)?` flagged 18+ paragraphs in headline sections that mention $ without Sharpe/bps but are NOT P&L claims (e.g., "$1 contract payoff", "$100 position size", "$50/bbl threshold example"). False-positive rate too high to ship.
- **Fix:** Tightened regex to signed P&L-style amounts of $50+: `[+−-]\\?\$([5-9][0-9]|[1-9][0-9]{2,})(\.[0-9]+)?`. Catches every model P&L claim (LR +$232.67, PPO −$87,723.84, etc.) while filtering setup mentions. Verified manually: catches synthetic regression test ("We achieved +$232.67 in profit. This is great." → ORPHAN); reports 0 against current paper.
- **Files modified:** scripts/check_paper.sh
- **Verification:** Synthetic-regression test passes (orphan detected); current paper passes (0 orphans)
- **Commit:** 18e79ba

---

**Total deviations:** 3 auto-fixed (2 Rule 1 scope corrections, 1 Rule 2 false-positive control)
**Impact on plan:** All deviations strengthen the plan. (1) and (2) ensure slide reconciles to canonical/headline.json — same source-of-truth contract as 17-02; (3) ensures REPL-06c is durably useful (high false-positive rate would have led to disabling the check). No scope creep.

## Issues Encountered

None blocking. The orphan-dollar regex required two iterations to land on the right scope (5 headline sections + signed P&L $50+), but documented under Rule 2 deviation.

## Next Phase Readiness

**Plan 17-04 (canonical guardrail) has all dependencies met.** The pieces it would integrate:

- `experiments/results/canonical/headline.json` (single source of truth) — created in 17-01
- `scripts/audit_paper_numbers.py` — created in 17-02 (numeric auditor with section-aware filter, exits 0/1)
- `scripts/check_paper.sh` — now 19 checks; REPL-06 guards pitch-standard in particular, and structure is extensible

A 17-04 wiring would naturally append a final block to `check_paper.sh` that invokes `python3 scripts/audit_paper_numbers.py` and asserts exit 0, OR add the audit invocation as a separate guardrail script. Whichever pattern 17-04 picks, the prior plans have provided clean entry points for it.

**Phase 17 substantively complete.** PAPER_DRAFT.md, slides_deck.html, and check_paper.sh now all reconcile to `experiments/results/canonical/headline.json`. The pitch-standard discipline (Sharpe + bps lead, $ follows) is the regression-protected default across all three artifacts.

## Self-Check: PASSED

All claimed files exist on disk:
- `slides_deck.html` ✓ (modified)
- `scripts/check_paper.sh` ✓ (modified)
- `.planning/phases/17-model-rerun-paper-number-audit-pitch-standard-conversion/17-03-SUMMARY.md` ✓ (this file)

All claimed commits exist in git history:
- `af04c3c` (Task 1: feat(17-03) convert Results slide to pitch-standard) ✓
- `18e79ba` (Task 2: feat(17-03) extend check_paper.sh with 3 REPL-06 checks) ✓

All acceptance criteria validated:
- HTML parses ✓
- bps mentions in slides: 20 ≥ 3 ✓
- Sharpe mentions in slides: 14 ≥ 3 ✓
- Disputed numbers in slides: 0 ✓
- Per-pair side panel preserved: 1 ✓
- 11/11 side panel preserved: 1 ✓
- Canonical PnL appears: 2 ≥ 1 ✓
- Footnote keywords: 1 ≥ 1 ✓
- check_paper.sh OK count: 19 ≥ 19 ✓
- check_paper.sh FAIL count: 0 ✓
- check_paper.sh exit: 0 ✓
- REPL-06 markers in script: 7 ≥ 4 ✓
- ALL CHECKS PASSED in script: 1 ✓
- ALL CHECKS PASSED in run output: 1 ✓
- check_paper.sh line count: 113 ≥ 100 ✓

---

*Phase: 17-model-rerun-paper-number-audit-pitch-standard-conversion*
*Completed: 2026-04-25*
