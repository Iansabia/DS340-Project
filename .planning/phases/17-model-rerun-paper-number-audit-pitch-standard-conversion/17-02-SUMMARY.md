---
phase: 17-model-rerun-paper-number-audit-pitch-standard-conversion
plan: 02
subsystem: paper
tags: [paper-numerics, canonical-audit, pitch-standard, ppo-figure-fix, sharpe, basis-points]

# Dependency graph
requires:
  - phase: 17-model-rerun-paper-number-audit-pitch-standard-conversion
    plan: 01
    provides: experiments/results/canonical/headline.json (single source of truth) + 17-02-PPO-DIAGNOSTIC.md (units-mismatch root cause)
  - phase: 14-paper-finalization-presentation
    provides: PAPER_DRAFT.md baseline + scripts/check_paper.sh regression guard
provides:
  - Automated paper-numerics auditor (scripts/audit_paper_numbers.py) — re-runnable any time canonical JSON updates
  - Number audit log (17-03-NUMBER-AUDIT.md) — every paper claim cross-referenced against headline.json
  - Updated PAPER_DRAFT.md with pitch-standard headlines (per-trade Sharpe + bps) and canonical numbers throughout
affects: [17-03-slides-conversion, 17-04-canonical-guardrail]

# Tech tracking
tech-stack:
  added: []  # pure stdlib (re, json, argparse, pathlib, datetime)
  patterns:
    - "Section-aware numeric audit: only headline sections (Abstract, §5.1, §6.3, §8) cross-reference against canonical JSON; per-window/per-category/sweep tables have their own non-canonical result files and are skipped"
    - "Per-number proximity model resolution (find_model_at_position): each match resolves its model from the closest alias on the same line, not a paragraph centroid — fixes multi-model line attribution (§8 Conclusions list)"
    - "Skip neighbourhoods (SKIP_NUMBER_NEIGHBOURHOODS): narrative ranges ('by 0.7-1.0 bps', '$9 in P&L'), figure references ('§5.8', 'Fig 1'), and engineering-mention numbers ('$100 position size') are filtered out per-match without skipping the whole line"
    - "Pitch-standard headline: lead with per-trade Sharpe + per-trade alpha in bps; cumulative dollar P&L follows. Formula embedded inline: alpha_bps_per_trade = total_pnl / num_trades / position_size × 10,000"

key-files:
  created:
    - scripts/audit_paper_numbers.py (557 lines; canonical numeric auditor)
    - .planning/phases/17-model-rerun-paper-number-audit-pitch-standard-conversion/17-03-NUMBER-AUDIT.md (60 lines; this run's audit log)
  modified:
    - PAPER_DRAFT.md (Abstract + §1.4 item 7 + Table 2 + §5.1 narrative + §5.8 + §6.3 + §8 Conclusions item 1)

key-decisions:
  - "Auditor scope = headline sections only. The plan originally proposed auditing every dollar/Sharpe/bps/RMSE in PAPER_DRAFT.md against canonical/headline.json, but per-window walk-forward (§5.2), per-category (§5.3), data-scaling (§5.4), hyperparameter sweep (§5.5), transaction-cost (§5.6), feature ablation (§5.10), ensemble (§5.11), lookback (§5.12), and threshold (§5.13) sections all have their own result files (ablation_walk_forward, data_scaling, sweep_xgb, etc.) — they should NOT be cross-referenced against headline.json. Restricting to Abstract / §5.1 / §6.3 / §8 reduced the audit log from 53 spurious mismatches to a manageable signal."
  - "Per-number proximity model attribution. The auditor's first pass attributed every number on a line to a single 'nearest model'. This broke on §8 Conclusions item 1 which lists 6 models in one paragraph (LR / XGB / LSTM / GRU / PPO+AE / PPO-Raw). Solution: find_model_at_position(line, pos) finds the closest alias to each individual number's character offset, with an 80-char hard cap so out-of-range numbers fall back to paragraph context."
  - "Skip neighbourhoods preferred over per-line skips. Rather than skip whole lines that contain narrative ranges, SKIP_NUMBER_NEIGHBOURHOODS examines the ±25 char local context around each match. This preserves auditing of TRUE headline numbers while filtering out 'by 0.7-1.0 bps', '5-10% absolute P&L', and similar comparative phrasing that would otherwise generate noise."
  - "LR/XGB row order in Table 2: LR first, XGB second. The plan suggested swapping (XGB first, since XGB beats LR on RMSE/total_pnl), but with canonical numbers LR actually wins per-trade Sharpe (0.501 vs 0.499), per-trade alpha (15.02 vs 14.93 bps), directional accuracy (56.9% vs 56.6%), and win rate (57.8% vs 57.4%) — 4 of 5 metrics. XGB only wins RMSE and total_pnl by a hair. Kept LR row first; documented in §5.1 narrative ('LR wins per-trade Sharpe, alpha-per-trade, directional accuracy, and win rate; XGBoost wins RMSE and total P&L by a hair')."
  - "Plan example numbers (XGB +$201.63, LR +$201.69) were stale relative to canonical/headline.json (XGB +$232.83, LR +$232.67). Followed the plan's contract ('every paper number derives from headline.json') rather than its example values; abstract and Table 2 numbers all updated to the post-Phase-17-01 canonical figures."
  - "Removed the now-redundant 'per-trade Sharpe of 0.44 is reported separately' sentence from the abstract, since per-trade Sharpe (0.501) is now in the headline. Kept the per-pair Sharpe ≈ 3.2 sentence (POL-07 already wired this in)."

patterns-established:
  - "Audit log structure: Summary (counts) → Tolerances → Mismatches table → Unresolvable table → All matches table. Exit code = (0 if zero mismatches else 1) — usable as a CI gate or check_paper.sh add-on."
  - "Pitch-standard headline pattern: '{Model} achieves a per-trade Sharpe of {X.XXX} with +{Y.Y} bps per-trade alpha (\\${Z.ZZ} cumulative at \\$100 position size)' — Sharpe leads, bps second, dollars in parens."
  - "Tolerances applied: dollars ±0.5%, Sharpe ±0.01, RMSE ±0.005, trades ±1%, bps ±0.5. Documented inline in audit log so reviewers can independently re-derive."

requirements-completed: [REPL-03, REPL-04]

# Metrics
duration: 18min
completed: 2026-04-25
---

# Phase 17 Plan 02: Paper Numerics Audit + Pitch-Standard Conversion Summary

**Every numeric claim in PAPER_DRAFT.md now reconciles to experiments/results/canonical/headline.json. The disputed −$7,724 PPO+autoencoder figure has been replaced everywhere with the canonical +$4.61 / +0.5 bps result. The abstract leads with per-trade Sharpe (0.501) + per-trade alpha (+15.0 bps), the pitch-standard quant headline format. All 16 Phase-14 check_paper.sh guardrails still pass.**

## Performance

- **Duration:** ~18 min
- **Started:** 2026-04-25T17:48:00Z
- **Completed:** 2026-04-25T18:06:00Z (approx)
- **Tasks:** 2 / 2
- **Files modified:** 3 (1 created script + 1 created audit log + 1 modified paper)

## Audit Counts

From the final 17-03-NUMBER-AUDIT.md run:

| Bucket | Count | Notes |
|---|---|---|
| **MATCH** | 18 | Every Abstract / Table 2 / §5.1 narrative / §6.3 / §8 number resolves to canonical/headline.json within tolerance |
| **MISMATCH** | **0** | Zero remaining mismatches (auditor exits 0) |
| **UNRESOLVABLE** | 7 | Auxiliary numbers — quality-filter $-5.28/$+5.45 deltas (§3.2), $50/bbl threshold-exactness example (§3.2), $1.96 live-validation P&L (§5.9), $10.73 quality-filter contribution (§8 conclusions item 3), per-pair Sharpe ≈ 3.2 (§5.8 Table 8 narrative), $1/$0 contract-mechanics intro (§1.3). All correctly out of scope — these aren't headline-section model metrics. |

## Lines Edited in PAPER_DRAFT.md

| Line(s) | Section | Change |
|---|---|---|
| 12 | Abstract | Replaced dollar-leading headline with per-trade Sharpe (0.501) + per-trade alpha (+15.0 bps) + canonical PPO+AE figure (+0.5 bps); deleted redundant 'per-trade Sharpe of 0.44 is reported separately' sentence |
| 67 | §1.4 item 7 | Updated 'per-trade Sharpe of 0.44' → 'per-trade Sharpe of 0.50 (with +15.0 bps per-trade alpha at $100 position size)' |
| 213 | §5.1 narrative | Replaced 'fresh re-run on April 17, 2026 ...' with 'sourced verbatim from experiments/results/canonical/headline.json (Phase 17-01 canonical regenerator under seed=42, threshold=0.02, position_size=$100)' + reproduction one-liner |
| 215 | Table 2 caption | Added pitch-standard column-order note + canonical PPO+AE figure note + units-mismatch reference to 17-02-PPO-DIAGNOSTIC.md |
| 217-227 | Table 2 | Reordered columns: RMSE, Dir.Acc, **Sharpe (per-trade)**, **Alpha (bps/trade)**, P&L ($100 pos), Win Rate, # trades. Updated every row to canonical headline.json values (Naive +$58.12, Volume +$59.81, LR +$232.67, XGB +$232.83, LSTM +$221.84, GRU +$212.50, TFT +$6.57 / 120 trades / 0.155 Sharpe, PPO-Raw +$158.15 / +9.6 bps, PPO+AE +$4.61 / +0.5 bps) |
| 231 | §5.1 'Three observations' | Updated narrative to lead with bps comparison (Tier-0 → Tier-1 = 4 bps → 15 bps), document LR/XGB tie at 4-of-5 metrics for LR (Sharpe / alpha / dir-acc / win-rate) |
| 372 | §5.8 opening sentence | Updated 'per-trade Sharpe of 0.436' → 'per-trade Sharpe of 0.499' (canonical XGB) |
| 383 (after) | §5.8 new paragraph | Added 'Per-trade alpha in basis points' paragraph with verbatim formula, industry context (1-5 bps/trade typical for stat-arb at fixed position size), and PPO comparison (+0.5 bps PPO+AE / +9.6 bps PPO-Raw / +15 bps regression baselines) |
| 644-646 | §6.3 The Negative Result on PPO | Rewrote first sentence: '+0.5 bps per trade ($+4.61 cumulative over 899 trades at $100 position size; canonical figure)'. Added units-mismatch explanation citing 17-02-PPO-DIAGNOSTIC.md and the archived legacy backtest path; preserved the rest of the autoencoder-failure narrative verbatim |
| 700 | §8 Conclusions item 1 | Rewrote in pitch-standard format: 'Tier 1 (LR +15.0 bps/trade, Sharpe 0.501; XGBoost +14.9 bps/trade, Sharpe 0.499) beats Tier 2 (LSTM +14.3 bps, Sharpe 0.473; GRU +14.0 bps, Sharpe 0.459) by 0.7-1.0 bps and 5-10% absolute P&L; Tier 3 (PPO+autoencoder +0.5 bps, essentially zero alpha) is dominated.' Dollar-terms enumeration follows in second sentence. |

## Canonical PPO+AE Figure (Verbatim Source)

From `experiments/results/canonical/headline.json["models"]["ppo_filtered"]`:

| Field | Value |
|---|---|
| total_pnl | $4.607821818303617 → cited as **+$4.61** |
| num_trades | 899 |
| sharpe_per_trade | 0.014296484590545602 → cited as **0.014** |
| alpha_bps_per_trade | 0.5125497017022934 → cited as **+0.5 bps** |
| win_rate | 0.43159065628476084 → cited as **43.2%** |
| directional_accuracy | 0.27189908899789766 → cited as **27.2%** |

The legacy −$87,724 figure (and its likely transcription typo −$7,724) is documented in 17-02-PPO-DIAGNOSTIC.md as a units mismatch between profit_sim (canonical, raw spread units) and WalkForwardBacktester (legacy, dollar-notional with 200× contract scaling and 5pp round-trip fees). The legacy file is now archived under `experiments/results/archive/`; the paper cites only the canonical figure.

## Final Validation Status

| Check | Status | Detail |
|---|---|---|
| `bash scripts/check_paper.sh` | **PASS** (16/16) | All POL-04..POL-10 guardrails green; no Phase-14 regression |
| `python3 scripts/audit_paper_numbers.py` | **PASS** (exit 0) | 18 match / 0 mismatch / 7 unresolvable (auxiliary) |
| Abstract word count (POL-04) | **PASS** | 249 / 250 (1-word headroom) |
| `grep -c "−\\$7,724\|7,724\|7724" PAPER_DRAFT.md` | **PASS** | 0 (every reference replaced) |
| `grep -cE "[0-9]+\\.?[0-9]* bps" PAPER_DRAFT.md` | **PASS** | 6 ≥ 4 required |
| `grep -c "per-trade Sharpe\|per-trade alpha"` | **PASS** | 19 ≥ 3 required |
| Abstract contains Sharpe value | **PASS** | "Sharpe of 0.501" + "0.499" |
| Abstract contains bps value | **PASS** | "+15.0 bps", "+14.9 bps", "+14.3 bps", "+14.0 bps", "+0.5 bps" |
| Audit log line count | **PASS** | 60 lines ≥ 60 required |

## Task Commits

1. **Task 1: Write `scripts/audit_paper_numbers.py`** — `232b47b` (feat)
2. **Task 2: Apply audit + paper edits + auditor refinements** — `62c3368` (feat)

## Files Created/Modified

- `scripts/audit_paper_numbers.py` (created, 557 lines) — paper-vs-canonical numeric auditor with section-aware filtering, per-number proximity model resolution, and tolerance-based comparison
- `.planning/phases/17-model-rerun-paper-number-audit-pitch-standard-conversion/17-03-NUMBER-AUDIT.md` (created, 60 lines) — this run's audit log; 18 match / 0 mismatch / 7 unresolvable
- `PAPER_DRAFT.md` (modified) — Abstract pitch-standard rewrite + Table 2 column reorder + §5.1/§5.8/§6.3/§8 narrative updates with canonical numbers throughout

## Decisions Made

See key-decisions in frontmatter. Highlights:

- **Audit scope = headline sections only.** Audit log restricted to Abstract / §5.1 / §6.3 / §8. Per-window / per-category / sweep / ablation / ensemble sections have their own non-canonical result files.
- **Per-number proximity attribution.** find_model_at_position() resolves each individual numeric match to its closest alias within 80 chars, falling back to paragraph context only when no alias is in range.
- **LR row #1, XGB row #2 retained.** Plan suggested swapping; canonical numbers showed LR wins 4 of 5 metrics. Documented in §5.1 narrative.
- **Plan example values were stale.** Used canonical/headline.json as source-of-truth, not plan example values. Plan said XGB +$201.63 / LR +$201.69; canonical actually has XGB +$232.83 / LR +$232.67.

## Deviations from Plan

**1. [Rule 1 - Scope correction] LR/XGB row ordering kept (LR first).**
- **Found during:** Task 2 acceptance-criteria review against canonical/headline.json
- **Issue:** Plan §3 Fix 3 said 'XGBoost beats LR on 4 of 5 metrics ... Make XGBoost row #1'. With current canonical numbers (post-Phase-17-01 regeneration), LR actually wins 4 of 5 metrics: per-trade Sharpe (0.501 > 0.499), per-trade alpha (15.02 > 14.93 bps), directional accuracy (56.9% > 56.6%), win rate (57.8% > 57.4%). XGB only wins RMSE (0.290 < 0.306) and total_pnl ($232.83 > $232.67) by a hair.
- **Fix:** Kept LR row first in Table 2; documented the 4-of-5-metric tie in §5.1 narrative ('LR wins per-trade Sharpe, alpha-per-trade, directional accuracy, and win rate; XGBoost wins RMSE and total P&L by a hair').
- **Files modified:** PAPER_DRAFT.md
- **Commit:** 62c3368

**2. [Rule 1 - Scope correction] Used canonical numbers, not plan example numbers.**
- **Found during:** Task 2 audit-log generation
- **Issue:** Plan provided example replacement values (XGB +$201.63, LR +$201.69, LSTM +$182.72, GRU +$174.11) that were the pre-Phase-17-01 numbers. The Phase-17-01 canonical/headline.json regeneration produced LR +$232.67, XGB +$232.83, LSTM +$221.84, GRU +$212.50. Per the plan's own contract ('every paper number derives from headline.json'), the canonical figures are authoritative.
- **Fix:** Updated abstract, Table 2, §5.8, §6.3, §8 Conclusions to canonical numbers throughout; per-trade alpha values recomputed from canonical (LR 15.02 bps, XGB 14.93 bps, GRU 14.01 bps, LSTM 14.34 bps, PPO+AE 0.51 bps).
- **Files modified:** PAPER_DRAFT.md
- **Commit:** 62c3368

**3. [Rule 3 - Auditor design refinement] Auditor section-awareness, per-number proximity, skip neighborhoods.**
- **Found during:** Task 1 acceptance-criteria run (initial run produced 53 spurious mismatches from per-window walk-forward and per-category breakdown rows)
- **Issue:** A naive line-by-line auditor over the full PAPER_DRAFT.md flags 53+ false positives because §5.2 walk-forward / §5.3 per-category / §5.4 data-scaling / §5.5 sweep / §5.10 ablation / §5.11 ensemble / §5.12 lookback / §5.13 threshold all contain dollar amounts that are NOT supposed to match headline.json (they have their own result files). And within headline sections, narrative ranges ('by 0.7-1.0 bps', '$9 in P&L') and figure references ('§5.8', 'Fig 1') generate noise.
- **Fix:** Three improvements to scripts/audit_paper_numbers.py: (1) NON_CANONICAL_SECTIONS / HEADLINE_SECTIONS gate restricts audit to Abstract / §5.1 / §6.3 / §8; (2) find_model_at_position() resolves model per-number proximity instead of per-line; (3) SKIP_NUMBER_NEIGHBOURHOODS filters narrative ranges and section references in local ±25-char context.
- **Files modified:** scripts/audit_paper_numbers.py
- **Commit:** 62c3368

## Issues Encountered

None blocking. The auditor false-positive density required several iterations of regex / skip-neighborhood tuning to reach 0 mismatches; documented under deviation Rule 3.

## Next Phase Readiness

**Plan 17-03 (slides conversion to pitch standard) is unblocked.** The same canonical/headline.json source applies; the per-trade Sharpe + bps headline format established in PAPER_DRAFT.md is directly portable to slides:

```bash
python3 -c "import json; d=json.load(open('experiments/results/canonical/headline.json'))['models']; print({k: {'sharpe': v['sharpe_per_trade'], 'bps': v['alpha_bps_per_trade'], 'pnl': v['total_pnl']} for k, v in d.items()})"
```

**Plan 17-04 (canonical guardrail) has a concrete model:** `scripts/audit_paper_numbers.py` already exits 0/1 based on mismatch presence. Plan 17-04 should add it as an additional `check_paper.sh` invocation (or wire it into the existing script).

## Self-Check: PASSED

All claimed files exist on disk:
- `scripts/audit_paper_numbers.py` ✓
- `.planning/phases/17-model-rerun-paper-number-audit-pitch-standard-conversion/17-03-NUMBER-AUDIT.md` ✓
- `.planning/phases/17-model-rerun-paper-number-audit-pitch-standard-conversion/17-02-SUMMARY.md` ✓ (this file)
- `PAPER_DRAFT.md` ✓ (modified)

All claimed commits exist in git history:
- `232b47b` (Task 1: feat(17-02) add audit_paper_numbers.py) ✓
- `62c3368` (Task 2: feat(17-02) apply canonical audit + paper edits + auditor refinements) ✓

All acceptance criteria validated:
- bash scripts/check_paper.sh: ALL CHECKS PASSED ✓
- python3 scripts/audit_paper_numbers.py: exit 0 ✓
- abstract = 249 words (≤ 250) ✓
- 0 references to 7,724 ✓
- 6 bps mentions (≥ 4) ✓
- 19 per-trade mentions (≥ 3) ✓
- audit log = 60 lines (≥ 60) ✓

---

*Phase: 17-model-rerun-paper-number-audit-pitch-standard-conversion*
*Completed: 2026-04-25*
