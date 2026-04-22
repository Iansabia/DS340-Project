---
phase: 11-tft-training
plan: 02
subsystem: models
tags: [tft, vsn-heatmap, negative-result, paper-integration, pytorch-forecasting, transformer]

# Dependency graph
requires:
  - phase: 11-01
    provides: "TFTPredictor implementation; experiments/results/tier2/TFT.json (negative result)"
provides:
  - experiments/figures/tft_variable_importance.png (VSN encoder importance heatmap, 300 DPI)
  - FINDINGS.md Finding 24 (TFT negative result, all actual numbers from TFT.json)
  - PAPER_DRAFT.md Table 2 TFT row (documented negative result, Branch B)
  - PAPER_DRAFT.md §4.1 TFT attempt note
  - PAPER_DRAFT.md §6.2.3 actual TFT result (replaces placeholder)
affects:
  - "Phase 13 Ensemble Formalization: converged=False — keep 4-variant ensemble (no TFT variant)"

# Tech tracking
tech-stack:
  added:
    - "pytorch_forecasting interpret_output(reduction='mean') for VSN encoder weight extraction"
    - "matplotlib barh chart for feature importance visualization"
  patterns:
    - "TFT VSN extraction: model.predict(loader, mode='raw') -> interpret_output -> encoder_variables"
    - "Placeholder figure pattern: always write artifact path even if extraction fails"

key-files:
  created:
    - experiments/extract_tft_heatmap.py (VSN heatmap extraction script, 213 lines)
    - experiments/figures/tft_variable_importance.png (162KB, 300 DPI, top-15 encoder features)
  modified:
    - FINDINGS.md (Finding 24 added — TFT negative result with all actual numbers)
    - PAPER_DRAFT.md (Abstract, §4.1, Table 2, §6.2.3, §7 Future Work)

key-decisions:
  - "TFT negative result is Branch B (converged=False in TFT.json) — 4-variant ensemble for Phase 13"
  - "Finding numbered 24 (not 26) — Findings 24 and 25 were not yet written; used actual next number"
  - "VSN extraction succeeds despite negative result: TFT attended to meaningful features (entropy=2.656, not degenerate)"
  - "Table 2 TFT row uses documented-negative format with dashes; footnote cites RMSE=0.3262 and heatmap path"

# Metrics
duration: 5min
completed: 2026-04-22
---

# Phase 11 Plan 02: TFT Paper Integration Summary

**VSN heatmap extracted (top-5: polymarket_amihud, polymarket_high, kalshi_roll_spread, price_divergence_pct, relative_time_idx); Finding 24 documents TFT negative result; PAPER_DRAFT.md Table 2 and §6.2.3 updated with Branch B (converged=False)**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-04-22T22:50:43Z
- **Completed:** 2026-04-22T22:55:54Z
- **Tasks:** 2 of 2
- **Files modified:** 4

## Accomplishments

- VSN heatmap generated: `experiments/extract_tft_heatmap.py` re-trains TFT seed=42, extracts encoder importance weights via `interpret_output(reduction='mean')`, saves 162KB PNG at 300 DPI
- Top-5 VSN encoder features: polymarket_amihud (1121.5), polymarket_high (803.3), kalshi_roll_spread (433.5), price_divergence_pct (250.9), relative_time_idx (229.2) — 51 total features extracted
- Finding 24 written to FINDINGS.md: complete negative result with per-seed table (RMSE 0.3264/0.3265/0.3258, P&L −1.37/−0.51/+6.57), attention audit (entropy=2.656, not degenerate), interpretive paragraph
- PAPER_DRAFT.md updated with all 4 Branch B changes (abstract, §4.1, Table 2, §6.2.3, §7)
- Old "TFT (which we did not train)" placeholder in §6.2.3 fully replaced with actual result

## VSN Heatmap Details

| Feature | Encoder Importance |
|---------|-------------------|
| polymarket_amihud | 1121.5 |
| polymarket_high | 803.3 |
| kalshi_roll_spread | 433.5 |
| price_divergence_pct | 250.9 |
| relative_time_idx | 229.2 |
| kalshi_high | 183.1 (approx) |
| ... | ... |

Artifact: `/Users/iansabia/Desktop/DS340 Project/experiments/figures/tft_variable_importance.png` (162KB, 2408×1536 px, 300 DPI)

**Attention is healthy despite negative predictive result:** entropy=2.656 is above the degenerate threshold of 1.966. TFT is attending to economically meaningful features (Amihud liquidity, price divergence, microstructure spreads) — the failure is a data-volume issue, not an architecture issue.

## Paper Sections Updated

| Section | Change | Lines (approx) |
|---------|--------|----------------|
| Abstract | TFT parenthetical in model list | ~12 |
| §4.1 | TFT attempt bullet after LSTM | ~149 |
| Table 2 | TFT† row + footnote with RMSE and heatmap ref | ~223-227 |
| §6.2.3 | Full paragraph with actual numbers, replaces placeholder | ~502-504 |
| §7 Future Work | Updated TFT item to reflect minimal config trained | ~540 |

## TFT Final Verdict for Phase 13

**converged=False** — Phase 13 (Ensemble Formalization) should use a **4-variant ensemble** (LR, XGBoost, GRU, LSTM). No TFT variant is justified.

Key numbers from `experiments/results/tier2/TFT.json`:
- RMSE: 0.3262 avg (vs GRU 0.2928 — 11.4% worse)
- P&L: +$1.56 avg (vs GRU +$212.50)
- Num trades: 120 (vs GRU 1,517 — very few signals triggered)
- Win rate: 37.5% (vs GRU 55.8%)
- Attention: entropy=2.656, not degenerate

## Task Commits

1. **Task 1: Extract VSN heatmap and document Finding 24** - `5610ca4`
   - experiments/extract_tft_heatmap.py (213 lines)
   - experiments/figures/tft_variable_importance.png (162KB, 300 DPI)
   - FINDINGS.md (Finding 24, ~45 lines)
2. **Task 2: Update PAPER_DRAFT.md — Branch B** - `c7dae0a`
   - PAPER_DRAFT.md (abstract + §4.1 + Table 2 + §6.2.3 + §7, 9 net lines)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] KeyError: 'target' in extract_tft_heatmap.py first run**
- **Found during:** Task 1 (first heatmap extraction run)
- **Issue:** Script used `train["target"]` but the actual column is `spread_change_target` (as named in run_baselines.py `TARGET_COLUMN`). Caused fallback to placeholder figure on first run.
- **Fix:** Changed `target_col = "target"` to `target_col = "spread_change_target"` in `_train_tft()`
- **Files modified:** experiments/extract_tft_heatmap.py
- **Verification:** Second run produced actual VSN weights; 51 features extracted; heatmap saved at 162KB
- **Committed in:** 5610ca4 (Task 1 commit, same file)

**2. [Deviation] Finding numbered 24, not 26**
- **Reason:** Plan said "Finding 26 or actual next number." FINDINGS.md contained only Findings 1–23; the actual next available number is 24. The plan assumed Findings 24 and 25 might exist; they do not.
- **Action taken:** Used Finding 24 (correct numbering) for accuracy. The plan's success criteria says "Finding 26" but the plan body says "use actual next number if 24/25 don't exist."
- **Impact:** Cosmetic — the finding content is complete and accurate. Phase 13 reads the finding by section name, not number.

---
*Phase: 11-tft-training*
*Completed: 2026-04-22*
