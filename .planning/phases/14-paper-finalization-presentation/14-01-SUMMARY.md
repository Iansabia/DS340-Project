---
phase: 14-paper-finalization-presentation
plan: 01
subsystem: plotting + paper
tags: [ieee, figures, abstract, sharpe, pol-01, pol-02, pol-03, pol-04, pol-06, pol-07, pol-08]
dependency_graph:
  requires:
    - scienceplots==2.2.1 (already in requirements.txt)
    - matplotlib==3.10.8
    - experiments/results/{walk_forward,data_scaling,transaction_costs,shap,backtest,bootstrap_ci,ablation_lookback,ablation_threshold,ensemble}/
  provides:
    - src/plotting/ieee_style.py (canonical import: `from src.plotting.ieee_style import apply_ieee_style, save_ieee_fig, OKABE_ITO`)
    - scripts/regenerate_figures.py (reproduction entry point)
    - experiments/results/tft/vsn_importance.json (persisted VSN weights)
    - 11 IEEE-styled 300 DPI figures (+ sibling PDFs)
    - Abstract ≤ 250 words with per-pair Sharpe ≈ 3.2 headline
  affects:
    - Plan 14-02 (figure-reference renumbering — unblocked; can read regenerated PNGs + updated abstract)
    - Plan 14-03 (README — will reference `python scripts/regenerate_figures.py` entry point)
tech_stack:
  added:
    - src/plotting helper module (new package)
  patterns:
    - Consolidated figure regeneration via a single entry point (scripts/regenerate_figures.py) rather than editing each experiment script
    - Persist VSN weights as JSON so downstream re-renders don't retrain TFT
    - Sibling PDF export alongside PNG for LaTeX compatibility
key_files:
  created:
    - src/plotting/__init__.py
    - src/plotting/ieee_style.py
    - tests/plotting/__init__.py
    - tests/plotting/test_ieee_style.py
    - scripts/regenerate_figures.py
    - experiments/results/tft/vsn_importance.json
    - 11 PNG figures (regenerated) + 11 sibling PDFs
  modified:
    - PAPER_DRAFT.md (abstract, §1.4 item 7, §8 item 5)
    - experiments/extract_tft_heatmap.py (persist VSN weights to JSON)
    - .gitignore (lightning_logs/, checkpoints/, graphify-out/, .claude/)
decisions:
  - Wrote a consolidated scripts/regenerate_figures.py that reads result JSONs on disk and re-renders all 11 paper figures, rather than surgically editing 11 separate experiment scripts. Keeps IEEE styling in one place and makes regeneration idempotent.
  - Persisted TFT VSN encoder weights to experiments/results/tft/vsn_importance.json after the one-off retrain, so future figure regenerations do not re-train TFT.
  - Kept the abstract at 244 words (not 248) to leave room for any downstream Plan 14-02 tweaks that might re-expand it.
  - Sibling PDF export is best-effort (wrapped in try/except) so the PDF backend is never a hard dependency.
metrics:
  duration: "~20 min (includes 8 min TFT retrain)"
  completed: 2026-04-23
  tasks: 3/3
  commits: 3
  files_created: 17
  files_modified: 3
---

# Phase 14 Plan 01: Figure Restyle + Abstract/Sharpe Scrub Summary

**One-liner:** Built the canonical `src.plotting.ieee_style` helper, regenerated all 11 paper-referenced figures under IEEE SciencePlots style at 300 DPI with colorblind-safe Okabe-Ito palette, and replaced the stale per-trade Sharpe headline in the abstract and §1.4 with the per-pair ≈ 3.2 framing from Table 8.

## What We Built

### Task 1 — IEEE Style Helper (Wave 0)

`src/plotting/ieee_style.py` exposes:

- `apply_ieee_style()` — applies SciencePlots `['science', 'ieee', 'no-latex']` with the 8-color Okabe-Ito colorblind-safe palette, variable linestyle + marker cycle (for B&W readability), `savefig.dpi=300`, and `image.cmap='cividis'` for heatmaps. Falls back gracefully to plain-matplotlib rcParams if `scienceplots` fails to import.
- `save_ieee_fig(fig, path, dpi=300)` — saves PNG at 300 DPI with `bbox_inches='tight'` and attempts a sibling PDF export (swallowed on backend failure).
- `OKABE_ITO` — 8-hex-string palette constant.

Test coverage: 6 pytest tests (imports, palette structure, rcParams mutations, PNG persistence, tolerant PDF sibling) — **6/6 green** on first run.

**Commit:** 6c9bca4 — `feat(14-01): add IEEE style helper (POL-01/02/03 Wave 0)`

### Task 2 — Regenerate 11 Figures

`scripts/regenerate_figures.py` is the canonical reproduction entry point. It reads existing result JSONs on disk and re-renders:

| Fig | Output Path | Source |
|---|---|---|
| 1 | `experiments/figures/walk_forward_pnl.png` | `experiments/results/walk_forward/log.jsonl` |
| 2 | `experiments/results/data_scaling/pnl_at_2pp_vs_data.png` | `experiments/results/data_scaling/log.jsonl` |
| 3 | `experiments/figures/walk_forward_sharpe.png` | `experiments/results/walk_forward/log.jsonl` |
| 4 | `experiments/figures/transaction_cost_sensitivity.png` | `experiments/results/transaction_costs/sensitivity_results.json` |
| 5 | `experiments/figures/shap_bar_plot.png` | `experiments/results/shap/xgboost_feature_importance.csv` |
| 6 | `experiments/figures/backtest_equity_curves.png` | `experiments/results/backtest/{lr,xgboost,gru,lstm}.json` |
| 7 | `experiments/figures/bootstrap_ci_rmse.png` | `experiments/results/bootstrap_ci/bootstrap_results.json` |
| 8 | `experiments/figures/experiment2_lookback_pnl.png` | `experiments/results/ablation_lookback/*.json` |
| 9 | `experiments/figures/experiment3_threshold_heatmap.png` | `experiments/results/ablation_threshold/*.json` |
| 10 | `experiments/figures/tft_variable_importance.png` | `experiments/results/tft/vsn_importance.json` (new) |
| 11 | `experiments/figures/ensemble_weight_sweep.png` | `experiments/results/ensemble/summary.json` |

**Run:** `PYTHONPATH=$(pwd) python scripts/regenerate_figures.py` — all 11 renderers print `OK`.

**Special handling:**

- **Figure 2** carries the SCAL-03/POL-08 cap annotation on-figure (`plateau at N=6,802, fixed pair universe`) via a red dotted vertical line + text — readable without body text.
- **Figure 9** uses `cmap='cividis'` on the threshold heatmap (colorblind-safe, perceptually uniform).
- **Figure 10** (TFT VSN) sources weights from `experiments/results/tft/vsn_importance.json`, which we populated by running a patched `experiments/extract_tft_heatmap.py` once. The patch added JSON persistence so future regenerations do not re-train TFT (~8 min on Apple Silicon CPU).
- Each PNG has a sibling PDF vector export next to it for LaTeX compatibility.

**Commit:** e0d2683 — `feat(14-01): regenerate 11 paper figures under IEEE style at 300 DPI (POL-01/02/03/06/08)`

### Task 3 — Abstract Trim + §1.4 + §8 Sharpe Scrub

- **Abstract:** 315 → **244 words** (under the 250-word cap). Now leads with per-pair Sharpe ≈ 3.2 (Table 8) and reports per-trade 0.44 for transparency. Dropped verbose feature-list parenthetical, long SCC deployment description, and the TFT-did-not-converge note (which lives in Table 2's footnote already).
- **§1.4 item 7:** Replaced the stale `per-trade Sharpe of 0.59 annualizes to 4.3` claim and the dead `§4.6` cross-reference with the correct `§5.8, Table 8` pointer and the per-pair 3.2 / per-trade 0.44 framing.
- **§8 Conclusions item 5:** Strengthened from `honest annualized Sharpe is 2–4` to `per-pair annualized Sharpe is ≈ 3.2 (robust range 2–4 under realistic slippage assumptions)` for cross-consistency with Table 8.

**Commit:** 7f7bdb1 — `docs(14-01): trim abstract to 244 words, fix stale per-trade Sharpe headline (POL-04, POL-07)`

## Verification

| Check | Target | Actual |
|---|---|---|
| `pytest tests/plotting/test_ieee_style.py -q` | 6 passed | **6 passed** |
| `python scripts/regenerate_figures.py` | 11 `OK` lines | **11 OK** |
| Abstract word count | ≤ 250 | **244** |
| `grep -c "per-pair annualized Sharpe" PAPER_DRAFT.md` | ≥ 2 | **3** |
| `grep -cE "≈ 3\.2\|per-pair.*3\.2\|3\.2.*per-pair" PAPER_DRAFT.md` | ≥ 3 | **5** |
| stale `0.59 annualizes` / `4.3` claims | 0 | **0** |
| dead `§4.6` cross-reference | 0 | **0** |
| `grep -c "Table 8" PAPER_DRAFT.md` | ≥ 2 | **4** |
| Abstract contains `per-pair` | ≥ 1 | **1** |
| All 11 output PNGs present, non-zero | OK | **OK** |

## Deviations from Plan

**1. [Rule 3 - Blocking issue] TFT VSN weights were not persisted to any file on disk**

- **Found during:** Task 2 (Figure 10 source-data discovery).
- **Issue:** The plan's Fig-10 spec said "read `experiments/results/tft/*.json` or variable_importance file", but no such file existed — `experiments/extract_tft_heatmap.py` only drew the PNG and discarded the numbers.
- **Fix:** Patched `experiments/extract_tft_heatmap.py` to also dump weights to `experiments/results/tft/vsn_importance.json` after extraction, then ran it once to populate the JSON (TFT retrain ~8 min on Apple Silicon CPU). Downstream regenerations now read the JSON and never retrain.
- **Files modified:** `experiments/extract_tft_heatmap.py`, new `experiments/results/tft/vsn_importance.json`.
- **Commit:** e0d2683.

**2. [Rule 2 - Missing critical functionality] .gitignore did not cover generated training artifacts**

- **Found during:** Task 2 commit staging.
- **Issue:** The TFT retrain created `lightning_logs/` and `checkpoints/` at repo root, plus pre-existing untracked `graphify-out/` and `.claude/` dirs. These would accumulate noise in git status forever.
- **Fix:** Added `lightning_logs/`, `checkpoints/`, `graphify-out/`, `.claude/` to `.gitignore`.
- **Files modified:** `.gitignore`.
- **Commit:** e0d2683 (bundled with Task 2).

**3. [Informational — not a deviation] DPI metadata reads as 299.9994 instead of 300**

- The plan's acceptance criterion expected `PIL.Image.info.get('dpi')` to print `300` (or `300.0`). Actual value reads as `299.9994`. This is a well-known matplotlib `bbox_inches='tight'` + PIL pHYs-chunk rounding artifact — the PNG is saved at exactly 300 DPI; PIL's reconstruction of pixels-per-meter → inches introduces a ~10⁻⁴ floating-point error. Rounding to int yields 300 for every file. No fix needed — the figures are genuinely 300 DPI.

## Downstream Integration Points

For **Plan 14-02** (figure-reference renumbering):

- The 11 regenerated PNGs are on disk at their canonical paths (same file names as before, re-stamped at 300 DPI).
- Abstract word budget: 244 — 14-02 has 6 words of headroom before breaching the 250-word cap.
- `§4.6` cross-reference is already removed; 14-02 should not re-introduce it.
- Table 8 is now referenced 4 times — 14-02's cross-reference scrub can treat Table 8 as canonical-correctly-named.

For **Plan 14-03** (README + slides):

- The canonical regeneration command is `PYTHONPATH=$(pwd) python scripts/regenerate_figures.py` — reference this in the README's "Reproducing every paper figure" section.
- Figure 10 requires `experiments/results/tft/vsn_importance.json` on disk; Plan 14-03's reproduction docs should call out that users who want a genuine re-train must run `python experiments/extract_tft_heatmap.py` first, else the regeneration uses the checked-in weights.

## Data Reconstruction Required

Only one figure needed manual data reconstruction:

- **Fig 10 (TFT VSN):** Weights were not persisted by the original `extract_tft_heatmap.py`. We patched the script to persist and ran it once. Future runs read the JSON.

All other 10 figures consumed pre-existing JSON/CSV artifacts on disk with no reconstruction needed.

## Commits

| Hash | Message |
|---|---|
| 6c9bca4 | feat(14-01): add IEEE style helper (POL-01/02/03 Wave 0) |
| e0d2683 | feat(14-01): regenerate 11 paper figures under IEEE style at 300 DPI (POL-01/02/03/06/08) |
| 7f7bdb1 | docs(14-01): trim abstract to 244 words, fix stale per-trade Sharpe headline (POL-04, POL-07) |

## Requirements Satisfied

- **POL-01** — SciencePlots IEEE style applied: `src/plotting/ieee_style.py` + 11 figures use it.
- **POL-02** — Colorblind-safe palette + B&W readability: Okabe-Ito 8-color palette + linestyle/marker cycle.
- **POL-03** — 300 DPI on all saved figures: `savefig.dpi=300` in helper + explicit `dpi=300` in `save_ieee_fig`.
- **POL-04** — Abstract ≤ 250 words: 244.
- **POL-07** — Headline Sharpe = per-pair ≈ 3.2: abstract + §1.4 item 7 + §8 item 5 all consistent.

## Self-Check: PASSED

All 17 created/modified files exist on disk. All 3 commits (6c9bca4, e0d2683, 7f7bdb1) are present in git history.
