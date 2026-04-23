---
phase: 14-paper-finalization-presentation
plan: 02
subsystem: paper integrity
tags: [paper, pol-05, pol-06, pol-08, pol-09, pol-10, references, tables, figures, limitations]
dependency_graph:
  requires:
    - Plan 14-01 outputs (abstract at 244 words, §4.6 dead ref already removed, Figure 2 cap annotation on-figure)
    - PAPER_DRAFT.md in post-14-01 state
  provides:
    - PAPER_DRAFT.md with unique Table 1-10 numbering, 11-entry Appendix B figure list in in-text order, expanded §6.4 (8 items), alphabetized 14-entry reference list
    - scripts/check_paper.sh — 18-check grep/awk/sed validator for POL-04/05/06/07/08/09/10
  affects:
    - Plan 14-03 (README + slides — can now reference canonical Tables 9/10 and Fig. 11; paper structure is stable)
tech_stack:
  added:
    - scripts/check_paper.sh (new bash validator)
  patterns:
    - POL-* requirements enforced via lightweight grep/awk/sed checks rather than code-level tests
    - Explicit `sed -n '/^## X$/,/^## Y$/p'` range pattern (not `awk '/^## X/,/^## /'`) to avoid matching the start marker as the end marker
key_files:
  created:
    - scripts/check_paper.sh
  modified:
    - PAPER_DRAFT.md (Tables 6/7 renumbered to 9/10 in §5.10/§5.11; §1.4 item 6 Table 6→7 bug fix; "Figure 2b" → "Fig. 10"; [Insert Figure] placeholder → standard Fig. 11 prose; Appendix B rewritten in in-text order with 11 figures; §6.4 appended items 6-8; References alphabetized with Cont et al. 2014 added)
decisions:
  - Added `[Anonymous]` arXiv entry as reference #1 (sorts before alphabetical authors); Cont, Kukanov & Stoikov (2014) inserted as #4 alphabetically, resolving stale in-text citation at §6.2.1 line 536
  - Rule 1 bug fix applied during Task 1: §1.4 item 6 said "(Table 6)" for the transaction-cost analysis, but transaction costs are Table 7 — fixed in same commit as the renumber work
  - Validator uses `sed -n '/^## X$/,/^## Y$/p'` for section ranges instead of `awk '/^## X/,/^## /'` — the awk pattern fails when both patterns match the same line (caught during initial validator run; POL-09 reported 0 until fixed)
  - Kept original reference wording (author order, journal titles, volume/page numbers) untouched; only re-ordered and added the missing Cont entry
metrics:
  duration: "~5 min"
  completed: 2026-04-23
  tasks: 3/3
  commits: 3
  files_created: 2 (scripts/check_paper.sh + this summary)
  files_modified: 1 (PAPER_DRAFT.md)
---

# Phase 14 Plan 02: Paper Integrity Pass Summary

**One-liner:** Renumbered double-assigned Tables 6/7 to 9/10, rebuilt Appendix B in in-text appearance order with 11 figures, replaced the last `[Insert Figure]` placeholder with standard `Fig. 11` prose, appended three live-reconciliation limitations to §6.4, alphabetized the reference list (adding the missing Cont et al. 2014 entry), and committed `scripts/check_paper.sh` as an 18-check validator that enforces the state on future edits.

## What We Built

### Task 1 — Table/Figure renumber + dead-reference fixes

**Commit:** `a19b3bb` — `docs(14-02): renumber Tables 6/7 collisions to 9/10, reorder Appendix B figures, fix dead cross-refs (POL-05, POL-06, POL-10)`

**Final Table numbering scheme (for Plan 14-03 README):**

| Number    | Section | Content                          | Status    |
| --------- | ------- | -------------------------------- | --------- |
| Table 1   | §3.3    | Feature taxonomy                 | unchanged |
| Table 2   | §5.1    | Headline backtest                | unchanged |
| Table 3   | §5.3    | Per-category                     | unchanged |
| Table 3b  | §5.2    | Walk-forward Sharpe by window    | unchanged |
| Table 4   | §5.2    | Walk-forward per-window P&L      | unchanged |
| Table 5   | §5.4    | Data scaling                     | unchanged |
| Table 6   | §5.5    | XGBoost hyperparam sweep         | unchanged |
| Table 7   | §5.6    | Transaction cost sensitivity     | unchanged |
| Table 8   | §5.8    | Sharpe accounting                | unchanged |
| **Table 9**  | §5.10   | **Feature ablation (was Table 6)**  | **renumbered** |
| **Table 10** | §5.11   | **Ensemble variants (was Table 7)** | **renumbered** |

**Final Figure numbering scheme (for Plan 14-03 slides):**

| Fig. | File                                                                 | Section     |
| ---- | -------------------------------------------------------------------- | ----------- |
| 1    | `experiments/figures/walk_forward_pnl.png`                           | §5.2        |
| 2    | `experiments/results/data_scaling/pnl_at_2pp_vs_data.png`            | §5.4 (with SCAL-03 cap annotation on-figure from Plan 14-01) |
| 3    | `experiments/figures/walk_forward_sharpe.png`                        | §5.2 supplemental |
| 4    | `experiments/figures/transaction_cost_sensitivity.png`               | §5.6        |
| 5    | `experiments/figures/shap_bar_plot.png`                              | §5.7        |
| 6    | `experiments/figures/backtest_equity_curves.png`                     | §5.x        |
| 7    | `experiments/figures/bootstrap_ci_rmse.png`                          | §5.x        |
| 8    | `experiments/figures/experiment2_lookback_pnl.png`                   | §5.x (Experiment 2) |
| 9    | `experiments/figures/experiment3_threshold_heatmap.png`              | §5.x (Experiment 3) |
| 10   | `experiments/figures/tft_variable_importance.png`                    | §6.2.3 (was "Figure 2b" ghost) |
| 11   | `experiments/figures/ensemble_weight_sweep.png`                      | §5.11 (was `[Insert Figure]` placeholder) |

**Edits in this commit:**

- §5.10 Table 6 header → **Table 9** (feature ablation) + one in-text body ref
- §5.11 Table 7 header → **Table 10** (ensemble variants)
- §1.4 item 6: "(Table 6)" stale ref → "(Table 7)" — pre-existing Rule 1 bug (transaction-cost analysis was always Table 7 in §5.6)
- Line 229 TFT footnote: "Figure 2b" → "Fig. 10" (ghost reference resolved)
- Line 492: `[Insert Figure: experiments/figures/ensemble_weight_sweep.png — Caption: "Figure 11: ..."]` → standard `**Fig. 11** (\`experiments/figures/ensemble_weight_sweep.png\`) plots...` prose
- Appendix B: rewritten in in-text appearance order; bullets re-formatted to `- **Fig. N** — path — caption — section`; list now contains 11 entries (was 9)

### Task 2 — §6.4 Limitations expansion + References alphabetization

**Commit:** `4506c2d` — `docs(14-02): expand §6.4 limitations with live-cohort items, alphabetize references (POL-05, POL-08)`

**§6.4 now has 8 items:**

| # | Item                                    | Source                                 |
| - | --------------------------------------- | -------------------------------------- |
| 1 | Short test window                       | original (pre-14-02)                   |
| 2 | Paper trading only                      | original                               |
| 3 | Survivorship bias (training-data level) | original (preserved — POL-08 regression check) |
| 4 | Settlement divergence risk              | original                               |
| 5 | Regime-specific edge                    | original                               |
| 6 | **Live-cohort truncation (April 11+)**  | **NEW — pair_id schema fix force-closed pre-April-11 positions** |
| 7 | **Category-tagging gaps in live data**  | **NEW — 59% of live trades bucketed as `other` via `derive_category_from_ticker` gaps** |
| 8 | **Crypto regime flip within reconciliation window** | **NEW — 5-day sign flip; Finding 23** |

**References alphabetization (POL-05):**

14 entries in final alphabetical order:

1. `[Anonymous]. (2026). Matched filter feature engineering for investor flow prediction. arXiv:2601.07131.`
2. Amihud 2002
3. Burgi, Tuccella & Zitzewitz 2026
4. **Cont, Kukanov & Stoikov 2014 (NEW — was missing despite inline citation at §6.2.1 line 536)**
5. Corwin & Schultz 2012
6. Grinsztajn 2022
7. Kyle 1985
8. Lundberg & Lee 2017
9. Manski 2006
10. Parkinson 1980
11. Reimers & Gurevych 2019
12. Roll 1984
13. Schulman 2017
14. Wolfers & Zitzewitz 2004

### Task 3 — scripts/check_paper.sh + final POL-10 sweep

**Commit:** `f94a4e7` — `docs(14-02): add paper-integrity validator and clear residual TODOs (POL-10)`

**`scripts/check_paper.sh`** is a ~80-line bash validator with 18 checks, runs in ~1 second. Path: `scripts/check_paper.sh` (Plan 14-03 should run this as part of its pre-submission sweep).

**Checks:**

| Category | Check                                 | Enforces         |
| -------- | ------------------------------------- | ---------------- |
| POL-04   | Abstract word count ≤ 250              | 244/250          |
| POL-05   | References count ≥ 14                  | 14 entries       |
| POL-05   | Cont entry present                     | 2 hits (in-text + ref) |
| POL-05   | Alphabetical order (skipping `[Anonymous]` prefix) | `sort -c` silent |
| POL-06   | `^\*\*Table 6` count == 1              | XGBoost sweep only |
| POL-06   | `^\*\*Table 7` count == 1              | Transaction costs only |
| POL-06   | `^\*\*Table 9` count == 1              | Feature ablation only |
| POL-06   | `^\*\*Table 10` count == 1             | Ensemble variants only |
| POL-06   | Appendix B figure bullets ≥ 11         | All 11 figs      |
| POL-07   | "per-pair" mentions ≥ 3                | 7 hits           |
| POL-07   | Stale `0.59 annualize` / `4.3` claims  | 0 hits           |
| POL-08   | "survivorship" in §6.4                 | 2 hits (items 3 + preserved wording elsewhere) |
| POL-08   | "live-cohort" / "pair_id schema" in §6.4 | 1 hit (item 6) |
| POL-09   | "claude" / "anthropic" in Acknowledgments | 1 hit         |
| POL-10   | `TODO\|FIXME\|XXX\|\[Insert\|TBD`      | 0 hits           |
| POL-10   | `§4\.6\|Figure 2b\|Fig\. 2b`           | 0 hits           |

## Verification

| Check                                               | Target          | Actual          |
| --------------------------------------------------- | --------------- | --------------- |
| `bash scripts/check_paper.sh` exit code             | 0               | **0**           |
| Number of `[OK]` / `[FAIL]` lines                    | 16 OK / 0 FAIL  | **16 OK / 0 FAIL** |
| `grep -c '^\*\*Table 6' PAPER_DRAFT.md`             | 1               | **1**           |
| `grep -c '^\*\*Table 10' PAPER_DRAFT.md`            | 1               | **1**           |
| `grep -cE 'TODO\|FIXME\|XXX\|\[Insert\|TBD' PAPER_DRAFT.md` | 0        | **0**           |
| Appendix B figure bullets                           | 11              | **11**          |
| §6.4 numbered items                                  | ≥ 8             | **8**           |
| References entries                                   | 14              | **14**          |
| Cont et al. reference entry                          | 1               | **1**           |

## Deviations from Plan

**1. [Rule 1 - Bug] Pre-existing stale cross-reference at §1.4 item 6: "Table 6" referenced transaction-cost analysis but that's always been Table 7 (§5.6)**

- **Found during:** Task 1 (during full grep sweep for "Table 6" refs).
- **Issue:** §1.4 item 6 on line 65 said "**A transaction-cost sensitivity analysis** (Table 6)" but the transaction-cost table is Table 7 (§5.6). This was not part of the renumber work — it was a separate, older bug unnoticed until this pass.
- **Fix:** Edited "(Table 6)" → "(Table 7)" in same commit as the renumber work.
- **Files modified:** PAPER_DRAFT.md line 65.
- **Commit:** a19b3bb.

**2. [Rule 3 - Blocking issue] Initial `check_paper.sh` used fragile `awk '/^## X/,/^## /'` range pattern for POL-09 — pattern matched start-marker as end-marker**

- **Found during:** Task 3 first validator run (POL-09 reported `ai_disclosure = 0` despite Acknowledgments containing the Claude/Anthropic disclosure).
- **Issue:** `awk '/^## Acknowledgments/,/^## /'` range matches the start line as the end line because the start line itself matches `^## `, so the range collapses to a single line that contains just the heading.
- **Fix:** Replaced with `sed -n '/^## Acknowledgments$/,/^## References$/p'` — explicit start anchor (trailing `$`) and explicit end marker.
- **Files modified:** scripts/check_paper.sh (POL-09 block only).
- **Commit:** f94a4e7 (bundled with initial script creation).

## Downstream Integration Points

For **Plan 14-03** (README + slides):

- Reference Tables 9 (feature ablation) and 10 (ensemble variants) by their new numbers in the README's reproduction table. The canonical mapping is in the Task 1 table above.
- For slides: Fig. 11 (ensemble weight sweep) is the newly-resolved placeholder; if slides need a "weight sensitivity" visual they should point at `experiments/figures/ensemble_weight_sweep.png` and cite "Fig. 11".
- Plan 14-03's pre-submission sweep should run `bash scripts/check_paper.sh` (returns 0) as the final integrity gate before calling the paper "done".

### Residual issues for 14-03's cover-to-cover review

- Appendix A (line ~672) has Table-reference comments like `# Table 6` and `# Table 7` next to bash commands — these are currently pointing at the reproduction scripts (XGBoost sweep, transaction costs), which are still Tables 6 & 7, so no renumber needed. But 14-03 should verify the script names themselves (e.g., `run_xgb_hyperparam_sweep` vs. actual filename on disk) when writing the new README.
- `§5.x` appearing in Appendix B bullets for Figs. 6/7/8/9 is intentional — the body text uses those figures but without a consistent specific §N.M anchor; if 14-03 wants to tighten, it can trace each figure to its precise in-text citation.
- The `scripts/check_paper.sh` `[Anonymous]` alphabetical-order handling is a deliberate design choice: skipping the `[`-prefixed entry in the sort check lets authors stay in A-Z order while `[Anonymous]` sits at position 1. If the reviewer prefers `[Anonymous]` at the end, re-order and re-run the validator — the `references_alphabetical` check will still pass.

## Commits

| Hash    | Message                                                                                   |
| ------- | ----------------------------------------------------------------------------------------- |
| a19b3bb | docs(14-02): renumber Tables 6/7 collisions to 9/10, reorder Appendix B figures, fix dead cross-refs (POL-05, POL-06, POL-10) |
| 4506c2d | docs(14-02): expand §6.4 limitations with live-cohort items, alphabetize references (POL-05, POL-08) |
| f94a4e7 | docs(14-02): add paper-integrity validator and clear residual TODOs (POL-10)              |

## Requirements Satisfied

- **POL-05** — Reference list alphabetized (14 entries) and Cont, Kukanov & Stoikov (2014) added to resolve the stale in-text citation at §6.2.1.
- **POL-06** — All Table 1–10 numbers unique; 11-entry Appendix B figure list in in-text appearance order; "Figure 2b" ghost reference replaced with "Fig. 10".
- **POL-08** — §6.4 Limitations has 8 items including survivorship (preserved), live-cohort truncation (item 6), category-tagging gaps (item 7), crypto regime flip (item 8).
- **POL-09** — Acknowledgments section contains the Anthropic Claude disclosure (unchanged from Plan 14-01 state; regression-check verified green in `check_paper.sh`).
- **POL-10** — Zero TODO/FIXME/XXX/[Insert/TBD markers and zero dead cross-refs (§4.6, Figure 2b) in PAPER_DRAFT.md. `scripts/check_paper.sh` enforces this on future edits.

## Self-Check: PASSED

- File `scripts/check_paper.sh` exists and is executable (`test -x` passes).
- All 3 commits (a19b3bb, 4506c2d, f94a4e7) present in `git log --oneline -5`.
- `bash scripts/check_paper.sh` exits 0, prints "ALL CHECKS PASSED", and all 16 named checks return `[OK]`.
- PAPER_DRAFT.md modifications verified: 4 unique Table headers (6/7/9/10), 11-bullet Appendix B, 8-item §6.4, 14-entry References with Cont.
