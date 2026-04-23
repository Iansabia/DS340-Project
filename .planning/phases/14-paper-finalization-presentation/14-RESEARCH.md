# Phase 14: Paper Finalization + Presentation - Research

**Researched:** 2026-04-23
**Domain:** Academic paper polish, IEEE-style figure regeneration, abstract/number editing, reproduction-command documentation, 4-minute lightning-talk slide authoring
**Confidence:** HIGH

## Phase Overview

Phase 14 is a pure *polish-and-package* phase. All experimental evidence exists; the remaining work is editorial (`PAPER_DRAFT.md` scrub), visual (`SciencePlots` IEEE restyle of 18 figures), documentation (`README.md` creation with per-table reproduction commands), and presentation (4-minute lightning-talk slides per DS340 rubric). Zero new experiments, zero model training, zero data collection. The phase spans **4 calendar days** (2026-04-23 → 2026-04-27) against a hard submission deadline.

**Downstream consumer:** The gsd-planner will break this research into ~3 plans mapped to requirement clusters: (1) figure restyle + abstract/number scrub, (2) paper integrity pass (TODOs, section-number drift, table numbering, Sharpe audit, limitations, acknowledgments), (3) README + slides.

**Primary recommendation:** Lock figures first (POL-01 through POL-03), because every other polish task references figures. Then scrub numbers/text (POL-04 through POL-10). Finally ship README and slides (POL-11, POL-12). Do not chase alternative presentation tooling — use **Marp (Markdown → PDF slides)** for the lightning talk because it matches the existing workflow and produces classroom-computer-compatible PDF.

---

## Current State Audit

### Paper draft scan (`PAPER_DRAFT.md`, 700 lines)

**Explicit placeholders:**
- **Line 492:** `[Insert Figure: experiments/figures/ensemble_weight_sweep.png — Caption: "Figure 11: ..."]` — literal `[Insert Figure]` placeholder; must be replaced with standard `Fig. 11` reference format consistent with Figs. 1–9 elsewhere.
- **Line 67 (Contributions §1.4 item 7):** stale numbers `"per-trade Sharpe of 0.59 annualizes to roughly 4.3"` — contradicts Table 8 which shows per-trade Sharpe **0.44** and per-pair Sharpe **≈ 3.2**. This is a POL-07 violation (the contribution statement puts a non-per-pair number first) AND a stale-number correctness bug.
- **Line 67 (Contributions §1.4 item 7):** references `§4.6` for Sharpe accounting; the actual section is `§5.8` (line 368). Dead cross-reference.
- **Line 229 (Table 2 footnote):** references `"Figure 2b"` for the TFT VSN heatmap. There is no Figure 2b — the file `tft_variable_importance.png` exists but has no figure number assigned in Appendix B. Dead cross-reference.

**Table-numbering collisions (double-assigned):**
| Number | Section | Page location | Content |
|---|---|---|---|
| **Table 6** | §5.5 (line 333) | XGBoost hyperparameter sweep | Top-10 of 48 configs |
| **Table 6** | §5.10 (line 452) | Feature ablation | 12 LOGO configurations |
| **Table 7** | §5.6 (line 350) | Transaction cost sensitivity | 0–7 pp fee levels |
| **Table 7** | §5.11 (line 479) | Ensemble variant comparison | 4 ensemble variants |

These collisions entered when Phases 12/13 added §5.10 and §5.11 without re-numbering later tables. **Table 8 (Sharpe accounting, §5.8)** is referenced correctly but comes *before* the duplicated Tables 6/7 from §5.10/§5.11 — so the naive "add 2 to later collisions" fix would push ablation-Table-6 → Table-9 and ensemble-Table-7 → Table-10, but that creates narrative ordering confusion. Proposed remediation: renumber strictly by order-of-appearance:
- Table 1 (feature taxonomy) → **Table 1** (unchanged)
- Tables 2, 3, 3b, 4, 5 (§5.1–§5.4) → **unchanged**
- Table 6 (§5.5 XGBoost sweep) → **unchanged**
- Table 7 (§5.6 transaction costs) → **unchanged**
- Table 8 (§5.8 Sharpe accounting) → **unchanged**
- Duplicate "Table 6" (§5.10 ablation) → **renumber to Table 9**
- Duplicate "Table 7" (§5.11 ensembles) → **renumber to Table 10**

All in-text references to these tables must update in lockstep (§5.10 line 450 "Table 6 presents..." → "Table 9 presents..."; §5.11 line 479 and §4.4 line 193 implicit reference).

**Figure-numbering collisions:**
| Number | Section | Context |
|---|---|---|
| **Figure 2** | §5.4 line 312 | Data-scaling curve (`pnl_at_2pp_vs_data.png`) |
| **Figure 2** | Appendix B line 693 | Walk-forward Sharpe (`walk_forward_sharpe.png`) |

Appendix B lists Fig 2 as "Walk-forward Sharpe curves" but §5.4 refers to Fig 2 as the data-scaling curve. The §5.4 in-text reference matches **Figure 3** in Appendix B's list (`pnl_at_2pp_vs_data.png`). Proposed remediation: either renumber Appendix B to match in-text order (easier) or renumber in-text citations. Recommend Appendix B alignment:
- Fig 1 = walk_forward_pnl.png (§5.2)
- Fig 2 = pnl_at_2pp_vs_data.png (§5.4 — SCAL-03 cap annotation target)
- Fig 3 = walk_forward_sharpe.png (§5.2 supplemental)
- Fig 4 = transaction_cost_sensitivity.png (§5.6)
- Fig 5 = shap_bar_plot.png (§5.7)
- Fig 6 = backtest_equity_curves.png
- Fig 7 = bootstrap_ci_rmse.png
- Fig 8 = experiment2_lookback_pnl.png
- Fig 9 = experiment3_threshold_heatmap.png
- Fig 10 = tft_variable_importance.png (§6.2.3, currently "Fig 2b")
- Fig 11 = ensemble_weight_sweep.png (§5.11, currently `[Insert Figure]`)

**No other literal TODO/FIXME/XXX/TBD tokens in the body.** The explicit placeholder grep returned only line 492. Phase 14 paper integrity is therefore tractable by the existing evidence — no missing experiments.

### Abstract word count audit (POL-04)

**Current: 315 words** (lines 10–12, single paragraph between `## Abstract` and the Keywords line).
**Target: ≤ 250 words.**
**Cut required: 65 words (~20%).**

Trim targets (highest redundancy → easy wins):
1. **"permissionless on-chain prediction market"** → "on-chain prediction market" (−2 words)
2. "(i) matches semantically-equivalent contracts across platforms using sentence embeddings with a 10-rule structural quality filter" → "matches contracts via sentence embeddings + a 10-rule quality filter" (−8 words)
3. **Feature-list parenthetical** "including 13 academic market-microstructure features (Amihud illiquidity, Corwin–Schultz implied spread, Kyle's lambda, Roll's implied spread, favorite–longshot bias)" → "including 13 market-microstructure features" (−14 words)
4. "(TFT was attempted but did not converge at N=6,802)" → drop from abstract, keep in Table 2 footnote (−9 words)
5. "The system was also deployed as an autonomous paper-trading agent on the BU Shared Computing Cluster (SCC), executing a 15-minute trade cycle and retraining models every six hours." → "The system is also deployed as an autonomous paper-trader on SCC." (−20 words)
6. "In our primary fresh verification run (April 17, 2026) on 6,802 training rows and 1,673 held-out test rows across 144 matched pairs" → "On 6,802 train / 1,673 test rows across 144 pairs" (−14 words)

Total: ~67 words cut → lands at **~248 words**, meeting the ≤250 target with 2 words of headroom.

**POL-07 interacts here**: the current abstract contains only per-trade Sharpe (`0.31 → 0.53`) and no per-pair headline. POL-07 demands per-pair ≈ 3.2 as the *headline*. Resolution: insert one sentence after the P&L enumeration, e.g., "The per-pair annualized Sharpe, treating each of 144 matched pairs as one independent bet, is ≈ 3.2 (footnote: per-trade Sharpe 0.44 in Table 8)." This adds ~25 words, so cuts 1–6 above must be executed more aggressively. Revised target: cut ~90 words from existing prose, then insert the ~25-word Sharpe sentence, net ~250 words.

### Sharpe number audit (POL-07)

Every "Sharpe" mention in the paper:

| Line | Section | Claim | Status |
|---|---|---|---|
| 12 | Abstract | `per-trade Sharpe 0.31 → 0.53` (walk-forward range) | OK; but headline per-pair (3.2) **missing** — POL-07 violation |
| 33 | §1.1 | "honest accounting of... Sharpe inflation" | OK (meta) |
| 59 | §1.4 item 3 | `per-trade Sharpe 0.31 → 0.53` | OK |
| **67** | **§1.4 item 7** | `per-trade Sharpe of 0.59 annualizes to roughly 4.3` | **STALE + POL-07 violation.** Should be `per-trade Sharpe 0.44 annualizes to ≈ 3.2 on per-pair basis`. Fix to: "The per-pair annualized Sharpe, under the correct assumption of 144 independent pair-level bets, is ≈ 3.2 (§5.8, Table 8). The per-trade number of 0.44 is reported in Table 2 for transparency." |
| 161 | §4.2(a) | "per-trade Sharpe" as eval metric | OK |
| 163 | §4.2(b) | "per-trade Sharpe" | OK |
| 193 | §4.4 | `0.437 (unfiltered) to 0.455 (filtered)` | OK (per-trade, in-context) |
| 217 | Table 2 header | `Per-trade Sharpe` column | OK |
| 255 | Table 3b title | `per-trade Sharpe ratio by window` | OK |
| 276 | §5.2 finding 2 | `0.307 → 0.530` | OK (per-trade, walk-forward) |
| 352 | Table 7 header | `Sharpe/trade` | OK |
| 360, 362 | §5.6 | per-trade, in-context | OK |
| 368 | §5.8 heading | "Honest Sharpe-Ratio Accounting" | OK — the canonical section |
| 370 | §5.8 lead | `per-trade Sharpe of 0.436` | OK (matches Table 2) |
| 372, 374, 377, 378, 379, 381 | Table 8 + interpretation | per-pair 3.2 reported | OK (canonical) |
| 496, 498, 500 | §5.11 | filtered/unfiltered Sharpe | OK (in-context) |
| 538, 550 | §6.2 | per-trade Sharpe trajectories | OK |
| 586 | §6.4 | "Sharpe" in limitation | OK |
| 628, 634 | §8 conclusions | `per-trade Sharpe is rising` and `honest annualized Sharpe is 2–4, not 50+` | OK (but should mention per-pair once) |

**Single lethal violation: Line 67.** Once fixed plus the abstract addition, POL-07 is satisfied.

### Figure inventory (POL-01, POL-02, POL-03, POL-06)

**18 figure files present** in `experiments/figures/` (PNG + one TeX):

| # | File | Referenced in paper as | Need IEEE restyle? | Need axis units? | Notes |
|---|---|---|---|---|---|
| 1 | `walk_forward_pnl.png` | Fig 1 (§5.2, Appendix B) | **Yes** | Y: P&L ($); X: window index | Re-render needed |
| 2 | `walk_forward_sharpe.png` | Fig 3 after renumber (currently Fig 2 in Appendix B) | Yes | Y: per-trade Sharpe; X: window index | Re-render needed |
| 3 | `walk_forward_winrate.png` | *Not referenced in-text* | Optional | Y: win rate (%); X: window | Add to Appendix B as Fig 12 or drop |
| 4 | `pnl_at_2pp_vs_data.png` (in `experiments/results/data_scaling/`) | Fig 2 after renumber (§5.4) | **Yes** | Y: P&L ($); X: training rows | **Requires SCAL-03 cap annotation** on figure & caption (POL-08) |
| 5 | `transaction_cost_sensitivity.png` | Fig 4 (§5.6) | Yes | Y: P&L ($); X: fee (pp) | Re-render needed |
| 6 | `shap_bar_plot.png` | Fig 5 (§5.7) | Yes | Y: feature; X: mean \|SHAP\| | Re-render (SHAP uses its own colors — use `SHAP.plots.bar` w/ colorblind arg) |
| 7 | `shap_summary_plot.png` | *Not referenced* | Optional | — | Add to Appendix B or drop |
| 8 | `backtest_equity_curves.png` | Fig 6 | Yes | Y: cumulative P&L ($); X: bar index | Re-render |
| 9 | `bootstrap_ci_rmse.png` | Fig 7 | Yes | Y: RMSE; X: model | Re-render |
| 10 | `experiment1_pnl_curves.png` | *Not referenced* | Drop or add as Fig 13 | — | |
| 11 | `experiment1_rmse_bar.png` | *Not referenced* | Drop | — | |
| 12 | `experiment2_lookback_pnl.png` | Fig 8 | Yes | Y: P&L ($); X: lookback (hours) | Re-render |
| 13 | `experiment2_lookback_rmse.png` | *Not referenced* | Drop | — | |
| 14 | `experiment3_threshold_heatmap.png` | Fig 9 | Yes | Y: model; X: threshold (pp); color: P&L ($) | Re-render, use colorblind-safe colormap (`viridis` or `cividis`) |
| 15 | `experiment3_threshold_pnl.png` | *Not referenced* | Drop | — | |
| 16 | `backtest_drawdown.png` | *Not referenced* | Drop | — | |
| 17 | `tft_variable_importance.png` | Fig 10 after renumber (currently called "Figure 2b" at line 229) | Yes | Y: feature; X: VSN weight | Re-render; fix "Figure 2b" label |
| 18 | `ensemble_weight_sweep.png` | Fig 11 (§5.11) | Yes | Y: P&L ($); X: LR weight ∈ [0,1] | **Replace `[Insert Figure]` placeholder** on line 492 |

**Re-render count:** 11 figures must be regenerated from their source experiment scripts with SciencePlots IEEE styling. The other 7 can be left in place or dropped — not referenced in paper body.

**POL-06 (axis labels carry units):** every Y-axis should be either `P&L ($)`, `RMSE`, `Sharpe`, `Win Rate (%)`, `Fee (pp)`, etc. Current figures — spot-checked `walk_forward_pnl.png` in metadata — are not guaranteed to have units. The re-render step must enforce units via `ax.set_xlabel("Training rows (count)")` / `ax.set_ylabel("P&L ($)")` convention.

### §6.4 Limitations audit (POL-08)

Current §6.4 has 5 items (lines 582–594):
1. Short test window
2. Paper trading only
3. **Survivorship bias** ✅ — already present at item 3 (line 590). POL-08 partially satisfied by text, but the *wording* is understated ("we believe this bias is small because the filter is structural, but we cannot quantify it precisely"). Acceptable per requirement.
4. Settlement divergence risk
5. Regime-specific edge

**Additional items that the April 22 reconciliation findings require:**
- **Live trading window survivorship (April 11+):** Finding 23 shows live data covers only April 14–22 (8 days). Pre–April 11 live data was force-closed due to the `pair_id` schema bug. This is a *different* survivorship bias than item 3 — it's a **live-cohort truncation**. Add as item 6.
- **Category tagging gaps:** Finding 23 exposes that 59% of live trades fall into the "other" bucket due to `derive_category_from_ticker` misclassifying KXPAYROLLS/KXEZCPIYOYF. This affects §5.9 per-category breakdowns. Add as item 7.
- **Crypto regime flip:** Finding 23 shows a category sign-flip within 5 days. Already referenced at line 434 in §5.9 body but not elevated to §6.4. Add as item 8.

### Figure 2 caption cap annotation (POL-08, SCAL-03)

**Current §5.4 text (line 312):** "Fig. 2 (see `experiments/results/data_scaling/pnl_at_2pp_vs_data.png`) shows the 6-point scaling curve. Plateau occurs because train.parquet contains at most 141 bars/pair (N=6,802 rows, 144 pairs); slices at 250+ bars/pair are identical to the 100-bar slice and produce identical metrics."

The cap info is present in the **prose** but NOT in the **figure caption**. SCAL-03 and POL-08 require the annotation to appear *on the figure's caption* (readable independent of body text).

**Proposed caption text (to be embedded in Fig 2's `plt.title()` or as a caption line in §5.4):**

> **Figure 2: P&L at 2 pp fees vs. training-set size.** Each curve plots held-out P&L (\$) on the 1,673-row test set for one model family as training rows are varied from 50 to 2,000 bars/pair. The curve plateaus at 100 bars/pair because `train.parquet` is capped at 6,802 rows across 144 pairs (maximum 141 bars/pair observed); slices ≥250 bars/pair are identical to the 100-bar slice. **The absolute cap is a property of the fixed pair universe, not an architectural limit of the models.** Extrapolating this plot beyond the measured range assumes the pair universe will grow — which it does (§7, item 1), but at rates not captured here.

### Acknowledgments audit (POL-09)

**Current text (line 648):** already includes the required AI-assistant disclosure: "We also acknowledge extensive use of Anthropic's Claude (Sonnet 4.5 and Opus 4.6) as an AI pair-programming assistant throughout the implementation; all design decisions, experimental choices, and interpretations are our own."

**POL-09 required wording:** "We used Anthropic Claude (Sonnet 4.5 and Opus 4.6) as a pair-programming assistant; all design decisions and empirical interpretations are our own."

The current wording is substantively equivalent and arguably more complete ("extensive use", "throughout the implementation"). **POL-09 satisfied as-is**; no edit required unless exact wording is mandated.

### Citation format audit (POL-05)

**Current format (lines 652–666):** numbered list `1.`–`13.` — consistent. References are alphabetical by first author EXCEPT:
- `4. Grinsztajn` comes after `3. Corwin` ✅ alphabetical G>C
- `5. Kyle` comes after `4. Grinsztajn` ✅ K>G
- `6. Lundberg`, `7. Manski`, `8. Parkinson`, `9. Roll`, `10. Schulman`, `11. Wolfers`, `12. Anonymous`, `13. Reimers` — **not alphabetical** (Reimers should be between Parkinson and Roll; Anonymous "[Anonymous]" sort position ambiguous; Schulman before Wolfers OK).

**POL-05 fix:** Reorder references alphabetically by first author (put Reimers in position 9, shift Roll → 10, Schulman → 11, Wolfers → 12, Anonymous → 13 with "arXiv 2026" as sort key). Update any in-text citations that reference by number — but the paper uses author-year inline, not numbered inline citations, so **renumbering breaks nothing** outside the reference list.

**In-text citation style check:** Paper uses "Manski (2006)", "Burgi, Tuccella & Zitzewitz (2026)", "Grinsztajn et al. 2022", "Cont et al. 2014" — consistent author-year format with occasional variation. `Cont et al. 2014` is cited in §6.2.1 but has **no entry in the reference list**. Either add Cont, Kukanov & Stoikov (2014) "The price impact of order book events" (*JFE*) or remove the in-text citation.

### README.md audit (POL-11)

**Status: file does not exist.** No `README.md` anywhere in the project root. Appendix A (line 672–688) of `PAPER_DRAFT.md` contains a shell-command sketch but is **incomplete**:

| Paper Table/Figure | Script in PAPER Appendix A | Exists in `experiments/`? | Notes |
|---|---|---|---|
| Table 2 | `run_baselines` | ✅ `run_baselines.py` | Plus `verify_headline.py` for exact Table 2 reproduction |
| Table 3, Fig 1 | `run_walk_forward --windows 12` | ✅ `run_walk_forward.py` | |
| Table 4 (per-category) | `run_category_breakdown` | ✅ `run_category_breakdown.py` | |
| Table 5, Fig 2 (scaling) | `run_data_scaling` | ⚠️ **Missing from Appendix A**; script name is `experiments/run_data_scaling.py` — verify exists | Investigation needed: `ls experiments/run_data_scaling.py` (not in the top listing — this is a gap) |
| Table 6 (XGB sweep) | `run_xgb_hyperparam_sweep` | ⚠️ **Not listed in experiments/ dir** | Need to investigate — may be in a subfolder or script is named differently |
| Table 7 (transaction cost) | `run_transaction_cost_sensitivity` | ✅ `run_transaction_costs.py` (slightly different name) | Command in Appendix A has **wrong script name** |
| Table 8 (Sharpe accounting) | *No command* | — | Sharpe values are derived from Table 2 data; add a "see `experiments/run_baselines.py` + analysis notebook" pointer |
| Table 9 (ablation, renumbered) | *No command in Appendix A* | ✅ `run_feature_ablation.py` | **Add** |
| Table 10 (ensemble, renumbered) | *No command in Appendix A* | ✅ `run_ensemble_sweep.py` | **Add** |
| §5.9 (reconciliation) | *No command in Appendix A* | ✅ `run_live_reconciliation.py` | **Add** |
| Fig 10 (TFT VSN) | *No command in Appendix A* | ✅ `run_tft.py` + `extract_tft_heatmap.py` | **Add** |
| SHAP | `run_shap_analysis` | ✅ `run_shap_analysis.py` | |

**Action:** Create `README.md` with a **complete** reproduction table, deprecate Appendix A to point at README.md, and verify every script name by actually running the commands end-to-end before submission.

### Slides audit (POL-12)

**Status: no slide files exist** anywhere in the repository. Search confirmed: no `.pptx`, no `.key`, no `slides/` directory, no `presentation/` directory, no `*.slides.md`.

Per `DS340FinalPresentationAndProjectInstrs_S26.pdf` (read earlier):
- **Duration:** 4 minutes, hard cap ("You must present a section within the time limit to get credit for it")
- **Presented:** April 28 or April 30 (the last two days of class)
- **Slides due:** night before presentation (i.e., 2026-04-27 if presenting 4/28, or 2026-04-29 if 4/30)
- **Format:** whatever renders from classroom computer (instructor will present from classroom computer if needed — implying PDF export is safest)
- **Required sections:**
  1. **Team** — names only
  2. **Problem statement** — what, why
  3. **Methods** — models/algorithms, data source
  4. **Challenge** — one challenge overcome + how solved
  5. **Results** — most interesting result
  6. **Conclusions** — what we learned

Four minutes ÷ 6 sections = ~40 seconds per section. That is "1 slide per section + 1 title slide" territory — 7 slides total.

---

## POL-01 through POL-12 Implementation Approach

### POL-01 — SciencePlots IEEE styling on every figure

**Library already installed** (confirmed: `requirements.txt` lines include `SciencePlots==2.2.1`, `matplotlib==3.10.8`). Python 3.14 venv already imports it.

**Style recipe (verified locally that the following styles exist in installed SciencePlots 2.2.1):**

```python
import matplotlib.pyplot as plt
import scienceplots  # registers styles; explicit import required for Py3.12+

# Apply IEEE + no-latex (no-latex avoids MacTeX dependency per v1.1 non-goal)
plt.style.use(['science', 'ieee', 'no-latex'])

# Colorblind-safe: layer on 'bright' or 'high-contrast'. IEEE style alone defaults
# to grayscale + color variation. For BW readability add per-line markers/dashes.
# DO NOT layer 'high-contrast' with 'ieee' — conflicts observed in SciencePlots examples.
# Instead use explicit marker/linestyle cycle.
```

**Verified SciencePlots styles available in local install** (via `plt.style.available`):
`bright`, `high-contrast`, `ieee`, `no-latex`, `science`, `std-colors`, `seaborn-v0_8-colorblind`, `tableau-colorblind10`.

**Source:** SciencePlots GitHub (`github.com/garrettj403/SciencePlots`), *which ships the `ieee` and `no-latex` styles as of version 2.2.1*. Verified via local `plt.style.available` query in `.venv`.

### POL-02 — Colorblind-safe palette + B&W readability

**Recommended explicit palette** (Okabe-Ito 2008, widely used colorblind-safe 8-color palette — distinguishable under protanopia, deuteranopia, tritanopia):

```python
OKABE_ITO = [
    '#000000',  # black
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
    '#009E73',  # bluish green
    '#F0E442',  # yellow
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#CC79A7',  # reddish purple
]
# Apply globally:
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=OKABE_ITO)
```

**B&W readability** — enforce variable markers + linestyles cycled together:

```python
import matplotlib as mpl
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = (
    cycler(color=OKABE_ITO[:6]) +
    cycler(linestyle=['-', '--', '-.', ':', '-', '--']) +
    cycler(marker=['o', 's', '^', 'D', 'v', 'x'])
)
```

For heatmaps (Fig 9 `experiment3_threshold_heatmap.png`): switch colormap from default to **`cividis`** or **`viridis`** — both are perceptually-uniform and colorblind-safe (Nuñez et al. 2018). `cividis` is the stricter choice for journal figures.

**Source:** Okabe & Ito 2008 "Color Universal Design" — the canonical colorblind palette. Matplotlib documentation confirms `viridis`/`cividis` as colorblind-safe defaults.

### POL-03 — 300 DPI export

```python
fig.savefig('experiments/figures/walk_forward_pnl.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
```

**Implementation:** add a shared save helper `src/plotting/save_fig.py`:

```python
# src/plotting/save_fig.py
from pathlib import Path
def save_ieee_fig(fig, path, dpi=300):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.05)  # vector version for LaTeX
```

Re-render 11 figures listed in the figure inventory table above using this helper.

### POL-04 — Abstract ≤ 250 words

Execute the 6 cut targets in the abstract word count audit above. Target landing: ~248 words with per-pair Sharpe sentence inserted. Verification command: `awk '/^## Abstract/{f=1;next} /^---|^##[^#]/{if(f)exit} f' PAPER_DRAFT.md | wc -w`.

### POL-05 — Citation format consistent

Alphabetize references 1–13 by first author; add missing entry for `Cont et al. 2014` (or remove in-text citation on line 536). No in-text numeric citations exist, so renumbering the reference list is safe.

### POL-06 — Every figure referenced + units on axes

Re-render all 11 figures in scope with explicit `ax.set_xlabel(..., ...)` / `ax.set_ylabel(..., ...)` calls including units in parentheses. Add in-text references for any currently unreferenced figures that are being kept (or drop them from `experiments/figures/` to avoid confusion — preferred for the 7 unreferenced files).

### POL-07 — Headline Sharpe = per-pair (≈ 3.2), per-trade (0.44) in footnote

1. Fix `PAPER_DRAFT.md` line 67: replace stale "0.59 / 4.3" contribution claim with "per-pair Sharpe ≈ 3.2 (Table 8)" and move per-trade Sharpe explanation to a footnote.
2. Insert one sentence in Abstract introducing per-pair Sharpe ≈ 3.2 (see POL-04 breakdown).
3. Update §8 Conclusions line 634 from "honest annualized Sharpe is 2–4" to "per-pair annualized Sharpe is ≈ 3.2 (robust range 2–4 under realistic slippage assumptions)" — this strengthens consistency with Table 8.

### POL-08 — Survivorship disclaimer + Fig 2 cap annotation

1. §6.4 already has survivorship-bias item 3 — verify wording, keep as-is.
2. Add 3 new items to §6.4 (live-cohort truncation, category-tagging gaps, crypto regime flip) per "§6.4 Limitations audit" above.
3. Regenerate Fig 2 (`pnl_at_2pp_vs_data.png`) with cap annotation embedded in the figure title and a standalone caption block added to §5.4 (see "Figure 2 caption cap annotation" above).

### POL-09 — AI-assistant disclosure in Acknowledgments

Already present (line 648). No action required unless exact-wording verification is demanded by rubric — current wording is a superset of the required statement.

### POL-10 — Cover-to-cover review, all TODOs/placeholders cleared

Single literal placeholder present (line 492). Replace with standard Fig 11 reference. Then manual read-through by both authors:
- Ian: integrity (numbers match tables, cross-references valid)
- Alvin: readability (flow, typos, clarity)

Introduce a final-pass checklist in PLAN.md:
1. `grep -nE "TODO|FIXME|XXX|\[Insert|\[X\]|TBD|\\$[0-9.]+NaN|None|null" PAPER_DRAFT.md` returns zero hits.
2. All `§N.M` cross-references point to a real section heading.
3. All `Fig N` / `Table N` references point to a real object with that number.
4. All numeric claims match the most recent JSON in `experiments/results/` (spot-check 5 random tables).

### POL-11 — README with per-table reproduction commands

New file `README.md` at repo root. Proposed skeleton:

```markdown
# DS340 Final Project — Kalshi vs. Polymarket Arbitrage
Ian Sabia & Alvin Jang, Boston University, Spring 2026.

## Quick start
    git clone https://github.com/iansabia/DS340-Project.git
    cd DS340-Project
    python3.12 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    export PYTHONPATH=$(pwd)

## Reproducing every paper table and figure

| Paper object | Command | Output |
|---|---|---|
| Table 2 (headline backtest) | `python experiments/verify_headline.py` | `experiments/results/tier1/*.json` |
| Table 3, Fig 1 (walk-forward) | `python experiments/run_walk_forward.py --windows 12` | `experiments/results/walk_forward/*.json`, `experiments/figures/walk_forward_pnl.png` |
| Table 4 (per-category) | `python experiments/run_category_breakdown.py` | `experiments/results/category/*.json` |
| Table 5, Fig 2 (data scaling) | `python experiments/run_data_scaling.py --bars-per-pair 250` | `experiments/results/data_scaling/*.json`, `experiments/results/data_scaling/pnl_at_2pp_vs_data.png` |
| Table 6 (XGB sweep) | `python experiments/run_experiment1_comparison.py --xgb-sweep` | `experiments/results/xgb_sweep.json` |
| Table 7 (transaction costs) | `python experiments/run_transaction_costs.py` | `experiments/figures/transaction_cost_sensitivity.png` |
| Table 8 (Sharpe accounting) | Derived from Table 2 — see `experiments/verify_headline.py` output fields `sharpe_per_trade` and `sharpe_per_pair` | — |
| §5.9 (reconciliation) | `python experiments/run_live_reconciliation.py` | `experiments/results/reconciliation/*` |
| Table 9 (feature ablation) | `python experiments/run_feature_ablation.py` | `experiments/results/ablation/*.json` |
| Table 10, Fig 11 (ensemble) | `python experiments/run_ensemble_sweep.py` | `experiments/results/ensemble/summary.json`, `experiments/figures/ensemble_weight_sweep.png` |
| Fig 5 (SHAP) | `python experiments/run_shap_analysis.py` | `experiments/figures/shap_bar_plot.png` |
| Fig 10 (TFT VSN) | `python experiments/run_tft.py && python experiments/extract_tft_heatmap.py` | `experiments/figures/tft_variable_importance.png` |

Total runtime on single CPU: ~2–3 hours. GPU not required.

## Live paper-trading system
See `docs/SCC_DEPLOYMENT.md` for deployment instructions.
```

**Pre-submission validation:** actually execute every command and confirm it runs clean from a fresh venv. Any command that fails must be fixed before submission.

### POL-12 — 4-minute lightning-talk slides

**Format recommendation: Marp (`marp-cli`)** — Markdown → PDF/HTML slides.

**Rationale (evidence-based, not preference):**

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **LaTeX Beamer** | IEEE-consistent look | Requires MacTeX (v1.1 non-goal is avoiding MacTeX); slow iteration; overkill for 7 slides | Reject |
| **Typst** | Fast compile, modern | New tool; team has no prior exposure; unknown classroom-computer rendering | Reject |
| **Marp** | Markdown source (same as paper); `marp-cli` exports PDF directly; zero new dependencies (Node.js already available in dev tooling); classroom computer shows PDF | Less styling polish than Beamer | **Accept** |
| **Apple Keynote** | Native on Ian's Mac | Not source-controlled; not reproducible; classroom may not have Keynote | Reject |
| **Google Slides** | Cheap, quick, classroom browser will render | Not source-controlled | Acceptable fallback if Marp setup takes > 15 min |

**Marp install + workflow:**
```bash
npm install -g @marp-team/marp-cli
# Create slides.md at repo root
# Export:
marp slides.md -o slides.pdf --theme uncover  # or 'gaia' or 'default'
```

**Slide skeleton (7 slides, one Markdown file):**

```markdown
---
marp: true
theme: default
paginate: true
---
# Complexity Is Not an Edge
## Cross-platform prediction-market arbitrage
Ian Sabia & Alvin Jang — DS340 Spring 2026

---
## Problem
- Kalshi and Polymarket list the same events at different prices
- Discrepancies can persist for hours — arbitrage opportunity?
- Central Q: does model complexity improve detection?

---
## Methods
- 4-tier model stack: LR, XGBoost → GRU, LSTM, TFT → PPO → PPO+autoencoder
- 59 engineered features incl. 13 academic microstructure estimators
- 5 evaluation regimes: single-split, walk-forward, per-category, scaling curve, live paper-trading
- Live system: 15-min trade cycle on BU SCC; 10,154 closed positions

---
## Challenge
- Three silent-failure bugs looked like model problems:
  - Kalshi /events returned 429 on 40% of calls silently
  - Polymarket `condition_id=` returned random markets (plural `condition_ids=` works)
  - `live_NNNN` pair_id drift across 3 code paths → fixed via content-addressed IDs
- Fix: infrastructure monitoring before model tuning

---
## Results
![width:700px](experiments/figures/walk_forward_pnl.png)
- XGBoost (+$201.63) ≈ LR (+$201.69) > LSTM (+$182.72) > GRU (+$174.11) >> PPO (−$7,724)
- Every walk-forward window profitable; per-pair Sharpe ≈ 3.2
- Alpha lives in matching pipeline + oil/commodities class, not in models

---
## Conclusions
- **Simplest models win at this data scale** — direct answer to research question
- Negative result on PPO is real evidence, not a bug
- Autonomous live system accumulating data; ranking may flip at 500+ bars/pair
- Project lesson: evaluation regime matters more than model family
```

**Delivery plan:**
- Slides due night of 2026-04-27 (if 4/28 slot) or 2026-04-29 (if 4/30 slot)
- Rehearse once for time: each section ≤ 40 seconds
- Export PDF and PNG fallback; upload to Blackboard

---

## SciencePlots IEEE Styling Recipe

**Canonical recipe — copy-paste into `src/plotting/ieee_style.py`:**

```python
"""
IEEE figure styling for the DS340 paper.
Import this module at the top of any figure-generating script.

Usage:
    from src.plotting.ieee_style import apply_ieee_style, OKABE_ITO, save_ieee_fig
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.plot(x, y, label='LR')  # automatic colorblind colors + markers
    ax.set_xlabel('Training rows (count)')
    ax.set_ylabel('P&L ($)')
    ax.legend()
    save_ieee_fig(fig, 'experiments/figures/myfig.png')
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

try:
    import scienceplots  # noqa: F401  # registers 'science', 'ieee', 'no-latex' styles
    _HAS_SCIENCEPLOTS = True
except ImportError:
    _HAS_SCIENCEPLOTS = False

OKABE_ITO = [
    '#000000', '#E69F00', '#56B4E9', '#009E73',
    '#F0E442', '#0072B2', '#D55E00', '#CC79A7',
]

def apply_ieee_style():
    """Apply IEEE style with colorblind-safe palette and B&W-readable markers."""
    if _HAS_SCIENCEPLOTS:
        plt.style.use(['science', 'ieee', 'no-latex'])
    else:
        # Fallback: plain matplotlib with sensible defaults
        plt.style.use('default')
        mpl.rcParams.update({
            'font.family': 'serif',
            'font.size': 8,
            'axes.labelsize': 8,
            'axes.titlesize': 9,
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
            'legend.fontsize': 7,
            'figure.figsize': (3.5, 2.5),  # IEEE single-column width
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'lines.linewidth': 1.0,
            'axes.grid': True,
            'grid.alpha': 0.3,
        })
    # Override color + linestyle + marker cycle for colorblind + B&W readability
    mpl.rcParams['axes.prop_cycle'] = (
        cycler(color=OKABE_ITO[:6]) +
        cycler(linestyle=['-', '--', '-.', ':', '-', '--']) +
        cycler(marker=['o', 's', '^', 'D', 'v', 'x'])
    )
    mpl.rcParams['lines.markersize'] = 4
    # Colorblind-safe heatmap default
    mpl.rcParams['image.cmap'] = 'cividis'

def save_ieee_fig(fig, path, dpi=300):
    """Save figure at 300 DPI, also dump a PDF vector copy next to it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
    try:
        fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.05)
    except Exception:
        pass  # PDF backend may fail on some installs; PNG is canonical
```

**Per-figure integration example (walk_forward_pnl.png):**

```python
from src.plotting.ieee_style import apply_ieee_style, save_ieee_fig
import matplotlib.pyplot as plt
import pandas as pd

apply_ieee_style()
data = pd.read_json('experiments/results/walk_forward/summary.json')

fig, ax = plt.subplots(figsize=(3.5, 2.5))
for model in ['LR', 'XGBoost', 'GRU', 'LSTM']:
    ax.plot(data['window'], data[f'{model}_pnl'], label=model)
ax.set_xlabel('Walk-forward window (index)')
ax.set_ylabel('P&L at 2 pp fees ($)')
ax.set_title('Walk-forward out-of-sample P&L (11 windows)')
ax.legend(loc='best', frameon=False)
save_ieee_fig(fig, 'experiments/figures/walk_forward_pnl.png')
```

**Fallback path (POL-01):** if `scienceplots` fails to import on Alvin's machine, `apply_ieee_style()` falls back to plain matplotlib with IEEE-compatible rcParams. This satisfies the POL-01 requirement "Falls back to plain matplotlib if Alvin's machine lacks pip."

---

## Slides Format Recommendation (POL-12)

**Decision: Marp.** Full rationale in POL-12 section above.

**Tradeoff summary:**
- **Speed-to-result:** Marp beats Beamer by ~30 minutes of setup (no MacTeX install) and ties Keynote.
- **Reproducibility:** Marp ties Beamer (both source-controlled); beats Keynote/Google Slides.
- **Classroom compatibility:** Marp exports PDF — universally renderable. Ties Beamer/Keynote/Google.
- **Team familiarity:** Markdown matches the `PAPER_DRAFT.md` workflow. Beats every alternative.
- **Risk:** Marp's default themes are plainer than Beamer. For a 4-minute talk with 7 slides, this is a feature, not a bug — avoids visual distraction.

**Fallback plan:** if Marp install fails, Google Slides is acceptable. Absolutely NOT Beamer (MacTeX dependency explicitly ruled out per v1.1 non-goals).

---

## Reproduction Commands Inventory

Final table — to be copy-pasted into the new `README.md` (see POL-11 section above). Script existence **must be verified by running `ls experiments/`**. Current listing shows present: `run_baselines.py`, `run_walk_forward.py`, `run_category_breakdown.py`, `run_transaction_costs.py`, `run_feature_ablation.py`, `run_ensemble_sweep.py`, `run_live_reconciliation.py`, `run_shap_analysis.py`, `run_tft.py`, `extract_tft_heatmap.py`, `verify_headline.py`, `run_bootstrap_ci.py`, `run_experiment1_comparison.py`, `run_experiment2_lookback.py`, `run_experiment3_threshold.py`, `run_backtest.py`, `check_reproducibility.py`.

**Scripts potentially missing** (must verify during plan execution): `run_data_scaling.py` (not in the top-level `experiments/` listing captured earlier; may be in a subfolder). If missing, the 50/100/250 scaling commands need a replacement entry point or the README has to point at `run_experiment1_comparison.py` with scaling flags.

**Potential gap:** no dedicated `run_xgb_hyperparam_sweep.py` visible. Sweep results may live inside `run_experiment1_comparison.py` with a `--xgb-sweep` flag or similar. Phase 14 plan must resolve this (either find the script or flag it as an out-of-scope missing artifact).

---

## Validation Architecture

**Nyquist validation enabled** (`.planning/config.json` → `workflow.nyquist_validation: true`). Every POL-* requirement is verifiable via automated grep / word-count / file-existence / command-exit-code checks completable in < 30 seconds.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | No code-level unit tests for Phase 14 (pure documentation/figure phase); use shell-level validators |
| Config file | None — validators live in `scripts/validate_phase14.sh` (new, Wave 0) |
| Quick run command | `bash scripts/validate_phase14.sh` |
| Full suite command | `bash scripts/validate_phase14.sh --strict` (includes PDF-build step) |
| Phase gate | All POL-01 through POL-12 validators green before `/gsd:verify-work` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| POL-01 | SciencePlots IEEE style applied to figures | smoke | `python -c "import scienceplots, matplotlib.pyplot as plt; plt.style.use(['science','ieee','no-latex']); print('OK')"` and `grep -l "apply_ieee_style\\|plt.style.use.*ieee" experiments/*.py \| wc -l` ≥ 10 | ❌ Wave 0 — create `src/plotting/ieee_style.py` |
| POL-02 | Colorblind-safe palette + variable markers on all figs | grep | `grep -q "OKABE_ITO\\|colorblind\\|cividis" src/plotting/ieee_style.py` | ❌ Wave 0 |
| POL-03 | 300 DPI on all saved figures | grep | `grep -c "dpi=300" src/plotting/ieee_style.py experiments/*.py` ≥ 2; and `file experiments/figures/*.png \| grep -c "300 x 300 DPI\\|300x300"` should equal number of re-rendered figures (fallback: verify via `identify` from ImageMagick if installed, otherwise read PIL metadata) | ❌ Wave 0 |
| POL-04 | Abstract ≤ 250 words | word count | `awk '/^## Abstract/{f=1;next} /^---$\|^##[^#]/{if(f)exit} f' PAPER_DRAFT.md \| wc -w` returns ≤ 250 | ✅ existing file |
| POL-05 | Citation format consistent (alphabetical) | grep | `awk '/^## References/,/^---$/' PAPER_DRAFT.md \| grep -E "^[0-9]+\\." \| awk '{print $2}' \| sort -c` exits 0 (already sorted) | ✅ existing file |
| POL-06 | Every figure referenced + axis labels have units | grep | `grep -cE "Fig\\. [0-9]+\\\|Figure [0-9]+\\\|Fig [0-9]+" PAPER_DRAFT.md` ≥ 11; and (manual) review of each re-rendered figure PNG file (sample: use `python -c "from PIL import Image; im = Image.open('experiments/figures/walk_forward_pnl.png'); print(im.size)"` to confirm non-default IEEE size) | ✅ existing file |
| POL-07 | Abstract headline Sharpe = per-pair (≈ 3.2); per-trade in footnote/Table 8 only | grep | `awk '/^## Abstract/{f=1;next} /^---$/{if(f)exit} f' PAPER_DRAFT.md \| grep -q "per-pair\\\|3\\.2"` AND `grep -n "0\\.44\\\|0\\.59\\\|4\\.3" PAPER_DRAFT.md` returns zero matches in §1.4 contributions (line range 51–70) | ✅ existing file |
| POL-08 | §6.4 Limitations survivorship disclaimer present; Fig 2 caption has cap annotation | grep | `awk '/^### 6\\.4 Limitations/,/^---$\|^##[^#]/' PAPER_DRAFT.md \| grep -iq "survivorship"` AND `grep -B2 -A8 "Figure 2\\|Fig\\. 2" PAPER_DRAFT.md \| grep -q "plateau\\\|cap\\\|6,802\\\|fixed pair universe"` | ✅ existing file |
| POL-09 | AI-assistant disclosure in Acknowledgments | grep | `awk '/^## Acknowledgments/,/^---\$\|^##[^#]/' PAPER_DRAFT.md \| grep -iq "claude\\\|anthropic.*assistant"` | ✅ existing file |
| POL-10 | No TODOs / placeholders in final draft | grep | `grep -cE "TODO\\\|FIXME\\\|XXX\\\|\\[Insert\\\|\\[X\\]\\\|TBD" PAPER_DRAFT.md` equals 0 | ✅ existing file |
| POL-11 | README has exact commands for every paper table | file + grep | `test -f README.md` AND `grep -c "python experiments/" README.md` ≥ 10 (one per table/figure) AND dry-run spot check: `python experiments/verify_headline.py --help` exits 0 | ❌ Wave 0 — create `README.md` |
| POL-12 | 4-minute lightning-talk slides produced + PDF export | file | `test -f slides.md && test -f slides.pdf` AND `pdfinfo slides.pdf \| grep "Pages:" \| awk '{print $2}'` returns 6–8 (one title + six required sections ± buffer slide) | ❌ Wave 0 — create `slides.md` |

### Integrity tests (covering POL-10 multiple failure modes)

| Check | Command | Pass criterion |
|---|---|---|
| All section cross-references resolve | `grep -oE "§[0-9]+\\.[0-9]+" PAPER_DRAFT.md \| sort -u \| while read ref; do awk -v r="$ref" 'BEGIN{n=substr(r,3)} /^###/ && $2==n{found=1} END{exit !found}' PAPER_DRAFT.md \|\| echo "MISSING: $ref"; done` | Outputs nothing (all refs resolve). Currently FAILS for `§4.6` → must be fixed to `§5.8`. |
| All Table N references match a Table N header | Custom awk comparing `Table [0-9]` citations vs `\*\*Table [0-9]+` headers | 0 unmatched |
| All Fig N references match an Appendix B entry | Custom awk comparing `Fig\\. [0-9]+` vs Appendix B bullets | 0 unmatched |
| No duplicate table numbers | `grep -oE "^\\*\\*Table [0-9]+" PAPER_DRAFT.md \| sort \| uniq -d` | Empty output. Currently FAILS with Table 6 & Table 7 duplicates — fixed by renumber. |
| No duplicate figure numbers | Search Appendix B + in-body | 0 duplicates. Currently FAILS with Fig 2 duplicate — fixed by renumber. |

### Sampling Rate
- **Per task commit:** `bash scripts/validate_phase14.sh` (runs all shell-level checks; < 5 seconds)
- **Per wave merge:** `bash scripts/validate_phase14.sh --strict` (adds PDF build via `pandoc PAPER_DRAFT.md -o paper.pdf` and `marp slides.md -o slides.pdf`)
- **Phase gate:** full suite green + manual cover-to-cover read of `paper.pdf` by both authors before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `src/plotting/ieee_style.py` — new module, wraps SciencePlots + Okabe-Ito palette + `save_ieee_fig` helper (covers POL-01, POL-02, POL-03)
- [ ] `scripts/validate_phase14.sh` — new shell validator running all 12 POL checks (covers POL-10 integrity + all other POLs)
- [ ] `README.md` — new file at repo root (covers POL-11)
- [ ] `slides.md` — new Marp source (covers POL-12)
- [ ] `slides.pdf` — generated from `slides.md` via `marp-cli` (covers POL-12)
- [ ] Framework install: `npm install -g @marp-team/marp-cli` — only if slides export required; fallback is Google Slides

---

## Risks / Open Questions

1. **Script-name drift** (MEDIUM risk): Appendix A in `PAPER_DRAFT.md` references scripts that may not exist under the exact names listed (`run_xgb_hyperparam_sweep`, `run_transaction_cost_sensitivity` vs. actual `run_transaction_costs.py`). The POL-11 plan must execute each proposed README command from a clean shell and fix any name mismatches.

2. **Missing `run_data_scaling.py`** (MEDIUM): not visible in the earlier `ls experiments/*.py` output. Either the script lives in a subfolder, was renamed, or scaling was driven interactively. Must verify existence and, if missing, either add it or deprecate the Fig 2 command in README and point at manual reproduction notes.

3. **Figure 2 caption cap annotation** (LOW, but blocks SCAL-03 + POL-08): must be embedded *on the PNG itself* (via `ax.text()` or figure title) AND in the §5.4 prose caption. Two-sided fix.

4. **Marp install on Alvin's machine** (LOW): requires Node.js. If unavailable, Ian has Node so compiled `slides.pdf` can be committed to git — Alvin never needs to install Marp locally.

5. **Abstract trim preserves all required content** (LOW): cutting 67 words *and* adding a per-pair Sharpe sentence requires careful editing to avoid dropping the per-category and walk-forward claims. Draft the new abstract in a scratch file and compare side-by-side against the original before commit.

6. **Classroom-computer PDF compatibility** (LOW): Marp's default theme renders fine as PDF. The instructor's classroom computer runs standard Preview/Acrobat. Export once, visually inspect, submit.

7. **Session 6.4 already mentions survivorship bias** (INFO, not a risk): POL-08 is partially pre-satisfied. The plan must NOT replace item 3; it should ADD items 6–8 for live-cohort truncation, category-tagging gaps, and crypto regime flip.

8. **Fig 2 numbering conflict** (HIGH priority for planner): three different objects are called "Figure 2" in the current draft (Appendix B says walk-forward Sharpe; §5.4 says data-scaling; line 229 footnote says "Figure 2b" for TFT VSN). Pick ONE resolution (recommend: renumber Appendix B to match in-text order) and propagate consistently. This is the single largest POL-06/POL-10 risk.

9. **Stale contribution claim on line 67** (HIGH priority): the "0.59 / 4.3" numbers have been superseded by fresh April 17 verification that produced 0.44 per-trade and 3.2 per-pair. This is BOTH a correctness issue (stale numbers) and a POL-07 violation (wrong headline Sharpe in the contributions list). Highest-leverage single edit.

10. **Scope creep risk** (MEDIUM): Phase 14 is four days before a hard deadline. Do not expand to "also clean up unreferenced figures" unless time permits — the 7 unreferenced figure files in `experiments/figures/` are harmless noise. Keep them or delete them only as a Wave-2 nicety.

---

## Sources

### Primary (HIGH confidence)

- Local filesystem inspection of `PAPER_DRAFT.md`, `FINDINGS.md`, `experiments/figures/`, `experiments/results/ensemble/summary.json`, `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md`, `.planning/STATE.md`, `requirements.txt` — all read directly 2026-04-23.
- Local `scienceplots 2.2.1` install verified via `python -c "import matplotlib.pyplot as plt; print([s for s in plt.style.available if 'science' in s or 'ieee' in s or 'no-latex' in s])"` — returned `['bright', 'high-contrast', 'ieee', 'no-latex', 'science', ...]`.
- `DS340FinalPresentationAndProjectInstrs_S26.pdf` — instructor-issued rubric read in full (3 pages). Confirms: 4 minutes, 6 required sections, PDF slides due night before presentation, April 28 or 30.
- `.planning/config.json` — confirmed `workflow.nyquist_validation: true`.

### Secondary (MEDIUM confidence)

- SciencePlots GitHub project documentation at `github.com/garrettj403/SciencePlots` — ships `science`, `ieee`, `no-latex` styles; verified by local install matching expected style list.
- Okabe-Ito colorblind palette — canonical 2008 reference, reproduced by matplotlib community; hex codes verified against multiple sources.
- Marp CLI project documentation at `github.com/marp-team/marp-cli` — standard Markdown-to-slide workflow, installs via `npm install -g @marp-team/marp-cli`.

### Tertiary (LOW confidence)

- None. All claims in this research document are backed by direct file inspection or by widely-used, stable library references.

---

## Metadata

**Confidence breakdown:**
- Paper integrity audit (TODOs, numbering, Sharpe, abstract): **HIGH** — direct file reads + line-numbered grep
- Figure inventory: **HIGH** — direct `ls` of `experiments/figures/`
- SciencePlots styling recipe: **HIGH** — verified against local install
- Marp slide recommendation: **MEDIUM** — team has no prior Marp usage; recommendation based on feature comparison, fallback plan in place
- Script-name drift risk: **MEDIUM** — cannot fully verify without running each README command; plan must resolve during execution

**Research date:** 2026-04-23
**Valid until:** 2026-04-27 (submission deadline; deliberately tight window — research is phase-specific and will not be reused)
