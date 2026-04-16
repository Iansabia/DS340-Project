# Project Research Summary — v1.1

> **MILESTONE SCOPE:** This summary covers the **v1.1 "Extended Evidence & Submission" milestone only**.
> All four source files (STACK.md, FEATURES.md, ARCHITECTURE.md, PITFALLS.md) **extend, not replace**,
> the v1.0 research from 2026-04-01. The v1.0 baseline pipeline (Python 3.12, PyTorch, SB3, sklearn,
> SHAP, matplotlib, full data/matching/feature/model stack) is **already built and working**.
> v1.1 adds: TFT, live-vs-backtest reconciliation, feature ablation, ensemble formalization,
> 250-bar scaling checkpoint, and paper polish. The v1.0 SUMMARY.md above this one remains as history.

**Project:** DS340 Final — Kalshi vs. Polymarket Price Discrepancies
**Domain:** Cross-platform prediction market arbitrage (ML complexity-vs-performance analysis)
**Researched:** 2026-04-16
**Deadline:** 2026-04-27 (11 days from research date)
**Confidence:** HIGH overall

---

## Executive Summary

The v1.0 system is a working, live paper-trading pipeline that fetches features from Kalshi and Polymarket, runs LR + XGBoost predictions with a concordance filter, and executes simulated trades on SCC on a 6-hour cycle. What v1.1 must deliver is the evidence layer that turns a working system into a publishable paper: a fourth model tier (TFT to complete the originally-promised model suite), a rigorous live-vs-backtest reconciliation (the paper's unique asset — almost no student paper has real live data), a systematic feature ablation (makes the parsimony thesis defensible), a formalized ensemble matching the live deployment, the 250-bar scaling checkpoint (fills the only pending item in Table 5), and publication-quality figures. The central complexity-vs-performance thesis needs all of these to be cohesive and credible.

The highest-risk capability is TFT. At 6,802 training rows and a 1.7 samples-per-parameter ratio, TFT is operating below every rule-of-thumb for safe training. The honest expected outcome is "TFT ties or loses to GRU/LSTM," which directly extends the simplicity-wins finding to a third architecture family and is a valid paper result. Hard time-box to one day; a negative result ("TFT did not converge at this dataset scale") is explicitly paper-worthy. The second critical risk is live-vs-backtest reconciliation: the MEMORY context flags an active pair_id schema bug (3 code paths disagree on `live_NNNN` meaning, live P&L since April 9 is suspect). This must be diagnosed and resolved before any reconciliation numbers can appear in the paper — budget 1–2 days for the schema audit before producing the comparison table.

All other capabilities are low-risk extensions of existing infrastructure. The architectural discipline for v1.1 is strict: five new files, approximately 10 lines of modifications to three existing files, zero changes to the `BasePredictor` interface, zero new frameworks. Integration elegance is a means to finishing before April 27, not an end in itself.

---

## Key Findings

### Top Stack Additions (from STACK.md)

The v1.0 stack is intact except for **environment drift**: `pytorch-forecasting` and `stable-baselines3` are NOT currently installed in `.venv/`, and Python has drifted from 3.12 to 3.14. Before any v1.1 phase executes, run `.venv/bin/pip install pytorch-forecasting==1.7.0 --dry-run` to confirm Python 3.14 wheel availability. If wheels are missing, rebuild the venv on Python 3.12.

**Add exactly three libraries:**

| Library | Version | Purpose | Why this one |
|---------|---------|---------|-------------|
| `pytorch-forecasting` | 1.7.0 | TFT training | `TimeSeriesDataSet` plumbing already exists from Phase 3; switching to `neuralforecast` or `darts` would cost a week of re-formatting |
| `quantstats` | 0.0.81 | Live-vs-backtest tearsheet | `qs.reports.html(returns, benchmark)` produces the comparison we need; active 2026 release; pyfolio-reloaded is heavier with a worse dependency tree |
| `SciencePlots` | 2.2.1 | Publication figures | One-liner `plt.style.use(['science', 'ieee', 'no-latex'])` gives IEEE styling; use `no-latex` variant to skip MacTeX install requirement for Alvin |

**No additional libraries.** Feature ablation uses `sklearn.inspection.permutation_importance` (already in sklearn 1.8.0) plus a 30-line custom LOGO loop. Ensemble uses a custom `EnsemblePredictor(BasePredictor)` via numpy; sklearn's `VotingRegressor` is incompatible with our Tier-2 models' 3D tensor inputs.

**Rejected (with rationale):** `neuralforecast`, `darts`, `pyfolio-reloaded`, `tueplots`, `mlforecast`, `optuna` — all add integration cost, framework conflict, or dep-tree complexity with no proportional benefit in 11 days.

### Top Features to Build (from FEATURES.md)

Six capabilities with clear must-ship scope and defer lists:

**C1 — TFT on small data** (must-ship; highest risk; starts the TFT→Ensemble critical path)
- Table stakes: `hidden_size=8–16`, `attention_head=1`, `dropout=0.3–0.5`, `QuantileLoss`, identical eval protocol to GRU/LSTM, VSN attention heatmap, `GroupNormalizer` per-pair
- Key differentiator: VSN attention heatmap and temporal attention heatmap (free output of `model.interpret_output`; shows which features TFT actually uses vs. SHAP)
- Hard stop: if val_loss does not beat GRU within 1 day, report "TFT did not converge on 6,802 rows" as the finding — this directly supports the simplicity-wins thesis

**C2 — Live vs. backtest reconciliation** (must-ship; unique paper asset; pair_id bug blocks this)
- Table stakes: defined reconciliation window (Apr 9–25, ~1,632 cycles), trade-level pairing, summary comparison table, tracking error, exit-reason attribution, category-level breakdown, explicit paper-trading caveats
- Key differentiator: live-only oil-vs-non-oil breakdown (confirms or refutes Finding 6 in live data)
- Prerequisite: pair_id schema audit must complete before any reconciliation numbers can be trusted

**C3 — Feature ablation** (must-ship; fast to execute; critical for parsimony claim)
- Table stakes: LOGO over 5 feature groups on LR + XGBoost, delta-P&L table with bootstrap CIs, two-model comparison (LR vs. XGBoost separately)
- Expected finding: Polymarket microstructure group is load-bearing (consistent with SHAP polymarket_vwap dominance from Finding 5); classical microstructure group is droppable (consistent with Finding 12)
- Critical discipline: pre-register protocol before running; use three-way train/ablation-holdout/final-test split to avoid p-hacking

**C4 — Ensemble formalization** (must-ship; documents the live deployment; low risk)
- Table stakes: four variants (LR, LR+XGB equal-weight, LR+LSTM, LR+XGB majority-vote), concordance filter audit showing both filtered and unfiltered P&L, explicit statement that ensemble matches live deployment
- Key differentiator: ensemble weight sensitivity sweep (LR-weight 0.0→1.0 in steps of 0.1; one plot; shows the choice is not cherry-picked)

**C5 — 250-bar scaling checkpoint** (passive wait; fills Table 5 row; no development work needed)
- Auto-retrain fires when live system accumulates 250 bars/pair (ETA 12–24h from 2026-04-16)
- Just collect the output and fill in Table 5; update Finding 22 from "pending" to actual numbers

**C6 — Paper polish** (must-ship last; can partially parallelize with C4)
- Table stakes: color-blind palette (`plt.style.use(['science', 'ieee', 'no-latex'])` or seaborn-colorblind), variable line styles + markers, 300 DPI, abstract under 250 words (current draft is 412 words), consistent citation format, all 9 figures referenced in text, axis labels with units

**Defer to v2+/out of scope:** TFT in live deployment, PatchTST/Autoformer implementations, FRTB P&L attribution decomposition, stacking regression meta-model, Optuna HPO.

### Top Architectural Integration Points (from ARCHITECTURE.md)

The guiding principle is explicit: fit into the existing skeleton, never reshape it. Every new component either implements an existing ABC or adds to `experiments/` in the existing style.

**New files (5 modules + wrappers):**

| File | Responsibility | Pattern |
|------|---------------|---------|
| `src/models/tft.py` | `TFTPredictor(BasePredictor)` — hides `TimeSeriesDataSet` inside `fit()`, exposes row-aligned `predict()` | Mirrors `GRUPredictor` exactly |
| `src/models/ensemble.py` | `EnsemblePredictor(BasePredictor)` with weighted average and concordance mode | `BasePredictor` subclass; picklable for live deployment |
| `src/analysis/__init__.py` + `reconciliation.py` | Pure reconciliation logic; joins `positions.db` + backtest predictions | New `src/analysis/` subpackage; logic testable without spinning up CLI |
| `experiments/run_tft.py` | Thin wrapper over `run_tier2_with_seeds` | ~80 LOC |
| `experiments/run_feature_ablation.py` | Filters `X[subset]` before `model.fit()`; saves to `results/ablation/` | ~200 LOC; never modifies `BasePredictor` interface |
| `experiments/run_ensemble_sweep.py` | Instantiates `EnsemblePredictor` variants | ~100 LOC |
| `experiments/run_live_reconciliation.py` | CLI wrapper over `src/analysis/reconciliation.py` | ~40 LOC |

**Modified files (~10 lines total across 3 files):**
- `experiments/run_baselines.py` — add `TFTPredictor` import, insert into `tier2` list and `_MODEL_ORDER` (≤5 lines)
- `experiments/run_walk_forward.py` — add `'tft'` to the sequence-model branch (≤3 lines)

**Five integration contracts that must be honored:**
1. `TFTPredictor.fit(X, y)` consumes `(X_with_group_id_column, y_array)` exactly as `GRUPredictor` does
2. Reconciliation uses `src.evaluation.profit_sim.simulate_profit` — not a parallel P&L calculator
3. Reconciliation reads `positions.db` via `PositionManager.get_closed_positions()`, not raw SQLite
4. Feature ablation filters at experiment boundary (`X[subset]`), never inside `fit()`
5. `EnsemblePredictor` is picklable via `BasePredictor.save/load` for potential live deployment

**Test discipline (tiered by risk, not uniform):**
- Full unit + integration: `src/analysis/reconciliation.py` (wrong timestamp alignment = wrong paper numbers), `src/models/ensemble.py` (candidate for live deployment)
- Smoke test only: `src/models/tft.py`, `experiments/run_feature_ablation.py`
- No tests needed: thin experiment wrappers

### Top Pitfalls to Design Against (from PITFALLS.md)

These seven translate directly into acceptance criteria in REQUIREMENTS.md:

**P1 — TFT silent overfitting on 6,802 rows** (Phase 11, SEVERITY: HIGH)
- Prevention: `hidden_size=16`, `dropout=0.3`, early stopping on walk-forward P&L not in-sample loss, attention entropy audit after each run
- Acceptance criterion: if `entropy(attention_weights) < 0.5 × log(n_features)` or `max_variable_weight > 0.8`, log a warning and report as degenerate; 1-day time-box, no extension

**P2 — Live-vs-backtest gap treated as noise** (Phase 9, SEVERITY: CRITICAL)
- Prevention: trade-level pairing on `(pair_id, entry_ts_bucket)`, single fee-function module used by both systems, UTC timestamps everywhere, price-source alignment (VWAP vs. mid-price) explicitly documented
- Acceptance criterion: `only_live + only_bt < 20% of matched_trades`; any gap above 20% is diagnosed and named in paper §5.9

**P3 — Ablation p-hacking via test-set selection** (Phase 12, SEVERITY: HIGH)
- Prevention: three-way split (train / ablation-holdout / final-test); minimum feature set selected on ablation-holdout only; protocol pre-registered before running; all ablation runs reported, not just winners
- Acceptance criterion: protocol doc written and committed before `run_feature_ablation.py` executes

**P4 — Concordance filter inflating Sharpe by shrinking denominator** (Phase 13, SEVERITY: CRITICAL)
- Prevention: always report both filtered and unfiltered P&L side-by-side; simulate P&L on rejected trades; flag if rejected trades are profitable (concordance filter is hurting real P&L while inflating paper Sharpe)
- Acceptance criterion: ensemble results table has explicit columns for "with concordance" and "without concordance"; rejection rate reported

**P5 — Sharpe reported without per-pair independence correction** (Phase 14, SEVERITY: CRITICAL)
- Prevention: headline Sharpe in every table must use `per_pair_sharpe()` from existing `src/evaluation/sharpe.py`; per-trade Sharpe in footnote only; annualized Sharpe computed only on daily aggregates
- Acceptance criterion: no Sharpe above 4.0 in any table without explicit per-pair independence disclaimer; abstract cites per-pair-corrected number

**P6 — Reproducibility failure from unseeded stochastic training** (Phase 8, SEVERITY: CRITICAL)
- Prevention: `torch.manual_seed`, `numpy.random.seed`, CUDNN deterministic flags, and DataLoader worker seeds all set via shared `src/utils/seed.py` called at top of every training script; re-run each model twice to verify identical results
- Acceptance criterion: running `run_baselines.py` twice in succession gives identical Table 2 numbers within 1%

**P7 — Scaling curve plateau presented as universal evidence** (Phase 10/14, SEVERITY: MEDIUM-HIGH)
- Prevention: annotate training-set cap on scaling figure x-axis; text explicitly states "plateau is an artifact of the fixed pair universe, not a fundamental architectural ranking"
- Acceptance criterion: Figure caption for Table 5 / scaling plot includes the cap annotation

---

## Implications for Roadmap

### Critical Path and Phase Structure

The ARCHITECTURE.md dependency analysis and FEATURES.md critical path agree on this structure:

```
Phase 8 ──┬──▶ Phase 9  (reconciliation — leaf, start immediately)    ──┐
          │                                                               │
          ├──▶ Phase 10 (250-bar wait — passive, no dev time)            ┤
          │                                                               │
          ├──▶ Phase 11 (TFT — longest dev task)                        ┤──▶ Phase 13 ──▶ Phase 14
          │                                                               │   (ensemble)   (paper)
          └──▶ Phase 12 (ablation — fast, independent)                  ┘
```

Phases 9, 10, 11, 12 can all run in parallel after Phase 8. Phase 13 depends on Phase 11 (for optional 4-model ensemble; design 13 to ship without TFT as a fallback). Phase 14 waits for 13.

**Phase 8 — Environment and baseline verification** (blocks everything)
- Deliver: clean venv (pytorch-forecasting installed, Python 3.14 dry-run verified), seed discipline implemented, v1.0 Table 2 numbers reproduced identically
- Must avoid: P6 (unseeded training gives different numbers each run)

**Phase 9 — Live-vs-backtest reconciliation** (leaf; start immediately after Phase 8)
- Deliver: pair_id schema audit resolved, `src/analysis/reconciliation.py`, trade-level comparison table, tracking error, category breakdown, §5.9 paper section
- Must avoid: P2 (treating gap as noise); using parallel P&L simulator; opening `positions.db` directly
- Research flag: schema audit needs user input (see Open Questions #1)

**Phase 10 — 250-bar scaling checkpoint** (passive; parallelize with 11 and 12)
- Deliver: GRU/LSTM row in Table 5, Finding 22 updated; 1–2 hours of analysis work when retrain fires
- Must avoid: P7 (plateau without cap annotation)

**Phase 11 — TFT implementation** (starts TFT→Ensemble critical path)
- Deliver: `src/models/tft.py`, `experiments/run_tft.py`, TFT row in Table 2, VSN heatmap figure
- Must avoid: P1 (default hyperparameters); spending >1 day; declaring TFT success without attention entropy check
- Research flag: MEDIUM — pre-specify dropout, hidden_size, and stop criteria in REQUIREMENTS.md as concrete acceptance criteria; do not leave as "tune if needed"

**Phase 12 — Feature ablation** (fast and independent; parallelize with 11)
- Deliver: `experiments/run_feature_ablation.py`, ablation table with bootstrap CIs, "minimum sufficient feature set" discussion
- Must avoid: P3 (p-hacking via test-set selection); LOFO on individual features instead of LOGO groups

**Phase 13 — Ensemble formalization** (waits on Phase 11)
- Deliver: `src/models/ensemble.py`, ensemble sweep results, concordance filter audit, ensemble table added to paper
- Must avoid: P4 (concordance filter inflating Sharpe); framing ensemble as the main contribution rather than as deployment documentation

**Phase 14 — Paper finalization** (terminal; waits for all)
- Deliver: SciencePlots IEEE styling on all figures, abstract trimmed to <250 words, citation style consistency, all TODOs cleared, final PDF reviewed cover-to-cover
- Must avoid: P5 (per-trade Sharpe in abstract); P6 (cherry-picked walk-forward window as headline)

### Research Flags

**Phases needing deeper research or user validation during planning:**
- **Phase 9 (Reconciliation):** The pair_id schema bug must be diagnosed with the user before phase planning can complete. Which code path is canonical for `live_NNNN`? Without this, reconciliation planning is speculative.
- **Phase 11 (TFT):** Hyperparameters for the 6,802-row / 47-bar-avg regime need to be pre-specified as acceptance criteria, not left as implementation-time decisions.

**Phases with clear patterns (no additional research needed):**
- Phase 12 (Ablation): LOGO over 5 groups on LR/XGBoost is fully specified.
- Phase 13 (Ensemble): `EnsemblePredictor(BasePredictor)` design is architecturally clear.
- Phase 14 (Paper polish): SciencePlots `no-latex` variant is the decided approach.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All library versions PyPI-verified April 2026; Python 3.14 wheel compatibility is a known risk with a concrete mitigation (dry-run before proceeding) |
| Features | HIGH | Strong literature anchors for all 6 capabilities; existing system provides internal calibration for what is feasible in 11 days |
| Architecture | HIGH | Grounded in the shipped v1.0 codebase; every new component extends an existing, tested contract |
| Pitfalls | HIGH | Well-established quantitative finance literature (Bailey & de Prado, Lim et al.) plus direct project evidence from v1.0 findings |

**Overall confidence:** HIGH

### Open Questions for the Requirements Author

These must be resolved with the user before REQUIREMENTS.md is finalized:

1. **Pair_id schema bug (BLOCKING Phase 9):** Which of the 3 code paths is the canonical `live_NNNN` interpretation? Live P&L since April 9 is suspect until this is diagnosed. The reconciliation section cannot produce trustworthy numbers without this answer.

2. **TFT time-box go/no-go gate:** Should Phase 11 have a formal go/no-go checkpoint between TFT implementation and Phase 13 — i.e., if TFT does not converge, Phase 13 proceeds without TFT as an ensemble member? Or should Phase 11 always be considered complete (with the negative result), and Phase 13 always gets a TFT-excluding ensemble as its baseline?

3. **Live trading end date:** FEATURES.md states the reconciliation window as "April 9–25." Confirm: should live trading continue through April 25 (2 days before submission), or should a cutoff date be set earlier to freeze the dataset?

4. **Ensemble deployment in v1.1 scope:** ARCHITECTURE.md notes that `EnsemblePredictor` could replace the hardcoded LR/XGB average in `src/live/strategy.py`. Is updating the live deployment in scope for v1.1, or is it strictly deferred to v1.2?

5. **SciencePlots teammate agreement:** Confirm that Alvin's machine can use `SciencePlots` with the `no-latex` variant. If any figures are being generated on SCC (which may not have a display), confirm a non-interactive matplotlib backend is set.

### Gaps to Address

- **Env drift verification:** The `pip install pytorch-forecasting==1.7.0 --dry-run` test on Python 3.14 has not been run. This is the first action in Phase 8.
- **Reproducibility baseline:** Whether current GRU/LSTM results reproduce with seeds is unknown. Phase 8 must verify before any Tier-2 numbers are cited.
- **Modified result files:** Git status shows `experiments/results/tier1/*.json` and `tier2/*.json` are modified on disk. Phase 8 must reconcile these against the paper's current Table 2 numbers before adding new results on top.

---

## Sources

### Primary (HIGH confidence — PyPI verified 2026-04-16)
- [pytorch-forecasting 1.7.0](https://pypi.org/project/pytorch-forecasting/) — TFT, TimeSeriesDataSet API
- [quantstats 0.0.81](https://pypi.org/project/quantstats/) — live-vs-backtest tearsheet
- [SciencePlots 2.2.1](https://pypi.org/project/SciencePlots/) — publication figure styling
- [sklearn 1.8.0 permutation_importance](https://scikit-learn.org/stable/modules/permutation_importance.html) — feature ablation
- [PyTorch Forecasting TFT API ref + GitHub Issue #1322](https://github.com/sktime/pytorch-forecasting/issues/1322) — small-data hyperparameter guidance

### Primary (HIGH confidence — academic literature)
- Lim et al. (2020), "Temporal Fusion Transformers," ScienceDirect — TFT design and small-data limitations (Sec 5.3)
- Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio" — Pitfalls 3, 4, 5
- ArXiv 2603.16886 (2026) — TFT ranks 6–8th of 10 architectures on financial data; ModernTCN and PatchTST lead
- Kaplan et al. (2020), Hoffmann et al. (2022) — neural scaling laws (context for scaling-curve pitfall)

### Secondary (MEDIUM confidence — project-specific)
- v1.0 FINDINGS.md — Finding 2 (GRU underperforms XGBoost), Finding 5 (polymarket_vwap dominates SHAP), Finding 12 (microstructure features neutral at 47 bars), Finding 17 (per-pair Sharpe correction), Finding 21 (per-category model rankings)
- QuantConnect Reconciliation Documentation — live-vs-backtest reconciliation framework
- v1.0 ARCHITECTURE.md — `BasePredictor` contract, `TimeSeriesDataSet` plumbing, `save_results` schema

---

*Research completed: 2026-04-16*
*Milestone: v1.1 Extended Evidence & Submission*
*Source files synthesized: STACK.md, FEATURES.md, ARCHITECTURE.md, PITFALLS.md (all v1.1-specific)*
*Ready for roadmap: yes*
