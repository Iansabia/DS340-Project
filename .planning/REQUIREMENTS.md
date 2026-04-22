# Requirements: Kalshi vs. Polymarket Price Discrepancies

**Defined:** 2026-04-01
**Core Value:** Empirically answer whether model complexity improves cross-platform prediction market arbitrage detection

## v1 Requirements

### Data Pipeline

- [x] **DATA-01**: Kalshi API adapter that handles live/historical endpoint split via `/historical/cutoff`
- [x] **DATA-02**: Polymarket API adapter that queries Gamma for metadata, CLOB for prices, Data API for trade records
- [x] **DATA-03**: Polymarket price history reconstruction from trade records for resolved markets
- [x] **DATA-04**: Rate limiting and local caching for both platform APIs
- [x] **DATA-05**: Automated retry with exponential backoff for transient API failures
- [x] **DATA-06**: Raw data storage with timestamps in `data/raw/`

### Market Matching

- [x] **MATCH-01**: Keyword-based first-pass candidate matching across platforms
- [x] **MATCH-02**: Sentence-transformer semantic similarity scoring for fuzzy matching
- [ ] **MATCH-03**: Manual curation interface for reviewing and accepting/rejecting matched pairs
- [x] **MATCH-04**: Match confidence scoring per pair
- [ ] **MATCH-05**: Settlement criteria comparison documentation for each matched pair

### Feature Engineering

- [x] **FEAT-01**: Time-aligned hourly feature vectors (spread, volume, bid-ask spread, price velocity)
- [x] **FEAT-02**: Temporal train/test split enforced (no look-ahead bias)
- [x] **FEAT-03**: Low-liquidity market filtering (remove markets with <10 trades)
- [x] **FEAT-04**: Output format compatible with PyTorch Forecasting TimeSeriesDataSet
- [x] **FEAT-05**: Processed dataset saved to `data/processed/`

### Models -- Tier 1 (Regression Baselines)

- [x] **MOD-01**: Linear Regression trained on spread prediction
- [x] **MOD-02**: XGBoost trained on spread prediction
- [x] **MOD-03**: Naive baseline (spread always closes fully)
- [x] **MOD-04**: Volume baseline (higher-volume platform is always correct)

### Models -- Tier 2 (Time Series)

- [x] **MOD-05**: GRU trained on spread prediction with hourly sequences
- [x] **MOD-06**: LSTM trained on spread prediction with hourly sequences
- [x] **MOD-07**: TFT via PyTorch Forecasting (droppable if timeline tight)

### Models -- Tier 3 (RL)

- [x] **MOD-08**: PPO agent acting directly on raw microstructure features
- [x] **MOD-09**: Autoencoder trained on normal spread behavior for anomaly detection
- [x] **MOD-10**: PPO agent with autoencoder signal filter (acts on flagged opportunities)

### Evaluation

- [x] **EVAL-01**: Regression metrics computed for all models (RMSE, MAE, directional accuracy)
- [x] **EVAL-02**: Profit simulation for all models (P&L, win rate, Sharpe ratio)
- [x] **EVAL-03**: SHAP interpretability analysis on best-performing models
- [x] **EVAL-04**: Bootstrap confidence intervals on key metrics

### Experiments

- [x] **EXP-01**: Experiment 1 -- Complexity-vs-performance comparison across all tiers (centerpiece)
- [x] **EXP-02**: Experiment 2 -- Historical window length ablation (6h, 24h, 72h, 7d)
- [x] **EXP-03**: Experiment 3 -- Minimum spread threshold ablation (no min, >2pp, >5pp, >10pp)
- [x] **EXP-04**: Transaction cost sensitivity analysis

### Deliverables

- [ ] **DEL-01**: Final paper documenting methodology, experiments, and findings
- [ ] **DEL-02**: Lightning talk slides

## v2 Requirements

(Not applicable -- single-submission academic project)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Live trading / real-money execution | Historical backtesting only -- academic project |
| External event features (news, sentiment) | Microstructure-only by design to isolate signal |
| Pretrained models | All models trained from scratch per project rules |
| Third-party data aggregators | Direct API ingestion only |
| Transaction cost modeling in simulation | Acknowledged via sensitivity analysis but not modeled in P&L |
| Web dashboard or mobile app | Research project -- scripts and notebooks only |
| Model ensembling | Would obscure complexity comparison -- each model evaluated independently |
| Political/entertainment markets | Insufficient cross-platform overlap for matched pairs |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Complete |
| DATA-02 | Phase 1 | Complete |
| DATA-03 | Phase 1 | Complete |
| DATA-04 | Phase 1 | Complete |
| DATA-05 | Phase 1 | Complete |
| DATA-06 | Phase 1 | Complete |
| MATCH-01 | Phase 2 | Complete |
| MATCH-02 | Phase 2 | Complete |
| MATCH-03 | Phase 2 | Pending |
| MATCH-04 | Phase 2 | Complete |
| MATCH-05 | Phase 2 | Pending |
| FEAT-01 | Phase 3 | Complete |
| FEAT-02 | Phase 3 | Complete |
| FEAT-03 | Phase 3 | Complete |
| FEAT-04 | Phase 3 | Complete |
| FEAT-05 | Phase 3 | Complete |
| MOD-01 | Phase 4 | Complete |
| MOD-02 | Phase 4 | Complete |
| MOD-03 | Phase 4 | Complete |
| MOD-04 | Phase 4 | Complete |
| EVAL-01 | Phase 4 | Complete |
| EVAL-02 | Phase 4 | Complete |
| MOD-05 | Phase 5 | Complete |
| MOD-06 | Phase 5 | Complete |
| MOD-07 | Phase 5 | Complete |
| MOD-08 | Phase 6 | Complete |
| MOD-09 | Phase 6 | Complete |
| MOD-10 | Phase 6 | Complete |
| EVAL-03 | Phase 7 | Complete |
| EVAL-04 | Phase 7 | Complete |
| EXP-01 | Phase 7 | Complete |
| EXP-02 | Phase 7 | Complete |
| EXP-03 | Phase 7 | Complete |
| EXP-04 | Phase 7 | Complete |
| DEL-01 | Phase 14 | Pending |
| DEL-02 | Phase 14 | Pending |

**Coverage:**
- v1 requirements: 36 total
- Mapped to phases: 36
- Unmapped: 0

---

## v1.1 Requirements -- Extended Evidence & Submission

**Defined:** 2026-04-17
**Milestone goal:** Strengthen every pillar of the paper's evidence base -- scaling, model variety, feature understanding, execution realism -- so the April 27 submission is credible and signals continued deployment.
**Deadline:** 2026-04-27 (paper + code)

Each requirement derives from the v1.1 research (`.planning/research/SUMMARY.md`) and incorporates acceptance criteria from the 7 identified pitfalls (P1-P7).

### Phase 8 -- Environment & Baseline Verification

Gating phase; blocks all downstream work.

- [x] **ENV-01**: `pytorch-forecasting==1.7.0 --dry-run` succeeds on the current `.venv` Python 3.14, or the venv is rebuilt on Python 3.12. Must complete before any v1.1 model training runs.
- [x] **ENV-02**: Three target libraries installed and importable: `pytorch-forecasting==1.7.0`, `quantstats==0.0.81`, `SciencePlots==2.2.1`.
- [x] **ENV-03**: Shared seed utility at `src/utils/seed.py` applied at top of every training script (torch, numpy, Python random, CUDNN deterministic flag, DataLoader worker seed). Guards P6.
- [x] **ENV-04**: Running `experiments/verify_headline.py` twice in succession produces identical Table 2 numbers within 1% tolerance. Guards P6.
- [x] **ENV-05**: All current `PAPER_DRAFT.md` Table 2 numbers reproduce from the clean environment (reconciles the `experiments/results/tier1/*.json` modified-on-disk files with the paper's current numbers).

### Phase 9 -- Live vs Backtest Reconciliation

Leaf phase; starts immediately after Phase 8. **Unblocked** -- the pair_id schema bug was audited and confirmed resolved on 2026-04-17.

- [x] **RECON-01**: New `src/analysis/` subpackage with `reconciliation.py` module. Pure analysis logic, testable without CLI.
- [x] **RECON-02**: Reconciliation window: **April 11, 2026 -> April 25, 2026** (post pair_id fix through 2-day submission buffer). Pre-April-11 positions excluded (exit_reason=`force_close_schema_fix`).
- [x] **RECON-03**: Trade-level pairing on `(pair_id, entry_ts_bucket)` between `positions.db` closed positions and backtest predictions regenerated over same timestamps. Guards P2.
- [x] **RECON-04**: Single shared fee function (`src/evaluation/profit_sim.simulate_profit`) used by both live-capture and backtest-comparison sides. No parallel P&L calculator. Guards P2.
- [x] **RECON-05**: Summary comparison table: live P&L vs simulated P&L, tracking error, total trades matched, only-live count, only-backtest count.
- [x] **RECON-06**: Category-level breakdown (oil vs non-oil) comparing live vs simulated P&L per category. Directly tests Finding 6 on live data.
- [x] **RECON-07**: Exit-reason attribution table (TAKE_PROFIT, TIME_STOP, STOP_LOSS, MOMENTUM, RESOLUTION_EXIT) -- live vs simulated counts and P&L.
- [x] **RECON-08**: Acceptance gate: `(only_live + only_backtest) / matched_trades < 20%`. Any gap >= 20% is diagnosed and named in paper section 5.9.
- [x] **RECON-09**: New paper section 5.9 "Live vs Backtest Reconciliation" written with findings, explicit paper-trading caveats (no slippage, no partial fills).
- [x] **RECON-10**: `experiments/run_live_reconciliation.py` CLI wrapper (~40 LOC) over `src/analysis/reconciliation.py`.

### Phase 10 -- 250-Bar Scaling Checkpoint

Passive wait; parallelizes with Phases 11 and 12.

- [x] **SCAL-01**: 250-bar auto-retrain checkpoint output captured from SCC (ETA 12-24h from milestone start; may already have fired).
- [x] **SCAL-02**: Table 5 in paper updated with 3rd scale point (50 / 100 / 250 bars/pair).
- [x] **SCAL-03**: Figure 2 regenerated with explicit training-set-cap annotation ("plateau at N=6,802, fixed pair universe"). Guards P7.
- [x] **SCAL-04**: Finding 22 in `FINDINGS.md` filled in with actual numbers (currently marked pending).
- [x] **SCAL-05**: Paper section 5.4 updated to either confirm "ranking invariant across 5x data growth" or document any ranking shift.

### Phase 11 -- TFT (Temporal Fusion Transformer)

Starts the TFT -> Ensemble critical path.

- [x] **TFT-01**: New `src/models/tft.py` implementing `TFTPredictor(BasePredictor)` -- mirrors `GRUPredictor` pattern, hides `TimeSeriesDataSet` plumbing inside `fit()`, exposes row-aligned `predict()`.
- [x] **TFT-02**: Hyperparameters pre-specified for small-data regime: `hidden_size=8`, `attention_head_size=1`, `dropout=0.3`, `QuantileLoss`, `GroupNormalizer` per-pair. No implementation-time tuning. Guards P1.
- [x] **TFT-03**: Evaluated on identical protocol to GRU/LSTM: single-split backtest + walk-forward.
- [x] **TFT-04**: Hard 1-day time-box. If val_loss does not beat GRU within 24 hours, report "TFT did not converge at N=6,802" as a paper finding and move on. Go/no-go: **Option B** -- always include Phase 11 outcome (success or documented negative result); Phase 13 proceeds regardless with TFT-excluding baseline, and if TFT worked it's added as one more variant.
- [ ] **TFT-05**: Attention entropy audit after training. Flag degenerate if `entropy(attention_weights) < 0.5 * log(n_features)` or `max_variable_weight > 0.8`. Guards P1.
- [x] **TFT-06**: New `experiments/run_tft.py` thin wrapper (~80 LOC) over `run_tier2_with_seeds`.
- [ ] **TFT-07**: TFT row added to Tables 2 and 3 in paper. Paper section 4.1 updated to list 5 tiers if TFT converged, or note TFT attempt under section 6 Discussion if it did not.
- [ ] **TFT-08**: VSN feature-weight heatmap produced via `model.interpret_output` and saved to `experiments/figures/tft_variable_importance.png` (included as a differentiator figure).

### Phase 12 -- Feature Ablation Study

Fast and independent; parallelizes with Phase 11.

- [ ] **ABLA-01**: Pre-registered ablation protocol committed as `.planning/ablation_protocol.md` **before** `run_feature_ablation.py` executes. Guards P3.
- [ ] **ABLA-02**: LOGO (leave-one-group-out) across 5 feature groups: (a) raw aligned OHLCV, (b) cross-platform basics (spread/mid/divergence), (c) rolling/momentum, (d) classical microstructure (Amihud/Kyle/Roll/Corwin-Schultz), (e) prediction-market-specific (favorite-longshot etc.). **Not LOFO** over individual features (noisier).
- [ ] **ABLA-03**: Three-way temporal split: train / ablation-holdout / final-test. Minimum sufficient feature set selected on ablation-holdout only; final test untouched until after selection is frozen. Guards P3.
- [ ] **ABLA-04**: Bootstrap 95% CIs on per-group P&L deltas (1,000 resamples).
- [ ] **ABLA-05**: Ablation table reports ALL runs, not only favorable ones.
- [ ] **ABLA-06**: Two-model comparison (LR vs XGBoost) separately -- feature importance differs by model family.
- [ ] **ABLA-07**: New `experiments/run_feature_ablation.py` (~200 LOC). Filters `X[subset]` at experiment boundary; never modifies `BasePredictor.fit()` signature.
- [ ] **ABLA-08**: Paper section 5.X "Feature Ablation" added with table and parsimony discussion.

### Phase 13 -- Ensemble Formalization

Waits on Phase 11; ships with TFT-excluding baseline regardless.

- [ ] **ENSM-01**: New `src/models/ensemble.py` implementing `EnsemblePredictor(BasePredictor)`. Picklable via `BasePredictor.save/load`.
- [ ] **ENSM-02**: Four ensemble variants evaluated: (a) LR alone, (b) LR + XGBoost equal-weight, (c) LR + LSTM, (d) majority-vote LR + XGBoost + LSTM. If TFT converged: 5th variant adds TFT.
- [ ] **ENSM-03**: Concordance filter audit with BOTH filtered and unfiltered P&L in the same table. Includes rejection rate and P&L on rejected trades. If rejected trades are profitable in aggregate, flag as "concordance filter is hurting real P&L while inflating paper Sharpe." Guards P4.
- [ ] **ENSM-04**: Ensemble-weight sensitivity sweep: LR-weight from 0.0 to 1.0 in 0.1 increments (XGBoost-weight = 1 - LR-weight). One plot showing sweep is not cherry-picked.
- [ ] **ENSM-05**: `EnsemblePredictor` **not** wired into `src/live/strategy.py` during v1.1 (live deployment stays hardcoded to current LR+XGB average; rollout is a future-work bullet). Guards "breaking live system" risk.
- [ ] **ENSM-06**: New `experiments/run_ensemble_sweep.py` (~100 LOC).
- [ ] **ENSM-07**: Paper section 4.4 (Live System Architecture) rewritten with evidence-based ensemble justification; new ensemble table added to section 5.

### Phase 14 -- Paper Finalization + Presentation

Terminal phase; consumes results from all prior phases.

- [ ] **POL-01**: `SciencePlots` IEEE styling applied to every figure via `plt.style.use(['science', 'ieee', 'no-latex'])`. Falls back to plain matplotlib if Alvin's machine lacks pip.
- [ ] **POL-02**: Colorblind-safe palette across all plots (`seaborn.set_theme(palette='colorblind')` or SciencePlots default). Variable line styles and markers for B&W print readability.
- [ ] **POL-03**: All figures saved at 300 DPI.
- [ ] **POL-04**: Abstract trimmed to <= 250 words (current draft: ~412 words).
- [ ] **POL-05**: Citation format consistent throughout (numbered bracket style, alphabetical by first author in references).
- [ ] **POL-06**: Every figure referenced in text with caption; axis labels carry units.
- [ ] **POL-07**: Headline Sharpe in abstract uses per-pair-corrected number (approximately 3.2), not per-trade (0.44). Per-trade Sharpe in footnote/Table 8 only. Guards P5.
- [ ] **POL-08**: Survivorship-bias disclaimer appears in section 6.4 Limitations. Scaling-curve cap annotation appears on Figure 2 caption. Guards P7.
- [ ] **POL-09**: AI-assistant disclosure in Acknowledgments: "We used Anthropic Claude (Sonnet 4.5 and Opus 4.6) as a pair-programming assistant; all design decisions and empirical interpretations are our own."
- [ ] **POL-10**: Final PDF reviewed cover-to-cover; all TODOs / placeholders cleared.
- [ ] **POL-11**: Code README updated with exact reproduction commands for every paper table.
- [ ] **POL-12**: 4-minute lightning-talk slides covering Team / Problem / Methods / Challenge / Results / Conclusions. Delivered before Apr 28.

### v1.1 Non-Goals (Explicit Out of Scope)

| Item | Reason |
|---|---|
| TFT in live deployment | v1.1 adds research evidence only; live stays hardcoded LR+XGB |
| PatchTST, Autoformer, TimesNet | Beyond scope; mentioned in section 7 Future Work as candidates |
| FRTB P&L attribution decomposition | Overkill for academic paper; simple tracking error suffices |
| Stacking regression meta-model | Not justified at 6,802 rows; simple ensembles only |
| Optuna / Ray Tune for HPO | XGBoost swept manually; TFT pre-specified; no new HPO |
| LOFO over 59 individual features | Too noisy at N=6,802; LOGO over 5 groups only |
| Real-money live trading | Paper trading through submission; real-money = future work |
| Ensemble class wired into live `strategy.py` | Risk of breaking live system mid-v1.1; deferred |
| LaTeX-based figures | `SciencePlots no-latex` variant used to avoid MacTeX install |

### Traceability -- v1.1

| Requirement | Phase | Status |
|---|---|---|
| ENV-01 | Phase 8 | Complete |
| ENV-02 | Phase 8 | Complete |
| ENV-03 | Phase 8 | Complete |
| ENV-04 | Phase 8 | Complete |
| ENV-05 | Phase 8 | Complete |
| RECON-01 | Phase 9 | Complete |
| RECON-02 | Phase 9 | Complete |
| RECON-03 | Phase 9 | Complete |
| RECON-04 | Phase 9 | Complete |
| RECON-05 | Phase 9 | Complete |
| RECON-06 | Phase 9 | Complete |
| RECON-07 | Phase 9 | Complete |
| RECON-08 | Phase 9 | Complete |
| RECON-09 | Phase 9 | Complete |
| RECON-10 | Phase 9 | Complete |
| SCAL-01 | Phase 10 | Complete |
| SCAL-02 | Phase 10 | Complete |
| SCAL-03 | Phase 10 | Complete |
| SCAL-04 | Phase 10 | Complete |
| SCAL-05 | Phase 10 | Complete |
| TFT-01 | Phase 11 | Complete |
| TFT-02 | Phase 11 | Complete |
| TFT-03 | Phase 11 | Complete |
| TFT-04 | Phase 11 | Complete |
| TFT-05 | Phase 11 | Pending |
| TFT-06 | Phase 11 | Complete |
| TFT-07 | Phase 11 | Pending |
| TFT-08 | Phase 11 | Pending |
| ABLA-01 | Phase 12 | Pending |
| ABLA-02 | Phase 12 | Pending |
| ABLA-03 | Phase 12 | Pending |
| ABLA-04 | Phase 12 | Pending |
| ABLA-05 | Phase 12 | Pending |
| ABLA-06 | Phase 12 | Pending |
| ABLA-07 | Phase 12 | Pending |
| ABLA-08 | Phase 12 | Pending |
| ENSM-01 | Phase 13 | Pending |
| ENSM-02 | Phase 13 | Pending |
| ENSM-03 | Phase 13 | Pending |
| ENSM-04 | Phase 13 | Pending |
| ENSM-05 | Phase 13 | Pending |
| ENSM-06 | Phase 13 | Pending |
| ENSM-07 | Phase 13 | Pending |
| POL-01 | Phase 14 | Pending |
| POL-02 | Phase 14 | Pending |
| POL-03 | Phase 14 | Pending |
| POL-04 | Phase 14 | Pending |
| POL-05 | Phase 14 | Pending |
| POL-06 | Phase 14 | Pending |
| POL-07 | Phase 14 | Pending |
| POL-08 | Phase 14 | Pending |
| POL-09 | Phase 14 | Pending |
| POL-10 | Phase 14 | Pending |
| POL-11 | Phase 14 | Pending |
| POL-12 | Phase 14 | Pending |

**Coverage (v1.1):**
- Requirements: 55 total (5 + 10 + 5 + 8 + 8 + 7 + 12)
- Mapped to phases: 55
- Unmapped: 0

---
*v1.1 requirements defined: 2026-04-17*
*v1.1 traceability expanded: 2026-04-17 (roadmap creation)*
*Last updated: 2026-04-17*
