# Roadmap: Kalshi vs. Polymarket Price Discrepancies

## Milestones

- **v1.0 Core Pipeline** - Phases 1-7.3 (shipped 2026-04-08)
- **v1.1 Extended Evidence & Submission** - Phases 8-14 (in progress, deadline 2026-04-27)

## Phases

<details>
<summary>v1.0 Core Pipeline (Phases 1-7.3) - SHIPPED 2026-04-08</summary>

### Phase 1: Data Ingestion
**Goal**: Raw historical market data from both platforms is reliably available on disk for downstream processing
**Depends on**: Nothing (first phase)
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06
**Success Criteria** (what must be TRUE):
  1. Running the Kalshi ingestion script produces parquet files in `data/raw/` containing OHLCV candlestick data, correctly handling the live/historical endpoint split
  2. Running the Polymarket ingestion script produces parquet files in `data/raw/` containing reconstructed price histories from trade records, with metadata from Gamma API
  3. Re-running either script skips already-cached data and respects rate limits without manual intervention
  4. Both adapters recover from transient API failures via automated retry with backoff
**Plans**: 3/3 complete

### Phase 2: Market Matching
**Goal**: A verified registry of matched contract pairs across Kalshi and Polymarket with confidence scores and settlement documentation
**Depends on**: Phase 1
**Requirements**: MATCH-01, MATCH-02, MATCH-03, MATCH-04, MATCH-05
**Plans**: Complete

### Phase 02.1: Trade-Based Data Reconstruction (INSERTED)
**Goal:** Raw trades pulled from both platforms, reconstructed into 4-hour OHLCV+VWAP candles with microstructure features, and aligned cross-platform with forward-fill and staleness decay.
**Depends on:** Phase 2
**Plans:** 2/2 complete

### Phase 3: Feature Engineering
**Goal**: A processed, model-ready dataset with derived microstructure features computed from 4-hour aligned data, temporally split into train/test sets with PyTorch Forecasting compatibility
**Depends on**: Phase 2.1
**Requirements**: FEAT-01, FEAT-02, FEAT-03, FEAT-04, FEAT-05
**Plans**: 1/1 complete

### Phase 4: Regression Baselines and Evaluation Framework
**Goal**: Tier 1 models are trained and evaluated, the evaluation framework exists for all future models, and the TA check-in deliverable is ready
**Depends on**: Phase 3
**Requirements**: MOD-01, MOD-02, MOD-03, MOD-04, EVAL-01, EVAL-02
**Plans**: 2/2 complete

### Phase 5: Time Series Models
**Goal**: Recurrent and attention-based models are trained and evaluated, testing whether temporal structure improves spread prediction
**Depends on**: Phase 4
**Requirements**: MOD-05, MOD-06, MOD-07
**Plans**: 5/5 complete

### Phase 6: RL and Autoencoder
**Goal**: PPO trading agents and the autoencoder anomaly detector are trained, testing whether RL and anomaly detection improve trading performance
**Depends on**: Phase 4
**Requirements**: MOD-08, MOD-09, MOD-10
**Plans**: 5/5 complete

### Phase 7: Experiments and Interpretability
**Goal**: The three planned experiments are executed, SHAP analysis reveals feature importance, and all results have statistical rigor via confidence intervals
**Depends on**: Phase 5, Phase 6
**Requirements**: EXP-01, EXP-02, EXP-03, EXP-04, EVAL-03, EVAL-04
**Plans**: 5/5 complete

### Phase 07.1: Walk-Forward Backtesting (INSERTED)
**Goal:** Produce honest, paper-ready backtested Sharpe ratios for all 8 models via a walk-forward portfolio simulator with realistic transaction costs
**Depends on:** Phase 7
**Plans:** 2/2 complete

### Phase 07.2: Live Paper Trading and Data Collection (INSERTED)
**Goal:** Deploy a live paper trading system that collects data from Kalshi+Polymarket, runs all 8 models on live bars, logs paper trades, and supports periodic retraining
**Depends on:** Phase 7.1
**Plans:** 3/3 complete

### Phase 07.3: Adaptive Trading System (INSERTED)
**Goal:** Build an adaptive trading system with multi-bar position management, contract-horizon-based bar intervals and rule-based exit strategy, deployed on Oracle Cloud VM
**Depends on:** Phase 7.2
**Plans:** 4/4 complete

</details>

---

## v1.1 Extended Evidence & Submission

**Milestone Goal:** Strengthen every pillar of the paper's evidence base -- scaling, model variety, feature understanding, execution realism -- while the live system continues accumulating data. Produce a submission-ready paper (due Apr 27) that credibly signals continued deployment post-submission.

**Deadline:** 2026-04-27

- [ ] **Phase 8: Environment & Baseline Verification** - Clean venv, seed discipline, reproduce all Table 2 numbers identically
- [ ] **Phase 9: Live vs Backtest Reconciliation** - Trade-level comparison of live paper-trading P&L against backtest predictions
- [ ] **Phase 10: 250-Bar Scaling Checkpoint** - Third scale point in Table 5, test ranking invariance across 5x data growth
- [ ] **Phase 11: TFT Training** - Temporal Fusion Transformer on 6,802 rows with pre-specified small-data hyperparameters
- [ ] **Phase 12: Feature Ablation** - LOGO over 5 feature groups on LR + XGBoost with pre-registered protocol
- [ ] **Phase 13: Ensemble Formalization** - Evidence-based production ensemble with concordance filter audit
- [ ] **Phase 14: Paper Finalization + Presentation** - IEEE-styled figures, trimmed abstract, final PDF, lightning talk slides

## Phase Details

### Phase 8: Environment & Baseline Verification
**Goal**: Every v1.1 experiment runs on a reproducible, verified foundation where Table 2 numbers are confirmed and training is deterministic
**Depends on**: Nothing (gating phase for v1.1; blocks all downstream phases)
**Requirements**: ENV-01, ENV-02, ENV-03, ENV-04, ENV-05
**Success Criteria** (what must be TRUE):
  1. `pytorch-forecasting==1.7.0`, `quantstats==0.0.81`, and `SciencePlots==2.2.1` are installed and importable in `.venv/` (Python 3.12 rebuild if 3.14 wheels unavailable)
  2. A shared `src/utils/seed.py` is called at the top of every training script and running `experiments/verify_headline.py` twice produces identical Table 2 numbers within 1% tolerance (guards P6)
  3. All PAPER_DRAFT.md Table 2 numbers reconcile against the `experiments/results/tier1/*.json` files currently on disk (no unexplained divergence between paper and code)
**Plans**: TBD

Plans:
- [ ] 08-01: TBD
- [ ] 08-02: TBD

### Phase 9: Live vs Backtest Reconciliation
**Goal**: A trade-level comparison proves that the live paper-trading system and the backtest simulator agree on P&L within documented tolerances, giving the paper unique live-deployment evidence
**Depends on**: Phase 8
**Requirements**: RECON-01, RECON-02, RECON-03, RECON-04, RECON-05, RECON-06, RECON-07, RECON-08, RECON-09, RECON-10
**Success Criteria** (what must be TRUE):
  1. `src/analysis/reconciliation.py` pairs live closed positions (April 11-25) against backtest predictions on `(pair_id, entry_ts_bucket)` using the single shared fee function from `src/evaluation/profit_sim.simulate_profit` (guards P2)
  2. Summary comparison table shows live P&L vs simulated P&L, tracking error, matched/only-live/only-backtest counts, and `(only_live + only_backtest) / matched_trades < 20%` acceptance gate passes (or gap is diagnosed and named in paper)
  3. Category-level (oil vs non-oil) and exit-reason attribution breakdowns are produced, directly testing Finding 6 on live data
  4. Paper section 5.9 "Live vs Backtest Reconciliation" is written with findings and explicit paper-trading caveats (no slippage, no partial fills)
**Plans**: TBD

Plans:
- [ ] 09-01: TBD
- [ ] 09-02: TBD

### Phase 10: 250-Bar Scaling Checkpoint
**Goal**: The third scale point (250 bars/pair) fills Table 5 and either confirms ranking invariance across 5x data growth or documents a ranking shift
**Depends on**: Phase 8
**Requirements**: SCAL-01, SCAL-02, SCAL-03, SCAL-04, SCAL-05
**Success Criteria** (what must be TRUE):
  1. 250-bar auto-retrain checkpoint output is captured from SCC and Table 5 in the paper contains all three scale points (50 / 100 / 250 bars/pair)
  2. Figure 2 is regenerated with explicit training-set-cap annotation ("plateau at N=6,802, fixed pair universe") so the scaling curve is not misread as universal evidence (guards P7)
  3. Paper section 5.4 explicitly states whether model rankings are invariant across the 5x data growth or documents any ranking shift
**Plans**: TBD

Plans:
- [ ] 10-01: TBD

### Phase 11: TFT Training
**Goal**: The deferred Temporal Fusion Transformer is trained with pre-specified small-data hyperparameters and produces either a competitive result or a documented negative finding that extends the simplicity-wins thesis
**Depends on**: Phase 8
**Requirements**: TFT-01, TFT-02, TFT-03, TFT-04, TFT-05, TFT-06, TFT-07, TFT-08
**Success Criteria** (what must be TRUE):
  1. `src/models/tft.py` implements `TFTPredictor(BasePredictor)` with pre-specified hyperparameters (`hidden_size=8`, `attention_head_size=1`, `dropout=0.3`, `QuantileLoss`, `GroupNormalizer`) and is evaluated on the identical protocol to GRU/LSTM (guards P1)
  2. If TFT converges (val_loss beats GRU within 1-day time-box): TFT row appears in Tables 2 and 3, attention entropy audit passes (`entropy >= 0.5 * log(n_features)` and `max_variable_weight < 0.8`), and VSN heatmap is saved to `experiments/figures/`
  3. If TFT does not converge: "TFT did not converge at N=6,802" is reported as a finding under Discussion section 6, and Phase 13 proceeds with TFT-excluding baseline
  4. Either outcome (success or documented negative result) produces a paper-ready finding that extends the complexity-vs-performance analysis
**Plans**: TBD

Plans:
- [ ] 11-01: TBD
- [ ] 11-02: TBD

### Phase 12: Feature Ablation
**Goal**: A pre-registered LOGO ablation study identifies the minimum sufficient feature set for profitable trading and quantifies which feature groups are load-bearing vs droppable
**Depends on**: Phase 8
**Requirements**: ABLA-01, ABLA-02, ABLA-03, ABLA-04, ABLA-05, ABLA-06, ABLA-07, ABLA-08
**Success Criteria** (what must be TRUE):
  1. `.planning/ablation_protocol.md` is committed before `run_feature_ablation.py` executes, pre-registering the 5 feature groups, three-way split design, and analysis plan (guards P3)
  2. LOGO ablation runs on both LR and XGBoost separately with bootstrap 95% CIs on per-group P&L deltas, and the results table reports ALL runs (not only favorable ones)
  3. Minimum sufficient feature set is selected on ablation-holdout split only; final-test split is untouched until after selection is frozen
  4. Paper section 5.X "Feature Ablation" contains the ablation table and parsimony discussion
**Plans**: TBD

Plans:
- [ ] 12-01: TBD
- [ ] 12-02: TBD

### Phase 13: Ensemble Formalization
**Goal**: The live deployment's LR+XGBoost ensemble is formally evaluated alongside alternative ensemble variants, with an honest concordance filter audit that surfaces any Sharpe inflation
**Depends on**: Phase 11 (TFT outcome determines whether 5th ensemble variant is included)
**Requirements**: ENSM-01, ENSM-02, ENSM-03, ENSM-04, ENSM-05, ENSM-06, ENSM-07
**Success Criteria** (what must be TRUE):
  1. `src/models/ensemble.py` implements `EnsemblePredictor(BasePredictor)` and evaluates 4+ ensemble variants (LR alone, LR+XGB equal-weight, LR+LSTM, majority-vote; plus TFT variant if Phase 11 succeeded)
  2. Concordance filter audit shows BOTH filtered and unfiltered P&L in the same table, includes rejection rate and P&L on rejected trades, and flags if rejected trades are profitable in aggregate (guards P4)
  3. Ensemble-weight sensitivity sweep (LR-weight 0.0 to 1.0 in 0.1 increments) produces a single plot proving the weight choice is not cherry-picked
  4. `EnsemblePredictor` is NOT wired into `src/live/strategy.py` during v1.1 (live deployment stays hardcoded to current LR+XGB average; guards "breaking live system" risk)
**Plans**: TBD

Plans:
- [ ] 13-01: TBD
- [ ] 13-02: TBD

### Phase 14: Paper Finalization + Presentation
**Goal**: The paper and slides are submission-ready with IEEE-styled figures, corrected Sharpe numbers, all TODOs cleared, and explicit limitations documented
**Depends on**: Phase 13 (and transitively all prior phases)
**Requirements**: POL-01, POL-02, POL-03, POL-04, POL-05, POL-06, POL-07, POL-08, POL-09, POL-10, POL-11, POL-12
**Success Criteria** (what must be TRUE):
  1. Every figure uses `SciencePlots` IEEE styling with colorblind-safe palette, variable line styles/markers for B&W readability, and 300 DPI export; every figure is referenced in text with caption and axis labels carrying units
  2. Abstract is 250 words or fewer; headline Sharpe in abstract uses per-pair-corrected number (approximately 3.2), not per-trade (0.44); per-trade Sharpe appears only in footnote/Table 8 (guards P5)
  3. Survivorship-bias disclaimer appears in section 6.4 Limitations; scaling-curve cap annotation appears on Figure 2 caption (guards P7); AI-assistant disclosure in Acknowledgments
  4. Final PDF reviewed cover-to-cover with all TODOs/placeholders cleared, code README updated with exact reproduction commands for every paper table, and 4-minute lightning-talk slides completed
**Plans**: TBD

Plans:
- [ ] 14-01: TBD
- [ ] 14-02: TBD
- [ ] 14-03: TBD

## Progress

**v1.1 Execution Order:**

Phase 8 gates everything. After Phase 8, four phases run in parallel. Phase 13 waits for Phase 11. Phase 14 waits for all.

```
          +---> Phase 9  (reconciliation, leaf)
          |
Phase 8 --+---> Phase 10 (250-bar wait, passive)
          |                                         +---> Phase 14 (paper, terminal)
          +---> Phase 11 (TFT) ----------> Phase 13 (ensemble) ---+
          |
          +---> Phase 12 (ablation, independent)
```

**Milestone Deadline:**
- Final Submission (April 27): All v1.1 phases complete

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 8. Environment & Baseline Verification | v1.1 | 0/0 | Not started | - |
| 9. Live vs Backtest Reconciliation | v1.1 | 0/0 | Not started | - |
| 10. 250-Bar Scaling Checkpoint | v1.1 | 0/0 | Not started | - |
| 11. TFT Training | v1.1 | 0/0 | Not started | - |
| 12. Feature Ablation | v1.1 | 0/0 | Not started | - |
| 13. Ensemble Formalization | v1.1 | 0/0 | Not started | - |
| 14. Paper Finalization + Presentation | v1.1 | 0/0 | Not started | - |

---
*Roadmap created: 2026-04-01*
*v1.0 shipped: 2026-04-08*
*v1.1 roadmap created: 2026-04-17*
*Last updated: 2026-04-17*
