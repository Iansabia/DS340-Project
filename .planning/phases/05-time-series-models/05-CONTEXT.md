---
phase: 5
slug: time-series-models
type: context
created: 2026-04-05
source: 4 parallel research agents + direct data inspection; user-approved via research-first flow
status: Ready for planning
---

# Phase 5: Time Series Models — Context

**Central question for this phase:** Does temporal structure improve spread prediction over the flat-tabular Tier 1 baselines?

<domain>
## Phase Boundary

**In scope:**
- GRU model (Tier 2 recurrent baseline)
- LSTM model (Tier 2 recurrent alternative)
- TFT *deferral documentation* with roadmap-compliant rationale
- Shared sequence-model utilities (windowing, early stopping, seeding, scaling)
- Tier 2 evaluation harness that writes results in the Tier-1-compatible JSON schema
- Integration into the single cross-tier comparison table

**Out of scope (deferred to later phases):**
- TFT *implementation* (see D9 below — explicit roadmap deferral)
- RL models (Phase 6)
- Cross-tier experiment framework, SHAP, bootstrap CIs (Phase 7)
- Upstream data bug: `kalshi_order_flow_imbalance` 100% NaN (document and drop, do not fix)
- Hyperparameter sweeps beyond the 3-seed reported runs
</domain>

<decisions>
## Implementation Decisions (LOCKED from research)

### Data shape facts
- Bars are **4-hour bars** (not hourly — verified from timestamp deltas)
- 144 matched pairs, 6,946 train rows, 1,817 test rows, 39 columns total
- Train sequence length per pair: min=5, median=32, mean=48.2, max=141 (bars)
- Test sequence length per pair: min=2, median=8, mean=12.6, max=36 (bars)
- Train/test per-pair gap = 1 bar (4 hours) → temporally continuous
- `time_idx` and `group_id` already present and reset per split

### Features and target
- **Target:** `spread_change = spread.shift(-1) - spread` computed per pair_id (identical to Phase 4)
- **Usable features:** 33 (all numeric columns MINUS `timestamp`, `pair_id`, `time_idx`, `group_id`, `spread`, `kalshi_order_flow_imbalance`)
- **Drop `kalshi_order_flow_imbalance`** — 100% NaN (all 6946+1817 rows). Upstream bug, out of scope.
- **Forward-fill within pair (then 0.0)** for `price_velocity` and `spread_volatility` first-row NaN (1 per pair)
- **Bool → float {0.0, 1.0}** for `kalshi_has_trade` and `polymarket_has_trade` BEFORE scaling
- **StandardScaler** fit on train features ONLY; apply to val + test
- **Do NOT scale the target** (P&L sim needs raw spread units)

### Sequence construction
- **Lookback: 6 bars (24 hours)** — captures full day-cycle, keeps 140/144 train pairs, 6,085 training windows
- **Stride: 1** (overlapping sliding windows)
- **Prediction horizon: 1 bar** (next 4-hour spread change)
- **Windows MUST NOT cross pair_id boundaries**
- **Test warm-up strategy:** for each test row, assemble lookback from concatenated (train ∪ test) sorted by timestamp per pair — so test can produce one prediction per row (row-aligned with Tier 1)

### Train/val split
- **90/10 within-pair chronological** split of train.parquet
- Last 10% of each pair's bars (by timestamp) → validation
- Used for early stopping ONLY — not for hyperparameter tuning
- Test.parquet untouched (held out exactly as Tier 1)

### GRU architecture
- hidden_size = **64**
- num_layers = **1**
- input_dropout = **0.3** (applied to input features, not recurrent)
- bidirectional = **False**
- output = Linear(64 → 1), no activation (raw regression)

### LSTM architecture
- hidden_size = **32** (smaller because LSTM has ~40% more params than GRU at same hidden size)
- num_layers = **1**
- input_dropout = **0.3**
- bidirectional = **False**
- output = Linear(32 → 1)

### Training protocol (same for GRU and LSTM)
- loss = **MSE** (target is light-tailed: skew=0.17, kurt=-0.21; MSE preferred over Huber)
- optimizer = **AdamW**, lr=**1e-3**, weight_decay=**1e-4**
- batch_size = **64**
- max_epochs = **100**
- early_stopping patience = **10**, min_delta = **1e-4** (on validation MSE)
- LR scheduler = **ReduceLROnPlateau**, factor=0.5, patience=5
- gradient_clip max_norm = **1.0**
- seeds reported = **{42, 123, 456}** — runs mean ± std
- device = **auto** (CUDA if available, else CPU — CPU sufficient given <6k windows, <35 features)

### BasePredictor contract
- Inherit from `src.models.base.BasePredictor`
- Implement: `name` property, `fit(X_train, y_train) → self`, `predict(X) → 1-D ndarray`
- Inherit `evaluate()` from base (no override needed)
- Windowing is **INTERNAL** to the model class (cache train during fit; build windows during predict)
- `predict(X)` MUST return one prediction per input row (shape `(len(X),)`) — use warm-up stitching to satisfy this

### TFT
- **DEFERRED** per roadmap success criterion #3's explicit deferral clause
- Written rationale (for VERIFICATION.md):
  > TFT is deferred from Phase 5 per the roadmap's explicit deferral clause. At hidden_size=16, TFT has ~10k parameters against ~5–6k training windows (param-to-sample ratio ≈ 1.9, far above the 0.01 safety threshold). PyTorch Forecasting's canonical examples use datasets 200×+ larger. Transformers overfit pathologically at this data scale, while GRU/LSTM provide equivalent Tier 2 representational coverage with proven stability. Deferring TFT preserves the 22-day runway for Phase 6 (RL) and Phase 7 (experiments), which carry higher research-question weight. TFT is re-examined post-deadline if dataset expansion yields ≥20k windows.

### Evaluation + comparison table
- Results written to `experiments/results/tier2/gru.json` and `experiments/results/tier2/lstm.json`
- JSON schema identical to Tier 1 (keys: `model_name`, `metrics`, `timestamp`, `threshold`, `n_train_rows`, `n_test_rows`, `n_features`, `pnl_series`)
- Extend `experiments/run_baselines.py` with `--tier {1,2,both}` CLI flag
- `format_comparison_table()` auto-assembles Tier 1 + Tier 2 into the single comparison table (roadmap success criterion #4)
- Report seed mean ± std in the comparison table for Tier 2 models (add `seed` and `mean_rmse`/`std_rmse` to the `extra` field)

### Testing (TDD Iron Law enforced)
- Write tests FIRST (red), implement (green), refactor
- Tests live in `tests/models/test_sequence_utils.py`, `test_gru.py`, `test_lstm.py`
- Use `conftest.py` fixtures for synthetic sequence data (small N, known outputs)
- Must-pass: BasePredictor inheritance, predict-before-fit raises, shape contracts, seed reproducibility, evaluate() returns Tier-1-compatible keys

### Performance expectation (documented in SUMMARY.md)
- GRU/LSTM are expected to be **competitive with** but **likely NOT beat** XGBoost (RMSE=0.286) at this dataset size
- Realistic RMSE band: GRU 0.29–0.33, LSTM 0.30–0.34
- This underperformance, if it occurs, is a **valid and publishable finding** supporting the project's complexity-vs-performance thesis — NOT a failure

### Claude's Discretion
- Exact filenames for test fixtures
- Internal representation of the cached training data (pd.DataFrame vs torch.Tensor vs np.ndarray)
- Training-loop implementation details (manual loop vs. PyTorch Lightning — **prefer manual loop** to avoid adding a heavy dependency for 2 models)
- Print statements / progress bars during training
- Logging format (but MUST log per-epoch val_loss for reproducibility)
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project-level
- `CLAUDE.md` — model tier architecture, GRU-preferred, TFT deferral policy, TDD requirement
- `.planning/REQUIREMENTS.md` — MOD-05 (GRU), MOD-06 (LSTM), MOD-07 (TFT)
- `.planning/ROADMAP.md` — Phase 5 goal + 4 success criteria (especially #3's deferral clause and #4's comparison-table requirement)

### Research + validation
- `.planning/phases/05-time-series-models/05-RESEARCH.md` — full research synthesis (this phase)

### Phase 4 patterns to replicate
- `.planning/phases/04-regression-baselines/04-SUMMARY.md` — what was built in Phase 4
- `src/models/base.py` — BasePredictor contract (inheritance target)
- `src/models/xgboost_model.py` — concrete wrapper pattern to follow
- `src/models/linear_regression.py` — simpler wrapper pattern
- `experiments/run_baselines.py` — training/eval harness to extend (source of truth for feature extraction, target computation, data loading, results saving)
- `experiments/results/tier1/xgboost.json` — exact JSON schema to match
- `experiments/results/tier1/linear_regression.json` — schema reference
- `src/evaluation/` — metrics + profit simulation (unchanged, inherited via BasePredictor)
- `tests/models/` — test patterns to mirror (test file naming, fixture usage)

### Data
- `data/processed/train.parquet` — 6946 rows / 144 pairs
- `data/processed/test.parquet` — 1817 rows / 144 pairs
- `src/features/schemas.py` — column definitions
- `src/features/build_features.py` — understand how train/test was constructed
</canonical_refs>

<specifics>
## Specific Requirements

- **Must appear in single comparison table with Tier 1** (roadmap success criterion #4)
- Seeds `{42, 123, 456}` for Tier 2 reported results; report mean ± std
- 4-hour bars → `BARS_PER_YEAR = 2190` for annualized Sharpe (already used by Tier 1 profit sim — verify)
- TFT deferral must be written into VERIFICATION.md as documented rationale
- No new heavy dependencies (no PyTorch Lightning, no pytorch-forecasting for this phase — torch is already in deps)
</specifics>

<deferred>
## Deferred Ideas

- **TFT implementation** — post-deadline stretch goal if dataset expands
- Hyperparameter grid search (beyond 3 reported seeds)
- Bidirectional variants (legitimate ablation but saved for Phase 7 experiments)
- Multi-step-ahead prediction (k > 1)
- Augmentation (sequence jittering, vertical flip) — only if first-pass validation curves suggest underfitting
- Per-group monitoring (per-pair validation error) — Phase 7 analysis
- Fixing upstream `kalshi_order_flow_imbalance` NaN bug — separate phase / decimal gap-closure
</deferred>

---

*Phase: 05-time-series-models*
*Context gathered: 2026-04-05 via research-first workflow*
*4 parallel research agents + direct data verification*
