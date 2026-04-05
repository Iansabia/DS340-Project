---
phase: 5
slug: time-series-models
type: research
created: 2026-04-05
sources: 4 parallel research agents + direct data inspection
---

# Phase 5 Research: Time Series Models (GRU / LSTM / TFT)

## Question Answered
**What is the best way to execute Phase 5 given the expanded 144-pair dataset?**

---

## Data Reality (verified from `data/processed/`)

| Fact | Value |
|------|-------|
| Train parquet | 6,946 rows / 144 pairs / 39 columns |
| Test parquet | 1,817 rows / 144 pairs / 39 columns |
| **Bar interval** | **4 hours** (NOT hourly — verified from timestamps) |
| Train sequence length per pair | min=5, p25=20, median=32, mean=48.2, p90=118, max=141 bars |
| Test sequence length per pair | min=2, p25=6, median=8, mean=12.6, p90=30, max=36 bars |
| Train/test gap per pair (median) | 4 hours (1 bar — continuous split) |
| Target column | `spread` (already present); **derived target** is `spread.shift(-1) - spread` per pair |
| `time_idx`, `group_id` | Already populated, reset to 0 per split |
| PyTorch Forecasting compatibility | TimeSeriesDataSet-ready |

### NaN audit (critical finding)

| Column | NaN % | Action |
|--------|-------|--------|
| `kalshi_order_flow_imbalance` | **100%** (all rows NaN) | **DROP** — bug upstream in Phase 3 feature engineering (denominator=0). Out of scope for Phase 5. |
| `price_velocity` | 2.1% (1 per pair, first row) | Forward-fill within pair, then 0.0 |
| `spread_volatility` | 2.1% (1 per pair, first row) | Forward-fill within pair, then 0.0 |
| `polymarket_order_flow_imbalance` | 0% | Use as-is (contrary to one agent's claim) |
| All others | 0% | Use as-is |

**Usable features: 33** (39 total − 5 non-features [`timestamp`, `pair_id`, `time_idx`, `group_id`, `spread`] − 1 all-NaN [`kalshi_order_flow_imbalance`])

### Training-window counts by lookback (derived target: horizon=1)

| Lookback (bars) | Lookback (hours) | Windows | Pairs kept (train) | Pairs kept (test) |
|-----------------|------------------|---------|-------------------|-------------------|
| 4  | 16h  | 6370 | 144 | 111 |
| **6**  | **24h**  | **6085** | **140** | **89** |
| 8  | 32h  | 5805 | 136 | 62 |
| 12 | 48h  | 5263 | 126 | 33 |
| 24 | 96h  | 3906 | 98  | 3  |

---

## Locked Decisions (from research synthesis + data facts)

### D1. Lookback length: **6 bars (24 hours)**
- Captures a full day-cycle on 4-hour bars
- Keeps 140/144 pairs (97%) in training
- Yields 6,085 training windows — best windows-per-pair-kept tradeoff
- Agents disagreed (one said 12h=3 bars, another said 12 bars), but they assumed hourly data. Reconciled to 6 bars = 24h after bar-interval verification.

### D2. Warm-up context for test predictions: **stitch train tail + test by timestamp per pair**
- Train and test are temporally continuous (1-bar gap per pair)
- For each test row at time_idx=t (test-local), assemble lookback from concatenated (train ∪ test) sorted by timestamp
- Avoids wasting the first `lookback-1` test rows per pair
- Enables one prediction per test row (keeps row-alignment with Tier 1 harness)

### D3. Prediction horizon: **1 bar (next 4-hour spread change)**
- Matches Phase 4 target: `spread_change = spread.shift(-1) - spread` grouped by pair_id
- Identical regression target to XGBoost/LinearRegression → direct comparability
- Multi-step deferred (no domain justification)

### D4. Train/val split: **90/10 within-pair chronological, from train.parquet only**
- Last 10% of each pair's bars (by timestamp) becomes validation
- Preserves temporal ordering, no cross-pair leakage
- 90% = ~6250 rows = ~5475 windows for training
- 10% = ~695 rows = ~610 windows for validation + early stopping

### D5. Feature scaling: **StandardScaler on all 33 features, fit on train, apply to val+test**
- Target `spread_change` kept in raw units (not scaled) — P&L simulation expects raw spreads
- Bool columns (`kalshi_has_trade`, `polymarket_has_trade`) cast to float {0.0, 1.0} BEFORE scaling
- Scaler saved alongside model for inference reproducibility

### D6. GRU architecture
| Hyperparameter | Value |
|----------------|-------|
| hidden_size | 64 |
| num_layers | 1 |
| input_dropout | 0.3 (on input features, not recurrent) |
| bidirectional | **False** (no peeking into future within window) |
| output head | Linear(64 → 1) |
| input features | 33 (pre-scaled) |

### D7. LSTM architecture
| Hyperparameter | Value |
|----------------|-------|
| hidden_size | 32 (smaller — LSTM has ~40% more params per unit than GRU) |
| num_layers | 1 |
| input_dropout | 0.3 |
| bidirectional | False |
| output head | Linear(32 → 1) |
| input features | 33 (pre-scaled) |

### D8. Training protocol (GRU and LSTM identical)
| Hyperparameter | Value |
|----------------|-------|
| loss | MSE (target is light-tailed: skew=0.17, kurt=-0.21) |
| optimizer | AdamW |
| learning rate | 1e-3 |
| weight decay (L2) | 1e-4 |
| batch size | 64 |
| max epochs | 100 |
| early stopping patience | 10 (on val MSE) |
| early stopping min_delta | 1e-4 |
| LR scheduler | ReduceLROnPlateau, factor=0.5, patience=5 |
| gradient clip | max_norm=1.0 |
| seeds (reported) | {42, 123, 456} — report mean ± std |
| device | auto (CUDA if available, else CPU) |

### D9. TFT: **DEFER** with documented rationale
Per roadmap's explicit deferral clause (Phase 5 success criterion #3). Rationale for VERIFICATION.md:

> TFT is deferred from Phase 5 per the roadmap's explicit deferral clause. At hidden_size=16, TFT has ~10k parameters against ~5–6k training windows (param-to-sample ratio ≈ 1.9, well above the 0.01 safety threshold). PyTorch Forecasting's canonical examples use datasets 200×+ larger. Transformers overfit pathologically on datasets this small, and GRU/LSTM provide equivalent Tier 2 representational coverage with proven stability. Deferring TFT preserves the remaining 22-day timeline for Phase 6 (RL) and Phase 7 (experiments/interpretability), which carry higher research-question weight. TFT is re-examined post-deadline if dataset expansion enables a ≥20k-window training set.

### D10. Evaluation integration
- Models inherit `src.models.base.BasePredictor`
- Expose flat `fit(X: pd.DataFrame, y: np.ndarray)` and `predict(X: pd.DataFrame) → np.ndarray` interface
- Windowing is **internal** to each model (cache train data during fit, build windows during predict)
- Results saved to `experiments/results/tier2/gru.json` and `experiments/results/tier2/lstm.json`
- Exact JSON schema matches Tier 1 (`model_name`, `metrics`, `timestamp`, `threshold`, `n_train_rows`, `n_test_rows`, `n_features`, `pnl_series`)
- Extended CLI in `experiments/run_baselines.py`: `--tier {1,2,both}` flag
- `format_comparison_table()` auto-picks up Tier 2 JSONs — same single table as roadmap success criterion #4

---

## Validation Architecture

**Reference:** `references/tdd.md`. Tests mirror `tests/models/` pattern from Phase 4.

### Dimension 1: Unit — windowing utility
- **test_create_sequences_shape**: 30-row input, lookback=6 → (24, 6, n_features) windows
- **test_create_sequences_respects_group_boundaries**: 2 pairs × 20 rows, lookback=6 → no window crosses pair boundary
- **test_create_sequences_too_short_pair_skipped**: pair with 3 rows < lookback=6 → 0 windows from that pair

### Dimension 2: Unit — model interfaces
- **test_gru_inherits_base_predictor / test_lstm_inherits_base_predictor**
- **test_predict_before_fit_raises**: `RuntimeError`
- **test_fit_returns_self**: chainability
- **test_predict_returns_1d_flat_array**: shape == (len(X),)
- **test_name_property**: stable strings "GRU" / "LSTM"

### Dimension 3: Reproducibility
- **test_seed_reproducibility**: two fits with same seed → identical predictions (atol=1e-5)

### Dimension 4: Integration — evaluate() produces Tier-1-compatible metrics
- **test_evaluate_returns_expected_keys**: {rmse, mae, directional_accuracy, total_pnl, num_trades, win_rate, sharpe_ratio, sharpe_per_trade, pnl_series}
- **test_evaluate_on_synthetic_data**: 200 rows × 10 features → evaluate runs without error

### Dimension 5: End-to-end integration
- **test_tier2_harness_writes_json**: running harness produces `experiments/results/tier2/gru.json` and `lstm.json` with valid schema
- **test_comparison_table_includes_tier2**: `format_comparison_table(tier="both")` includes GRU and LSTM rows

### Dimension 6: Regression against Tier 1 baseline
- **test_sanity_gru_beats_naive**: on held-out test set, GRU RMSE < NaivePredictor RMSE (sanity only, no strong claim against XGBoost)

### Dimension 7: Deferral documentation
- **test_tft_deferral_documented**: VERIFICATION.md contains the deferral rationale from D9

### Dimension 8: Nyquist sampling (per gsd-nyquist-auditor)
- Every unit test covers a single behavior; integration tests hit end-to-end paths

---

## Integration Contract (from BasePredictor + run_baselines.py)

**Required methods on each new model class:**

```python
class GRUPredictor(BasePredictor):
    def __init__(self, hidden_size=64, num_layers=1, dropout=0.3,
                 learning_rate=1e-3, weight_decay=1e-4, batch_size=64,
                 max_epochs=100, patience=10, lookback=6, random_state=42): ...

    @property
    def name(self) -> str: return "GRU"

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> "GRUPredictor": ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
    # must return shape (len(X),) — 1 prediction per input row
```

**Inherits** `evaluate(X_test, y_test, threshold=0.02)` from `BasePredictor` — no override needed.

---

## Realistic Performance Expectation (honest)

**GRU/LSTM are expected to be competitive with, but likely NOT beat, XGBoost on this dataset size.**

Reasoning:
- XGBoost param ratio: ~0.001 samples/param (trees are efficient in small-N regimes)
- GRU @ hidden=64: ~0.3 samples/param (borderline-overparameterized)
- Temporal structure may be weak: 33 engineered features already encode most of the signal
- 144-group heterogeneity punishes sequence models harder than trees

Expected RMSE band:
- XGBoost (Tier 1 baseline): 0.286
- GRU realistic: 0.29–0.33
- LSTM realistic: 0.30–0.34

**This is a VALID and PUBLISHABLE finding** for the project's central research question: *"Does increasing model complexity improve arbitrage detection over simpler regression approaches?"* For this dataset scale, the answer is likely **no for Tier 2 alone** — supporting the broader complexity-vs-performance thesis.

---

## Top Gotchas

1. **Test sequences short (median 8 bars, 89 pairs < 12 bars):** Must warm up test windows from train tail (D2) or models will produce fewer predictions than Tier 1 has rows.
2. **`kalshi_order_flow_imbalance` 100% NaN:** Drop, don't impute. Upstream bug out of Phase 5 scope.
3. **First row per pair:** `price_velocity` + `spread_volatility` NaN. Forward-fill within pair, not globally.
4. **Target must NOT be scaled:** P&L simulation reads raw spread deltas.
5. **Bool feature casting:** Cast `kalshi_has_trade`/`polymarket_has_trade` to float BEFORE StandardScaler or scaler will error.
6. **Reproducibility:** Must seed numpy, torch, and torch.cuda (if present). Use `set_seed()` helper.
7. **CPU-only fine:** 33 features × 6-bar windows × ~6k examples trains in < 2 min on CPU per model.

---

## Files That Must Exist After Phase 5

| Path | Purpose |
|------|---------|
| `src/models/sequence_utils.py` | `create_sequences`, `EarlyStopping`, `set_seed`, `get_device`, scaler helpers |
| `src/models/gru.py` | `GRUPredictor(BasePredictor)` |
| `src/models/lstm.py` | `LSTMPredictor(BasePredictor)` |
| `src/models/__init__.py` (updated) | exports GRUPredictor, LSTMPredictor |
| `experiments/run_baselines.py` (updated) | `--tier` CLI flag; writes to `tier1/` OR `tier2/` |
| `experiments/results/tier2/gru.json` | GRU eval results (Tier-1-compatible schema) |
| `experiments/results/tier2/lstm.json` | LSTM eval results |
| `tests/models/test_sequence_utils.py` | Windowing + helpers |
| `tests/models/test_gru.py` | GRU interface + integration |
| `tests/models/test_lstm.py` | LSTM interface + integration |
| `.planning/phases/05-time-series-models/05-VERIFICATION.md` | Final verification (includes TFT deferral rationale) |

---

*Phase 5 research complete. Decisions locked. Ready for planning.*
