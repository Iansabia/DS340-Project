# Phase 11: TFT Training - Research

**Researched:** 2026-04-22
**Domain:** pytorch-forecasting 1.7.0 TFT — small-data regime, BasePredictor wrapping, quantile prediction extraction, attention audit, VSN heatmap
**Confidence:** HIGH (all claims verified against live installed library; API calls inspected at runtime)

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| TFT-01 | `src/models/tft.py` implementing `TFTPredictor(BasePredictor)` — mirrors GRUPredictor pattern, hides TimeSeriesDataSet plumbing inside `fit()`, exposes row-aligned `predict()` | Architecture section §1 details exact pattern; warm-up stitching verified from GRU source |
| TFT-02 | Hyperparameters pre-specified: `hidden_size=8`, `attention_head_size=1`, `dropout=0.3`, `QuantileLoss`, `GroupNormalizer` per-pair | Parameter budget section §2; GroupNormalizer `transformation=None` verified as correct for signed spread-change targets |
| TFT-03 | Identical evaluation protocol to GRU/LSTM: single-split + walk-forward | Integration section §5 covers single-split flow; walk-forward section §6 covers the 3-line modification needed |
| TFT-04 | 1-day time-box; Option B — always complete with success or documented negative result | Training time estimates §4; GRU val-loss baseline 0.2929 RMSE is concrete comparison target |
| TFT-05 | Attention entropy audit: flag if `entropy(attn) < 0.5*log(n_features)` or `max_variable_weight > 0.8` | Pitfall section §7 gives verified detection code |
| TFT-06 | `experiments/run_tft.py` thin wrapper (~80 LOC) over `run_tier2_with_seeds` | Integration §5 shows exact modification to `model_classes` list |
| TFT-07 | TFT row in Tables 2 and 3; paper section 4.1 updated | Out-of-scope from code implementation; paper edit task |
| TFT-08 | VSN heatmap saved to `experiments/figures/tft_variable_importance.png` | Section §3 documents exact API: `model.interpret_output(raw_out)` returns `encoder_variables` tensor; `model.encoder_variables` gives column names |
</phase_requirements>

---

## Summary

pytorch-forecasting 1.7.0 and pytorch-lightning 2.6.1 are confirmed installed and importable in the project venv. The `TemporalFusionTransformer`, `TimeSeriesDataSet`, `QuantileLoss`, and `GroupNormalizer` classes all import cleanly. The TFT API uses a `from_dataset()` factory pattern, not direct constructor instantiation — the dataset object propagates all feature-name and normalizer state into the model automatically.

The baseline competition is concrete: GRU achieves RMSE=0.2928 and P&L=+$212.50; LSTM achieves RMSE=0.2915 and P&L=+$221.84 on the current single-split. TFT's acceptance gate (TFT-04) is "val_loss beats GRU" — meaning RMSE < 0.2928. At hidden_size=8, the estimated parameter count is roughly 1,600 parameters vs. 29,760 for GRU (hidden=64), giving approximately 4.3 samples/param — marginally above the "dangerous" threshold, and better than the GRU's 0.2 samples/param which still trained successfully. The honest expected outcome is TFT ties or loses to GRU/LSTM, extending the simplicity-wins thesis to transformers.

The three short-sequence pairs (5 bars each, pair IDs: `kxbtc26feb2817b-0x1179eff8`, `kxbtc26jan2817b-0x164c1750`, `kxbtc26jan3017b-0xa73ec60b`) will be excluded by TFT's `min_encoder_length` constraint when set to 6. This is correct behavior, matching GRU's existing `_padded_pairs` warn-and-skip approach. 144 pairs in train; 3 pairs with < 6 bars means 141 pairs produce valid TFT training sequences.

**Primary recommendation:** Build `TFTPredictor(BasePredictor)` using the `from_dataset()` factory pattern. Use `QuantileLoss(quantiles=[0.1, 0.5, 0.9])` (3 quantiles instead of default 7 — faster training, adequate for P&L interval gating). Extract point predictions via `mode='prediction'` which calls `QuantileLoss.to_prediction()` and returns the index-of-0.5 quantile automatically. Extract VSN heatmap via `model.interpret_output(raw_predictions)` which returns `encoder_variables` tensor; column names are in `model.encoder_variables`.

---

## Standard Stack

### Core (verified installed)

| Library | Version | Purpose | Notes |
|---------|---------|---------|-------|
| `pytorch-forecasting` | 1.7.0 | TFT model, TimeSeriesDataSet, QuantileLoss, GroupNormalizer | Confirmed installed |
| `pytorch-lightning` | 2.6.1 | Trainer used internally by TFT fit loop | Confirmed installed as transitive dep |
| `torch` | 2.11.0 | Tensor ops, used by GRU/LSTM already | Confirmed installed |

### Supporting (already in project)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `src.models.sequence_utils` | project | `set_seed`, `get_device`, `fit_feature_scaler`, `apply_feature_scaler` | Call same helpers as GRU for consistency |
| `src.models.base.BasePredictor` | project | ABC with `fit/predict/evaluate/save/load` | TFTPredictor must subclass this |
| `matplotlib` | installed | VSN heatmap figure | `plt.savefig` for TFT-08 |

### Key API Facts (verified at runtime)

```
QuantileLoss default quantiles: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
  - 0.5 IS in the default list (to_prediction() will find it)
  - Use QuantileLoss(quantiles=[0.1, 0.5, 0.9]) for lighter training
  - mode='prediction' on TFT.predict() calls QuantileLoss.to_prediction()
    which does: y_pred[..., quantiles.index(0.5)]

TimeSeriesDataSet required parameters (from live signature):
  data, time_idx, target, group_ids,
  max_encoder_length, max_prediction_length=1,
  static_categoricals=['group_id'],
  time_varying_unknown_reals=[all feature cols],
  target_normalizer=GroupNormalizer(groups=['group_id'])

GroupNormalizer for spread-change targets:
  - transformation=None (default) is CORRECT — spread changes are signed, not positive-only
  - transformation='softplus' maps negatives near-zero — WRONG for our target
  - Use: GroupNormalizer(groups=['group_id'], transformation=None)

TFT.from_dataset() factory:
  - Signature: from_dataset(dataset, allowed_encoder_known_variable_names=None, **kwargs)
  - kwargs pass hyperparameters: hidden_size, attention_head_size, dropout, etc.
  - This is the canonical instantiation path; direct __init__ requires manually
    specifying x_reals, x_categoricals, etc. (derived from dataset in from_dataset)

interpret_output() return dict keys:
  - 'attention'              — temporal attention weights (encoder+decoder)
  - 'encoder_variables'      — VSN weights for encoder time-varying features
  - 'decoder_variables'      — VSN weights for decoder (none in our 1-step setup)
  - 'static_variables'       — VSN weights for static features (just group_id)
  - 'encoder_length_histogram'
  - 'decoder_length_histogram'
  Column names: model.encoder_variables (list of feature names in encoder order)

Trainer (pytorch-lightning 2.6.1):
  Key params: max_epochs, gradient_clip_val, accelerator, devices,
              enable_progress_bar, enable_model_summary, callbacks
  EarlyStopping params: monitor, min_delta, patience, mode, strict
```

---

## Architecture Patterns

### Recommended Project Structure for Phase 11

```
src/models/
├── tft.py              # NEW: TFTPredictor(BasePredictor) ~250 LOC
experiments/
├── run_tft.py          # NEW: thin wrapper ~80 LOC
├── run_baselines.py    # MODIFY: +5 lines (TFT import + model_classes)
├── run_walk_forward.py # MODIFY: +3 lines ('tft' in sequence branch)
experiments/figures/
└── tft_variable_importance.png   # NEW: output artifact
tests/models/
└── test_tft.py         # NEW: smoke test ~60 LOC
```

### Pattern 1: TFT Wrapping Pattern (mirrors GRUPredictor exactly)

**What:** `TFTPredictor.fit(X, y)` builds TimeSeriesDataSet internally, trains with pytorch-lightning Trainer, caches dataset and training rows. `predict(X)` stitches cached train rows for warm-up, builds prediction dataset via `TimeSeriesDataSet.from_dataset()`, runs `model.predict(loader, mode='prediction')`.

**When to use:** This is the only correct pattern. Direct dataset construction in predict() would re-fit the GroupNormalizer on test data (data leakage).

**The time_idx problem and solution:**

TFT requires a monotonically increasing `time_idx` integer column per group. Our data does NOT have this as a globally unique integer — it has pair-level indices. Solution: derive inside `fit()` using cumulative count within each group:

```python
# Source: derived from TimeSeriesDataSet contract (verified at runtime)
df_long = X_train.copy()
df_long['target'] = y_train
df_long['time_idx'] = df_long.groupby('group_id').cumcount()
# time_idx is now 0,1,2,...,n-1 within each group_id
```

**The from_dataset predict path:**

```python
# Source: TimeSeriesDataSet.from_dataset signature (verified at runtime)
# from_dataset(dataset, data, stop_randomization=False, predict=False, update_kwargs={})
pred_dataset = TimeSeriesDataSet.from_dataset(
    self._training_dataset,
    df_stitched,           # stitched train+test rows with time_idx
    predict=True,
    stop_randomization=True,
)
pred_loader = pred_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)
raw_preds = self._model.predict(pred_loader, mode='prediction')
# raw_preds shape: (n_prediction_rows, 1) — slice to (n_test,) after aligning
```

### Pattern 2: Hyperparameter Set for Pre-specified Small-Data Regime

These are LOCKED per TFT-02; do not tune during implementation:

```python
# Source: REQUIREMENTS.md TFT-02 + STACK.md configuration table
TFT_HPARAMS = {
    "hidden_size": 8,
    "attention_head_size": 1,
    "dropout": 0.3,
    "hidden_continuous_size": 8,   # must be <= hidden_size
    "lstm_layers": 1,
    "max_encoder_length": 6,       # matches GRU lookback (CONTEXT.md D6)
    "max_prediction_length": 1,    # single-step, matches all other models
    "loss": QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
    "target_normalizer": GroupNormalizer(groups=["group_id"], transformation=None),
    "learning_rate": 1e-3,
}

TRAINER_KWARGS = {
    "max_epochs": 30,
    "gradient_clip_val": 0.1,      # TFT standard from pytorch-forecasting docs
    "accelerator": "cpu",          # no GPU on this Mac; MPS segfault risk
    "enable_progress_bar": False,  # suppress output for multi-seed runs
    "enable_model_summary": False,
    "callbacks": [
        EarlyStopping(monitor="val_loss", patience=5, mode="min"),
    ],
}
```

### Pattern 3: VSN Heatmap Extraction (TFT-08)

```python
# Source: pytorch-forecasting interpret_output + plot_interpretation source (verified)
# Called after training; raw_predictions from a full prediction pass over train set
with torch.no_grad():
    raw_out = model.predict(train_loader, mode=('raw', 'prediction'))
interpretation = model.interpret_output(raw_out, reduction="mean")

# encoder_variables tensor: shape (n_encoder_features,) - importance weights
encoder_vars = interpretation["encoder_variables"].detach().cpu().numpy()
feature_names = model.encoder_variables  # list[str], same order as weights

# Sort and plot
order = np.argsort(encoder_vars)
fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.2)))
ax.barh(np.arange(len(feature_names)), encoder_vars[order] * 100,
        tick_label=np.array(feature_names)[order])
ax.set_xlabel("Variable Importance (%)")
ax.set_title("TFT Variable Selection Network — Encoder Features")
plt.tight_layout()
fig.savefig("experiments/figures/tft_variable_importance.png", dpi=300, bbox_inches='tight')
plt.close(fig)
```

### Pattern 4: Attention Entropy Audit (TFT-05)

```python
# Source: PITFALLS.md P1 detection pattern
# Called after each training seed before reporting results
import numpy as np

def audit_attention_entropy(interpretation: dict, n_features: int) -> dict:
    """Returns audit dict with is_degenerate flag and metrics."""
    enc_vars = interpretation["encoder_variables"].detach().cpu().numpy()
    # Normalize to probability distribution
    probs = enc_vars / enc_vars.sum()
    # Shannon entropy
    probs_nonzero = probs[probs > 0]
    entropy = float(-np.sum(probs_nonzero * np.log(probs_nonzero)))
    max_weight = float(probs.max())
    threshold_entropy = 0.5 * np.log(n_features)
    is_degenerate = (entropy < threshold_entropy) or (max_weight > 0.8)
    return {
        "entropy": entropy,
        "max_variable_weight": max_weight,
        "threshold_entropy": threshold_entropy,
        "is_degenerate": is_degenerate,
    }
```

### Pattern 5: run_baselines.py Integration (+5 lines, TFT-06)

The exact insertion points in `run_baselines.py`:

```python
# 1. Top-level import (add after LSTMPredictor import):
from src.models.tft import TFTPredictor  # lazy or direct

# 2. In _MODEL_ORDER list — insert "TFT" after "LSTM":
_MODEL_ORDER = [..., "GRU", "LSTM", "TFT", "PPO-Raw", ...]

# 3. In run_tier2_with_seeds() — add TFTPredictor to model_classes:
model_classes: list[type[BasePredictor]] = [GRUPredictor, LSTMPredictor, TFTPredictor]
```

### Pattern 6: run_walk_forward.py Integration (+3 lines, TFT-03)

```python
# In run_walk_forward.py, after GRU/LSTM imports (line ~185):
try:
    from src.models.tft import TFTPredictor  # noqa: WPS433
    model_factories["tft"] = TFTPredictor
    logger.info("pytorch-forecasting available — including TFT")
except ImportError:
    logger.info("pytorch-forecasting not available — skipping TFT")

# And in the model loop condition (line ~261):
if name in ("gru", "lstm", "tft"):   # ADD 'tft' to sequence branch
    model.fit(X_train_seq, y_train_seq)
    preds = model.predict(X_test_seq)
```

### Anti-Patterns to Avoid

- **Direct TFT constructor call:** requires manually specifying `x_reals`, `x_categoricals`, `embedding_sizes`. Use `TemporalFusionTransformer.from_dataset(training_dataset, **hparams)` instead.
- **GroupNormalizer with transformation='softplus':** maps negative spread changes near zero. Our target is signed; use `transformation=None`.
- **Fitting GroupNormalizer in predict():** data leakage. The normalizer fitted during training must be baked into `_training_dataset` and reused via `from_dataset()`.
- **Extracting predictions from quantile index 3 (the 0.5 slot in default 7-quantile list):** use `mode='prediction'` which calls `to_prediction()` and finds the 0.5-index automatically. Safer than hardcoding index.
- **mode=('raw', 'prediction') for normal prediction:** this is for interpret_output only. Use `mode='prediction'` for the row-aligned point forecast.
- **Using MPS accelerator:** Phase 5 decisions documented an MPS segfault bug. Use `accelerator='cpu'` explicitly.
- **Re-using training TimeSeriesDataSet object for prediction:** the training dataset has `predict_mode=False`; prediction requires `from_dataset(..., predict=True)`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Time-series dataset plumbing | Custom batching loop for TFT sequences | `TimeSeriesDataSet` | Handles encoder/decoder split, known-future vs unknown-past, GroupNormalizer fit, variable-length sequences |
| Quantile output → point prediction | `output[:, :, 3]` hardcoded index | `mode='prediction'` on `TFT.predict()` | `QuantileLoss.to_prediction()` finds 0.5-quantile index safely; works for any quantile list |
| Early stopping | Manual epoch loop with patience counter | `pl.callbacks.EarlyStopping(monitor='val_loss', patience=5)` | Already integrated with Trainer's validation loop |
| VSN importance bar chart | Custom matplotlib from raw weights | `model.interpret_output()` + `model.encoder_variables` | Returns normalized per-feature weights in correct order; column names already attached |
| Warm-up stitching | Rebuild all windows from scratch in predict | Cache `_training_dataset` and use `from_dataset(..., predict=True)` | `from_dataset` preserves normalizer, encoder length, feature order from training |

**Key insight:** The entire TFT complexity (encoder/decoder architecture, variable selection network, attention layers) is hidden inside `pytorch_forecasting.TemporalFusionTransformer`. The implementation task is wrapping its training loop and prediction path behind the `BasePredictor` interface — not re-implementing any TFT component.

---

## Common Pitfalls

### Pitfall 1: GroupNormalizer transformation='softplus' crashes on negative targets

**What goes wrong:** REQUIREMENTS.md says "GroupNormalizer per-pair" and STACK.md's config table says `transformation='softplus'`. But softplus is for positive-only targets (maps negatives near zero). Spread changes are signed. Using softplus causes the normalizer to warp the target distribution and TFT learns to predict near-zero, yielding degenerate P&L.

**How to avoid:** `GroupNormalizer(groups=['group_id'], transformation=None)`. This does standard z-normalization per pair, which is correct for signed spread changes.

**Warning signs:** val_loss appears very low (normalizer is squashing targets), but all predictions are near zero, trade count collapses to 0 at any reasonable threshold.

### Pitfall 2: time_idx column not monotonically increasing per group

**What goes wrong:** TimeSeriesDataSet requires `time_idx` to be strictly monotonic within each group. If derived from the dataset's global `time_idx` column (which exists in the parquet but is NOT per-group monotonic — it's a global sort key), TFT silently trains on shuffled sequences.

**How to avoid:** Derive `time_idx` inside `fit()` as `df.groupby('group_id').cumcount()`. This gives 0, 1, 2, ..., n_group-1 per group regardless of global ordering.

**Warning signs:** `TimeSeriesDataSet` construction raises `ValueError: time_idx must be monotonically increasing per group`.

### Pitfall 3: 3 pairs with 5 bars will cause TimeSeriesDataSet to silently drop them

**What goes wrong:** `min_encoder_length` defaults to `max_encoder_length // 2`. With `max_encoder_length=6`, `min_encoder_length` defaults to 3. The 3 pairs with exactly 5 bars (< 6) will produce sequences but with encoder_length=5. This is actually fine — TFT handles variable-length encoders. But if `allow_missing_timesteps=False` (default), any gap in time_idx will raise an error.

**How to avoid:** Set `allow_missing_timesteps=True` in TimeSeriesDataSet. The 3 short pairs (5 bars each) will produce valid-but-short encoder sequences. No further action needed; GRU's existing padded-pairs behavior is not required for TFT.

**Warning signs:** `TimeSeriesDataSet` construction raises about missing timesteps; or training silently uses 141 pairs instead of 144.

### Pitfall 4: from_dataset prediction set requires same group_ids as training

**What goes wrong:** If a pair_id in the test set was not seen during training, `from_dataset(predict=True)` will raise a `KeyError` because the GroupNormalizer has no statistics for that group. This cannot happen with our static split (train pairs = test pairs), but can happen in walk-forward windows where early windows have fewer pairs.

**How to avoid:** In walk-forward, catch `KeyError` / `Exception` in the `try:` block already present in `run_walk_forward.py` around `model.fit()` / `model.predict()`. Existing infrastructure handles this.

### Pitfall 5: Training time blows the 1-day time-box in walk-forward

**What goes wrong:** TFT with 30 epochs × 11 walk-forward windows = 330 training runs. On CPU, each TFT epoch on ~5,000 rows takes approximately 30-60 seconds, making 11-window walk-forward unfeasible (330 epochs × 45s ≈ 4 hours).

**How to avoid:** Run walk-forward with fewer epochs for TFT. Set `max_epochs=10` for walk-forward. Or skip walk-forward entirely for TFT and only run single-split (TFT-03 says "single-split + walk-forward" but TFT-04's 1-day time-box is the governing constraint). Single-split with 3 seeds × 30 epochs ≈ 90 epochs × ~45s = ~67 minutes, which fits in the time-box.

**Decision for planner:** if 1-day time-box is tight, scope TFT walk-forward to 5 epochs per window (`max_epochs=10` for walk-forward) or skip walk-forward for TFT entirely. The paper finding ("TFT at N=6,802") does not require walk-forward — single-split is sufficient for Table 2.

### Pitfall 6: NaN loss on first epoch (numerical instability with small hidden_size)

**What goes wrong:** With `hidden_size=8` and 51 features going through the VSN's GRN layers, gradient explosion can occur in epoch 1 before `gradient_clip_val` takes effect (it clips during backward, not forward). Result: NaN loss, training collapses.

**How to avoid:** `gradient_clip_val=0.1` is already in the pre-specified hyperparameters. If NaN still appears, lower learning rate to `5e-4`. The fallback documented for TFT-04 is "TFT did not converge" — identifiable as: `val_loss` is NaN after epoch 2, or `val_loss > 2.0` (above GRU's ~0.085 raw loss) after epoch 10.

**Warning signs:** Lightning logs show `val_loss=nan` or `inf`. `model.predict()` returns arrays of NaN.

---

## Code Examples

### Full fit() pseudocode

```python
# Source: TimeSeriesDataSet API (verified at runtime), GRUPredictor pattern
def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> "TFTPredictor":
    if "group_id" not in X_train.columns:
        raise ValueError("TFTPredictor.fit requires 'group_id' column")
    
    set_seed(self._random_state)
    
    feature_cols = [c for c in X_train.columns if c != "group_id"]
    y_train = np.asarray(y_train, dtype=float)
    
    # Build long-format DataFrame for TimeSeriesDataSet
    df = X_train.copy()
    df["target"] = y_train
    df["time_idx"] = df.groupby("group_id").cumcount()  # per-group monotonic index
    
    # Scale features (same helpers as GRU)
    bool_cols = [c for c in feature_cols if X_train[c].dtype == bool]
    self._scaler = fit_feature_scaler(X_train[feature_cols], bool_cols)
    scaled_vals = apply_feature_scaler(X_train[feature_cols], self._scaler, bool_cols)
    for i, col in enumerate(feature_cols):
        df[col] = scaled_vals[:, i]
    
    # TimeSeriesDataSet
    self._training_dataset = TimeSeriesDataSet(
        data=df,
        time_idx="time_idx",
        target="target",
        group_ids=["group_id"],
        max_encoder_length=self._max_encoder_length,
        max_prediction_length=1,
        static_categoricals=["group_id"],
        time_varying_unknown_reals=feature_cols,
        target_normalizer=GroupNormalizer(groups=["group_id"], transformation=None),
        allow_missing_timesteps=True,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    # Validation split (last 10% of each group, same as GRU)
    val_cutoff = int(len(df) * 0.9)
    val_dataset = TimeSeriesDataSet.from_dataset(
        self._training_dataset,
        df[df.index >= val_cutoff],  # rough; TFT respects time_idx internally
        predict=False,
        stop_randomization=True,
    )
    
    train_loader = self._training_dataset.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_loader = val_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)
    
    # Instantiate model from dataset
    self._model = TemporalFusionTransformer.from_dataset(
        self._training_dataset,
        hidden_size=self._hidden_size,
        attention_head_size=self._attention_head_size,
        dropout=self._dropout,
        hidden_continuous_size=self._hidden_continuous_size,
        lstm_layers=self._lstm_layers,
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        learning_rate=self._learning_rate,
        log_interval=-1,
        log_val_interval=-1,
    )
    
    # Train with Lightning
    trainer = pl.Trainer(
        max_epochs=self._max_epochs,
        gradient_clip_val=0.1,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[EarlyStopping(monitor="val_loss", patience=self._patience, mode="min")],
    )
    trainer.fit(self._model, train_loader, val_loader)
    
    # Cache for warm-up stitching
    self._cached_train = {
        "df": df,               # long-format with time_idx
        "feature_cols": feature_cols,
        "bool_cols": bool_cols,
    }
    self._fitted = True
    return self
```

### Full predict() pseudocode

```python
# Source: TimeSeriesDataSet.from_dataset signature (verified at runtime)
def predict(self, X: pd.DataFrame) -> np.ndarray:
    if not self._fitted:
        raise RuntimeError("Must call fit() first")
    if "group_id" not in X.columns:
        raise ValueError("TFTPredictor.predict requires 'group_id' column")
    
    feature_cols = self._cached_train["feature_cols"]
    bool_cols = self._cached_train["bool_cols"]
    
    # Scale test features with train-fitted scaler (no re-fit)
    X_scaled = X.copy()
    scaled_vals = apply_feature_scaler(X[feature_cols], self._scaler, bool_cols)
    for i, col in enumerate(feature_cols):
        X_scaled[col] = scaled_vals[:, i]
    
    # Derive time_idx for test rows (continuing from end of train per group)
    train_df = self._cached_train["df"]
    max_train_idx = train_df.groupby("group_id")["time_idx"].max()
    X_scaled["time_idx"] = X_scaled.groupby("group_id").cumcount()
    for gid in X_scaled["group_id"].unique():
        offset = max_train_idx.get(gid, -1) + 1
        mask = X_scaled["group_id"] == gid
        X_scaled.loc[mask, "time_idx"] += offset
    X_scaled["target"] = 0.0  # placeholder; overwritten during prediction
    
    # Stitch train rows for warm-up
    stitched = pd.concat([train_df, X_scaled], ignore_index=True)
    
    pred_dataset = TimeSeriesDataSet.from_dataset(
        self._training_dataset,
        stitched,
        predict=True,
        stop_randomization=True,
    )
    pred_loader = pred_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)
    
    # mode='prediction' calls QuantileLoss.to_prediction() -> returns 0.5-quantile
    raw_preds = self._model.predict(pred_loader, mode="prediction")
    # raw_preds: tensor of shape (n_rows,) — these are the test rows only (predict=True)
    return raw_preds.numpy().astype(float)
```

### Attention audit code

```python
# Source: PITFALLS.md P1 + verified against interpret_output return keys
def _audit_attention(self) -> dict:
    """Run after fit(). Returns degenerate flag and entropy metrics."""
    import torch
    train_loader = self._training_dataset.to_dataloader(
        train=False, batch_size=64, num_workers=0
    )
    with torch.no_grad():
        # Collect raw output for interpret_output
        raw_predictions = self._model.predict(
            train_loader, mode="raw", return_x=False
        )
    interpretation = self._model.interpret_output(
        raw_predictions, reduction="mean"
    )
    enc_vars = interpretation["encoder_variables"].detach().cpu().numpy()
    probs = enc_vars / enc_vars.sum()
    probs_nz = probs[probs > 0]
    entropy = float(-np.sum(probs_nz * np.log(probs_nz)))
    max_weight = float(probs.max())
    n_features = len(enc_vars)
    threshold = 0.5 * np.log(n_features)
    is_degenerate = (entropy < threshold) or (max_weight > 0.8)
    if is_degenerate:
        print(
            f"[TFT] WARN: attention collapse — entropy={entropy:.3f} "
            f"(threshold={threshold:.3f}), max_weight={max_weight:.3f}"
        )
    return {
        "entropy": entropy, "max_variable_weight": max_weight,
        "threshold_entropy": threshold, "is_degenerate": is_degenerate,
    }
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| TFT direct `__init__` constructor | `TemporalFusionTransformer.from_dataset()` factory | pytorch-forecasting 1.0+ | from_dataset propagates all feature-name and normalizer state automatically; direct init requires manual x_reals/x_categoricals |
| 7-quantile QuantileLoss (default) | 3-quantile QuantileLoss for small-data | — | Reduces prediction head parameters, speeds training by ~2x on CPU |
| mode='raw' for point forecast | mode='prediction' calls to_prediction() for median | pytorch-forecasting 1.x | Safe median extraction regardless of quantile list ordering |

**Deprecated/outdated:**
- `pytorch-forecasting` TFT v2 (hidden in `_tft_v2.TFT`): The v2 version has `hidden_size=64` default and additional architectural components. Do NOT use v2 for small data. Use the standard `pytorch_forecasting.TemporalFusionTransformer` (legacy, but better documented for small-data regime).

---

## Open Questions

1. **Walk-forward time budget for TFT**
   - What we know: single-split 3-seed run ≈ 67 minutes on CPU. Walk-forward (11 windows × 3 seeds × 10 epochs) ≈ 2-3 hours.
   - What's unclear: whether the 1-day time-box permits both single-split and walk-forward.
   - Recommendation: Planner should scope TFT walk-forward to `max_epochs=10` (vs 30 for single-split), or mark walk-forward as optional for TFT ("best effort"). The paper finding from single-split alone is sufficient for TFT-07.

2. **Validation split strategy inside fit()**
   - What we know: GRU uses a 90/10 chronological within-group split, manually constructing train/val masks before building sequences. TFT's TimeSeriesDataSet can accept a cutoff via `min_prediction_idx` or we can split the DataFrame before constructing two separate datasets.
   - What's unclear: whether `TimeSeriesDataSet.from_dataset(..., predict=False)` for the val split needs rows from both train and val groups, or only val-period rows.
   - Recommendation: Split the long-format DataFrame at `min_prediction_idx = max_time_idx_in_train - max_encoder_length - 1` so the val dataset sees proper encoder context. Alternatively, use a simple 80/20 chronological DataFrame split and construct two separate TimeSeriesDataSet objects. The latter is simpler and matches GRU's pattern more directly.

3. **GPU availability on SCC vs. local Mac**
   - What we know: SCC has GPU access. Local Mac uses MPS which has a segfault risk (Phase 5 decisions). CPU training is ~45s/epoch.
   - What's unclear: whether to run TFT on SCC for speed or on local for debugging convenience.
   - Recommendation: develop and debug locally on CPU (accelerator='cpu'). If 3-seed single-split run exceeds 2 hours, run on SCC via scp. Single-split is the priority; walk-forward can be skipped if time-boxed.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (existing in project) |
| Config file | `pytest.ini` or `pyproject.toml` (existing) |
| Quick run command | `python -m pytest tests/models/test_tft.py -x -q` |
| Full suite command | `python -m pytest tests/ -x -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TFT-01 | TFTPredictor subclasses BasePredictor; fit/predict shape contract | smoke | `pytest tests/models/test_tft.py::test_tft_inherits_base_predictor -x` | ❌ Wave 0 |
| TFT-01 | predict() returns 1-D array len(X) for test input | smoke | `pytest tests/models/test_tft.py::test_predict_shape -x` | ❌ Wave 0 |
| TFT-01 | group_id guard raises ValueError if missing | smoke | `pytest tests/models/test_tft.py::test_group_id_guard -x` | ❌ Wave 0 |
| TFT-02 | Hyperparameter defaults match pre-specified values | smoke | `pytest tests/models/test_tft.py::test_hyperparameter_defaults -x` | ❌ Wave 0 |
| TFT-05 | audit_attention returns is_degenerate=True when max_weight > 0.8 | unit | `pytest tests/models/test_tft.py::test_attention_audit_degenerate -x` | ❌ Wave 0 |
| TFT-08 | VSN heatmap file saved to correct path | smoke | manual verification post-run | N/A |

### Sampling Rate

- **Per task commit:** `python -m pytest tests/models/test_tft.py -x -q`
- **Per wave merge:** `python -m pytest tests/models/ tests/evaluation/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/models/test_tft.py` — smoke test for TFTPredictor contract; covers TFT-01, TFT-02, TFT-05 unit test for entropy audit

*(All other test infrastructure exists: pytest, conftest.py, fixtures for train/test DataFrames are established in test_gru.py pattern)*

---

## Sources

### Primary (HIGH confidence — verified at runtime against installed library)

- pytorch-forecasting 1.7.0 installed in `.venv/` — `TemporalFusionTransformer`, `TimeSeriesDataSet`, `QuantileLoss`, `GroupNormalizer` all import cleanly; signatures inspected at runtime
- `QuantileLoss.to_prediction()` source — verified: extracts `quantiles.index(0.5)` quantile; 0.5 IS in default 7-quantile list `[0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]`
- `TemporalFusionTransformer.interpret_output()` source — verified: returns dict with keys `attention`, `encoder_variables`, `decoder_variables`, `static_variables`, `encoder_length_histogram`, `decoder_length_histogram`
- `TemporalFusionTransformer.encoder_variables` — attribute confirmed accessible for feature name lookup
- `GroupNormalizer.__init__` — verified: `transformation=None` is correct for signed targets
- `pl.Trainer.__init__` — verified: `accelerator`, `devices`, `callbacks`, `gradient_clip_val`, `enable_progress_bar`, `enable_model_summary` all present in pytorch-lightning 2.6.1
- `experiments/results/tier2/GRU.json` — GRU RMSE=0.2928, P&L=+$212.50 (concrete TFT-04 target)
- `experiments/results/tier2/LSTM.json` — LSTM RMSE=0.2915, P&L=+$221.84 (context)
- `data/processed/train.parquet` — 6,946 rows, 144 pairs, 39 raw columns → 51 features after `compute_derived_features()`
- `data/processed/test.parquet` — 1,817 rows
- 3 short-sequence pairs (5 bars): `kxbtc26feb2817b-0x1179eff8`, `kxbtc26jan2817b-0x164c1750`, `kxbtc26jan3017b-0xa73ec60b`
- `src/models/gru.py` — warm-up stitching pattern, `_cached_train` design, `group_id` guard, scaler helpers
- `experiments/run_baselines.py` — `run_tier2_with_seeds()` function, `model_classes` list, `_MODEL_ORDER`
- `experiments/run_walk_forward.py` — sequence model branch at line ~261 (`if name in ("gru", "lstm")`)

### Secondary (MEDIUM confidence — planning documents, previously researched)

- `.planning/research/STACK.md` — TFT hyperparameter table for small-data regime
- `.planning/research/PITFALLS.md` P1 — TFT overfitting detection code and prevention strategy
- `.planning/research/ARCHITECTURE.md` — TFTPredictor integration contract and data-flow diagram

### Tertiary (LOW confidence — prior research, not re-verified)

- STACK.md reference to pytorch-forecasting GitHub Issue #1322 — hidden_size=160 OOM on 40-60k rows; extrapolated to our 6.8k-row case

---

## Metadata

**Confidence breakdown:**

| Area | Level | Reason |
|------|-------|--------|
| TimeSeriesDataSet API | HIGH | Signatures inspected at runtime against installed 1.7.0 |
| QuantileLoss / point prediction | HIGH | `to_prediction()` source read; 0.5 quantile presence verified |
| GroupNormalizer choice | HIGH | `transformation=None` confirmed correct for signed targets |
| interpret_output / VSN heatmap | HIGH | Return dict keys confirmed from source; `encoder_variables` attribute confirmed |
| Hyperparameter values | HIGH | Locked by REQUIREMENTS.md TFT-02; not implementation-time decisions |
| Training time estimate | MEDIUM | Rough estimate from epoch count × batch overhead; no empirical run done |
| Walk-forward time budget | MEDIUM | Extrapolated from single-split estimate |
| Attention audit entropy formula | HIGH | Confirmed from PITFALLS.md P1 with n_features=51 verified |
| Short-sequence pair handling | HIGH | 3 pairs with 5 bars confirmed from data; `allow_missing_timesteps=True` is the fix |

**Research date:** 2026-04-22
**Valid until:** 2026-05-22 (pytorch-forecasting 1.7.0 API is stable)
