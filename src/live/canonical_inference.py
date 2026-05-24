"""Phase A canonical-oil XGBoost inference + shadow logging.

Module-level singleton (lazy-loaded on first use):

    canonical_model    — fitted XGBoostPredictor from models/canonical_oil/xgboost.pkl
    canonical_columns  — 50-feature column order (matches training)
    canonical_zero_var — 25 column names that had zero variance on train

Public helpers:

    is_enabled()       — True if CANONICAL_OIL_ENABLED=true (A2 / narrow-replace mode)
    is_shadow()        — True if CANONICAL_OIL_SHADOW=true (A1 / shadow mode)
    use_canonical(t)   — True if Kalshi ticker is in oil family
    predict(row_df)    — score a single-row DataFrame, returning float prediction
    log_shadow(...)    — append a JSONL record to canonical_predictions.jsonl

Design constraints from phase_a_v3.md:
    - Live read path is data/live/bars.parquet (the canonical symlink),
      never hardcode /projectnb.
    - Force-zero columns from zero_variance_columns.json at inference
      so live data distribution shifts in those columns can't perturb
      the model away from its training-time effective view.
    - Threshold 0.001 (scale-equivalent adaptation, not a tuning lever).
    - Oil-family only: KXWTI*, KXBRENT*, KXCRUDE, KXDIESEL,
      KXHEATINGOIL, KXGASOLINE, KXMEXCUBOIL.
    - Independent of LIVE_TRADING — Phase A is paper-only in all modes.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = REPO_ROOT / "models" / "canonical_oil"
LIVE_BARS_PATH = REPO_ROOT / "data" / "live" / "bars.parquet"
SHADOW_LOG_PATH = REPO_ROOT / "data" / "live" / "canonical_predictions.jsonl"

OIL_FAMILY_PREFIXES = (
    "KXWTI",
    "KXBRENT",
    "KXCRUDE",
    "KXDIESEL",
    "KXHEATINGOIL",
    "KXGASOLINE",
    "KXMEXCUBOIL",
)

PREDICTION_THRESHOLD = 0.001

# Module-level lazy state.
_lock = threading.Lock()
_loaded = False
_model = None
_columns: list[str] = []
_zero_var: frozenset[str] = frozenset()


def _read_bool_env(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def is_enabled() -> bool:
    """A2 narrow-replacement mode (canonical drives oil trade decisions).

    Defaults to False. Flip via CANONICAL_OIL_ENABLED=true at cutover.
    """
    return _read_bool_env("CANONICAL_OIL_ENABLED", False)


def is_shadow() -> bool:
    """A1 shadow mode (canonical scored + logged but not acted on).

    Defaults to True so the model produces parity data the moment the
    bundle is deployed. Turn off only after A2 stabilizes.
    """
    return _read_bool_env("CANONICAL_OIL_SHADOW", True)


def use_canonical(kalshi_ticker: str) -> bool:
    """Pair admission filter — canonical model serves the oil family only."""
    if not kalshi_ticker:
        return False
    return kalshi_ticker.startswith(OIL_FAMILY_PREFIXES)


def _ensure_loaded() -> None:
    global _loaded, _model, _columns, _zero_var
    if _loaded:
        return
    with _lock:
        if _loaded:
            return
        pkl_path = MODEL_DIR / "xgboost.pkl"
        cols_path = MODEL_DIR / "feature_columns.json"
        zv_path = MODEL_DIR / "zero_variance_columns.json"
        if not pkl_path.exists() or not cols_path.exists() or not zv_path.exists():
            raise FileNotFoundError(
                f"Canonical bundle incomplete in {MODEL_DIR}. Run scripts/export_canonical_oil.py."
            )
        with open(pkl_path, "rb") as f:
            _model = pickle.load(f)
        _columns = json.loads(cols_path.read_text())
        _zero_var = frozenset(json.loads(zv_path.read_text()))
        _loaded = True
        logger.info(
            "Canonical inference loaded: %d features, %d zero-variance, threshold=%.4f",
            len(_columns), len(_zero_var), PREDICTION_THRESHOLD,
        )


def build_row(features_df: pd.DataFrame) -> pd.DataFrame:
    """Project a feature DataFrame onto the canonical column schema.

    Accepts a single-row DataFrame produced by the existing strategy
    pipeline (compute_derived_features + fillna(0)). Returns a
    single-row DataFrame with EXACTLY the canonical 50 columns in fit
    order, with zero-variance columns force-zeroed regardless of
    incoming values.

    Missing canonical columns are filled with 0.0 (matching training-
    time fillna behavior on canonical_train).
    """
    _ensure_loaded()
    if len(features_df) != 1:
        raise ValueError(f"build_row expects 1 row, got {len(features_df)}")
    out_cols = {}
    for col in _columns:
        if col in _zero_var:
            out_cols[col] = 0.0
        elif col in features_df.columns:
            val = features_df[col].iloc[0]
            out_cols[col] = 0.0 if pd.isna(val) else float(val)
        else:
            out_cols[col] = 0.0
    return pd.DataFrame([out_cols], columns=_columns)


def predict(features_df: pd.DataFrame) -> float:
    """Score a single-row feature DataFrame, returning the canonical prediction."""
    _ensure_loaded()
    row = build_row(features_df)
    y = _model.predict(row)
    return float(np.asarray(y).reshape(-1)[0])


def would_trade(pred: float) -> bool:
    """Trade-rule: abs(pred) > threshold."""
    return abs(pred) > PREDICTION_THRESHOLD


def log_shadow(
    *,
    ts: int,
    pair_id: str,
    kalshi_ticker: str,
    canonical_pred: float,
    spread: float,
    legacy_avg_pred: Optional[float] = None,
    legacy_lr_pred: Optional[float] = None,
    legacy_xgb_pred: Optional[float] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Append a single shadow-prediction record to canonical_predictions.jsonl.

    Append-only, parent-dir auto-created. Designed to never raise — a
    logging failure here must not break the trading cycle.
    """
    try:
        SHADOW_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": int(ts),
            "ts_iso": datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat(),
            "pair_id": pair_id,
            "kalshi_ticker": kalshi_ticker,
            "canonical_pred": float(canonical_pred),
            "canonical_would_trade": bool(abs(canonical_pred) > PREDICTION_THRESHOLD),
            "canonical_threshold": PREDICTION_THRESHOLD,
            "spread": float(spread),
            "legacy_avg_pred": float(legacy_avg_pred) if legacy_avg_pred is not None else None,
            "legacy_lr_pred": float(legacy_lr_pred) if legacy_lr_pred is not None else None,
            "legacy_xgb_pred": float(legacy_xgb_pred) if legacy_xgb_pred is not None else None,
            "mode": "shadow" if is_shadow() and not is_enabled() else ("dual" if is_shadow() and is_enabled() else "live"),
        }
        if extra:
            rec["extra"] = extra
        with open(SHADOW_LOG_PATH, "a") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    except Exception as e:
        logger.warning("canonical shadow log failed: %s", e)
