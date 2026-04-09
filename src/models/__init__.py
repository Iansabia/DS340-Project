"""Prediction models for spread forecasting.

Core models (no PyTorch dependency) are imported eagerly.
PyTorch-dependent models (GRU, LSTM, PPO, Autoencoder) are imported
lazily to allow deployment on lightweight environments without torch.
"""
from src.models.base import BasePredictor
from src.models.linear_regression import LinearRegressionPredictor
from src.models.xgboost_model import XGBoostPredictor
from src.models.naive import NaivePredictor
from src.models.volume import VolumePredictor


def __getattr__(name: str):
    """Lazy imports for PyTorch-dependent models."""
    _lazy = {
        "GRUPredictor": "src.models.gru",
        "LSTMPredictor": "src.models.lstm",
        "PPORawPredictor": "src.models.ppo_raw",
        "PPOFilteredPredictor": "src.models.ppo_filtered",
        "AnomalyDetectorAutoencoder": "src.models.autoencoder",
    }
    if name in _lazy:
        import importlib
        mod = importlib.import_module(_lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BasePredictor",
    "LinearRegressionPredictor",
    "XGBoostPredictor",
    "NaivePredictor",
    "VolumePredictor",
    "GRUPredictor",
    "LSTMPredictor",
    "PPORawPredictor",
    "PPOFilteredPredictor",
    "AnomalyDetectorAutoencoder",
]
