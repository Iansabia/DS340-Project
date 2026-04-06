"""Prediction models for spread forecasting."""
from src.models.base import BasePredictor
from src.models.linear_regression import LinearRegressionPredictor
from src.models.xgboost_model import XGBoostPredictor
from src.models.naive import NaivePredictor
from src.models.volume import VolumePredictor
from src.models.gru import GRUPredictor
from src.models.lstm import LSTMPredictor
from src.models.ppo_raw import PPORawPredictor
from src.models.ppo_filtered import PPOFilteredPredictor
from src.models.autoencoder import AnomalyDetectorAutoencoder

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
