"""
Models module for cryptocurrency price prediction.

This module contains implementations of various machine learning models
specialized for time series forecasting of cryptocurrency prices.
"""

from .lstm_model import LSTMModel
from .ensemble_model import EnsembleModel

__all__ = ['LSTMModel', 'EnsembleModel']