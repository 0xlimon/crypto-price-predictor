"""
Data handling module for cryptocurrency price prediction.

This module contains components for loading and preprocessing cryptocurrency data.
"""

from .data_loader import CryptoDataLoader
from .data_processor import DataProcessor

__all__ = ['CryptoDataLoader', 'DataProcessor']