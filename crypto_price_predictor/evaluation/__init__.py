"""
Evaluation module for cryptocurrency price prediction.

This module contains tools for evaluating and visualizing the performance
of cryptocurrency price prediction models.
"""

from .metrics import (
    calculate_metrics, 
    plot_predictions, 
    plot_error_distribution, 
    evaluate_multiple_models,
    calculate_directional_accuracy
)

__all__ = [
    'calculate_metrics',
    'plot_predictions',
    'plot_error_distribution',
    'evaluate_multiple_models',
    'calculate_directional_accuracy'
]