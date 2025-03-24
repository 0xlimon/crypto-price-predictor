"""
Metrics and evaluation tools for cryptocurrency price prediction models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,
    r2_score, explained_variance_score
)
import logging
from scipy import stats

logger = logging.getLogger(__name__)


def calculate_metrics(
    y_true: Union[np.ndarray, pd.Series], 
    y_pred: Union[np.ndarray, pd.Series],
    prefix: str = ''
) -> Dict[str, float]:
    """
    Calculate a comprehensive set of regression metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        prefix: Prefix to add to metric names (for comparing multiple models)
        
    Returns:
        Dictionary of metrics
    """
    # Ensure arrays
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Calculate metrics
    metrics = {}
    
    # Standard regression metrics
    metrics[f'{prefix}mae'] = mean_absolute_error(y_true, y_pred)
    metrics[f'{prefix}mse'] = mean_squared_error(y_true, y_pred)
    metrics[f'{prefix}rmse'] = np.sqrt(metrics[f'{prefix}mse'])
    
    # Only calculate MAPE if no zeros in true values (avoid division by zero)
    if not np.any(y_true == 0):
        metrics[f'{prefix}mape'] = mean_absolute_percentage_error(y_true, y_pred)
    else:
        # Alternative: custom MAPE handling zeros with a small epsilon
        epsilon = 1e-10
        metrics[f'{prefix}mape'] = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    # R-squared and explained variance
    metrics[f'{prefix}r2'] = r2_score(y_true, y_pred)
    metrics[f'{prefix}explained_variance'] = explained_variance_score(y_true, y_pred)
    
    # Financial and trading specific metrics
    metrics[f'{prefix}directional_accuracy'] = calculate_directional_accuracy(y_true, y_pred)
    
    # Maximum absolute error
    metrics[f'{prefix}max_error'] = np.max(np.abs(y_true - y_pred))
    
    # Median absolute error
    metrics[f'{prefix}median_ae'] = np.median(np.abs(y_true - y_pred))
    
    # Residual statistics
    residuals = y_true - y_pred
    metrics[f'{prefix}residual_mean'] = np.mean(residuals)
    metrics[f'{prefix}residual_std'] = np.std(residuals)
    
    # Log the metrics
    metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(f"Model evaluation metrics: {metric_str}")
    
    return metrics


def calculate_directional_accuracy(
    y_true: Union[np.ndarray, pd.Series], 
    y_pred: Union[np.ndarray, pd.Series],
    lag: int = 1
) -> float:
    """
    Calculate directional accuracy (percentage of correct movement predictions).
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        lag: Lag for calculating price movements
        
    Returns:
        Directional accuracy score (0-1)
    """
    # Ensure arrays
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Calculate actual and predicted directions
    true_dir = np.sign(y_true[lag:] - y_true[:-lag])
    pred_dir = np.sign(y_pred[lag:] - y_true[:-lag])
    
    # Calculate accuracy
    matches = true_dir == pred_dir
    accuracy = np.mean(matches)
    
    return accuracy


def calculate_trading_metrics(
    y_true: Union[np.ndarray, pd.Series], 
    y_pred: Union[np.ndarray, pd.Series],
    initial_capital: float = 10000.0,
    transaction_cost_pct: float = 0.001
) -> Dict[str, float]:
    """
    Calculate trading-specific metrics based on a simple strategy.
    
    Args:
        y_true: True prices (should be original/unscaled)
        y_pred: Predicted prices (should be original/unscaled)
        initial_capital: Initial capital for simulated trading
        transaction_cost_pct: Transaction cost as percentage
        
    Returns:
        Dictionary of trading metrics
    """
    # Ensure arrays and same length
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Simulate a simple trading strategy:
    # Buy if predicted price is higher than current price
    # Sell if predicted price is lower than current price
    
    # Initialize portfolio and positions
    capital = initial_capital
    position = 0  # Number of coins held
    trades = 0
    returns = []
    
    # For each day (skip the first one)
    for i in range(1, len(y_true)):
        prev_price = y_true[i-1]
        current_price = y_true[i]
        
        # Predicted direction
        prediction = y_pred[i-1]  # Previous day's prediction for current day
        pred_direction = 1 if prediction > prev_price else -1
        
        # True direction
        true_direction = 1 if current_price > prev_price else -1
        
        # Trading logic
        if pred_direction > 0 and position == 0:  # Buy signal
            # Calculate how many coins we can buy
            transaction_cost = capital * transaction_cost_pct
            available_capital = capital - transaction_cost
            position = available_capital / prev_price
            capital = 0
            trades += 1
            
        elif pred_direction < 0 and position > 0:  # Sell signal
            # Sell all coins
            sale_value = position * prev_price
            transaction_cost = sale_value * transaction_cost_pct
            capital = sale_value - transaction_cost
            position = 0
            trades += 1
        
        # Calculate returns
        portfolio_value = capital + (position * current_price)
        returns.append(portfolio_value)
    
    # Calculate metrics
    final_value = returns[-1]
    profit_loss = final_value - initial_capital
    profit_loss_pct = (profit_loss / initial_capital) * 100
    
    # Buy and hold strategy for comparison
    coins_bought = initial_capital / y_true[0]
    hold_value = coins_bought * y_true[-1]
    hold_profit = hold_value - initial_capital
    hold_profit_pct = (hold_profit / initial_capital) * 100
    
    # Maximum drawdown
    peak = returns[0]
    max_drawdown = 0
    
    for value in returns:
        peak = max(peak, value)
        drawdown = (peak - value) / peak
        max_drawdown = max(max_drawdown, drawdown)
    
    # Sharpe ratio (simplified, assuming zero risk-free rate)
    returns_array = np.array(returns)
    daily_returns = returns_array[1:] / returns_array[:-1] - 1
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)  # Annualized
    
    # Compile results
    metrics = {
        'final_value': final_value,
        'profit_loss': profit_loss,
        'profit_loss_pct': profit_loss_pct,
        'num_trades': trades,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'hold_value': hold_value,
        'hold_profit': hold_profit,
        'hold_profit_pct': hold_profit_pct,
        'strategy_vs_hold': profit_loss_pct - hold_profit_pct
    }
    
    return metrics


def evaluate_forecast_horizon(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    steps: List[int] = [1, 3, 5, 7, 14],
    target_scaler=None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model performance across different forecast horizons.
    
    Args:
        model: Trained model with a predict method
        X_test: Test features
        y_test: Test targets
        steps: List of forecast horizons (steps ahead) to evaluate
        target_scaler: Scaler to inverse transform predictions
        
    Returns:
        Dictionary of metrics for each forecast horizon
    """
    results = {}
    
    # For each forecast horizon
    for step in steps:
        logger.info(f"Evaluating {step}-step ahead forecasts")
        
        # Skip steps that exceed the test set
        if step >= len(X_test):
            logger.warning(f"Step {step} exceeds test set length. Skipping.")
            continue
        
        # Get predictions
        y_pred = model.predict(X_test[:-step])
        
        # Shift test values to align with forecast horizon
        y_true = y_test[step:]
        
        # Truncate predictions to match shifted test values
        y_pred = y_pred[:len(y_true)]
        
        # Inverse transform if scaler is provided
        if target_scaler:
            y_pred = target_scaler.inverse_transform(y_pred)
            y_true = target_scaler.inverse_transform(y_true)
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred)
        results[f'horizon_{step}'] = metrics
    
    return results


def evaluate_multiple_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_scaler=None,
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate and compare multiple models.
    
    Args:
        models: Dictionary mapping model names to trained models
        X_test: Test features
        y_test: Test targets
        target_scaler: Scaler to inverse transform predictions
        verbose: Whether to print results
        
    Returns:
        Dictionary of metrics for each model
    """
    results = {}
    
    # For each model
    for model_name, model in models.items():
        logger.info(f"Evaluating model: {model_name}")
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Inverse transform if scaler is provided
        if target_scaler:
            y_pred = target_scaler.inverse_transform(y_pred)
            y_true = target_scaler.inverse_transform(y_test)
        else:
            y_true = y_test
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred, prefix=f'{model_name}_')
        results[model_name] = metrics
    
    # Print comparison if verbose
    if verbose:
        # Create DataFrame for easy comparison
        metrics_df = pd.DataFrame(results).T
        logger.info("\nModel comparison:")
        logger.info(metrics_df[['mae', 'rmse', 'mape', 'r2', 'directional_accuracy']])
    
    return results


def plot_predictions(
    y_true: Union[np.ndarray, pd.Series], 
    y_pred: Union[np.ndarray, pd.Series],
    dates: Optional[Union[pd.DatetimeIndex, List[str]]] = None,
    model_name: str = 'Model',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot predictions against actual values.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        dates: Dates corresponding to the data points
        model_name: Name of the model (for title)
        figsize: Figure size
        save_path: Path to save figure
    """
    # Ensure arrays
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create x-axis
    if dates is not None:
        x = dates
    else:
        x = range(len(y_true))
    
    # Plot data
    plt.plot(x, y_true, label='Actual', color='blue', linewidth=2)
    plt.plot(x, y_pred, label='Predicted', color='red', linestyle='--', linewidth=2)
    
    # Calculate metrics for title
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Add title and labels
    plt.title(f'{model_name} Predictions vs Actual\nMAE: {mae:.2f}, RMSE: {rmse:.2f}')
    plt.xlabel('Date' if dates is not None else 'Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # If dates are used, format x-axis
    if dates is not None:
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()


def plot_error_distribution(
    y_true: Union[np.ndarray, pd.Series], 
    y_pred: Union[np.ndarray, pd.Series],
    model_name: str = 'Model',
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> None:
    """
    Plot the distribution of prediction errors.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        model_name: Name of the model (for title)
        figsize: Figure size
        save_path: Path to save figure
    """
    # Ensure arrays
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Calculate errors
    errors = y_true - y_pred
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot histogram and KDE
    sns.histplot(errors, kde=True, stat='density', alpha=0.6)
    
    # Plot normal distribution for comparison
    mu, std = np.mean(errors), np.std(errors)
    x = np.linspace(mu - 3*std, mu + 3*std, 100)
    plt.plot(x, stats.norm.pdf(x, mu, std), 'r-', lw=2, 
             label=f'Normal Dist. $\mu$={mu:.2f}, $\sigma$={std:.2f}')
    
    # Add reference line at zero
    plt.axvline(x=0, color='darkgreen', linestyle='--', alpha=0.7, label='Zero Error')
    
    # Add title and labels
    plt.title(f'{model_name} Prediction Error Distribution')
    plt.xlabel('Error (Actual - Predicted)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add error stats text box
    stats_text = (
        f'Mean Error: {np.mean(errors):.4f}\n'
        f'Std Dev: {np.std(errors):.4f}\n'
        f'Skewness: {stats.skew(errors):.4f}\n'
        f'Kurtosis: {stats.kurtosis(errors):.4f}'
    )
    plt.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Error distribution plot saved to {save_path}")
    
    plt.show()


def plot_forecast_comparison(
    y_true: Union[np.ndarray, pd.Series],
    forecasts: Dict[str, np.ndarray],
    dates: Optional[Union[pd.DatetimeIndex, List[str]]] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot multiple forecasts against actual values.
    
    Args:
        y_true: True target values
        forecasts: Dictionary mapping model names to their predictions
        dates: Dates corresponding to the data points
        figsize: Figure size
        save_path: Path to save figure
    """
    # Ensure actual values are array
    y_true = np.asarray(y_true).flatten()
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create x-axis
    if dates is not None:
        x = dates
    else:
        x = range(len(y_true))
    
    # Plot actual values
    plt.plot(x, y_true, label='Actual', color='blue', linewidth=2)
    
    # Plot forecasts
    colors = plt.cm.tab10.colors
    for i, (model_name, y_pred) in enumerate(forecasts.items()):
        y_pred = np.asarray(y_pred).flatten()
        
        # Ensure lengths match
        if len(y_pred) > len(y_true):
            y_pred = y_pred[:len(y_true)]
        elif len(y_pred) < len(y_true):
            # Pad with NaN for visualization
            y_pred = np.pad(y_pred, (0, len(y_true) - len(y_pred)), 
                            constant_values=np.nan)
        
        plt.plot(x, y_pred, label=model_name, color=colors[i % len(colors)], 
                 linestyle='--', linewidth=1.5)
    
    # Add title and labels
    plt.title('Forecast Comparison')
    plt.xlabel('Date' if dates is not None else 'Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # If dates are used, format x-axis
    if dates is not None:
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Forecast comparison plot saved to {save_path}")
    
    plt.show()


def plot_metrics_comparison(
    metrics: Dict[str, Dict[str, float]],
    metric_names: List[str] = ['mae', 'rmse', 'mape', 'r2', 'directional_accuracy'],
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Plot comparison of multiple metrics across models.
    
    Args:
        metrics: Dictionary mapping model names to dictionaries of metrics
        metric_names: List of metric names to plot
        figsize: Figure size
        save_path: Path to save figure
    """
    # Create figure
    fig, axes = plt.subplots(nrows=len(metric_names), ncols=1, figsize=figsize, sharex=True)
    
    # If single metric, wrap axis in list
    if len(metric_names) == 1:
        axes = [axes]
    
    # Create DataFrame for sorting
    metrics_df = pd.DataFrame({model: {k: v for k, v in model_metrics.items() 
                                      if k in metric_names}
                              for model, model_metrics in metrics.items()})
    
    # For each metric
    for i, metric in enumerate(metric_names):
        # Sort by metric (ascending for error metrics, descending for accuracy)
        ascending = metric not in ['r2', 'directional_accuracy', 'explained_variance']
        sorted_models = metrics_df.loc[metric].sort_values(ascending=ascending).index
        
        # Extract values
        values = [metrics[model][metric] for model in sorted_models]
        
        # Plot
        bars = axes[i].barh(sorted_models, values, height=0.6)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            label_position = width if width < 0 else width + 0.01
            axes[i].text(label_position, bar.get_y() + bar.get_height()/2,
                        f'{width:.4f}', va='center')
        
        # Set title
        axes[i].set_title(f'{metric.upper()}')
        axes[i].grid(axis='x', linestyle='--', alpha=0.7)
    
    # Set global title
    fig.suptitle('Model Performance Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Metrics comparison plot saved to {save_path}")
    
    plt.show()


if __name__ == '__main__':
    # Example usage
    import sys
    import os
    sys.path.append(os.path.abspath('../..'))
    
    from crypto_price_predictor.data.data_loader import CryptoDataLoader
    from crypto_price_predictor.data.data_processor import DataProcessor
    from crypto_price_predictor.models.lstm_model import LSTMModel
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    # Load data
    loader = CryptoDataLoader()
    btc_data = loader.load_data('BTC-USD', interval='1d')
    
    # Process data
    processor = DataProcessor(
        sequence_length=30,
        forecast_horizon=1,
        feature_columns=['Close']
    )
    processed_data = processor.process_data(btc_data)
    
    # Create and train model
    model = LSTMModel(config={'epochs': 10})  # Few epochs for example
    model.train(processed_data)
    
    # Make predictions
    y_pred = model.predict(processed_data['X_test'], return_scaled=True)
    y_true = processed_data['y_test']
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    print(metrics)
    
    # Plot predictions
    # Need to inverse transform for plotting
    y_pred_orig = processor.target_scaler.inverse_transform(y_pred)
    y_true_orig = processor.target_scaler.inverse_transform(y_true)
    
    # Example date range
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Create some dummy dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=len(y_true))
    dates = pd.date_range(start=start_date, end=end_date, periods=len(y_true))
    
    # Plot
    plot_predictions(y_true_orig, y_pred_orig, dates=dates, model_name='LSTM')
    plot_error_distribution(y_true_orig, y_pred_orig, model_name='LSTM')