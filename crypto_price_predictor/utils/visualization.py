"""
Visualization utilities for cryptocurrency price prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import List, Dict, Optional, Union, Tuple, Any
import mplfinance as mpf
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Set default styles
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')


def plot_price_history(
    data: pd.DataFrame,
    columns: List[str] = ['Close'],
    title: str = 'Cryptocurrency Price History',
    figsize: Tuple[int, int] = (12, 6),
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    log_scale: bool = False,
    show_volume: bool = True,
    save_path: Optional[str] = None
) -> None:
    """
    Plot the price history of a cryptocurrency.
    
    Args:
        data: DataFrame with price data
        columns: List of price columns to plot (e.g., ['Close', 'Open'])
        title: Plot title
        figsize: Figure size
        start_date: Start date for the plot (if None, use all data)
        end_date: End date for the plot (if None, use all data)
        log_scale: Whether to use log scale for y-axis
        show_volume: Whether to show volume in a subplot
        save_path: Path to save the figure
    """
    # Ensure DataFrame has datetime index
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            logger.error(f"Could not convert index to datetime: {str(e)}")
            return
    
    # Filter by date if provided
    if start_date or end_date:
        if start_date:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            data = data[data.index >= start_date]
        
        if end_date:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            data = data[data.index <= end_date]
    
    # Create figure
    if show_volume and 'Volume' in data.columns:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
    
    # Plot price data
    for col in columns:
        if col in data.columns:
            ax1.plot(data.index, data[col], label=col, linewidth=2)
    
    # Set title and labels
    ax1.set_title(title, fontsize=14)
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend()
    
    # Set y-axis to log scale if requested
    if log_scale:
        ax1.set_yscale('log')
    
    # Format x-axis to show dates nicely
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    
    # Add volume subplot if requested
    if show_volume and 'Volume' in data.columns:
        try:
            # Check if Volume data is valid using proper method
            if data['Volume'].isnull().values.any():  # Use values.any() instead of any()
                logger.warning("Volume data contains NaN values, filling with zeros")
                data['Volume'] = data['Volume'].fillna(0)
            
            # Convert volume to float explicitly and handle any potential non-numeric values
            volume_data = data['Volume'].astype(float)
            
            # Plot volume as bar chart with scalar values
            ax2.bar(data.index, volume_data, color='gray', alpha=0.5, width=1)
            ax2.set_ylabel('Volume', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            
            # Add grid to volume subplot
            ax2.grid(True, alpha=0.3)
        except Exception as e:
            logger.error(f"Error plotting volume: {str(e)}")
            # Continue without volume subplot
            fig.delaxes(ax2)
            ax1.set_xlabel('Date', fontsize=12)
    else:
        ax1.set_xlabel('Date', fontsize=12)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Price history plot saved to {save_path}")
    
    plt.show()


def plot_technical_indicators(
    data: pd.DataFrame,
    indicators: Dict[str, List[str]] = None,
    title: str = 'Technical Indicators',
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Plot technical indicators for cryptocurrency analysis.
    
    Args:
        data: DataFrame with price data and indicators
        indicators: Dictionary mapping indicator groups to column names
            Example: {'trend': ['SMA_20', 'EMA_50'], 'oscillators': ['RSI_14']}
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
    """
    # Default indicators if none provided
    if indicators is None:
        indicators = {
            'trend': ['SMA_20', 'SMA_50', 'SMA_200'] if all(col in data.columns for col in ['SMA_20', 'SMA_50', 'SMA_200']) else [],
            'oscillators': ['RSI_14'] if 'RSI_14' in data.columns else [],
            'volatility': ['BB_High_20', 'BB_Low_20'] if all(col in data.columns for col in ['BB_High_20', 'BB_Low_20']) else []
        }
    
    # Count number of subplots needed
    num_plots = 1  # Price chart
    for group, cols in indicators.items():
        if group == 'trend' or group == 'volatility':
            # These go on the price chart
            pass
        elif cols:  # Only add subplot if group has indicators
            num_plots += 1
    
    # Create figure with appropriate number of rows
    fig, axes = plt.subplots(num_plots, 1, figsize=figsize, sharex=True)
    
    # If only one plot, wrap axes in list
    if num_plots == 1:
        axes = [axes]
    
    # Plot price and trend/volatility indicators on the first subplot
    ax_idx = 0
    ax = axes[ax_idx]
    
    # Plot price
    ax.plot(data.index, data['Close'], label='Close', color='blue', linewidth=2)
    
    # Plot trend indicators on same subplot
    if 'trend' in indicators:
        for col in indicators['trend']:
            if col in data.columns:
                ax.plot(data.index, data[col], label=col, alpha=0.7)
    
    # Plot volatility bands on same subplot
    if 'volatility' in indicators:
        for col in indicators['volatility']:
            if col in data.columns:
                if 'High' in col:
                    ax.plot(data.index, data[col], label=col, alpha=0.3, linestyle='--', color='green')
                elif 'Low' in col:
                    ax.plot(data.index, data[col], label=col, alpha=0.3, linestyle='--', color='red')
                else:
                    ax.plot(data.index, data[col], label=col, alpha=0.3)
    
    # Add legend and labels
    ax.set_title(f"{title} - Price and Trend", fontsize=12)
    ax.set_ylabel('Price', fontsize=10)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot other indicator groups in separate subplots
    for group, cols in indicators.items():
        if group not in ['trend', 'volatility'] and cols:
            ax_idx += 1
            ax = axes[ax_idx]
            
            for col in cols:
                if col in data.columns:
                    ax.plot(data.index, data[col], label=col)
            
            # Add horizontal lines for reference (e.g., overbought/oversold levels for RSI)
            if group == 'oscillators':
                if any('RSI' in col for col in cols):
                    ax.axhline(y=70, color='r', linestyle='--', alpha=0.3)
                    ax.axhline(y=30, color='g', linestyle='--', alpha=0.3)
            
            # Add legend and labels
            ax.set_title(f"{group.capitalize()} Indicators", fontsize=12)
            ax.set_ylabel(group.capitalize(), fontsize=10)
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
    
    # Set x-axis format for the last subplot
    axes[-1].set_xlabel('Date', fontsize=10)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Technical indicators plot saved to {save_path}")
    
    plt.show()


def plot_correlation_matrix(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
    title: str = 'Feature Correlation Matrix',
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'coolwarm',
    save_path: Optional[str] = None
) -> None:
    """
    Plot a correlation matrix for selected features.
    
    Args:
        data: DataFrame with data
        columns: List of columns to include in correlation matrix
        title: Plot title
        figsize: Figure size
        cmap: Colormap for the heatmap
        save_path: Path to save the figure
    """
    # Select columns to use
    if columns:
        # Filter only columns that exist in data
        valid_columns = [col for col in columns if col in data.columns]
        if not valid_columns:
            logger.error(f"None of the specified columns exist in the data")
            return
        df = data[valid_columns]
    else:
        # Use all numeric columns by default
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        df = data[numeric_cols]
    
    # Calculate correlation matrix
    corr = df.corr()
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Set up the matplotlib figure
    plt.figure(figsize=figsize)
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True) if cmap == 'coolwarm' else cmap
    
    # Draw the heatmap with the mask
    sns.heatmap(
        corr, 
        mask=mask, 
        cmap=cmap,
        vmax=1.0, 
        vmin=-1.0, 
        center=0,
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .5},
        annot=True, 
        fmt=".2f"
    )
    
    # Adjust layout
    plt.title(title, fontsize=14)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Correlation matrix plot saved to {save_path}")
    
    plt.show()


def plot_feature_importance(
    importance: Union[np.ndarray, List[float]],
    feature_names: List[str],
    title: str = 'Feature Importance',
    figsize: Tuple[int, int] = (10, 8),
    max_features: int = 20,
    save_path: Optional[str] = None
) -> None:
    """
    Plot feature importance from a model.
    
    Args:
        importance: Array of feature importance values
        feature_names: Names of features
        title: Plot title
        figsize: Figure size
        max_features: Maximum number of features to show
        save_path: Path to save the figure
    """
    # Convert to numpy array
    importance = np.array(importance)
    
    # Sort by importance
    indices = np.argsort(importance)
    
    # Select top features
    if len(indices) > max_features:
        indices = indices[-max_features:]
    
    # Reverse order for plotting
    indices = indices[::-1]
    
    # Get feature names and importance values
    names = [feature_names[i] for i in indices]
    values = importance[indices]
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create horizontal bar chart
    plt.barh(range(len(names)), values, align='center')
    plt.yticks(range(len(names)), names)
    
    # Add labels and title
    plt.xlabel('Importance')
    plt.title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
    
    plt.show()


def create_candlestick_chart(
    data: pd.DataFrame,
    title: str = 'Cryptocurrency Candlestick Chart',
    figsize: Tuple[int, int] = (14, 8),
    volume: bool = True,
    mav: Union[List[int], int, None] = (10, 20, 50),
    save_path: Optional[str] = None
) -> None:
    """
    Create a candlestick chart for cryptocurrency data.
    
    Args:
        data: DataFrame with OHLCV data
        title: Chart title
        figsize: Figure size
        volume: Whether to include volume subplot
        mav: Moving average windows (e.g., [10, 20, 50])
        save_path: Path to save the figure
    """
    # Ensure data has required columns
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in data.columns for col in required_cols):
        logger.error(f"Data must have Open, High, Low, Close columns")
        return
    
    # Create a copy and ensure all OHLC data is float type
    df = data.copy()
    for col in required_cols:
        # Convert to numeric first, coercing any errors to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Then fill any NaNs with a reasonable value
        if df[col].isnull().values.any():
            logger.warning(f"Column {col} contains non-numeric values, filling NaNs with mean")
            df[col] = df[col].fillna(df[col].mean())
    
    # Ensure DataFrame has datetime index
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            logger.error(f"Could not convert index to datetime: {str(e)}")
            return
    
    # Set up style
    mc = mpf.make_marketcolors(
        up='green', down='red',
        wick={'up': 'green', 'down': 'red'},
        volume={'up': 'green', 'down': 'red'}
    )
    
    # Create style without figsize parameter (not supported in some versions)
    s = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle='-',
        y_on_right=False
    )
    
    # Set up kwargs
    kwargs = {
        'type': 'candle',
        'volume': volume,
        'title': title,
        'figsize': figsize,
        'style': s,
        'mav': mav,
        'warn_too_much_data': 10000
    }
    
    # Add save path if provided
    if save_path:
        kwargs['savefig'] = save_path
    
    # Create plot
    try:
        mpf.plot(df, **kwargs)
        if save_path:
            logger.info(f"Candlestick chart saved to {save_path}")
    except Exception as e:
        logger.error(f"Error creating candlestick chart: {str(e)}")


def plot_seasonal_decomposition(
    data: pd.DataFrame,
    column: str = 'Close',
    period: Optional[int] = None,
    title: str = 'Seasonal Decomposition',
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Plot seasonal decomposition of time series data.
    
    Args:
        data: DataFrame with time series data
        column: Column to decompose
        period: Period for decomposition (e.g., 7 for weekly, 30 for monthly)
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
    """
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
    except ImportError:
        logger.error("statsmodels is required for seasonal decomposition")
        return
    
    # Determine period if not provided
    if period is None:
        # Infer period from data frequency
        if isinstance(data.index, pd.DatetimeIndex):
            if data.index.freq == 'D' or data.index.inferred_freq == 'D':
                period = 7  # Weekly for daily data
            elif data.index.freq == 'H' or data.index.inferred_freq == 'H':
                period = 24  # Daily for hourly data
            elif data.index.freq == 'M' or data.index.inferred_freq == 'M':
                period = 12  # Yearly for monthly data
            else:
                period = 7  # Default to weekly
        else:
            period = 7  # Default
    
    # Check if series is long enough
    if len(data) < 2 * period:
        logger.error(f"Data length ({len(data)}) is too short for period ({period})")
        return
    
    # Get the column data
    if column not in data.columns:
        logger.error(f"Column {column} not found in data")
        return
    
    series = data[column]
    
    # Decompose
    try:
        decomposition = seasonal_decompose(series, model='additive', period=period)
        
        # Plot
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # Original
        decomposition.observed.plot(ax=axes[0], title='Original')
        axes[0].set_ylabel('Value')
        
        # Trend
        decomposition.trend.plot(ax=axes[1], title='Trend')
        axes[1].set_ylabel('Trend')
        
        # Seasonal
        decomposition.seasonal.plot(ax=axes[2], title='Seasonality')
        axes[2].set_ylabel('Seasonality')
        
        # Residual
        decomposition.resid.plot(ax=axes[3], title='Residuals')
        axes[3].set_ylabel('Residuals')
        
        # Set main title
        fig.suptitle(title, fontsize=14)
        
        # Format x-axis
        if isinstance(data.index, pd.DatetimeIndex):
            axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            axes[3].xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Seasonal decomposition plot saved to {save_path}")
        
        plt.show()
    
    except Exception as e:
        logger.error(f"Error performing seasonal decomposition: {str(e)}")


if __name__ == '__main__':
    # Example usage
    import sys
    import os
    sys.path.append(os.path.abspath('../..'))
    
    from crypto_price_predictor.data.data_loader import CryptoDataLoader
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    # Load data
    loader = CryptoDataLoader()
    btc_data = loader.load_data('BTC-USD', interval='1d')
    
    # Create example plots
    plot_price_history(btc_data, columns=['Close'], title='Bitcoin Price History')
    
    # Add some technical indicators
    btc_data['SMA_20'] = btc_data['Close'].rolling(window=20).mean()
    btc_data['SMA_50'] = btc_data['Close'].rolling(window=50).mean()
    btc_data['RSI_14'] = 50 + 10 * np.random.randn(len(btc_data))  # Dummy RSI
    
    # Plot with indicators
    plot_technical_indicators(btc_data)
    
    # Correlation matrix
    cols_to_correlate = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50']
    plot_correlation_matrix(btc_data, columns=cols_to_correlate)
    
    # Candlestick chart
    create_candlestick_chart(btc_data)