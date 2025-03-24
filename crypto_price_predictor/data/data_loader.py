"""
Module for loading cryptocurrency data from various sources.
"""

import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Union, List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CryptoDataLoader:
    """
    Class for loading cryptocurrency data from online sources.
    
    This class provides methods to fetch historical price data for cryptocurrencies
    using the yfinance library, which retrieves data from Yahoo Finance.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the CryptoDataLoader.
        
        Args:
            cache_dir: Optional directory path to cache downloaded data
        """
        self.cache_dir = cache_dir
        self._data_cache = {}  # In-memory cache
        
    def load_data(self, 
                 ticker: str, 
                 start_date: Optional[Union[str, datetime]] = None,
                 end_date: Optional[Union[str, datetime]] = None,
                 interval: str = '1d',
                 include_indicators: bool = False) -> pd.DataFrame:
        """
        Load historical cryptocurrency data.
        
        Args:
            ticker: Cryptocurrency ticker symbol (e.g., 'BTC-USD', 'ETH-USD')
            start_date: Start date for historical data (default: 2 years ago)
            end_date: End date for historical data (default: today)
            interval: Data interval (1d, 1h, etc.)
            include_indicators: Whether to include basic technical indicators
            
        Returns:
            DataFrame containing the historical price data
            
        Raises:
            ValueError: If the ticker is invalid or data cannot be retrieved
        """
        # Set default date range if not provided
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            
        if start_date is None:
            start_date = end_date - timedelta(days=730)  # 2 years
        elif isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        
        # Create cache key
        cache_key = f"{ticker}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}_{interval}"
        
        # Check if data is in cache
        if cache_key in self._data_cache:
            logger.info(f"Using cached data for {ticker}")
            return self._data_cache[cache_key].copy()
        
        # Download data
        try:
            logger.info(f"Downloading data for {ticker} from {start_date} to {end_date}")
            data = yf.download(
                tickers=ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False
            )
            
            if data.empty:
                raise ValueError(f"No data available for {ticker} in the specified date range")
            
            # Validate and clean data
            data = self._validate_and_clean_data(data)
            
            # Add basic indicators if requested
            if include_indicators:
                data = self._add_basic_indicators(data)
                
            # Cache the data
            self._data_cache[cache_key] = data.copy()
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data for {ticker}: {str(e)}")
            raise ValueError(f"Failed to load data for {ticker}: {str(e)}")
    
    def load_multiple(self, 
                     tickers: List[str], 
                     start_date: Optional[Union[str, datetime]] = None,
                     end_date: Optional[Union[str, datetime]] = None,
                     interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple cryptocurrencies.
        
        Args:
            tickers: List of cryptocurrency ticker symbols
            start_date: Start date for historical data
            end_date: End date for historical data
            interval: Data interval
            
        Returns:
            Dictionary mapping ticker symbols to their respective DataFrames
        """
        result = {}
        for ticker in tickers:
            try:
                result[ticker] = self.load_data(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval
                )
            except ValueError as e:
                logger.warning(f"Skipping {ticker}: {str(e)}")
                continue
                
        return result
    
    def _validate_and_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean the downloaded data.
        
        Args:
            data: Raw DataFrame from yfinance
            
        Returns:
            Cleaned DataFrame
        """
        # Check for missing values
        if data.isnull().sum().sum() > 0:
            # Forward fill missing values
            data = data.fillna(method='ffill')
            # If there are still NaN values at the beginning, backward fill
            data = data.fillna(method='bfill')
            
        # Ensure index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # Sort by date
        data = data.sort_index()
        
        return data
    
    def _add_basic_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic technical indicators to the data.
        
        Args:
            data: Price DataFrame
            
        Returns:
            DataFrame with additional technical indicators
        """
        # Simple Moving Averages
        data['SMA_7'] = data['Close'].rolling(window=7).mean()
        data['SMA_30'] = data['Close'].rolling(window=30).mean()
        
        # Exponential Moving Averages
        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
        
        # Trading Volume Moving Average
        data['Volume_MA_20'] = data['Volume'].rolling(window=20).mean()
        
        # Relative Strength Index (RSI) - simplified calculation
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        
        rs = gain / loss
        data['RSI_14'] = 100 - (100 / (1 + rs))
        
        return data


if __name__ == '__main__':
    # Example usage
    loader = CryptoDataLoader()
    btc_data = loader.load_data('BTC-USD', interval='1d', include_indicators=True)
    print(btc_data.head())
    print(f"Loaded {len(btc_data)} days of BTC data")