"""
Module for creating advanced features for cryptocurrency price prediction.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union, Dict
import logging
from scipy import stats
import ta  # Technical analysis library

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Class for creating advanced features from cryptocurrency data.
    
    This class provides methods to generate technical indicators,
    volatility metrics, trend features, and more to improve
    prediction model performance.
    """
    
    def __init__(self, feature_groups: Optional[List[str]] = None):
        """
        Initialize the FeatureEngineer.
        
        Args:
            feature_groups: List of feature groups to use
                Available groups: 'trend', 'momentum', 'volatility', 
                'volume', 'pattern', 'custom'
        """
        self.feature_groups = feature_groups or [
            'trend', 'momentum', 'volatility', 'volume'
        ]
        
        # Track created features
        self.created_features = []
    
    def _sanitize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Replace infinity values with NaN and handle extremely large values.
        
        Args:
            data: DataFrame with features
            
        Returns:
            DataFrame with sanitized features
        """
        # Replace infinity values with NaN
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # Find columns with extremely large values and cap them
        for col in data.columns:
            if data[col].dtype in [np.float64, np.float32]:
                # Cap extremely large values (beyond float64 safe range)
                max_safe = 1e+300  # Well below float64 max but still very large
                min_safe = -1e+300
                data[col] = data[col].clip(min_safe, max_safe)
        
        return data
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features based on the selected feature groups.
        
        Args:
            data: DataFrame with OHLCV price data
            
        Returns:
            DataFrame with added features
        """
        logger.info(f"Engineering features for groups: {self.feature_groups}")
        
        # Reset feature list
        self.created_features = []
        
        # Create a copy to avoid modifying the original
        df = data.copy()
        
        # Validate and prepare data
        df = self._validate_and_prepare_data(df)
        
        # Apply feature engineering based on selected groups
        if 'trend' in self.feature_groups:
            df = self._add_trend_features(df)
            df = self._sanitize_features(df)
        
        if 'momentum' in self.feature_groups:
            df = self._add_momentum_features(df)
            df = self._sanitize_features(df)
        
        if 'volatility' in self.feature_groups:
            df = self._add_volatility_features(df)
            df = self._sanitize_features(df)
        
        if 'volume' in self.feature_groups:
            df = self._add_volume_features(df)
            df = self._sanitize_features(df)
        
        if 'pattern' in self.feature_groups:
            df = self._add_pattern_features(df)
            df = self._sanitize_features(df)
        
        if 'custom' in self.feature_groups:
            df = self._add_custom_crypto_features(df)
            df = self._sanitize_features(df)
        
        # Final sanitization before returning
        df = self._sanitize_features(df)
        
        # Drop NaN values created by indicators that use windows
        df = df.dropna()
        
        logger.info(f"Created {len(self.created_features)} new features")
        
        return df
    
    def _validate_and_prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and prepare data for feature engineering.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Prepared DataFrame
        """
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except Exception as e:
                raise ValueError(f"Cannot convert index to datetime: {str(e)}")
        
        # Sort by date
        data = data.sort_index()
        
        return data
    
    def _add_trend_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend-based technical indicators.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with added trend features
        """
        # Ensure Close is 1D for TA library and convert to pandas Series if needed
        if hasattr(data['Close'], 'values') and len(data['Close'].values.shape) > 1:
            close_values = data['Close'].values.flatten()
            close_series = pd.Series(close_values, index=data.index)
        else:
            close_series = data['Close']
        
        # Simple Moving Averages
        for window in [7, 21, 50, 200]:
            data[f'SMA_{window}'] = ta.trend.sma_indicator(close_series, window=window)
            self.created_features.append(f'SMA_{window}')
        
        # Exponential Moving Averages
        for window in [9, 21, 50]:
            data[f'EMA_{window}'] = ta.trend.ema_indicator(close_series, window=window)
            self.created_features.append(f'EMA_{window}')
        
        # MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(close_series)
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        data['MACD_Diff'] = macd.macd_diff()
        self.created_features.extend(['MACD', 'MACD_Signal', 'MACD_Diff'])
        
        # Parabolic SAR
        # Ensure High, Low are also 1D and convert to pandas Series if needed
        if hasattr(data['High'], 'values') and len(data['High'].values.shape) > 1:
            high_values = data['High'].values.flatten()
            high_series = pd.Series(high_values, index=data.index)
        else:
            high_series = data['High']
            
        if hasattr(data['Low'], 'values') and len(data['Low'].values.shape) > 1:
            low_values = data['Low'].values.flatten()
            low_series = pd.Series(low_values, index=data.index)
        else:
            low_series = data['Low']
        
        data['SAR'] = ta.trend.PSARIndicator(
            high_series, low_series, close_series
        ).psar()
        self.created_features.append('SAR')
        
        # Moving Average Crossovers (as binary signals)
        data['SMA_50_200_Cross'] = np.where(
            data['SMA_50'] > data['SMA_200'], 1, 0
        )
        self.created_features.append('SMA_50_200_Cross')
        
        # ADX (Average Directional Index)
        adx = ta.trend.ADXIndicator(high_series, low_series, close_series)
        data['ADX'] = adx.adx()
        data['DI_plus'] = adx.adx_pos()
        data['DI_minus'] = adx.adx_neg()
        self.created_features.extend(['ADX', 'DI_plus', 'DI_minus'])
        
        # Ichimoku Cloud
        ichimoku = ta.trend.IchimokuIndicator(high_series, low_series)
        data['Ichimoku_A'] = ichimoku.ichimoku_a()
        data['Ichimoku_B'] = ichimoku.ichimoku_b()
        self.created_features.extend(['Ichimoku_A', 'Ichimoku_B'])
        
        return data
    
    def _add_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum-based technical indicators.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with added momentum features
        """
        # Ensure price data is 1D and convert to pandas Series
        if hasattr(data['Close'], 'values') and len(data['Close'].values.shape) > 1:
            close_values = data['Close'].values.flatten()
            close_series = pd.Series(close_values, index=data.index)
        else:
            close_series = data['Close']
            
        if hasattr(data['High'], 'values') and len(data['High'].values.shape) > 1:
            high_values = data['High'].values.flatten()
            high_series = pd.Series(high_values, index=data.index)
        else:
            high_series = data['High']
            
        if hasattr(data['Low'], 'values') and len(data['Low'].values.shape) > 1:
            low_values = data['Low'].values.flatten()
            low_series = pd.Series(low_values, index=data.index)
        else:
            low_series = data['Low']
        
        # RSI (Relative Strength Index)
        for window in [7, 14, 21]:
            data[f'RSI_{window}'] = ta.momentum.RSIIndicator(
                close_series, window=window
            ).rsi()
            self.created_features.append(f'RSI_{window}')
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            high_series, low_series, close_series
        )
        data['Stoch_k'] = stoch.stoch()
        data['Stoch_d'] = stoch.stoch_signal()
        self.created_features.extend(['Stoch_k', 'Stoch_d'])
        
        # Rate of Change
        for window in [7, 14, 21]:
            data[f'ROC_{window}'] = ta.momentum.ROCIndicator(
                close_series, window=window
            ).roc()
            self.created_features.append(f'ROC_{window}')
        
        # William's %R
        for window in [14, 21]:
            data[f'Williams_R_{window}'] = ta.momentum.WilliamsRIndicator(
                high_series, low_series, close_series, lbp=window
            ).williams_r()
            self.created_features.append(f'Williams_R_{window}')
        
        # Awesome Oscillator
        data['Awesome_Osc'] = ta.momentum.AwesomeOscillatorIndicator(
            high_series, low_series
        ).awesome_oscillator()
        self.created_features.append('Awesome_Osc')
        
        # Price momentum as percent change
        for window in [1, 3, 7, 14]:
            data[f'Price_Pct_Change_{window}d'] = data['Close'].pct_change(window)
            self.created_features.append(f'Price_Pct_Change_{window}d')
        
        return data
    
    def _add_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility-based technical indicators.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with added volatility features
        """
        # Ensure price data is 1D and convert to pandas Series
        if hasattr(data['Close'], 'values') and len(data['Close'].values.shape) > 1:
            close_values = data['Close'].values.flatten()
            close_series = pd.Series(close_values, index=data.index)
        else:
            close_series = data['Close']
            
        if hasattr(data['High'], 'values') and len(data['High'].values.shape) > 1:
            high_values = data['High'].values.flatten()
            high_series = pd.Series(high_values, index=data.index)
        else:
            high_series = data['High']
            
        if hasattr(data['Low'], 'values') and len(data['Low'].values.shape) > 1:
            low_values = data['Low'].values.flatten()
            low_series = pd.Series(low_values, index=data.index)
        else:
            low_series = data['Low']
        
        # Bollinger Bands
        for window in [20, 30]:
            bb = ta.volatility.BollingerBands(
                close_series, window=window, window_dev=2
            )
            data[f'BB_High_{window}'] = bb.bollinger_hband()
            data[f'BB_Low_{window}'] = bb.bollinger_lband()
            data[f'BB_Width_{window}'] = bb.bollinger_wband()
            data[f'BB_Pct_{window}'] = bb.bollinger_pband()
            self.created_features.extend([
                f'BB_High_{window}', f'BB_Low_{window}', 
                f'BB_Width_{window}', f'BB_Pct_{window}'
            ])
        
        # Average True Range
        for window in [14, 21]:
            data[f'ATR_{window}'] = ta.volatility.AverageTrueRange(
                high_series, low_series, close_series, window=window
            ).average_true_range()
            self.created_features.append(f'ATR_{window}')
        
        # Keltner Channel
        kc = ta.volatility.KeltnerChannel(
            high_series, low_series, close_series
        )
        data['KC_High'] = kc.keltner_channel_hband()
        data['KC_Low'] = kc.keltner_channel_lband()
        data['KC_Width'] = data['KC_High'] - data['KC_Low']
        self.created_features.extend(['KC_High', 'KC_Low', 'KC_Width'])
        
        # Historical Volatility (standard deviation of returns)
        for window in [7, 14, 30]:
            data[f'Hist_Vol_{window}d'] = data['Close'].pct_change().rolling(window).std()
            self.created_features.append(f'Hist_Vol_{window}d')
        
        # High-Low Range as percentage of close
        data['HL_Range_Pct'] = (data['High'] - data['Low']) / data['Close']
        self.created_features.append('HL_Range_Pct')
        
        return data
    
    def _add_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based technical indicators.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with added volume features
        """
        # Ensure price data is 1D and convert to pandas Series
        if hasattr(data['Close'], 'values') and len(data['Close'].values.shape) > 1:
            close_values = data['Close'].values.flatten()
            close_series = pd.Series(close_values, index=data.index)
        else:
            close_series = data['Close']
            
        if hasattr(data['High'], 'values') and len(data['High'].values.shape) > 1:
            high_values = data['High'].values.flatten()
            high_series = pd.Series(high_values, index=data.index)
        else:
            high_series = data['High']
            
        if hasattr(data['Low'], 'values') and len(data['Low'].values.shape) > 1:
            low_values = data['Low'].values.flatten()
            low_series = pd.Series(low_values, index=data.index)
        else:
            low_series = data['Low']
            
        if hasattr(data['Volume'], 'values') and len(data['Volume'].values.shape) > 1:
            volume_values = data['Volume'].values.flatten()
            volume_series = pd.Series(volume_values, index=data.index)
        else:
            volume_series = data['Volume']
        
        # Volume Moving Averages (calculated manually since volume_sma_indicator doesn't exist)
        for window in [7, 14, 30]:
            data[f'Volume_SMA_{window}'] = volume_series.rolling(window=window).mean()
            self.created_features.append(f'Volume_SMA_{window}')
        
        # Accumulation/Distribution Line
        data['ADL'] = ta.volume.acc_dist_index(
            high_series, low_series, close_series, volume_series
        )
        self.created_features.append('ADL')
        
        # On-Balance Volume
        data['OBV'] = ta.volume.OnBalanceVolumeIndicator(
            close_series, volume_series
        ).on_balance_volume()
        self.created_features.append('OBV')
        
        # Chaikin Money Flow
        for window in [20, 30]:
            data[f'CMF_{window}'] = ta.volume.ChaikinMoneyFlowIndicator(
                high_series, low_series, close_series, volume_series, window=window
            ).chaikin_money_flow()
            self.created_features.append(f'CMF_{window}')
        
        # Volume / Price Change Ratio (with safeguard against division by zero)
        epsilon = 1e-10  # Small value to prevent division by zero
        data['Vol_Price_Ratio'] = data['Volume'] / ((data['High'] - data['Low']) + epsilon)
        self.created_features.append('Vol_Price_Ratio')
        
        # Daily volume change percentage
        data['Volume_Pct_Change_1d'] = data['Volume'].pct_change()
        self.created_features.append('Volume_Pct_Change_1d')
        
        # Normalized volume (z-score over rolling window)
        for window in [30, 90]:
            rolling_mean = data['Volume'].rolling(window=window).mean()
            rolling_std = data['Volume'].rolling(window=window).std()
            data[f'Volume_Z_{window}'] = (data['Volume'] - rolling_mean) / rolling_std
            self.created_features.append(f'Volume_Z_{window}')
        
        return data
    
    def _add_pattern_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add pattern-based technical indicators.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with added pattern features
        """
        # Doji pattern (close near open)
        doji_threshold = 0.001
        data['Doji'] = ((data['Close'] - data['Open']).abs() / data['Open']) < doji_threshold
        data['Doji'] = data['Doji'].astype(int)
        self.created_features.append('Doji')
        
        # Price gaps (difference between current open and previous close)
        data['Gap_Up'] = (data['Open'] > data['Close'].shift(1)).astype(int)
        data['Gap_Down'] = (data['Open'] < data['Close'].shift(1)).astype(int)
        self.created_features.extend(['Gap_Up', 'Gap_Down'])
        
        # Trend reversals based on moving average crosses
        data['Trend_Reversal_Up'] = ((data['SMA_7'] > data['SMA_21']) & 
                                    (data['SMA_7'].shift(1) <= data['SMA_21'].shift(1))).astype(int)
        data['Trend_Reversal_Down'] = ((data['SMA_7'] < data['SMA_21']) & 
                                      (data['SMA_7'].shift(1) >= data['SMA_21'].shift(1))).astype(int)
        self.created_features.extend(['Trend_Reversal_Up', 'Trend_Reversal_Down'])
        
        # Daily candle classification
        data['Bullish_Candle'] = (data['Close'] > data['Open']).astype(int)
        data['Bearish_Candle'] = (data['Close'] < data['Open']).astype(int)
        self.created_features.extend(['Bullish_Candle', 'Bearish_Candle'])
        
        # Consecutive up/down days
        for n in [2, 3]:
            # n consecutive up days
            condition = True
            for i in range(n):
                condition = condition & (data['Close'].shift(i) > data['Open'].shift(i))
            data[f'Consec_{n}_Up'] = condition.astype(int)
            
            # n consecutive down days
            condition = True
            for i in range(n):
                condition = condition & (data['Close'].shift(i) < data['Open'].shift(i))
            data[f'Consec_{n}_Down'] = condition.astype(int)
            
            self.created_features.extend([f'Consec_{n}_Up', f'Consec_{n}_Down'])
        
        return data
    
    def _add_custom_crypto_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add cryptocurrency-specific custom features.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with added custom features
        """
        # Daily log returns (with safeguard against zero or negative values)
        epsilon = 1e-10  # Small value to prevent log of zero or negative values
        data['Log_Return'] = np.log((data['Close'] / data['Close'].shift(1)).clip(epsilon, None))
        self.created_features.append('Log_Return')
        
        # Volatility ratio (High-Low range / Previous day's close)
        epsilon = 1e-10  # Small value to prevent division by zero
        data['Volatility_Ratio'] = (data['High'] - data['Low']) / (data['Close'].shift(1) + epsilon)
        self.created_features.append('Volatility_Ratio')
        
        # Temporal features (day of week, month)
        data['Day_of_Week'] = data.index.dayofweek
        data['Month'] = data.index.month
        data['Quarter'] = data.index.quarter
        data['Day_of_Month'] = data.index.day
        data['Week_of_Year'] = data.index.isocalendar().week
        self.created_features.extend(['Day_of_Week', 'Month', 'Quarter', 'Day_of_Month', 'Week_of_Year'])
        
        # Weekend effect (one-hot encoding for weekends)
        data['Is_Weekend'] = (data.index.dayofweek >= 5).astype(int)
        self.created_features.append('Is_Weekend')
        
        # Market regime features
        # - Bull market (price above long-term MA)
        # - Bear market (price below long-term MA)
        data['Bull_Market'] = (data['Close'] > data['SMA_200']).astype(int)
        data['Bear_Market'] = (data['Close'] < data['SMA_200']).astype(int)
        self.created_features.extend(['Bull_Market', 'Bear_Market'])
        
        # Volatility regime (based on ATR percentile)
        atr_percentile = (
            data['ATR_14'].rolling(window=90)
            .apply(lambda x: stats.percentileofscore(x, x.iloc[-1]))
        )
        data['High_Volatility_Regime'] = (atr_percentile > 80).astype(int)
        data['Low_Volatility_Regime'] = (atr_percentile < 20).astype(int)
        self.created_features.extend(['High_Volatility_Regime', 'Low_Volatility_Regime'])
        
        # Price distance from moving averages (as percentage)
        for ma in ['SMA_50', 'SMA_200']:
            epsilon = 1e-10  # Small value to prevent division by zero
            data[f'Dist_From_{ma}_Pct'] = (data['Close'] - data[ma]) / (data[ma] + epsilon) * 100
            self.created_features.append(f'Dist_From_{ma}_Pct')
        
        return data
    
    def select_important_features(self, 
                                 data: pd.DataFrame, 
                                 target_column: str,
                                 n_features: int = 20,
                                 method: str = 'correlation') -> List[str]:
        """
        Select most important features based on correlation or other methods.
        
        Args:
            data: DataFrame with features and target
            target_column: Target column name
            n_features: Number of features to select
            method: Feature selection method
            
        Returns:
            List of selected feature names
        """
        if method == 'correlation':
            # Calculate correlation with target
            correlations = data.corr()[target_column].abs().sort_values(ascending=False)
            # Remove target itself
            correlations = correlations.drop(target_column)
            # Get top n_features
            selected = correlations.head(n_features).index.tolist()
            
            return selected
        
        # Add other methods as needed (mutual information, feature importance from models, etc.)
        
        logger.warning(f"Unknown feature selection method: {method}")
        return list(data.columns)


if __name__ == '__main__':
    # Example usage
    import sys
    import os
    sys.path.append(os.path.abspath('..'))
    
    from crypto_price_predictor.data.data_loader import CryptoDataLoader
    
    # Load data
    loader = CryptoDataLoader()
    btc_data = loader.load_data('BTC-USD', interval='1d')
    
    # Create features
    engineer = FeatureEngineer(feature_groups=['trend', 'momentum', 'volatility', 'volume'])
    
    # Add features
    enhanced_data = engineer.engineer_features(btc_data)
    
    print(f"Original data shape: {btc_data.shape}")
    print(f"Enhanced data shape: {enhanced_data.shape}")
    print(f"New features: {engineer.created_features}")
    
    # Select important features
    important_features = engineer.select_important_features(
        enhanced_data, 'Close', n_features=15
    )
    print(f"Top 15 important features: {important_features}")