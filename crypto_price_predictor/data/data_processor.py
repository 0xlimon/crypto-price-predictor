"""
Module for preprocessing cryptocurrency data for model training.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Class for preprocessing cryptocurrency data for time series forecasting.
    
    This class handles operations like normalization, sequence creation,
    train-test splitting, and feature engineering for time series data.
    """
    
    def __init__(self, 
                scale_method: str = 'minmax',
                test_size: float = 0.2,
                sequence_length: int = 60,
                forecast_horizon: int = 1,
                feature_columns: Optional[List[str]] = None):
        """
        Initialize the DataProcessor.
        
        Args:
            scale_method: Scaling method ('minmax' or 'standard')
            test_size: Proportion of data to use for testing
            sequence_length: Number of time steps for sequence creation
            forecast_horizon: Number of future steps to predict
            feature_columns: Columns to use as features (default: ['Close'])
        """
        self.scale_method = scale_method
        self.test_size = test_size
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.feature_columns = feature_columns or ['Close']
        
        # Initialize scalers
        self.feature_scaler = None
        self.target_scaler = None
        self._initialize_scalers()
    
    def _initialize_scalers(self):
        """Initialize the feature and target scalers based on scale_method."""
        if self.scale_method == 'minmax':
            self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
            self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        elif self.scale_method == 'standard':
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scale_method}")
    
    def process_data(self, data: pd.DataFrame, target_column: str = 'Close') -> Dict:
        """
        Process the data for model training and testing.
        
        Args:
            data: DataFrame containing cryptocurrency data
            target_column: Column to predict
            
        Returns:
            Dictionary containing processed data:
                - X_train: Training features
                - y_train: Training targets
                - X_test: Testing features
                - y_test: Testing targets
                - feature_scaler: Fitted feature scaler
                - target_scaler: Fitted target scaler
        """
        logger.info("Processing data for model training and testing")
        
        # Validate data
        self._validate_data(data)
        
        # Prepare features and target
        feature_data, target_data = self._prepare_features_and_target(data, target_column)
        
        # Scale the data
        scaled_features, scaled_target = self._scale_data(feature_data, target_data)
        
        # Create sequences
        X, y = self._create_sequences(scaled_features, scaled_target)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = self._train_test_split(X, y)
        
        # Return processed data
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler
        }
    
    def _validate_data(self, data: pd.DataFrame):
        """
        Validate the input data.
        
        Args:
            data: DataFrame to validate
            
        Raises:
            ValueError: If data is invalid
        """
        if data.empty:
            raise ValueError("Input data is empty")
        
        # Check if feature columns exist in the data
        missing_columns = [col for col in self.feature_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in data: {missing_columns}")
        
        # Check for NaN values
        if data[self.feature_columns].isnull().any().any():
            raise ValueError("Data contains NaN values")
    
    def _prepare_features_and_target(self, 
                                    data: pd.DataFrame, 
                                    target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare feature and target arrays.
        
        Args:
            data: Input DataFrame
            target_column: Column to use as target
            
        Returns:
            Tuple of (feature_data, target_data)
        """
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Extract features
        feature_data = data[self.feature_columns].values
        
        # Extract target (can be same as a feature)
        target_data = data[[target_column]].values
        
        return feature_data, target_data
    
    def _scale_data(self, 
                   feature_data: np.ndarray, 
                   target_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale feature and target data.
        
        Args:
            feature_data: Feature array
            target_data: Target array
            
        Returns:
            Tuple of (scaled_features, scaled_target)
        """
        # Fit and transform features
        scaled_features = self.feature_scaler.fit_transform(feature_data)
        
        # Fit and transform target
        scaled_target = self.target_scaler.fit_transform(target_data)
        
        return scaled_features, scaled_target
    
    def _create_sequences(self, 
                         scaled_features: np.ndarray, 
                         scaled_target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series forecasting.
        
        Args:
            scaled_features: Scaled feature array
            scaled_target: Scaled target array
            
        Returns:
            Tuple of (X, y) where X contains sequences and y contains targets
        """
        X, y = [], []
        
        # Total number of time steps to use
        total_steps = len(scaled_features) - self.sequence_length - self.forecast_horizon + 1
        
        for i in range(total_steps):
            # Add sequence of features
            X.append(scaled_features[i:i+self.sequence_length])
            
            # Add target (next value after sequence)
            target_idx = i + self.sequence_length + self.forecast_horizon - 1
            y.append(scaled_target[target_idx])
        
        return np.array(X), np.array(y)
    
    def _train_test_split(self, 
                         X: np.ndarray, 
                         y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature sequences
            y: Target values
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Determine split index
        split_idx = int(len(X) * (1 - self.test_size))
        
        # Split the data
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")
        
        return X_train, X_test, y_train, y_test
    
    def inverse_transform_predictions(self, 
                                     scaled_predictions: np.ndarray) -> np.ndarray:
        """
        Transform scaled predictions back to original scale.
        
        Args:
            scaled_predictions: Array of scaled predictions
            
        Returns:
            Array of predictions in original scale
        """
        # Reshape if needed
        if scaled_predictions.ndim == 1:
            scaled_predictions = scaled_predictions.reshape(-1, 1)
            
        # Inverse transform
        return self.target_scaler.inverse_transform(scaled_predictions)
    
    def create_future_sequences(self, 
                              data: pd.DataFrame, 
                              n_future_steps: int) -> np.ndarray:
        """
        Create sequences for future predictions.
        
        Args:
            data: Most recent data
            n_future_steps: Number of future steps to predict
            
        Returns:
            Array of feature sequences for future predictions
        """
        # Extract the most recent data points
        recent_data = data.tail(self.sequence_length)
        
        # Extract features
        feature_data = recent_data[self.feature_columns].values
        
        # Scale data
        scaled_features = self.feature_scaler.transform(feature_data)
        
        # Create sequence
        future_sequence = scaled_features[-self.sequence_length:]
        future_sequence = future_sequence.reshape(1, self.sequence_length, len(self.feature_columns))
        
        return future_sequence


if __name__ == '__main__':
    # Example usage
    from data_loader import CryptoDataLoader
    
    # Load data
    loader = CryptoDataLoader()
    btc_data = loader.load_data('BTC-USD', interval='1d')
    
    # Process data
    processor = DataProcessor(
        scale_method='minmax',
        sequence_length=30,
        forecast_horizon=1,
        feature_columns=['Close', 'Volume', 'High', 'Low']
    )
    
    processed_data = processor.process_data(btc_data, target_column='Close')
    
    print(f"X_train shape: {processed_data['X_train'].shape}")
    print(f"y_train shape: {processed_data['y_train'].shape}")