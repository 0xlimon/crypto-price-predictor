"""
LSTM (Long Short-Term Memory) model for cryptocurrency price prediction.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
from datetime import datetime
import time

# Configure TensorFlow logging
tf.get_logger().setLevel('ERROR')

# Configure module logging
logger = logging.getLogger(__name__)


class LSTMModel:
    """
    LSTM-based model for cryptocurrency price prediction.
    
    This class implements a Long Short-Term Memory (LSTM) neural network
    for time series forecasting of cryptocurrency prices.
    """
    
    def __init__(self, 
                config: Optional[Dict[str, Any]] = None,
                model_path: Optional[str] = None):
        """
        Initialize the LSTM model.
        
        Args:
            config: Model configuration dictionary
            model_path: Path to load a pre-trained model
        """
        # Default configuration
        self.default_config = {
            'lstm_layers': [64, 64],    # Units in each LSTM layer
            'dense_layers': [32],       # Units in each Dense layer
            'dropout_rate': 0.2,        # Dropout rate for regularization
            'learning_rate': 0.001,     # Learning rate for optimization
            'batch_size': 32,           # Batch size for training
            'epochs': 100,              # Maximum epochs for training
            'patience': 15,             # Patience for early stopping
            'forecast_horizon': 1,      # Steps ahead to forecast
            'sequence_length': 60,      # Length of input sequences
            'bidirectional': False,     # Whether to use bidirectional LSTM
            'loss': 'mean_squared_error',  # Loss function
            'metrics': ['mae'],         # Metrics to track
            'save_dir': './models'      # Directory to save models
        }
        
        # Apply provided configuration over default
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
            
        # Initialize model
        self.model = None
        
        # Load model if path provided
        if model_path:
            self.load(model_path)
            
        # Track training history
        self.history = None
        
        # Feature scaler for inverse transformation
        self.target_scaler = None
        
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build the LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
        """
        logger.info(f"Building LSTM model with input shape {input_shape}")
        
        # Create sequential model
        model = Sequential()
        
        # Add LSTM layers
        for i, units in enumerate(self.config['lstm_layers']):
            # First layer needs input shape
            if i == 0:
                if len(self.config['lstm_layers']) == 1:
                    # Single LSTM layer (return sequences=False)
                    model.add(LSTM(
                        units=units,
                        input_shape=input_shape,
                        return_sequences=False,
                        dropout=self.config['dropout_rate']
                    ))
                else:
                    # First layer of multiple (return sequences=True)
                    model.add(LSTM(
                        units=units,
                        input_shape=input_shape,
                        return_sequences=True,
                        dropout=self.config['dropout_rate']
                    ))
            else:
                # Last LSTM layer
                if i == len(self.config['lstm_layers']) - 1:
                    model.add(LSTM(
                        units=units,
                        return_sequences=False,
                        dropout=self.config['dropout_rate']
                    ))
                else:
                    # Middle LSTM layer
                    model.add(LSTM(
                        units=units,
                        return_sequences=True,
                        dropout=self.config['dropout_rate']
                    ))
            
            # Add batch normalization after each LSTM layer
            model.add(BatchNormalization())
        
        # Add Dense layers
        for units in self.config['dense_layers']:
            model.add(Dense(units=units, activation='relu'))
            model.add(Dropout(self.config['dropout_rate']))
        
        # Output layer (single value for regression)
        model.add(Dense(units=self.config['forecast_horizon']))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss=self.config['loss'],
            metrics=self.config['metrics']
        )
        
        # Save model
        self.model = model
        
        # Log model summary
        self.model.summary(print_fn=lambda x: logger.info(x))
        
    def train(self, 
             processed_data: Dict[str, Any],
             validation_split: float = 0.1,
             verbose: int = 1) -> Dict[str, Any]:
        """
        Train the LSTM model.
        
        Args:
            processed_data: Dictionary with processed data
            validation_split: Portion of training data to use for validation
            verbose: Verbosity mode
            
        Returns:
            Training history
        """
        # Extract data
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        # Save target scaler for later inverse transformations
        self.target_scaler = processed_data['target_scaler']
        
        # Build model if it doesn't exist
        if self.model is None:
            self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Create callbacks
        callbacks = self._create_callbacks()
        
        # Train model
        logger.info(f"Training LSTM model with {X_train.shape[0]} samples")
        start_time = time.time()
        
        history = self.model.fit(
            X_train, y_train,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        training_time = time.time() - start_time
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        
        # Save history
        self.history = history.history
        
        return self.history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded")
        
        logger.info(f"Evaluating model on {X_test.shape[0]} test samples")
        
        # Evaluate model
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Create metrics dictionary
        metrics = {}
        for i, metric_name in enumerate(['loss'] + self.config['metrics']):
            metrics[metric_name] = results[i]
            
        logger.info(f"Test metrics: {metrics}")
        
        return metrics
    
    def predict(self, 
               X: np.ndarray, 
               return_scaled: bool = False) -> np.ndarray:
        """
        Make predictions with the model.
        
        Args:
            X: Input features
            return_scaled: Whether to return scaled predictions
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded")
        
        # Make predictions
        scaled_predictions = self.model.predict(X)
        
        # Return scaled or original values
        if return_scaled or self.target_scaler is None:
            return scaled_predictions
        else:
            # Inverse transform to original scale
            return self.target_scaler.inverse_transform(scaled_predictions)
    
    def predict_future(self, 
                      recent_data: Union[pd.DataFrame, np.ndarray],
                      days_ahead: int = 30,
                      plot: bool = False) -> pd.DataFrame:
        """
        Predict future values recursively.
        
        Args:
            recent_data: Most recent data for initial sequence
            days_ahead: Number of days to predict
            plot: Whether to plot the predictions
            
        Returns:
            DataFrame with date index and predicted prices
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded")
        
        # If DataFrame is provided, extract the sequence
        if isinstance(recent_data, pd.DataFrame):
            # Need to process this data similar to training data
            raise NotImplementedError(
                "Future prediction from DataFrame not yet implemented. "
                "Please provide pre-processed sequence data."
            )
        
        # Make a deep copy of the input sequence
        sequence = recent_data.copy()
        
        # Ensure sequence has the correct shape for LSTM input
        # For LSTM, we need shape (samples, time steps, features)
        if len(sequence.shape) == 3 and sequence.shape[0] == 1:
            # Already in shape (1, time_steps, features)
            reshaped_sequence = sequence
        elif len(sequence.shape) == 2:
            # Shape is (time_steps, features), need to add batch dimension
            reshaped_sequence = np.expand_dims(sequence, axis=0)  # Makes (1, time_steps, features)
        else:
            raise ValueError(f"Unexpected sequence shape: {sequence.shape}. Expected (time_steps, features) or (1, time_steps, features)")
        
        future_predictions = []
        current_sequence = reshaped_sequence.copy()
        
        # Predict one day at a time
        for _ in range(days_ahead):
            # Get prediction for next day
            next_pred = self.model.predict(current_sequence)
            future_predictions.append(next_pred[0])
            
            # Update sequence for next prediction - shift the time steps and add new prediction
            # Keeping the batch dimension (1) intact
            if len(current_sequence.shape) == 3:
                # Roll along time steps axis (axis=1)
                new_sequence = np.roll(current_sequence[0], -1, axis=0)
                # Set the last time step to be the new prediction
                new_sequence[-1] = next_pred
                # Add batch dimension back
                current_sequence = np.expand_dims(new_sequence, axis=0)
            else:
                raise ValueError(f"Unexpected sequence shape after prediction: {current_sequence.shape}")
        
        # Convert predictions to original scale
        if self.target_scaler:
            future_predictions = self.target_scaler.inverse_transform(
                np.array(future_predictions).reshape(-1, 1)
            ).flatten()
        else:
            future_predictions = np.array(future_predictions).flatten()
        
        # Create date index for future predictions
        last_date = datetime.now()  # Should ideally be based on input data
        future_dates = pd.date_range(
            start=last_date, periods=days_ahead+1, freq='D'
        )[1:]  # Skip today
        
        # Create DataFrame with predictions
        predictions_df = pd.DataFrame({
            'Predicted_Price': future_predictions
        }, index=future_dates)
        
        # Plot if requested
        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(predictions_df.index, predictions_df['Predicted_Price'])
            plt.title('Future Price Predictions')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.grid(True)
            plt.show()
        
        return predictions_df
    
    def _create_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """
        Create callbacks for model training.
        
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config['patience'],
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001
        )
        callbacks.append(reduce_lr)
        
        # Model checkpoint
        os.makedirs(self.config['save_dir'], exist_ok=True)
        checkpoint_path = os.path.join(
            self.config['save_dir'],
            f"lstm_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        )
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True
        )
        callbacks.append(checkpoint)
        
        return callbacks
    
    def save(self, path: Optional[str] = None) -> str:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
            
        Returns:
            Path where model was saved
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        if path is None:
            # Create default path
            os.makedirs(self.config['save_dir'], exist_ok=True)
            path = os.path.join(
                self.config['save_dir'],
                f"lstm_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
            )
        else:
            # Ensure path has a valid extension (.keras or .h5)
            if not path.endswith('.keras') and not path.endswith('.h5'):
                path = f"{path}.h5"
        
        self.model.save(path)
        logger.info(f"Model saved to {path}")
        
        return path
    
    def load(self, path: str) -> None:
        """
        Load a saved model.
        
        Args:
            path: Path to the saved model
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        self.model = load_model(path)
        logger.info(f"Model loaded from {path}")
    
    def plot_training_history(self) -> None:
        """
        Plot the training history.
        """
        if self.history is None:
            raise ValueError("No training history available")
        
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['loss'], label='Training Loss')
        if 'val_loss' in self.history:
            plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        
        # Plot metrics
        if 'mae' in self.history:
            plt.subplot(1, 2, 2)
            plt.plot(self.history['mae'], label='Training MAE')
            if 'val_mae' in self.history:
                plt.plot(self.history['val_mae'], label='Validation MAE')
            plt.title('Mean Absolute Error')
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # Example usage
    import sys
    import os
    sys.path.append(os.path.abspath('../..'))
    
    from crypto_price_predictor.data.data_loader import CryptoDataLoader
    from crypto_price_predictor.data.data_processor import DataProcessor
    
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
    model_config = {
        'lstm_layers': [64, 32],
        'epochs': 50,
        'batch_size': 32
    }
    
    model = LSTMModel(config=model_config)
    
    # Train model
    history = model.train(processed_data)
    
    # Evaluate model
    results = model.evaluate(processed_data['X_test'], processed_data['y_test'])
    
    # Make predictions
    predictions = model.predict(processed_data['X_test'])
    
    # Save model
    model.save()