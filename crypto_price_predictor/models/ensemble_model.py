"""
Ensemble model for cryptocurrency price prediction.

This module implements an ensemble approach that combines multiple models
to improve prediction accuracy and reliability.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import os
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

from .lstm_model import LSTMModel

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Ensemble model for cryptocurrency price prediction.
    
    This class implements various ensemble techniques that combine multiple
    machine learning models to create more accurate and robust predictions.
    """
    
    def __init__(self, 
                ensemble_type: str = 'stacking',
                base_models: Optional[List[str]] = None,
                config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ensemble model.
        
        Args:
            ensemble_type: Type of ensemble ('voting', 'stacking', or 'boosting')
            base_models: List of model names to include in the ensemble
            config: Configuration for models and ensemble
        """
        # Default configuration
        self.default_config = {
            'voting_weights': None,          # Weights for voting ensemble
            'cv_folds': 5,                   # Number of cross-validation folds
            'model_params': {                # Parameters for base models
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'random_state': 42
                },
                'gradient_boosting': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 3,
                    'random_state': 42
                },
                'xgboost': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 3,
                    'random_state': 42
                },
                'svr': {
                    'kernel': 'rbf',
                    'C': 1.0,
                    'epsilon': 0.1
                },
                'lstm': {
                    'lstm_layers': [64, 32],
                    'dense_layers': [16],
                    'dropout_rate': 0.2,
                    'epochs': 50,
                    'batch_size': 32
                }
            },
            'meta_model': 'linear',          # Meta-model for stacking
            'save_dir': './models'           # Directory to save models
        }
        
        # Apply provided configuration over default
        self.config = self.default_config.copy()
        if config:
            # Deep update for nested dictionaries
            for key, value in config.items():
                if (key in self.config and isinstance(self.config[key], dict) 
                    and isinstance(value, dict)):
                    self.config[key].update(value)
                else:
                    self.config[key] = value
        
        # Validate ensemble type
        valid_types = ['voting', 'stacking', 'boosting']
        if ensemble_type not in valid_types:
            raise ValueError(f"Ensemble type must be one of {valid_types}")
        self.ensemble_type = ensemble_type
        
        # Set base models
        self.available_models = {
            'random_forest': self._create_random_forest,
            'gradient_boosting': self._create_gradient_boosting,
            'xgboost': self._create_xgboost,
            'svr': self._create_svr,
            'linear': self._create_linear,
            'ridge': self._create_ridge,
            'lasso': self._create_lasso,
            'elastic_net': self._create_elastic_net,
            'mlp': self._create_mlp,
            'lstm': self._create_lstm
        }
        
        # Default base models if none provided
        if not base_models:
            if ensemble_type == 'voting':
                base_models = ['random_forest', 'gradient_boosting', 'xgboost']
            elif ensemble_type == 'stacking':
                base_models = ['random_forest', 'gradient_boosting', 'xgboost', 'ridge']
            else:  # boosting
                base_models = ['gradient_boosting']
        
        # Validate base models
        for model_name in base_models:
            if model_name not in self.available_models:
                raise ValueError(f"Unknown model: {model_name}. Available models: {list(self.available_models.keys())}")
        
        self.base_model_names = base_models
        
        # Initialize model collections
        self.base_models = []
        self.meta_model = None
        
        # Track training data and predictions
        self.feature_scaler = None
        self.target_scaler = None
        self.base_predictions = None
        self.history = {}
    
    def train(self, 
             processed_data: Dict[str, Any],
             validation_split: float = 0.1,
             verbose: int = 1) -> Dict[str, Any]:
        """
        Train the ensemble model.
        
        Args:
            processed_data: Dictionary with processed data
            validation_split: Proportion of training data to use for validation
            verbose: Verbosity level
            
        Returns:
            Training history dictionary
        """
        # Extract data
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        
        # Save scalers for later inversions
        self.feature_scaler = processed_data['feature_scaler']
        self.target_scaler = processed_data['target_scaler']
        
        # Reshape data if needed for traditional ML models
        # LSTM models expect 3D data, others expect 2D
        if X_train.ndim == 3:
            X_train_2d = X_train.reshape(X_train.shape[0], -1)
            X_test_2d = X_test.reshape(X_test.shape[0], -1)
        else:
            X_train_2d = X_train
            X_test_2d = X_test
        
        # Train based on ensemble type
        if self.ensemble_type == 'voting':
            return self._train_voting_ensemble(
                X_train_2d, y_train, X_test_2d, y_test, X_train, X_test, verbose
            )
        elif self.ensemble_type == 'stacking':
            return self._train_stacking_ensemble(
                X_train_2d, y_train, X_test_2d, y_test, X_train, X_test, verbose
            )
        else:  # boosting
            return self._train_boosting_ensemble(
                X_train_2d, y_train, X_test_2d, y_test, verbose
            )
    
    def _train_voting_ensemble(self, 
                              X_train_2d: np.ndarray,
                              y_train: np.ndarray,
                              X_test_2d: np.ndarray,
                              y_test: np.ndarray,
                              X_train_3d: np.ndarray,
                              X_test_3d: np.ndarray,
                              verbose: int) -> Dict[str, Any]:
        """
        Train a voting ensemble.
        
        Args:
            X_train_2d: 2D training features
            y_train: Training targets
            X_test_2d: 2D test features
            y_test: Test targets
            X_train_3d: 3D training features (for LSTM)
            X_test_3d: 3D test features (for LSTM)
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        logger.info(f"Training voting ensemble with {len(self.base_model_names)} base models")
        
        # Create and train base models
        test_predictions = []
        model_metrics = {}
        weights = self.config['voting_weights']
        
        # If weights provided, validate length
        if weights and len(weights) != len(self.base_model_names):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({len(self.base_model_names)})")
        
        # Use equal weights if not specified
        if not weights:
            weights = [1.0 / len(self.base_model_names)] * len(self.base_model_names)
        
        # Normalize weights to sum to 1
        weights = np.array(weights) / sum(weights)
        
        # Train each base model
        for i, model_name in enumerate(self.base_model_names):
            logger.info(f"Training base model: {model_name}")
            
            # Get model creation function
            model_creator = self.available_models[model_name]
            
            # Create and train model (handle LSTM specially)
            if model_name == 'lstm':
                model = model_creator()
                model_hist = model.train({
                    'X_train': X_train_3d,
                    'y_train': y_train,
                    'target_scaler': self.target_scaler
                }, validation_split=0.1)
                
                # Get predictions on test set
                preds = model.predict(X_test_3d, return_scaled=True)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, preds)
                mse = mean_squared_error(y_test, preds)
                r2 = r2_score(y_test, preds)
            else:
                # For traditional ML models
                model = model_creator()
                model.fit(X_train_2d, y_train)
                
                # Get predictions on test set
                preds = model.predict(X_test_2d).reshape(-1, 1)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, preds)
                mse = mean_squared_error(y_test, preds)
                r2 = r2_score(y_test, preds)
                
                # Store model history (empty for non-LSTM models)
                model_hist = {}
            
            # Save model metrics
            model_metrics[model_name] = {
                'mae': mae,
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2,
                'weight': weights[i]
            }
            
            # Add model and predictions
            self.base_models.append(model)
            test_predictions.append(preds)
            
            logger.info(f"{model_name} metrics - MAE: {mae:.4f}, RMSE: {np.sqrt(mse):.4f}")
        
        # Combine predictions using weighted average
        self.base_predictions = test_predictions
        ensemble_preds = np.zeros_like(test_predictions[0])
        for i, preds in enumerate(test_predictions):
            ensemble_preds += weights[i] * preds
        
        # Calculate ensemble metrics
        ensemble_mae = mean_absolute_error(y_test, ensemble_preds)
        ensemble_mse = mean_squared_error(y_test, ensemble_preds)
        ensemble_r2 = r2_score(y_test, ensemble_preds)
        
        logger.info(f"Ensemble metrics - MAE: {ensemble_mae:.4f}, RMSE: {np.sqrt(ensemble_mse):.4f}")
        
        # Save history
        self.history = {
            'ensemble_metrics': {
                'mae': ensemble_mae,
                'mse': ensemble_mse,
                'rmse': np.sqrt(ensemble_mse),
                'r2': ensemble_r2
            },
            'model_metrics': model_metrics,
            'training_history': {name: {} for name in self.base_model_names}
        }
        
        # Add LSTM training history if available
        for i, model_name in enumerate(self.base_model_names):
            if model_name == 'lstm' and hasattr(self.base_models[i], 'history'):
                self.history['training_history'][model_name] = self.base_models[i].history
        
        return self.history
    
    def _train_stacking_ensemble(self, 
                                X_train_2d: np.ndarray,
                                y_train: np.ndarray,
                                X_test_2d: np.ndarray,
                                y_test: np.ndarray,
                                X_train_3d: np.ndarray,
                                X_test_3d: np.ndarray,
                                verbose: int) -> Dict[str, Any]:
        """
        Train a stacking ensemble.
        
        Args:
            X_train_2d: 2D training features
            y_train: Training targets
            X_test_2d: 2D test features
            y_test: Test targets
            X_train_3d: 3D training features (for LSTM)
            X_test_3d: 3D test features (for LSTM)
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        logger.info(f"Training stacking ensemble with {len(self.base_model_names)} base models")
        
        # Create cross-validation folds
        tscv = TimeSeriesSplit(n_splits=self.config['cv_folds'])
        
        # Initialize arrays to hold predictions
        cv_preds = np.zeros((X_train_2d.shape[0], len(self.base_model_names)))
        test_preds = np.zeros((X_test_2d.shape[0], len(self.base_model_names)))
        
        # Train each base model with cross-validation
        model_metrics = {}
        
        for model_idx, model_name in enumerate(self.base_model_names):
            logger.info(f"Training base model: {model_name}")
            
            # Initialize test predictions for this model
            test_model_preds = np.zeros((X_test_2d.shape[0], self.config['cv_folds']))
            
            # Cross-validation training for out-of-fold predictions
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train_2d)):
                # Get fold data
                fold_X_train, fold_X_val = X_train_2d[train_idx], X_train_2d[val_idx]
                fold_y_train, fold_y_val = y_train[train_idx], y_train[val_idx]
                
                # Handle LSTM models specially
                if model_name == 'lstm':
                    fold_X_train_3d = X_train_3d[train_idx]
                    fold_X_val_3d = X_train_3d[val_idx]
                    
                    # Create and train LSTM model
                    model = self._create_lstm()
                    model.train({
                        'X_train': fold_X_train_3d,
                        'y_train': fold_y_train,
                        'target_scaler': self.target_scaler
                    }, validation_split=0.1, verbose=0)
                    
                    # Get out-of-fold predictions
                    val_fold_preds = model.predict(fold_X_val_3d, return_scaled=True).flatten()
                    
                    # Get test predictions for this fold
                    test_fold_preds = model.predict(X_test_3d, return_scaled=True).flatten()
                else:
                    # Create and train traditional ML model
                    model = self.available_models[model_name]()
                    model.fit(fold_X_train, fold_y_train)
                    
                    # Get out-of-fold predictions
                    val_fold_preds = model.predict(fold_X_val).flatten()
                    
                    # Get test predictions for this fold
                    test_fold_preds = model.predict(X_test_2d).flatten()
                
                # Store out-of-fold predictions
                cv_preds[val_idx, model_idx] = val_fold_preds
                
                # Store test predictions for this fold
                test_model_preds[:, fold_idx] = test_fold_preds
            
            # Average test predictions across folds
            test_preds[:, model_idx] = np.mean(test_model_preds, axis=1)
            
            # Calculate metrics for this base model
            model_mae = mean_absolute_error(y_test, test_preds[:, model_idx])
            model_mse = mean_squared_error(y_test, test_preds[:, model_idx])
            model_r2 = r2_score(y_test, test_preds[:, model_idx])
            
            model_metrics[model_name] = {
                'mae': model_mae,
                'mse': model_mse,
                'rmse': np.sqrt(model_mse),
                'r2': model_r2
            }
            
            logger.info(f"{model_name} CV metrics - MAE: {model_mae:.4f}, RMSE: {np.sqrt(model_mse):.4f}")
        
        # Train meta-model on out-of-fold predictions
        if self.config['meta_model'] == 'linear':
            meta_model = LinearRegression()
        elif self.config['meta_model'] == 'ridge':
            meta_model = Ridge(alpha=0.5)
        else:
            meta_model = self.available_models[self.config['meta_model']]()
        
        meta_model.fit(cv_preds, y_train)
        self.meta_model = meta_model
        
        # Make ensemble predictions on test data
        ensemble_preds = meta_model.predict(test_preds).reshape(-1, 1)
        
        # Calculate ensemble metrics
        ensemble_mae = mean_absolute_error(y_test, ensemble_preds)
        ensemble_mse = mean_squared_error(y_test, ensemble_preds)
        ensemble_r2 = r2_score(y_test, ensemble_preds)
        
        logger.info(f"Stacking ensemble metrics - MAE: {ensemble_mae:.4f}, RMSE: {np.sqrt(ensemble_mse):.4f}")
        
        # Save predictions and history
        self.base_predictions = test_preds
        
        # Train final base models on all data
        self.base_models = []
        for model_name in self.base_model_names:
            if model_name == 'lstm':
                model = self._create_lstm()
                model.train({
                    'X_train': X_train_3d,
                    'y_train': y_train,
                    'target_scaler': self.target_scaler
                }, validation_split=0.1, verbose=0)
            else:
                model = self.available_models[model_name]()
                model.fit(X_train_2d, y_train)
            self.base_models.append(model)
        
        # Save history
        self.history = {
            'ensemble_metrics': {
                'mae': ensemble_mae,
                'mse': ensemble_mse,
                'rmse': np.sqrt(ensemble_mse),
                'r2': ensemble_r2
            },
            'model_metrics': model_metrics
        }
        
        return self.history
    
    def _train_boosting_ensemble(self, 
                               X_train_2d: np.ndarray,
                               y_train: np.ndarray,
                               X_test_2d: np.ndarray,
                               y_test: np.ndarray,
                               verbose: int) -> Dict[str, Any]:
        """
        Train a boosting ensemble.
        
        Args:
            X_train_2d: 2D training features
            y_train: Training targets
            X_test_2d: 2D test features
            y_test: Test targets
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        logger.info("Training boosting ensemble")
        
        # For boosting, we use a single GradientBoostingRegressor or XGBoost
        if 'xgboost' in self.base_model_names:
            model = self._create_xgboost()
        else:
            model = self._create_gradient_boosting()
        
        # Train the model
        if isinstance(model, xgb.XGBRegressor):
            eval_set = [(X_train_2d, y_train), (X_test_2d, y_test)]
            model.fit(
                X_train_2d, y_train,
                eval_set=eval_set,
                eval_metric='mae',
                verbose=verbose
            )
            # Get evaluation results
            results = model.evals_result()
        else:
            model.fit(X_train_2d, y_train)
            results = {}  # No history for sklearn GBR
        
        # Make predictions
        test_preds = model.predict(X_test_2d).reshape(-1, 1)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, test_preds)
        mse = mean_squared_error(y_test, test_preds)
        r2 = r2_score(y_test, test_preds)
        
        logger.info(f"Boosting ensemble metrics - MAE: {mae:.4f}, RMSE: {np.sqrt(mse):.4f}")
        
        # Save the model and history
        self.base_models = [model]
        self.history = {
            'ensemble_metrics': {
                'mae': mae,
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2
            },
            'training_history': results
        }
        
        return self.history
    
    def predict(self, 
               X: np.ndarray, 
               return_scaled: bool = False) -> np.ndarray:
        """
        Make predictions with the ensemble model.
        
        Args:
            X: Input features
            return_scaled: Whether to return scaled predictions
            
        Returns:
            Array of predictions
        """
        if not self.base_models:
            raise ValueError("Model has not been trained")
        
        # Reshape data if needed
        if X.ndim == 3 and self.ensemble_type != 'boosting':
            X_2d = X.reshape(X.shape[0], -1)
        else:
            X_2d = X
        
        # Make predictions based on ensemble type
        if self.ensemble_type == 'voting':
            return self._predict_voting_ensemble(X, X_2d, return_scaled)
        elif self.ensemble_type == 'stacking':
            return self._predict_stacking_ensemble(X, X_2d, return_scaled)
        else:  # boosting
            return self._predict_boosting_ensemble(X_2d, return_scaled)
    
    def _predict_voting_ensemble(self, 
                               X_3d: np.ndarray,
                               X_2d: np.ndarray,
                               return_scaled: bool) -> np.ndarray:
        """
        Make predictions with a voting ensemble.
        
        Args:
            X_3d: 3D input features (for LSTM)
            X_2d: 2D input features (for traditional ML)
            return_scaled: Whether to return scaled predictions
            
        Returns:
            Array of predictions
        """
        # Get weights
        weights = self.config['voting_weights']
        if not weights:
            weights = [1.0 / len(self.base_models)] * len(self.base_models)
        
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        # Get predictions from each model
        predictions = []
        for i, (model, model_name) in enumerate(zip(self.base_models, self.base_model_names)):
            if model_name == 'lstm':
                preds = model.predict(X_3d, return_scaled=True)
            else:
                preds = model.predict(X_2d).reshape(-1, 1)
            predictions.append(preds)
        
        # Combine predictions
        ensemble_preds = np.zeros_like(predictions[0])
        for i, preds in enumerate(predictions):
            ensemble_preds += weights[i] * preds
        
        # Return scaled or original values
        if return_scaled or self.target_scaler is None:
            return ensemble_preds
        else:
            return self.target_scaler.inverse_transform(ensemble_preds)
    
    def _predict_stacking_ensemble(self, 
                                 X_3d: np.ndarray,
                                 X_2d: np.ndarray,
                                 return_scaled: bool) -> np.ndarray:
        """
        Make predictions with a stacking ensemble.
        
        Args:
            X_3d: 3D input features (for LSTM)
            X_2d: 2D input features (for traditional ML)
            return_scaled: Whether to return scaled predictions
            
        Returns:
            Array of predictions
        """
        # Get predictions from base models
        base_preds = np.zeros((X_2d.shape[0], len(self.base_models)))
        
        for i, (model, model_name) in enumerate(zip(self.base_models, self.base_model_names)):
            if model_name == 'lstm':
                preds = model.predict(X_3d, return_scaled=True).flatten()
            else:
                preds = model.predict(X_2d).flatten()
            base_preds[:, i] = preds
        
        # Meta-model prediction
        ensemble_preds = self.meta_model.predict(base_preds).reshape(-1, 1)
        
        # Return scaled or original values
        if return_scaled or self.target_scaler is None:
            return ensemble_preds
        else:
            return self.target_scaler.inverse_transform(ensemble_preds)
    
    def _predict_boosting_ensemble(self, 
                                 X: np.ndarray,
                                 return_scaled: bool) -> np.ndarray:
        """
        Make predictions with a boosting ensemble.
        
        Args:
            X: Input features
            return_scaled: Whether to return scaled predictions
            
        Returns:
            Array of predictions
        """
        # Get predictions from boosting model
        preds = self.base_models[0].predict(X).reshape(-1, 1)
        
        # Return scaled or original values
        if return_scaled or self.target_scaler is None:
            return preds
        else:
            return self.target_scaler.inverse_transform(preds)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X: Test features
            y: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Make predictions
        predictions = self.predict(X, return_scaled=True)
        
        # Calculate metrics
        mae = mean_absolute_error(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, predictions)
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        
        logger.info(f"Evaluation metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        return metrics
    
    def save(self, path: Optional[str] = None) -> str:
        """
        Save the ensemble model to disk.
        
        Args:
            path: Path to save the model
            
        Returns:
            Path where model was saved
        """
        if not self.base_models:
            raise ValueError("No model to save")
        
        if path is None:
            # Create default path
            os.makedirs(self.config['save_dir'], exist_ok=True)
            path = os.path.join(
                self.config['save_dir'],
                f"ensemble_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            )
        
        # For LSTM models, we need to save separately
        lstm_models = {}
        for i, (model, model_name) in enumerate(zip(self.base_models, self.base_model_names)):
            if model_name == 'lstm':
                # Save LSTM model
                lstm_dir = os.path.dirname(path)
                lstm_filename = f"lstm_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
                lstm_path = os.path.join(lstm_dir, lstm_filename)
                model.save(lstm_path)
                
                # Replace with path
                lstm_models[i] = lstm_path
        
        # Create saveble object
        save_dict = {
            'ensemble_type': self.ensemble_type,
            'base_model_names': self.base_model_names,
            'config': self.config,
            'history': self.history,
            'lstm_models': lstm_models
        }
        
        # LSTM models are saved separately, for other models we can pickle
        base_models_to_save = []
        for i, model in enumerate(self.base_models):
            if i not in lstm_models:
                base_models_to_save.append(model)
            else:
                base_models_to_save.append(None)  # Placeholder
        
        save_dict['base_models'] = base_models_to_save
        save_dict['meta_model'] = self.meta_model
        
        # Save scalers if available
        if self.feature_scaler:
            save_dict['feature_scaler'] = self.feature_scaler
        if self.target_scaler:
            save_dict['target_scaler'] = self.target_scaler
        
        # Save the model
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"Ensemble model saved to {path}")
        
        return path
    
    def load(self, path: str) -> None:
        """
        Load a saved model.
        
        Args:
            path: Path to the saved model
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load the model
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        # Restore model attributes
        self.ensemble_type = save_dict['ensemble_type']
        self.base_model_names = save_dict['base_model_names']
        self.config = save_dict['config']
        self.history = save_dict['history']
        self.meta_model = save_dict['meta_model']
        
        # Restore scalers
        if 'feature_scaler' in save_dict:
            self.feature_scaler = save_dict['feature_scaler']
        if 'target_scaler' in save_dict:
            self.target_scaler = save_dict['target_scaler']
        
        # Restore base models
        self.base_models = save_dict['base_models']
        
        # Load LSTM models if any
        lstm_models = save_dict['lstm_models']
        for idx, lstm_path in lstm_models.items():
            idx = int(idx)  # Convert to int (pickle keys are strings)
            if os.path.exists(lstm_path):
                lstm_model = LSTMModel()
                lstm_model.load(lstm_path)
                self.base_models[idx] = lstm_model
            else:
                logger.warning(f"LSTM model file not found: {lstm_path}")
        
        logger.info(f"Ensemble model loaded from {path}")
    
    def plot_model_comparison(self) -> None:
        """
        Plot performance comparison of different models.
        """
        if not self.history or 'model_metrics' not in self.history:
            raise ValueError("No training history available")
        
        metrics = self.history['model_metrics']
        model_names = list(metrics.keys())
        mae_values = [metrics[name]['mae'] for name in model_names]
        rmse_values = [metrics[name]['rmse'] for name in model_names]
        
        # Add ensemble metrics
        if 'ensemble_metrics' in self.history:
            model_names.append('Ensemble')
            mae_values.append(self.history['ensemble_metrics']['mae'])
            rmse_values.append(self.history['ensemble_metrics']['rmse'])
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot MAE
        plt.subplot(1, 2, 1)
        plt.bar(model_names, mae_values)
        plt.title('Mean Absolute Error')
        plt.ylabel('MAE')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot RMSE
        plt.subplot(1, 2, 2)
        plt.bar(model_names, rmse_values)
        plt.title('Root Mean Squared Error')
        plt.ylabel('RMSE')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, feature_names: Optional[List[str]] = None) -> None:
        """
        Plot feature importance for models that support it.
        
        Args:
            feature_names: List of feature names
        """
        if not self.base_models:
            raise ValueError("No trained models available")
        
        # Find models that support feature importance
        for i, (model, model_name) in enumerate(zip(self.base_models, self.base_model_names)):
            if model_name in ['random_forest', 'gradient_boosting', 'xgboost']:
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    
                    # Create feature names if not provided
                    if not feature_names:
                        feature_names = [f'Feature {i+1}' for i in range(len(importances))]
                    
                    # Sort by importance
                    indices = np.argsort(importances)[::-1]
                    sorted_names = [feature_names[i] for i in indices]
                    sorted_importances = importances[indices]
                    
                    # Plot
                    plt.figure(figsize=(10, 6))
                    plt.title(f'Feature Importance ({model_name})')
                    plt.bar(range(len(sorted_importances)), sorted_importances, align='center')
                    plt.xticks(range(len(sorted_importances)), sorted_names, rotation=90)
                    plt.tight_layout()
                    plt.show()
                    
                    # Only plot one model's importance
                    break
    
    # Model creation methods
    def _create_random_forest(self) -> RandomForestRegressor:
        """Create a Random Forest regressor."""
        params = self.config['model_params']['random_forest']
        return RandomForestRegressor(**params)
    
    def _create_gradient_boosting(self) -> GradientBoostingRegressor:
        """Create a Gradient Boosting regressor."""
        params = self.config['model_params']['gradient_boosting']
        return GradientBoostingRegressor(**params)
    
    def _create_xgboost(self) -> xgb.XGBRegressor:
        """Create an XGBoost regressor."""
        params = self.config['model_params']['xgboost']
        return xgb.XGBRegressor(**params)
    
    def _create_svr(self) -> SVR:
        """Create a Support Vector regressor."""
        params = self.config['model_params']['svr']
        return SVR(**params)
    
    def _create_linear(self) -> LinearRegression:
        """Create a Linear regressor."""
        return LinearRegression()
    
    def _create_ridge(self) -> Ridge:
        """Create a Ridge regressor."""
        return Ridge(alpha=0.5)
    
    def _create_lasso(self) -> Lasso:
        """Create a Lasso regressor."""
        return Lasso(alpha=0.1)
    
    def _create_elastic_net(self) -> ElasticNet:
        """Create an ElasticNet regressor."""
        return ElasticNet(alpha=0.1, l1_ratio=0.5)
    
    def _create_mlp(self) -> MLPRegressor:
        """Create a Multi-layer Perceptron regressor."""
        return MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500)
    
    def _create_lstm(self) -> LSTMModel:
        """Create an LSTM model."""
        return LSTMModel(config=self.config['model_params']['lstm'])


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
        feature_columns=['Close', 'Volume', 'High', 'Low']
    )
    processed_data = processor.process_data(btc_data)
    
    # Create ensemble model
    ensemble = EnsembleModel(
        ensemble_type='voting',
        base_models=['random_forest', 'gradient_boosting', 'ridge']
    )
    
    # Train model
    history = ensemble.train(processed_data)
    
    # Evaluate model
    metrics = ensemble.evaluate(processed_data['X_test'], processed_data['y_test'])
    
    # Save model
    ensemble.save()