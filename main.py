#!/usr/bin/env python
"""
Cryptocurrency Price Prediction

Main entry point for the cryptocurrency price prediction application.
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from crypto_price_predictor.data.data_loader import CryptoDataLoader
from crypto_price_predictor.data.data_processor import DataProcessor
from crypto_price_predictor.features.feature_engineering import FeatureEngineer
from crypto_price_predictor.models.lstm_model import LSTMModel
from crypto_price_predictor.models.ensemble_model import EnsembleModel
from crypto_price_predictor.evaluation.metrics import (
    calculate_metrics, plot_predictions, plot_error_distribution,
    evaluate_multiple_models, calculate_directional_accuracy,
    plot_forecast_comparison
)
from crypto_price_predictor.utils.visualization import (
    plot_price_history, plot_technical_indicators,
    plot_correlation_matrix, create_candlestick_chart
)
from crypto_price_predictor.utils.helpers import (
    get_execution_time, save_results_to_csv, ensure_dir_exists,
    format_large_number, setup_logging
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Cryptocurrency Price Prediction')
    
    # General options
    parser.add_argument('--ticker', type=str, default='BTC-USD',
                        help='Cryptocurrency ticker symbol (default: BTC-USD)')
    parser.add_argument('--days', type=int, default=730,
                        help='Number of days of historical data to use (default: 730)')
    parser.add_argument('--interval', type=str, default='1d',
                        help='Data interval (1d, 1h, etc. - default: 1d)')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Directory to save results (default: ./results)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Log file path (default: None)')
    
    # Sub-commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Data command
    data_parser = subparsers.add_parser('data', help='Download and process data')
    data_parser.add_argument('--save', action='store_true',
                            help='Save processed data to CSV')
    data_parser.add_argument('--visualize', action='store_true',
                            help='Visualize the data')
    
    # Features command
    features_parser = subparsers.add_parser('features', help='Engineer features')
    features_parser.add_argument('--groups', type=str, nargs='+',
                                default=['trend', 'momentum', 'volatility', 'volume'],
                                help='Feature groups to include')
    features_parser.add_argument('--visualize', action='store_true',
                                help='Visualize features')
    features_parser.add_argument('--correlation', action='store_true',
                                help='Show correlation matrix')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train prediction models')
    train_parser.add_argument('--model', type=str, default='lstm',
                            choices=['lstm', 'ensemble'],
                            help='Model type to train (default: lstm)')
    train_parser.add_argument('--epochs', type=int, default=100,
                            help='Number of epochs for training (default: 100)')
    train_parser.add_argument('--sequence-length', type=int, default=60,
                            help='Sequence length for LSTM (default: 60)')
    train_parser.add_argument('--save-model', action='store_true',
                            help='Save the trained model')
    train_parser.add_argument('--config', type=str,
                            help='Path to model configuration file')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--model-path', type=str, required=True,
                                help='Path to the saved model')
    predict_parser.add_argument('--days-ahead', type=int, default=30,
                                help='Number of days to predict ahead (default: 30)')
    predict_parser.add_argument('--plot', action='store_true',
                                help='Plot predictions')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    evaluate_parser.add_argument('--model-path', type=str, required=True,
                                help='Path to the saved model')
    evaluate_parser.add_argument('--comparison', action='store_true',
                                help='Compare with other models')
    
    # Full pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full prediction pipeline')
    pipeline_parser.add_argument('--model', type=str, default='ensemble',
                                choices=['lstm', 'ensemble'],
                                help='Model type to train (default: ensemble)')
    pipeline_parser.add_argument('--epochs', type=int, default=100,
                                help='Number of epochs for training (default: 100)')
    pipeline_parser.add_argument('--forecast-days', type=int, default=30,
                                help='Days to forecast ahead (default: 30)')
    pipeline_parser.add_argument('--save-all', action='store_true',
                                help='Save all artifacts (data, model, predictions)')
    
    return parser.parse_args()


def setup_environment(args):
    """Set up the environment (logging, directories, etc.)."""
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(args.log_file, log_level)
    
    # Create output directory
    ensure_dir_exists(args.output_dir)
    
    logger.info(f"Environment set up complete. Output directory: {args.output_dir}")
    return logger


@get_execution_time
def load_data(args):
    """Load cryptocurrency data."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data for {args.ticker} with interval {args.interval}")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    # Initialize data loader
    loader = CryptoDataLoader()
    
    # Load data
    data = loader.load_data(
        ticker=args.ticker,
        start_date=start_date,
        end_date=end_date,
        interval=args.interval
    )
    
    logger.info(f"Loaded {len(data)} records from {data.index.min()} to {data.index.max()}")
    
    return data


@get_execution_time
def process_data(data, args, feature_engineering=True):
    """Process and engineer features for the data."""
    logger = logging.getLogger(__name__)
    
    # Engineer features if requested
    if feature_engineering:
        logger.info("Engineering features")
        
        # Feature groups to use
        feature_groups = args.groups if hasattr(args, 'groups') else [
            'trend', 'momentum', 'volatility', 'volume'
        ]
        
        # Initialize feature engineer
        engineer = FeatureEngineer(feature_groups=feature_groups)
        
        # Add features
        data = engineer.engineer_features(data)
        
        logger.info(f"Added {len(engineer.created_features)} new features")
        
        # Show correlation matrix if requested
        if hasattr(args, 'correlation') and args.correlation:
            logger.info("Generating correlation matrix")
            plot_correlation_matrix(data, title=f"{args.ticker} Feature Correlation")
    
    # Define parameters for data processor
    sequence_length = args.sequence_length if hasattr(args, 'sequence_length') else 60
    
    # Initialize data processor
    processor = DataProcessor(
        sequence_length=sequence_length,
        forecast_horizon=1,
        feature_columns=list(data.columns)  # Use all available columns
    )
    
    # Process data for model training
    processed_data = processor.process_data(data, target_column='Close')
    
    logger.info(f"Data processed. Training set: {processed_data['X_train'].shape}, "
               f"Test set: {processed_data['X_test'].shape}")
    
    return data, processed_data


@get_execution_time
def train_model(processed_data, args):
    """Train a prediction model."""
    logger = logging.getLogger(__name__)
    logger.info(f"Training {args.model} model")
    
    # Configure model
    if args.config:
        # Load configuration from file
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            'lstm_layers': [128, 64],
            'dense_layers': [32],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': args.epochs,
            'save_dir': os.path.join(args.output_dir, 'models')
        }
    
    # Create model
    if args.model == 'lstm':
        model = LSTMModel(config=config)
    else:  # ensemble
        model = EnsembleModel(
            ensemble_type='stacking',
            base_models=['random_forest', 'gradient_boosting', 'xgboost', 'lstm'],
            config=config
        )
    
    # Train model
    logger.info("Starting model training")
    history = model.train(processed_data)
    
    # Evaluate on test set
    metrics = model.evaluate(processed_data['X_test'], processed_data['y_test'])
    
    logger.info(f"Model training complete. Test metrics: {metrics}")
    
    # Save model if requested
    if args.save_model:
        model_path = os.path.join(args.output_dir, 'models', 
                                 f"{args.model}_{args.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5")
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
    
    return model, history, metrics


@get_execution_time
def make_predictions(model, data, processed_data, args):
    """Make predictions with the trained model."""
    logger = logging.getLogger(__name__)
    
    # Make predictions on test data
    y_pred = model.predict(processed_data['X_test'])
    y_true = processed_data['target_scaler'].inverse_transform(processed_data['y_test'])
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    logger.info(f"Prediction metrics: MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, "
               f"Directional Accuracy: {metrics['directional_accuracy']:.4f}")
    
    # Plot predictions if requested
    if hasattr(args, 'plot') and args.plot:
        # Get dates for test set
        test_dates = data.index[-len(y_true):]
        
        # Plot predictions vs actual
        plot_predictions(y_true, y_pred, dates=test_dates, 
                        model_name=args.model.upper(), 
                        save_path=os.path.join(args.output_dir, 'predictions_plot.png'))
        
        # Plot error distribution
        plot_error_distribution(y_true, y_pred, model_name=args.model.upper(),
                               save_path=os.path.join(args.output_dir, 'error_distribution.png'))
    
    # Predict future if days_ahead is specified
    if hasattr(args, 'days_ahead') and args.days_ahead > 0:
        logger.info(f"Predicting {args.days_ahead} days ahead")
        
        # For LSTM model, we can use its specific future prediction method
        if hasattr(model, 'predict_future'):
            future_predictions = model.predict_future(
                processed_data['X_test'][-1:], 
                days_ahead=args.days_ahead,
                plot=args.plot if hasattr(args, 'plot') else False
            )
        else:
            # For other models, we need to implement recursive prediction
            logger.warning("Future prediction not implemented for this model type")
            future_predictions = None
        
        if future_predictions is not None:
            logger.info(f"Future predictions:\n{future_predictions.head()}")
            
            # Save predictions if output dir is specified
            predictions_path = os.path.join(args.output_dir, 
                                          f"{args.ticker}_predictions_{datetime.now().strftime('%Y%m%d')}.csv")
            future_predictions.to_csv(predictions_path)
            logger.info(f"Future predictions saved to {predictions_path}")
    
    return y_pred, y_true, metrics


def run_data_command(args):
    """Execute the 'data' command."""
    logger = logging.getLogger(__name__)
    
    # Load data
    data = load_data(args)
    
    # Visualize if requested
    if args.visualize:
        # Plot price history
        plot_price_history(data, title=f"{args.ticker} Price History", 
                          save_path=os.path.join(args.output_dir, f"{args.ticker}_price_history.png"))
        
        # Create candlestick chart
        create_candlestick_chart(data, title=f"{args.ticker} Candlestick Chart",
                                save_path=os.path.join(args.output_dir, f"{args.ticker}_candlestick.png"))
    
    # Save to CSV if requested
    if args.save:
        file_path = os.path.join(args.output_dir, f"{args.ticker}_data.csv")
        data.to_csv(file_path)
        logger.info(f"Data saved to {file_path}")
    
    return data


def run_features_command(args):
    """Execute the 'features' command."""
    logger = logging.getLogger(__name__)
    
    # Load data
    data = load_data(args)
    
    # Engineer features
    engineer = FeatureEngineer(feature_groups=args.groups)
    enhanced_data = engineer.engineer_features(data)
    
    logger.info(f"Created {len(engineer.created_features)} new features")
    
    # Visualize if requested
    if args.visualize:
        # Plot technical indicators
        indicators = {
            'trend': [col for col in enhanced_data.columns if 'SMA' in col or 'EMA' in col],
            'oscillators': [col for col in enhanced_data.columns if 'RSI' in col or 'ROC' in col],
            'volatility': [col for col in enhanced_data.columns if 'BB' in col or 'ATR' in col]
        }
        
        plot_technical_indicators(enhanced_data, indicators=indicators,
                                title=f"{args.ticker} Technical Indicators",
                                save_path=os.path.join(args.output_dir, f"{args.ticker}_indicators.png"))
    
    # Show correlation if requested
    if args.correlation:
        # Select subset of important features
        important_features = ['Close', 'Volume']
        important_features.extend([col for col in enhanced_data.columns if 'SMA' in col][:2])
        important_features.extend([col for col in enhanced_data.columns if 'RSI' in col][:1])
        important_features.extend([col for col in enhanced_data.columns if 'MACD' in col][:1])
        
        plot_correlation_matrix(enhanced_data, columns=important_features,
                               title=f"{args.ticker} Feature Correlation",
                               save_path=os.path.join(args.output_dir, f"{args.ticker}_correlation.png"))
    
    return enhanced_data


def run_train_command(args):
    """Execute the 'train' command."""
    # Load data
    data = load_data(args)
    
    # Process data (with feature engineering)
    _, processed_data = process_data(data, args)
    
    # Train model
    model, history, metrics = train_model(processed_data, args)
    
    return model, history, metrics


def run_predict_command(args):
    """Execute the 'predict' command."""
    logger = logging.getLogger(__name__)
    
    # Load data
    data = load_data(args)
    
    # Process data (with feature engineering)
    data, processed_data = process_data(data, args)
    
    # Load model
    if args.model_path.endswith('.h5'):
        model = LSTMModel(model_path=args.model_path)
        logger.info(f"Loaded LSTM model from {args.model_path}")
    elif args.model_path.endswith('.pkl'):
        model = EnsembleModel()
        model.load(args.model_path)
        logger.info(f"Loaded ensemble model from {args.model_path}")
    else:
        logger.error(f"Unknown model type for file: {args.model_path}")
        return None
    
    # Make predictions
    y_pred, y_true, metrics = make_predictions(model, data, processed_data, args)
    
    return y_pred, y_true, metrics


def run_evaluate_command(args):
    """Execute the 'evaluate' command."""
    logger = logging.getLogger(__name__)
    
    # Load data
    data = load_data(args)
    
    # Process data (with feature engineering)
    data, processed_data = process_data(data, args)
    
    # Load model
    if args.model_path.endswith('.h5'):
        model = LSTMModel(model_path=args.model_path)
        model_type = 'lstm'
        logger.info(f"Loaded LSTM model from {args.model_path}")
    elif args.model_path.endswith('.pkl'):
        model = EnsembleModel()
        model.load(args.model_path)
        model_type = 'ensemble'
        logger.info(f"Loaded ensemble model from {args.model_path}")
    else:
        logger.error(f"Unknown model type for file: {args.model_path}")
        return None
    
    # Evaluate model
    metrics = model.evaluate(processed_data['X_test'], processed_data['y_test'])
    
    # Make predictions for visualization
    y_pred = model.predict(processed_data['X_test'])
    y_true = processed_data['target_scaler'].inverse_transform(processed_data['y_test'])
    
    # Plot predictions
    test_dates = data.index[-len(y_true):]
    plot_predictions(y_true, y_pred, dates=test_dates, 
                    model_name=model_type.upper(),
                    save_path=os.path.join(args.output_dir, 'evaluation_plot.png'))
    
    # Compare with other models if requested
    if args.comparison:
        logger.info("Training comparison models")
        
        # Create other models for comparison
        models = {
            'LSTM': model  # Already loaded model
        }
        
        # Train a simple model for comparison
        if model_type != 'lstm':
            lstm_config = {
                'lstm_layers': [64, 32],
                'epochs': 50,
                'batch_size': 32
            }
            models['Simple LSTM'] = LSTMModel(config=lstm_config)
            models['Simple LSTM'].train(processed_data)
        
        # Train ensemble if not already loaded
        if model_type != 'ensemble':
            ensemble_config = {
                'lstm_layers': [64, 32],
                'epochs': 50
            }
            models['Ensemble'] = EnsembleModel(
                ensemble_type='voting',
                base_models=['random_forest', 'gradient_boosting', 'ridge'],
                config=ensemble_config
            )
            models['Ensemble'].train(processed_data)
        
        # Compare models
        comparison_results = evaluate_multiple_models(
            models, processed_data['X_test'], processed_data['y_test'],
            target_scaler=processed_data['target_scaler']
        )
        
        # Save comparison results
        comparison_path = os.path.join(args.output_dir, 'model_comparison.csv')
        save_results_to_csv(comparison_results, comparison_path)
        logger.info(f"Comparison results saved to {comparison_path}")
        
        # Get predictions from all models
        forecasts = {}
        for model_name, model_obj in models.items():
            forecasts[model_name] = model_obj.predict(processed_data['X_test'])
        
        # Plot comparison
        plot_forecast_comparison(
            y_true, forecasts, dates=test_dates,
            save_path=os.path.join(args.output_dir, 'model_comparison_plot.png')
        )
    
    return metrics


def run_pipeline_command(args):
    """Execute the 'pipeline' command (full prediction pipeline)."""
    logger = logging.getLogger(__name__)
    logger.info("Running full prediction pipeline")
    
    # 1. Load data
    data = load_data(args)
    
    # 2. Process data and engineer features
    data, processed_data = process_data(data, args, feature_engineering=True)
    
    # 3. Train model
    args.model = args.model  # Use model specified in pipeline command
    args.save_model = args.save_all  # Save model if save_all is specified
    model, history, training_metrics = train_model(processed_data, args)
    
    # 4. Make predictions
    args.days_ahead = args.forecast_days  # Use forecast days specified in pipeline command
    args.plot = True  # Always plot in pipeline mode
    y_pred, y_true, prediction_metrics = make_predictions(model, data, processed_data, args)
    
    # 5. Save results if requested
    if args.save_all:
        # Save data
        data_path = os.path.join(args.output_dir, f"{args.ticker}_data.csv")
        data.to_csv(data_path)
        
        # Save metrics
        metrics_path = os.path.join(args.output_dir, f"{args.ticker}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(prediction_metrics, f, indent=2)
        
        logger.info(f"Results saved to {args.output_dir}")
    
    logger.info("Pipeline completed successfully")
    return model, prediction_metrics


def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up environment
    logger = setup_environment(args)
    logger.info(f"Starting cryptocurrency price prediction for {args.ticker}")
    
    try:
        # Execute the requested command
        if args.command == 'data':
            data = run_data_command(args)
        elif args.command == 'features':
            enhanced_data = run_features_command(args)
        elif args.command == 'train':
            model, history, metrics = run_train_command(args)
        elif args.command == 'predict':
            y_pred, y_true, metrics = run_predict_command(args)
        elif args.command == 'evaluate':
            metrics = run_evaluate_command(args)
        elif args.command == 'pipeline':
            model, metrics = run_pipeline_command(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
        
        logger.info("Command executed successfully")
        return 0
        
    except Exception as e:
        logger.exception(f"Error executing command: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())