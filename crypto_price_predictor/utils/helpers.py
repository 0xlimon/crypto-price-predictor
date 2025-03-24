"""
Helper utilities for cryptocurrency price prediction.
"""

import os
import time
import csv
import json
import pickle
import logging
from functools import wraps
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, cast
import pandas as pd
import numpy as np
from datetime import datetime

# For function that returns the same type as its first argument
T = TypeVar('T')

logger = logging.getLogger(__name__)


def get_execution_time(func: Callable) -> Callable:
    """
    Decorator to measure and log the execution time of a function.
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function that logs execution time
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Format time nicely
        if execution_time < 0.001:
            time_str = f"{execution_time * 1000000:.2f} μs"
        elif execution_time < 1:
            time_str = f"{execution_time * 1000:.2f} ms"
        elif execution_time < 60:
            time_str = f"{execution_time:.2f} sec"
        elif execution_time < 3600:
            time_str = f"{execution_time / 60:.2f} min"
        else:
            time_str = f"{execution_time / 3600:.2f} hr"
        
        logger.info(f"Function '{func.__name__}' executed in {time_str}")
        return result
    
    return wrapper


def save_results_to_csv(
    results: Union[Dict, List, pd.DataFrame], 
    filepath: str,
    include_timestamp: bool = True,
    overwrite: bool = False
) -> str:
    """
    Save results to a CSV file.
    
    Args:
        results: Results to save (dictionary, list of dictionaries, or DataFrame)
        filepath: Path to save the file
        include_timestamp: Whether to include timestamp in filename
        overwrite: Whether to overwrite existing file
        
    Returns:
        Path where the file was saved
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(filepath)
    ensure_dir_exists(directory)
    
    # Add timestamp to filename if requested
    if include_timestamp:
        filename, ext = os.path.splitext(filepath)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = f"{filename}_{timestamp}{ext}"
    
    # Check if file exists and handle accordingly
    if os.path.exists(filepath) and not overwrite:
        filename, ext = os.path.splitext(filepath)
        filepath = f"{filename}_new{ext}"
        logger.warning(f"File already exists, saving as {filepath}")
    
    try:
        # Convert to DataFrame if needed
        if isinstance(results, dict):
            df = pd.DataFrame([results])
        elif isinstance(results, list):
            df = pd.DataFrame(results)
        elif isinstance(results, pd.DataFrame):
            df = results
        else:
            raise ValueError(f"Unsupported result type: {type(results)}")
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        logger.info(f"Results saved to {filepath}")
        
    except Exception as e:
        logger.error(f"Error saving results to CSV: {str(e)}")
        raise
    
    return filepath


def load_model_from_config(
    config_path: str, 
    model_type: str = 'ensemble',
    custom_params: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Load a model from a configuration file.
    
    Args:
        config_path: Path to the configuration file (JSON)
        model_type: Type of model to load ('lstm', 'ensemble', etc.)
        custom_params: Additional parameters to override configuration
        
    Returns:
        Loaded model instance
    """
    try:
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Override with custom parameters if provided
        if custom_params:
            for key, value in custom_params.items():
                if key in config:
                    if isinstance(config[key], dict) and isinstance(value, dict):
                        # Deep update for nested dictionaries
                        config[key].update(value)
                    else:
                        config[key] = value
                else:
                    config[key] = value
        
        # Import appropriate model class
        if model_type.lower() == 'lstm':
            from crypto_price_predictor.models.lstm_model import LSTMModel
            model = LSTMModel(config=config)
        elif model_type.lower() == 'ensemble':
            from crypto_price_predictor.models.ensemble_model import EnsembleModel
            model = EnsembleModel(config=config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Load weights if specified in config
        if 'model_path' in config and config['model_path']:
            model.load(config['model_path'])
        
        logger.info(f"Model loaded from config: {config_path}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model from config: {str(e)}")
        raise


def ensure_dir_exists(directory: str) -> str:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Directory path to create
        
    Returns:
        The directory path
    """
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")
    
    return directory


def format_large_number(
    number: Union[int, float], 
    decimal_places: int = 2,
    use_suffix: bool = True
) -> str:
    """
    Format a large number with appropriate suffix (K, M, B, T).
    
    Args:
        number: Number to format
        decimal_places: Number of decimal places to show
        use_suffix: Whether to use suffixes (K, M, B, T)
        
    Returns:
        Formatted number as string
    """
    if not use_suffix:
        return f"{number:,.{decimal_places}f}"
    
    abs_num = abs(number)
    
    if abs_num < 1000:
        return f"{number:.{decimal_places}f}"
    elif abs_num < 1000000:
        return f"{number/1000:.{decimal_places}f}K"
    elif abs_num < 1000000000:
        return f"{number/1000000:.{decimal_places}f}M"
    elif abs_num < 1000000000000:
        return f"{number/1000000000:.{decimal_places}f}B"
    else:
        return f"{number/1000000000000:.{decimal_places}f}T"


def load_data_safe(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = '1d',
    retries: int = 3,
    retry_delay: int = 5
) -> pd.DataFrame:
    """
    Safely load cryptocurrency data with retries.
    
    Args:
        ticker: Cryptocurrency ticker symbol
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        interval: Data interval
        retries: Number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        DataFrame with cryptocurrency data
    """
    from crypto_price_predictor.data.data_loader import CryptoDataLoader
    
    loader = CryptoDataLoader()
    
    for attempt in range(retries):
        try:
            data = loader.load_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                interval=interval
            )
            return data
        except Exception as e:
            if attempt < retries - 1:
                logger.warning(f"Error loading data (attempt {attempt+1}/{retries}): {str(e)}")
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to load data after {retries} attempts: {str(e)}")
                raise
    
    # This should never be reached due to the raised exception above
    raise RuntimeError("Unexpected error in load_data_safe")


def get_available_models() -> Dict[str, str]:
    """
    Get available pre-trained models in the models directory.
    
    Returns:
        Dictionary mapping model names to file paths
    """
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    
    if not os.path.exists(models_dir):
        return {}
    
    available_models = {}
    
    # Look for model files
    for filename in os.listdir(models_dir):
        filepath = os.path.join(models_dir, filename)
        
        # Skip directories
        if os.path.isdir(filepath):
            continue
        
        # Check file extension
        _, ext = os.path.splitext(filename)
        if ext.lower() in ['.h5', '.pkl', '.model']:
            model_name = os.path.splitext(filename)[0]
            available_models[model_name] = filepath
    
    return available_models


def create_directory_structure() -> None:
    """
    Create the standard directory structure for the project.
    """
    # Base directories
    base_dir = os.path.dirname(os.path.dirname(__file__))
    directories = [
        os.path.join(base_dir, 'models'),
        os.path.join(base_dir, 'data', 'raw'),
        os.path.join(base_dir, 'data', 'processed'),
        os.path.join(base_dir, 'results'),
        os.path.join(base_dir, 'logs'),
        os.path.join(base_dir, 'config')
    ]
    
    # Create each directory
    for directory in directories:
        ensure_dir_exists(directory)
    
    logger.info("Directory structure created")


def setup_logging(
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Set up logging for the application.
    
    Args:
        log_file: Path to log file (if None, only console logging is used)
        log_level: Logging level
        console: Whether to log to console
        
    Returns:
        Configured logger
    """
    # Create logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Add file handler if log_file is specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        ensure_dir_exists(log_dir)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)
    
    return root_logger


if __name__ == "__main__":
    # Example usage
    
    # Set up logging
    logger = setup_logging()
    
    # Example of timing decorator
    @get_execution_time
    def slow_function(n: int) -> int:
        time.sleep(0.5)
        return n * n
    
    result = slow_function(5)
    print(f"Result: {result}")
    
    # Example of formatting large numbers
    large_num = 1234567890
    formatted = format_large_number(large_num)
    print(f"Formatted number: {formatted}")
    
    # Create directory structure
    create_directory_structure()