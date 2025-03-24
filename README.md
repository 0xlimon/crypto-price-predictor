# Cryptocurrency Price Prediction

A comprehensive machine learning system for predicting cryptocurrency prices using time series analysis and deep learning techniques. This project offers a modular architecture for data collection, feature engineering, model training, prediction, and evaluation.

## Features

- **Multiple timeframe support**: Daily, hourly, 30-minute, and other intervals
- **Advanced prediction models**:
  - LSTM (Long Short-Term Memory) neural networks
  - Ensemble methods (Random Forest, Gradient Boosting, XGBoost)
- **Technical indicators**: Over 60 engineered features from price data
- **Comprehensive evaluation**: Multiple metrics and visualization tools
- **Flexible command-line interface**: Individual commands for each step or full pipeline execution

## Project Structure

```
crypto_price_predictor/
│
├── data/                      # Data handling components
│   ├── data_loader.py         # Loading cryptocurrency data from sources
│   └── data_processor.py      # Preprocessing and preparing data for models
│
├── features/                  # Feature engineering
│   └── feature_engineering.py # Creating technical indicators and features
│
├── models/                    # Model implementations
│   ├── lstm_model.py          # LSTM neural network for time series
│   └── ensemble_model.py      # Ensemble methods implementation
│
├── evaluation/                # Evaluation components
│   └── metrics.py             # Metrics calculation and visualization
│
├── utils/                     # Utility modules
│   ├── visualization.py       # Data and results visualization
│   └── helpers.py             # Helper functions
│
└── main.py                    # Main entry point with CLI
```

## Requirements

- Python 3.8+
- Dependencies:
  ```
  numpy>=1.19.5
  pandas>=1.3.0
  scikit-learn>=0.24.2
  tensorflow>=2.6.0
  matplotlib>=3.4.3
  seaborn>=0.11.2
  yfinance>=0.1.63
  ta>=0.7.0 (Technical Analysis library)
  ```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd cryptocurrency-price-prediction
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

The system provides a command-line interface with multiple commands for different stages of the prediction process.

### Basic Command Structure

```
python main.py [general options] command [command options]
```

### General Options

- `--ticker`: Cryptocurrency ticker symbol (default: BTC-USD)
- `--days`: Number of days of historical data (default: 730)
- `--interval`: Data interval (default: 1d, options: 1d, 1h, 30m, etc.)
- `--output-dir`: Directory to save results (default: ./results)
- `--verbose`: Enable verbose output
- `--log-file`: Path to log file

### Available Commands

1. **data**: Download and process data
   ```
   python main.py --ticker BTC-USD --days 730 data --save --visualize
   ```
   Options:
   - `--save`: Save data to CSV
   - `--visualize`: Create price history and candlestick charts

2. **features**: Engineer features from price data
   ```
   python main.py --ticker BTC-USD --days 730 features --groups trend momentum volatility volume --visualize --correlation
   ```
   Options:
   - `--groups`: Feature groups to include
   - `--visualize`: Create technical indicator charts
   - `--correlation`: Generate correlation matrix

3. **train**: Train prediction models
   ```
   python main.py --ticker BTC-USD --days 730 train --model lstm --epochs 100 --sequence-length 60 --save-model
   ```
   Options:
   - `--model`: Model type (lstm or ensemble)
   - `--epochs`: Number of training epochs
   - `--sequence-length`: Sequence length for LSTM
   - `--save-model`: Save the trained model
   - `--config`: Path to model configuration file

4. **predict**: Make predictions with a trained model
   ```
   python main.py --ticker BTC-USD --days 730 predict --model-path ./results/models/lstm_BTC-USD_YYYYMMDD_HHMMSS.h5 --days-ahead 30 --plot
   ```
   Options:
   - `--model-path`: Path to the saved model (required)
   - `--days-ahead`: Number of days to forecast
   - `--plot`: Generate prediction charts

5. **evaluate**: Evaluate model performance
   ```
   python main.py --ticker BTC-USD --days 730 evaluate --model-path ./results/models/lstm_BTC-USD_YYYYMMDD_HHMMSS.h5 --comparison
   ```
   Options:
   - `--model-path`: Path to the saved model (required)
   - `--comparison`: Compare with other models

6. **pipeline**: Run full prediction pipeline
   ```
   python main.py --ticker BTC-USD --days 730 pipeline --model lstm --epochs 100 --forecast-days 30 --save-all
   ```
   Options:
   - `--model`: Model type (lstm or ensemble)
   - `--epochs`: Number of training epochs
   - `--forecast-days`: Days to forecast ahead
   - `--save-all`: Save all outputs (data, model, predictions)

## Working with 30-Minute Interval Data

To work with 30-minute interval data, use the `--interval 30m` parameter. This higher frequency data requires different considerations compared to daily data.

### Loading 30-Minute Data

```
python main.py --ticker BTC-USD --interval 30m --days 30 data --save
```

Notes:
- Use fewer days (15-30 recommended) to avoid excessive data volume
- 30 days of 30-minute data generates approximately 1,440 data points

### Training with 30-Minute Data

```
python main.py --ticker BTC-USD --interval 30m --days 30 train --model lstm --epochs 50 --sequence-length 48 --save-model
```

Recommendations for 30-minute data:
- `--sequence-length 48`: Uses 24 hours of 30-minute data points as context (48 intervals)
- `--epochs 50-100`: Sufficient for 30-minute data
- LSTM models typically perform better than ensemble models for high-frequency data

### Predicting with 30-Minute Models

```
python main.py --ticker BTC-USD --interval 30m --days 30 predict --model-path ./results/models/lstm_BTC-USD_YYYYMMDD_HHMMSS.h5 --days-ahead 2 --plot
```

Notes:
- For 30-minute data, shorter prediction horizons (1-3 days) are recommended
- `--days-ahead 2` predicts 96 intervals (48 per day)

### Full Pipeline with 30-Minute Data

```
python main.py --ticker BTC-USD --interval 30m --days 30 pipeline --model lstm --epochs 50 --forecast-days 2 --save-all
```

## Understanding the Results

### Model Training Output

The training process outputs several metrics:
- **Loss**: Mean Squared Error (MSE) between predictions and actual values
- **MAE**: Mean Absolute Error, shows average prediction deviation
- **Validation metrics**: Indicate model performance on unseen data

Example output:
```
Test metrics: {'loss': 0.0023914, 'mae': 0.03846}
```
This means predictions are on average about 3.8% away from actual values.

### Prediction Output

Predictions are saved as:
- CSV file with timestamp and predicted values
- Visualization charts comparing actual vs. predicted values
- Error distribution graphs

### Saved Files

All outputs are saved in the `./results` directory:
- Raw data: `./results/BTC-USD_data.csv`
- Trained models: `./results/models/lstm_BTC-USD_YYYYMMDD_HHMMSS.h5`
- Predictions: `./results/BTC-USD_predictions_YYYYMMDD.csv`
- Charts: Various PNG files in `./results`

## Advanced Configuration

### Model Configuration File

You can create a JSON configuration file for fine-tuning model parameters:

```json
{
  "lstm_layers": [64, 32],
  "dense_layers": [16],
  "dropout_rate": 0.2,
  "learning_rate": 0.001,
  "batch_size": 32,
  "epochs": 100,
  "sequence_length": 60
}
```

Use with:
```
python main.py --ticker BTC-USD train --model lstm --config my_config.json
```

## Troubleshooting

### Common Issues

1. **Error with feature engineering for 30-minute data**:
   - Solution: The code now handles high-frequency data properly, ensuring features are calculated correctly.

2. **Out of memory errors with large datasets**:
   - Solution: Reduce the number of days or use a lower frequency interval.

3. **Model saving errors**:
   - Solution: Ensure the model path has a valid extension (.h5).

4. **Poor prediction accuracy**:
   - Solution: Experiment with different sequence lengths, more training epochs, or different model architectures.

## Examples

### Example 1: Predict Bitcoin Price with Daily Data

```
# Load and visualize data
python main.py --ticker BTC-USD --days 730 data --save --visualize

# Train LSTM model
python main.py --ticker BTC-USD --days 730 train --model lstm --epochs 100 --sequence-length 60 --save-model

# Make predictions
python main.py --ticker BTC-USD --days 730 predict --model-path ./results/models/lstm_BTC-USD_YYYYMMDD_HHMMSS.h5 --days-ahead 30 --plot
```

### Example 2: Short-term Prediction with 30-Minute Data

```
# Load 30-minute data
python main.py --ticker BTC-USD --interval 30m --days 30 data --save

# Train model with 30-minute data
python main.py --ticker BTC-USD --interval 30m --days 30 train --model lstm --epochs 50 --sequence-length 48 --save-model

# Predict next 2 days (96 intervals of 30 minutes)
python main.py --ticker BTC-USD --interval 30m --days 30 predict --model-path ./results/models/lstm_BTC-USD_YYYYMMDD_HHMMSS.h5 --days-ahead 2 --plot
```

### Example 3: Complete Pipeline Run

```
python main.py --ticker ETH-USD --interval 1h --days 90 pipeline --model lstm --epochs 75 --forecast-days 7 --save-all
```

## License

MIT