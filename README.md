# Cryptocurrency Price Prediction

A comprehensive machine learning system for predicting cryptocurrency prices using time series analysis and deep learning techniques. This project offers a modular architecture for data collection, feature engineering, model training, prediction, and evaluation.

## Features

- **Multiple Timeframe Support**: Daily, hourly, 30-minute, and custom intervals
- **Advanced Models**: LSTM neural networks and ensemble methods
- **Extensive Feature Engineering**: Technical indicators, trend analysis, and volatility metrics
- **Comprehensive Evaluation**: Multiple performance metrics and visualization tools
- **Flexible Pipeline**: End-to-end workflow or individual components

## Project Structure

```
crypto_price_predictor/
├── data/
│   ├── data_loader.py        # Handles data acquisition from sources
│   └── data_processor.py     # Transforms and prepares data for models
├── features/
│   └── feature_engineering.py # Creates technical indicators and features
├── models/
│   ├── lstm_model.py         # LSTM neural network implementation
│   └── ensemble_model.py     # Ensemble method implementations
├── evaluation/
│   └── metrics.py            # Performance metrics and visualization
└── utils/
    ├── helpers.py            # Utility functions
    └── visualization.py      # Visualization tools
```

## Requirements

- Python 3.8+
- TensorFlow 2.x
- pandas
- numpy
- scikit-learn
- matplotlib
- yfinance
- ta (Technical Analysis library)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The system provides both a complete pipeline and individual components that can be used separately.

### Basic Commands

**Data Collection:**
```bash
python main.py --ticker BTC-USD --days 180 data --save
```

**Feature Engineering:**
```bash
python main.py --ticker BTC-USD --days 180 features --groups trend momentum volatility --visualize
```

**Model Training:**
```bash
python main.py --ticker BTC-USD --days 180 train --model lstm --epochs 100 --sequence-length 60 --save-model
```

**Prediction:**
```bash
python main.py --ticker BTC-USD --days 180 predict --model-path ./results/models/lstm_BTC-USD_20250324_HHMMSS.h5 --days-ahead 30 --plot
```

**Evaluation:**
```bash
python main.py --ticker BTC-USD --days 180 evaluate --model-path ./results/models/lstm_BTC-USD_20250324_HHMMSS.h5
```

**Full Pipeline:**
```bash
python main.py --ticker BTC-USD --days 180 pipeline --model lstm --forecast-days 30 --save-all
```

### Working with 30-Minute Interval Data

This system fully supports 30-minute interval data for more granular analysis and prediction. Use the `--interval 30m` parameter with any command.

**Collecting 30-Minute Data:**
```bash
python main.py --ticker BTC-USD --interval 30m --days 30 data --save
```

**Training with 30-Minute Data:**
```bash
python main.py --ticker BTC-USD --interval 30m --days 30 train --model lstm --epochs 50 --sequence-length 48 --save-model
```

**Predicting with 30-Minute Data:**
```bash
python main.py --ticker BTC-USD --interval 30m --days 30 predict --model-path ./results/models/lstm_BTC-USD_YYYYMMDD_HHMMSS.h5 --days-ahead 2 --plot
```

**Full Pipeline with 30-Minute Data:**
```bash
python main.py --ticker BTC-USD --interval 30m --days 30 pipeline --model lstm --forecast-days 2 --save-all
```

#### Recommended Parameters for 30-Minute Data

When working with 30-minute interval data, consider these recommended parameters:

- **days**: 15-30 days (yields approximately 720-1440 data points)
- **sequence_length**: 48 (represents 24 hours of 30-minute data)
- **epochs**: 50-100 (higher frequency data may need more training epochs)
- **forecast-days**: 1-3 days (short-term forecasts are more accurate for higher frequency data)

#### Notes for 30-Minute Data

1. **Data Volume**: 30-minute data creates 48 data points per day, so even 30 days provides substantial training data.
2. **Prediction Horizon**: For high-frequency data, shorter prediction horizons (1-3 days) typically yield more accurate results.
3. **Hardware Requirements**: Training with high-frequency data may require more computational resources and time.

## Command Reference

### Common Parameters

- `--ticker`: Cryptocurrency symbol (e.g., BTC-USD, ETH-USD)
- `--days`: Number of historical days to use for data
- `--interval`: Data time interval (1d, 1h, 30m, etc.)
- `--output-dir`: Directory to save results

### Data Command

```bash
python main.py --ticker BTC-USD --days 180 data [options]
```

Options:
- `--save`: Save the downloaded data to a CSV file
- `--visualize`: Generate price charts
- `--output-dir PATH`: Specify output directory

### Feature Engineering Command

```bash
python main.py --ticker BTC-USD --days 180 features [options]
```

Options:
- `--groups GROUP1 [GROUP2 ...]`: Feature groups to include (trend, momentum, volatility, volume)
- `--visualize`: Generate feature visualization
- `--correlation`: Show feature correlation heatmap
- `--save`: Save engineered features to CSV

### Training Command

```bash
python main.py --ticker BTC-USD --days 180 train [options]
```

Options:
- `--model MODEL`: Model type (lstm or ensemble)
- `--epochs N`: Number of training epochs
- `--sequence-length N`: Length of input sequences
- `--save-model`: Save the trained model
- `--config PATH`: JSON file with model configuration

### Prediction Command

```bash
python main.py --ticker BTC-USD --days 180 predict [options]
```

Options:
- `--model-path PATH`: Path to the trained model
- `--days-ahead N`: Number of days to predict
- `--plot`: Generate prediction chart

### Evaluation Command

```bash
python main.py --ticker BTC-USD --days 180 evaluate [options]
```

Options:
- `--model-path PATH`: Path to the trained model
- `--comparison`: Compare with benchmark models

### Pipeline Command

```bash
python main.py --ticker BTC-USD --days 180 pipeline [options]
```

Options:
- `--model MODEL`: Model type (lstm or ensemble)
- `--forecast-days N`: Number of days to predict
- `--save-all`: Save data, model, and predictions
- `--config PATH`: JSON file with model configuration

## Advanced Configuration

You can provide advanced configuration via JSON files:

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

Example usage:
```bash
python main.py --ticker BTC-USD --days 180 train --model lstm --config my_config.json
```

## Troubleshooting

### Common Issues with 30-Minute Data

1. **Infinity or NaN values**: 
   - The system now handles this automatically by sanitizing data after feature engineering.
   - If you see related errors, try reducing the amount of data or using the daily interval first.

2. **Model Saving Errors**: 
   - Ensure you have write permissions to the output directory.
   - The system now automatically adds the .h5 extension to model files.

3. **CUDA Errors**: 
   - These are warnings and don't affect functionality. The system will use CPU if GPU is not available.

4. **Shape Errors During Prediction**: 
   - The latest version properly handles input sequence shapes for predictions.
   - If problems persist, ensure your model was trained with the same parameters.

## Results and Interpretation

The model outputs several evaluation metrics:

- **MAE (Mean Absolute Error)**: Average absolute difference between predictions and actual values
- **RMSE (Root Mean Squared Error)**: Root of the average squared differences
- **Directional Accuracy**: Percentage of correct price movement directions predicted
- **R²**: Coefficient of determination (how well the model explains the variance)

A typical successful model might show:
- MAE < 5% of the price
- Directional Accuracy > 55%

## Example Results

Training a model with 30-minute Bitcoin data typically yields results like:

```
Test metrics: {'loss': 0.0023914019111543894, 'mae': 0.03846069425344467}
```

This indicates mean absolute error of around 3.8%, which is quite good for cryptocurrency price prediction.

## License

This project is available under the MIT License.