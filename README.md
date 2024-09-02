# Stock Market Prediction using Random Forest Classifier

## Overview

This project is a demonstration of using machine learning to predict stock market trends using historical data. The script employs a `RandomForestClassifier` from the `scikit-learn` library to make predictions on the S&P 500 index (^GSPC) data, obtained via the `yfinance` library. The main objective is to predict whether the closing price of the S&P 500 index will increase or decrease on the following trading day.

## Prerequisites

To run this script, you need to have the following Python libraries installed:

- `yfinance`
- `scikit-learn`
- `matplotlib`
- `pandas`

You can install these libraries using pip:

```bash
pip install yfinance scikit-learn matplotlib pandas
```
## How the Script Works

### 1. Data Retrieval
The script retrieves historical data for the S&P 500 index (^GSPC) using the `yfinance` library. It pulls data for the maximum available period.

### 2. Data Preprocessing
- The dataset is cleaned by removing unnecessary columns like `Dividends` and `Stock Splits`.
- A new column `Tomorrow` is created to store the closing price of the next day.
- A `Target` column is introduced to indicate whether the closing price on the next day is higher than the current day's closing price (binary classification: 1 for increase, 0 for decrease).

### 3. Model Training
- The data is split into training and testing sets. The training set contains all data except the last 100 rows, which are reserved for testing.
- The model is a `RandomForestClassifier` initialized with 100 estimators and a minimum split size of 100. It is trained using the predictors: `Close`, `Volume`, `Open`, `High`, and `Low`.

### 4. Prediction and Backtesting
- The script makes predictions on the test set and evaluates the precision score.
- A backtesting function iteratively trains and tests the model over different time horizons to validate its performance.

### 5. Feature Engineering
- New features are created based on rolling averages and trend calculations over multiple time horizons (e.g., 2, 5, 60, 250, and 1000 days).
- The model is retrained using these new features, and predictions are made using a threshold of 0.6 for positive classification.

### 6. Visualization
The script visualizes the predictions and trends using `matplotlib` to help assess the model's performance visually.

### 7. Output
- The precision score of the model's predictions is printed to give an indication of its accuracy.
- Value counts of predictions and a trend line are plotted.


