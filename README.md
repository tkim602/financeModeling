# financeModeling (stock price prediction) 

## Introduction:
This repository is dedicated to building and improving a robust stock price prediction model using LSTM and other advanced machine learning techniques. The project aims to handle multiple stocks, extract meaningful features, and enhance prediction accuracy with continuous iteration and innovation.

I am constantly working on refining the model, expanding support for a broader range of stocks, and integrating advanced data-driven approaches to deliver more precise predictions. Contributions, suggestions, and collaborations are highly encouraged to make this project even better.

## Features: 
- Stock price prediction using LSTM.
- Data pre-processing for financial datasets.
- Model evaluation with metrics like RMSE and accuracy.
- Visualization of stock trends.
- Prediction of future stock prices for specific dates. 

## Installation:
1. Clone the repository
   ```python
   git clone https://github.com/tkim602/financeModeling.git
   cd financeModeling
   ```
2. Set up a Python virtual environment
   ```python
   python3 -m venv stock_env
   source stock_env/bin/activate
   ```
3. Install the dependencies
   ```python
   pip install -r requirements.txt
   ```
## Usage
1. Activate the virtual enviroment
   ```python
   source stock_env/bin/activate
   ```
2. Run the main script
   ```python
   python main.py
   ```
3. Predict stock price
   - Ensure lstm_model_weights.weights.h5 and scaler.pkl are available in the project directory. These files are generated after running main.py.
   1. Run the prediction script
   ```python
   python predict.py
   ```
   2. Input the desired prediction date when prompted
   ```python
   Enter the date for prediction (YYYY-MM-DD): 2024-12-29
   ```
   3. The predicted stock price for the given date will be displayed
   ```python
   Predicted price for 2024-12-29: $129.32
   ```
## Directory Structure
 - main.py: Script for training the LSTM model.
 - predict.py: Script for predicting future stock prices.
 - features.py: Contains feature engineering functions.
 - indicators.py: Includes financial indicators like RSI, MACD, and Bollinger Bands.
 - config.py: Configuration for dataset paths and model parameters.
 - data_processing.py: Functions for loading and processing financial data.
 - evaluate.py: Model evaluation metrics and visualization tools.









