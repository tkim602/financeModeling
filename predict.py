import os
import numpy as np
import pandas as pd
import joblib
from config import DATE_COL, SEQ_LENGTH
from data_processing import load_data
from features import create_lstm_features
from modeling import scale_data, build_lstm

def predict_future_price(df, model, scaler, feature_cols, seq_length, predict_date):
    df = create_lstm_features(df)
    df = df[df[DATE_COL] <= predict_date]
    
    if len(df) < seq_length:
        raise ValueError("Not enough data to make predictions for the given date.")
    
    recent_data = df[feature_cols].iloc[-seq_length:].values
    print("Recent raw data:", recent_data)
    
    recent_data_scaled = scaler.transform(recent_data)
    print("Scaled data for prediction:", recent_data_scaled)

    recent_data = df[feature_cols].iloc[-seq_length:].values
    recent_data_scaled = scaler.transform(recent_data)
    recent_data_scaled = recent_data_scaled.reshape(1, seq_length, len(feature_cols))
    predicted_log_price = model.predict(recent_data_scaled)
    predicted_price = np.expm1(predicted_log_price[0, 0])
    return predicted_price

if __name__ == "__main__":
    if not os.path.exists("lstm_model_weights.weights.h5"):
        print("Error: 'lstm_model_weights.weights.h5' not found. Please train the model and save weights.")
        exit(1)

    df = load_data()
    feature_cols = [
        'Log_Close', 'Log_Return', 'Log_Volume', 'MA5', 'MA20',
        'RSI', 'MACD_Line', 'MACD_Signal', 'Boll_Mid', 'Boll_Up', 'Boll_Down',
        'ATR', 'Day_of_Week'
    ]
    seq_length = SEQ_LENGTH
    model = build_lstm((seq_length, len(feature_cols)))
    model.load_weights("lstm_model_weights.weights.h5")
    
    predict_date = input("Enter the date for prediction (YYYY-MM-DD): ")
    predict_date = pd.Timestamp(predict_date).tz_localize("UTC")

    scaler = joblib.load("scaler.pkl") 
    predicted_price = predict_future_price(df, model, scaler, feature_cols, seq_length, predict_date)
    print(f"Predicted price for {predict_date.date()}: ${predicted_price:.2f}")
