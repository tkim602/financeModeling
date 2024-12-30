import pandas as pd
import numpy as np
from indicators import compute_rsi, compute_macd, compute_bollinger_bands
from config import DATE_COL, SEQ_LENGTH

def create_lstm_features(df):
    df['Log_Close'] = np.log1p(df['Close'])
    df['Log_Return'] = df['Log_Close'].diff()
    df['Log_Volume'] = np.log1p(df['Volume'])
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['RSI'] = compute_rsi(df['Close'])
    macd_line, signal_line, macd_hist = compute_macd(df['Close'])
    df['MACD_Line'] = macd_line
    df['MACD_Signal'] = signal_line
    df['Boll_Mid'], df['Boll_Up'], df['Boll_Down'] = compute_bollinger_bands(df['Close'])
    df['ATR'] = df['High'] - df['Low']  # Example feature: Average True Range (improvement)
    df['Day_of_Week'] = df[DATE_COL].dt.dayofweek
    df['Target_Log_Close'] = df['Log_Close'].shift(-SEQ_LENGTH)
    df.dropna(inplace=True)
    return df

def make_sequence_data(df, feature_cols, target_col, seq_len, stride=1):
    X, y = [], []
    data = df[feature_cols].values
    labels = df[target_col].values
    for i in range(0, len(data) - seq_len, stride):
        X.append(data[i:i+seq_len])
        y.append(labels[i+seq_len])
    return np.array(X), np.array(y)
