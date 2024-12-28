import pandas as pd
import numpy as np
from config import DATE_COL, SEQ_LENGTH

def create_lstm_features(df):
    df['Log_Close'] = np.log1p(df['Close'])
    df['Log_MA5'] = df['Log_Close'].rolling(5).mean()
    df['Log_MA20'] = df['Log_Close'].rolling(20).mean()
    df['Log_Volume'] = np.log1p(df['Volume'])
    df['Volume_MA5'] = df['Log_Volume'].rolling(5).mean()
    df['Log_Close_Target'] = df['Log_Close'].shift(-SEQ_LENGTH)
    df.dropna(inplace=True)
    return df

def make_sequence_data(df, feature_cols, target_col, seq_len):
    X, y = [], []
    data = df[feature_cols].values
    labels = df[target_col].values
    for i in range(len(data) - seq_len):
        X.append(data[i : i+seq_len])
        y.append(labels[i + seq_len])
    return np.array(X), np.array(y)
