import numpy as np
import pandas as pd
from config import DATE_COL, TRAIN_END, VAL_END, TEST_START, SEQ_LENGTH
from data_processing import load_data
from features import create_lstm_features, make_sequence_data
from modeling import scale_data, build_lstm, train_lstm
from evaluate import evaluate_prediction, plot_result

def main():
    df = load_data()
    df = create_lstm_features(df)
    train_df = df[df[DATE_COL] <= TRAIN_END].copy()
    val_df = df[(df[DATE_COL] > TRAIN_END) & (df[DATE_COL] <= VAL_END)].copy()
    test_df = df[df[DATE_COL] > TEST_START].copy()

    feature_cols = ['Log_Close', 'Log_MA5', 'Log_MA20', 'Log_Volume', 'Volume_MA5']
    target_col = 'Log_Close_Target'

    train_X, train_y = make_sequence_data(train_df, feature_cols, target_col, SEQ_LENGTH)
    val_X, val_y = make_sequence_data(val_df, feature_cols, target_col, SEQ_LENGTH)
    test_X, test_y = make_sequence_data(test_df, feature_cols, target_col, SEQ_LENGTH)

    train_X_flat = train_X.reshape(-1, len(feature_cols))
    val_X_flat = val_X.reshape(-1, len(feature_cols))
    test_X_flat = test_X.reshape(-1, len(feature_cols))

    train_s, val_s, test_s, scaler = scale_data(train_X_flat, val_X_flat, test_X_flat)

    train_X_s = train_s.reshape(train_X.shape[0], SEQ_LENGTH, len(feature_cols))
    val_X_s = val_s.reshape(val_X.shape[0], SEQ_LENGTH, len(feature_cols))
    test_X_s = test_s.reshape(test_X.shape[0], SEQ_LENGTH, len(feature_cols))

    model = build_lstm((SEQ_LENGTH, len(feature_cols)))
    model = train_lstm(model, train_X_s, train_y, val_X_s, val_y)

    pred_log = model.predict(test_X_s)
    true_log = test_y

    pred_price = np.expm1(pred_log)
    true_price = np.expm1(true_log)

    evaluate_prediction(true_price, pred_price)
    plot_result(true_price, pred_price)

if __name__ == "__main__":
    main()
