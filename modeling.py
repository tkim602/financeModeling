import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from config import SEQ_LENGTH, EPOCHS, BATCH_SIZE

def scale_data(train, val, test):
    scaler = MinMaxScaler()
    scaler.fit(train)
    train_s = scaler.transform(train)
    val_s = scaler.transform(val)
    test_s = scaler.transform(test)
    return train_s, val_s, test_s, scaler

def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

def train_lstm(model, X_train, y_train, X_val, y_val):
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val), callbacks=[es], verbose=1)
    return model
