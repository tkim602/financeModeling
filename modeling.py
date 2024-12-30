import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from config import SEQ_LENGTH, EPOCHS, BATCH_SIZE

def scale_data(train, val, test):
    scaler = MinMaxScaler()
    scaler.fit(train)
    train_s = scaler.transform(train)
    val_s = scaler.transform(val)
    test_s = scaler.transform(test)
    return train_s, val_s, test_s, scaler

from tensorflow.keras import Input

def build_lstm(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape)) 
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(256, return_sequences=False)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(loss=tf.keras.losses.Huber(delta=1.0), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    return model

def train_lstm(model, X_train, y_train, X_val, y_val):
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
              validation_data=(X_val, y_val), callbacks=[es, lr_scheduler], verbose=1)
    return model
