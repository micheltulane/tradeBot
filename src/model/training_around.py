__author__ = "Michel Tulane"
#File created 24-JUL-2019

# Generic imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# NN Model related imports
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras import layers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import   Input, Dense, LSTM, GRU, Embedding, Activation
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

# Execution variables
TRAIN_MODEL = True
# BATCH_SIZE = 256
# SEQ_LENGTH =
DATA_PATH = "../../data/5min_fetched/"

tf.keras.backend.clear_session()  # Reset notebook state.


# Import data
df = pd.read_excel(os.path.abspath(DATA_PATH + "/train_test_data.xlsx"))

print(df.shape)
x_data = df.drop(columns=["day", "time", "USDT_BTC_expected_price_change_15min"])
y_data = df[["USDT_BTC_expected_price_change_15min"]]
print(x_data.shape)
print(y_data.shape)

x_data_ndarray = x_data.values
y_data_ndarray = y_data.values

train_split = 0.85
num_train = int(train_split * len(x_data_ndarray))
num_test = len(x_data_ndarray)-num_train

x_train = x_data_ndarray[0:num_train]
x_test = x_data_ndarray[num_train:]
print(len(x_train) + len(x_test))

y_train = y_data_ndarray[0:num_train]
y_test = y_data_ndarray[num_train:]
print(len(y_train) + len(y_test))

num_x_signals = x_data_ndarray.shape[1]
print(num_x_signals)

num_y_signals = y_data_ndarray.shape[1]
print(num_y_signals)


x_scaler = MinMaxScaler()
print(x_scaler.fit(x_data_ndarray))

y_scaler = MinMaxScaler()
print(y_scaler.fit(y_data_ndarray))

# Scale training and test sets
x_train_scaled = x_scaler.transform(x_train)
x_test_scaled = x_scaler.transform(x_test)

y_train_scaled = y_scaler.transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

print("Min x:", np.min(x_train_scaled))
print("Max x:", np.max(x_train_scaled))
print("Min y:", np.min(y_train_scaled))
print("Max y:", np.max(y_train_scaled))


pass


