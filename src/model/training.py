__author__ = "Michel Tulane"
#File created 24-JUL-2019

# Generic imports
import os
import time
import numpy as np
import json
import pandas as pd
import feather
import matplotlib.pyplot as plt
from datetime import datetime

# NN Model related imports
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from tensorflow import keras
from keras import layers
from tensorflow.python.keras.models import Sequential, model_from_json
from tensorflow.python.keras.layers import Input, Dense, LSTM, GRU, SimpleRNN, Embedding, Activation, CuDNNLSTM
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

# Execution variables
TRAIN_MODEL = True

DATA_PATH = "../../data/5min_fetched/"
TRAINING_LOG_PATH = "../../logs/training/"
MODEL_CHECKPOINT_PATH = TRAINING_LOG_PATH + "checkpoint/model.keras"
TENSORBOARD_PATH = TRAINING_LOG_PATH + "/tensorboard"
BUY_THRESHOLD = 0.1
SELL_THRESHOLD = -0.1
tf.keras.backend.clear_session()  # Reset notebook state.

# Execution timestamp (for lod file naming...)
now = int(time.time())

# Import data
print("Importing data from disk...")
# df = pd.read_excel(os.path.abspath(DATA_PATH + "/train_test_data.xlsx"))
df = feather.read_dataframe(os.path.abspath(DATA_PATH + "/USDT_BTC.feather"))

print("Dataframe shame: {shape}".format(shape=df.shape))
x_data = df.drop(columns=["day", "time", "USDT_BTC_expected_price_change_15min", "USDT_BTC_expected_price_change_5min"])
y_data = df[["USDT_BTC_expected_price_change_5min"]]
target_names = ["BTC_expected_price_change_5min"]
print(x_data.shape)
print(y_data.shape)

x_data_ndarray = x_data.values
y_data_ndarray = y_data.values

train_split = 0.9
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

print("TRAINING SET SCALED:")
print("Min x :", np.min(x_train_scaled))
print("Max x :", np.max(x_train_scaled))
print("Min y :", np.min(y_train_scaled))
print("Max y: ", np.max(y_train_scaled))
print("Shape x: ", x_train_scaled.shape)
print("Shape y: ", y_train_scaled.shape)


print("TEST SET SCALED:")
print("Min x :", np.min(x_test_scaled))
print("Max x :", np.max(x_test_scaled))
print("Min y :", np.min(y_test_scaled))
print("Max y :", np.max(y_test_scaled))
print("Shape x: ", x_test_scaled.shape)
print("Shape y: ", y_test_scaled.shape)


def batch_generator(batch_size, sequence_length):
    """
    Generator function for creating random batches of training-data.
    """

    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(num_train - sequence_length)

            # Copy the sequences of data starting at this index.
            x_batch[i] = x_train_scaled[idx:idx + sequence_length]
            y_batch[i] = y_train_scaled[idx:idx + sequence_length]

        yield (x_batch, y_batch)


batch_size = 64
sequence_length = 1024
generator = batch_generator(batch_size=batch_size,
                            sequence_length=sequence_length)

validation_data = (np.expand_dims(x_test_scaled, axis=0),
                   np.expand_dims(y_test_scaled, axis=0))

model = Sequential()
lstm_units = 512
model.add(CuDNNLSTM(units=lstm_units,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
model.add(Dense(num_y_signals, activation='sigmoid'))

warmup_steps = 64


def loss_mse_warmup(y_true, y_pred):
    """
    Calculate the Mean Squared Error between y_true and y_pred,
    but ignore the beginning "warmup" part of the sequences.

    y_true is the desired output.
    y_pred is the model's output.
    """

    # The shape of both input tensors are:
    # [batch_size, sequence_length, num_y_signals].

    # Ignore the "warmup" parts of the sequences
    # by taking slices of the tensors.
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]

    # These sliced tensors both have this shape:
    # [batch_size, sequence_length - warmup_steps, num_y_signals]

    # Calculate the MSE loss for each value in these tensors.
    # This outputs a 3-rank tensor of the same shape.
    loss = tf.losses.mean_squared_error(labels=y_true_slice,
                                        predictions=y_pred_slice)

    # Keras may reduce this across the first axis (the batch)
    # but the semantics are unclear, so to be sure we use
    # the loss across the entire tensor, we reduce it to a
    # single scalar with the mean function.
    loss_mean = tf.reduce_mean(loss)

    return loss_mean

optimizer = RMSprop(lr=1e-3)

model.compile(loss=loss_mse_warmup, optimizer=optimizer)

model.summary()

callback_checkpoint = ModelCheckpoint(filepath=MODEL_CHECKPOINT_PATH,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)

callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=5, verbose=1)

callback_tensorboard = TensorBoard(log_dir=TENSORBOARD_PATH,
                                   histogram_freq=0,
                                   write_graph=False)

callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       min_lr=1e-4,
                                       patience=0,
                                       verbose=1)

callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard,
             callback_reduce_lr]

if TRAIN_MODEL:
    model.fit_generator(generator=generator,
                        epochs=30,
                        steps_per_epoch=100,
                        validation_data=validation_data,
                        callbacks=callbacks)

try:
    model.load_weights(MODEL_CHECKPOINT_PATH)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)

result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
                        y=np.expand_dims(y_test_scaled, axis=0))

print("loss (test-set):", result)


def plot_comparison(start_idx, length=100, train=True):
    """
    Plot the predicted and true output-signals.

    :param start_idx: Start-index for the time-series.
    :param length: Sequence-length to process and plot.
    :param train: Boolean whether to use training- or test-set.
    """

    if train:
        # Use training-data.
        x = x_train_scaled
        y_true = y_train
    else:
        # Use test-data.
        x = x_test_scaled
        y_true = y_test

    # End-index for the sequences.
    end_idx = start_idx + length

    # Select the sequences from the given start-index and
    # of the given length.
    x = x[start_idx:end_idx]
    y_true = y_true[start_idx:end_idx]

    # Input-signals for the model.
    x = np.expand_dims(x, axis=0)

    # Use the model to predict the output-signals.
    y_pred = model.predict(x)

    # The output of the model is between 0 and 1.
    # Do an inverse map to get it back to the scale
    # of the original data-set.
    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])

    # For each output-signal.
    for signal in range(len(target_names)):
        # Get the output-signal predicted by the model.
        signal_pred = y_pred_rescaled[:, signal]

        # Get the true output-signal from the data-set.
        signal_true = y_true[:, signal]

        # Make the plotting-canvas bigger.
        plt.figure(figsize=(15, 5))

        # Plot and compare the two signals.
        plt.plot(signal_true, label='true')
        plt.plot(signal_pred, label='pred')

        # Plot grey box for warmup-period.
        p = plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)

        # Plot labels etc.
        plt.ylabel(target_names[signal])
        plt.legend()
        plt.show()


# plot_comparison(start_idx=0, length=20000, train=False)


# test another simulation method (compute entire prediction at once...)
def simulate_model_on_history(hist_length=100, start_idx=0, length=100, train=False):
    """
    Parse historical data and use trained model to make buy/sell decisions

    :param hist_length: length of previous data used to predict values
    :param start_idx: Start-index for the time-series.
    :param length: Sequence-length to process and plot.
    :param train: Boolean whether to use training- or test-set.
    """
    account_value = 1000
    account_is_usdt = True
    buy_cnt = 0
    sell_cnt = 0
    hodl_cnt = 0

    if train:
        # Use training-data.
        x = x_train_scaled
    else:
        # Use test-data.
        x = x_test_scaled

    # predict all data range
    x_all_exp = np.expand_dims(x, axis=0)
    y_all_pred = model.predict(x_all_exp)
    y = y_scaler.inverse_transform(y_all_pred[0])

    start_value = account_value
    for i in range(length):
        hist_time = start_idx + i
        time = hist_time + hist_length

        print("Hist index: {hist}, Time: {time}".format(hist=hist_time, time=time))

        # Input-signals for the model.
        x_sim = x[hist_time:time]
        x_sim_exp = np.expand_dims(x_sim, axis=0)
        x_sim_rescaled = x_scaler.inverse_transform(x_sim_exp[0])

        # Use already predicted data
        y_pred_rescaled = y[hist_time:time]

        # check last prediction value for price increase or decrease
        last_prediction = y_pred_rescaled[-1][0]
        current_price = x_sim_rescaled[-1][0]

        print("Last prediction: {pred}".format(pred=last_prediction))
        print("Current price: {price}".format(price=current_price))

        if (last_prediction > BUY_THRESHOLD) and account_is_usdt and (i != length-1):
            print("Buying")
            account_value = account_value/current_price
            account_is_usdt = False
            buy_cnt += 1

        elif ((last_prediction < SELL_THRESHOLD) and not account_is_usdt) or (i == length-1 and not account_is_usdt):
            print("Selling")
            account_value = account_value*current_price
            account_is_usdt = True
            sell_cnt += 1

        else:
            print("Hodl")
            hodl_cnt += 1

    print("Start value: {startval} , end value: {endval}".format(startval=start_value, endval=account_value))
    print("Buy count: {buys} , Sell count: {sells}, Hodl count: {hodls}".format(buys=buy_cnt,
                                                                                sells=sell_cnt,
                                                                                hodls=hodl_cnt))


simulate_model_on_history(hist_length=sequence_length, start_idx=0, length=20000, train=False)


def save_model_package_info():
    print("Saving model package info...")
    with open(str(now) + "_model_save.json", "w") as json_file:
        json_file.write(model.to_json())
        joblib.dump([x_scaler, y_scaler], TRAINING_LOG_PATH + str(now) + "_scalers.joblib")


save_model_package_info()


pass


