__author__ = "Michel Tulane"
#File created 13-OCT-2018

# Generic imports
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tradeBot.src.utils.datautils as dut

# NN Model related imports
from sklearn import preprocessing as pp
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, LSTM, GRU, Embedding, Activation
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

# Execution variables
TRAIN_MODEL = False

# Load ETH/BTC Poloniex exchange data (hourly)
filename = os.path.abspath('../../data/Poloniex_ETHBTC_1h.csv')
exch_data = dut.read_cryptodatadownload_csv(filename=filename)

# Take date string and generate lists of days of the week and hour
days = []
hours = []
for index, hour in exch_data['Date'].iteritems():
    date_obj = datetime.strptime(hour, '%Y-%m-%d %I-%p')
    days.append(date_obj.weekday())
    hours.append(date_obj.hour)

# Append created lists to the dataframe
exch_data['Hour'] = hours
exch_data['Day'] = days

# Delete unused list objects
del hours
del days

# Generate target data (hourly close price, shifted back 1hr)
target_names = ['Close']
shift_steps = 1
exch_data['Prediction'] = exch_data[target_names[0]].shift(+shift_steps)

# Plot Closing prices(part of input data) and predicted future closing prices (output data)
# ax = exch_data.plot( y='Close', color='red', linewidth=1)
# exch_data.plot(y='Prediction', color='blue', linewidth=1, ax=ax)
# plt.show()

# Create input and output numpy arrays
# Input signals (in order) : Day, Hour, Open, High, Low , Close, Volume From, Volume To

x_data = np.rot90(np.array([np.array(exch_data['Day'].values[shift_steps:-1], dtype=np.float64),
                  np.array(exch_data['Hour'].values[shift_steps:-1], dtype=np.float64),
                  np.array(exch_data['Open'].values[shift_steps:-1], dtype=np.float64),
                  np.array(exch_data['High'].values[shift_steps:-1], dtype=np.float64),
                  np.array(exch_data['Low'].values[shift_steps:-1], dtype=np.float64),
                  np.array(exch_data['Close'].values[shift_steps:-1], dtype=np.float64),
                  np.array(exch_data['Volume From'].values[shift_steps:-1], dtype=np.float64),
                  np.array(exch_data['Volume To'].values[shift_steps:-1], dtype=np.float64)]))

y_data = np.array(exch_data['Prediction'].values[shift_steps:-1], dtype=np.float64)
y_data = y_data.reshape((len(y_data), 1))

# Flip x and y data to get back in chronological order
x_data = np.flip(x_data, axis=0)
y_data = np.flip(y_data, axis=0)

print("Shape:", x_data.shape)
print("Type:", x_data.dtype)

print("Shape:", y_data.shape)
print("Type:", y_data.dtype)

# Split training and testing data
train_split = 0.9
num_data = len(x_data)
num_train = int(train_split * num_data)
num_test = num_data - num_train

x_train = x_data[0:num_train]
x_test = x_data[num_train:]
y_train = y_data[0:num_train]
y_test = y_data[num_train:]
num_x_signals = x_data.shape[1]
num_y_signals = y_data.shape[1]

# Standardize data......................................................................................................
# Create input and output scalers
x_st_scaler = pp.StandardScaler()
y_st_scaler = pp.StandardScaler()

# Train standard scalers on whole data
x_st_scaler.fit(x_data)
y_st_scaler.fit(y_data)

# Apply trained standard scalers on all data
x_data_stand = x_st_scaler.transform(x_data)
x_train_stand = x_st_scaler.transform(x_train)
x_test_stand = x_st_scaler.transform(x_test)

y_data_stand = y_st_scaler.transform(y_data)
y_train_stand = y_st_scaler.transform(y_train)
y_test_stand = y_st_scaler.transform(y_test)


# Normalize data........................................................................................................
# Create input and output scalers
x_mm_scaler = pp.MinMaxScaler()
y_mm_scaler = pp.MinMaxScaler()

# Train scalers on standardized data
x_mm_scaler.fit(x_data_stand)
y_mm_scaler.fit(y_data_stand)

# Normalize training and test data
x_data_scaled = x_mm_scaler.transform(x_data_stand)
x_train_scaled = x_mm_scaler.transform(x_train_stand)
x_test_scaled = x_mm_scaler.transform(x_test_stand)

y_data_scaled = y_mm_scaler.transform(y_data_stand)
y_train_scaled = y_mm_scaler.transform(y_train_stand)
y_test_scaled = y_mm_scaler.transform(y_test_stand)


# Generate random training batches of training data
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

# Batch size: number of sequences to use in one iteration (mini-batch mode)
# Tune this parameter depending on system load during training
batch_size = 128
# Equivalent to 4 weeks with 1hr timesteps
sequence_length = 24*7*4
generator = batch_generator(batch_size=batch_size,
                            sequence_length=sequence_length)
x_batch, y_batch = next(generator)
print(x_batch.shape)
print(y_batch.shape)

validation_data = (np.expand_dims(x_test_scaled, axis=0),
                   np.expand_dims(y_test_scaled, axis=0))

# Create RNN model
model = Sequential()
model.add(LSTM(units=128,
               return_sequences=True,
               input_shape=(None, num_x_signals,)))

model.add(Dense(num_y_signals, activation='relu'))

warmup_steps = 50


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


# Compile model
optimizer = RMSprop(lr=1e-3)
model.compile(loss=loss_mse_warmup, optimizer=optimizer)
model.summary()

# Create callbacks for writing checkpoints during training
path_checkpoint = os.path.abspath('../../logs/training/model_checkpoint.keras')
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)

callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=5, verbose=1)

callback_tensorboard = TensorBoard(log_dir=os.path.abspath('../../logs/training/'),
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
    # Train the neural net
    model.fit_generator(generator=generator,
                        epochs=20,
                        steps_per_epoch=100,
                        validation_data=validation_data,
                        callbacks=callbacks)
else:
    # Load savec weights
    try:
        model.load_weights(path_checkpoint)
    except Exception as error:
        print("Error trying to load checkpoint.")
        print(error)


# Test model loss on test-set
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
    y_pred_rescaled = y_st_scaler.inverse_transform(y_mm_scaler.inverse_transform(y_pred[0]))

    print(y_pred_rescaled.shape)
    print(y_true.shape)
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


plot_comparison(start_idx=5000, length=10000, train=True)

pass

