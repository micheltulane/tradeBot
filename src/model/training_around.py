__author__ = "Michel Tulane"
#File created 24-JUL-2019

# Generic imports
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# NN Model related imports
import tensorflow as tf
from tensorflow import keras
from keras import layers
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import   Input, Dense, LSTM, GRU, Embedding, Activation
# from tensorflow.python.keras.optimizers import RMSprop
# from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

# Execution variables
TRAIN_MODEL = True

tf.keras.backend.clear_session()  # Reset notebook state.



