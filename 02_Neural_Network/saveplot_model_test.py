import tensorflow as tf
from tensorflow import keras, convert_to_tensor
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Dropout, LSTM
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
from kerastuner.tuners import RandomSearch
import kerastuner as kt
from kerastuner.engine.hyperparameters import HyperParameters
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd

import pickle



input = tf.keras.Input(shape=(50, 3), dtype='int32', name='input')
x = tf.keras.layers.Flatten(input_shape=(50, 3))(input)

x = tf.keras.layers.Dense(96, activation='relu')(x)
x = tf.keras.layers.Dense(96, activation='relu')(x)
x = tf.keras.layers.Dense(160, activation='elu')(x)

output = tf.keras.layers.Dense(50, activation='sigmoid', name='output')(x)
model = tf.keras.Model(inputs=[input], outputs=[output])
dot_img_file = 'model_1.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)