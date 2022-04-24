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



def getData2():
    ################################################################
    #TRAIN DATA
    ################################################################


    ################################################################
    #TEST DATA
    ################################################################
    n=50
    df_test = pd.read_csv('C:/Users/Anwender/Desktop/Studienarbeit_Data/zeros_filled_shuffled/test.csv')
    df_test['x']=df_test['x'].div(200)
    df_test['y']=df_test['x'].div(200)
    df_test['color']=df_test['color'].div(2)

    data_x_y_test=df_test.values
    data_x_y_test = np.hsplit(data_x_y_test, [3,4])
    data_x_test=data_x_y_test[0]
    data_y_test=data_x_y_test[1]

    data_x_split_test = [data_x_test[x:x+n] for x in range(0, len(data_x_test), n)]
    data_y_split_test = [data_y_test[x:x+n] for x in range(0, len(data_y_test), n)]
    test_labels=[np.concatenate(a) for a in data_y_split_test]
    test_fatures=convert_to_tensor(data_x_split_test)
    test_labels=convert_to_tensor(test_labels)
    print("read test data\n")

    # print(test_labels)
    # print(test_fatures)
    return test_fatures,  test_labels
test_fatures,  test_labels= getData2()


def build_model2():
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-5,
    decay_steps=1000,
    decay_rate=0.9)

    model = keras.Sequential()
    #model.add(Normalization(axis=None))

    model.add(Flatten(input_shape=(50, 3)))
    #model.add(Dense(128, activation='elu'))
    model.add(Dense(96, activation='elu'))
    model.add(Dense(96, activation='relu'))
    model.add(Dense(160, activation='elu'))
    #model.add(Dense(64, activation='relu'))
    
        # Dense(64, activation='elu'),
        # Dropout(rate=0.2),
    
    model.add(Dense(50, activation='sigmoid'))
   

    opt = keras.optimizers.Adam(learning_rate=lr_schedule)


    model.compile(optimizer=opt,
                loss=keras.losses.BinaryCrossentropy(
        from_logits=False,),
                metrics=['categorical_accuracy'])
    return model

model2=build_model2()
filepath="C:/Users/Anwender/Desktop/MODEL_1"
model = keras.models.load_model(filepath)
filepath2="IMAGE.png"
plot_model(model2, to_file=filepath2)
# test_loss, test_acc = model.evaluate(test_fatures,  test_labels, verbose=2)

# print('\nTest accuracy:', test_acc)


