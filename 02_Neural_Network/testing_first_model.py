# TensorFlow and tf.keras
#import tensorflow as tf
from tensorflow import keras, convert_to_tensor
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Dropout
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

def split_dataframe(df, chunk_size = 4): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks


#df = pd.read_csv('02_Neural_Network/data.csv')


# #print(df.to_string()) 
# newdf=split_dataframe(df)

# print(newdf)
#dataf=([[[1,2,3,4],[1,2,3,4]],[[1,2,3,4],[1,2,3,4]]])
#newdata=np.split(my_data, [1,3])



#my_data = np.genfromtxt('01_TrackGenerator_VL/tracks/ALL.csv', delimiter=',', encoding="utf8",skip_header=1)



def getData():
    ################################################################
    #TRAIN DATA
    ################################################################
    n=80
    df = pd.read_csv('C:/Users/Anwender/Desktop/Studienarbeit_Data/zeros_filled/training.csv')
    df['x']=df['x'].div(200)
    df['y']=df['x'].div(200)
    df['color']=df['color'].div(2)

    data_x_y=df.values
    data_x_y = np.hsplit(data_x_y, [3,4])
    data_x=data_x_y[0]
    data_y=data_x_y[1]

    data_x_split_train = [data_x[x:x+n] for x in range(0, len(data_x), n)]
    data_y_split_train = [data_y[x:x+n] for x in range(0, len(data_y), n)]
    train_labels=[np.concatenate(a) for a in data_y_split_train]
    train_fatures=convert_to_tensor(data_x_split_train)
    train_labels=convert_to_tensor(train_labels)
    print("read training data\n")
    # print(train_labels)
    # print(train_fatures)

    ################################################################
    #TEST DATA
    ################################################################
    df_test = pd.read_csv('C:/Users/Anwender/Desktop/Studienarbeit_Data/zeros_filled/test.csv')
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
    return train_fatures, train_labels, test_fatures,  test_labels


train_fatures, train_labels, test_fatures,  test_labels = getData()


def build_model():
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=10000,
    decay_rate=0.6)

    model = keras.Sequential([
        Flatten(input_shape=(80, 3)),
        Dense(128, activation='elu'),
        Dropout(rate=0.2),
        Dense(128, activation='elu'),
        Dropout(rate=0.2),

        Dense(80, activation='sigmoid')
    ])

    opt = keras.optimizers.RMSprop(learning_rate=lr_schedule)


    model.compile(optimizer=opt,
                loss=keras.losses.BinaryCrossentropy(
        from_logits=False,),
                metrics=['categorical_accuracy'])
    return model

model=build_model()
model.fit(train_fatures, train_labels, batch_size=50, epochs=100)

test_loss, test_acc = model.evaluate(test_fatures,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)