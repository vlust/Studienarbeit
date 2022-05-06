
import tensorflow as tf
from tensorflow import keras, convert_to_tensor
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Dropout, LSTM
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from kerastuner.tuners import RandomSearch
import kerastuner as kt
from kerastuner.engine.hyperparameters import HyperParameters
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
from VL_trackGenerator import *

import pickle

import random

def visualize_all(trackdata, conedata):
                #   sort track data             
                x, y = visualize_track(trackdata)
                yellow_x, yellow_y, blue_x, blue_y=visualize_cones(conedata)

                plt.plot(x, y)
                plt.plot(yellow_x,yellow_y,'*',color='orange')
                plt.plot(blue_x,blue_y,'*',color='blue')

                plt.axis('scaled')
                plt.show()

def show_track(trackdata):
        x, y = visualize_track(trackdata)
        plt.plot(x, y)
        plt.axis('scaled')
        plt.show()

def visualize_track(track_data):
        return map(list, zip(*track_data))
        
def visualize_cones(conedata):
        yellow_cones = [x for x in conedata if x[2]==1]
        blue_cones = [x for x in conedata if x[2]==2]
        yellow_x, yellow_y, _, _ = map(list, zip(*yellow_cones))
        blue_x, blue_y, _, _ = map(list, zip(*blue_cones))

        return yellow_x, yellow_y, blue_x, blue_y
        
def show_cones(conedata):
        yellow_x, yellow_y, blue_x, blue_y=visualize_cones(conedata)
        plt.plot(yellow_x,yellow_y,'*',color='orange')
        plt.plot(blue_x,blue_y,'*',color='blue')
        plt.axis('scaled')
        plt.show()

def getData():
    n=50
    ################################################################
    #TEST DATA
    ################################################################
    df_test = pd.read_csv('C:/Users/Anwender/Desktop/Studienarbeit_Data/zeros_filled_shuffled_3class/test.csv')
    for i in range(len(df_test['target'])):
        if df_test['target'][i] == 2:
            df_test.iat[i, df_test.columns.get_loc('target')] = 0
    print(df_test)
    # df_test['x']=df_test['x'].div(200)
    # df_test['y']=df_test['x'].div(200)
    df_test['color']=df_test['color'].div(2)

    data_x_y_test=df_test.values
    data_x_y_test = np.hsplit(data_x_y_test, [3,4])
    data_x_test=data_x_y_test[0]
    data_y_test=data_x_y_test[1]

    data_x_split_test = [data_x_test[x:x+n] for x in range(0, len(data_x_test), n)]
    data_y_split_test = [data_y_test[x:x+n] for x in range(0, len(data_y_test), n)]
    test_labels=[np.concatenate(a) for a in data_y_split_test]
    
    test_fatures=data_x_split_test
    #test_fatures=convert_to_tensor(data_x_split_test)
    #test_labels=convert_to_tensor(test_labels)
    print("read test data\n")

    # print(test_labels)
    # print(test_fatures)
    return  test_fatures,  test_labels

# test_features,  test_labels = getData()
# test_features_x=[]
# test_features_y=[]
# for track in test_features:
#     test_features_x.append([x[0] for x in track])
#     test_features_y.append([x[1] for x in track])




# sum1=[x.sum() for x in test_labels]


# # plt.scatter(test_features_x[0],test_features_y[0])
# # plt.show()

# predictions = np.loadtxt('test2.txt', dtype=int)

# sum2=50-predictions.sum(axis=1)

# # print(sum1.shape)
# # print(sum2.shape)
# indexs=[]

# for i in range(len(sum1)):

#     if sum1[i].sum() == sum2[i].sum() and min(map(abs,test_features_y[i]))!=0 and sum2[i]<50:
#         indexs.append(i)
# print(len(indexs))

# sample_list = random.choices(indexs, k=50)
# print()
# for index in indexs:
#     x1=[]
#     x2=[]
#     y1=[]
#     y2=[]
#     for i in range(len(predictions[index])):
#         if predictions[index][i]:
#             x1.append(test_features_x[index][i])
#             y1.append(test_features_y[index][i])
#         else:
#             x2.append(test_features_x[index][i])
#             y2.append(test_features_y[index][i])

#     plt.clf()
#     fig, axs = plt.subplots(1)
#     axs.scatter(x1, y1)
#     axs.scatter(x2, y2)
#     plt.axis('scaled')
#     plt.savefig(f'Analyse/images/image_{index}.png')

df = pd.read_csv('history.csv')
print(df)
x=[1, 2,3, 4, 5]
y1=[]
y2=[]
for i in range(5):
    y1.append(df['loss'][i])
    y2.append(df['accuracy'][i])

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(x, y1, 'g-')
ax2.plot(x, y2, 'b-')

ax1.set_xlabel('Epochen')
ax1.set_ylabel('Loss', color='g')
ax2.set_ylabel('Accuracy', color='b')

plt.show()