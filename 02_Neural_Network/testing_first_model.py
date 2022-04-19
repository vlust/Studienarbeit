# TensorFlow and tf.keras
import tensorflow as tf

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
    df = pd.read_csv('01_TrackGenerator_VL/tracks/training.csv')
    df['x']=df['x'].div(200)
    df['y']=df['x'].div(200)
    df['color']=df['color'].div(2)

    data_x_y=df.values
    data_x_y = np.hsplit(data_x_y, [3,4])
    data_x=data_x_y[0]
    data_y=data_x_y[1]

    data_x_split_train = [data_x[x:x+80] for x in range(0, len(data_x), 80)]
    data_y_split_train = [data_y[x:x+80] for x in range(0, len(data_y), 80)]
    train_labels=[np.concatenate(a) for a in data_y_split_train]
    train_fatures=tf.convert_to_tensor(data_x_split_train)
    train_labels=tf.convert_to_tensor(train_labels)
    print("read training data\n")
    # print(train_labels)
    # print(train_fatures)

    ################################################################
    #TEST DATA
    ################################################################
    df_test = pd.read_csv('01_TrackGenerator_VL/tracks/test.csv')
    df_test['x']=df_test['x'].div(200)
    df_test['y']=df_test['x'].div(200)
    df_test['color']=df_test['color'].div(2)

    data_x_y_test=df_test.values
    data_x_y_test = np.hsplit(data_x_y_test, [3,4])
    data_x_test=data_x_y_test[0]
    data_y_test=data_x_y_test[1]

    data_x_split_test = [data_x_test[x:x+80] for x in range(0, len(data_x_test), 80)]
    data_y_split_test = [data_y_test[x:x+80] for x in range(0, len(data_y_test), 80)]
    test_labels=[np.concatenate(a) for a in data_y_split_test]
    test_fatures=tf.convert_to_tensor(data_x_split_test)
    test_labels=tf.convert_to_tensor(test_labels)
    print("read test data\n")

    # print(test_labels)
    # print(test_fatures)
    return train_fatures, train_labels, test_fatures,  test_labels

train_fatures, train_labels, test_fatures,  test_labels = getData()

# fashion_mnist = tf.keras.datasets.fashion_mnist

# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# # plt.figure()
# # plt.imshow(train_images[0])
# # plt.colorbar()
# # plt.grid(False)
# # plt.show()

# train_images = train_images / 255.0

# test_images = test_images / 255.0

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# # plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(80, 3)),
    #tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128, activation='elu'),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(128, activation='elu'),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(80, activation='sigmoid')
])

opt = tf.keras.optimizers.RMSprop(learning_rate=0.00001)


model.compile(optimizer=opt,
              loss=tf.keras.losses.BinaryCrossentropy(
    from_logits=False,),
              metrics=['categorical_accuracy'])

model.fit(train_fatures, train_labels, batch_size=50, epochs=100)

test_loss, test_acc = model.evaluate(test_fatures,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)