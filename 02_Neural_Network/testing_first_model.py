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

my_data = np.genfromtxt('02_Neural_Network/data.csv', delimiter=',', encoding="utf8",skip_header=1)


#df = pd.read_csv('02_Neural_Network/data.csv')
# #print(df.to_string()) 
# newdf=split_dataframe(df)

# print(newdf)
#dataf=([[[1,2,3,4],[1,2,3,4]],[[1,2,3,4],[1,2,3,4]]])
#newdata=np.split(my_data, [1,3])

newdata = [my_data[x:x+2] for x in range(0, len(my_data), 2)]
#
data=tf.convert_to_tensor(newdata)
print(data)
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

# # plt.figure(figsize=(10,10))
# # for i in range(25):
# #     plt.subplot(5,5,i+1)
# #     plt.xticks([])
# #     plt.yticks([])
# #     plt.grid(False)
# #     plt.imshow(train_images[i], cmap=plt.cm.binary)
# #     plt.xlabel(class_names[train_labels[i]])
# # plt.show()

# model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10)
# ])

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# model.fit(train_images, train_labels, epochs=10)

# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

# print('\nTest accuracy:', test_acc)