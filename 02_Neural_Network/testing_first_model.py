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



my_data = np.genfromtxt('01_TrackGenerator_VL/tracks/ALL.csv', delimiter=',', encoding="utf8",skip_header=1)
data_x_y=my_data/200
print(data_x_y)
data_x_y = np.hsplit(data_x_y, [3,4])
data_x=data_x_y[0]
data_y=data_x_y[1]
data_y*=200
#print(data_y)
# print((data_x.shape))
# print(data_y.shape)
data_x_split = [data_x[x:x+80] for x in range(0, len(data_x), 80)]
data_y_split = [data_y[x:x+80] for x in range(0, len(data_y), 80)]

newlabels=[np.concatenate(a) for a in data_y_split]


data_fatures=tf.convert_to_tensor(data_x_split)
data_labels=tf.convert_to_tensor(newlabels)
# print(data_labels)
# print(data_fatures)



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
    tf.keras.layers.Dense(256, activation='elu'),
    #tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Dense(128, activation='elu'),
    tf.keras.layers.Dense(80, activation='sigmoid')
])

opt = tf.keras.optimizers.RMSprop(learning_rate=0.00005)


model.compile(optimizer=opt,
              loss=tf.keras.losses.BinaryCrossentropy(
    from_logits=False,),
              metrics=['categorical_accuracy'])

model.fit(data_fatures, data_labels, batch_size=50, epochs=20)

test_loss, test_acc = model.evaluate(data_fatures,  data_labels, verbose=2)

print('\nTest accuracy:', test_acc)