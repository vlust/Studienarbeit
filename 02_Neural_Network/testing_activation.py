import tensorflow as tf
import numpy as np

a = tf.constant([-20, -1.0, 1.0, 1.0, 20], dtype = tf.float32)
b = tf.keras.activations.sigmoid(a)
b.numpy()
c=np.round(b)
print(c)