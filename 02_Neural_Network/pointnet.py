import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt

VAL_SPLIT = 0.2
NUM_SAMPLE_POINTS = 50
BATCH_SIZE = 32
EPOCHS = 60
INITIAL_LR = 1e-3

def getData():
    ################################################################
    #TRAIN DATA
    ################################################################
    n=50
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
    train_fatures=tf.convert_to_tensor(data_x_split_train)
    train_labels=tf.convert_to_tensor(train_labels)
    print("read training data\n")
    # print(train_labels)
    # print(train_fatures)

    ################################################################
    #TEST DATA
    ################################################################
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
    test_fatures=tf.convert_to_tensor(data_x_split_test)
    test_labels=tf.convert_to_tensor(test_labels)
    print("read test data\n")

    # print(test_labels)
    # print(test_fatures)
    return train_fatures, train_labels, test_fatures,  test_labels


train_fatures, train_labels, test_fatures,  test_labels = getData()

def conv_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    x = layers.Conv1D(filters, kernel_size=1, padding="valid", name=f"{name}_conv")(x)
    x = layers.BatchNormalization(momentum=0.0, name=f"{name}_batch_norm")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)


def mlp_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    x = layers.Dense(filters, name=f"{name}_dense")(x)
    x = layers.BatchNormalization(momentum=0.0, name=f"{name}_batch_norm")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)


class OrthogonalRegularizer(keras.regularizers.Regularizer):
    """Reference: https://keras.io/examples/vision/pointnet/#build-a-model"""

    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.identity = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.identity))

    # def get_config(self):
    #     config = super(TransformerEncoder, self).get_config()
    #     config.update({"num_features": self.num_features, "l2reg_strength": self.l2reg})
    #     return config
def transformation_net(inputs: tf.Tensor, num_features: int, name: str) -> tf.Tensor:
    """
    Reference: https://keras.io/examples/vision/pointnet/#build-a-model.

    The `filters` values come from the original paper:
    https://arxiv.org/abs/1612.00593.
    """
    x = conv_block(inputs, filters=64, name=f"{name}_1")
    x = conv_block(x, filters=128, name=f"{name}_2")
    x = conv_block(x, filters=1024, name=f"{name}_3")
    x = layers.GlobalMaxPooling1D()(x)
    x = mlp_block(x, filters=512, name=f"{name}_1_1")
    x = mlp_block(x, filters=256, name=f"{name}_2_1")
    return layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=keras.initializers.Constant(np.eye(num_features).flatten()),
        activity_regularizer=OrthogonalRegularizer(num_features),
        name=f"{name}_final",
    )(x)


def transformation_block(inputs: tf.Tensor, num_features: int, name: str) -> tf.Tensor:
    transformed_features = transformation_net(inputs, num_features, name=name)
    transformed_features = layers.Reshape((num_features, num_features))(
        transformed_features
    )
    return layers.Dot(axes=(2, 1), name=f"{name}_mm")([inputs, transformed_features])

def get_shape_segmentation_model(num_points: int, num_classes: int) -> keras.Model:
    input_points = keras.Input(shape=(None, 3))

    # PointNet Classification Network.
    transformed_inputs = transformation_block(
        input_points, num_features=3, name="input_transformation_block"
    )
    features_64 = conv_block(transformed_inputs, filters=64, name="features_64")
    features_128_1 = conv_block(features_64, filters=128, name="features_128_1")
    features_128_2 = conv_block(features_128_1, filters=128, name="features_128_2")
    transformed_features = transformation_block(
        features_128_2, num_features=128, name="transformed_features"
    )
    features_512 = conv_block(transformed_features, filters=512, name="features_512")
    features_2048 = conv_block(features_512, filters=2048, name="pre_maxpool_block")
    global_features = layers.MaxPool1D(pool_size=num_points, name="global_features")(
        features_2048
    )
    global_features = tf.tile(global_features, [1, num_points, 1])

    # Segmentation head.
    segmentation_input = layers.Concatenate(name="segmentation_input")(
        [
            features_64,
            features_128_1,
            features_128_2,
            transformed_features,
            features_512,
            global_features,
        ]
    )
    segmentation_features = conv_block(
        segmentation_input, filters=128, name="segmentation_features"
    )
    outputs = layers.Conv1D(
        num_classes, kernel_size=1, activation="softmax", name="segmentation_head"
    )(segmentation_features)
    return keras.Model(input_points, outputs)

x, y = train_fatures, train_labels
total_training_examples=len(train_fatures)
num_points = x.shape[1]
num_classes = y.shape[-1]

segmentation_model = get_shape_segmentation_model(num_points, num_classes)
segmentation_model.summary()

training_step_size = total_training_examples // BATCH_SIZE
total_training_steps = training_step_size * EPOCHS
print(f"Total training steps: {total_training_steps}.")

lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[training_step_size * 15, training_step_size * 15],
    values=[INITIAL_LR, INITIAL_LR * 0.5, INITIAL_LR * 0.25],
)

steps = tf.range(total_training_steps, dtype=tf.int32)
lrs = [lr_schedule(step) for step in steps]

plt.plot(lrs)
plt.xlabel("Steps")
plt.ylabel("Learning Rate")
plt.show()

def run_experiment(epochs):

    segmentation_model = get_shape_segmentation_model(num_points, num_classes)
    segmentation_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    checkpoint_filepath = "/tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )

    history = segmentation_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[checkpoint_callback],
    )

    segmentation_model.load_weights(checkpoint_filepath)
    return segmentation_model, history


#segmentation_model, history = run_experiment(epochs=EPOCHS)