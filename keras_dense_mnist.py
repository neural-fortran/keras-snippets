"""
Adapted from https://keras.io/examples/vision/mnist_convnet/
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Model / data parameters
num_classes = 10
training_set_size = 60000
testing_set_size = 10000
image_size = 784

# Load the data, split between train and test sets, and scale to [0, 1] range
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (training_set_size, image_size)).astype("float32") / 255
x_test = np.reshape(x_test, (testing_set_size, image_size)).astype("float32") / 255

# Make sure images have shape (image_size, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential([
    keras.Input(image_size),
    layers.Dense(30, activation="sigmoid"),
    layers.Dense(num_classes, activation="softmax"),
])

model.summary()

batch_size = 128
epochs = 10

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1
)

model.save("keras_dense_mnist.h5")
