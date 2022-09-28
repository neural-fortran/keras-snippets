"""
Create a minimal network with a reshape layer.
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

input_size = 784
output_shape = 28, 28, 1

model = keras.Sequential([
    keras.Input(input_size),
    layers.Reshape(output_shape)
])

model.summary()
model.save("keras_reshape.h5")
