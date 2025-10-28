import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Quick compute
x = tf.random.normal([1000, 1000])
y = tf.reduce_sum(x)
print("Compute OK, sum =", y.numpy())

# Tiny model
model = keras.Sequential([
    layers.Dense(8, activation="relu", input_shape=(4,)),
    layers.Dense(1)
])
model.compile(optimizer="adam", loss="mse")

X = np.random.randn(256, 4).astype("float32")
y = np.random.randn(256, 1).astype("float32")

hist = model.fit(X, y, epochs=2, verbose=0)
print("Train OK, final loss:", hist.history["loss"][-1])

exit()