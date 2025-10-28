# Verify-TensorFlow-Installation---Functionality
This guide provides a clear, step-by-step process for testing your TensorFlow installation using a simple Python script. It walks you through setting up the required environment, running the test code, and verifying successful computation and model training. Ideal for beginners and developers, it ensures that TensorFlow and NumPy are Installed.

# A Step-by-Step Guide to Using the TensorFlow Test Code

## 📋 Prerequisites

-   Python installed on your system (preferably the latest version)
-   TensorFlow installed (`pip install tensorflow`)
-   NumPy installed (`pip install numpy`)
-   A code editor or IDE (Integrated Development Environment) of your
    choice

------------------------------------------------------------------------

## 🧩 Step 1: Save the Code

Copy the code below and save it as `tensorflow_test.py`:

``` python
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
```

------------------------------------------------------------------------

## ▶️ Step 2: Run the Code

1.  Open a terminal or command prompt.\
2.  Navigate to the directory where you saved the file:

``` bash
cd path/to/your/directory
```

3.  Run the code:

``` bash
python tensorflow_test.py
```

If you're using a virtual environment, activate it before running the
code.

------------------------------------------------------------------------

## ✅ What to Expect

If everything is set up correctly, you should see:

1.  **Compute OK, sum = `<some_value>`** --- confirms TensorFlow
    computation works.\
2.  **Train OK, final loss: `<some_value>`** --- confirms model training
    works.

------------------------------------------------------------------------

## 🧰 Troubleshooting

-   If TensorFlow or NumPy is not installed:

    ``` bash
    pip install tensorflow numpy
    ```

-   If the file cannot be found, ensure you're in the correct
    directory.\

-   For any other issues, double-check installations or environment
    setup.

------------------------------------------------------------------------

## 🧠 Summary

This simple TensorFlow test script verifies that your environment is
correctly configured for deep learning tasks using TensorFlow and NumPy.

