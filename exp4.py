# pyright: reportMissingModuleSource=false
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# 1) Load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2) Preprocess
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., np.newaxis]  # (N, 28, 28, 1)
x_test = x_test[..., np.newaxis]

# 3) Build CNN model
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax"),
])

# 4) Compile + Train
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.1, verbose=2)

# 5) Evaluate
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print("Test accuracy:", round(acc, 4))

# 6) Predict first 5 test images
pred = model.predict(x_test[:5], verbose=0).argmax(axis=1)
print("Predictions:", pred)
print("Actual:", y_test[:5])
