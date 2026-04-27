import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, datasets

# 1. Load and prepare data (Flattened for simplicity)
(x_train, _), (x_test, _) = datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

# 2. Build the Simple Autoencoder
model = models.Sequential([
    # Encoder: Compress 784 pixels down to 32 
    layers.Dense(128, activation="relu", input_shape=(784,)),
    layers.Dense(32, activation="relu"),

    # Decoder: Reconstruct 32 back up to 784
    layers.Dense(128, activation="relu"),
    layers.Dense(784, activation="sigmoid")
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# 3. Train (Input and Target are both 'x_train')
model.fit(x_train, x_train, epochs=10, batch_size=256, validation_data=(x_test, x_test))

# 4. Test and Visualize
preds = model.predict(x_test)

plt.figure(figsize=(10, 4))
for i in range(5):
    # Original
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.axis("off")

    # Reconstructed
    plt.subplot(2, 5, i + 6)
    plt.imshow(preds[i].reshape(28, 28), cmap="gray")
    plt.axis("off")
plt.show()