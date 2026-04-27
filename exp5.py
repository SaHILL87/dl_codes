# pyright: reportMissingModuleSource=false
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# 1) Load MNIST (labels not needed for autoencoder)
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# 2) Preprocess
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_test_img = x_test.copy()
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))

# 3) Build Autoencoder
encoder = models.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
])

decoder = models.Sequential([
    layers.Input(shape=(64,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(784, activation="sigmoid"),
])

autoencoder = models.Sequential([encoder, decoder])
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

# 4) Train
autoencoder.fit(x_train, x_train, epochs=3, batch_size=128, validation_split=0.1, verbose=2)

# 5) Evaluate reconstruction
test_loss = autoencoder.evaluate(x_test, x_test, verbose=0)
print("Test reconstruction loss:", round(float(test_loss), 6))

# 6) Show compression + sample reconstruction error
encoded = encoder.predict(x_test[:5], verbose=0)
recon = autoencoder.predict(x_test[:5], verbose=0)
mse = np.mean((x_test[:5] - recon) ** 2, axis=1)

print("Original dimension:", x_test.shape[1])
print("Encoded dimension:", encoded.shape[1])
print("MSE for first 5 test images:", np.round(mse, 6))

# 7) Plot original vs reconstructed output images
n_show = 5
orig = x_test_img[:n_show]
recon_img = recon.reshape((-1, 28, 28))[:n_show]

fig, axes = plt.subplots(2, n_show, figsize=(2 * n_show, 4))
for i in range(n_show):
    axes[0, i].imshow(orig[i], cmap="gray")
    axes[0, i].set_title("Original")
    axes[0, i].axis("off")

    axes[1, i].imshow(recon_img[i], cmap="gray")
    axes[1, i].set_title("Reconstructed")
    axes[1, i].axis("off")

plt.tight_layout()
plot_file = "exp5_output_plot.png"
plt.savefig(plot_file, dpi=120)
plt.close(fig)
