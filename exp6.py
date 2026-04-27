# LSTM implementation

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Create sine wave data
time = np.linspace(0, 100, 1000)
data = np.sin(time)

# Prepare sequences
X = []
y = []
seq_length = 50

for i in range(len(data) - seq_length):
    X.append(data[i:i+seq_length])
    y.append(data[i+seq_length])

X = np.array(X).reshape(-1, seq_length, 1)
y = np.array(y)

# Split data
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, input_shape=(seq_length, 1)),
    tf.keras.layers.Dense(1)
])

# Compile
model.compile(optimizer='adam', loss='mse')

# Train
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Predict
pred = model.predict(X_test)

# Plot results
plt.plot(y_test[:100], label="Actual")
plt.plot(pred[:100], label="Predicted")
plt.legend()
plt.show()