import numpy as np


def activate(net):
    net = np.clip(net, -50, 50)
    return 1.0 / (1.0 + np.exp(-net))


def read_data(n, m):
    x_data = np.zeros((n, m), dtype=float)
    t_data = np.zeros(n, dtype=float)
    print("Enter: f1 f2 ... fm label(0/1)")
    for i in range(n):
        row = np.array(list(map(float, input("Row " + str(i + 1) + ": ").split())))
        x_data[i] = row[:m]
        t_data[i] = row[m]
    return x_data, t_data


n = int(input("No. of samples: "))
m = int(input("No. of features: "))
x_data, t_data = read_data(n, m)
alpha = float(input("Learning rate: "))
max_epochs = int(input("Max epochs: "))

w = np.zeros(m, dtype=float)
b = 0.0

for epoch in range(max_epochs):
    loss = 0.0
    for i in range(n):
        y = activate(np.dot(w, x_data[i]) + b)
        error = t_data[i] - y
        grad = error * y * (1 - y)
        w += alpha * grad * x_data[i]
        b += alpha * grad
        loss += error * error

    mse = loss / n
    print("Epoch", epoch + 1, "mse=", round(mse, 6), "w=", w, "b=", b)
    if mse < 0.001:
        break

print("Final w=", w)
print("Final b=", b)

while True:
    s = input("Test sample (or q): ")
    if s.lower() == "q":
        break

    x = np.array(list(map(float, s.split())))
    if len(x) != m:
        print("Enter exactly", m, "values")
        continue

    y = activate(np.dot(w, x) + b)
    print("Output=", round(float(y), 4), "Predicted class:", 1 if y >= 0.5 else 0)
