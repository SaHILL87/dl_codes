import numpy as np


def activate(net):
    return 1 if net >= 0 else -1


def read_data(n, m):
    x_data = np.zeros((n, m), dtype=float)
    t_data = np.zeros(n, dtype=int)
    print("Enter: f1 f2 ... fm label(0/1 or -1/1)")
    for i in range(n):
        row = np.array(list(map(float, input("Row " + str(i + 1) + ": ").split())))
        x_data[i] = row[:m]
        label = int(row[m])
        t_data[i] = -1 if label == 0 else 1
    return x_data, t_data


n = int(input("No. of samples: "))
m = int(input("No. of features: "))
x_data, t_data = read_data(n, m)
alpha = float(input("Learning rate: "))
max_epochs = int(input("Max epochs: "))

w = np.zeros(m, dtype=float)
b = 0.0

for epoch in range(max_epochs):
    errors = 0
    for i in range(n):
        y = activate(np.dot(w, x_data[i]) + b)
        if y != t_data[i]:
            w += alpha * t_data[i] * x_data[i]
            b += alpha * t_data[i]
            errors += 1

    print("Epoch", epoch + 1, "errors=", errors, "w=", w, "b=", b)
    if errors == 0:
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
    print("Predicted class:", 1 if y == 1 else 0)
