import numpy as np
from sklearn.datasets import load_breast_cancer

np.random.seed(0)

# ── Primitives (never touch these) ────────────────────────────────────────────

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

def bce_loss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def get_metrics(y_true, p):
    pred = (p >= 0.5).astype(int)
    tp = np.sum((pred == 1) & (y_true == 1))
    tn = np.sum((pred == 0) & (y_true == 0))
    fp = np.sum((pred == 1) & (y_true == 0))
    fn = np.sum((pred == 0) & (y_true == 1))
    acc  = (tp + tn) / len(y_true)
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return acc, prec, rec, f1

def split_and_standardize(X, y, test_ratio=0.2):
    idx = np.random.permutation(len(X))
    cut = int(len(X) * (1 - test_ratio))
    X_tr, X_te = X[idx[:cut]], X[idx[cut:]]
    y_tr, y_te = y[idx[:cut]], y[idx[cut:]]
    mu, sigma = X_tr.mean(0), X_tr.std(0) + 1e-9
    return (X_tr - mu) / sigma, (X_te - mu) / sigma, y_tr, y_te

def gradients(Xb, yb, w, b):
    err = sigmoid(Xb @ w + b) - yb
    return (Xb.T @ err) / len(Xb), err.mean()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Write your batch strategy here
# ─────────────────────────────────────────────────────────────────────────────

def get_batches(X, y, batch_size=32):
    idx = np.random.permutation(len(X))

    # --- MINI-BATCH (default shown) ---
    for s in range(0, len(idx), batch_size):
        yield X[idx[s:s+batch_size]], y[idx[s:s+batch_size]]

    # --- VANILLA: replace everything above with ---
    # yield X[idx], y[idx]

    # --- SGD: replace everything above with ---
    # for i in idx:
    #     yield X[i:i+1], y[i:i+1]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Write your optimizer update here
# ─────────────────────────────────────────────────────────────────────────────

def update(s, gw, gb, lr):

    # --- ADAM (default shown) ---
    b1, b2, eps = 0.9, 0.999, 1e-8
    s["t"] += 1
    s["mw"] = b1 * s["mw"] + (1 - b1) * gw
    s["mb"] = b1 * s["mb"] + (1 - b1) * gb
    s["sw"] = b2 * s["sw"] + (1 - b2) * gw ** 2
    s["sb"] = b2 * s["sb"] + (1 - b2) * gb ** 2
    mwh = s["mw"] / (1 - b1 ** s["t"]);  mbh = s["mb"] / (1 - b1 ** s["t"])
    swh = s["sw"] / (1 - b2 ** s["t"]);  sbh = s["sb"] / (1 - b2 ** s["t"])
    s["w"] -= lr * mwh / (np.sqrt(swh) + eps)
    s["b"] -= lr * mbh / (np.sqrt(sbh) + eps)

    # --- GD ---
    # s["w"] -= lr * gw
    # s["b"] -= lr * gb

    # --- MOMENTUM ---
    # mu = 0.9
    # s["vw"] = mu * s["vw"] - lr * gw
    # s["vb"] = mu * s["vb"] - lr * gb
    # s["w"] += s["vw"];  s["b"] += s["vb"]

    # --- RMSPROP ---
    # rho, eps = 0.9, 1e-8
    # s["sw"] = rho * s["sw"] + (1 - rho) * gw ** 2
    # s["sb"] = rho * s["sb"] + (1 - rho) * gb ** 2
    # s["w"] -= lr * gw / (np.sqrt(s["sw"]) + eps)
    # s["b"] -= lr * gb / (np.sqrt(s["sb"]) + eps)

    # --- ADAGRAD ---
    # eps = 1e-8
    # s["sw"] += gw ** 2;  s["sb"] += gb ** 2
    # s["w"] -= lr * gw / (np.sqrt(s["sw"]) + eps)
    # s["b"] -= lr * gb / (np.sqrt(s["sb"]) + eps)

    # --- NAG (also change gradient computation in train() below) ---
    # mu = 0.9
    # s["vw"] = mu * s["vw"] - lr * gw
    # s["vb"] = mu * s["vb"] - lr * gb
    # s["w"] += s["vw"];  s["b"] += s["vb"]

# ── Training loop (never touch this) ─────────────────────────────────────────

def train(X, y, lr, epochs=80, batch_size=32):
    d = X.shape[1]
    s = dict(w=np.zeros(d), b=0.0,
             vw=np.zeros(d), vb=0.0,
             mw=np.zeros(d), mb=0.0,
             sw=np.zeros(d), sb=0.0, t=0)
    history = []

    for _ in range(epochs):
        for Xb, yb in get_batches(X, y, batch_size):

            # For NAG: change this line to:
            # gw, gb = gradients(Xb, yb, s["w"] + 0.9*s["vw"], s["b"] + 0.9*s["vb"])
            gw, gb = gradients(Xb, yb, s["w"], s["b"])

            update(s, gw, gb, lr)

        history.append(float(bce_loss(y, sigmoid(X @ s["w"] + s["b"]))))

    return s["w"], s["b"], history

# ── Main (never touch this) ───────────────────────────────────────────────────

if __name__ == "__main__":
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = split_and_standardize(X.astype(float), y.astype(int))

    w, b, history = train(X_tr, y_tr, lr=0.01)

    p_te = sigmoid(X_te @ w + b)
    acc, prec, rec, f1 = get_metrics(y_te, p_te)
    print(f"Final train loss : {history[-1]:.6f}")
    print(f"Acc / Prec / Rec / F1 : {acc:.4f} / {prec:.4f} / {rec:.4f} / {f1:.4f}")