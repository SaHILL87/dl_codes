import numpy as np
from sklearn.datasets import load_wine

np.random.seed(0)

# ── Primitives ────────────────────────────────────────────────────────────────

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

def bce_loss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def get_metrics(y_true, p):
    y_pred = (p >= 0.5).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    acc  = (tp + tn) / len(y_true)
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return acc, prec, rec, f1

# ── Data helpers ──────────────────────────────────────────────────────────────

def split_data(X, y, test_ratio=0.2):
    idx = np.random.permutation(len(X))
    cut = int(len(X) * (1 - test_ratio))
    return X[idx[:cut]], X[idx[cut:]], y[idx[:cut]], y[idx[cut:]]

def standardize(X_tr, X_te):
    mu, sigma = X_tr.mean(0), X_tr.std(0) + 1e-9
    return (X_tr - mu) / sigma, (X_te - mu) / sigma

def get_batches(X, y, strategy, batch_size):
    idx = np.random.permutation(len(X))
    if strategy == "vanilla":
        yield X[idx], y[idx]
    elif strategy == "sgd":
        for i in idx:
            yield X[i:i+1], y[i:i+1]
    elif strategy == "mini_batch":
        for s in range(0, len(idx), batch_size):
            b = idx[s:s+batch_size]
            yield X[b], y[b]

# ── Gradient ──────────────────────────────────────────────────────────────────

def gradients(Xb, yb, w, b):
    err = sigmoid(Xb @ w + b) - yb
    return (Xb.T @ err) / len(Xb), err.mean()

# ── Optimizer update rules ────────────────────────────────────────────────────

def update_gd(s, gw, gb, lr, **_):
    s["w"] -= lr * gw
    s["b"] -= lr * gb

def update_momentum(s, gw, gb, lr, mu=0.9, **_):
    s["vw"] = mu*s["vw"] - lr*gw
    s["vb"] = mu*s["vb"] - lr*gb
    s["w"]  += s["vw"]             
    s["b"]  += s["vb"]

update_nag = update_momentum          # NAG uses the same update; look-ahead is handled in train()

def update_rmsprop(s, gw, gb, lr, rho=0.9, eps=1e-8, **_):
    s["sw"] = rho*s["sw"] + (1-rho)*gw**2
    s["sb"] = rho*s["sb"] + (1-rho)*gb**2
    s["w"] -= lr * gw / (np.sqrt(s["sw"]) + eps)
    s["b"] -= lr * gb / (np.sqrt(s["sb"]) + eps)

def update_adagrad(s, gw, gb, lr, eps=1e-8, **_):
    s["sw"] += gw**2;  s["sb"] += gb**2
    s["w"]  -= lr * gw / (np.sqrt(s["sw"]) + eps)
    s["b"]  -= lr * gb / (np.sqrt(s["sb"]) + eps)

def update_adam(s, gw, gb, lr, b1=0.9, b2=0.999, eps=1e-8, **_):
    s["t"] += 1
    s["mw"] = b1*s["mw"] + (1-b1)*gw;   s["mb"] = b1*s["mb"] + (1-b1)*gb
    s["sw"] = b2*s["sw"] + (1-b2)*gw**2; s["sb"] = b2*s["sb"] + (1-b2)*gb**2
    mwh = s["mw"]/(1-b1**s["t"]);  mbh = s["mb"]/(1-b1**s["t"])
    swh = s["sw"]/(1-b2**s["t"]);  sbh = s["sb"]/(1-b2**s["t"])
    s["w"] -= lr * mwh / (np.sqrt(swh) + eps)
    s["b"] -= lr * mbh / (np.sqrt(sbh) + eps)

OPTIMIZERS = {
    "gd": update_gd, "momentum": update_momentum, "nag": update_nag,
    "rmsprop": update_rmsprop, "adagrad": update_adagrad, "adam": update_adam,
}

# ── Training loop ─────────────────────────────────────────────────────────────

def train(X, y, optimizer, strategy, lr, epochs=80, batch_size=32):
    d = X.shape[1]
    s = dict(w=np.zeros(d), b=0.0,
             vw=np.zeros(d), vb=0.0,
             mw=np.zeros(d), mb=0.0,
             sw=np.zeros(d), sb=0.0, t=0)
    update = OPTIMIZERS[optimizer]
    history = []

    for _ in range(epochs):
        for Xb, yb in get_batches(X, y, strategy, batch_size):
            if optimizer == "nag":                       # look-ahead step
                gw, gb = gradients(Xb, yb, s["w"] + 0.9*s["vw"],
                                           s["b"] + 0.9*s["vb"])
            else:
                gw, gb = gradients(Xb, yb, s["w"], s["b"])
            update(s, gw, gb, lr)

        history.append(float(bce_loss(y, sigmoid(X @ s["w"] + s["b"]))))

    p = sigmoid(X @ s["w"] + s["b"])
    return s["w"], s["b"], float(bce_loss(y, p)), history

# ── Main ──────────────────────────────────────────────────────────────────────

LR = {"gd": 0.05, "momentum": 0.03, "nag": 0.03,
      "rmsprop": 0.005, "adagrad": 0.2, "adam": 0.01}

VALID_COMBOS = [(opt, strat)
                for opt in LR
                for strat in ("vanilla", "mini_batch", "sgd")]

if __name__ == "__main__":
    # using load_wine mapped to binary (class 0 vs others) for a different dataset
    X, y = load_wine(return_X_y=True)
    y = (y == 0).astype(int)

    X_tr, X_te, y_tr, y_te = split_data(X.astype(float), y.astype(int))
    X_tr, X_te = standardize(X_tr, X_te)

    # ← only line you change for the exam
    selected = ("adam", "sgd") # (optimizer, strategy)

    assert selected in VALID_COMBOS, f"Invalid combo. Choose from:\n{VALID_COMBOS}"

    opt, strategy = selected
    w, b, train_loss, history = train(X_tr, y_tr, opt, strategy, LR[opt])

    acc, prec, rec, f1 = get_metrics(y_te, sigmoid(X_te @ w + b))
    print(f"Combo : {opt}+{strategy}")
    print(f"Train loss : {train_loss:.6f}")
    print(f"Acc/Prec/Rec/F1: {acc:.4f} / {prec:.4f} / {rec:.4f} / {f1:.4f}")
