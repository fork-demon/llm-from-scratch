"""
Module 03 -- A complete neural network + backpropagation in NumPy.

What happens when you run this file:
  1. Generates a 3-class spiral dataset (impossible for linear models).
  2. Trains a LINEAR classifier  -> watch it plateau (~50-54%).
  3. Trains a 2-hidden-layer MLP -> watch it succeed (~98-99%).
  4. Gradient-checks the backprop against numerical gradients.
  5. (Optional) Plots decision boundaries if matplotlib is installed.

Expected Output:
  ============================================================
  1) LINEAR model on the spiral (this should struggle)
  ============================================================
    final loss 0.7331, accuracy 0.540  <- the linear ceiling

  ============================================================
  2) MLP (2 -> 64 -> 64 -> 3) on the same data
  ============================================================
    step    0  loss 1.1000  acc 0.470
    step  400  loss 0.0826  acc 0.970
    step  800  loss 0.0399  acc 0.987
    step 1200  loss 0.0321  acc 0.987
    step 1600  loss 0.0285  acc 0.987
    final accuracy 0.987  <- nonlinearity earns its keep

  ============================================================
  3) Gradient check (backprop vs numerical oracle)
  ============================================================
    worst diff ~1e-12 -> PASS

Every gradient here is computed by the three rules:
  linear:   d_W = X.T @ d_out ;  d_X = d_out @ W.T ;  d_b = sum(d_out)
  relu:     d_x = d_out * (x > 0)
  softmax+cross-entropy at the logits:  d_logits = probs - one_hot
"""
import numpy as np

rng = np.random.default_rng(0)


# ----------------------------------------------------------------------
# Data: three interleaved spiral arms, one per class.
# ----------------------------------------------------------------------
def make_spiral(points_per_class=100, num_classes=3):
    N, K = points_per_class, num_classes
    X = np.zeros((N * K, 2))
    y = np.zeros(N * K, dtype=int)
    for k in range(K):
        ix = range(N * k, N * (k + 1))
        r = np.linspace(0.0, 1.0, N)                                  # radius
        t = np.linspace(k * 4, (k + 1) * 4, N) + rng.normal(0, 0.2, N)  # angle
        X[ix] = np.column_stack([r * np.sin(t), r * np.cos(t)])
        y[ix] = k
    return X, y


# ----------------------------------------------------------------------
# The shared pieces
# ----------------------------------------------------------------------
def softmax(logits):
    # subtract the row max first: exp() of big numbers overflows, and
    # softmax is unchanged by shifting -- the classic numerical-stability trick.
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(probs, y):
    # loss = mean over batch of  -log(prob assigned to the CORRECT class)
    # "average surprise". 1e-12 guards log(0).
    return -np.log(probs[np.arange(len(y)), y] + 1e-12).mean()


# ----------------------------------------------------------------------
# Model 1: LINEAR classifier (to watch it fail)
# ----------------------------------------------------------------------
def train_linear(X, y, steps=300, lr=1.0):
    n_in, n_out = X.shape[1], y.max() + 1
    W = 0.01 * rng.normal(size=(n_in, n_out))
    b = np.zeros(n_out)

    for step in range(steps):
        logits = X @ W + b                       # forward
        probs = softmax(logits)
        loss = cross_entropy(probs, y)

        d_logits = probs.copy()                  # backward: "gradient is the miss"
        d_logits[np.arange(len(y)), y] -= 1
        d_logits /= len(y)

        W -= lr * (X.T @ d_logits)               # update
        b -= lr * d_logits.sum(axis=0)

    acc = (softmax(X @ W + b).argmax(axis=1) == y).mean()
    return loss, acc, W, b


# ----------------------------------------------------------------------
# Model 2: the MLP.  Architecture: 2 -> 64 -> 64 -> 3
# ----------------------------------------------------------------------
class MLP:
    def __init__(self, sizes=(2, 64, 64, 3)):
        # Small random init. Why not zeros? All-zero weights make every
        # neuron identical -> identical gradients -> they can never
        # differentiate ("symmetry breaking" needs randomness).
        self.sizes = sizes
        self.W = [0.1 * rng.normal(size=(a, b)) for a, b in zip(sizes, sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]

    # ---- forward: save intermediate values; backprop will need them ----
    def forward(self, X):
        B, D = X.shape
        assert D == self.sizes[0], f"Expected input dim {self.sizes[0]}, got {D}"
        self.cache = [X]                   # activations layer by layer
        h = X
        for i in range(len(self.W) - 1):
            h = np.maximum(0, h @ self.W[i] + self.b[i])   # linear + ReLU hinge
            self.cache.append(h)
        logits = h @ self.W[-1] + self.b[-1]               # last layer: no ReLU
        return logits

    def loss(self, X, y):
        return cross_entropy(softmax(self.forward(X)), y)

    # ---- backward: walk the assembly line in reverse ----
    def backward(self, logits, y):
        n = len(y)
        d_W = [None] * len(self.W)
        d_b = [None] * len(self.b)

        probs = softmax(logits)
        d = probs.copy()                            # d = blame flowing backward
        d[np.arange(n), y] -= 1                     # d_logits = probs - one_hot
        d /= n

        for i in reversed(range(len(self.W))):
            h_in = self.cache[i]                    # what this layer consumed
            d_W[i] = h_in.T @ d                     # rule 1: blame -> weights
            d_b[i] = d.sum(axis=0)
            if i > 0:
                d = d @ self.W[i].T                 # rule 1: blame -> layer input
                d = d * (self.cache[i] > 0)         # rule 2: ReLU gate
        return d_W, d_b

    def step(self, d_W, d_b, lr):
        for i in range(len(self.W)):
            self.W[i] -= lr * d_W[i]
            self.b[i] -= lr * d_b[i]

    def accuracy(self, X, y):
        return (self.forward(X).argmax(axis=1) == y).mean()


def train_mlp(X, y, steps=2000, lr=0.5):
    net = MLP()
    for step in range(steps):
        logits = net.forward(X)
        loss = cross_entropy(softmax(logits), y)
        d_W, d_b = net.backward(logits, y)
        net.step(d_W, d_b, lr)
        if step % 400 == 0:
            print(f"  step {step:4d}  loss {loss:.4f}  acc {net.accuracy(X, y):.3f}")
    return net


# ----------------------------------------------------------------------
# Decision Boundary Visualizer (Terminal fallback + Matplotlib)
# ----------------------------------------------------------------------
def plot_boundaries(X, y, linear_predict_fn, mlp_predict_fn, save_path=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  [Notice] Install matplotlib (`pip install matplotlib`) to view visual boundary plots.")
        return

    plt.figure(figsize=(10, 4.5))
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # 1. Linear model
    plt.subplot(1, 2, 1)
    Z_lin = linear_predict_fn(grid).reshape(xx.shape)
    plt.contourf(xx, yy, Z_lin, alpha=0.3, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=15, cmap=plt.cm.Spectral, edgecolors="k", linewidth=0.5)
    plt.title("Linear Model (Stuck at ~50%)")
    plt.xlabel("x0")
    plt.ylabel("x1")

    # 2. MLP model
    plt.subplot(1, 2, 2)
    Z_mlp = mlp_predict_fn(grid).reshape(xx.shape)
    plt.contourf(xx, yy, Z_mlp, alpha=0.3, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=15, cmap=plt.cm.Spectral, edgecolors="k", linewidth=0.5)
    plt.title("MLP Model (Hinges bend space ~98%)")
    plt.xlabel("x0")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"  Decision boundary plot saved to: {save_path}")
    else:
        plt.show()


# ----------------------------------------------------------------------
# Gradient check: backprop vs. the brute-force oracle from module 02.
# If these agree to ~1e-6, the backprop is provably correct.
# ----------------------------------------------------------------------
def gradient_check(net, X, y, n_checks=8, h=1e-5):
    logits = net.forward(X)
    d_W, _ = net.backward(logits, y)

    print("  param        backprop      numerical     |diff|")
    worst = 0.0
    for _ in range(n_checks):
        li = rng.integers(0, len(net.W))            # random layer
        r = rng.integers(0, net.W[li].shape[0])     # random weight in it
        c = rng.integers(0, net.W[li].shape[1])

        orig = net.W[li][r, c]
        net.W[li][r, c] = orig + h                  # nudge up
        lp = net.loss(X, y)
        net.W[li][r, c] = orig - h                  # nudge down
        lm = net.loss(X, y)
        net.W[li][r, c] = orig                      # restore

        numerical = (lp - lm) / (2 * h)             # centered difference
        analytic = d_W[li][r, c]
        diff = abs(numerical - analytic)
        worst = max(worst, diff)
        print(f"  W{li}[{r:2d},{c:2d}]   {analytic:12.8f}  {numerical:12.8f}  {diff:.2e}")
    status = "PASS" if worst < 1e-5 else "FAIL"
    print(f"  worst diff {worst:.2e} -> {status}")


if __name__ == "__main__":
    X, y = make_spiral()

    print("=" * 60)
    print("1) LINEAR model on the spiral (this should struggle)")
    print("=" * 60)
    loss, acc, W_lin, b_lin = train_linear(X, y)
    print(f"  final loss {loss:.4f}, accuracy {acc:.3f}  <- the linear ceiling\n")

    print("=" * 60)
    print("2) MLP (2 -> 64 -> 64 -> 3) on the same data")
    print("=" * 60)
    net = train_mlp(X, y)
    print(f"  final accuracy {net.accuracy(X, y):.3f}  <- nonlinearity earns its keep\n")

    print("=" * 60)
    print("3) Gradient check (backprop vs numerical oracle)")
    print("=" * 60)
    gradient_check(net, X[:50], y[:50])

    # Save visualization to diagrams folder
    import os
    diag_dir = os.path.join(os.path.dirname(__file__), "..", "diagrams")
    if os.path.exists(diag_dir):
        plot_path = os.path.join(diag_dir, "03-spiral-boundary.png")
        plot_boundaries(X, y, 
                        lambda g: (g @ W_lin + b_lin).argmax(axis=1),
                        lambda g: net.forward(g).argmax(axis=1),
                        save_path=plot_path)
