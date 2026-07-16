"""
Module 02 -- Gradient descent from absolute scratch.
Three stages. Run the file; each stage prints its own story.

Only dependency: numpy.
"""
import numpy as np

rng = np.random.default_rng(42)


# ----------------------------------------------------------------------
# STAGE A: one parameter, gradient derived by hand
# ----------------------------------------------------------------------
# Task: we secretly generate data with y = 3.0 * x (plus noise) and see
# if gradient descent can discover the 3.0 without being told.
#
# Model:       pred = w * x
# Loss (MSE):  L = mean((w*x - y)^2)
#
# The derivative dL/dw, by the chain rule (high-school calculus):
#   d/dw (w*x - y)^2  =  2 * (w*x - y) * x
# averaged over the dataset. That's the whole "gradient".

def stage_a():
    print("=" * 60)
    print("STAGE A: fit y = w*x, gradient computed by hand-derived formula")
    print("=" * 60)

    x = rng.uniform(-2, 2, size=100)
    y = 3.0 * x + rng.normal(0, 0.1, size=100)   # ground truth w = 3.0

    w = 0.0            # start knowing nothing
    lr = 0.1           # learning rate = step size

    for step in range(30):
        pred = w * x
        loss = np.mean((pred - y) ** 2)
        grad = np.mean(2 * (pred - y) * x)   # dL/dw -- the hand-derived formula
        w -= lr * grad                       # THE update. This line is all of deep learning.
        if step % 5 == 0:
            print(f"  step {step:3d}  w = {w:7.4f}  loss = {loss:8.5f}")

    print(f"  final w = {w:.4f}   (true answer: 3.0)\n")


# ----------------------------------------------------------------------
# STAGE B: numerical gradients -- the brute-force oracle
# ----------------------------------------------------------------------
# Deriving formulas by hand doesn't scale. But the *definition* of a
# derivative gives us a universal (slow) gradient machine:
#
#   dL/dw  ~=  (L(w + h) - L(w)) / h        for tiny h
#
# i.e. literally: nudge the parameter, re-measure the loss, divide.
# This works for ANY loss function with zero calculus. It's O(n_params)
# loss evaluations per step -- uselessly slow for big models, but the
# perfect *test oracle* for the fast method (backprop) in module 03.

def stage_b():
    print("=" * 60)
    print("STAGE B: same fit, gradient measured numerically (no calculus)")
    print("=" * 60)

    x = rng.uniform(-2, 2, size=100)
    y = 3.0 * x + rng.normal(0, 0.1, size=100)

    def loss_fn(w):
        return np.mean((w * x - y) ** 2)

    w, lr, h = 0.0, 0.1, 1e-5

    for step in range(30):
        grad = (loss_fn(w + h) - loss_fn(w)) / h   # nudge & re-measure
        w -= lr * grad
        if step % 5 == 0:
            print(f"  step {step:3d}  w = {w:7.4f}  loss = {loss_fn(w):8.5f}")

    print(f"  final w = {w:.4f}   -- same answer, zero calculus\n")


# ----------------------------------------------------------------------
# STAGE C: the real thing -- two parameters, mini-batches, loss curve
# ----------------------------------------------------------------------
# Model: y = w*x + b.  We now:
#   * estimate gradients from random BATCHES (stochastic gradient descent)
#   * track the loss over time and print it as an ASCII chart
# This structure -- forward, loss, gradients, update, repeat over batches --
# is *identical* in shape to how GPT-4 was trained. Only the model between
# "forward" and "loss" gets fancier.

def stage_c():
    print("=" * 60)
    print("STAGE C: y = w*x + b, mini-batch SGD, loss curve")
    print("=" * 60)

    N = 1000
    x = rng.uniform(-2, 2, size=N)
    y = 3.0 * x - 1.5 + rng.normal(0, 0.3, size=N)   # truth: w=3.0, b=-1.5

    w, b = 0.0, 0.0
    lr, batch_size = 0.05, 32
    history = []

    for step in range(400):
        idx = rng.integers(0, N, size=batch_size)     # random mini-batch
        xb, yb = x[idx], y[idx]

        pred = w * xb + b                             # 1. forward
        err = pred - yb
        loss = np.mean(err ** 2)                      # 2. loss

        grad_w = np.mean(2 * err * xb)                # 3. gradients
        grad_b = np.mean(2 * err)                     #    (dL/db: chain rule, x-term is 1)

        w -= lr * grad_w                              # 4. update
        b -= lr * grad_b
        history.append(loss)

    # crude ASCII loss curve: one row per 40 steps
    print("  loss over training (each row = 40 steps):")
    for i in range(0, 400, 40):
        chunk = np.mean(history[i:i + 40])
        bar = "#" * max(1, int(chunk * 12))
        print(f"  steps {i:3d}-{i+39:3d}  {chunk:7.4f}  {bar}")

    print(f"\n  learned  w = {w:.3f}, b = {b:.3f}   (truth: 3.0, -1.5)")
    print("  You just trained a model with the exact loop that trains GPTs.\n")


if __name__ == "__main__":
    stage_a()
    stage_b()
    stage_c()
