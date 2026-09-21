"""Lesson 2.1 (gradient descent) and Part 3 (neural networks, backprop)."""
import numpy as np

import mlp_numpy as m


def test_softmax_rows_sum_to_one_and_do_not_overflow():
    p = m.softmax(np.array([[1000.0, 999.0], [0.0, 0.0]]))
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.allclose(p[1], [0.5, 0.5])


def test_cross_entropy_is_low_when_confident_and_right():
    probs = np.array([[0.9, 0.1]])
    assert m.cross_entropy(probs, np.array([0])) < m.cross_entropy(probs, np.array([1]))


def test_backprop_matches_numerical_gradient():
    """The gradient check from the backprop lesson, as an assertion."""
    X, y = m.make_spiral(points_per_class=10)
    net = m.MLP(sizes=(2, 8, 3))
    d_W, _ = net.backward(net.forward(X), y)
    h = 1e-5
    for li, r, c in [(0, 0, 0), (0, 1, 3), (1, 2, 1)]:
        orig = net.W[li][r, c]
        net.W[li][r, c] = orig + h
        lp = net.loss(X, y)
        net.W[li][r, c] = orig - h
        lm = net.loss(X, y)
        net.W[li][r, c] = orig
        assert abs((lp - lm) / (2 * h) - d_W[li][r, c]) < 1e-6


def test_mlp_beats_linear_model_on_spiral():
    X, y = m.make_spiral()
    _, linear_acc = m.train_linear(X, y)
    net = m.MLP()
    for _ in range(600):
        logits = net.forward(X)
        net.step(*net.backward(logits, y), lr=0.5)
    assert net.accuracy(X, y) > linear_acc + 0.15
