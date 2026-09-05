import unittest
import numpy as np

# Tests can import directly by adding parent paths
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'phase1-foundations')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'phase3-transformers')))

from mlp_numpy import MLP, softmax, cross_entropy, make_spiral
from attention_numpy import attention

class TestFoundations(unittest.TestCase):
    def test_mlp_gradient_check(self):
        rng = np.random.default_rng(42)
        X, y = make_spiral(points_per_class=10, num_classes=3)
        net = MLP(sizes=(2, 16, 3))
        logits = net.forward(X)
        d_W, d_b = net.backward(logits, y)
        
        # Numerical gradient check on one layer weight
        W = net.W[0]
        i, j = 0, 0
        orig = W[i, j]
        h = 1e-5
        
        W[i, j] = orig + h
        l1 = net.loss(X, y)
        W[i, j] = orig - h
        l2 = net.loss(X, y)
        W[i, j] = orig
        
        num_grad = (l1 - l2) / (2 * h)
        analytical_grad = d_W[0][i, j]
        
        diff = abs(num_grad - analytical_grad)
        self.assertLess(diff, 1e-5, f"Gradient check failed: diff={diff}")

    def test_attention_shapes_and_causal_mask(self):
        T, D = 4, 8
        x = np.random.randn(T, D)
        Wq = np.random.randn(D, D) / np.sqrt(D)
        Wk = np.random.randn(D, D) / np.sqrt(D)
        Wv = np.random.randn(D, D) / np.sqrt(D)

        out, weights = attention(x, Wq, Wk, Wv, causal=True)
        self.assertEqual(out.shape, (T, D))
        self.assertEqual(weights.shape, (T, T))

        # Check rows sum to 1
        np.testing.assert_allclose(weights.sum(axis=-1), np.ones(T), rtol=1e-5)

        # Check causal mask: upper triangle (strict) must be 0
        for i in range(T):
            for j in range(i + 1, T):
                self.assertAlmostEqual(weights[i, j], 0.0, places=6)

if __name__ == '__main__':
    unittest.main()
