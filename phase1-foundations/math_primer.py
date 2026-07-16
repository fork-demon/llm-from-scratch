"""
Module 00 -- Verify every hand computation in the math primer.

Run it: every section prints its checks and asserts they're right.
Then BREAK it: change a number in a "by hand" value and watch the
assertion catch you. Math you can unit-test is math you can trust.
"""
import numpy as np


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ----------------------------------------------------------------------
section("1. VECTORS: lists of numbers with arithmetic")
v = np.array([2, 1, 3])
w = np.array([1, 0, 2])
print(f"  v + w    = {v + w}          (element-wise add)")
print(f"  2 * v    = {2 * v}          (scaling)")
assert (v + w == np.array([3, 1, 5])).all()

length = np.sqrt((np.array([3, 4]) ** 2).sum())      # Pythagoras, unchanged
print(f"  length of [3,4] = {length}   (school Pythagoras, any dimension)")
assert length == 5.0


# ----------------------------------------------------------------------
section("2. DOT PRODUCT: multiply pairwise, add up")
a = np.array([2, 1, 3])
b = np.array([4, 0, 2])
by_hand = 2 * 4 + 1 * 0 + 3 * 2                       # = 14, as on the page
print(f"  by hand : (2x4)+(1x0)+(3x2) = {by_hand}")
print(f"  numpy   : a @ b = {a @ b}")
assert a @ b == by_hand == 14

# dot product as similarity meter (cat/dog/truck from the page)
cat = np.array([0.9, 0.8, -0.5])
dog = np.array([0.8, 0.9, -0.4])
truck = np.array([-0.6, -0.8, 0.9])
print(f"  cat.dog   = {cat @ dog:+.2f}   (similar -> large positive)")
print(f"  cat.truck = {cat @ truck:+.2f}   (opposite -> negative)")
assert cat @ dog > 1.5 and cat @ truck < -1.5

def cosine(x, y):
    return (x @ y) / (np.sqrt((x**2).sum()) * np.sqrt((y**2).sum()))
print(f"  cosine(cat, dog)   = {cosine(cat, dog):+.3f}   (range -1..+1)")
print(f"  cosine(cat, truck) = {cosine(cat, truck):+.3f}")


# ----------------------------------------------------------------------
section("4. MATRIX x VECTOR: a batch of dot products (one per row)")
W = np.array([[1, 0, 2],
              [0, 3, 1]])
x = np.array([2, 1, 1])
row1 = 1 * 2 + 0 * 1 + 2 * 1        # dot of row 1 with x
row2 = 0 * 2 + 3 * 1 + 1 * 1        # dot of row 2 with x
print(f"  row 1 . x = {row1},  row 2 . x = {row2}   (by hand)")
print(f"  W @ x     = {W @ x}                (numpy agrees)")
print(f"  shapes: {W.shape} @ {x.shape} -> {(W @ x).shape}   inner 3s match, outer 2 out")
assert (W @ x == np.array([4, 4])).all()


# ----------------------------------------------------------------------
section("5. MATRIX x MATRIX: every row of A dotted with every column of B")
A = np.array([[1, 2],
              [3, 0]])
B = np.array([[1, 0, 1],
              [2, 1, 0]])
by_hand = np.array([[5, 2, 1],
                    [3, 0, 3]])     # worked cell-by-cell on the page
print(f"  A @ B =\n{A @ B}")
assert (A @ B == by_hand).all()
print(f"  shapes: {A.shape} @ {B.shape} -> {(A @ B).shape}")

# order matters:
C = np.array([[0, 1], [1, 0]])      # a swap
D = np.array([[2, 0], [0, 1]])      # a stretch
print(f"  C@D =\n{C @ D}\n  D@C =\n{D @ C}   <- NOT equal: order matters")
assert not (C @ D == D @ C).all()

# one matmul processes every token at once (why GPUs won):
tokens = np.array([[1.0, 2.0],      # token 0's vector
                   [3.0, 4.0],      # token 1's vector
                   [5.0, 6.0]])     # token 2's vector
transform = np.array([[1.0, 0.0, 1.0],
                      [0.0, 1.0, 1.0]])
out = tokens @ transform            # (3,2) @ (2,3) -> (3,3): all 3 tokens at once
print(f"  3 tokens transformed in ONE multiply -> shape {out.shape}")


# ----------------------------------------------------------------------
section("6. TRANSPOSE: flip rows/columns; mostly shape plumbing")
A = np.array([[1, 2, 3],
              [4, 5, 6]])
print(f"  A ({A.shape}) ->  A.T ({A.T.shape}):\n{A.T}")
assert A.T.shape == (3, 2)

# the Q @ K.T pattern from attention: all-pairs dot products
X = np.array([[1.0, 0.0],
              [0.0, 1.0],
              [1.0, 1.0]])           # 3 token vectors, dim 2
pairwise = X @ X.T                    # (3,2) @ (2,3) -> (3,3)
print(f"  X @ X.T = all-pairs similarities:\n{pairwise}")
assert pairwise[0, 2] == X[0] @ X[2]  # cell (i,j) is exactly token_i . token_j


# ----------------------------------------------------------------------
section("7. DERIVATIVE: the nudge experiment")
def nudge_derivative(f, x, h=1e-6):
    return (f(x + h) - f(x)) / h      # nudge, re-measure, divide

d = nudge_derivative(lambda x: x ** 2, 3.0)
print(f"  f(x)=x^2 at x=3: measured {d:.4f}, formula 2x says 6")
assert abs(d - 6) < 1e-3

# verify the whole formula table from the page:
table = [
    ("c*x (c=5) ", lambda x: 5 * x,  lambda x: 5.0),
    ("x^2       ", lambda x: x ** 2, lambda x: 2 * x),
    ("e^x       ", np.exp,           np.exp),
    ("ln(x)     ", np.log,           lambda x: 1 / x),
]
x0 = 2.0
for name, f, formula in table:
    measured, expected = nudge_derivative(f, x0), formula(x0)
    print(f"  {name} at x=2: measured {measured:8.4f}, formula {expected:8.4f}")
    assert abs(measured - expected) < 1e-3


# ----------------------------------------------------------------------
section("8. CHAIN RULE: amplification factors multiply")
# pipeline: x -> square -> times 5, at x = 2
stage1_amp = 2 * 2.0            # d(x^2)/dx = 2x = 4 at x=2
stage2_amp = 5.0                # d(5u)/du = 5
end_to_end = stage1_amp * stage2_amp
measured = nudge_derivative(lambda x: 5 * x ** 2, 2.0)
print(f"  stage amps: {stage1_amp} x {stage2_amp} = {end_to_end}")
print(f"  nudge experiment on the whole pipeline: {measured:.4f}")
assert abs(measured - end_to_end) < 1e-3
print("  -> backprop (module 03) is this, run station by station.")


print("\nALL CHECKS PASS. Now change a 'by hand' number above and rerun --")
print("the assertion will catch it. Math you can unit-test.")
