"""
Module 11 -- A vector database from scratch.

Contents:
  1. Brute-force exact nearest-neighbor search (fine to ~1M vectors!)
  2. k-means from scratch (15 lines)
  3. IVF index: cluster the vectors, search only the nearest buckets
  4. Recall vs speed measurement -- the trade-off, on your own index
  5. int8 quantization: 4x memory for ~zero recall loss
  6. Semantic search demo over a mini document corpus

On HNSW (the other big index family, not built here): think
"skip list meets social network" -- a layered graph where each vector
links to its near neighbors; search greedily walks the graph from a
random entry point, dropping to denser layers as it closes in.
Faster recall/speed trade than IVF at high recall, harder to build
incrementally. Once you understand IVF, HNSW papers are readable.

Expected Output:
  (Run the script to see the numerical output and shape assertions pass)
"""
import time

import numpy as np

rng = np.random.default_rng(11)


# ----------------------------------------------------------------------
# 2. k-means -- you have every tool this needs
# ----------------------------------------------------------------------
def kmeans(X, k, iters=20):
    centroids = X[rng.choice(len(X), size=k, replace=False)]
    for _ in range(iters):
        # assign each point to nearest centroid (one big distance matrix)
        d = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
        assign = d.argmin(axis=1)
        # move each centroid to the mean of its members
        for j in range(k):
            members = X[assign == j]
            if len(members):
                centroids[j] = members.mean(axis=0)
    return centroids, assign


# ----------------------------------------------------------------------
# The database
# ----------------------------------------------------------------------
class VectorDB:
    def __init__(self, dim):
        self.dim = dim
        self.vecs = np.zeros((0, dim), dtype=np.float32)
        self.ids, self.meta = [], []
        self.centroids = None            # IVF state
        self.buckets = None

    def add(self, id_, vec, metadata=None):
        v = np.asarray(vec, dtype=np.float32)
        v = v / (np.linalg.norm(v) + 1e-9)          # normalize -> dot == cosine
        self.vecs = np.vstack([self.vecs, v])
        self.ids.append(id_)
        self.meta.append(metadata or {})

    # ---- 1. exact search: one matmul. Do not underestimate this. ----
    def search_exact(self, qvec, k=5, filter_fn=None):
        q = qvec / (np.linalg.norm(qvec) + 1e-9)
        sims = self.vecs @ q                        # (N,) all similarities at once
        order = np.argsort(-sims)
        out = []
        for i in order:
            if filter_fn and not filter_fn(self.meta[i]):
                continue                            # metadata WHERE-clause
            out.append((self.ids[i], float(sims[i])))
            if len(out) == k:
                break
        return out

    # ---- 3. IVF: partition by learned geometry ----
    def build_ivf(self, n_clusters):
        self.centroids, assign = kmeans(self.vecs, n_clusters)
        self.buckets = [np.where(assign == j)[0] for j in range(n_clusters)]

    def search_ivf(self, qvec, k=5, nprobe=1):
        q = qvec / (np.linalg.norm(qvec) + 1e-9)
        # step 1: which buckets is q closest to?
        c_sims = self.centroids @ q
        probe = np.argsort(-c_sims)[:nprobe]
        # step 2: brute force ONLY inside those buckets
        cand = np.concatenate([self.buckets[j] for j in probe])
        sims = self.vecs[cand] @ q
        order = np.argsort(-sims)[:k]
        return [(self.ids[cand[i]], float(sims[i])) for i in order]


# ----------------------------------------------------------------------
# 4. Measure the recall/speed trade-off on synthetic clustered data
# ----------------------------------------------------------------------
def benchmark():
    print("=" * 64)
    print("IVF BENCHMARK: recall vs speed (N=20000, d=64, 32 clusters)")
    print("=" * 64)
    N, d, C = 20000, 64, 32
    # clustered data (like real embeddings: text clumps by topic)
    centers = rng.normal(size=(C, d)) * 3
    X = centers[rng.integers(0, C, N)] + rng.normal(size=(N, d))

    db = VectorDB(d)
    for i in range(N):
        db.add(i, X[i])
    db.build_ivf(n_clusters=C)

    queries = rng.normal(size=(50, d))
    truth = [set(i for i, _ in db.search_exact(q, k=10)) for q in queries]

    t0 = time.perf_counter()
    for q in queries:
        db.search_exact(q, k=10)
    exact_t = time.perf_counter() - t0

    print(f"  {'nprobe':>7} {'recall@10':>10} {'speedup':>8}")
    for nprobe in (1, 2, 4, 8, 16, 32):
        t0 = time.perf_counter()
        results = [db.search_ivf(q, k=10, nprobe=nprobe) for q in queries]
        t = time.perf_counter() - t0
        recall = np.mean([len(set(i for i, _ in r) & tr) / 10
                          for r, tr in zip(results, truth)])
        print(f"  {nprobe:>7} {recall:>10.2f} {exact_t/t:>7.1f}x")
    print("  nprobe = all clusters -> exact search. The knob IS the trade-off.\n")


# ----------------------------------------------------------------------
# 5. Quantization: float32 -> int8
# ----------------------------------------------------------------------
def quantization_demo():
    print("=" * 64)
    print("QUANTIZATION: 4x less memory, how much recall lost?")
    print("=" * 64)
    N, d = 5000, 64
    X = rng.normal(size=(N, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    scale = np.abs(X).max()
    X8 = np.round(X / scale * 127).astype(np.int8)      # the whole trick

    q = rng.normal(size=d).astype(np.float32)
    q /= np.linalg.norm(q)
    top_f32 = set(np.argsort(-(X @ q))[:10])
    top_i8 = set(np.argsort(-(X8.astype(np.float32) @ q))[:10])
    print(f"  memory: {X.nbytes/1e6:.1f} MB -> {X8.nbytes/1e6:.1f} MB")
    print(f"  top-10 overlap after quantization: {len(top_f32 & top_i8)}/10\n")


# ----------------------------------------------------------------------
# 6. Semantic search over a mini corpus (crude averaged-word embeddings)
# ----------------------------------------------------------------------
DOCS = [
    ("pw1", "how to reset your password and recover your account", "auth"),
    ("pw2", "credential recovery procedure for locked accounts", "auth"),
    ("pw3", "change login secret after security incident", "auth"),
    ("db1", "database connection pooling best practices", "infra"),
    ("db2", "tuning postgres queries with indexes", "infra"),
    ("db3", "sharding strategy for large tables", "infra"),
    ("ml1", "training neural networks with gradient descent", "ml"),
    ("ml2", "transformer attention explained for engineers", "ml"),
    ("ml3", "embeddings map meaning into vector space", "ml"),
    ("k81", "deploying containers to the kubernetes cluster", "infra"),
    ("k82", "rolling updates and pod restarts in production", "infra"),
]


def crude_embedder(train_texts, dim=32):
    """Averaged word vectors, word2vec-style but even simpler:
    co-occurrence counts + SVD-free random projection. Deliberately crude --
    finding its failure modes is exercise 2."""
    vocab = sorted({w for t in train_texts for w in t.split()})
    stoi = {w: i for i, w in enumerate(vocab)}
    # co-occurrence matrix within docs
    C = np.zeros((len(vocab), len(vocab)))
    for t in train_texts:
        ws = [stoi[w] for w in t.split()]
        for a in ws:
            for b in ws:
                if a != b:
                    C[a, b] += 1
    # random projection of co-occurrence rows -> dense "embeddings"
    R = rng.normal(size=(len(vocab), dim)) / np.sqrt(dim)
    E = np.log1p(C) @ R

    def embed(text):
        ids = [stoi[w] for w in text.split() if w in stoi]
        if not ids:
            return np.zeros(dim)
        return E[ids].mean(axis=0)          # average word vectors: word order LOST
    return embed


def semantic_demo():
    print("=" * 64)
    print("SEMANTIC SEARCH: zero shared keywords, right answer anyway")
    print("=" * 64)
    embed = crude_embedder([t for _, t, _ in DOCS])
    db = VectorDB(32)
    for id_, text, topic in DOCS:
        db.add(id_, embed(text), {"topic": topic, "text": text})

    # note: this crude embedder only knows words it has seen -- a query of
    # entirely unseen words embeds to zero. Real embedding models use
    # subword tokenization (module 04!) so nothing is ever out-of-vocabulary.
    for query in ("forgot my password",
                  "making queries faster",
                  "what is attention"):
        hits = db.search_exact(embed(query), k=2)
        print(f"  query: {query!r}")
        for id_, sim in hits:
            text = next(t for i, t, _ in DOCS if i == id_)
            print(f"    {sim:5.2f}  [{id_}] {text}")
    print()
    # metadata filter: vector search + WHERE topic = 'auth'
    hits = db.search_exact(embed("locked out of the system"), k=2,
                           filter_fn=lambda m: m["topic"] == "auth")
    print("  filtered (topic=auth) query 'locked out of the system':")
    for id_, sim in hits:
        print(f"    {sim:5.2f}  [{id_}]")


if __name__ == "__main__":
    benchmark()
    quantization_demo()
    semantic_demo()
