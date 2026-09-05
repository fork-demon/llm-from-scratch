"""
Module 12 -- A complete RAG pipeline from scratch. No API keys.

  python mini_rag.py                 # normal demo queries
  python mini_rag.py --show-failure  # watch a retrieval miss happen

Pipeline:
  INGEST: docs -> chunk (with overlap) -> embed -> vector store
  QUERY : question -> embed -> top-k retrieve -> assemble prompt
          -> answer FROM CONTEXT ONLY (extractive stand-in for an LLM)

Swapping in a real LLM changes ~3 lines (see bottom of file).

Expected Output:
  (Run the script to see the numerical output and shape assertions pass)
"""
import sys

import numpy as np

rng = np.random.default_rng(12)

# ----------------------------------------------------------------------
# A tiny internal "wiki" -- the knowledge that is NOT in any model's weights
# ----------------------------------------------------------------------
CORPUS = {
    "onboarding.md": (
        "New engineers get laptop access on day one. "
        "The onboarding buddy is assigned by the team lead. "
        "All new hires must complete security training within two weeks. "
        "Production access requires completing the incident response course. "
        "The engineering handbook lives in the internal wiki."
    ),
    "deploy-policy.md": (
        "Deployments to production happen through the CI pipeline only. "
        "Manual deploys are forbidden except during a declared incident. "
        "Every deploy requires two approvals on the pull request. "
        "Rollbacks are triggered from the deploy dashboard. "
        "Deploy freezes apply during the last week of each quarter."
    ),
    "oncall.md": (
        "The oncall rotation changes every Monday at 10am. "
        "Primary oncall must acknowledge pages within five minutes. "
        "Secondary oncall is paged if the primary does not respond. "
        "After an incident the oncall engineer writes the postmortem. "
        "Postmortems are blameless and due within three business days."
    ),
    "expenses.md": (
        "Engineers may expense up to 500 dollars per year for learning materials. "
        "Conference travel requires manager approval in advance. "
        "Receipts must be submitted within thirty days of purchase. "
        "Home office equipment is budgeted separately at 1000 dollars."
    ),
}


# ----------------------------------------------------------------------
# 1. CHUNKING -- undignified string splitting, outsized quality impact
# ----------------------------------------------------------------------
def chunk(text, source, sentences_per_chunk=2, overlap=1):
    sents = [s.strip() + "." for s in text.split(".") if s.strip()]
    chunks, i = [], 0
    step = max(1, sentences_per_chunk - overlap)
    while i < len(sents):
        body = " ".join(sents[i:i + sentences_per_chunk])
        chunks.append({"text": body, "source": source, "pos": i})
        i += step
    return chunks


# ----------------------------------------------------------------------
# 2. EMBEDDING -- a crude but honest embedder:
#    TF-IDF bag of words (rare words matter, "the" doesn't), lightly
#    smoothed by a co-occurrence matrix so weight bleeds onto words that
#    appear together in the corpus (a whiff of semantics). Real systems
#    use a transformer encoder (module 11 doc); the pipeline is identical.
# ----------------------------------------------------------------------
def tokenize(text):
    return text.lower().replace(".", "").replace(",", "").split()


def build_embedder(texts, smooth=0.3):
    vocab = sorted({w for t in texts for w in tokenize(t)})
    stoi = {w: i for i, w in enumerate(vocab)}
    Vn = len(vocab)

    # idf: words in every chunk (the, is, ...) get ~zero weight
    df = np.zeros(Vn)
    for t in texts:
        for w in set(tokenize(t)):
            df[stoi[w]] += 1
    idf = np.log(len(texts) / (df + 1e-9))

    # co-occurrence (within a chunk), row-normalized -> "related words"
    C = np.zeros((Vn, Vn))
    for t in texts:
        ws = [stoi[w] for w in tokenize(t)]
        for a in ws:
            for b in ws:
                if a != b:
                    C[a, b] += 1
    C = C / (C.sum(axis=1, keepdims=True) + 1e-9)

    def embed(text):
        v = np.zeros(Vn)
        for w in tokenize(text):
            if w in stoi:
                v[stoi[w]] += idf[stoi[w]]        # tf-idf part
        v = v + smooth * (v @ C)                  # bleed onto related words
        return v / (np.linalg.norm(v) + 1e-9)
    return embed


# ----------------------------------------------------------------------
# 3. VECTOR STORE -- module 11, minimal version (exact search: N is tiny)
# ----------------------------------------------------------------------
class Store:
    def __init__(self):
        self.vecs, self.chunks = [], []

    def add(self, vec, chunk_):
        self.vecs.append(vec)
        self.chunks.append(chunk_)

    def search(self, qvec, k=3):
        sims = np.array(self.vecs) @ qvec
        order = np.argsort(-sims)[:k]
        return [(self.chunks[i], float(sims[i])) for i in order]


# ----------------------------------------------------------------------
# 4. PROMPT ASSEMBLY -- printed in full: see exactly what the LLM gets
# ----------------------------------------------------------------------
def assemble_prompt(question, retrieved):
    ctx = "\n".join(f"[{i+1}] ({c['source']}) {c['text']}"
                    for i, (c, _) in enumerate(retrieved))
    return (
        "Answer using ONLY the context below. If the answer is not in the\n"
        "context, say 'Not found in the provided context.'\n\n"
        f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"
    )


# ----------------------------------------------------------------------
# 5. THE "LLM" -- an honest extractive stand-in. It can ONLY answer from
#    context, so it demonstrates grounding + citation + refusal, which is
#    exactly the behavior the prompt above asks a real LLM for.
# ----------------------------------------------------------------------
def extractive_answer(question, retrieved, embed, threshold=0.35):
    qv = embed(question)
    best, best_sim, best_cite = None, -1, None
    for i, (c, _) in enumerate(retrieved):
        for sent in c["text"].split("."):
            if not sent.strip():
                continue
            sim = float(embed(sent) @ qv)
            if sim > best_sim:
                best, best_sim, best_cite = sent.strip(), sim, (i + 1, c["source"])
    if best_sim < threshold:
        return f"Not found in the provided context. (best match {best_sim:.2f})"
    return f"{best}. [source {best_cite[0]}: {best_cite[1]}] (sim {best_sim:.2f})"


# ----------------------------------------------------------------------
# Wire it together
# ----------------------------------------------------------------------
def main(show_failure=False):
    # INGEST
    all_chunks = []
    for src, text in CORPUS.items():
        all_chunks.extend(chunk(text, src))
    embed = build_embedder([c["text"] for c in all_chunks])
    store = Store()
    for c in all_chunks:
        store.add(embed(c["text"]), c)
    print(f"ingested {len(CORPUS)} docs -> {len(all_chunks)} chunks\n")

    if show_failure:
        # negation/paraphrase-heavy query: crude embeddings often miss.
        queries = ["can I deploy without the pipeline",
                   "what is the wifi password"]   # genuinely absent
    else:
        queries = [
            "how quickly must I acknowledge pages",
            "what is the budget for learning materials",
            "who writes the postmortem after an incident",
            "when do deploy freezes apply",
        ]

    for q in queries:
        print("=" * 64)
        print(f"QUERY: {q}")
        print("=" * 64)
        retrieved = store.search(embed(q), k=3)
        print("retrieved chunks:")
        for c, sim in retrieved:
            print(f"  {sim:5.2f}  ({c['source']}) {c['text'][:60]}...")
        prompt = assemble_prompt(q, retrieved)
        print("\n--- prompt sent to the 'LLM' " + "-" * 30)
        print(prompt)
        print("-" * 60)
        print("ANSWER:", extractive_answer(q, retrieved, embed))
        print()

    # To use a real LLM instead of the extractive stand-in, replace the
    # extractive_answer() call with (any chat API):
    #
    #   response = client.messages.create(
    #       model=..., max_tokens=300,
    #       messages=[{"role": "user", "content": prompt}])
    #   answer = response.content[0].text
    #
    # The pipeline shape does not change. That is the point of RAG.


if __name__ == "__main__":
    main(show_failure="--show-failure" in sys.argv)
