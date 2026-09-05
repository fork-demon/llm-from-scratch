# Module 11 — Vector Databases & Semantic Search From Scratch

> **⏱️ Time:** ~1-2 weeks.  
> **What you'll build:** Core mechanics and conceptual understanding.

---

## 🎯 TL;DR
1. **The Core Problem:** The challenge this module solves.
2. **The Mechanism:** How we solve it using first principles.
3. **The Payoff:** What you can do with this new capability.

---


**Time: about 2 weeks. Code: `vector_db.py`. This is the comfort-zone module — a systems problem wearing ML clothes. Enjoy being the expert in the room again.**

## The Problem

Try this search in any keyword-based system: the user types *"how do I reset my password"* and the answer lives in a document titled *"Credential recovery procedure for locked accounts."* Zero words in common. Keyword search returns nothing, and it's not a bug — matching words is all keyword search *is*, and these two texts share none.

But you spent Phase 2 learning that meaning has a geometry. If both texts become vectors — module 01's points-in-space — then "reset my password" and "credential recovery procedure" land *near each other*, because they mean nearly the same thing, regardless of sharing no vocabulary. So semantic search is three steps:

1. Embed every document as a vector, ahead of time.
2. Embed the incoming query as a vector.
3. Return the documents whose vectors are nearest to the query's. (Nearest by what measure? The dot product / cosine — module 00's agreement meter, in its third starring role.)

A **vector database** is the storage plus the index that makes step 3 fast when you have millions of documents. That's the whole product category. By the end of this module you'll have built one, and vendor marketing in this space will never impress you again.

## The Missing Piece: Embedding Whole Texts, Not Words

Your module 05 model embeds *words*. Search needs one vector for a whole *sentence or document*. Where do those come from?

Modern **embedding models** (the sentence-transformers family, the OpenAI/Voyage embedding APIs) are transformer encoders — your module 08 architecture with one variation you can reason about: the causal mask is *removed*. Why? The mask existed to hide the future during next-token training. But when you're representing a complete document, there's no future to hide — every token may look at every token. Trained how? By a **contrastive** game, which is module 05's fake-task trick with a new task: take pairs of texts known to be related (duplicate questions from forums, a query and the doc users clicked), *pull* their embeddings together; take unrelated pairs, *push* them apart. Millions of repetitions, and the geometry that crystallizes is exactly the one search needs: related-in-the-retrieval-sense = nearby.

For our from-scratch build, the code uses a deliberately crude document embedder — essentially *averaged word vectors*, built with module 05 machinery — because it keeps the entire pipeline yours, with no API keys and no black boxes. Crude has a curriculum benefit: its failure modes are *instructive*. Averaging destroys word order ("dog bites man" and "man bites dog" embed identically), and exercise 2 has you hunt down a query where this matters. Swapping in a real embedding model later is a two-line change; the pipeline doesn't move.

## Finding the Nearest Vectors: Honest Answer First

Here's the part vendors would prefer you not know. The exact method — compute the query's dot product against *all* N stored vectors, take the top k — is **one matrix-vector multiply** (module 00, section 4, literally). NumPy does this so fast that up to something like a *million* vectors, brute force is completely fine. A huge fraction of real "vector database" workloads need nothing beyond this. Always benchmark the dumb thing first; in this domain the dumb thing is startlingly good.

But at tens of millions of vectors, O(N·d) per query eventually hurts. And here's an interesting wall: in high dimensions there is **no exact shortcut** — the tree tricks that work in 2D or 3D (KD-trees and friends) collapse in 768 dimensions (module 01's "high-dimensional space is weird" striking again, this time against you). So every production index accepts a trade: **approximate** nearest neighbors — a little recall given up for orders of magnitude of speed.

## IVF: The Index You'll Build

Of the approximate index families, we build **IVF** ("inverted file index" — the default in FAISS, Meta's ubiquitous library), because it's the clearest and because you already know both of its ingredients:

**Setup (once):** cluster all your vectors into, say, 32 buckets using **k-means** — an algorithm you'll write in ~15 lines, and it's pleasingly simple: assign every point to its nearest center, move each center to the average of its points, repeat until settled.

**Query (every time):** compare the query against just the 32 bucket *centers* (32 dot products instead of a million). Pick the closest bucket or few. Brute-force search *only inside those buckets*.

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 420" font-family="sans-serif">
  <rect width="760" height="420" fill="#ffffff"/>
  <text x="380" y="26" text-anchor="middle" font-size="17" font-weight="bold" fill="#333">IVF index: cluster the vectors, search only the nearest buckets</text>

  <!-- cluster A -->
  <ellipse cx="180" cy="150" rx="95" ry="70" fill="#4A90D9" opacity="0.10"/>
  <g fill="#4A90D9" opacity="0.75">
    <circle cx="150" cy="130" r="4"/><circle cx="190" cy="115" r="4"/><circle cx="215" cy="150" r="4"/>
    <circle cx="160" cy="175" r="4"/><circle cx="200" cy="180" r="4"/><circle cx="135" cy="155" r="4"/>
  </g>
  <path d="M 172 142 l 16 0 m -8 -8 l 0 16" stroke="#1a5da8" stroke-width="3" fill="none"/>
  <text x="180" y="240" text-anchor="middle" font-size="12" fill="#4A90D9" font-weight="bold">bucket 1</text>

  <!-- cluster B (the winner) -->
  <ellipse cx="430" cy="140" rx="95" ry="70" fill="#55A868" opacity="0.14"/>
  <g fill="#55A868" opacity="0.85">
    <circle cx="400" cy="120" r="4"/><circle cx="445" cy="110" r="4"/><circle cx="465" cy="150" r="4"/>
    <circle cx="410" cy="165" r="4"/><circle cx="450" cy="172" r="4"/><circle cx="385" cy="145" r="4"/>
  </g>
  <path d="M 422 132 l 16 0 m -8 -8 l 0 16" stroke="#2e7d4f" stroke-width="3" fill="none"/>
  <text x="430" y="235" text-anchor="middle" font-size="12" fill="#55A868" font-weight="bold">bucket 2 ← searched</text>

  <!-- cluster C -->
  <ellipse cx="640" cy="180" rx="85" ry="65" fill="#E8734A" opacity="0.10"/>
  <g fill="#E8734A" opacity="0.75">
    <circle cx="615" cy="160" r="4"/><circle cx="655" cy="150" r="4"/><circle cx="670" cy="190" r="4"/>
    <circle cx="620" cy="205" r="4"/><circle cx="660" cy="210" r="4"/>
  </g>
  <path d="M 632 172 l 16 0 m -8 -8 l 0 16" stroke="#c0563a" stroke-width="3" fill="none"/>
  <text x="640" y="270" text-anchor="middle" font-size="12" fill="#E8734A" font-weight="bold">bucket 3</text>

  <!-- query -->
  <defs>
    <marker id="qv" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 z" fill="#9b59b6"/>
    </marker>
  </defs>
  <g>
    <circle cx="390" cy="320" r="8" fill="#9b59b6"/>
    <text x="390" y="348" text-anchor="middle" font-size="12" fill="#9b59b6" font-weight="bold">query vector</text>
  </g>
  <!-- compare to centroids -->
  <line x1="382" y1="312" x2="200" y2="165" stroke="#ccc" stroke-width="1.5" stroke-dasharray="5,4" marker-end="url(#qv)"/>
  <line x1="392" y1="310" x2="428" y2="155" stroke="#9b59b6" stroke-width="2.5" marker-end="url(#qv)"/>
  <line x1="398" y1="312" x2="620" y2="195" stroke="#ccc" stroke-width="1.5" stroke-dasharray="5,4" marker-end="url(#qv)"/>

  <text x="255" y="300" font-size="11" fill="#888">step 1: compare query to the C centroids (+) only</text>
  <text x="470" y="248" font-size="11" fill="#9b59b6">step 2: brute-force INSIDE the winning bucket(s)</text>

  <rect x="60" y="365" width="640" height="44" rx="10" fill="#f6f8fa" stroke="#ddd"/>
  <text x="380" y="383" text-anchor="middle" font-size="12" fill="#333">nprobe = how many buckets to check: 1 = fastest (misses neighbors near borders), all = exact search.</text>
  <text x="380" y="401" text-anchor="middle" font-size="12" fill="#888">A hash-partition scheme where the partition function is learned geometry (k-means). The knob IS the recall/speed trade-off.</text>
</svg>

*Ask the centers first, then search only the winning neighborhood*

If you're thinking "that's just hash partitioning where the partition function is learned from the data's geometry" — yes. Exactly. That sentence *is* IVF.

The design has one knob, and understanding it means understanding every ANN benchmark you'll ever read: **nprobe**, the number of buckets to search. Probe 1 bucket: fastest, but a true neighbor sitting just across a bucket border gets missed. Probe more: slower, better recall. Probe all: you've reinvented exact search. The code *measures* this trade-off on your own index — a printed table of recall vs. speedup as nprobe grows — so the recall/latency curves in vendor whitepapers become graphs you've personally generated.

(The other big index family, **HNSW**, gets two honest paragraphs in the code comments: think "skip list meets social network" — a layered graph where search greedily hops between neighbors, dropping to denser layers as it closes in. Different trade-offs, same spirit. Once IVF is yours, HNSW papers are readable.)

## The Rest of a Real Vector Database (all mapped to things you know)

What separates your 150-liner from Pinecone? Packaging, mostly — the physics is identical. The remaining features, each mapped to familiar ground: **metadata filtering** (`WHERE topic='auth'` intersected with vector search — thornier than it sounds with an approximate index: filter first or search first? Real systems agonize); **upserts and index drift** (data changes, clusters go stale, k-means gets re-run periodically — it's compaction); **sharding** (the same partition logic, spread across machines); and **quantization** (store vectors as int8 instead of float32 — 4× memory saved, and the code demonstrates in ten lines that the top-10 results barely change; the same trick module 10 mentioned for model weights).


> **🛑 CHECKPOINT:** Run the code and modify it before proceeding. Reading without running is an illusion of knowledge.

## The Code

`vector_db.py` — a `VectorDB` class with `add()` and `search()`, exact brute-force search (one matmul, respect it), k-means from scratch, the IVF index with the nprobe knob, a benchmark harness that prints your own recall-vs-speedup table on 20,000 synthetic clustered vectors, the int8 quantization demo, and a small end-to-end semantic search over a built-in corpus — where "forgot my password" correctly retrieves the credential-recovery document, zero keywords shared. The embedder is the crude module-05-style one, trained inline; the whole file runs in seconds with no network.

## Modify-It Exercises

1. Re-run the benchmark with 4, 16, and 64 clusters at fixed N. Where's the sweet spot? Compare against the folk rule "clusters ≈ √N" — does your data agree?
2. Find a query where the crude averaged-word embedder fails embarrassingly but plain keyword matching would have won. (Hunting hint: negation — "not required" averages to nearly the same vector as "required" — or word order.) Then write two sentences on why production search is almost always **hybrid**: classic keyword scoring (BM25) *plus* vectors, fused. Your failure case is the argument.
3. Add a similarity-metric option: raw dot product vs. cosine. Construct a case where they *rank differently* (hint: a long rambling document vs. a short focused one — raw dot product rewards length; is that relevance?). You've just had the debate every vector-DB team has had.
4. Implement `delete(id)`. Now delete 30% of your vectors and look at what's left of your IVF buckets. Congratulations — you've discovered why vector databases have background compaction jobs, by needing one.
5. *(Stretch)* Swap the crude embedder for a real one (a local sentence-transformers model, or any embedding API) and re-run your exercise-2 failure case. Watch it get fixed. That before/after is the value of contrastively-trained embeddings, measured on your own machine.

## Best External Resources

- Pinecone's *"Faiss: The Missing Manual"* series — the best applied writing on ANN indexes; read the IVF and HNSW chapters after building yours.
- Jay Alammar, *"The Illustrated Retrieval Transformer"* — a bridge to exactly where we're heading next.

Next: **12 — RAG.** The vector database meets the language model, and everything you've built in this course clicks together into one pipeline.
