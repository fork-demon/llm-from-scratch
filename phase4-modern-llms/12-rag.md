# Module 12 — RAG: Wiring It All Together

> **⏱️ Time:** ~1-2 weeks.  
> **What you'll build:** Core mechanics and conceptual understanding.

---

## 🎯 TL;DR
1. **The Core Problem:** The challenge this module solves.
2. **The Mechanism:** How we solve it using first principles.
3. **The Payoff:** What you can do with this new capability.

---


**Time: about 2 weeks. Code: `mini_rag.py`. The capstone — every module in the course shows up somewhere in this one pipeline.**

## The Problem (which you derived yourself in module 10)

Let's collect what you know about model weights as a knowledge store, because module 10 handed you a full indictment: knowledge is frozen at training time (yesterday's news isn't in there), updating it means expensive retraining, sparse facts are stored unreliably, your company's private documents were never in the training data at all, and — the structural kicker — the architecture has no `NOT FOUND`, so where knowledge is missing, plausible fabrication comes out instead.

If a colleague described a datastore like that, you'd stop them halfway through: *"Don't store the data there. Fetch it at request time."*

That's RAG — **Retrieval-Augmented Generation** — and you've now derived it rather than memorized the acronym:

> Don't teach the model your facts. At request time, *find* the relevant documents (module 11's semantic search), paste them into the prompt (module 09's context window), and ask the model to answer *from what it just read*.

Notice what this does to the model's job. It shifts from **recall** ("what do you remember about our deploy policy?") to **reading comprehension** ("here are three paragraphs of our deploy policy — answer using them"). LLMs are mediocre and hallucination-prone at recall; they are *strong* at reading comprehension over text sitting right in the context. RAG moves the work to where the model is good. Architecturally, it's a pattern you've built a dozen times: the LLM is a stateless compute node, and RAG bolts a data layer in front of it.

## The Pipeline

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 460" font-family="sans-serif">
  <rect width="760" height="460" fill="#ffffff"/>
  <text x="380" y="26" text-anchor="middle" font-size="17" font-weight="bold" fill="#333">RAG: fetch knowledge at request time; the LLM does reading comprehension, not recall</text>

  <defs>
    <marker id="flow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 z" fill="#888"/>
    </marker>
  </defs>

  <!-- ingest lane -->
  <text x="60" y="70" font-size="13" fill="#4A90D9" font-weight="bold">INGEST (offline)</text>
  <rect x="60" y="85" width="110" height="46" rx="8" fill="#f6f8fa" stroke="#bbb"/>
  <text x="115" y="105" text-anchor="middle" font-size="12" fill="#333">your docs</text>
  <text x="115" y="121" text-anchor="middle" font-size="10" fill="#888">wiki, PDFs, tickets</text>

  <rect x="220" y="85" width="110" height="46" rx="8" fill="#f6f8fa" stroke="#bbb"/>
  <text x="275" y="105" text-anchor="middle" font-size="12" fill="#333">chunk</text>
  <text x="275" y="121" text-anchor="middle" font-size="10" fill="#888">~100s of tokens + overlap</text>

  <rect x="380" y="85" width="110" height="46" rx="8" fill="#eef4fb" stroke="#4A90D9"/>
  <text x="435" y="105" text-anchor="middle" font-size="12" fill="#333">embed</text>
  <text x="435" y="121" text-anchor="middle" font-size="10" fill="#888">text → vector (mod 05/11)</text>

  <rect x="540" y="85" width="130" height="46" rx="8" fill="#eef4fb" stroke="#4A90D9" stroke-width="2"/>
  <text x="605" y="105" text-anchor="middle" font-size="12" fill="#333">vector DB</text>
  <text x="605" y="121" text-anchor="middle" font-size="10" fill="#888">IVF index (mod 11)</text>

  <line x1="170" y1="108" x2="212" y2="108" stroke="#888" stroke-width="2" marker-end="url(#flow)"/>
  <line x1="330" y1="108" x2="372" y2="108" stroke="#888" stroke-width="2" marker-end="url(#flow)"/>
  <line x1="490" y1="108" x2="532" y2="108" stroke="#888" stroke-width="2" marker-end="url(#flow)"/>

  <!-- query lane -->
  <text x="60" y="205" font-size="13" fill="#55A868" font-weight="bold">QUERY (online)</text>
  <rect x="60" y="220" width="110" height="46" rx="8" fill="#f6f8fa" stroke="#bbb"/>
  <text x="115" y="240" text-anchor="middle" font-size="12" fill="#333">question</text>
  <text x="115" y="256" text-anchor="middle" font-size="10" fill="#888">"when do deploys freeze?"</text>

  <rect x="220" y="220" width="110" height="46" rx="8" fill="#eef4fb" stroke="#4A90D9"/>
  <text x="275" y="240" text-anchor="middle" font-size="12" fill="#333">embed</text>
  <text x="275" y="256" text-anchor="middle" font-size="10" fill="#888">same model as ingest!</text>

  <rect x="380" y="220" width="110" height="46" rx="8" fill="#eefaf1" stroke="#55A868"/>
  <text x="435" y="240" text-anchor="middle" font-size="12" fill="#333">top-k search</text>
  <text x="435" y="256" text-anchor="middle" font-size="10" fill="#888">nearest chunks by dot product</text>

  <line x1="170" y1="243" x2="212" y2="243" stroke="#888" stroke-width="2" marker-end="url(#flow)"/>
  <line x1="330" y1="243" x2="372" y2="243" stroke="#888" stroke-width="2" marker-end="url(#flow)"/>
  <line x1="605" y1="131" x2="465" y2="212" stroke="#4A90D9" stroke-width="2" marker-end="url(#flow)" stroke-dasharray="5,4"/>

  <!-- prompt assembly -->
  <rect x="540" y="200" width="160" height="90" rx="8" fill="#f6f8fa" stroke="#bbb"/>
  <text x="620" y="222" text-anchor="middle" font-size="12" fill="#333" font-weight="bold">assemble prompt</text>
  <text x="620" y="242" text-anchor="middle" font-size="10" fill="#666">"Answer ONLY from this</text>
  <text x="620" y="256" text-anchor="middle" font-size="10" fill="#666">context: [chunk1][chunk2]...</text>
  <text x="620" y="270" text-anchor="middle" font-size="10" fill="#666">Question: ..."</text>
  <line x1="490" y1="243" x2="532" y2="243" stroke="#888" stroke-width="2" marker-end="url(#flow)"/>

  <!-- LLM + answer -->
  <rect x="280" y="340" width="200" height="52" rx="10" fill="#fdf3ee" stroke="#E8734A" stroke-width="2"/>
  <text x="380" y="362" text-anchor="middle" font-size="13" fill="#333" font-weight="bold">LLM (modules 06–09)</text>
  <text x="380" y="380" text-anchor="middle" font-size="10" fill="#888">reads the context, not its weights</text>
  <line x1="600" y1="290" x2="465" y2="338" stroke="#888" stroke-width="2" marker-end="url(#flow)"/>

  <rect x="60" y="340" width="160" height="52" rx="8" fill="#eefaf1" stroke="#55A868" stroke-width="2"/>
  <text x="140" y="362" text-anchor="middle" font-size="12" fill="#333">grounded answer</text>
  <text x="140" y="380" text-anchor="middle" font-size="10" fill="#888">with citations to chunks</text>
  <line x1="272" y1="366" x2="228" y2="366" stroke="#888" stroke-width="2" marker-end="url(#flow)"/>

  <text x="380" y="435" text-anchor="middle" font-size="12" fill="#888">When a RAG system is bad, it is almost always retrieval (left half), not the LLM — debug there first.</text>
</svg>

*Two lanes: an offline lane that files your documents into a vector index, and an online lane that fetches, assembles, and answers*

Each arrow in that picture hides one real design decision. Let's take them in order, honestly.

**Chunking — the unglamorous decision that dominates quality.** You can't embed a whole book as one vector: module 11 showed you that averaging smears meaning, and a 300-page average is meaning-flavored mush. You also can't embed lone sentences: "It requires two approvals" is useless without knowing what "it" is. The working compromise is chunks of a few hundred tokens with some *overlap* between neighbors — overlap so that a fact straddling a boundary survives in at least one piece. Here's the part nobody says loudly enough: chunking is undignified string-splitting, there is no elegant theory of it, only heuristics (respect paragraph boundaries, keep headings attached to their sections) — and it affects final answer quality more than most fancy choices downstream. Real RAG systems are most often bad *here*, at the least glamorous stage. Exercise 1 makes sure you never treat it as a detail.

**Retrieval — module 11, verbatim.** Embed the query with the *same* model used at ingest (mixing embedding models is like comparing hashes from different hash functions — a classic silent failure), then top-k nearest chunks. Two production notes for your mental shelf: **hybrid search** (keyword BM25 + vectors, fused) beats either alone — your module 11 exercise 2 failure case is the reason; and the single highest-ROI upgrade in the industry is a **reranker** — a slower, smarter model that reorders the top-50 candidates down to a final 5. Cheap-fast-recall stage feeding an expensive-precise-precision stage: you've built this two-tier shape in other systems; it has the same justification here.

**Prompt assembly — where hallucination gets handcuffed.** The retrieved chunks and the question get formatted into a prompt with an instruction that carries most of RAG's safety value: *"Answer using ONLY the context below. If the answer is not in the context, say so."* Recall module 10: hallucination is unconstrained plausible-text generation. This instruction converts the task into constrained extraction — the answer is either in the provided text or the correct response is "not found." It *reduces* fabrication rather than eliminating it (the model can still blend context with its priors), but the reduction is large, and it's free.

**Generation, with receipts.** Because *you* chose which chunks went into the prompt, you can cite them — "per deploy-policy.md, section 2" — and a user can check. Citations are the accountability that raw LLM answers structurally lack (module 10: no truth register). One caution so you don't over-trust: grounded ≠ guaranteed-faithful. Mature systems add a verification pass — does the cited chunk actually support the sentence citing it?

## When RAG Fails — Every Failure Mode Traces to a Module You've Done

This section is the practical payoff of the whole course. When your RAG system misbehaves, you now have the diagnostic map:

**Retrieval missed.** The right document exists but wasn't in the top-k — bad chunking cut the fact in half, or the embedder was blind to the phrasing (module 11 exercise 2: negation blindness — "not required" embeds beside "required"). Downstream, the model — handed irrelevant context — answers from its weights instead. You get module 10's hallucination *with* a false aura of groundedness. Worst failure mode in the business, and it started at retrieval.

**Context stuffing.** "Let's retrieve top-20, to be safe." Now the actually-relevant paragraph is buried under nineteen near-relevant ones. Two costs: the good signal is diluted, and there's a measured phenomenon called **lost in the middle** — models attend well to the start and end of a long context and poorly to the middle (module 07's attention, spread thin over long spans, empirically favors the edges). More context is not monotonically better; curation beats volume.

**Contradictory sources.** The 2023 policy says X, the 2024 revision says Y, retrieval returned both. The model has no mechanism for adjudicating truth (module 10) — only plausibility — so you get X, Y, or a confident smoothie of both. The fix isn't a smarter model; it's boring data hygiene: version metadata and module 11's metadata filtering at query time.

The meta-lesson, worth the price of the module: **when a RAG system is bad, it is almost always the retrieval, not the LLM.** The model is the flashy component, so it gets the blame and the tuning attention — but debug the boring data layer first. You now have a principled reason to, which is more than most teams debugging RAG can say.


> **🛑 CHECKPOINT:** Run the code and modify it before proceeding. Reading without running is an illusion of knowledge.

## The Code

`mini_rag.py` — the complete pipeline, self-contained, no API keys. A small built-in "company wiki" (deploy policy, oncall runbook, expense rules), the chunker with overlap, module 11's embedder and vector store inline, retrieval, and prompt assembly — with the assembled prompt *printed in full* each query, so you see exactly what an LLM would receive. Standing in for the LLM: an honest *extractive* answerer that can only quote sentences from the retrieved chunks, cite its source, or say "not found in the provided context" — which makes grounding, citation, and refusal all visible in a component with no ability to fabricate. Run `python mini_rag.py --show-failure` to watch a deliberately engineered retrieval miss and learn its signature. The file's closing comment shows the ~3 lines that swap the extractive stand-in for a real LLM API — the pipeline's shape doesn't change at all, which is rather the point.

## Modify-It Exercises

1. Change the chunk size: one sentence, three sentences, whole document. Run the same five queries against all three. Watch answers appear and disappear based purely on how you split strings. You will never again believe chunking is a detail.
2. Add a `date` field to the metadata, create a contradicting 2024 version of one document, and make "what does the *current* policy say?" answer correctly via a metadata filter. You've implemented the fix for the contradictory-sources failure.
3. Log the top retrieval score for each query. Pick a threshold below which the system answers "I don't have information on this" *without even calling the answerer*. Tune it against a few in-corpus and out-of-corpus questions. You've just built the honesty feature that module 10 proved the bare architecture lacks.
4. Build the evaluation before you need it: ten (question → chunk-that-should-be-retrieved) pairs, and a script measuring how often the right chunk lands in the top-k. Now re-run it while varying chunk size and k. Congratulations — you're doing *retrieval engineering*, which is what production RAG work actually is, day to day.
5. *(The full-circle exercise.)* Swap in a real LLM API and re-run `--show-failure`. Where your extractive answerer said "not found," watch the real LLM confidently synthesize an answer from irrelevant context. Sit with the difference. Then write three sentences on what this means for monitoring RAG in production. This is the entire course's arc — from "text becomes tokens" to "here is precisely why and where the impressive machine deceives" — closed in one experiment.

## Best External Resources

- Lewis et al. 2020, the paper that named RAG — read §1–2 for the framing; the specific architecture is dated.
- Anthropic's *"Contextual Retrieval"* engineering post — a practical, modern upgrade path starting from exactly the pipeline you just built.

## The End — Look at What's on Your Disk

Built by you, with your own hands, small enough to hold in your head: a tokenizer; a neural network with backprop you wrote and gradient-checked; word embeddings whose geometry you watched crystallize; a language model that converged to the count table and then beat it; attention, verified to twelve decimal places; a working GPT that learned Shakespeare's shape from scratch; a KV cache with a measured 10× speedup; a vector database with a benchmarked recall knob; and a RAG pipeline whose failure modes you can diagnose by module number.

Where to next depends on which itch survived: **interpretability** (transformer-circuits.pub — you now hold exactly the prerequisites); **training at scale** (nanoGPT → llm.c, and the Chinchilla and Llama papers, which will read as engineering docs); **inference systems** (the vLLM/PagedAttention paper — it's virtual memory for your module 09 KV cache, and you'll grin at the title alone); or **applied AI engineering** (evals, agents, tool use — where your production instincts are a scarce and valuable asset).

And read *Attention Is All You Need* one final time on your way out. Seven modules ago it was a famous paper. Now it's a design doc for a system you've shipped.
