import { Lesson, Why, Problem, MentalModel, TryIt, Numbers, TheMath, CodeIt, BreakIt, Exercises, CheckYourself, Remember, RealLLM } from '../components/lesson'
import { Callout, DeepDive, Equation, Flow, G, Term, ToyVsReal, WhyExists } from '../components/ui'
import { Code } from '../components/Code'
import { CodeExercise } from '../components/python'
import { Exercise, ExplainBack } from '../components/exercise'
import { RagPlayground } from '../interactive/RagPlayground'
import { IvfLab } from '../interactive/IvfLab'
import { RagTwoLanes } from '../illustrations/RagTwoLanes'

export default function RagLesson() {
  return (
    <Lesson id="rag">
      <Why>
        <p className="lede">Ask any LLM this:</p>
        <div className="card center" style={{ fontFamily: 'var(--serif)', fontSize: 21 }}>“How quickly must I acknowledge a page when I am on call at our company?”</div>
        <p>The honest answer is in <code>oncall.md</code> on your company wiki: five minutes. The model has never seen that file. It was not in the training data, so it is not in the weights.</p>
        <p>You know from <a href="#/lesson/why-llms-know">Why LLMs know things</a> what happens next. The model has no “I have no record of this” mechanism. It produces the most plausible continuation: “Typically within 15 minutes”. Fluent, confident, wrong.</p>
        <p>The same gap appears for anything <b>private</b> (your documents), anything <b>recent</b> (after training stopped), and anything <b>rare</b> (seen too few times to be stored reliably).</p>
        <Callout kind="idea">
          Part 9 asks one question three times: <b>what exactly changes?</b> In this lesson the answer is: <b>only the prompt</b>. The model and its weights stay exactly as they are. We look the answer up ourselves and paste it in front of the question.
        </Callout>
      </Why>

      <Problem>
        <p>So far, “How does an LLM know?” had one answer: patterns stored in weights during training. Now the question is: <b>what if the information is not inside the model?</b></p>
        <WhyExists
          problem="The model must answer from information it was never trained on: private, recent, or rare."
          naive="Train it on that information (fine-tune on the company wiki)."
          fails="Slow and costly to repeat every time a document changes. Facts seen a few times are stored unreliably. And you still cannot tell which document an answer came from."
          idea="Do not store the facts in the model at all. At question time, find the relevant text and put it into the prompt. The model’s job changes from remembering to reading."
          tradeoff="You now own a search system. If search brings back the wrong text, the model answers from the wrong text, or from its own guesses."
        />
        <Term
          name="Retrieval-augmented generation (RAG)"
          plain={<>Before calling the model, search your own documents for passages related to the question, and paste them into the prompt with the instruction “answer from this”.</>}
          example={<>Question: “when do deploy freezes apply?” → search finds a passage from <code>deploy-policy.md</code> → prompt = instruction + that passage + the question.</>}
          formal={<>answer = LLM(prompt(question, top-k chunks by embedding similarity to the question)). Weights unchanged.</>}
        />
        <Callout kind="dev">You have built this shape before. The LLM is a stateless compute node. RAG puts a data layer in front of it: <b>retrieval + context injection</b>. It is a search engine whose results page is read by a model instead of by a person.</Callout>
      </Problem>

      <MentalModel>
        <Callout kind="analogy">
          A closed-book exam tests what you memorised. An <b>open-book exam</b> lets you look things up, and tests whether you can read and use what you found. RAG turns every question into an open-book question.
          <br /><br />
          Where the analogy stops: a student chooses which page to open. Here the model chooses nothing. <em>Your code</em> picks the pages, by a fairly blunt similarity search, before the model sees the question. If your code opens the wrong page, the “student” never finds out that a better page existed.
        </Callout>
        <p>The pipeline has two halves that run at different times: ingest (once, offline) and query (every question). They meet at one table.</p>
        <RagTwoLanes />
        <p>Follow the highlighted rows. The three best-scoring chunks leave the table, land under “Context:” in the prompt, and come out as the citation. Nothing else in the picture is new to the model: the box marked “LLM, unchanged” is the same model with the same weights.</p>
        <p>Two pieces are new. The rest you already own.</p>
        <Term
          name="Chunk"
          plain={<>A passage of a document, small enough to be about one thing. Documents are cut into chunks because we retrieve passages, not whole files.</>}
          example={<>Two sentences of <code>oncall.md</code>. Neighbouring chunks share a sentence (“overlap”) so a fact that straddles a cut survives in at least one chunk.</>}
          formal={<>Production systems use chunks of a few hundred <G t="token">tokens</G>, split on headings and paragraphs where possible.</>}
        />
        <Term
          name="Vector database"
          plain={<>A store of (vector, text) pairs that answers one query: “which stored vectors point in the most similar direction to this one?”</>}
          example={<>Embed the question, take the <G t="dot-product">dot product</G> with every stored vector, sort, keep the top 3.</>}
          formal={<>Nearest-neighbour search under <G t="cosine">cosine similarity</G>, exact or approximate.</>}
        />
        <p>“Top-k” here means the k best-scoring chunks. It is the same idea as top-k sampling in <a href="#/lesson/inference">Inference</a> (keep the k best, drop the rest), applied to chunks instead of tokens.</p>
        <p>The vectors come from an <G t="embedding">embedding</G> model. In <a href="#/lesson/embeddings">lesson 4.2</a> one <em>word</em> became a vector. Here a whole <em>passage</em> becomes one vector, so that passages about similar things land close together.</p>
      </MentalModel>

      <TryIt title="Run the whole pipeline yourself">
        <p>Start with the first question and walk stages 1 to 6. Pay most attention to stage 5: that text is everything the LLM would ever see.</p>
        <RagPlayground />
        <h3>At scale: do we have to compare with every vector?</h3>
        <p>With 19 chunks, search is 19 dot products. As you saw in <a href="#/lesson/matrices">Matrices</a>, comparing one question against N stored vectors is a single matrix multiply. That stays fast up to around a million vectors. It is worth knowing how far the boring solution goes.</p>
        <p>At hundreds of millions of vectors, comparing against everything is too slow and too costly. The fix is the same one a database uses: an <b>index</b> that lets you skip most of the data. One simple kind is called IVF, for “inverted file”. Group the vectors into clusters once, then at query time search only the few clusters nearest the question. The number of clusters you search is called <b>nprobe</b>, and it is the one dial.</p>
        <IvfLab />
        <p>The real benchmark in the repo does this with 20,000 vectors of 64 numbers in 32 clusters, and asks for the 10 nearest neighbours. Exact search would find all 10. <b>Recall@10</b> is the share of those true 10 that the index actually found, so 0.51 means it missed about half of them:</p>
        <div className="table-scroll">
          <table className="plain mono" style={{ fontSize: 14 }}>
            <thead><tr><th>nprobe (clusters searched)</th><th>1</th><th>2</th><th>4</th><th>8</th><th>16</th><th>32 (all)</th></tr></thead>
            <tbody>
              <tr><td>recall@10</td><td>0.51</td><td>0.64</td><td>0.82</td><td>0.91</td><td>0.99</td><td>1.00</td></tr>
              <tr><td>speed vs exact search</td><td>12.3×</td><td>9.0×</td><td>4.1×</td><td>2.1×</td><td>0.8×</td><td>0.5×</td></tr>
            </tbody>
          </table>
        </div>
        <p className="muted" style={{ fontSize: 14.5 }}>Output of <code>python phase4-modern-llms/vector_db.py</code> on the machine used to write this lesson. Recall is reproducible (the data is seeded). The speed-ups are timings and will differ on yours.</p>
        <p>Read the two ends. At nprobe = 1 the search is 12 times faster and <b>misses half</b> of the true neighbours. At nprobe = 32 it searches everything, finds everything, and is <em>slower</em> than plain exact search, because it pays for the index on top. The knob <em>is</em> the trade-off. For RAG this matters because a missed neighbour is a passage that never reaches the prompt.</p>
      </TryIt>

      <Numbers>
        <p>“Similar direction” is measured by cosine similarity, which you met in <a href="#/lesson/vectors">lesson 1.1</a>: the dot product with both lengths divided out. Here it is by hand, with a vocabulary of just three words: <span className="mono">[pages, acknowledge, budget]</span>.</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th /><th>vector</th><th>dot with question</th><th>length</th><th>cosine</th></tr></thead>
            <tbody>
              <tr><td>question</td><td className="mono">[1, 1, 0]</td><td /><td className="mono">√2 = 1.41</td><td /></tr>
              <tr><td>chunk about paging</td><td className="mono">[2, 1, 0]</td><td className="mono">1·2 + 1·1 + 0·0 = 3</td><td className="mono">√5 = 2.24</td><td className="mono">3 / (1.41 × 2.24) = <b>0.95</b></td></tr>
              <tr><td>chunk about expenses</td><td className="mono">[0, 1, 2]</td><td className="mono">1·0 + 1·1 + 0·2 = 1</td><td className="mono">√5 = 2.24</td><td className="mono">1 / (1.41 × 2.24) = <b>0.32</b></td></tr>
            </tbody>
          </table>
        </div>
        <p>The paging chunk wins, 0.95 to 0.32. That is all “retrieval” is: this calculation, once per stored chunk, then sort.</p>
        <p>Where do the numbers in the vectors come from? In our crude embedder, each word’s weight is how <em>rare</em> it is. With 19 chunks:</p>
        <div className="table-scroll">
          <table className="plain mono" style={{ fontSize: 14 }}>
            <thead><tr><th>word</th><th>appears in … of 19 chunks</th><th>weight = ln(19 / count)</th></tr></thead>
            <tbody>
              <tr><td>the</td><td>14</td><td>0.31</td></tr>
              <tr><td>oncall</td><td>4</td><td>1.56</td></tr>
              <tr><td>pages</td><td>2</td><td>2.25</td></tr>
              <tr><td>laptop</td><td>1</td><td>2.94</td></tr>
            </tbody>
          </table>
        </div>
        <p>A word that appears almost everywhere tells you almost nothing about which chunk you want, so it gets almost no weight. This scheme has a name, TF-IDF (term frequency × inverse document frequency). The rarity weight is the IDF part, which dates from the 1970s (Spärck Jones, 1972). Counting how often the word occurs in the chunk is the TF part.</p>
      </Numbers>

      <TheMath>
        <p>The whole of query time fits on one line. Nothing in it touches the model’s <G t="parameters">parameters</G>.</p>
        <Equation
          label="Answer equals LLM of the prompt built from the question and the top k chunks"
          symbols={[
            ['q', 'the user’s question, as text'],
            ['embed( )', 'the embedding model: text in, one vector out. The SAME model must be used for chunks and for questions'],
            [<>c<sub>i</sub></>, 'a stored chunk of text'],
            ['cos( , )', 'cosine similarity: dot product divided by both lengths. 1 = same direction, 0 = unrelated, −1 = opposite (our word-count vectors never go below 0; learned embeddings can)'],
            [<>top<sub>k</sub></>, 'keep the k chunks with the highest similarity. It always returns k chunks, even if all of them are bad'],
            ['prompt( , )', 'plain string formatting: instruction + chunks + question'],
            ['LLM( )', 'the unchanged model, doing ordinary next-token prediction on that string'],
          ]}
        >
          answer = LLM( prompt( q, top<sub>k</sub> of the c<sub>i</sub> ranked by cos( embed(q), embed(c<sub>i</sub>) ) ) )
        </Equation>
        <DeepDive title="Why cosine and not plain distance?">
          <p>A long chunk repeats words, so its raw vector is longer than a short chunk’s, without being more relevant. Dividing by the lengths removes that. In practice vectors are usually scaled to length 1 when they are stored. After that, cosine is the dot product, and search is one matrix multiply. For length-1 vectors, ranking by straight-line distance gives the same order as ranking by cosine, so the choice stops mattering.</p>
        </DeepDive>
      </TheMath>

      <CodeIt>
        <p>The pipeline from the playground, in the order the data flows. First, cutting text into overlapping windows of sentences:</p>
        <Code source="phase4-modern-llms/mini_rag.py" title="1. chunking">{`
def chunk(text, source, sentences_per_chunk=2, overlap=1):
    sents = [s.strip() + "." for s in text.split(".") if s.strip()]
    chunks, i = [], 0
    step = max(1, sentences_per_chunk - overlap)
    while i < len(sents):
        body = " ".join(sents[i:i + sentences_per_chunk])
        chunks.append({"text": body, "source": source, "pos": i})
        i += step
    return chunks
`}</Code>
        <p>It is string splitting. There is no elegant theory of chunking, only heuristics, and it has an outsized effect on quality.</p>
        <p>Second, the embedder. The core is four lines: add up each known word’s rarity weight, leak a little weight onto words that tend to appear alongside, scale to length 1.</p>
        <Code source="phase4-modern-llms/mini_rag.py" title="2. embedding (inside build_embedder)">{`
def embed(text):
    v = np.zeros(Vn)                        # one slot per vocabulary word
    for w in tokenize(text):
        if w in stoi:                       # unknown words are silently skipped
            v[stoi[w]] += idf[stoi[w]]      # rare words count more
    v = v + smooth * (v @ C)                # bleed onto co-occurring words
    return v / (np.linalg.norm(v) + 1e-9)   # length 1: dot product == cosine
`}</Code>
        <Callout kind="warn">This embedder is deliberately crude. It matches <em>strings</em>, not meanings. A real system replaces this one function with a Transformer encoder. Every other line of the file stays the same, which is why it is worth studying.</Callout>
        <p>Third, the store. Exact search is the matrix multiply you know:</p>
        <Code source="phase4-modern-llms/mini_rag.py" title="3. vector store">{`
def search(self, qvec, k=3):
    sims = np.array(self.vecs) @ qvec        # every similarity at once
    order = np.argsort(-sims)[:k]            # best k
    return [(self.chunks[i], float(sims[i])) for i in order]
`}</Code>
        <p>Fourth, the step that gives RAG its name. It is an f-string:</p>
        <Code source="phase4-modern-llms/mini_rag.py" title="4. prompt assembly">{`
def assemble_prompt(question, retrieved):
    ctx = "\\n".join(f"[{i+1}] ({c['source']}) {c['text']}"
                    for i, (c, _) in enumerate(retrieved))
    return (
        "Answer using ONLY the context below. If the answer is not in the\\n"
        "context, say 'Not found in the provided context.'\\n\\n"
        f"Context:\\n{ctx}\\n\\nQuestion: {question}\\nAnswer:"
    )
`}</Code>
        <p>Finally the “LLM”. To run offline, the file uses a stand-in that can only quote: it returns the retrieved sentence most similar to the question, or refuses if the best similarity is under 0.35.</p>
        <Code source="phase4-modern-llms/mini_rag.py" title="5. the extractive stand-in (shortened)">{`
def extractive_answer(question, retrieved, embed, threshold=0.35):
    qv = embed(question)
    best, best_sim, best_cite = None, -1, None
    for i, (c, _) in enumerate(retrieved):
        for sent in c["text"].split("."):
            sim = float(embed(sent) @ qv)
            if sim > best_sim:
                best, best_sim, best_cite = sent.strip(), sim, (i + 1, c["source"])
    if best_sim < threshold:
        return f"Not found in the provided context. (best match {best_sim:.2f})"
    return f"{best}. [source {best_cite[0]}: {best_cite[1]}] (sim {best_sim:.2f})"
`}</Code>
        <p>And the IVF index from the lab is two steps: rank the cluster centres, then brute-force only inside the chosen buckets.</p>
        <Code source="phase4-modern-llms/vector_db.py" title="IVF search">{`
def search_ivf(self, qvec, k=5, nprobe=1):
    q = qvec / (np.linalg.norm(qvec) + 1e-9)
    c_sims = self.centroids @ q                  # 1: which clusters is q nearest to?
    probe = np.argsort(-c_sims)[:nprobe]
    cand = np.concatenate([self.buckets[j] for j in probe])
    sims = self.vecs[cand] @ q                   # 2: exact search, but only in there
    order = np.argsort(-sims)[:k]
    return [(self.ids[cand[i]], float(sims[i])) for i in order]
`}</Code>
      </CodeIt>

      <BreakIt>
        <p>Use the “Break it” buttons in the playground. For each one, predict which stage fails <em>before</em> you look.</p>
        <ul>
          <li><b>Different words, same meaning.</b> The right chunk scrapes in at 0.33. Rephrase the question with words from the document (“are manual deploys forbidden”) and watch the top score jump to 0.57. Retrieval quality is capped by the embedder.</li>
          <li><b>The answer is not there</b>, then set sentences per chunk to 1. The refusal turns into a confident, cited, <em>wrong</em> answer about the onboarding buddy (similarity 0.38, just over the threshold). Nothing about the question changed. Only the chunking did. A citation proves where a sentence came from, not that it answers the question.</li>
          <li><b>Chunking splits the answer.</b> Read the prompt in stage 5. The deadline is not in it at all. No model, however large, can answer from text it was not given.</li>
          <li><b>k too small</b>, then push k to 6. Now the prompt is mostly irrelevant text. With a real model, more context is not free: it costs tokens, and relevant passages buried in the middle of a long context are used less reliably.</li>
          <li><b>Fix a gap without training.</b> In stage 1 press “Add a document” (it mentions the wifi password), then ask the wifi question again. You updated what the system knows in one second. No weight changed.</li>
        </ul>
        <Callout kind="established">
          RAG does not stop <G t="hallucination">hallucination</G>. It changes the odds. If the right passage is in the prompt, the model is doing reading comprehension, which it is good at. If retrieval brings back garbage, the model answers from garbage, or falls back on its weights, now with a false look of being grounded. And even with the right passage, the model can ignore or misread it. When a RAG system is bad, debug retrieval first.
        </Callout>
      </BreakIt>

      <Exercises>
        <CodeExercise id="rag-code-retrieve" />
        <Exercise
          id="rag-calc-cosine"
          type="calculate"
          title="Cosine by hand"
          answer={{ value: 0.96, tolerance: 0.006 }}
          answerLabel="cosine similarity"
          hints={['Dot product first: 3×4 + 4×3.', 'Lengths: √(3² + 4²) = 5 for both vectors.', 'cosine = 24 / (5 × 5).']}
          solution={<><p>a·b = 12 + 12 = 24. Both lengths are √25 = 5. Cosine = 24 / 25 = <b>0.96</b>.</p><p>The vectors are different, but they point in nearly the same direction, so a vector store would rank b as a very close match for a.</p></>}
        >
          <p>A question embeds to <code>a = [3, 4]</code> and a chunk to <code>b = [4, 3]</code>. What is their cosine similarity? (Two decimals.)</p>
        </Exercise>

        <Exercise
          id="rag-predict-weights"
          type="predict"
          title="What changed?"
          hints={['List the components: documents, chunks, vectors, prompt, model weights.', 'Which of those exist only at query time, and which one is sent to the model?']}
          solution={<p>Nothing inside the model changed: not one weight. The <em>vector store</em> gained a few rows at ingest time, and from then on the <em>prompt</em> for related questions contains the new text. That is why RAG updates are instant and reversible (delete the document and the knowledge is gone), and why RAG cannot teach the model a new <em>behaviour</em> such as a house style. Behaviour lives in weights. That is the next lesson.</p>}
        >
          <p>You add a new policy document to a RAG system and re-run ingest. A colleague says “great, the model has learned our new policy”. Predict: which numbers inside the LLM are different from yesterday? What actually changed, and where does it live?</p>
        </Exercise>

        <Exercise
          id="rag-debug-miss"
          type="debug"
          title="Diagnose the miss"
          hints={['Walk the stages in order: is the fact in a document? In one chunk? Was that chunk retrieved? Was it in the prompt? Did the model use it?', 'The log shows the top-3 similarities are all low and close together (0.23, 0.22, 0.22). What does a flat, low score profile tell you about the match between the question’s words and the corpus?', 'Compare the vocabulary: the user wrote “time off” and “sick”. The document says “leave of absence” and “medical”.']}
          solution={<><p>The failing stage is <b>embedding/retrieval</b>, not the LLM. The document exists and is chunked sensibly, but the question shares no vocabulary with it, so its chunk never entered the top-k. The flat, low similarity scores are the signature: nothing matched well. The model then did what it always does with a prompt that lacks the answer: it produced a plausible one from its weights.</p><p>Fixes, in order of cost: a better embedding model (one that places “sick” near “medical”); hybrid search that combines keyword and vector scores; a score threshold below which the system says “I do not have this” without calling the model; an evaluation set of (question → chunk that should be retrieved) pairs so you notice the next miss before users do.</p></>}
        >
          <p>A user asks an HR bot: “how much time off do I get when I am sick?”. The bot answers “10 days per year”, citing nothing. The real policy (“Medical leave of absence: up to 30 days”) is in <code>leave-policy.md</code>, which was ingested. The retrieval log shows:</p>
          <Code lang="text">{`
0.23  (expenses.md)   Receipts must be submitted within thirty days...
0.22  (onboarding.md) All new hires must complete security training...
0.22  (oncall.md)     Postmortems are blameless and due within three...
`}</Code>
          <p>Which stage failed: chunking, embedding/retrieval, prompt assembly, or the LLM? How can you tell from the log?</p>
        </Exercise>

        <Exercise
          id="rag-modify-chunks"
          type="modify"
          title="Change the chunk size in the real file"
          hints={['Run python phase4-modern-llms/mini_rag.py and note the similarity of the top chunk for “how quickly must I acknowledge pages” (0.47).', 'In main(), change chunk(text, src) to chunk(text, src, sentences_per_chunk=5, overlap=0): each document becomes one chunk.', 'Compare the top retrieval score again, and look at how long the printed prompt has become.']}
          solution={<p>With whole documents as chunks there are only 4 chunks, and the top retrieval score for the paging question drops from 0.47 to <b>0.28</b>: the one relevant sentence is diluted by four irrelevant ones in the same vector. The prompt also grows to three whole documents. The stand-in still answers correctly because it re-scores individual sentences, but a vector store at scale would now rank that document much lower. Too small loses context (the split-answer failure); too large blurs the vector. That tension is why chunking is a real design decision.</p>}
        >
          <p>Open <code>mini_rag.py</code>. Predict what happens to the top similarity score for the first demo query if every document becomes a single chunk. Then change <code>sentences_per_chunk</code> and check.</p>
        </Exercise>

        <ExplainBack
          id="rag-explain"
          prompt="Your product manager asks: “If we add RAG, will the chatbot stop making things up?” Answer in three or four sentences, without jargon."
          modelAnswer={<p>It will make things up less often, not never. RAG does not change the model. It searches our documents for passages related to the question and puts them into the prompt, so the model can read the answer instead of recalling it, and we can show which document the answer came from. But if the search brings back the wrong passage, or nothing relevant, the model will still produce a fluent answer, and it may now look trustworthy because it has a citation next to it. So the quality of the search decides the quality of the answers, and we should measure and monitor the search, and let the system say “not found”.</p>}
        />
      </Exercises>

      <CheckYourself
        questions={[
          {
            q: 'What does RAG change about the LLM?',
            options: ['Its weights, slightly, each time a document is added', 'Nothing. It changes the text of the prompt the model receives', 'Its tokenizer, so that company terms become single tokens', 'Its attention mechanism, so that it can read a database'],
            answer: 1,
            explain: 'The model is untouched. Retrieval and prompt assembly are ordinary code that runs before the model call.',
          },
          {
            q: 'Why must the question be embedded with the same embedding model as the chunks?',
            options: ['Otherwise the vectors have different lengths and the code crashes', 'Because similarity only means something between vectors from the same space; two models place the same text in unrelated positions', 'Because embedding models are licensed per corpus', 'It is only a performance optimisation'],
            answer: 1,
            explain: 'It is like comparing hashes from two different hash functions. Often nothing crashes: you get meaningless neighbours.',
          },
          {
            q: 'Top-k retrieval is asked a question that no document covers. What does it return?',
            options: ['An empty list', 'An error', 'The k least-bad chunks, with low similarity scores', 'The chunks most recently added'],
            answer: 2,
            explain: 'Top-k always returns k items. Noticing that they are all poor matches (a score threshold, a refusal instruction) is something you must add.',
          },
          {
            q: 'An IVF index with nprobe = 1 is much faster than exact search. What do you pay?',
            options: ['Nothing, it is strictly better', 'More memory for the same results', 'Some true nearest neighbours are missed, because they sit in clusters that were not searched', 'The similarity scores become approximate'],
            answer: 2,
            explain: 'The scores it computes are exact. The approximation is in which vectors it never looks at. In the repo benchmark, nprobe = 1 found only 51% of the true top 10.',
          },
          {
            q: 'A RAG bot gives a wrong answer with a citation. Where should you look first?',
            options: ['The model’s temperature setting', 'The retrieval log: which chunks were put into the prompt, and with what scores', 'The size of the model', 'The system prompt wording'],
            answer: 1,
            explain: 'Most RAG failures start at retrieval: the right text never reached the prompt. Read the exact prompt before blaming the model.',
          },
        ]}
      />

      <Remember
        items={[
          <><b>What changes: the prompt. Not the weights.</b> RAG is retrieval + context injection around an unchanged model.</>,
          <>Ingest time: documents → <b>chunks</b> → <b>embeddings</b> → vector database. Query time: embed the question → <b>top-k</b> by cosine → paste into the prompt → LLM → answer with citation.</>,
          <>It moves the model’s job from <b>recall</b> (unreliable) to <b>reading comprehension</b> (strong), and makes knowledge instantly updatable and citable.</>,
          <>Exact search is one matrix multiply. Approximate indexes such as IVF search only the nearest clusters: a <b>recall vs speed</b> knob.</>,
          <>RAG does not stop hallucination. <b>Garbage retrieved, garbage answered.</b> When it fails, check chunking, embedding and k before the model.</>,
        ]}
      />

      <RealLLM>
        <Flow horizontal steps={[{ label: 'Plain LLM', sub: 'knowledge in weights only' }, { label: 'This lesson: RAG', sub: 'changes the prompt' }, { label: 'Fine-tuning', sub: 'changes the weights' }, { label: 'Agents', sub: 'changes the code around it' }]} active={1} />
        <ToyVsReal
          toy={<ul><li>4 documents, 19 chunks of 2 sentences</li><li>Bag-of-words TF-IDF embedder with 123 dimensions, one per word</li><li>Exact search over a Python list</li><li>An extractive stand-in that can only quote one sentence</li></ul>}
          real={<ul><li>Millions of documents; chunks of a few hundred tokens that respect headings</li><li>A Transformer encoder producing dense vectors, often combined with keyword search (“hybrid”) and a slower re-ranking model over the top candidates</li><li>An approximate index (IVF, or the graph-based HNSW) with metadata filters</li><li>A real LLM that can combine passages, and can also ignore or misread them</li></ul>}
        />
        <Callout kind="established">The shape is the same in every production system: chunk, embed, index, retrieve, assemble a prompt, generate. “Search the web” and “chat with your PDF” features are this pipeline. Long context windows reduce how much you must retrieve, but do not remove the need: someone still has to decide what goes in the prompt, and you pay for every token.</Callout>
        <Callout kind="research">How faithfully models use retrieved context is an active research area. Measured effects include weaker use of information in the middle of long contexts, and models preferring what is in their weights when it conflicts with the context. Treat “grounded” as a property you test for, not one you get by construction.</Callout>
      </RealLLM>
    </Lesson>
  )
}
