import type { ReactNode } from 'react'
import { Lesson, Why, Remember, RealLLM } from '../components/lesson'
import { Callout, DeepDive, Flow, G, ToyVsReal } from '../components/ui'
import { Code } from '../components/Code'
import { Exercise, ExplainBack } from '../components/exercise'
import { MilestoneTracker } from '../interactive/MilestoneTracker'
import { MILESTONES } from '../lib/milestones'

/** One milestone = one section: goal, the idea, the task, the test, the checklist. */
function Milestone({ id, goal, children }: { id: string; goal: ReactNode; children: ReactNode }) {
  const n = MILESTONES.findIndex((m) => m.id === id)
  const m = MILESTONES[n]
  return (
    <section className="section" id={`milestone-${id}`} data-phase="build">
      <div className="section-head" style={{ minWidth: 0 }}><span className="section-kicker" style={{ whiteSpace: 'normal', overflowWrap: 'anywhere' }}>Milestone {n + 1} of {MILESTONES.length} · {m.files.map((f) => f.split('/').pop()).join(' + ')}</span></div>
      <h2>{n + 1}. {m.title}</h2>
      <p className="lede"><b>Goal:</b> {goal}</p>
      {children}
      <MilestoneTracker only={id} />
    </section>
  )
}

export default function CapstoneLesson() {
  return (
    <Lesson id="capstone">
      <Why title="Can you wire every piece together yourself?">
        <p className="lede">Launch day. At 10 a.m. the Paisa Pal support bot goes live for every customer. By 10:20 the team WhatsApp group is all green ticks and one cake photo from Dev, who is not even on the team.</p>
        <p>The bot answers from the policy PDFs, checks transaction status with a tool, stays inside its budgets, and passed 200 real tickets in the eval. Riya knows each of those pieces from a lesson.</p>
        <p>That evening she stays back anyway. Kabir stops at her desk with his bag on his shoulder. “Happy?” “Almost,” she says. “I want to wire every piece together myself. Small. Once.” He smiles. “Don’t memorise it. Build it.”</p>
        <p>So far, every program in the repository ran alone. The tokenizer never met the GPT. The GPT never met the retriever. The agent used a scripted stand-in instead of a model.</p>
        <p>In this capstone you connect them:</p>
        <ul>
          <li>your tokenizer feeds your Transformer;</li>
          <li>you train it, then improve how it generates;</li>
          <li>you put retrieved text into its prompt;</li>
          <li>finally you hand it to an agent loop as a tool.</li>
        </ul>
        <Flow horizontal steps={MILESTONES.map((m) => ({ label: m.title }))} />
        <Callout kind="warn" label="Honest expectations">
          The result will be a model with under a million parameters, trained for minutes on one small text file. It will write Shakespeare-flavoured gibberish. It will <b>not</b> answer questions, even with perfect retrieved context in its prompt, because it was never trained to.
          <br /><br />
          That is fine. The point is not the output. The point is that <b>every piece of the system is yours</b>, and that you will know exactly which piece a real LLM makes bigger.
        </Callout>
        <p>Plan for about four hours, spread over several sittings. Riya’s took three evenings and a Sunday morning.</p>
        <p>Each milestone has the same five parts: the goal, the idea in a few sentences, the task, a small test you write, and a validation checklist.</p>
      </Why>

      <section className="section" id="setup" data-phase="build">
        <div className="section-head"><span className="section-kicker">Before you start</span></div>
        <h2>Setup, and how the tests work</h2>
        <p>Make a folder <code>capstone/</code> at the repository root for your own files. Run everything from the repository root. The phase folders have hyphens in their names, so they cannot be imported as packages; add them to the path instead:</p>
        <Code
          title="capstone/paths.py"
          show={`print("Python now searches these folders first, in this order:")
for folder in sys.path[:5]:
    print("  ", folder)`}
        >{`
import sys
for folder in ("phase2-language", "phase3-transformers",
               "phase4-modern-llms", "phase5-agents", "capstone"):
    sys.path.insert(0, folder)
`}</Code>
        <p>The repository already has a <code>tests/</code> folder with <G t="softmax">softmax</G>, tokenizer, attention, KV cache, RAG and agent tests. Run them now:</p>
        <Code>{`
pip install numpy pytest torch
pytest                 # everything in tests/test_*.py
pytest -k kv_cache     # only tests whose name contains "kv_cache"
`}</Code>
        <p>Read a few of them. <code>tests/conftest.py</code> does the same path trick as above. For this capstone, add <code>"capstone"</code> to the folder list in <code>tests/conftest.py</code>, and put your own tests in a new file <code>tests/test_capstone.py</code>. Each milestone below asks you to add one or two tests to it.</p>
        <Callout kind="dev">
          Why tests, in a course about understanding? Because in numerical code almost nothing crashes. A wrong axis, a missing mask or a stale cache all produce numbers of the right shape. A three-line assertion about a property that <em>must</em> hold (rows sum to 1, the round trip is lossless, the cache changes nothing) is how ML engineers catch silent bugs.
        </Callout>
        <MilestoneTracker />
      </section>

      <Milestone id="tokenizer" goal="a BPE tokenizer trained on text you chose, that never crashes and never loses a character.">
        <p>The model works on integers, so something has to turn text into integers and back. <G t="bpe">BPE</G> starts from single characters and repeatedly merges the most frequent adjacent pair into a new token, so common words become one token and rare words stay as pieces. If you want a refresher, revisit <a href="#/lesson/tokenization">Tokenization</a>.</p>
        <p>Until now <code>tiny_gpt.py</code> cheated: it used one token per character. Your tokenizer will replace that.</p>
        <Exercise
          id="capstone-m1-tokenizer"
          type="implement"
          title="Train it on your own text, and make it robust"
          hints={[
            'Any plain text of 200 KB or more works: shakespeare.txt (tiny_gpt.py downloads it into whichever folder you ran it from; copy it to the repository root), your own notes, a book from Project Gutenberg. Train on a slice such as text[:200_000] with vocab_size=400. Pure-Python BPE is slow on a whole megabyte.',
            'Now call tok.encode("Zebra 42!"). If any of those characters was not in the training slice you get a KeyError, because the base vocabulary is built from the training text only. Look at the first line of train().',
            'Smallest fix: give train() an extra argument alphabet="" and build the base vocabulary from set(text) | set(alphabet). Pass alphabet=full_text + string.printable. Every character now has an id, and merges still come only from the training slice.',
          ]}
          solution={<>
            <Code title="the one-line change in BPETokenizer.train">{`
def train(self, text, vocab_size, verbose=True, alphabet=""):
    # base vocabulary: every character we may EVER need gets an id
    chars = sorted(set(text) | set(alphabet))
`}</Code>
            <Code title="capstone/m1_tokenizer.py">{`
import paths, pickle, string
from bpe_tokenizer import BPETokenizer

text = open("shakespeare.txt").read()
tok = BPETokenizer()
tok.train(text[:200_000], vocab_size=400, verbose=False,
          alphabet=text + string.printable)
ids = tok.encode(text[:50_000])
print(50_000 / len(ids), "characters per token")
pickle.dump(tok, open("capstone/tok.pkl", "wb"))
`}</Code>
            <p>Why this and not an <code>&lt;unk&gt;</code> token? Because an unknown-token would make <code>decode(encode(s))</code> lose information. Real tokenizers solve the same problem by starting from the 256 possible bytes instead of characters, so nothing can ever be unseen. The comment inside <code>encode()</code> points at that byte fallback as a follow-up exercise if you want to go further.</p>
          </>}
        >
          <p>Train <code>BPETokenizer</code> on a text file of your choice. Then break it: encode a string containing characters your training text never had. Fix the crash without giving up the lossless round trip. Save the trained tokenizer to disk.</p>
        </Exercise>
        <p><b>The test you write.</b> The defining property of a tokenizer is that it is lossless:</p>
        <Code title="tests/test_capstone.py">{`
import pickle
tok = pickle.load(open("capstone/tok.pkl", "rb"))

def test_round_trip_is_lossless():
    for s in ["To be, or not to be", "What is a cat?", "Zebra 42!"]:
        assert tok.decode(tok.encode(s)) == s

def test_it_actually_compresses():
    s = "the king and the queen"
    ids = tok.encode(s)
    assert len(ids) < len(s) and max(ids) < len(tok.vocab)
`}</Code>
      </Milestone>

      <Milestone id="transformer" goal="your GPT reads your BPE token ids, and you have proved two properties of its attention.">
        <p>A token id is only a row number. The <G t="embedding">embedding</G> table turns it into a vector, a second table adds <G t="positional">position</G>, and a stack of Transformer blocks (attention to move information between tokens, an MLP to process it) turns those vectors into one score per vocabulary entry. You built each part in <a href="#/lesson/attention">Attention</a>, <a href="#/lesson/transformer-block">The Transformer block</a> and <a href="#/lesson/build-gpt">Build GPT</a>.</p>
        <p>The model does not care what a token <em>is</em>. Swapping characters for BPE tokens changes exactly one number in it: <code>vocab_size</code>.</p>
        <Exercise
          id="capstone-m2-plug"
          type="modify"
          title="Replace characters with your tokens"
          hints={[
            'In main() of tiny_gpt.py, four lines build the character vocabulary: chars, stoi, itos and data. Those are the only lines that know about characters, apart from the decode inside sample().',
            'data = torch.tensor(tok.encode(text)) and cfg.vocab_size = len(tok.vocab). Use len(tok.vocab), not 400: training stops early if no pair occurs twice.',
            'Encoding a full megabyte takes a minute or so in pure Python. Do it once and cache it: torch.save(data, "capstone/data.pt").',
          ]}
          solution={<>
            <Code title="in tiny_gpt.py main(), replacing the chars / stoi / itos lines">{`
text = load_text()
tok = pickle.load(open("capstone/tok.pkl", "rb"))
data = torch.tensor(tok.encode(text), dtype=torch.long)   # cache me
...
cfg.vocab_size = len(tok.vocab)
...
def sample(n_tokens=200):
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(start, n_tokens)[0].tolist()
    return tok.decode(out)
`}</Code>
            <p>Nothing inside <code>GPT</code>, <code>Block</code> or <code>CausalSelfAttention</code> changes. The embedding table has about 400 rows instead of about 65, and so does the output layer. Add <code>import paths, pickle</code> at the top so the pickled tokenizer class can be found. Notice what you gained: if your tokenizer averages two characters per token, the same 64-token context window now covers twice as much text.</p>
          </>}
        >
          <p>Copy <code>tiny_gpt.py</code> to <code>capstone/my_gpt.py</code> and change it so that it trains on the ids from your milestone 1 tokenizer. Do not train yet. Just run a forward pass on one batch and print the shape of the logits.</p>
        </Exercise>
        <p><b>The tests you write.</b> “Attention rows sum to 1” already exists in <code>tests/test_transformers.py</code> for the NumPy version: read it. Add the property that matters most for a language model, on your PyTorch GPT: no token may see the future.</p>
        <Code title="tests/test_capstone.py">{`
import torch
from my_gpt import GPT, Config

def small_model():
    cfg = Config(); cfg.vocab_size = 50; cfg.dropout = 0.0
    return GPT(cfg).eval()

def test_logits_shape_and_causality():
    model = small_model()
    a = torch.randint(50, (1, 10))
    b = a.clone(); b[0, -1] = (b[0, -1] + 1) % 50    # change ONLY the last token
    la, _ = model(a); lb, _ = model(b)
    assert la.shape == (1, 10, 50)
    assert torch.allclose(la[0, :-1], lb[0, :-1], atol=1e-5)
`}</Code>
        <p>If that second assertion fails, the causal mask is broken, and the model can cheat during training by reading the answer.</p>
      </Milestone>

      <Milestone id="training" goal="a trained model whose validation loss fell, and a saved weights file.">
        <p>If you have watched a tiny GPT train in the browser version of this course, this is the same loop, now in your own files.</p>
        <p><G t="pretraining">Training</G> is one loop: take a batch of text, ask the model for next-token probabilities at every position, measure how surprised it was by the real next token (<G t="cross-entropy">cross-entropy</G> loss), and nudge every parameter slightly downhill. A held-out validation split tells you whether it is learning patterns or memorising. See <a href="#/lesson/training-gpt">Training GPT</a>.</p>
        <Exercise
          id="capstone-m3-train"
          type="experiment"
          title="Train it, and read the loss honestly"
          hints={[
            'Start with python capstone/my_gpt.py --quick to check that nothing crashes, then do the full run. It prints train and validation loss and a sample every 500 steps.',
            'Your loss is per token, and your tokens are longer than one character, so it is not comparable to the character-level run. Divide by your characters-per-token figure from milestone 1 to get loss per character.',
            'A reference point: guessing uniformly among 400 tokens costs ln(400) ≈ 5.99 per token. Anything your model achieves below that is knowledge it extracted from the text.',
          ]}
          solution={<>
            <p>You should see validation loss fall quickly at first and then flatten, while the samples go from random token soup to word-like fragments to lines that <em>look</em> like a play: speaker names, line breaks, archaic words. Even early samples contain whole words, because common words are single tokens and no longer need to be spelled. That can make an undertrained model look better than it is, so trust the loss, not the look.</p>
            <p>If train loss keeps falling while validation loss turns upward, the model has started to memorise: with a 1 MB corpus and a few thousand steps this is mild, but raise the step count and you will see it. Save the result:</p>
            <Code>{`
torch.save(model.state_dict(), "capstone/gpt.pt")
# later:  model = GPT(cfg); model.load_state_dict(torch.load("capstone/gpt.pt")); model.eval()
`}</Code>
          </>}
        >
          <p>Train your BPE-token GPT. Record train and validation loss at the first and last evaluation. Compare the final loss <em>per character</em> with the original character-level model. Save the weights.</p>
        </Exercise>
        <p><b>The test you write.</b> A full training run is too slow for a unit test. The standard trick is to check that the model <em>can</em> learn: give it one tiny batch and let it memorise it.</p>
        <Code title="tests/test_capstone.py">{`
def test_can_overfit_one_batch():
    torch.manual_seed(0)
    cfg = Config(); cfg.vocab_size = 50; cfg.dropout = 0.0
    model = GPT(cfg)
    x = torch.randint(50, (4, 16)); y = torch.roll(x, -1, dims=1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(200):
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]
`}</Code>
        <p>If a model cannot even memorise 64 tokens, something in the forward pass, the loss or the optimiser wiring is broken, and no amount of data will fix it.</p>
      </Milestone>

      <Milestone id="inference" goal="generation with top-k and top-p, and a tested understanding of the KV cache.">
        <p>The network outputs probabilities; choosing a token from them is ordinary code outside the network. <G t="temperature">Temperature</G> sharpens or flattens the distribution, <G t="top-k">top-k</G> keeps the k most likely tokens, and top-p keeps the smallest set whose probabilities reach p. Separately, the <G t="kv-cache">KV cache</G> stores each past token’s key and value vectors so a new token does not recompute them. See <a href="#/lesson/inference">Inference</a>.</p>
        <Exercise
          id="capstone-m4-sampling"
          type="implement"
          title="Add top-k and top-p to GPT.generate"
          hints={[
            'Look at sample_demo() in kv_cache_demo.py: it does both in NumPy on one distribution. You need the same logic on a (B, vocab) torch tensor, between the softmax and torch.multinomial.',
            'top-k: torch.topk(probs, k).values[:, [-1]] is the k-th largest probability in each row. Zero everything smaller, then divide by the row sum.',
            'top-p: sort descending, take the cumulative sum, and keep a token if the mass ranked above it is still below p. torch.sort returns the indices you need to scatter the kept probabilities back into vocabulary order.',
          ]}
          solution={<>
            <Code title="inside generate(), after probs = F.softmax(...)">{`
if top_k is not None:
    kth = torch.topk(probs, top_k).values[:, [-1]]
    probs = probs.masked_fill(probs < kth, 0.0)
if top_p is not None:
    sp, si = torch.sort(probs, descending=True)
    above = sp.cumsum(-1) - sp            # mass ranked above each token
    sp = sp.masked_fill(above >= top_p, 0.0)
    probs = torch.zeros_like(probs).scatter(-1, si, sp)
probs = probs / probs.sum(-1, keepdim=True)   # renormalise
`}</Code>
            <p>Add <code>top_k=None, top_p=None</code> to the signature. The top-ranked token always has zero mass above it, so at least one token always survives. With <code>top_k=1</code> only one token is left, <code>multinomial</code> has no choice, and generation becomes greedy and repeatable. At temperature 0.8 with <code>top_k=20</code> your samples should lose most of their junk tokens.</p>
          </>}
        >
          <p>Extend <code>GPT.generate</code> in your <code>my_gpt.py</code> with optional <code>top_k</code> and <code>top_p</code>. Then generate 200 tokens at temperatures 0.3, 0.8 and 1.5 and describe what changes.</p>
        </Exercise>
        <p><b>The tests you write.</b> One for your sampler, one for the cache. The cache test is the most important test in all of inference: an optimisation must not change the answer.</p>
        <Code title="tests/test_capstone.py">{`
import kv_cache_demo as kv

def test_top_k_1_is_repeatable():
    model = small_model()                      # .eval(): dropout off
    start = torch.zeros((1, 1), dtype=torch.long)
    assert torch.equal(model.generate(start, 20, top_k=1),
                       model.generate(start, 20, top_k=1))

def test_cached_equals_naive():
    p = kv.make_params()
    assert kv.generate_naive(p, [1, 7, 3], 30) == kv.generate_cached(p, [1, 7, 3], 30)
`}</Code>
        <DeepDive title="Stretch goal: a KV cache inside your PyTorch GPT">
          <p><code>kv_cache_demo.py</code> caches in its own small NumPy model. Porting the idea to <code>my_gpt.py</code> is a good afternoon: <code>CausalSelfAttention.forward</code> takes an optional cache, computes q, k, v for the new token only, appends k and v to the cache, and attends over the whole cache. The position embedding must use the token’s true position, not 0. Then write the same equivalence test against your uncached <code>generate</code> with <code>top_k=1</code>.</p>
        </DeepDive>
      </Milestone>

      <Milestone id="rag" goal="text retrieved from your own documents ends up in your model’s prompt, with no weight changed.">
        <p>A model only “knows” what training pressed into its parameters. <G t="rag">RAG</G> works around that without touching the model: cut your documents into chunks, turn each into a vector, find the chunks nearest to the question, and paste them into the prompt. The model then reads them as part of the text so far. See <a href="#/lesson/rag">Retrieval-augmented generation</a>.</p>
        <Exercise
          id="capstone-m5-rag"
          type="implement"
          title="Retrieve, assemble, generate"
          hints={[
            'Replace CORPUS in a copy of mini_rag.py with three or four short documents of your own. main() shows the wiring: chunk, build_embedder, Store.add, Store.search, assemble_prompt.',
            'assemble_prompt returns a string. Your model needs token ids: torch.tensor([tok.encode(prompt)]). If this raises KeyError, your milestone 1 alphabet fix is missing a character.',
            'The prompt is far longer than 64 tokens. Look at the first line inside the generate loop: what does idx[:, -context_len:] do to it?',
          ]}
          solution={<>
            <Code title="capstone/m5_rag.py: load what you built in milestones 1 to 4">{`
import paths, pickle, torch
from mini_rag import chunk, build_embedder, Store, assemble_prompt, extractive_answer
from my_gpt import GPT, Config

tok = pickle.load(open("capstone/tok.pkl", "rb"))
cfg = Config(); cfg.vocab_size = len(tok.vocab)
model = GPT(cfg); model.load_state_dict(torch.load("capstone/gpt.pt")); model.eval()
`}</Code>
            <Code title="…then, after building embed and store from YOUR documents as mini_rag.main() does">{`
q = "who approves conference travel"          # ask about your own documents
retrieved = store.search(embed(q), k=2)
prompt = assemble_prompt(q, retrieved)

ids = torch.tensor([tok.encode(prompt)])
out = model.generate(ids, 60, temperature=0.8, top_k=20)
print(prompt)
print("MY GPT :", tok.decode(out[0, ids.shape[1]:].tolist()))
print("BASELINE:", extractive_answer(q, retrieved, embed))
`}</Code>
            <p>Retrieval works: the right chunk is in the prompt. Your GPT’s “answer” is still Shakespeare-flavoured noise, for two honest reasons. First, its context window is 64 tokens, so <code>generate</code> crops the prompt and the model sees only the tail. Second, and more important, it was only ever trained to continue Shakespeare. Nothing taught it that text after “Answer:” should use the text after “Context:”.</p>
            <p>That second ability is what large-scale pretraining and instruction tuning buy. The plumbing you wrote is exactly what production RAG does; swap your model for a real one (the comment at the bottom of <code>mini_rag.py</code> shows the three lines) and the same prompt produces a grounded answer. The extractive baseline shows what “grounded” looks like in the meantime.</p>
          </>}
        >
          <p>Index documents you wrote yourself with <code>mini_rag.py</code>. For one question, retrieve the top chunks, assemble the prompt, and feed it to your trained GPT from milestone 3. Print the full prompt, your model’s continuation, and the extractive baseline’s answer. Explain the difference.</p>
        </Exercise>
        <p><b>The tests you write.</b> Test the two claims RAG rests on: the retrieved text really reaches the prompt, and the model itself is untouched.</p>
        <Code title="tests/test_capstone.py">{`
from mini_rag import assemble_prompt
# embed, store = ...   build them from YOUR documents, as _pipeline() does in tests/test_systems.py
def test_retrieved_context_reaches_the_prompt():
    q = "who approves conference travel"
    retrieved = store.search(embed(q), k=2)
    prompt = assemble_prompt(q, retrieved)
    assert retrieved[0][0]["text"] in prompt and q in prompt

def test_rag_changes_no_weights():
    model = small_model()
    before = [p.clone() for p in model.parameters()]
    model.generate(torch.randint(50, (1, 30)), 10)
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
`}</Code>
      </Milestone>

      <Milestone id="tools" goal="an agent loop that calls your GPT as a tool.">
        <p>A model can only write text. An <G t="agent">agent</G> is a loop in <em>your</em> code around it: the model writes a request in an agreed format, your code parses it, runs a real function, appends the result to the context, and calls the model again. The model never executes anything. See <a href="#/lesson/agents">Agents</a>.</p>
        <p><code>mini_agent.py</code> uses a scripted stand-in as its “brain” so the loop can be studied offline. Your tiny GPT cannot follow the TOOL/ARGS format either, so it cannot be the brain. But it can be a <em>tool</em>: a verse generator that the agent calls.</p>
        <Exercise
          id="capstone-m6-tool"
          type="implement"
          title="Register your model as a tool"
          hints={[
            'A tool is a plain function plus a description: look at the TOOLS dict. Write write_verse(opening: str) -> str that encodes the opening, calls model.generate and decodes the result.',
            'run_agent takes model= as an argument. Write your own small scripted model: if there is no “RESULT:” in the context yet, ask for the tool; otherwise answer with the result.',
            'Gotcha: SYSTEM_PROMPT is a string built once, at import, from TOOLS. A tool registered later is callable but is not advertised in the prompt. The scripted model does not care. A real LLM would never learn the tool exists. Register the tool inside your copy of the file above SYSTEM_PROMPT, or rebuild the string.',
          ]}
          solution={<>
            <Code title="capstone/m6_agent.py">{`
import paths, pickle, torch
import mini_agent as agent

def write_verse(opening: str) -> str:
    ids = torch.tensor([tok.encode(opening)])
    out = model.generate(ids, 40, temperature=0.8, top_k=20)
    return tok.decode(out[0].tolist())

agent.TOOLS["write_verse"] = {"fn": write_verse,
    "desc": 'continues a line of verse. args: {"opening": str}'}

def verse_model(context):
    if "RESULT:" not in context:
        return ('Thought: I should use the verse tool.\\n'
                'TOOL: write_verse\\nARGS: {"opening": "O for a"}')
    return "Thought: done.\\nANSWER: " + context.split("RESULT:")[-1].strip()

if __name__ == "__main__":        # so a test can import verse_model without loading a model
    from my_gpt import GPT, Config
    tok = pickle.load(open("capstone/tok.pkl", "rb"))
    cfg = Config(); cfg.vocab_size = len(tok.vocab)
    model = GPT(cfg); model.load_state_dict(torch.load("capstone/gpt.pt")); model.eval()
    print(agent.run_agent("Write me a verse", model=verse_model))
`}</Code>
            <p>Read the printed trace. Step 1: the “brain” emits text containing <code>TOOL: write_verse</code>. Then <code>run_agent</code>, not the model, looks the name up in <code>TOOLS</code> and calls your function. The result is appended to the scratchpad as <code>RESULT: …</code>. Step 2: the brain sees that result in its context and answers. The line <code>result = TOOLS[name]["fn"](**args)</code> is the only place where anything is ever <em>done</em>.</p>
          </>}
        >
          <p>Wrap your trained GPT’s generation as a function, register it in <code>mini_agent.TOOLS</code>, and drive <code>run_agent</code> with a small scripted model of your own that calls it. Then find the line in <code>run_agent</code> where the tool actually runs.</p>
        </Exercise>
        <p><b>The tests you write.</b> Test the loop with a fake tool, so the test needs no trained model and runs in milliseconds. Also test the unhappy path: a loop that touches the real world must not crash on bad model output.</p>
        <Code title="tests/test_capstone.py">{`
import mini_agent as agent
from m6_agent import verse_model

def test_agent_calls_my_tool():
    calls = []
    agent.TOOLS["write_verse"] = {"desc": "fake",
        "fn": lambda opening: calls.append(opening) or "a verse"}
    out = agent.run_agent("Write me a verse", model=verse_model, verbose=False)
    assert calls == ["O for a"] and "a verse" in out

def test_unknown_tool_does_not_crash():
    bad = lambda ctx: "ANSWER: gave up" if "RESULT:" in ctx else "TOOL: nope\\nARGS: {}"
    assert agent.run_agent("x", model=bad, verbose=False) == "gave up"
`}</Code>
        <p className="muted" style={{ fontSize: 15 }}>The first test can import <code>verse_model</code> without loading any weights because <code>m6_agent.py</code> loads the model inside its <code>if __name__ == "__main__":</code> guard. The fake tool replaces <code>write_verse</code>, so the real model is never needed.</p>
      </Milestone>

      <section className="section" id="explain" data-phase="practice">
        <div className="section-head"><span className="section-kicker">Wrap up</span></div>
        <h2>Explain what you built</h2>
        <ExplainBack
          id="capstone-explain"
          prompt="Describe your finished system to another developer in one paragraph. For each of the six pieces, say what goes in and what comes out. Then say which single piece you would have to scale up the most to get a useful assistant, and why."
          modelAnswer={<p>The tokenizer turns a string into a list of integer ids and back, without loss. The GPT takes ids, looks up an embedding for each, adds position, runs them through Transformer blocks and outputs a score for every vocabulary entry at every position. Training adjusted its parameters by gradient descent so that the real next token became more probable. Inference turns the last position’s scores into probabilities, trims them with temperature, top-k and top-p, samples one id, appends it and repeats, with a KV cache to avoid recomputing the past. RAG retrieves chunks of my documents by vector similarity and places them in the prompt; no weights change. The agent is a loop in my code that parses tool requests out of model text, runs the function and feeds the result back. The piece to scale is the model and its training: parameters, data and compute, plus instruction tuning. Every other piece is already structurally the same as in production.</p>}
        />
      </section>

      <Remember
        items={[
          <>The whole stack is <b>six small programs</b> with narrow interfaces: strings ↔ ids, ids → scores, scores → one id, question → prompt, text → action.</>,
          <>The model never sees text and never acts. <b>Tokenizers, samplers, retrievers and agent loops are ordinary code around it.</b></>,
          <>RAG changes the <b>prompt</b>. Training and fine-tuning change the <b>weights</b>. An agent adds a <b>loop</b>. Nothing else is going on.</>,
          <>Test <b>properties that must hold</b>: lossless round trip, rows sum to 1, no peeking at the future, cached equals naive, weights unchanged by retrieval.</>,
          <>Your model is tiny and writes gibberish. The <b>architecture is not the difference</b> between it and a real LLM. Scale and training data are.</>,
        ]}
      />

      <RealLLM>
        <ToyVsReal
          toy={<ul><li>BPE over characters, a few hundred merges, pure Python</li><li>Under 1M parameters, 64-token context, minutes on a CPU</li><li>TF-IDF-style embeddings, exact search over a few chunks</li><li>Regex-parsed TOOL/ARGS text, scripted brain</li></ul>}
          real={<ul><li>BPE over bytes, around 100,000 merges, optimised native code</li><li>Billions of parameters, context of 100,000+ tokens, months on GPU clusters, then instruction and preference tuning</li><li>Learned Transformer embeddings, approximate search over millions of chunks</li><li>Structured tool-call formats the model was trained to emit, with the same loop around them</li></ul>}
        />
        <Callout kind="established">
          None of the six interfaces changes at scale. A production stack still has a tokenizer in front, a sampler behind, retrieval that edits the prompt, and a loop that executes tool calls. When you next read the documentation of an LLM API or an agent framework, you should be able to map every parameter onto something you wrote here.
        </Callout>
        <p>What the launch-day bot has that your six pieces do not is mostly <em>training</em>, not new plumbing. A big model can teach a small one (<a href="#/lesson/distillation">Small models from big ones</a>). Post-training teaches it what it should refuse (<a href="#/lesson/alignment-safety">Alignment and safety</a>). An image encoder lets it read a payment screenshot (<a href="#/lesson/multimodal">Models that see and hear</a>). And none of it is fully readable from the inside yet (<a href="#/lesson/interpretability">Looking inside the model</a>).</p>
        <p>Late on Sunday, Riya’s tiny GPT, called as a tool by her own agent loop, writes four lines of nonsense verse. She reads them aloud to an empty flat and laughs. Every character of it went through code she wrote.</p>
        <p>One thing left: can you rebuild the whole picture <a href="#/lesson/from-memory">from memory</a>?</p>
      </RealLLM>
    </Lesson>
  )
}
