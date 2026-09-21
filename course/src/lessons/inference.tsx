import { Lesson, Why, Problem, MentalModel, TryIt, Numbers, TheMath, CodeIt, BreakIt, Exercises, CheckYourself, Remember, RealLLM, BeforeMovingOn } from '../components/lesson'
import { Callout, DeepDive, Equation, Flow, G, Term, ToyVsReal, WhyExists } from '../components/ui'
import { Code } from '../components/Code'
import { CodeExercise } from '../components/python'
import { Exercise, ExplainBack, OrderExercise } from '../components/exercise'
import { SamplingPlayground } from '../interactive/SamplingPlayground'
import { KvCacheLab } from '../interactive/KvCacheLab'
import { CachedStep } from '../illustrations/CachedStep'

export default function InferenceLesson() {
  return (
    <Lesson id="inference">
      <Why>
        <p className="lede">Training is over. The weights are frozen. Now you use the model, and two things are odd.</p>
        <div className="grid-2">
          <div className="card">
            <h4 style={{ fontSize: 17, marginBottom: 6 }}>Same question, different answers</h4>
            <p>Ask a chatbot “What is a cat?” twice. You get two different answers. But the model is a fixed function: same weights, same input. Where does the difference come from?</p>
          </div>
          <div className="card">
            <h4 style={{ fontSize: 17, marginBottom: 6 }}>A pause, then a stream</h4>
            <p>Paste in a long document. There is a noticeable wait, then words stream out at a steady pace. Why is the first word slow and the rest fast?</p>
          </div>
        </div>
        <p>Both have small, mechanical answers. The first is a dice roll that happens <em>after</em> the network has finished. The second is a cache. Using a trained model to produce output is called <b>inference</b>, and this lesson is about what happens around the network while it runs.</p>
        <Callout kind="idea">
          The model’s job ends at the <G t="logits">logits</G>: one score per token in the vocabulary. <b>Choosing</b> a token from those scores, and <b>not redoing work</b> while generating thousands of them, are both ordinary code that lives outside the network.
        </Callout>
      </Why>

      <Problem>
        <p>You know training well by now. Inference runs the same network in a very different way:</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th></th><th>Training</th><th>Inference</th></tr></thead>
            <tbody>
              <tr><td><b>Answers</b></td><td>Known: the next token is sitting in the text</td><td>Unknown: the model must produce it</td></tr>
              <tr><td><b>Positions</b></td><td>All T positions predicted in parallel, in one pass</td><td>One new token at a time: each one depends on the one before</td></tr>
              <tr><td><b>After the forward pass</b></td><td>Loss, then a backward pass, then a weight update</td><td>Pick a token, append it, go again. No gradients, weights never change</td></tr>
              <tr><td><b>Randomness</b></td><td>Which batch comes next</td><td>Which token gets drawn</td></tr>
              <tr><td><b>Main cost</b></td><td>Compute, and memory for every intermediate result (needed by backprop)</td><td>Memory for a cache of past work (the KV cache, explained below), and the speed of reading it</td></tr>
            </tbody>
          </table>
        </div>
        <p>That leaves two problems to solve.</p>
        <WhyExists
          problem="The network hands back about 50,000 scores. We need exactly one token."
          naive="Always take the token with the highest score (this is called greedy decoding)."
          fails="It is deterministic, so every answer to a prompt is identical, and it easily falls into loops: “the cat sat on the cat sat on the cat…”. The likeliest next word, chosen every time, does not make the best sentence."
          idea="Turn scores into probabilities and roll a weighted die. Then give the user knobs that reshape the die before the roll: temperature, top-k, top-p."
          tradeoff="Randomness brings variety, and risk: one unlucky draw is fed back in and every later token has to live with it."
        />
        <WhyExists
          problem="The generation loop from Build GPT feeds the whole sequence through the model for every new token."
          naive="Accept it. To produce token 1,001, run all 1,000 previous tokens through every layer again."
          fails="Those 1,000 tokens have not changed, and neither has anything computed from them. Work per token grows with the length, so total work grows with the square of the length."
          idea="Keep the intermediate results that later tokens need (each old token’s key and value vectors) and compute only the new token."
          tradeoff="Speed is bought with memory. The cache grows with every token, and that memory is what limits context length and how many users one GPU can serve."
        />
      </Problem>

      <MentalModel>
        <h3>Half A: choosing a token</h3>
        <p>After <a href="#/lesson/softmax">softmax</a>, the scores are a probability for every token. <G t="sampling">Sampling</G> means rolling a die with those probabilities painted on its faces. “mat” at 40% wins four rolls in ten, not every roll. That single roll is why the same prompt gives different answers.</p>
        <p>Three knobs reshape the die before it is rolled:</p>
        <Term
          name="Temperature"
          plain={<>A contrast knob. Divide every logit by T before softmax. Small T stretches the gaps between scores, so the favourite dominates. Large T shrinks the gaps, so long shots get a real chance.</>}
          example={<>Logits [2, 1, 0]. T = 1 gives 67% / 24% / 9%. T = 0.5 gives 87% / 12% / 2%. T = 2 gives 51% / 31% / 19%.</>}
          formal={<>p = softmax(logits / T). As T → 0 this becomes “always pick the largest” (greedy). T = 1 leaves the model’s own probabilities untouched.</>}
        />
        <Term
          name="Top-k and top-p"
          plain={<>Two ways to cut off the tail before rolling. <b>Top-k</b>: keep the k most likely tokens. <b>Top-p</b>: keep the smallest group of top tokens whose probabilities add up to at least p. Either way, the rest get probability 0 and the survivors are rescaled to sum to 1.</>}
          example={<>Probabilities 50%, 20%, 15%, 10%, 5%. Top-k = 2 keeps the first two. Top-p = 0.9 keeps four (50 + 20 + 15 = 85 is not yet 90; adding 10 gets there).</>}
          formal={<>Top-p is also called nucleus sampling. It adapts: when the model is sure it keeps 1 or 2 tokens, when it is unsure it keeps hundreds.</>}
        />
        <p>Why cut the tail at all? A real vocabulary has 50,000 or more tokens. Almost all of them are nonsense at any given moment, each with a tiny probability. But tens of thousands of tiny probabilities add up to a real chance of drawing <em>one</em> of them.</p>
        <p>And a bad token is not just one bad word: it is appended to the input, and every later prediction is conditioned on it. This is also why high temperatures fall apart: flattening the distribution hands that tail a large share of the die.</p>

        <h3>Half B: not repeating yourself</h3>
        <p>Recall what attention does for the newest token: its <span className="q">query</span> is compared with the <span className="k">key</span> of every earlier token, and the matching <span className="v">values</span> are blended. So to produce the next token, the model needs the keys and values of <em>all</em> earlier tokens, in every layer.</p>
        <p>Now the insight, and it comes straight from the <a href="#/lesson/masks-and-heads">causal mask</a>. Token 5 can only look at tokens 1 to 5. So nothing about token 5, in any layer, depends on tokens 6, 7, 8… When token 9 arrives, token 5’s key and value are <b>exactly what they were before</b>. Recomputing them gives the same numbers, every single step.</p>
        <Callout kind="dev">
          A pure function called again and again with the same arguments: you know this one. It is <b>memoisation</b>. Store each token’s <span className="k">k</span> and <span className="v">v</span> the first time, per layer, and read them back afterwards. That store is the <G t="kv-cache">KV cache</G>.
        </Callout>
        <p>With a cache, this is everything one new token costs, in one layer. Only the newest token goes in. It makes one <span className="q">q</span>, one <span className="k">k</span> and one <span className="v">v</span>; the k and v are appended to this layer’s cache; then q is compared with every cached k, and the weights blend the cached v.</p>
        <CachedStep />
        <p>Why is the <span className="q">query</span> not cached? Because nobody ever needs it again. A token’s query is used once, when that token looks back. Keys and values are read by every future token.</p>
        <Term
          name="Prefill and decode"
          plain={<><b>Prefill</b>: run the whole prompt through the model once, all tokens in parallel, to fill the cache and get the first new token. <b>Decode</b>: then produce tokens one at a time, each step processing a single token against the cache.</>}
          example={<>A 3,000-token document plus a question: one big prefill pass (the wait), then maybe 200 small decode steps (the stream).</>}
          formal={<>Prefill is limited by raw compute. Decode does very little arithmetic per step but must read all the weights and the whole cache from memory each time, so it is limited by memory speed. That is why “time to first token” and “tokens per second” are reported separately.</>}
        />
      </MentalModel>

      <TryIt title="Two experiments">
        <h3>A. Reshape the die, then roll it</h3>
        <p>Try the presets from left to right. With <b>Greedy</b>, press “Sample 100”: one token, a hundred times. With <b>Too hot</b>, watch the junk tokens start to win rolls, then rescue it with top-k.</p>
        <SamplingPlayground />
        <h3>B. Generate with and without a cache</h3>
        <p>Step through six tokens. Check the table each time: the recomputed keys and the cached keys are the same numbers.</p>
        <KvCacheLab />
      </TryIt>

      <Numbers>
        <h3>Temperature by hand</h3>
        <p>Three tokens, logits <span className="mono">[2, 1, 0]</span>. Divide by T, exponentiate, divide by the total.</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>T</th><th>logits ÷ T</th><th>e<sup>…</sup></th><th>total</th><th>probabilities</th></tr></thead>
            <tbody>
              <tr><td className="mono">0.5</td><td className="mono">4, 2, 0</td><td className="mono">54.60, 7.39, 1.00</td><td className="mono">62.99</td><td className="mono"><b>0.87, 0.12, 0.02</b></td></tr>
              <tr><td className="mono">1</td><td className="mono">2, 1, 0</td><td className="mono">7.39, 2.72, 1.00</td><td className="mono">11.11</td><td className="mono"><b>0.67, 0.24, 0.09</b></td></tr>
              <tr><td className="mono">2</td><td className="mono">1, 0.5, 0</td><td className="mono">2.72, 1.65, 1.00</td><td className="mono">5.37</td><td className="mono"><b>0.51, 0.31, 0.19</b></td></tr>
            </tbody>
          </table>
        </div>
        <p>The ranking never changes. Only the contrast does. The weakest token goes from a 2% chance to a 19% chance, just by turning T from 0.5 to 2.</p>
        <p>In the playground’s 11-token example the three junk tokens share 0.3% at T = 1 and 7.8% at T = 3. One roll in thirteen is garbage. Over a 100-token answer, that is about eight garbage tokens.</p>

        <h3>The cache by hand</h3>
        <p>Take a 7-billion-parameter model of the classic shape: 32 layers, 32 attention heads, 128 numbers per head, each number stored in 2 bytes (16-bit). For <b>one token</b> the cache must hold a key and a value, per head, per layer:</p>
        <p className="mono" style={{ fontSize: 14.5 }}>2 × 32 layers × 32 heads × 128 × 2 bytes = 524,288 bytes = <b>0.5 MB per token</b></p>
        <p>A 4,096-token conversation: 4,096 × 0.5 MB = <b>2 GB</b>. For one user. Ten users at once: 20 GB, more than the roughly 14 GB the model’s own weights take. This is the number that decides how long a context a provider can offer and how many conversations fit on one GPU.</p>
      </Numbers>

      <TheMath>
        <Equation
          label="Probability of token i equals e to the logit over T, divided by the sum of the same over all tokens"
          symbols={[
            [<>z<sub>i</sub></>, 'the logit (raw score) the model gave token i'],
            ['T', 'the temperature: T < 1 sharpens, T > 1 flattens, T = 1 changes nothing'],
            [<>p<sub>i</sub></>, 'the probability of drawing token i, before any top-k / top-p cut'],
            [<>Σ<sub>j</sub></>, 'sum over every token in the vocabulary, so that the probabilities add up to 1'],
          ]}
        >
          p<sub>i</sub> = e<sup>z<sub>i</sub> / T</sup> / Σ<sub>j</sub> e<sup>z<sub>j</sub> / T</sup>
        </Equation>
        <p>Dividing by T <em>before</em> softmax is the whole trick: it scales the <em>gaps</em> between logits, and softmax only cares about gaps. Dividing the probabilities afterwards would do nothing useful (you will debug that mistake below).</p>
        <Equation
          label="Cache bytes equals two times layers times k v heads times head dimension times tokens times bytes per number"
          symbols={[
            ['2', <>one <span className="k">key</span> vector and one <span className="v">value</span> vector</>],
            ['layers', 'every Transformer block has its own attention, so its own cache'],
            ['kv_heads × head_dim', 'the size of one key (or value) vector: all heads side by side'],
            ['tokens', 'prompt plus everything generated so far: the cache only grows'],
            ['bytes', '4 for 32-bit numbers, 2 for 16-bit, 1 for 8-bit'],
          ]}
        >
          cache bytes = 2 × layers × kv_heads × head_dim × tokens × bytes
        </Equation>
        <p>Everything in it is fixed by the model except <b>tokens</b>. The cache grows in a straight line with the conversation, for each user being served.</p>
        <DeepDive title="What the cache does not save">
          <p>The new token’s query still has to be compared with <em>every</em> cached key. That is T dot products per layer for token T, so the attention part of generating T tokens still adds up to roughly T²/2 comparisons. The quadratic cost of attention has not gone away.</p>
          <p>What the cache removes is everything else being redone for old tokens: their q, k, v projections, their feed-forward layers, their LayerNorms, in every block. Those are the bulk of the arithmetic. Without a cache, producing token T pushes all T tokens through the whole network again. With it: one token goes through, plus T cheap dot products per layer.</p>
        </DeepDive>
      </TheMath>

      <CodeIt>
        <h3>Sampling</h3>
        <p>Here is the inside of the generation loop from <code>tiny_gpt.py</code>. The network is one line. The sampling policy is the other three.</p>
        <Code source="phase3-transformers/tiny_gpt.py" title="GPT.generate, the body of the loop">{`
logits, _ = self(idx_cond)                      # the network: done
logits = logits[:, -1, :] / temperature         # last position only, then the contrast knob
probs = F.softmax(logits, dim=-1)               # scores -> probabilities
nxt = torch.multinomial(probs, num_samples=1)   # roll the weighted die
idx = torch.cat([idx, nxt], dim=1)              # feed it back in
`}</Code>
        <p>Top-k and top-p are a few more lines between softmax and the roll. From <code>kv_cache_demo.py</code>:</p>
        <Code source="phase3-transformers/kv_cache_demo.py" title="top-k, then top-p">{`
# top-k: keep k best, renormalize -- the garbage tail gets exactly 0
p = softmax(logits).copy()
cutoff = np.sort(p)[-k]
p[p < cutoff] = 0
p = p / p.sum()

# top-p: keep smallest set covering 90% of the mass
order = np.argsort(-probs)                       # most likely first
csum = np.cumsum(probs[order])                   # running total
keep = order[:np.searchsorted(csum, 0.9) + 1]    # up to and including the one that crosses 0.9
`}</Code>

        <h3>The cache</h3>
        <p>The naive loop calls <code>forward_full</code>, which runs every token. The cached loop calls <code>forward_step</code>, which runs one. Here is the attention part of <code>forward_step</code> for one layer:</p>
        <Code source="phase3-transformers/kv_cache_demo.py" title="forward_step: one token, one layer">{`
q = heads(x @ p[f"Wq{l}"], 1)        # x is ONE token: shape (1, D)
k = heads(x @ p[f"Wk{l}"], 1)
v = heads(x @ p[f"Wv{l}"], 1)

# append this token's K,V to the layer cache. Old K,V never change
cache[l]["K"] = np.concatenate([cache[l]["K"], k], axis=1)
cache[l]["V"] = np.concatenate([cache[l]["V"], v], axis=1)
Kc, Vc = cache[l]["K"], cache[l]["V"]            # (H, T_so_far, hd)

s = q @ Kc.transpose(0, 2, 1) / np.sqrt(HD)      # one query against every cached key
# no mask needed: the cache only CONTAINS the past.
x = x + unheads(softmax(s) @ Vc, 1) @ p[f"Wo{l}"]
`}</Code>
        <p>Notice the comment about the mask. The new token is the last one, and the cache holds only tokens before it, so there is no future to hide.</p>
        <Code source="phase3-transformers/kv_cache_demo.py" title="the two loops, side by side">{`
def generate_naive(p, prompt, n):
    ids = list(prompt)
    for _ in range(n):
        logits = forward_full(p, ids)          # reprocess EVERYTHING
        ids.append(int(np.argmax(logits)))     # greedy, for determinism
    return ids

def generate_cached(p, prompt, n):
    cache = empty_cache()
    for i, t in enumerate(prompt):             # "prefill": build cache from prompt
        logits = forward_step(p, t, i, cache)
    ids = list(prompt)
    for _ in range(n):                         # "decode": one token per step
        ids.append(int(np.argmax(logits)))
        logits = forward_step(p, ids[-1], len(ids) - 1, cache)
    return ids
`}</Code>
        <p>For simplicity this demo prefills one token at a time. Real systems push the whole prompt through in one parallel pass, which is much faster on a GPU, and fill the cache as a side effect.</p>
        <p>And the most important line in the file. A cache must never change the answer:</p>
        <Code source="phase3-transformers/kv_cache_demo.py" title="the regression test">{`
a = generate_naive(p, prompt, 20)
b = generate_cached(p, prompt, 20)
assert a == b, "cache changed the output -- that's a bug!"
`}</Code>
        <p>Running the file on a laptop CPU printed this. Your timings will differ, the trend will not:</p>
        <Code lang="output" title="python kv_cache_demo.py">{`
  naive and cached outputs identical: [1, 7, 3, 6, 20, 21, 38, 3, 28, 6]... OK

  new tokens  naive (s) cached (s)  speedup
          20      0.002      0.001     1.7x
          60      0.012      0.004     3.1x
         120      0.053      0.009     6.2x

  cache size at T=123: 123.0 KB for this toy.
`}</Code>
        <p>The speedup is not a constant. It <em>grows</em> with length, because the naive loop’s waste grows with length. At 120 tokens it is 6×. At 4,000 tokens the naive loop is unusable.</p>
      </CodeIt>

      <BreakIt>
        <p>Predict first, then check in the two labs.</p>
        <ul>
          <li><b>Greedy, whole sentences.</b> Select the Greedy preset and look at the three generated sentences. All identical, and stuck in “sat on the cat sat on the cat…” until max tokens cuts them off. Greedy can never leave a loop: the same input always gives the same choice.</li>
          <li><b>Max tokens = 3.</b> The sentence stops mid-thought. Max tokens is a hard budget, not a request to “write about 3 words”. The model does not know it is about to be cut off.</li>
          <li><b>T = 3, top-k off.</b> Count the junk. Then set top-k to 4. The temperature is still 3, but junk cannot be drawn: it has probability exactly 0.</li>
          <li><b>Top-p = 0.4 at T = 1.</b> How many tokens survive? (One: “mat” alone already holds 40.3%.) Top-p can quietly turn sampling into greedy.</li>
          <li><b>KV calculator: kv_heads from 32 to 8.</b> The cache shrinks 4×. That one change is grouped-query attention, which you will meet in <a href="#/lesson/modern-architecture">Modern LLM architecture</a>.</li>
          <li><b>KV calculator: 7B model, 32,000 tokens, 8 conversations.</b> Does it fit on a 24 GB GPU?</li>
        </ul>
      </BreakIt>

      <Exercises>
        <CodeExercise id="inference-code-topp" />
        <Exercise
          id="inference-temperature-calc"
          type="calculate"
          title="Turn the temperature down"
          answer={{ value: 0.88, tolerance: 0.011 }}
          answerLabel="P(yes) at T = 0.5"
          hints={[
            'Temperature is applied to the logits, before softmax: divide both logits by 0.5.',
            'The scaled logits are [2, 0]. Now softmax: e² / (e² + e⁰).',
            'e² = 7.39 and e⁰ = 1, so 7.39 / 8.39.',
          ]}
          solution={<><p>Scaled logits: [1, 0] ÷ 0.5 = [2, 0]. Softmax: 7.39 / (7.39 + 1) = <b>0.88</b>.</p><p>At T = 1 it would have been e¹ / (e¹ + 1) = 2.72 / 3.72 = 0.73. Halving the temperature doubled the gap between the logits, which moved “yes” from 73% to 88%. Halve it again (T = 0.25) and you get 98%. This is how T → 0 approaches greedy.</p></>}
        >
          <p>A model has two possible next tokens with logits <code>yes = 1</code> and <code>no = 0</code>. What is the probability of “yes” at temperature 0.5? (Two decimals.)</p>
        </Exercise>

        <Exercise
          id="inference-top-p-predict"
          type="predict"
          title="How many survive top-p?"
          answer={{ value: 4 }}
          answerLabel="number of tokens kept"
          hints={[
            'Sort from most to least likely (they already are) and keep a running total.',
            'Running totals: 0.50, 0.70, 0.85, 0.95. Top-p keeps tokens until the total reaches at least p.',
          ]}
          solution={<><p><b>4.</b> After three tokens the total is 0.85, which is still below 0.9, so the fourth is needed: 0.95 ≥ 0.9. The fifth (0.05) is cut.</p><p>The survivors are rescaled by dividing by 0.95: 0.526, 0.211, 0.158, 0.105. The common mistake is to answer 3 because “0.85 is close”. The rule is the <em>smallest set that reaches p</em>, so the token that crosses the line is included.</p></>}
        >
          <p>The probabilities after softmax are <code>[0.50, 0.20, 0.15, 0.10, 0.05]</code>. With top-p = 0.9, how many tokens can still be drawn?</p>
        </Exercise>

        <Exercise
          id="inference-kv-size"
          type="calculate"
          title="Size a cache"
          answer={{ value: 6.25, tolerance: 0.5 }}
          answerLabel="GB"
          hints={[
            'cache bytes = 2 × layers × kv_heads × head_dim × tokens × bytes_per_number.',
            'Per token: 2 × 40 × 40 × 128 × 2 = 819,200 bytes, about 0.78 MB.',
            'Times 8,192 tokens = 6,710,886,400 bytes. Divide by 1024³ (or by 10⁹: both are accepted).',
          ]}
          solution={<><p>2 × 40 × 40 × 128 × 2 = 819,200 bytes per token. Times 8,192 tokens = 6.7 billion bytes = <b>6.25 GB</b> (dividing by 1024³; 6.7 GB if you divide by 10⁹).</p><p>For <em>one</em> conversation. A server handling 16 of these at once needs 100 GB for caches alone. Check your answer in the calculator.</p></>}
        >
          <p>A 13B-class model: 40 layers, 40 attention heads of 128 numbers, no head sharing, 16-bit numbers. How many GB of KV cache does one 8,192-token conversation need?</p>
        </Exercise>

        <Exercise
          id="inference-debug"
          type="debug"
          title="Two silent bugs"
          hints={[
            'Bug 1 is in the sampling: where is the temperature applied, and do the numbers still sum to 1 afterwards?',
            'Bug 2 is in the cache: after this step, what does the cache contain? What will the NEXT token be able to look at?',
            'Dividing probabilities by T and renormalising gives back exactly the same probabilities. And a k, v that is never appended is a token that no later token can ever see.',
          ]}
          solution={<><p><b>Bug 1:</b> temperature is applied <em>after</em> softmax. Dividing every probability by the same T and renormalising changes nothing: (p/T) / Σ(p/T) = p. The knob is dead. (Without the renormalising line it would not even be a distribution.) Temperature must divide the <em>logits</em>.</p><p><b>Bug 2:</b> the new token’s k and v are used for this step (via <code>np.concatenate</code> into local variables) but never written back to <code>cache[l]</code>. The next token will attend over a cache that is missing this one, as if it had never been said. No crash, shapes are fine, the text just gets quietly worse. This is exactly what the <code>assert a == b</code> test exists to catch.</p></>}
        >
          <p>This code runs without errors. The temperature setting seems to do nothing, and long generations lose the thread. Find both bugs.</p>
          <Code>{`
# --- inside forward_step, for layer l ---
Kc = np.concatenate([cache[l]["K"], k], axis=1)
Vc = np.concatenate([cache[l]["V"], v], axis=1)
s = q @ Kc.transpose(0, 2, 1) / np.sqrt(HD)
x = x + unheads(softmax(s) @ Vc, 1) @ p[f"Wo{l}"]

# --- after the model, choosing the token ---
probs = softmax(logits)
probs = probs / temperature
probs = probs / probs.sum()
next_id = rng.choice(len(probs), p=probs)
`}</Code>
        </Exercise>

        <Exercise
          id="inference-implement-topk"
          type="implement"
          title="Add top-k to the real GPT"
          hints={[
            'Open phase3-transformers/tiny_gpt.py and find GPT.generate. Add a parameter top_k=None.',
            'After dividing by temperature and before softmax: find the k-th largest logit with torch.topk(logits, top_k), and set everything below it to -inf.',
            'v, _ = torch.topk(logits, top_k); logits[logits < v[:, [-1]]] = float("-inf"). Softmax turns −inf into exactly 0, and renormalises for free.',
          ]}
          solution={<><Code>{`
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -self.cfg.context_len:]
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)               # the k largest logits
            logits[logits < v[:, [-1]]] = float("-inf")    # everything below the k-th: out
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, nxt], dim=1)
    return idx
`}</Code><p>Masking the <em>logits</em> with −∞ is the same trick as the causal mask: softmax gives those entries probability 0 and the rest automatically sum to 1. With 65 characters, try <code>top_k=5</code> at temperature 1.0: fewer stray capitals and odd symbols in the middle of words.</p></>}
        >
          <p><code>GPT.generate</code> in <code>tiny_gpt.py</code> has temperature but no top-k. Add a <code>top_k</code> argument. Then generate with <code>top_k=5</code> and with <code>top_k=None</code> and compare the text. Stretch goal (from the repo notes): retrofit a KV cache into <code>generate</code> and time it at 500 tokens.</p>
        </Exercise>

        <ExplainBack
          id="inference-explain"
          prompt="A teammate asks: “Why can the KV cache store keys and values forever without them going stale? And why don’t we cache queries too?” Answer using what you know about the causal mask."
          modelAnswer={<p>Because of the causal mask, a token only ever looks at itself and earlier tokens. So everything computed for a token, in every layer, depends only on the tokens up to that point. Appending new tokens cannot change it. Its key and value vectors are therefore final the moment they are computed, which makes them safe to cache: recomputing would give identical numbers. Queries are different because of how they are used. A token’s query is used once, at the moment that token looks back at the others. No later token ever reads an earlier token’s query, only its key (to match against) and its value (to take content from). Caching queries would store something nobody asks for again.</p>}
        />
      </Exercises>

      <CheckYourself
        questions={[
          {
            q: 'You send the same prompt twice with temperature 0.8 and get different answers. Which part of the system produced the difference?',
            options: ['The sampling step after the network: a weighted random draw from the same probabilities', 'The Transformer blocks, which contain random numbers', 'The tokenizer, which splits text differently each time', 'The weights, which change slightly between calls'],
            answer: 0,
            explain: 'Same input, same weights, same logits. The randomness is one dice roll per token, outside the network, and an early different roll changes everything after it.',
          },
          {
            q: 'Why does very high temperature produce nonsense rather than just “more creative” text?',
            options: ['The model’s weights get hotter and less accurate', 'It shortens the context window, so the model loses track of the beginning of the sentence it is writing', 'It flattens the distribution, handing real probability to the huge tail of inappropriate tokens, and every bad token is fed back as input', 'It disables the causal mask'],
            answer: 2,
            explain: 'The tail is tens of thousands of tokens. Flatten the die and they win rolls. Errors then compound because generation is autoregressive.',
          },
          {
            q: 'Why is it safe to reuse the keys and values of old tokens?',
            options: ['They are approximately right, and the error is small', 'It is only safe for short sequences', 'Because keys and values are not learned', 'Because of the causal mask, nothing computed for an old token depends on later tokens, so recomputing would give identical numbers'],
            answer: 3,
            explain: 'The cache is exact, not an approximation. The demo asserts that naive and cached generation produce identical output.',
          },
          {
            q: 'You paste a 20,000-token document and ask for a one-word answer. Where does most of the time go?',
            options: ['Decode: generating the word, because an output token costs far more arithmetic than all the input tokens together', 'Prefill: running all 20,000 prompt tokens through the model to build the cache and get the first token', 'Tokenization', 'Sampling'],
            answer: 1,
            explain: 'The prompt must be processed before anything can be generated. Long prompts cost compute (prefill) and memory (the cache) even if the answer is tiny.',
          },
          {
            q: 'A provider wants to serve twice as many simultaneous conversations on the same GPUs. What is the most direct obstacle?',
            options: ['The vocabulary is too large', 'Softmax is too slow', 'Every conversation needs its own KV cache, and cache memory grows with layers × heads × tokens', 'Temperature settings conflict between users'],
            answer: 2,
            explain: 'The weights are shared by all users. The caches are not. That memory is what limits batch size and context length.',
          },
        ]}
      />

      <Remember
        items={[
          <>The network’s job ends at the <b>logits</b>. Picking a token is a policy applied <b>outside</b> the model: ÷ temperature → softmax → top-k / top-p → renormalise → roll the die.</>,
          <><b>Temperature</b> scales the gaps between logits. T → 0 is greedy (deterministic, loops easily). T &gt; 1 feeds the junk tail. <b>Top-k / top-p</b> cut the tail to exactly zero.</>,
          <>The same prompt gives different answers because a <b>random draw</b> happens at every token, and each draw is fed back in.</>,
          <><b>KV cache</b> = memoisation. The causal mask means old tokens’ <span className="k">keys</span> and <span className="v">values</span> never change, so compute q, k, v for one new token, append k and v, attend over the cache. Output is identical to the naive loop.</>,
          <><b>Prefill</b> (whole prompt, parallel, the wait) then <b>decode</b> (one token per step, the stream). Speed is bought with memory: <span className="mono">2 × layers × kv_heads × head_dim × tokens × bytes</span>, per conversation.</>,
        ]}
      />

      <RealLLM>
        <Flow horizontal steps={[{ label: 'prompt tokens' }, { label: 'prefill', sub: 'fills the KV cache' }, { label: 'logits' }, { label: 'sampling policy', sub: 'T, top-k, top-p' }, { label: 'one token', sub: 'append, decode again' }]} active={3} />
        <ToyVsReal
          toy={<ul><li>11 hand-picked logits, a toy word table</li><li>A 2-layer random-weight Transformer in NumPy</li><li>Cache of a few kilobytes</li><li>One sequence at a time, prefill token by token</li></ul>}
          real={<ul><li>50,000 to 200,000 logits per step, from the trained network</li><li>Dozens of layers, fused GPU kernels</li><li>Gigabytes of cache per long conversation, carefully paged in GPU memory</li><li>Many users batched together; prompts prefilled in parallel</li></ul>}
        />
        <Callout kind="established">The parameters you see in an LLM API are this lesson. <code>temperature</code>, <code>top_p</code>, <code>top_k</code>: reshaping the die. <code>max_tokens</code>: the hard budget. <code>stop</code> sequences and the end-of-sequence token: the stop conditions. “Prompt caching” on a pricing page is a KV cache kept <em>between</em> requests that share the same beginning, so the shared part is not prefilled again.</Callout>
        <Callout kind="established">A model has no memory between calls. A chat feels continuous because the application sends the whole conversation again every turn, and it is prefilled again (or fetched from a prompt cache). The <G t="context-window">context window</G> is the maximum length of that input, not a storage area.</Callout>
        <Callout kind="note" label="The order of the knobs">We applied temperature first, then top-k, then top-p. That is the order in this course’s code and in the widely used Hugging Face library. Not every implementation agrees: some cut the tail first and apply temperature last, which makes the surviving set independent of T. Hosted APIs do not always document their order. Practical advice, which several providers also give: adjust temperature <em>or</em> top-p, not both at once.</Callout>
        <Callout kind="research">Cache memory is the bottleneck of serving, so shrinking it is a busy area: sharing keys and values between heads (<G t="gqa">GQA</G>, next part), storing the cache in 8 or 4 bits, evicting or compressing old tokens, and attention variants that need less of it. How much quality each trick costs is an empirical question that is still being worked out. Also still debated: why sampling from the model’s own distribution (T = 1, no cuts) reads worse than slightly sharpened sampling, when that distribution is exactly what training optimised.</Callout>
        <DeepDive title="Is temperature 0 really deterministic?">
          <p>In our code, yes: argmax of the same numbers is the same token. In hosted services, often not quite. GPU arithmetic can give very slightly different logits depending on how requests are batched together, and when two tokens are nearly tied, a difference in the last decimal place flips the choice. After one flipped token, the rest of the answer differs. If you need reproducibility, do not rely on temperature 0 alone.</p>
        </DeepDive>
      </RealLLM>

      <BeforeMovingOn
        id="part-7"
        intro="You can now build, train and run a GPT. Before looking at what modern models changed, check that the whole stack from Parts 0 to 7 is still in your head. These questions reach back on purpose."
        questions={[
          {
            q: 'Part 1. Attention scores, the output layer and embedding similarity all rely on one operation to ask “how much do these two vectors agree?” Which?',
            options: ['The dot product: multiply element by element, add up', 'Softmax: exponentiate both vectors and divide by the total', 'The derivative', 'Concatenation'],
            answer: 0,
            explain: 'q · k in attention, hidden vector · each row of the output matrix for logits, and cosine similarity is a dot product of length-1 vectors.',
          },
          {
            q: 'Part 1. The logits are [5, 5, 5]. What does softmax return?',
            options: ['[1, 0, 0]', '[5/15, 5/15, 5/15] because it divides by the sum of logits', 'It is undefined', '[⅓, ⅓, ⅓]: only differences between logits matter, and there are none'],
            answer: 3,
            explain: 'Equal scores in, equal probabilities out. That is also why an untrained model with small starting weights begins at loss ln(V).',
          },
          {
            q: 'Parts 2 and 3. In one training step, what does backpropagation deliver, and who uses it?',
            options: ['The new weights; the model uses them immediately', 'The gradient of the loss for every parameter; the optimizer uses it to nudge each parameter', 'The validation loss; the engineer uses it', 'The next token; the sampler uses it'],
            answer: 1,
            explain: 'loss.backward() fills in gradients using the chain rule. opt.step() turns gradients into updates. Two separate jobs.',
          },
          {
            q: 'Part 4. Why does GPT work on sub-word tokens rather than whole words or single characters?',
            options: ['Sub-words are easier for humans to read', 'Because softmax needs at least 50,000 outputs', 'Single characters cannot be given embeddings and whole words cannot be given ids, so sub-words are the only option', 'Whole words give an endless vocabulary and fail on new words; characters make sequences very long. Sub-word pieces are the compromise'],
            answer: 3,
            explain: 'BPE keeps common chunks as single tokens and spells rare words from pieces, so nothing is ever out of vocabulary.',
          },
          {
            q: 'Part 4. What is an embedding table, mechanically?',
            options: ['A matrix with one learned row per token id; “embedding a token” is looking up its row', 'A dictionary of word definitions that the model consults whenever it meets a token it has not seen', 'A hash function', 'The softmax of the token id'],
            answer: 0,
            explain: 'Lookup by row index. The rows start random and are moved by gradient descent like any other weights.',
          },
          {
            q: 'Part 5. The bigram model and our playground MLP with context 1 both got stuck at a high loss. Why did more training not help?',
            options: ['The learning rate was wrong: with a smaller one, both would eventually have matched the longer-context models', 'They had too many parameters', 'The information needed to predict the next character was not in the input: one character of context is too little', 'Cross-entropy cannot go lower'],
            answer: 2,
            explain: 'No optimizer can extract information that is not there. More context was the fix, and attention is how to get it without a fixed window.',
          },
          {
            q: 'Part 6. In attention, what decides how much token A takes from token B?',
            options: ['The distance between them in the sentence: nearer tokens always receive more weight than farther ones', 'The dot product of A’s query with B’s key, turned into a weight by softmax (and forced to 0 if B comes after A)', 'The size of B’s value vector', 'A fixed table of word pairs'],
            answer: 1,
            explain: 'Query meets key to give a score, softmax gives weights, the weights blend the values. The causal mask removes the future.',
          },
          {
            q: 'Part 6. A Transformer block computes x = x + attention(LayerNorm(x)). What is the “x +” for?',
            options: ['It is the residual connection: the layer only adds a correction, and gradients get a direct path back through deep stacks', 'It doubles the signal so that it stays strong enough to survive the LayerNorm in the next block', 'It applies the causal mask', 'It converts vectors to logits'],
            answer: 0,
            explain: 'Without residuals, deep stacks are very hard to train. With them, each block refines the running representation.',
          },
          {
            q: 'Part 7. The model outputs logits of shape (B, T, vocab). During generation, which of them are used to choose the next token?',
            options: ['All of them, averaged over the T positions so that every token in the prompt gets a vote', 'Those at the first position', 'Only those at the last position: that position has seen the whole sequence so far', 'A random position'],
            answer: 2,
            explain: 'logits[:, -1, :]. In training every position is graded. In generation only the last one tells us what comes next.',
          },
        ]}
      >
        <OrderExercise
          id="part-7-pipeline"
          title="Rebuild the whole pipeline: text in, next token out"
          prompt={<p>You have now built every one of these steps. Put them in the order they run when a model produces one new token.</p>}
          correct={[
            'Tokenizer: text → token ids',
            'Embedding lookup: id → vector, plus position',
            'Transformer blocks × N: attention, then feed-forward',
            'Final LayerNorm, keep the last position',
            'Output head: vector → one logit per vocabulary token',
            'Divide the logits by the temperature',
            'Softmax: logits → probabilities',
            'Top-k / top-p: cut the tail, renormalise',
            'Sample one token id',
            'Append it to the sequence and go round again',
          ]}
          solutionNote={<p>Text becomes ids, ids become vectors, the blocks let the vectors exchange and process information, and the last position’s vector is scored against every vocabulary token. Everything after the logits is the sampling policy from this lesson, outside the network. With a KV cache, the second trip round the loop only pushes the one new token through steps 2 to 5.</p>}
        />
      </BeforeMovingOn>
    </Lesson>
  )
}
