import { RepoRunner } from '../components/RepoRunner'
import { CodeExercise } from '../components/python'
import { Lesson, Why, Problem, MentalModel, TryIt, Numbers, TheMath, CodeIt, BreakIt, Exercises, CheckYourself, Remember, RealLLM } from '../components/lesson'
import { Callout, DeepDive, Equation, G, Term, ToyVsReal, WhyExists } from '../components/ui'
import { Code } from '../components/Code'
import { Exercise, ExplainBack } from '../components/exercise'
import { QuantizeLab } from '../interactive/QuantizeLab'
import { SpeculativeLab } from '../interactive/SpeculativeLab'

export default function MakingModelsCheaperLesson() {
  return (
    <Lesson id="making-models-cheaper">
      <Why>
        <p className="lede">The meeting room is too cold, as usual. The finance lead has one slide up, with one number circled in red: the monthly GPU quote for the support bot.</p>
        <p>“Can we halve it?” she asks. Riya has already packed the batch as tightly as scheduling allows. Continuous batching keeps the slots full. Paged allocation of the <G t="kv-cache">KV cache</G> fits more conversations into the same memory.</p>
        <p>Kabir leans over and whispers, “All of that took the model as given. Now change the model.”</p>
        <p>So what is left to change? Three things are still on the table.</p>
        <div className="grid-3">
          <div className="card">
            <h4 style={{ fontSize: 17, marginBottom: 6 }}>Store each weight in fewer bytes</h4>
            <p>A decode step is a read of every weight. Read a quarter of the bytes and the step is up to four times faster, and the memory you freed holds more KV cache.</p>
          </div>
          <div className="card">
            <h4 style={{ fontSize: 17, marginBottom: 6 }}>Get more than one token per pass</h4>
            <p>The arithmetic units are nearly idle during decode. Something else could ride along in that same read of the weights.</p>
          </div>
          <div className="card">
            <h4 style={{ fontSize: 17, marginBottom: 6 }}>Use more than one GPU</h4>
            <p>When 140 GB of weights meet an 80 GB card, no amount of scheduling helps. The model has to be cut up.</p>
          </div>
        </div>
        <Callout kind="idea">
          One sentence from the last lesson carries this one: <b>a decode step costs one read of the weights, and the arithmetic units are idle while it happens</b>. Fewer bytes to read makes the step shorter. Spare arithmetic makes the step do more. Both attack the same number from opposite ends.
        </Callout>
      </Why>

      <Problem>
        <p>Two reminders, so this lesson stands on its own.</p>
        <p><b>Decode is memory-bound.</b> To produce one token the GPU reads every weight once. The arithmetic for that token takes roughly 1% of the time the read takes. For one stream, tokens per second is at most <span className="mono">bandwidth ÷ weight bytes</span>.</p>
        <p><b>Prefill is the opposite.</b> A whole prompt goes through in one pass, so there is far more arithmetic per byte read, and the arithmetic units are the limit. Both are worked out in <a href="#/lesson/inference-systems">Serving many users at once</a>.</p>
        <WhyExists
          problem="A 16-bit weight costs 2 bytes, and every one of them crosses the memory link for every single token."
          naive="Buy a GPU with more bandwidth."
          fails="Bandwidth did grow: about 2 TB/s on an A100 to about 8 TB/s on a B200, four times in two generations. But 16-bit arithmetic grew about seven times over the same span, and models grew faster than either. And no bandwidth helps a model that does not fit in the memory you have."
          idea="Store each weight in fewer bits, as a small integer or a tiny float, with a shared scaling factor, and expand it back inside the matrix multiply."
          tradeoff="A 4-bit or 8-bit number cannot represent the weight exactly. The rounding error changes the model’s output, by an amount that depends on how many weights share each scaling factor."
        />
        <WhyExists
          problem="Decode produces exactly one token per pass, however idle the arithmetic units are."
          naive="Make the model smaller so the pass is faster."
          fails="A smaller model is a different, worse model. You wanted this model’s answers."
          idea="Let something cheap guess the next few tokens, a small model or a light extra head on the big one, and have the big model check all the guesses in one pass, which costs it almost nothing extra."
          tradeoff="The guesses are often wrong, and the small model’s own time is spent either way. It helps least exactly when the server is busiest."
        />
      </Problem>

      <MentalModel title="Fewer bytes, more tokens per pass, more GPUs">
        <h3>1. Quantization: fewer bytes per weight</h3>
        <p>Every weight today is a 16-bit number: 2 bytes. The first idea is to store it in 8 bits or 4 bits instead, and live with a little rounding.</p>
        <Term
          name="Quantization"
          plain={<>Store each weight in fewer bits plus a shared scale, instead of a 16-bit float. The small number can be an integer (int8, int4) or a tiny float (FP8, FP4). 8 bits halves the bytes, 4 bits quarters them.</>}
          example={<>Eight weights with largest magnitude 0.035. Scale = 0.035 ÷ 7 = 0.005. The weight 0.021 is stored as round(0.021 ÷ 0.005) = 4 and read back as 4 × 0.005 = 0.020.</>}
          formal={<>Symmetric absmax integer quantization: scale = max|w| ÷ (2<sup>bits−1</sup> − 1), q = round(w ÷ scale), ŵ = q × scale. “Weight-only” means activations stay in 16-bit and weights are unpacked on the fly inside the matrix multiply.</>}
        />
        <p><b>int8</b> and <b>int4</b> mean 8-bit and 4-bit integers: 256 and 16 possible values per weight. A 16-bit float has tens of thousands.</p>
        <p><b>Weight-only</b> means only the stored weights shrink. Activations, the vectors flowing between layers, stay in 16-bit, and each weight is expanded back inside the matrix multiply.</p>
        <p>Decode is limited by bytes moved, so this is one of the few techniques that makes serving <b>smaller and faster at once</b>. Four times fewer bytes is up to four times more tokens per second for one stream, and the freed memory holds more KV cache.</p>
        <p>The danger is <b>outliers</b>. One scale has to stretch far enough to reach the largest value in its group. A single weight 25 times larger than the rest stretches the grid until every ordinary weight rounds to zero.</p>
        <p>The fix is more scales, so that each one covers fewer weights. This is the <b>granularity</b> of the quantization: one scale per tensor is the coarsest, then one per row (per output channel), then one per group of 32 to 128 weights.</p>
        <p>Scales are not free. Each one costs 16 bits of its own, so group-wise int4 really costs about 4.1 to 4.5 bits per weight.</p>
        <Callout kind="dev">
          This is lossy compression applied to a lookup table you have to read in full, on every request, forever. The usual trade-off is size against decompression time. Here decompression is almost free, because the arithmetic units were idle anyway. So the trade is size against accuracy alone.
        </Callout>
        <p>Real integer methods improve on plain rounding. You will meet these names on model cards, so here is one line each.</p>
        <ul>
          <li><b>LLM.int8()</b> (Dettmers et al., NeurIPS 2022) keeps the few outlier activation dimensions in 16-bit and quantizes the rest.</li>
          <li><b>GPTQ</b> (Frantar et al., ICLR 2023) rounds the weights one at a time and nudges the not-yet-rounded ones to make up for each rounding error.</li>
          <li><b>AWQ</b> (Lin et al., MLSys 2024) rescales the roughly 1% of weight channels that meet large activations, so they survive the grid.</li>
          <li><b>NF4</b> (QLoRA, Dettmers et al., NeurIPS 2023) spaces its 16 levels closer together in the middle, where bell-shaped weights actually sit.</li>
        </ul>

        <h3>Low-precision floats: FP8 and FP4</h3>
        <p>Integers are not the only small numbers. In data centres today the main route is <b>tiny floating-point formats</b>, because recent GPUs do their arithmetic on them directly.</p>
        <p>A float spends some of its bits on an <b>exponent</b> (how big) and the rest on a <b>mantissa</b> (how precise). With only 8 or 4 bits, you choose the split.</p>
        <Term
          name="FP8 (E4M3 and E5M2)"
          plain={<>An 8-bit float. E4M3 has 4 exponent bits and 3 mantissa bits: more precision, largest value 448. E5M2 has 5 and 2: less precision, but reaches 57,344.</>}
          example={<>Weights and activations usually use E4M3. Gradients in training, which swing across a wide range, have often used E5M2.</>}
          formal={<>Supported natively by NVIDIA Hopper (H100, H200) and later GPUs and by AMD’s MI300 series and later. Each tensor, or each block of a tensor, carries a higher-precision scale so that its values land inside the format’s range.</>}
        />
        <p>FP8 is now a standard way to serve large open models: the weights halve, and the matrix multiplies themselves run in 8 bits. It is used in training too. DeepSeek-V3 (December 2024) trained most of its matrix multiplies in FP8 E4M3, with one scale per small tile of activations and per 128 × 128 block of weights. It was one of the first openly documented training runs at that scale to do so.</p>
        <Term
          name="Microscaling FP4 (MXFP4, NVFP4)"
          plain={<>A 4-bit float (1 sign bit, 2 exponent, 1 mantissa) can only be 0, 0.5, 1, 1.5, 2, 3, 4 or 6, positive or negative. So every small block of weights gets its own scale.</>}
          example={<>MXFP4: blocks of 32 values share one 8-bit power-of-two scale, so 4 + 8 ÷ 32 = 4.25 bits per weight. NVFP4: blocks of 16 share an FP8 scale, plus one scale for the whole tensor, so about 4.5 bits.</>}
          formal={<>MXFP4 is defined in the Open Compute Project’s Microscaling (MX) specification (2023). NVFP4 is NVIDIA’s variant. Blackwell GPUs (B200, B300) compute on FP4 in hardware.</>}
        />
        <p>Here is the block-scale idea on a block of 4 weights (real MXFP4 blocks hold 32): <span className="mono">[0.30, −0.12, 0.05, 0.71]</span>.</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>step</th><th>arithmetic</th><th>result</th></tr></thead>
            <tbody>
              <tr><td>1. Largest value’s exponent</td><td className="mono">0.71 lies between 2⁻¹ and 2⁰</td><td className="mono"><b>−1</b></td></tr>
              <tr><td>2. Shared scale</td><td className="mono">2<sup>−1 − 2</sup> (FP4’s largest, 6, is 1.5 × 2²)</td><td className="mono"><b>0.125</b></td></tr>
              <tr><td>3. Divide by the scale</td><td className="mono">[2.4, −0.96, 0.4, 5.68]</td><td className="mono">off the grid</td></tr>
              <tr><td>4. Round to the FP4 grid</td><td className="mono">nearest of 0, 0.5, 1, 1.5, 2, 3, 4, 6</td><td className="mono"><b>[2, −1, 0.5, 6]</b></td></tr>
              <tr><td>5. Read back (× 0.125)</td><td className="mono">[0.25, −0.125, 0.0625, 0.75]</td><td className="mono">errors 0.05, 0.005, 0.0125, 0.04</td></tr>
            </tbody>
          </table>
        </div>
        <p>Stored: four 4-bit codes and one 8-bit exponent. Notice the grid. Its levels sit close together near zero and far apart near the top, so small weights keep their detail and the largest ones round more coarsely. That suits bell-shaped weights, the same reason NF4 spaces its levels that way.</p>
        <Callout kind="established">
          OpenAI’s open-weight gpt-oss models (August 2025) shipped their mixture-of-experts weights in MXFP4, which is how the 120B-parameter model fits on a single 80 GB GPU. Blackwell’s tensor cores run FP4 matrix multiplies natively, at a higher peak rate than FP8.
        </Callout>
        <p>The <b>KV cache</b> can be stored in low precision too, which raises the number of sequences that fit. vLLM offers an FP8 cache, and research such as KIVI (ICML 2024) goes as low as 2 bits.</p>
        <Callout kind="model">
          A widely reported regularity, not a law: 8-bit weights, integer or FP8, are usually close to lossless, and 4-bit weights cost a small but measurable amount of quality that many applications accept. It varies with the model, its size, the method and above all the task: long reasoning chains and code tend to be more sensitive than short chat. Do not trust a leaderboard for this. Run your own <a href="#/lesson/evals">evals</a> on the quantized model before you ship it.
        </Callout>

        <h3>2. Speculative decoding: spend the idle arithmetic</h3>
        <p>If the arithmetic units are idle during decode, give them something useful. The big model can score 5 positions in one pass in nearly the time it scores 1, exactly like a tiny prefill. It only needs to know which 5 tokens to score.</p>
        <Callout kind="analogy">
          Amma, when Riya explains this on the phone: “Like a student who writes out the whole sum and brings it to me. I tick five lines in the time it takes to check one.” At the first wrong line she corrects it and crosses out everything below. Where it stops: Amma’s ticking is yes or no. The real check is a probability rule, and it guarantees the final text is distributed exactly as if the big model had written it alone.
        </Callout>
        <Term
          name="Speculative decoding"
          plain={<>A small, fast draft model guesses the next few tokens. The big target model checks all the guesses in one pass. Guesses are kept or replaced by a rule that makes the final text distributed exactly as if the big model had written every token itself.</>}
          example={<>The draft proposes “sat on the mat”. The target accepts “sat”, “on”, “the”, rejects “mat” and replaces it with “sofa”. Four tokens for one pass of the big model.</>}
          formal={<>Accept a proposed token x with probability min(1, p(x) ÷ q(x)), where p is the target’s probability and q the draft’s. On the first rejection, draw a replacement from max(0, p − q) rescaled to sum to 1, and discard the later guesses. Leviathan et al. (ICML 2023) and Chen et al. (2023) both prove the output distribution equals the target’s.</>}
        />
        <p>The <b>target model</b> is the one whose answers you actually want. The <b>drafter</b> is whatever makes the guesses, and there are three common kinds.</p>
        <ul>
          <li><b>A separate small model</b>, a few hundred million parameters against the target’s tens of billions. This is the original recipe, and the one our demo uses.</li>
          <li><b>A draft head</b> on the target itself: one light extra layer that reads the target’s own hidden vectors and guesses ahead. EAGLE-3 (Li et al., 2025) is the best-known version, and heads of this kind are now a common choice in production engines.</li>
          <li><b>Multi-token prediction heads</b> trained with the model. DeepSeek-V3 was trained to predict one extra token at each position, and at serving time that head drafts. Its report says the extra token was accepted 85% to 90% of the time.</li>
        </ul>
        <p>The acceptance rule below is the same for all three. Only where the guesses come from changes.</p>
        <p>It is a latency technique that only works <em>because</em> decode is memory-bound. It does not help in two cases:</p>
        <ul>
          <li>the drafter is often wrong: little is accepted, and its own time is wasted;</li>
          <li>the server already runs large batches: the arithmetic is no longer idle, so verifying extra positions is no longer free.</li>
        </ul>
        <Callout kind="dev">
          Branch prediction, with a guarantee. A processor guesses the next instructions and throws the work away when it guessed wrong. Here the check is not just “was it right?” but a rule chosen so that the final text has exactly the distribution the big model would have produced on its own. A wrong guess costs time, never correctness.
        </Callout>

        <h3>3. When the model does not fit on one GPU</h3>
        <Term
          name="Tensor parallelism"
          plain={<>Split every weight matrix across several GPUs. Each GPU holds a slice, computes its part of every layer, and the parts are combined.</>}
          example={<>A 70B model in 16-bit is 140 GB. Across 4 GPUs of 80 GB each, every GPU holds 35 GB of weights and a quarter of each KV cache.</>}
          formal={<>In the Megatron-LM scheme (Shoeybi et al., 2019) the forward pass needs two all-reduce operations per Transformer layer. An all-reduce is the step where every GPU sends its partial result to the others and all of them end up with the sum. Each GPU reads only its slice, so the bandwidth-bound step gets faster too.</>}
        />
        <p><b>Tensor parallelism</b> cuts latency as well as memory. The price is that every layer now waits for the GPUs to talk to each other, twice. That needs the fast links inside one server, so it is rarely stretched across machines.</p>
        <p><b>Pipeline parallelism</b> splits by layers instead. GPU 1 runs layers 1 to 20, GPU 2 runs 21 to 40, and so on. It only passes activations between neighbours, so slower links are fine.</p>
        <p>But one token still visits every stage in turn, so latency does not improve, and a stage sits idle while it waits for the one before it. Those idle gaps are called bubbles (GPipe, Huang et al., 2019). You fill them by splitting the batch into smaller pieces and keeping several of them moving through the stages at once.</p>
        <p><b>Data parallelism</b> is the one you know: complete replicas behind a load balancer. It multiplies throughput and does nothing for one request’s latency. A large deployment typically combines them: tensor parallel inside a server so the model fits and responds quickly, then as many replicas as the traffic needs.</p>
        <Callout kind="idea">
          Notice how the three techniques divide the work. Quantization shrinks the bytes. Speculative decoding buys more tokens per read of those bytes. Parallelism spreads the bytes over more memory links. None of them changes the scheduler from <a href="#/lesson/inference-systems">the last lesson</a>: a real engine does all of it at once.
        </Callout>
      </MentalModel>

      <TryIt title="Two experiments">
        <h3>C. Quantize a layer</h3>
        <p>Start with int4 and one scale for the tensor. Then plant the outlier.</p>
        <QuantizeLab />
        <h3>D. Draft and verify</h3>
        <p>Step through a few rounds. Then drag the draft quality and watch the accepted tokens per pass.</p>
        <SpeculativeLab />
      </TryIt>

      <Numbers title="What int4 buys, on the same GPU">
        <p>Take the capacity plan from <a href="#/lesson/inference-systems">the last lesson</a> and change one thing: store the weights in int4 with group scales instead of 16-bit floats. Everything else is the same. 8B parameters, an 80 GB GPU with 2 TB/s of bandwidth, 10% of memory kept back, a KV cache of 131,072 bytes per token and 1,300 tokens per sequence at its longest.</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>step</th><th>16-bit</th><th>int4 with group scales</th></tr></thead>
            <tbody>
              <tr><td>Bytes per parameter</td><td className="mono">2</td><td className="mono">about 0.52</td></tr>
              <tr><td>Weights</td><td className="mono"><b>16 GB</b></td><td className="mono"><b>about 4.1 GB</b></td></tr>
              <tr><td>Memory left for KV caches</td><td className="mono">72 − 16 = <b>56 GB</b></td><td className="mono">72 − 4.1 = <b>67.9 GB</b></td></tr>
              <tr><td>Sequences that fit (÷ 170.4 MB)</td><td className="mono"><b>328</b></td><td className="mono"><b>398</b></td></tr>
              <tr><td>One-stream ceiling (2 TB/s ÷ weights)</td><td className="mono"><b>125 tokens/s</b></td><td className="mono"><b>about 480 tokens/s</b></td></tr>
            </tbody>
          </table>
        </div>
        <p>Two very different gains, from one change.</p>
        <ul>
          <li><b>For one user, almost four times the speed.</b> The weight read is the whole step, and it shrank by the full factor. This is the gain people notice on a laptop.</li>
          <li><b>For a loaded server, 21% more sequences.</b> The weights were only 16 of the 72 usable gigabytes, and the caches were already most of the rest. Freeing 12 GB helps, but it cannot help by four times, because the caches did not shrink.</li>
        </ul>
        <p>That gap is the thing to remember. Quantizing weights is a large win when the batch is small and a modest one when it is large, which is why <b>quantizing the KV cache</b> is a separate and increasingly important lever at scale.</p>
        <Callout kind="warn">
          The right column is a ceiling, and a more optimistic one than the left. Real int4 kernels unpack weights on the fly, which costs arithmetic that plain 16-bit kernels do not pay, and at large batches the arithmetic limit from the last lesson binds long before this ceiling does. Treat 480 as “the read is no longer what limits you”, not as a number to promise anyone.
        </Callout>
      </Numbers>

      <TheMath>
        <p>Two formulas, both about the acceptance rule, because that is where the surprise lives. Here p is the target’s probability for the proposed token and q the draft’s:</p>
        <Equation
          label="Accept with probability minimum of 1 and p over q; on rejection resample from the positive part of p minus q, normalised"
          symbols={[
            ['p(x), q(x)', 'probability of token x under the target and under the draft, in the same context'],
            ['min(1, p/q)', 'tokens the draft over-proposes (q > p) are thinned out; the rest are always kept'],
            ['max(0, p − q)', 'where the target wanted more mass than the draft offered: the replacement is drawn from here'],
          ]}
        >
          accept x with probability min(1, p(x) ÷ q(x)); else draw from max(0, p − q) ÷ Σ max(0, p − q)
        </Equation>
        <p>Why it is exact: x comes out by acceptance with probability q(x) × min(1, p(x)/q(x)) = min(p(x), q(x)). The rejected mass is 1 − Σ min(p, q) = Σ max(0, p − q), and the replacement step hands exactly max(0, p(x) − q(x)) back to x. The two parts add up to p(x). Be careful when reading the papers: Chen et al. use the letters the other way round (q for the target).</p>
        <Equation
          label="Expected tokens per target pass equals one minus alpha to the gamma plus one, over one minus alpha"
          symbols={[
            ['α', 'acceptance rate: the chance one proposal survives, Σ min(p, q), assumed the same at every position'],
            ['γ', 'tokens proposed per round'],
            ['γ + 1', 'the most one pass can yield: all proposals plus one bonus token from the target’s own scores'],
          ]}
        >
          E[tokens per target pass] = (1 − α<sup>γ+1</sup>) ÷ (1 − α)
        </Equation>
        <p>With α = 0.8 and γ = 4 that is 3.36 tokens per pass. It is a geometric series capped at γ + 1 (Leviathan et al., equation 1). The speed-up is smaller than this number, because the draft’s γ steps also take time.</p>
        <p>Quantization needs no equation beyond the absmax rule already given above: <span className="mono">scale = max|w| ÷ (2<sup>bits−1</sup> − 1)</span>, then <span className="mono">q = round(w ÷ scale)</span> and back as <span className="mono">ŵ = q × scale</span>. What matters about it is not the algebra but the measured error, which is next.</p>
        <DeepDive title="Why a bad draft can still be exact">
          <p>Nothing in the acceptance rule assumes the draft is any good. Take a draft that ignores the context entirely and proposes uniformly. Then q is flat, min(1, p/q) accepts rarely, and almost every position falls through to the residual draw from max(0, p − q) normalised, which for a flat q is close to p itself. The output is still the target’s distribution. What collapses is α, so E[tokens per pass] approaches 1, and you paid for the draft’s passes as well. Exactness is free; speed is not.</p>
        </DeepDive>
      </TheMath>

      <CodeIt>
        <h3>Quantization</h3>
        <p>The whole algorithm is three lines. Everything else in <code>quantize_demo.py</code> is about how many scales to use and measuring the error:</p>
        <Code source="phase6-engineering/quantize_demo.py" title="symmetric absmax quantization">{`
def qmax(bits):
    return 2 ** (bits - 1) - 1          # int8 -> 127, int4 -> 7

def quantize_block(w, bits):
    scale = np.abs(w).max() / qmax(bits)
    q = np.clip(np.round(w / scale), -qmax(bits), qmax(bits)).astype(np.int8)
    return q, scale                     # restore with q * scale
`}</Code>
        <Code source="phase6-engineering/quantize_demo.py" title="group-wise: one scale per `group` consecutive weights in a row">{`
G = W.reshape(rows, cols // group, group)
scale = np.abs(G).max(axis=2, keepdims=True) / qmax(bits)
q = np.clip(np.round(G / scale), -qmax(bits), qmax(bits))
W_hat = (q * scale).reshape(rows, cols)
`}</Code>
        <p>The file measures the error in the weights and, more usefully, in the layer’s output <code>y = x @ W.T</code>. Then it plants 8 outliers among 262,144 weights:</p>
        <div className="table-scroll">
          <table className="plain mono" style={{ fontSize: 13.5 }}>
            <thead><tr><th>bits</th><th>scales</th><th>output error, clean matrix</th><th>output error, 8 outlier weights</th></tr></thead>
            <tbody>
              <tr><td>8</td><td>tensor</td><td>1.02%</td><td>11.02%</td></tr>
              <tr><td>8</td><td>row</td><td>0.78%</td><td>2.09%</td></tr>
              <tr><td>4</td><td>tensor</td><td>18.49%</td><td><b>96.91%</b></td></tr>
              <tr><td>4</td><td>row</td><td>14.15%</td><td>22.72%</td></tr>
              <tr><td>4</td><td>group of 128</td><td>11.60%</td><td>12.79%</td></tr>
              <tr><td>4</td><td>group of 32</td><td>9.63%</td><td>9.90%</td></tr>
            </tbody>
          </table>
        </div>
        <p className="muted" style={{ fontSize: 14 }}>From sections 2 and 3 of <code>python phase6-engineering/quantize_demo.py</code>: W is 256 × 1024 with entries of typical size 0.02, outliers are ±1.0.</p>
        <p>Per-tensor int4 with outliers loses 97% of the signal: the step between levels becomes 0.14 while ordinary weights are about 0.02, so they all round to zero. Group-wise scales barely notice. These percentages are for plain rounding of one random layer. They are not what a real model loses: GPTQ and AWQ exist to do much better than this at the same 4 bits.</p>

        <h3>Speculative decoding</h3>
        <Code source="phase6-engineering/speculative_demo.py" title="one round: verify the draft’s proposals (simplified)">{`
for x in proposals:                          # the target scored all of them in ONE pass
    p, q = P[ctx], Q[ctx]
    if rng.random() < min(1.0, p[x] / q[x]):
        out.append(x)                        # accept
        ctx = x
    else:
        out.append(draw(residual(p, q), rng))   # replace from max(0, p - q), then stop
        return out
out.append(draw(P[ctx], rng))                # all accepted: one bonus token
`}</Code>
        <p>The file then tests the claim that matters, against the exact distribution over all three-token sequences:</p>
        <Code lang="output" title="python phase6-engineering/speculative_demo.py (section 3)">{`
  sampler                          total variation distance from the target
  target, one token at a time      0.0082   <- pure sampling noise at this N
  speculative (draft + verify)     0.0074   <- no larger than the noise: no bias
  draft alone                      0.5768   <- a different distribution
`}</Code>
        <p>That middle row is the whole point. The speculative sampler is no further from the target than the target is from itself at this sample size, so the acceptance rule introduces no bias that this test can see.</p>
        <RepoRunner path="phase6-engineering/quantize_demo.py" title="Run quantize_demo.py in your browser">
          <p>This is the whole file from the repository, running in your browser. Press Run to see what it prints, then edit a copy and change things.</p>
        </RepoRunner>
      </CodeIt>

      <BreakIt>
        <p>Predict first, then check.</p>
        <ul>
          <li><b>Lab C: int8, one scale for the tensor, plant the outlier.</b> Still usable. Now int4. Count the weights that rounded to zero.</li>
          <li><b>Lab C: int4, group of 32, then move the outlier.</b> Only the group containing it is damaged. How many ordinary weights were harmed, out of how many?</li>
          <li><b>Lab D: make the draft completely wrong.</b> Tokens per pass approach 1 but never go below it, and the text is still the target’s. What did you lose? (The time spent running the draft.)</li>
          <li><b>Lab D: make the draft perfect and raise the proposal count.</b> Tokens per pass rise towards γ + 1, not beyond it. Why is there always that cap? (One pass can only confirm the proposals it was given, plus one free token from the target’s own scores.)</li>
          <li><b>In <code>speculative_demo.py</code></b>, make the rejection branch draw from <code>p</code> instead of <code>residual(p, q)</code> and run <code>pytest tests/test_engineering_serving.py</code>. Which test catches it?</li>
          <li><b>In <code>quantize_demo.py</code></b>, raise the outlier magnitude from 1.0 to 10.0 and rerun. The per-tensor int4 error barely moves from 96.91%. Why can it not get much worse? (Once every ordinary weight already rounds to zero, a larger outlier changes nothing for them.)</li>
        </ul>
      </BreakIt>

      <Exercises>
        <CodeExercise id="making-models-cheaper-code-int8" />
        <Exercise
          id="making-models-cheaper-outlier"
          type="predict"
          title="One outlier, one scale"
          answer={{ value: 0, tolerance: 0 }}
          answerLabel="ordinary weights that survive as non-zero"
          hints={[
            'The scale must reach the largest magnitude in the group: scale = max|w| ÷ 7.',
            'scale = 1.4 ÷ 7 = 0.2. Each weight is stored as round(w ÷ 0.2).',
            '0.06 ÷ 0.2 = 0.3, −0.02 ÷ 0.2 = −0.1, 0.012 ÷ 0.2 = 0.06. What does each round to?',
          ]}
          solution={<><p>The scale becomes 0.2, and 0.3, −0.1 and 0.06 all round to 0. <b>None</b> of the ordinary weights survives. The group is stored as [0, 0, 0, 7].</p><p>Without the outlier the scale would be 0.06 ÷ 7 = 0.0086 and the three weights would be stored as 7, −2 and 1. This is why int4 needs small groups: a group of 32 contains the damage to 31 neighbours, a per-tensor scale spreads it over millions.</p></>}
        >
          <p>A group of four weights is quantized to int4 (levels −7 to 7) with one absmax scale: <code>[0.06, −0.02, 0.012, 1.4]</code>. How many of the three ordinary weights are stored as something other than 0?</p>
        </Exercise>

        <Exercise
          id="making-models-cheaper-int4-ceiling"
          type="calculate"
          title="The ceiling after quantization"
          answer={{ value: 149, tolerance: 5 }}
          answerLabel="tokens per second, upper bound"
          hints={[
            'The formula is unchanged: bandwidth ÷ weight bytes. Only the weight bytes change.',
            'int4 with group scales costs about 0.52 bytes per parameter, not 4 bits exactly, because each scale is stored too.',
            '13 × 10⁹ × 0.52 = about 6.7 GB. Now divide 1,000 GB/s by it.',
          ]}
          solution={<><p>13 billion parameters at about 0.52 bytes each is about 6.7 GB. 1,000 GB/s ÷ 6.7 GB ≈ <b>149 tokens per second</b>, up from about 38 in 16-bit.</p><p>Notice that the gain is not exactly 4. The scales are stored alongside the integers, so 4-bit weights cost 4.1 to 4.5 bits in practice. And this remains an upper bound: real int4 kernels spend arithmetic unpacking the weights, and KV cache reads are unchanged.</p></>}
        >
          <p>In <a href="#/lesson/inference-systems">the last lesson</a> you found that a 13B model in 16-bit on an accelerator with 1 TB/s of memory bandwidth tops out at about 38 tokens per second for one stream. Quantize the weights to int4 with group scales. What is the new upper bound?</p>
        </Exercise>

        <Exercise
          id="making-models-cheaper-accept"
          type="calculate"
          title="Tokens per pass"
          answer={{ value: 2.31, tolerance: 0.05 }}
          answerLabel="expected tokens per target pass"
          hints={[
            'Use E[tokens per pass] = (1 − α^(γ+1)) ÷ (1 − α), with α the acceptance rate and γ the number of proposals.',
            'α = 0.6, γ = 4, so γ + 1 = 5. You need 0.6 to the power 5.',
            '0.6⁵ = 0.07776. So (1 − 0.07776) ÷ 0.4.',
          ]}
          solution={<><p>(1 − 0.6⁵) ÷ (1 − 0.6) = 0.92224 ÷ 0.4 = <b>2.31</b> tokens per target pass. With α = 0.8 and the same γ it would be 3.36.</p><p>The acceptance rate matters far more than the number of proposals. Raising γ from 4 to 8 at α = 0.6 only lifts this from 2.31 to 2.47, because the series is already close to its limit of 1 ÷ (1 − α) = 2.5. Past that point you are running draft steps whose output is almost never reached. That is why a well-matched draft model beats a longer guess.</p></>}
        >
          <p>A draft model’s proposals survive the acceptance test 60% of the time, and it proposes 4 tokens per round. How many tokens does each pass of the target model produce, on average?</p>
        </Exercise>

        <Exercise
          id="making-models-cheaper-spec-debug"
          type="debug"
          title="A speculative decoder that is quietly wrong"
          hints={[
            'It runs, it is fast, and every emitted token has non-zero probability under the target. Check the probabilities, not the tokens.',
            'A token x is emitted either by acceptance, with probability min(p(x), q(x)), or by the rejection branch. What does the rejection branch add here?',
            'Total: min(p, q) + (1 − Σ min(p, q)) × p(x). Compare that with p(x) for a token where q(x) > p(x).',
          ]}
          solution={<><p>On rejection it draws from <code>p</code> instead of from the residual <code>max(0, p − q)</code>, rescaled. The output distribution becomes min(p, q) + (1 − Σ min) × p, which is not p.</p><p>Take p = [0.5, 0.5] and q = [0.9, 0.1]. Acceptance contributes [0.5, 0.1], and the rejected mass is 0.4. The bug adds 0.4 × [0.5, 0.5], giving [0.7, 0.3]. The correct residual is [0, 1], giving [0.5, 0.5]. The bug over-produces whatever the draft likes. Nothing crashes and the text looks fine, so only a statistical test finds it. That is why <code>speculative_demo.py</code> checks the one-step identity to machine precision and the three-token joint distribution empirically.</p></>}
        >
          <p>A colleague’s implementation is fast and the outputs look sensible, but an eval shows the quality is slightly below the target model’s. What is wrong?</p>
          <Code>{`
for x in proposals:
    p, q = P[ctx], Q[ctx]
    if rng.random() < min(1.0, p[x] / q[x]):
        out.append(x)
        ctx = x
    else:
        out.append(draw(p, rng))     # rejected: sample from the target instead
        return out
`}</Code>
        </Exercise>

        <Exercise
          id="making-models-cheaper-granularity"
          type="modify"
          title="Find the group size that pays"
          hints={[
            'quantize_demo.py already reports output error at group sizes 32 and 128. Add more group sizes to that sweep.',
            'Each scale is 16 bits shared by `group` weights, so the real cost is 4 + 16 ÷ group bits per weight.',
            'Tabulate bits per weight against output error for groups of 16, 32, 64, 128, 256 and the whole row.',
          ]}
          solution={<><p>The published sweep gives 9.63% error at a group of 32 and 11.60% at 128, on the clean matrix. In bits per weight those cost 4.5 and 4.125. So halving the group size repeatedly buys less and less error reduction while the storage cost climbs steeply below 32: a group of 16 would cost 5 bits per weight.</p><p>The point is that granularity is a knob with a price tag, not a free improvement, and the shape of the curve is what decides where the industry settled. Groups of 32 to 128 are what you will see on real model cards, and this is why.</p></>}
        >
          <p>Open <code>quantize_demo.py</code> and add group sizes 16, 64 and 256 to the int4 sweep. For each one, work out the real bits per weight (4 bits of integer plus one 16-bit scale shared by the group) and plot it against the measured output error. Where does shrinking the group stop being worth the storage?</p>
        </Exercise>

        <ExplainBack
          id="making-models-cheaper-explain"
          prompt="A colleague says: “We quantized to int4 and it is four times faster on my laptop. Let us ship it to the production server and cut the GPU bill by four.” Explain what will actually happen and why."
          modelAnswer={<p>The laptop runs one stream, where the step is almost entirely the read of the weights, so shrinking the weights by four shrinks the step by nearly four. The production server runs a large batch, and there the picture is different. The weight read is shared across every sequence in the batch, so it is already a small part of each user’s cost, while each sequence’s own KV cache reads and its own arithmetic are unchanged by quantizing the weights. On the capacity plan from the serving lesson, 16 GB of weights becoming about 4.1 GB frees 12 GB, which raises the sequences that fit from 328 to 398, about 21%, not 300%. Getting more than that means shrinking the caches too, by quantizing the KV cache or by a model with fewer K/V heads. And before any of it ships we have to run our own evals on the quantized model, because 4-bit weights cost a measurable amount of quality that depends on the task.</p>}
        />
      </Exercises>

      <CheckYourself
        questions={[
          {
            q: 'Why does weight-only int4 quantization speed up single-stream decoding, when it adds work to unpack each weight?',
            options: [
              'Because integer arithmetic is faster than floating-point arithmetic on every GPU',
              'Because the step is limited by moving weight bytes across the memory link, and the unpacking uses arithmetic units that were idle',
              'Because a 4-bit model has fewer parameters, so there is less of it to compute',
              'Because int4 weights let the model skip layers whose weights all rounded to zero',
            ],
            answer: 1,
            explain: 'Decode is memory-bound. Quartering the bytes shortens the part that dominates, and the extra unpacking work lands on arithmetic units that had nothing else to do. The parameter count is unchanged: only the bytes per parameter fell.',
          },
          {
            q: 'A team quantizes a layer to int4 with one scale for the whole tensor, and the output error jumps to nearly 97%. What happened?',
            options: [
              'A few very large weights forced the scale so wide that ordinary weights all rounded to zero',
              'The integers overflowed, because int4 cannot hold values above 7',
              'The activations were also quantized, which is never safe at 4 bits',
              'The layer was too large for one scale to be computed accurately in floating point',
            ],
            answer: 0,
            explain: 'The scale must reach the largest magnitude in its group. With outliers 25 times the typical weight, the step between levels becomes larger than the ordinary weights themselves. Group-wise scales fix it by containing each outlier to its own group.',
          },
          {
            q: 'Speculative decoding with a poor draft model produces text that is…',
            options: [
              'lower in quality, because rejected draft tokens leave traces in the final output',
              'distributed exactly as the target model’s own samples, only produced with less speed-up',
              'identical to greedy decoding from the target, because rejection removes the randomness',
              'a blend of the two models, weighted by how often the draft’s proposals were accepted',
            ],
            answer: 1,
            explain: 'The accept-or-replace rule makes the output distribution equal to the target’s for any draft. A bad draft only lowers the accepted tokens per pass, and can make the whole thing slower than plain decoding once the draft’s own cost is counted.',
          },
          {
            q: 'A server is already running a large batch, near the arithmetic limit. It turns on speculative decoding. What should the team expect?',
            options: [
              'A further speed-up, because speculative decoding always produces more than one token per pass',
              'Little or no gain, and possibly a loss, because the arithmetic that verification needs is no longer idle',
              'Lower output quality, because large batches lower the acceptance rate',
              'The same throughput but far lower memory use, because drafts need no KV cache',
            ],
            answer: 1,
            explain: 'Speculation is paid for out of idle arithmetic. At a large batch the arithmetic units are already busy, so verifying extra positions costs real time, and the draft model’s own passes cost time too. It is a latency technique for small batches.',
          },
          {
            q: 'A 140 GB model has to run on 80 GB GPUs. Tensor parallelism and pipeline parallelism both make it fit. What is the main difference for one user’s latency?',
            options: [
              'Tensor parallelism cuts latency because each GPU reads only its slice; pipeline parallelism does not, because one token still visits every stage in turn',
              'Pipeline parallelism cuts latency because stages overlap; tensor parallelism does not, because of the all-reduce',
              'Neither changes latency, because both hold the same total bytes of weights',
              'Both halve latency, but tensor parallelism needs twice the memory for activations',
            ],
            answer: 0,
            explain: 'Under tensor parallelism every GPU reads a fraction of the weights at the same time, so the bandwidth-bound step gets shorter, at the cost of two all-reduces per layer. Under pipeline parallelism the stages run one after another for a given token, so its latency is unchanged, and the gain is throughput once the bubbles are filled.',
          },
        ]}
      />

      <Remember
        items={[
          <><b>Quantization stores each weight in fewer bits plus a shared scale</b>: a small integer (int8, int4) or a tiny float (FP8, and FP4 formats such as MXFP4 with one scale per block of 32). Fewer bytes shrink memory and speed up decode, because decode is limited by the bytes read.</>,
          <><b>Outliers are what breaks it.</b> One scale must reach the largest value in its group, so a single huge weight can round its neighbours to zero. More scales, per row or per group of 32 to 128, contain the damage, at about 4.1 to 4.5 bits per weight.</>,
          <>The gain is <b>large for one stream and modest for a full server</b>, because a big batch is dominated by KV cache reads that quantizing the weights leaves untouched. Quantize the cache too if you need more.</>,
          <><b>Speculative decoding</b> has something cheap guess (a small model, an EAGLE-style head, a multi-token-prediction head) and the target check the guesses in one pass. The accept-or-replace rule keeps the target’s distribution <em>exactly</em>. It buys latency only while the arithmetic units are idle, so it fades at large batch sizes.</>,
          <>When the model does not fit: <b>tensor parallelism</b> splits every matrix and cuts latency at the cost of two all-reduces per layer, <b>pipeline parallelism</b> splits by layers and needs bubbles filled, <b>data parallelism</b> is replicas behind a load balancer.</>,
        ]}
      />

      <RealLLM>
        <ToyVsReal
          toy={<ul><li>Round-to-nearest quantization of one random matrix</li><li>Error measured on one layer’s output, not on any task</li><li>Speculative decoding between two 6-token Markov chains</li><li>Parallelism described, not run: one process, one machine</li></ul>}
          real={<ul><li>FP8 and FP4 (MXFP4, NVFP4) run natively by the hardware, plus GPTQ, AWQ and similar integer methods</li><li>Quality measured with task evals on the quantized model before it ships</li><li>EAGLE-style draft heads, native multi-token-prediction heads, draft models or n-gram lookups as the proposer, verified in batches</li><li>Tensor parallel inside a server over fast links, replicas across servers</li></ul>}
        />
        <p>Where you will meet these in practice:</p>
        <div className="table-scroll">
          <table className="plain">
            <tbody>
              <tr><td><b>Model cards</b></td><td>A name like “FP8”, “MXFP4”, “GPTQ int4, group 128” or “AWQ 4-bit” tells you the format, the method and the granularity, which is most of what you need to guess the quality cost.</td></tr>
              <tr><td><b>llama.cpp and GGUF</b></td><td>Integer quantization from 1.5 to 8 bits with named schemes such as Q4_K_M. This is what makes a 7B model run on a laptop.</td></tr>
              <tr><td><b>vLLM, SGLang and TensorRT-LLM</b></td><td>All serve FP8 and 4-bit weights, offer a low-precision KV cache, and support speculative decoding with draft models, EAGLE-style heads or a model’s own multi-token-prediction head.</td></tr>
              <tr><td><b>Tensor parallel size</b></td><td>A launch flag on every serving engine. Setting it to the number of GPUs in one server is the usual first answer to “the model does not fit”.</td></tr>
            </tbody>
          </table>
        </div>
        <Callout kind="established">
          The mechanisms here are published and open: LLM.int8() (NeurIPS 2022), the OCP Microscaling formats (2023), FP8 training in the DeepSeek-V3 report (2024), EAGLE-3 (2025), GPTQ (ICLR 2023), AWQ (MLSys 2024), NF4 and QLoRA (NeurIPS 2023), speculative decoding with its exactness proof (Leviathan et al., ICML 2023; Chen et al., 2023), Megatron-LM tensor parallelism (Shoeybi et al., 2019) and GPipe (Huang et al., 2019). The exactness of the acceptance rule is a proved theorem, not a measured tendency, and <code>speculative_demo.py</code> checks it empirically as well.
        </Callout>
        <Callout kind="model">
          Our quantization figures come from round-to-nearest on one random matrix. They show the shape of the effect, which is what they are for: how granularity fights outliers. They do not predict what a real model loses on a real task at 4 bits. Only your own evals do that.
        </Callout>
        <Callout kind="research">
          How much quality 4-bit and lower precision really costs on hard tasks, how far KV caches can be compressed, how to train draft models that match a target closely, and how to make speculation pay at large batch sizes are all active areas. How closed providers serve their models is not published: treat any specific claim about it as a guess.
        </Callout>
        <p>Riya takes a plan back to finance: FP8 weights and an FP8 cache, a draft head for the quiet hours, and a full eval run before either ships. The red circle gets smaller. Not half, but smaller, and she can explain every step of why.</p>
      </RealLLM>
    </Lesson>
  )
}
