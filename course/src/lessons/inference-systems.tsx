import { Lesson, Why, Problem, MentalModel, TryIt, Numbers, TheMath, CodeIt, BreakIt, Exercises, CheckYourself, Remember, RealLLM } from '../components/lesson'
import { Callout, DeepDive, Equation, G, Term, ToyVsReal, WhyExists } from '../components/ui'
import { Code } from '../components/Code'
import { Exercise, ExplainBack } from '../components/exercise'
import { BatchingSimLab } from '../interactive/BatchingSimLab'
import { PagedKvLab } from '../interactive/PagedKvLab'
import { QuantizeLab } from '../interactive/QuantizeLab'
import { SpeculativeLab } from '../interactive/SpeculativeLab'
import { DecodeIsMemoryBound } from '../illustrations/DecodeIsMemoryBound'
import { RequestTimeline } from '../illustrations/RequestTimeline'

export default function InferenceSystemsLesson() {
  return (
    <Lesson id="inference-systems">
      <Why>
        <p className="lede">You can run a model for one user. Now a thousand users arrive at once, and two facts about the hardware decide everything.</p>
        <div className="grid-2">
          <div className="card">
            <h4 style={{ fontSize: 17, marginBottom: 6 }}>A GPU serving one user is mostly idle</h4>
            <p>To produce one token, the GPU must read every weight from its memory once. For a 7B model in 16-bit that is about 14 GB per token. The arithmetic for that token takes about 1% of the time the read takes. The rest is waiting for bytes.</p>
          </div>
          <div className="card">
            <h4 style={{ fontSize: 17, marginBottom: 6 }}>Serving 32 users costs barely more than serving one</h4>
            <p>The same read of the weights can serve every sequence in the batch. In this lesson’s simulator a step for 32 users takes 12.5 ms instead of 10.1 ms, and produces 32 tokens instead of 1.</p>
          </div>
        </div>
        <p>So the engineering question is not “how fast is the model?”. It is “how many sequences can share each pass, and what does each user pay for the sharing?”. That is a scheduling and memory-management problem, and you already know most of the ideas: queues, tail latency, bin packing, paging.</p>
        <Callout kind="idea">
          LLM serving has one central trade-off: <b>throughput against latency</b>. Bigger batches make every token cheaper and every user slower. What limits the batch is not slots, it is the memory the <G t="kv-cache">KV caches</G> need. Almost every serving technique either packs the batch better or shrinks the bytes.
        </Callout>
      </Why>

      <Problem>
        <p>From <a href="#/lesson/inference">Inference</a> you know the two phases of one request. <b>Prefill</b> pushes the whole prompt through the model in one parallel pass and fills the KV cache. <b>Decode</b> then produces one token per pass. You also know the cache formula: <span className="mono">2 × layers × kv_heads × head_dim × tokens × bytes</span>, per conversation.</p>
        <p>That was one request. A server has many, arriving at random, with wildly different lengths.</p>
        <WhyExists
          problem="Many requests arrive at random times. Each wants a fast first token and a steady stream after it."
          naive="Serve them one at a time, in order. It is simple and each user gets the full GPU."
          fails="In the simulator, 8 requests per second against one-at-a-time serving gives 98 tokens per second of capacity for about 680 tokens per second of demand. The queue grows without limit: the median wait for a first token is 65 seconds, while the arithmetic units sit about 99% idle."
          idea="Batch: run many sequences through each pass, so one read of the weights produces many tokens."
          tradeoff="Each pass gets a little slower, so each user’s tokens arrive a little later. And every sequence in the batch needs its own KV cache in GPU memory."
        />
        <WhyExists
          problem="Requests in a batch do not finish together. One answer is 10 tokens long, another is 500."
          naive="Static batching, as in training: collect a full batch, run it until the longest sequence finishes, then start the next batch."
          fails="Finished sequences keep their slot and are padded until the longest one ends. New arrivals wait outside for the whole batch. In the simulator, 46,247 decode slot-steps are spent on padding, slot utilisation is 37%, and the median time to first token is 14.7 seconds."
          idea="Make the scheduling decision at every step, not every batch: a finished sequence leaves immediately and a waiting one takes its place."
          tradeoff="The batch is no longer a tidy rectangle, so kernels and memory management must cope with sequences of different lengths. A newcomer’s prefill also interrupts everyone’s decoding."
        />
      </Problem>

      <MentalModel title="One bottleneck, seven consequences">
        <h3>1. Two numbers: bandwidth for decode, arithmetic for prefill</h3>
        <p>A GPU has a large memory that holds the weights, and arithmetic units that do the matrix multiplies. Between them is a link with a fixed speed: the <b>memory bandwidth</b>. On current data-centre GPUs it is on the order of 2 to 5 TB per second.</p>
        <p>In a decode step, every weight is needed once, so all of them cross that link. For one sequence the arithmetic is tiny: one token’s vector times each matrix. The step takes as long as moving the bytes.</p>
        <DecodeIsMemoryBound />
        <p>That gives the most useful back-of-envelope in serving. For <em>one</em> stream:</p>
        <p className="mono" style={{ fontSize: 15 }}>tokens per second ≤ memory bandwidth ÷ bytes of weights = 2 TB/s ÷ 14 GB ≈ 143</p>
        <p>It is an upper bound. It ignores the KV cache reads, kernel overheads and sampling, and it assumes the whole model sits on one GPU. Real single-stream numbers are lower. But it explains the order of magnitude, and it tells you what helps: more bandwidth or fewer bytes. A faster arithmetic unit does nothing.</p>
        <p><b>Prefill is the opposite.</b> A 1,000-token prompt goes through in one pass: the weights are still read once, but now there is 1,000 times more arithmetic per byte read. Prefill is limited by arithmetic. That is why the two phases get separate metrics.</p>
        <Callout kind="dev">
          You have met this in a storage-bound service. When every request costs one disk seek and a microsecond of CPU, a faster CPU changes nothing, and the fix is to serve many requests per seek. Performance engineers call the ratio “arithmetic per byte moved” <b>arithmetic intensity</b>, and the picture of the two limits a <b>roofline</b>. Decode with a small batch sits far under the memory roof. Batching walks it toward the compute roof.
        </Callout>
        <Callout kind="established">
          The figures come from NVIDIA’s published specifications: an A100 80GB has 1,935 to 2,039 GB/s of memory bandwidth depending on the variant and a peak of 312 TFLOP/s in 16-bit, an H100 SXM 3.35 TB/s, an H200 4.8 TB/s. The arithmetic for one token is about 2 FLOPs per parameter (Kaplan et al., 2020, counting non-embedding parameters and ignoring the attention-over-context term). At a usable 150 TFLOP/s, 14 GFLOP takes 0.09 ms. The read takes 7 ms.
        </Callout>

        <h3>2. The metrics you must keep apart</h3>
        <RequestTimeline />
        <Term
          name="Time to first token (TTFT)"
          plain={<>How long the user stares at nothing: time in the queue plus the prefill of their prompt.</>}
          example={<>40 ms waiting for a slot + 48 ms to prefill 500 tokens = a TTFT of 88 ms.</>}
          formal={<>TTFT = first-token time − arrival time. It grows with prompt length (prefill is arithmetic-bound) and explodes when the server is overloaded (queueing).</>}
        />
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>metric</th><th>what it measures</th><th>who cares</th></tr></thead>
            <tbody>
              <tr><td><b>TTFT</b></td><td>queue wait + prefill</td><td>the user: does it feel responsive?</td></tr>
              <tr><td><b>Time per output token (TPOT)</b>, also called inter-token latency</td><td>the gap between streamed tokens: one decode step, plus any stalls</td><td>the user: does the stream keep up with reading?</td></tr>
              <tr><td><b>End-to-end latency</b></td><td>TTFT + (output tokens − 1) × TPOT</td><td>programs that wait for the whole answer, such as agents</td></tr>
              <tr><td><b>Throughput</b></td><td>tokens per second across all users</td><td>whoever pays for the GPUs</td></tr>
              <tr><td><b>Goodput</b></td><td>requests per second that met the latency objective</td><td>whoever answers for the SLO</td></tr>
            </tbody>
          </table>
        </div>
        <p>Report each latency as a distribution: <b>p50 and p99</b>, as for any service. Output lengths are heavily skewed, so averages hide the users who suffer. And a server can show excellent throughput while failing its users: throughput counts tokens, goodput counts requests that were served <em>well</em>.</p>

        <h3>3. Continuous batching: schedule every step, not every batch</h3>
        <Term
          name="Continuous batching (iteration-level scheduling)"
          plain={<>After every decode step, the scheduler looks again. Finished sequences leave. Waiting requests join if there is a slot and memory for them. The batch is a pool that changes all the time.</>}
          example={<>A 10-token answer and a 500-token answer start together. After 10 steps the short one leaves and a queued request takes its slot. With static batching that slot would be padded for 490 more steps.</>}
          formal={<>Introduced as iteration-level scheduling by Orca (Yu et al., OSDI 2022). vLLM and most current engines use it. NVIDIA’s TensorRT-LLM calls it in-flight batching.</>}
        />
        <p>One wrinkle remains. When a new request joins, its prefill is a big arithmetic-bound pass, and while it runs, the sequences that were decoding produce nothing. A long prompt makes everyone’s stream stutter. <b>Chunked prefill</b> fixes that: split the prompt into chunks and run one chunk per step <em>together with</em> the ongoing decodes, which ride along almost for free (SARATHI, Agrawal et al., 2023; Sarathi-Serve, OSDI 2024, calls the result stall-free scheduling). Our simulator does not chunk, so you will see those stalls in the timeline.</p>

        <h3>4. The KV cache is the real capacity limit</h3>
        <p className="mono" style={{ fontSize: 15 }}>GPU memory = weights + KV caches + activations and workspace</p>
        <p>The weights are fixed. Activations are small at inference. Whatever is left holds KV caches, and that decides how many sequences can run at once. So wasting KV memory is wasting batch size, which is wasting throughput.</p>
        <p>The waste comes from not knowing the future. A request may generate up to its <code>max_tokens</code>, so early systems reserved one contiguous slab for the worst case. Most answers are short, so most of the slab was never written. The vLLM paper measured existing systems using only 20.4% to 38.2% of their KV memory for actual token states.</p>
        <Term
          name="PagedAttention"
          plain={<>Cut KV memory into small fixed-size blocks. A sequence gets a new block only when its last one is full, from anywhere in memory. A per-sequence block table maps “my 3rd block” to a physical block.</>}
          example={<>Block size 16. A sequence holding 62 tokens owns 4 blocks, for example physical blocks 5, 6, 7 and 53. Only the last is partly empty: at most 15 token slots wasted per sequence.</>}
          formal={<>Kwon et al., SOSP 2023 (the vLLM paper). Default block size 16 tokens. The attention kernel follows the block table instead of assuming one contiguous array. The paper reports 2 to 4 times the throughput of earlier systems at the same latency.</>}
        />
        <Callout kind="dev">
          This is virtual memory. Blocks are pages, the block table is a page table, a sequence is a process with a contiguous <em>logical</em> address space and scattered physical pages. The same benefits follow. No external fragmentation, because any free block fits. Internal fragmentation limited to the last block. And <b>sharing</b>: two sequences with the same prompt prefix can map the same physical blocks, copy-on-write, exactly like forked processes.
        </Callout>
        <p>That sharing, kept across requests, is <b>prefix caching</b> (vLLM calls it automatic prefix caching, SGLang’s version is RadixAttention). A long system prompt or a shared document is prefilled once, and later requests skip straight to their own suffix. It is what “prompt caching” on an API price list means. It cuts TTFT and prefill cost. It does nothing for decode.</p>
        <p>When blocks run out mid-generation, the scheduler <b>preempts</b> a sequence: it frees its blocks and later recomputes them by prefilling again (vLLM can also swap them to CPU memory). Users see a pause, not an error.</p>

        <h3>5. Quantization: fewer bytes per weight</h3>
        <Term
          name="Quantization"
          plain={<>Store each weight as a small integer plus a shared scale, instead of a 16-bit float. int8 halves the bytes, int4 quarters them.</>}
          example={<>Eight weights with largest magnitude 0.035. Scale = 0.035 ÷ 7 = 0.005. The weight 0.021 is stored as round(0.021 ÷ 0.005) = 4 and read back as 4 × 0.005 = 0.020.</>}
          formal={<>Symmetric absmax quantization: scale = max|w| ÷ (2<sup>bits−1</sup> − 1), q = round(w ÷ scale), ŵ = q × scale. “Weight-only” means activations stay in 16-bit and weights are unpacked on the fly inside the matrix multiply.</>}
        />
        <p>Because decode is limited by bytes moved, weight-only quantization is one of the few techniques that makes serving <b>smaller and faster at once</b>: 4 times fewer bytes is up to 4 times more tokens per second for one stream, and the freed memory holds more KV cache.</p>
        <p>The danger is <b>outliers</b>. One scale must cover the largest value it serves. A single weight 25 times larger than the rest stretches the grid until ordinary weights all round to zero. The fix is more scales: one per row (per output channel), or one per group of 32 to 128 weights. Each scale costs 16 bits, so group-wise int4 really costs about 4.1 to 4.5 bits per weight.</p>
        <p>Real methods improve on plain rounding. <b>LLM.int8()</b> (Dettmers et al., NeurIPS 2022) keeps the few outlier activation dimensions in 16-bit. <b>GPTQ</b> (Frantar et al., ICLR 2023) rounds the weights in order and adjusts the not-yet-rounded ones to compensate, guided by approximate second-order information. <b>AWQ</b> (Lin et al., MLSys 2024) rescales the roughly 1% of weight channels that meet large activations. <b>NF4</b> (QLoRA, Dettmers et al., NeurIPS 2023) spaces its 16 levels to suit bell-shaped weights. The <b>KV cache</b> can be quantized too (vLLM, for one, offers an FP8 cache, and research such as KIVI, ICML 2024, goes as low as 2 bits), which raises the number of sequences that fit.</p>
        <Callout kind="model">
          A widely reported regularity, not a law: 8-bit weights are usually close to lossless, and 4-bit weights cost a small but measurable amount of quality that many applications accept. It varies with the model, its size, the method and above all the task: long reasoning chains and code tend to be more sensitive than short chat. Do not trust a leaderboard for this. Run your own <a href="#/lesson/evals">evals</a> on the quantized model before you ship it.
        </Callout>

        <h3>6. Speculative decoding: spend the idle arithmetic</h3>
        <p>If the arithmetic units are idle during decode, give them something useful. The big model can score 5 positions in one pass in nearly the time it scores 1, exactly like a tiny prefill. It just needs to know which 5 tokens to score.</p>
        <Term
          name="Speculative decoding"
          plain={<>A small, fast draft model guesses the next few tokens. The big target model checks all the guesses in one pass. Guesses are kept or replaced by a rule that makes the final text distributed exactly as if the big model had written every token itself.</>}
          example={<>The draft proposes “sat on the mat”. The target accepts “sat”, “on”, “the”, rejects “mat” and replaces it with “sofa”. Four tokens for one pass of the big model.</>}
          formal={<>Accept a proposed token x with probability min(1, p(x) ÷ q(x)), where p is the target’s probability and q the draft’s. On the first rejection, draw a replacement from max(0, p − q) rescaled to sum to 1, and discard the later guesses. Leviathan et al. (ICML 2023) and Chen et al. (2023) both prove the output distribution equals the target’s.</>}
        />
        <p>It is a latency technique that only works <em>because</em> decode is memory-bound. When it does not help: if the draft is often wrong (little is accepted, and the draft’s own time is wasted), or if the server is already running large batches (the arithmetic is no longer idle, so verifying extra positions is no longer free).</p>

        <h3>7. When the model does not fit on one GPU</h3>
        <Term
          name="Tensor parallelism"
          plain={<>Split every weight matrix across several GPUs. Each GPU holds a slice, computes its part of every layer, and the parts are combined.</>}
          example={<>A 70B model in 16-bit is 140 GB. Across 4 GPUs of 80 GB each, every GPU holds 35 GB of weights and a quarter of each KV cache.</>}
          formal={<>In the Megatron-LM scheme (Shoeybi et al., 2019) the forward pass needs two all-reduce operations per Transformer layer. Each GPU reads only its slice, so the bandwidth-bound step gets faster too.</>}
        />
        <p><b>Tensor parallelism</b> cuts latency as well as memory, but every layer now waits for communication, twice. It needs the fast links inside one server and is rarely stretched across machines.</p>
        <p><b>Pipeline parallelism</b> splits by layers instead: GPU 1 runs layers 1 to 20, GPU 2 runs 21 to 40, and so on. It only passes activations between neighbours, so slower links are fine. But one token still visits every stage in turn, so latency does not improve, and stages idle while waiting for work. Those gaps are called bubbles (GPipe, Huang et al., 2019), and they are filled by keeping several micro-batches in flight.</p>
        <p><b>Data parallelism</b> is the one you know: complete replicas behind a load balancer. It multiplies throughput and does nothing for one request’s latency. A large deployment typically combines them: tensor parallel inside a server so the model fits and responds quickly, then as many replicas as the traffic needs.</p>
      </MentalModel>

      <TryIt title="Four experiments">
        <h3>A. Three schedulers, one stream of requests</h3>
        <p>The simulator serves the same 200 requests with no batching, static batching and continuous batching. Look at the timeline first, then the numbers. Then raise “slots” and watch the curve at the bottom: throughput climbs while every user’s tokens slow down.</p>
        <BatchingSimLab />
        <h3>B. Where the KV memory goes</h3>
        <p>Same queue, same memory, two allocators. Step to around 20 and compare how many sequences each one fits.</p>
        <PagedKvLab />
        <h3>C. Quantize a layer</h3>
        <p>Start with int4 and one scale for the tensor. Then plant the outlier.</p>
        <QuantizeLab />
        <h3>D. Draft and verify</h3>
        <p>Step through a few rounds. Then drag the draft quality and watch the accepted tokens per pass.</p>
        <SpeculativeLab />
      </TryIt>

      <Numbers title="Capacity planning, by hand">
        <p>You are asked: how many users can one GPU serve, and what does a million tokens cost? Here is the whole estimate. Every input is an assumption you should replace with your own.</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>given</th><th>value</th></tr></thead>
            <tbody>
              <tr><td>Model</td><td>8B parameters in 16-bit, the Llama-3 8B shape: 32 layers, 8 K/V heads (GQA), head_dim 128</td></tr>
              <tr><td>GPU</td><td>80 GB of memory, 2 TB/s of bandwidth, 150 TFLOP/s usable. Keep 10% of memory back for activations and workspace.</td></tr>
              <tr><td>Traffic</td><td>prompts of 1,000 tokens, answers of 300 tokens</td></tr>
              <tr><td>Price</td><td>2.00 dollars per GPU-hour. This is an example figure, not a quote.</td></tr>
            </tbody>
          </table>
        </div>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>step</th><th>arithmetic</th><th>result</th></tr></thead>
            <tbody>
              <tr><td>1. Weights</td><td className="mono">8 × 10⁹ × 2 bytes</td><td className="mono"><b>16 GB</b></td></tr>
              <tr><td>2. Memory left for KV caches</td><td className="mono">80 × 0.9 − 16</td><td className="mono"><b>56 GB</b></td></tr>
              <tr><td>3. KV cache per token</td><td className="mono">2 × 32 × 8 × 128 × 2 bytes</td><td className="mono"><b>131,072 bytes</b></td></tr>
              <tr><td>4. KV cache per sequence, at its longest</td><td className="mono">1,300 tokens × 131,072</td><td className="mono"><b>170 MB</b></td></tr>
              <tr><td>5. Sequences that fit</td><td className="mono">56 GB ÷ 170.4 MB</td><td className="mono"><b>328</b></td></tr>
              <tr><td>6. One decode step at that batch: bytes</td><td className="mono">(16 GB + 328 × 1,150 × 131,072) ÷ 2 TB/s</td><td className="mono">8.0 + 24.7 = <b>32.7 ms</b></td></tr>
              <tr><td>… and arithmetic</td><td className="mono">328 × 2 × 8 × 10⁹ FLOP ÷ 150 TFLOP/s</td><td className="mono"><b>35.0 ms</b></td></tr>
              <tr><td>… so the step takes</td><td className="mono">max(32.7, 35.0) + 3 ms overhead</td><td className="mono"><b>38.0 ms</b></td></tr>
              <tr><td>7. Prefill of one prompt</td><td className="mono">1,000 × 16 GFLOP ÷ 150 TFLOP/s + 3 ms</td><td className="mono"><b>109.7 ms</b></td></tr>
              <tr><td>8. GPU time per request</td><td className="mono">109.7 + 300 steps × 38.0 ms ÷ 328 sharing</td><td className="mono">109.7 + 34.7 = <b>144.4 ms</b></td></tr>
              <tr><td>9. Capacity</td><td className="mono">1,000 ÷ 144.4</td><td className="mono"><b>6.9 requests/s = 2,077 output tokens/s</b></td></tr>
              <tr><td>10. Cost</td><td className="mono">2.00 ÷ (2,077 × 3,600 ÷ 10⁶)</td><td className="mono"><b>0.27 dollars per million output tokens</b></td></tr>
            </tbody>
          </table>
        </div>
        <p>Three things to notice. In step 6 the average sequence holds 1,150 tokens (the prompt plus half the answer), and at 328 sequences the KV reads are three times the weight read: at large batch, <em>the cache</em> is what crosses the memory link. The two limits nearly meet (32.7 against 35.0 ms): this batch sits at the corner of the roofline, where neither memory traffic nor arithmetic is idle. And in step 8, three quarters of the GPU time per request is <b>prefill</b>. One stream alone would get at most 2 TB/s ÷ 16 GB = 125 tokens per second, so batching bought about 17 times the throughput.</p>
        <Callout kind="warn">
          This is a ceiling from a napkin. It ignores the latency objective, kernel efficiency at this batch shape and everything else a benchmark would reveal. Use it to see which resource binds and to sanity-check a vendor’s numbers, then measure your own workload before you order hardware.
        </Callout>
        <DeepDive title="Now add the SLO: goodput with Little’s law">
          <p>At the ceiling, decode steps only get the GPU 24% of the time (34.7 of every 144.4 ms), because prefills keep interrupting. So each user sees a token every 38.0 ÷ 0.24 = <b>158 ms</b>, not every 38 ms. If your objective is 50 ms per token, the ceiling is useless.</p>
          <p>Find the load that meets it. Let λ be requests per second. Prefill takes a fraction 0.1097 × λ of the GPU. Little’s law gives the number of sequences in decode: L = λ × time in decode = λ × 300 × 0.050 s = 15 λ. At that batch the step is memory-bound: 3 + 8.0 + 0.0754 × 15 λ ms. Setting step ÷ (1 − 0.1097 λ) = 50 ms and solving gives <b>λ = 5.9 requests per second with 88 sequences in flight</b>, a 17.7 ms step, 1,769 tokens per second and 0.31 dollars per million.</p>
          <p>So the SLO costs about 15% of the capacity here, and the memory could hold 328 sequences but the objective only lets you run 88. Chunked prefill, or separate GPU pools for prefill and decode, exist to win that gap back.</p>
        </DeepDive>
      </Numbers>

      <TheMath>
        <p>Four small formulas carry the lesson. The first is the single-stream bound:</p>
        <Equation
          label="Tokens per second for one stream is at most memory bandwidth divided by the bytes of weights"
          symbols={[
            ['bandwidth', 'bytes per second between GPU memory and the arithmetic units'],
            ['weight bytes', 'parameters × bytes per parameter: 2 for 16-bit, about 0.52 for int4 with group scales'],
            ['≤', 'an upper bound: KV cache reads and overheads only make it slower'],
          ]}
        >
          tokens/s (one stream) ≤ bandwidth ÷ weight bytes
        </Equation>
        <p>The second is the step time for a batch of B sequences. It is the roofline idea: a step lasts as long as the slower of moving bytes and doing arithmetic.</p>
        <Equation
          label="Step time equals overhead plus the maximum of bytes moved over bandwidth and FLOPs needed over FLOPs per second"
          symbols={[
            ['W', 'bytes of weights: read once per step, shared by the whole batch'],
            [<>KV<sub>i</sub></>, 'bytes of KV cache of sequence i: read once per step, not shared'],
            ['B', 'sequences in the batch'],
            ['2N', 'FLOPs to push one token through a model with N parameters'],
            ['F', 'usable FLOPs per second'],
          ]}
        >
          t<sub>step</sub> ≈ overhead + max( (W + Σ<sub>i</sub> KV<sub>i</sub>) ÷ bandwidth , B × 2N ÷ F )
        </Equation>
        <p>With B small the left term wins and hardly depends on B: batching is nearly free. As B grows, either ΣKV or B × 2N catches up with W, and from there each extra sequence costs real time. Throughput is B ÷ t<sub>step</sub>, so it rises steeply and then flattens. Per-user latency is t<sub>step</sub>, so it only ever gets worse.</p>
        <p>The third is the speculative acceptance rule, with p the target’s probability for the proposed token and q the draft’s:</p>
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
        <DeepDive title="The critical batch size, in one line">
          <p>Ignore KV reads for a moment. The two terms are equal when B × 2N ÷ F = (bytes per parameter × N) ÷ bandwidth, so B* = (F ÷ bandwidth) × (bytes per parameter ÷ 2). The model size cancels. With 150 TFLOP/s and 2 TB/s in 16-bit, B* = 75: below about 75 sequences the GPU is waiting for bytes, above it for arithmetic. The simulator’s constants give 7 ÷ 0.09 = 78. KV reads move that point further out, and with long contexts the cache reads dominate before arithmetic ever does.</p>
        </DeepDive>
      </TheMath>

      <CodeIt>
        <h3>The simulator</h3>
        <p><code>batching_sim.py</code> runs no neural network. It simulates the scheduler around one. Its only physics is this cost model, the roofline formula with illustrative constants:</p>
        <Code source="phase6-engineering/batching_sim.py" title="the cost of one decode step, and of a prefill">{`
COST = {"weight_read_ms": 7.0,          # 14 GB / 2 TB/s, shared by the whole batch
        "kv_read_ms_per_tok": 0.00026,  # 0.5 MiB / 2 TB/s, per cached token
        "compute_ms_per_tok": 0.09,     # 2 x 7e9 FLOP / 150 TFLOP/s
        "overhead_ms": 3.0}

def decode_step_ms(n_seqs, ctx_tokens, cost=COST):
    memory = cost["weight_read_ms"] + cost["kv_read_ms_per_tok"] * ctx_tokens
    compute = cost["compute_ms_per_tok"] * n_seqs
    return cost["overhead_ms"] + max(memory, compute)

def prefill_ms(prompt_tokens, cost=COST):
    compute = cost["compute_ms_per_tok"] * prompt_tokens
    return cost["overhead_ms"] + max(cost["weight_read_ms"], compute)
`}</Code>
        <p>Static batching is a loop over rectangles. Everyone is padded to the longest prompt, and the batch runs for as many steps as the longest answer:</p>
        <Code source="phase6-engineering/batching_sim.py" title="static batching (simplified)">{`
pad_prompt = max(r["prompt"] for r in batch)     # prompts are padded to the longest
steps = max(r["out"] for r in batch)             # and everyone waits for the longest answer
for k in range(2, steps + 1):
    dur = decode_step_ms(n, n * (pad_prompt + k - 1), cost)
    live = sum(1 for r in batch if r["generated"] < r["out"])
    st["padded_steps"] += n - live               # finished, still occupying the slot
`}</Code>
        <p>Continuous batching makes its decision at the top of every iteration. Admission needs a free slot <em>and</em> free KV blocks:</p>
        <Code source="phase6-engineering/batching_sim.py" title="continuous batching: admission, every iteration (simplified)">{`
while waiting and len(running) + len(admitted) < max_batch:
    r = waiting[0]
    if cfg["kv_mode"] == "reserved":
        need = _blocks(r["prompt"] + MAX_NEW, bs)              # worst case, up front
    else:
        need = _blocks(r["prompt"] + r["generated"] + 1, bs)   # what it holds right now
    if st["free"] - need < headroom:
        break
    st["free"] -= need
    admitted.append(waiting.pop(0))
`}</Code>
        <p>If anything was admitted, the iteration is a prefill and the running sequences stall. Otherwise it is a decode step for everyone. Finished sequences free their slot and blocks at once. Running the file prints:</p>
        <Code lang="output" title="python phase6-engineering/batching_sim.py (200 requests, 8 arrivals per second)">{`
  policy     batch   tok/s  TTFT p50  TTFT p99  TPOT p50 TPOT p99   slots   padded  in SLO
                               (ms)      (ms)      (ms)     (ms)    used    steps
  none           1        98     65210    155285     10.1     10.2     100%        0      0%
  static        16       280     14675     35349     12.2     13.4      37%    46247      2%
  continuous    16       612        31       106     13.0     18.3      42%        0    100%
`}</Code>
        <p>Same GPU, same requests, same 16 slots. Continuous batching keeps up with the traffic and the other two fall behind, so their queues grow for as long as the traffic lasts. Notice that TPOT barely differs: the damage of bad scheduling lands on TTFT. And with 64 slots but only 16,384 tokens of KV memory, on a saturated server:</p>
        <Code lang="output" title="section 4 of the same run">{`
  kv mode    slots  mean running  peak   tok/s  TTFT p50  KV wasted  preemptions
  reserved      64          18.4    23    1084      8488        57%            0
  paged         64          38.0    59    1512      3356         2%            0
`}</Code>
        <p>Same memory, twice the sequences in flight. The vLLM paper measured even more waste than our 57% in the systems of its day: 62% to 80% of KV memory held no token state.</p>

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
      </CodeIt>

      <BreakIt>
        <p>Predict first, then check.</p>
        <ul>
          <li><b>Lab A: set the output skew to 0.</b> Every answer is now the same length. How much of static batching’s padding disappears? What is still wrong with it? (Padding falls sharply, but arrivals still wait outside for the whole batch, so TTFT stays bad.)</li>
          <li><b>Lab A: 1 arrival per second.</b> Throughput is the same for all three schedulers. Why? (Throughput cannot exceed demand. Look at TTFT instead: static batching waits up to 2 seconds for a batch that never fills.)</li>
          <li><b>Lab A: 64 slots, KV budget 16,384, then switch to “reserve up front”.</b> The curve at the bottom flattens early: more slots stop helping because memory, not slots, limits the batch.</li>
          <li><b>Lab A: KV budget 2,048 with paged blocks and 40 arrivals per second.</b> Preemptions appear. Tokens are never lost, but work is redone.</li>
          <li><b>Lab B: raise the max_tokens cap to 256.</b> Contiguous reservation gets worse although no request changed. Paged allocation does not move at all.</li>
          <li><b>Lab C: int8, one scale for the tensor, plant the outlier.</b> Still usable. Now int4. Count the weights that rounded to zero.</li>
          <li><b>Lab D: make the draft completely wrong.</b> Tokens per pass approach 1 but never go below it, and the text is still the target’s. What did you lose? (The time spent running the draft.)</li>
          <li><b>In <code>speculative_demo.py</code></b>, make the rejection branch draw from <code>p</code> instead of <code>residual(p, q)</code> and run <code>pytest tests/test_engineering_serving.py</code>. Which test catches it?</li>
        </ul>
      </BreakIt>

      <Exercises>
        <Exercise
          id="inference-systems-single-stream"
          type="calculate"
          title="The single-stream ceiling"
          answer={{ value: 38.5, tolerance: 1 }}
          answerLabel="tokens per second, upper bound"
          hints={[
            'Every generated token reads every weight once. How many bytes is that?',
            '13 billion parameters × 2 bytes = 26 GB per token.',
            '1 TB/s ÷ 26 GB = 1,000 ÷ 26.',
          ]}
          solution={<><p>26 GB must cross the memory link for each token. 1,000 GB/s ÷ 26 GB ≈ <b>38 tokens per second</b>, and real numbers will be lower.</p><p>Notice what is not in the formula: FLOPs. A chip with twice the arithmetic and the same bandwidth would produce the same 38. Quantizing to int4 (about 6.7 GB) raises the ceiling to about 149.</p></>}
        >
          <p>A 13B-parameter model in 16-bit runs on an accelerator with 1 TB/s of memory bandwidth. What is the upper bound on tokens per second for a single stream?</p>
        </Exercise>

        <Exercise
          id="inference-systems-padding"
          type="predict"
          title="Count the padding"
          answer={{ value: 540, tolerance: 0 }}
          answerLabel="wasted slot-steps"
          hints={[
            'The batch runs until its longest sequence finishes. How many steps is that?',
            '200 steps. A sequence that finished after 10 steps holds its slot, padded, for 190 more.',
            '(200 − 10) + (200 − 20) + (200 − 30) + (200 − 200).',
          ]}
          solution={<><p>190 + 180 + 170 + 0 = <b>540</b> wasted slot-steps out of 4 × 200 = 800. Two thirds of the batch’s capacity produced nothing.</p><p>One long answer is enough. That is why the skew of output lengths matters more than their average, and why continuous batching, which refills those three slots after 10, 20 and 30 steps, wins by so much on real traffic.</p></>}
        >
          <p>A static batch holds 4 sequences that will generate 10, 20, 30 and 200 tokens. How many slot-steps are spent on padding before the batch ends?</p>
        </Exercise>

        <Exercise
          id="inference-systems-capacity"
          type="calculate"
          title="Capacity without GQA"
          answer={{ value: 82, tolerance: 1 }}
          answerLabel="concurrent sequences"
          hints={[
            'Only step 3 of the worked example changes: the K/V head count goes from 8 to 32.',
            'KV per token = 2 × 32 × 32 × 128 × 2 = 524,288 bytes. Per sequence at 1,300 tokens: 681.6 MB.',
            '56 GB ÷ 681.6 MB, rounded down.',
          ]}
          solution={<><p>KV per token is 4 times larger: 524,288 bytes. A 1,300-token sequence needs 681.6 MB. 56 × 10⁹ ÷ 681.6 × 10⁶ = 82.2, so <b>82</b> sequences instead of 328.</p><p>GQA did not make the model faster for one user. It made the batch four times larger on the same GPU, which is where serving cost is decided. The opposite move also works: int4 weights (about 4.1 GB) leave 67.9 GB for caches, and 398 sequences fit.</p></>}
        >
          <p>Redo step 5 of the capacity plan for the same model <em>without</em> grouped-query attention: 32 K/V heads instead of 8. Everything else is unchanged. How many sequences fit?</p>
        </Exercise>

        <Exercise
          id="inference-systems-outlier"
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
          id="inference-systems-spec-debug"
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
          id="inference-systems-gqa-sim"
          type="modify"
          title="Give the simulated model GQA"
          hints={[
            'GQA with 8 of 32 K/V heads makes the cache 4 times smaller per token. Two constants in batching_sim.py depend on that.',
            'Divide COST["kv_read_ms_per_tok"] by 4. In section 4 of main(), the same 8 GiB now holds 65,536 tokens instead of 16,384.',
            'Pass a modified cost dict as the third argument of simulate(), and kv_budget_tokens=65536.',
          ]}
          solution={<><p>With the 4 times smaller cache, section 4 prints a mean of 47.1 running sequences and 1,814 tokens per second for <em>both</em> KV modes, and the peak is 64: the slot limit, not memory, is now what binds, so paging no longer matters at this setting. Raise <code>max_batch</code> and the gap returns. At 128 slots with ample memory, throughput rises from 1,869 to 2,070 tokens per second and TPOT p50 falls from 49.2 to 43.7 ms, because each sequence drags fewer cache bytes across the link.</p><p>The point: which resource binds depends on the configuration. A technique that doubles throughput on one setup can do nothing on another. Find the binding constraint first.</p></>}
        >
          <p>The simulator models a 7B model with plain multi-head attention: 0.5 MiB of cache per token. Change it to a GQA model with a quarter of the K/V heads, in the same 8 GiB of KV memory. Predict what happens to the reserved-versus-paged gap in section 4, then run it.</p>
        </Exercise>

        <ExplainBack
          id="inference-systems-explain"
          prompt="A product manager asks: “If the GPU can do 32 users for almost the price of one, why not run 1,000 users per GPU and cut our bill by 30?” Explain what stops you, using the two limits and the metrics from this lesson."
          modelAnswer={<p>Batching is nearly free only while the step is dominated by reading the weights, which every sequence shares. Two things end that. First, memory: every sequence needs its own KV cache, and once the caches fill the GPU no more sequences fit, however many slots we configure. Paged allocation, GQA and quantization push that limit out, they do not remove it. Second, time: each sequence adds its own cache reads and arithmetic to every step, so past some batch size the step gets slower in proportion, throughput flattens, and every user’s time per token keeps rising. Prefills for new arrivals also interrupt everyone’s stream. So throughput is bought with latency, and what we actually sell is goodput: requests that meet the TTFT and per-token objectives at p99. The right batch size is the largest one that still meets them, and we find it by measuring our own traffic.</p>}
        />
      </Exercises>

      <CheckYourself
        questions={[
          {
            q: 'A team replaces its GPUs with ones that have twice the arithmetic throughput and the same memory bandwidth. Single-user decode speed barely changes. Why?',
            options: [
              'Decoding one sequence is limited by reading the weights from memory for every token, not by arithmetic',
              'The new GPUs need a driver update before the extra arithmetic units can be used by the model',
              'Sampling from the vocabulary is the bottleneck, and sampling always runs on the CPU in any case',
              'Decode speed is fixed by the tokenizer, which runs at the same speed whatever hardware is used',
            ],
            answer: 0,
            explain: 'One token needs all the weights once and very little arithmetic. Tokens per second for one stream is bounded by bandwidth divided by weight bytes. Extra arithmetic helps prefill and large batches, not a single stream.',
          },
          {
            q: 'Why can a server add a second, third and tenth sequence to a decode batch at almost no cost in step time?',
            options: [
              'Because the sequences are averaged into a single vector before they go through the model',
              'Because one read of the weights serves every sequence in the batch, and arithmetic was idle',
              'Because the KV cache is shared between users, so extra sequences need no additional memory',
              'Because the GPU runs each sequence on a separate copy of the weights held in spare memory',
            ],
            answer: 1,
            explain: 'The expensive part of a step, moving the weights, is paid once per step. Each extra sequence adds only its own arithmetic and its own cache reads. KV caches are not shared, which is exactly what ends the free lunch.',
          },
          {
            q: 'Under load, static batching and continuous batching show almost the same time per output token, but time to first token differs by a factor of several hundred. What explains it?',
            options: [
              'Continuous batching uses a faster attention kernel for the first token of each sequence',
              'Static batching computes the first token twice, once for padding and once for the real answer',
              'With static batching, arrivals wait for a whole batch to end and finished sequences pad their slots; decode steps cost the same',
              'Continuous batching skips prefill for short prompts, which removes most of the first-token delay',
            ],
            answer: 2,
            explain: 'A decode step costs about the same under both. The difference is queueing: static batching holds slots hostage until the longest sequence ends, so new requests wait outside. That shows up in TTFT and in goodput, not in TPOT.',
          },
          {
            q: 'What does PagedAttention change, and what does it leave alone?',
            options: [
              'It compresses keys and values to 4 bits, which changes the attention output very slightly',
              'It moves old KV blocks to disk, so that context length is no longer limited by GPU memory',
              'It approximates attention over distant blocks, trading a little quality for a lot of memory',
              'It changes where KV blocks live in memory and how they are found; the attention result is the same',
            ],
            answer: 3,
            explain: 'It is memory management: fixed-size blocks, a block table per sequence, blocks allocated on demand and shareable. The attention computed over those blocks is the ordinary one. The gain is more sequences per GPU.',
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
        ]}
      />

      <Remember
        items={[
          <><b>Decode is memory-bound, prefill is compute-bound.</b> One stream gets at most <span className="mono">bandwidth ÷ weight bytes</span> tokens per second: about 143 for 14 GB on 2 TB/s. The arithmetic units are nearly idle.</>,
          <><b>Batching is nearly free</b> because the weight read is shared, until KV reads or arithmetic catch up, or KV memory runs out. Bigger batch: cheaper tokens, slower users. That is the trade-off.</>,
          <>Keep the metrics apart: <b>TTFT</b> (queue + prefill), <b>TPOT</b> (decode step + stalls), end-to-end, throughput, and <b>goodput</b> under an SLO, at p50 and p99.</>,
          <><b>Continuous batching</b> reschedules at every step, so nobody pads and nobody waits for a whole batch. <b>PagedAttention</b> allocates the KV cache in blocks on demand, like virtual memory, so twice as many sequences fit and prefixes can be shared.</>,
          <><b>Fewer bytes, or more tokens per pass.</b> Weight-only int8 and int4 shrink memory and speed up decode, with group-wise scales to survive outliers: measure the quality with your evals. Speculative decoding spends the idle arithmetic and keeps the target’s distribution exactly.</>,
        ]}
      />

      <RealLLM>
        <ToyVsReal
          toy={<ul><li>A scheduler simulator with a four-constant cost model: no GPU, no model</li><li>Prefill stalls every running sequence; no chunking</li><li>Block allocator with counts and a 64-block picture</li><li>Round-to-nearest quantization of one random matrix</li><li>Speculative decoding between two 6-token Markov chains</li></ul>}
          real={<ul><li>Engines measured on real traffic, with fused kernels for ragged batches and paged caches</li><li>Chunked prefill, prefix caches, priorities, sometimes separate prefill and decode pools</li><li>Tens of thousands of blocks, copy-on-write sharing, swapping to CPU memory</li><li>GPTQ, AWQ, FP8 and similar, with kernels that compute directly on packed weights</li><li>Draft models, extra prediction heads or n-gram lookups as the proposer, verified in batches</li></ul>}
        />
        <p>The engines you will meet, one line each:</p>
        <div className="table-scroll">
          <table className="plain">
            <tbody>
              <tr><td><b>vLLM</b></td><td>Open-source serving engine that began at UC Berkeley, built around PagedAttention, with continuous batching, chunked prefill and prefix caching.</td></tr>
              <tr><td><b>TensorRT-LLM</b></td><td>NVIDIA’s open-source library for optimised LLM inference on NVIDIA GPUs. Its name for continuous batching is in-flight batching.</td></tr>
              <tr><td><b>SGLang</b></td><td>Open-source serving framework whose RadixAttention reuses KV caches across requests that share prefixes, plus fast structured output.</td></tr>
              <tr><td><b>llama.cpp</b></td><td>LLM inference in plain C/C++ with the GGUF file format and integer quantization from 1.5 to 8 bits. It runs on CPUs, Apple silicon and consumer GPUs.</td></tr>
            </tbody>
          </table>
        </div>
        <Callout kind="established">
          The mechanisms in this lesson are published and open: iteration-level scheduling (Orca, OSDI 2022), PagedAttention (vLLM, SOSP 2023), chunked prefill (Sarathi-Serve, OSDI 2024), speculative decoding with its exactness proof (Leviathan et al., ICML 2023; Chen et al., 2023), and the quantization methods named above. That decode is limited by memory bandwidth and prefill by arithmetic follows from counting bytes and FLOPs, and you can verify it on any GPU by watching tokens per second as you change batch size.
        </Callout>
        <Callout kind="model">
          Our cost model is a roofline with four constants. Real step times also depend on kernel efficiency at each batch shape, attention cost growing with context, interconnect in multi-GPU setups, and scheduler overhead. Use the model to reason about <em>which</em> resource binds. Use a benchmark of your own traffic for the actual numbers.
        </Callout>
        <Callout kind="research">
          How to split prefill and decode across machines, how to schedule for goodput rather than throughput, how far KV caches can be compressed or evicted without hurting long-context quality, and how much quality 4-bit and lower precision really costs on hard tasks are all active areas. How closed providers serve their models is not published: treat any specific claim about it as a guess.
        </Callout>
      </RealLLM>
    </Lesson>
  )
}
