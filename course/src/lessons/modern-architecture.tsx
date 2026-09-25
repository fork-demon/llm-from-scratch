import type { ReactNode } from 'react'
import { Lesson, Why, Problem, MentalModel, TryIt, Numbers, TheMath, CodeIt, BreakIt, Exercises, CheckYourself, Remember, RealLLM } from '../components/lesson'
import { Callout, DeepDive, Equation, Flow, G, Term, ToyVsReal, WhyExists } from '../components/ui'
import { Code } from '../components/Code'
import { Exercise, ExplainBack } from '../components/exercise'
import { RopeLab } from '../interactive/RopeLab'
import { GqaCalculator } from '../interactive/GqaCalculator'
import { NormCompare } from '../interactive/NormCompare'

/** Problem, idea, trade-off: the three lines every repair in this lesson is told with. */
function Fix({ problem, idea, tradeoff }: { problem: ReactNode; idea: ReactNode; tradeoff: ReactNode }) {
  const rows: [string, ReactNode][] = [['Problem', problem], ['Idea', idea], ['Trade-off', tradeoff]]
  return (
    <div className="card" style={{ margin: '14px 0' }}>
      {rows.map(([label, body]) => (
        <p key={label} style={{ margin: '6px 0' }}><b className="acc">{label}.</b> {body}</p>
      ))}
    </div>
  )
}

export default function ModernArchitectureLesson() {
  return (
    <Lesson id="modern-architecture">
      <Why>
        <p className="lede">Friday afternoon. Kabir plugs his laptop into the big screen in the meeting room and opens a plain text file. “This is the config of DeepSeek-V3,” he says. “An open-weight model. Many 2025 and 2026 models borrow its design.”</p>
        <Code lang="output" title="config.json (a few lines of it)">{`
"num_hidden_layers": 61,
"hidden_size": 7168,
"rms_norm_eps": 1e-06,
"hidden_act": "silu",
"rope_scaling": { "type": "yarn", ... },
"kv_lora_rank": 512,
"n_routed_experts": 256,
"n_shared_experts": 1,
"num_experts_per_tok": 8,
`}</Code>
        <p>Riya reads it slowly. Layers: she built those. <code>hidden_size</code>: that is her <code>n_embd</code>, only 56 times wider. Then the lines get strange. RMS norm? Experts? Why only 8 of 256?</p>
        <p>Dev leans in from the doorway. “So it’s a totally new architecture. Way more parameters, way smarter.”</p>
        <p>Kabir smiles. “Let’s see how much of it Riya already wrote.”</p>
        <Callout kind="idea">
          None of these changes is a new architecture. Each one is a <b>repair</b>: a specific, nameable problem with the GPT-2 design, and a small fix. If you know the problem, the fix stays in your head. This lesson tells every repair the same way: the problem, the idea, and what it costs.
        </Callout>
      </Why>

      <Problem title="What goes wrong when you scale our tiny GPT">
        <p>Our tiny GPT has 4 layers, 128 numbers per token and a 64-token window. Now imagine turning every dial up a thousand times and serving it to a million users. Three kinds of pain appear.</p>
        <div className="grid-3">
          <div className="card">
            <h4 style={{ fontSize: 17, marginBottom: 6 }}>Position</h4>
            <p>Our model looks up a learned vector for position 0, 1, … 63. There is no row for position 64. And nothing in a lookup table says that positions 10 and 12 are “two apart”.</p>
          </div>
          <div className="card">
            <h4 style={{ fontSize: 17, marginBottom: 6 }}>Memory</h4>
            <p>The <G t="kv-cache">KV cache</G> from <a href="#/lesson/inference">Inference</a> grows with every token, every layer and every head. At long contexts it can rival the weights themselves.</p>
          </div>
          <div className="card">
            <h4 style={{ fontSize: 17, marginBottom: 6 }}>Cost per quality</h4>
            <p>When a training run costs millions, a layer that is 5% cheaper, or an MLP that is slightly better per parameter, is worth adopting.</p>
          </div>
        </div>
        <WhyExists
          problem="GPT-2’s design works, but it has a hard length limit, a KV cache that explodes at long context, and a few parts that do more work than they need to."
          naive="Keep the design exactly as it is and only make it bigger."
          fails="A position table cannot be read beyond its last row. The cache for a 70-billion-parameter model with plain multi-head attention at 128k tokens would need hundreds of gigabytes for a single conversation."
          idea="Keep the skeleton (blocks of attention + MLP with residuals). Swap individual parts for ones that fix a specific problem."
          tradeoff="Each swap has its own cost, and several were adopted mainly because experiments showed they work, not because theory predicted it. We will say which."
        />
      </Problem>

      <MentalModel title="Same skeleton, swapped parts">
        <p>Here is the whole lesson on one screen. The left column is what you built. The right column is what you would find in most open models today.</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>Part</th><th>Our tiny GPT (GPT-2 style)</th><th>Typical open model, 2024 to 2026</th><th>Fixes</th></tr></thead>
            <tbody>
              <tr><td>Normalisation</td><td>LayerNorm</td><td>RMSNorm (root mean square norm), often also on q and k (QK-norm)</td><td>cost, stability</td></tr>
              <tr><td>Position</td><td>learned table, added to embeddings</td><td>RoPE (rotary position embedding): rotate <span className="q">q</span> and <span className="k">k</span></td><td>position</td></tr>
              <tr><td>MLP</td><td>2 matrices + GELU</td><td>3 matrices, gated (SwiGLU)</td><td>quality per parameter</td></tr>
              <tr><td>Attention heads</td><td>one K/V head per query head</td><td>shared K/V heads (GQA), a compressed cache (MLA), or a sliding window in some layers</td><td>memory</td></tr>
              <tr><td>Attention execution</td><td>build the full T×T table</td><td>FlashAttention: same result, tiled</td><td>memory, speed</td></tr>
              <tr><td>Context window</td><td>64 (GPT-2: 1,024)</td><td>tens of thousands to over a hundred thousand tokens</td><td>usefulness</td></tr>
              <tr><td>Size</td><td>0.8M parameters, all used</td><td>billions; dense or, increasingly, mixture-of-experts (MoE)</td><td>capability per unit of compute</td></tr>
            </tbody>
          </table>
        </div>
        <p>Dev’s “way more parameters, way smarter” hides the most interesting row. In the biggest open models, most parameters sit idle for any one token. We will get there in part 8.</p>
        <Callout kind="analogy">
          Amma’s old pressure cooker and her new one work the same way: seal the lid, build steam, cook faster. But the new one has a better gasket, a safety valve that actually works, and a whistle she can hear from the balcony. Each change fixed one known annoyance.
          <br /><br />
          Where the analogy stops: cooker parts were redesigned from physics. Several of the swaps below were found by trying things and measuring the loss. We will mark those honestly.
        </Callout>
        <Callout kind="established">Everything in this lesson is taken from published papers and openly released model code and configs (Llama, Mistral, Qwen, Gemma, DeepSeek, Kimi, gpt-oss and similar families). The internals of closed commercial models are not public, so this lesson makes no claims about them.</Callout>
      </MentalModel>

      <TryIt title="The changes, one at a time">
        <h3>1. RMSNorm: a cheaper LayerNorm</h3>
        <p>In <a href="#/lesson/transformer-block">The Transformer block</a>, <G t="layernorm">LayerNorm</G> kept each token’s vector in a stable range. It subtracts the mean, divides by the spread, then applies a learned gain and bias. It runs twice per block, on every token.</p>
        <p>Which of those steps actually matters? Try it on four numbers.</p>
        <NormCompare />
        <Fix
          problem="LayerNorm computes two statistics (mean and spread) and carries two learned vectors (gain and bias), twice per block. Is all of that needed?"
          idea={<>Drop the mean subtraction and the bias. Divide the vector by its root-mean-square size, then multiply by a learned gain: <span className="mono">x / √(mean(x²)) × g</span>.</>}
          tradeoff="The output no longer has mean 0. Published comparisons found no consistent loss in quality from this. The saving is real but modest: normalisation is a small share of total compute."
        />
        <DeepDive title="RMSNorm: why it helps, where it appears, and QK-norm">
          <p><b>Why it helps.</b> Experiments (Zhang and Sennrich, 2019) found that the rescaling is what stabilises training; re-centring adds little. RMSNorm does less arithmetic and has fewer parameters for roughly the same result.</p>
          <p><b>Where it appears.</b> Llama, Mistral, Qwen, Gemma, DeepSeek and most open models since 2023. It was used earlier in T5.</p>
          <p><b>QK-norm.</b> A newer, related habit: apply an RMSNorm to each <span className="q">query</span> and <span className="k">key</span> vector just before the dot product. Very large training runs sometimes suffer loss spikes when attention scores grow huge and softmax saturates. Normalising q and k keeps the size of their dot product in check, which makes training more stable. OLMo 2, Qwen3 and Gemma 3 use it. It adds a tiny amount of work and is otherwise invisible: the attention formula stays the same.</p>
        </DeepDive>

        <h3>2. RoPE: position as rotation</h3>
        <p>Our tiny GPT adds a learned position vector to each token embedding: <code>x = tok_emb + pos_emb</code>. That has two problems.</p>
        <ul>
          <li><b>No row, no position.</b> The table has 64 rows. Position 64 does not exist. A model trained with 1,024 rows cannot read token 1,025.</li>
          <li><b>Distance is not built in.</b> Language cares mostly about <em>relative</em> position: “the adjective right before this noun”. A table of unrelated vectors forces the model to learn “two apart” separately for positions (3, 5), (40, 42), (900, 902)…</li>
        </ul>
        <p>The idea of <G t="rope">RoPE</G> (rotary position embedding) is to add nothing at all. Instead, just before the <span className="q">query</span>·<span className="k">key</span> dot product, <b>rotate</b> each vector by an angle proportional to its position.</p>
        <p>You know from <a href="#/lesson/vectors">lesson 1.1</a> that a dot product depends on the angle <em>between</em> two vectors. If both are rotated, only the difference in rotation survives.</p>
        <RopeLab />
        <Fix
          problem="A learned position table stops at its last row and does not express how far apart two tokens are."
          idea={<>Rotate <span className="q">q</span> and <span className="k">k</span> by (position × θ) inside every attention layer. The score <span className="q">q</span>·<span className="k">k</span> then depends only on the offset between the two positions.</>}
          tradeoff="It must be applied in every attention layer, to q and k, at the right positions (easy to get wrong with a KV cache). And “no table” does not mean “any length works”: models still degrade beyond the lengths they were trained on."
        />
        <DeepDive title="RoPE: why it helps and where it appears">
          <p><b>Why it helps.</b> Relative distance is built into the attention score, the same way at every position. There is no table to run out of, and no extra parameters. Values are not rotated: position only influences who attends to whom.</p>
          <p><b>Where it appears.</b> Llama, Mistral, Qwen, Gemma, DeepSeek, GPT-NeoX, PaLM: the default in open models. Introduced in the RoFormer paper (Su et al., 2021). Part 7 below shows how its rotation speeds are adjusted to stretch the context window.</p>
        </DeepDive>

        <h3>3. SwiGLU: a gated MLP</h3>
        <p>Our <G t="ffn">feed-forward layer</G> expands each token to 4× its width, applies GELU (a smooth ReLU), and projects back. Two matrices.</p>
        <p>A gated MLP uses three. Two of them read the input side by side. One result goes through a smooth switch, then the two are multiplied together, number by number.</p>
        <p>One path says <em>what</em> to write. The other says <em>how much of it to let through</em>. That is what “gated” means, and the G in SwiGLU.</p>
        <p>The switch itself is <b>Swish</b>, also written SiLU: the function z × sigmoid(z), a smooth version of the ReLU hinge you met in <a href="#/lesson/neurons">Neurons and layers</a>. It gives the Swi in the name.</p>
        <Fix
          problem="Most of a dense Transformer’s parameters sit in the MLPs. Any MLP design that gives lower loss for the same parameter count is worth a lot at scale."
          idea={<>Replace <span className="mono">W₂ · gelu(W₁x)</span> with <span className="mono">W₂ · ( swish(W₁x) ⊙ W₃x )</span>, where ⊙ multiplies matching entries. To keep the parameter count equal, the hidden width shrinks from 4d to about 8d/3.</>}
          tradeoff="Three matrix multiplies instead of two, and one more matrix to split across GPUs. And no satisfying theory: see the note below."
        />
        <DeepDive title="SwiGLU: why it helps and where it appears">
          <p><b>Why it helps.</b> In controlled comparisons (Shazeer, 2020) gated variants reached lower loss than ReLU or GELU MLPs with the same number of parameters and compute.</p>
          <p><b>Where it appears.</b> Llama, Mistral, Qwen, DeepSeek, PaLM, and the experts inside most MoE models. Gemma uses a close relative with GELU as the switch (GeGLU).</p>
        </DeepDive>
        <Callout kind="research">Why does multiplying two projections help? There are intuitions (the layer can compute products of features, which a plain MLP can only approximate). But the paper that introduced these variants offers no explanation and famously attributes the success to “divine benevolence”. Treat SwiGLU as an empirical result: it was measured to be better, so people use it.</Callout>

        <h3>4. The KV cache, and why it became the bottleneck</h3>
        <p>A quick recap from <a href="#/lesson/inference">Inference</a>. With a causal mask, the keys and values of past tokens never change, so we store them and only run the newest token through the model. That is the <G t="kv-cache">KV cache</G>.</p>
        <p>Every serving system uses it, so its size is a design constraint. For every token, in every layer, for every K/V head, we keep one key vector and one value vector. Most of the rest of this lesson is about shrinking that pile.</p>
        <Fix
          problem="Without a cache, generating token 1,001 reprocesses the previous 1,000 tokens, again, at every step."
          idea="Store each layer’s K and V for all past tokens. Process only the new token; let its query read the stored keys and values."
          tradeoff="Memory. The cache grows linearly with context length, and it is per conversation. At long contexts and many users it, not the weights, limits how many requests fit on a GPU."
        />
        <DeepDive title="KV cache: why it helps and where it appears">
          <p><b>Why it helps.</b> Each step runs only the new token through the layers. The past is read from the cache, not recomputed. Output is identical: you checked that in <code>kv_cache_demo.py</code>.</p>
          <p><b>Where it appears.</b> Every production inference stack. “Prompt caching” on API price lists is, in essence, this cache kept between requests for a prompt prefix that has not changed.</p>
        </DeepDive>

        <h3>5. GQA and MQA: share the keys and values</h3>
        <p>In <a href="#/lesson/masks-and-heads">multi-head attention</a> every head has its own <span className="q">Q</span>, <span className="k">K</span> and <span className="v">V</span>. But only K and V are cached. Queries are used once and thrown away.</p>
        <p>So here is a cheap question: do we really need as many K/V heads as query heads?</p>
        <Term
          name="Grouped-query attention (GQA)"
          plain={<>Query heads are split into groups. All query heads in a group read the <em>same</em> key head and value head. The extreme version, one K/V head shared by every query head, has its own name: multi-query attention, or MQA. Plain attention with one K/V head each is multi-head attention, MHA.</>}
          example={<>64 query heads, 8 K/V heads: query heads 0 to 7 share K/V head 0, heads 8 to 15 share K/V head 1, and so on. The cache holds 8 heads per layer, not 64.</>}
          formal={<>With H query heads and G key/value heads (G divides H), query head h attends using K/V head ⌊h / (H/G)⌋. MHA is G = H, MQA is G = 1.</>}
        />
        <GqaCalculator />
        <Fix
          problem="KV-cache memory (and the time spent reading it for every generated token) grows with the number of K/V heads."
          idea="Keep many query heads, so the model can still ask many different questions. Let groups of them share one K/V head."
          tradeoff="Fewer distinct keys and values means slightly less expressive attention. MQA showed measurable quality loss in some studies; GQA with around 8 K/V heads was reported to stay close to full multi-head quality (Ainslie et al., 2023)."
        />
        <DeepDive title="GQA: why it helps and where it appears">
          <p><b>Why it helps.</b> The cache shrinks by the factor (query heads ÷ K/V heads). At long context the cache is a large part of what each decoding step must read from memory, so a smaller cache also means faster decoding and more users per GPU.</p>
          <p><b>Where it appears.</b> GQA: Llama-2 70B, Llama-3 8B and 70B, Mistral 7B, Qwen2 and Qwen3, Gemma 2 and 3, gpt-oss. MQA: PaLM, Falcon-7B (Falcon-40B uses 8 K/V heads). Llama-2 7B still used plain multi-head attention. MQA itself is from Shazeer, 2019.</p>
        </DeepDive>

        <h3>5b. Beyond GQA: three more ways to shrink the cache</h3>
        <p>GQA was the first answer, and it is still everywhere. Since 2024, open models have tried three bolder ones. Each attacks a different factor of the cache formula.</p>
        <p><b>Compress what you store (MLA).</b> Multi-head latent attention, from DeepSeek-V2 and used again in DeepSeek-V3 and Kimi K2, does not cache keys and values at all. For each token it caches one short <em>compressed</em> vector (a “latent”), and rebuilds every head’s keys and values from it with learned matrices when needed.</p>
        <Fix
          problem="Even with GQA, each token stores full key and value vectors for several heads, in every layer."
          idea="Store one small latent vector per token per layer: in DeepSeek-V3, 512 numbers, plus a 64-number key part that carries the RoPE rotation. Expand it into per-head keys and values on the fly."
          tradeoff="More arithmetic per step to expand the latent, and a more complicated attention layer (RoPE needs its own separate part, because a rotation cannot pass through the compression). The published reports found quality comparable to or better than standard attention."
        />
        <p><b>Look back only so far (sliding windows).</b> In a <em>sliding-window</em> layer each token attends only to the last W tokens, so that layer’s cache never holds more than W tokens. Models interleave these “local” layers with ordinary “global” layers that still see everything.</p>
        <Fix
          problem="Every layer caches every past token, although many layers mostly use nearby context."
          idea="Make most layers local (window W), keep some global. Gemma 2 alternates local and global layers one to one, with a 4,096-token window. Gemma 3 uses five local layers per global one, with a 1,024-token window. gpt-oss alternates a 128-token banded window with full attention."
          tradeoff="Information from far back can only travel through the global layers. And the global layers still cache everything, so the cache still grows with context, only more slowly."
        />
        <p><b>Replace some attention with a fixed-size memory (hybrids).</b> A few recent models swap most attention layers for layers from the <em>linear attention</em> or <em>state-space</em> family. Such a layer keeps one fixed-size running summary instead of a growing cache, a little like the RNNs attention replaced, but designed to train in parallel. A few full-attention layers remain for exact look-back.</p>
        <Fix
          problem="Any full-attention layer has a cache that grows with every token, and T² prompt-reading compute."
          idea="Use mostly layers with a fixed-size state. Qwen3-Next mixes Gated DeltaNet layers with gated attention layers at roughly three to one; Kimi Linear mixes Kimi Delta Attention layers with MLA layers at roughly three to one."
          tradeoff="A fixed-size summary must forget something. Exact recall of a detail far back depends on the few full-attention layers that remain."
        />
        <Callout kind="research">MLA and interleaved sliding windows are established in widely used open models. Hybrid linear-attention designs are newer: how well they hold up on long, recall-heavy tasks compared with full attention, and what mix is best, is still being measured. Treat the ratios above as one lab’s choices, not settled rules.</Callout>

        <h3>6. FlashAttention: same maths, less memory traffic</h3>
        <p>Look at <code>tiny_gpt.py</code>: <code>att = q @ k.transpose(-2, -1)</code> builds the full T×T table of scores, for every head. Then softmax reads it and writes another T×T table.</p>
        <p>At T = 4,096 in 16-bit numbers, one table for one head is 4,096 × 4,096 × 2 bytes = 32 MiB. With 32 heads that is 1 GiB per layer, per sequence, written to GPU memory and read back, only to be thrown away.</p>
        <p>A GPU has a small amount of very fast on-chip memory and a large amount of slower main memory. For attention, the arithmetic is quick. Moving those big tables between the two memories is what takes the time.</p>
        <Fix
          problem="Standard attention writes T×T tables to the GPU’s large, slower memory. Memory use grows with T², and most of the time is spent moving data, not computing."
          idea="Cut Q, K and V into tiles small enough for the fast on-chip memory. Compute attention tile by tile, keeping a running softmax total so the final result is exactly right. Never store the full T×T table."
          tradeoff="It is a hand-written GPU kernel (a small program that runs directly on the GPU), tied to hardware details, and much harder to read or change than three lines of PyTorch. The T² arithmetic is still done; only the T² storage is avoided."
        />
        <DeepDive title="FlashAttention: why it helps and where it appears">
          <p><b>Why it helps.</b> Extra memory grows with T instead of T². Far fewer slow memory reads and writes, so it is several times faster on long sequences. The output is the same as standard attention up to floating-point rounding: it is not an approximation.</p>
          <p><b>Where it appears.</b> Inside PyTorch’s <code>F.scaled_dot_product_attention</code> and practically every training and serving stack. Dao et al., 2022, with later versions for newer GPUs.</p>
        </DeepDive>
        <Callout kind="dev">This is an I/O optimisation, like processing a huge file in blocks that fit in cache instead of seeking all over the disk. The algorithm’s answer is unchanged; its memory access pattern is what changed.</Callout>

        <h3>7. Longer context windows</h3>
        <p>GPT-2 read 1,024 tokens. Llama-2 reads 4,096, Llama-3 8,192, Llama-3.1 about 128 thousand (131,072 in its config). RoPE, a smaller cache and FlashAttention are what made this affordable. But affordable is not free.</p>
        <Fix
          problem="Whole codebases, long documents and long conversations do not fit in 1,024 tokens. Whatever does not fit, the model cannot see at all."
          idea="Pretrain on moderate lengths, then continue training briefly on long sequences with the RoPE rotations slowed down, so long distances look like distances the model already knows."
          tradeoff={<>Attention compute for reading a prompt grows with T²: going from 4,096 to 131,072 tokens is 32× more tokens but 1,024× more query-key scores. The KV cache grows linearly: a 70B-class model with GQA needs about 40 GiB of cache for one 131,072-token conversation.</>}
        />
        <DeepDive title="How the rotations are slowed: position interpolation, base scaling, YaRN">
          <p>A model trained on 4,096 positions has only ever seen rotation angles up to 4,096 × θ. Show it position 20,000 and the fast pairs have spun into angles it has never met.</p>
          <p><b>Position interpolation</b> squeezes the positions: divide every position by the stretch factor, so 16,000 tokens use the same angles 4,000 used to. <b>Base scaling</b> (sometimes called NTK-aware scaling) raises the RoPE base instead, which slows the slow pairs a lot and the fast pairs hardly at all, so nearby tokens stay sharp. <b>YaRN</b> combines the two per pair and adds a small correction to the attention scores. Qwen and DeepSeek models use YaRN; you saw <code>"type": "yarn"</code> in Kabir’s config.</p>
          <p>All of these need some further training on long text to work well. Which method is best is still argued about, and each lab tunes its own recipe.</p>
        </DeepDive>
        <Callout kind="research">“Fits in the window” is not the same as “is used well”. Studies such as “Lost in the Middle” (Liu et al., 2023) found that some models used information at the start and end of a long prompt much better than information in the middle. Newer models have improved on simple find-the-sentence tests, but how reliably models reason over very long inputs is still being measured. Test on your own task before trusting a number on a model card.</Callout>

        <h3>8. More parameters, and mixture-of-experts</h3>
        <p>The bluntest change is size. Our model has under a million <G t="parameters">parameters</G>. GPT-2 small had 124 million. Open models now range from about 1 billion to around a trillion. <a href="#/lesson/why-llms-know">Why LLMs know things</a> covered why bigger models reach lower loss.</p>
        <p>But in a normal (“dense”) model every parameter is used for every token. Twice the parameters means twice the compute per token. Mixture-of-experts breaks that link, and by 2025 it had become the usual design for the largest open-weight models.</p>
        <Fix
          problem="More parameters give a better model, but in a dense model the compute per token grows in step with parameter count."
          idea="In each block, replace the one MLP with many smaller parallel MLPs (“experts”) plus a small learned router. For each token the router picks a few experts, and only those run. Many designs also keep one “shared” expert that every token uses."
          tradeoff="All experts must still sit in GPU memory, so an MoE needs far more memory than a dense model of the same speed. The router must be trained to spread tokens evenly, and splitting experts across many GPUs makes training and serving harder."
        />
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>Open-weight model</th><th>experts per MoE layer</th><th>used per token</th><th>parameters: total / active</th></tr></thead>
            <tbody>
              <tr><td>Mixtral 8x7B (2023)</td><td className="mono">8</td><td className="mono">2</td><td className="mono">≈ 47B / ≈ 13B</td></tr>
              <tr><td>DeepSeek-V3 and R1</td><td className="mono">256 + 1 shared</td><td className="mono">8 + 1 shared</td><td className="mono">671B / ≈ 37B</td></tr>
              <tr><td>Qwen3-235B-A22B</td><td className="mono">128</td><td className="mono">8</td><td className="mono">235B / 22B</td></tr>
              <tr><td>Kimi K2</td><td className="mono">384 + 1 shared</td><td className="mono">8 + 1 shared</td><td className="mono">≈ 1T / ≈ 32B</td></tr>
              <tr><td>gpt-oss-120b</td><td className="mono">128</td><td className="mono">4</td><td className="mono">≈ 117B / ≈ 5.1B</td></tr>
            </tbody>
          </table>
        </div>
        <p>Read DeepSeek-V3’s row. Each token touches about 37B of 671B parameters: roughly 5.5%. Notice the trend too. Early MoEs had a few big experts. Newer ones have <b>many small, fine-grained experts</b>, pick several of them, and add a shared expert for the knowledge every token needs. Llama 4 Scout and Maverick are MoE models as well, and gpt-oss-20b uses 32 experts with 4 active.</p>
        <p><b>Why do active parameters decide the serving bill?</b> Two reasons. The arithmetic per token is about 2 operations per <em>active</em> parameter, so 37B active costs like a 37B dense model, not a 671B one. And at small batch sizes each decoding step is limited by reading weights from memory, and it only reads the experts it uses. What the total still decides is how much GPU memory you must buy to hold the model at all.</p>
        <DeepDive title="MoE: routing, balance, and where it appears">
          <p><b>The router</b> is one small matrix: it scores every expert for the current token, keeps the top few, and mixes their outputs using those scores. It is trained along with everything else.</p>
          <p><b>Balance.</b> Left alone, a router tends to send most tokens to a few favourite experts, leaving the rest untrained and some GPUs idle. Most designs add a small balancing term to the loss. DeepSeek-V3 relies mainly on adjusting a per-expert routing bias instead.</p>
          <p><b>Where it appears.</b> Mixtral, DeepSeek-V2 and V3, Qwen3’s large models, Kimi K2, Llama 4, gpt-oss. Dense models remain common at small and medium sizes, where memory is tight and simplicity pays.</p>
        </DeepDive>
        <Callout kind="model">“Expert” is a misleading name. Published analyses of open MoE models found that experts do not split into tidy human subjects such as “the maths expert”. Routing tends to follow lower-level token patterns. Think “sharded MLP with a learned shard key”, not “committee of specialists”.</Callout>
        <p>Riya turns to Dev. “So the biggest one on the list uses about 32 billion parameters per token. More parameters on disk, yes. Not more work per word.” Dev frowns at the table for a long moment. “Okay. That’s actually clever.”</p>
      </TryIt>
      <Numbers>
        <h3>The KV cache of two real models, by hand</h3>
        <p>For every token we store one key and one value (that is the 2), in every layer, for every K/V head, <code>head_dim</code> numbers each. In 16-bit precision each number is 2 bytes.</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th /><th>Llama-2 7B (MHA)</th><th>Llama-2 70B (GQA)</th><th>70B if it used MHA</th></tr></thead>
            <tbody>
              <tr><td>layers</td><td className="mono">32</td><td className="mono">80</td><td className="mono">80</td></tr>
              <tr><td>query heads</td><td className="mono">32</td><td className="mono">64</td><td className="mono">64</td></tr>
              <tr><td><span className="k">K</span>/<span className="v">V</span> heads</td><td className="mono">32</td><td className="mono"><b>8</b></td><td className="mono">64</td></tr>
              <tr><td>head_dim</td><td className="mono">128</td><td className="mono">128</td><td className="mono">128</td></tr>
              <tr><td>bytes per token<br /><span className="muted">2 × layers × KV heads × head_dim × 2</span></td><td className="mono">524,288<br />= 0.5 MiB</td><td className="mono">327,680<br />= 0.31 MiB</td><td className="mono">2,621,440<br />= 2.5 MiB</td></tr>
              <tr><td>× 4,096 tokens</td><td className="mono"><b>2.00 GiB</b></td><td className="mono"><b>1.25 GiB</b></td><td className="mono"><b>10.0 GiB</b></td></tr>
            </tbody>
          </table>
        </div>
        <p>Read the middle column again. The 70B model is ten times larger than the 7B model, yet its cache per token is <em>smaller</em>, because 8 K/V heads replaced 64. The saving is exactly 64 ÷ 8 = 8×.</p>
        <p>Stretch the context to 131,072 tokens and the 70B cache is 40 GiB with GQA, or 320 GiB with plain multi-head attention. For comparison, the 70B weights themselves take about 140 GB in 16-bit. Check all of these in the calculator.</p>
        <p>Now MLA. DeepSeek-V3 has 61 layers and caches 512 + 64 = 576 numbers per token per layer, with no separate key and value and no K/V heads:</p>
        <p className="mono" style={{ fontSize: 14.5 }}>61 layers × 576 × 2 bytes = 70,272 bytes ≈ 68.6 KiB per token</p>
        <p>That is about 4.7 times less than Llama-2 70B’s 327,680 bytes with GQA, in a model with roughly ten times the parameters. At 131,072 tokens it comes to about 8.6 GiB, against 40 GiB. (The calculator only models GQA, so check this one by hand.)</p>

        <h3>RoPE by hand</h3>
        <p>Take the simplest vectors: <span className="q mono">q = [1, 0]</span> and <span className="k mono">k = [1, 0]</span>, with θ = 30° per position. Put the query at position 5 and the key at position 3.</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th /><th>positions (5, 3)</th><th>positions (105, 103)</th></tr></thead>
            <tbody>
              <tr><td><span className="q">q</span> rotated by</td><td className="mono">5 × 30° = 150° → [−0.866, 0.5]</td><td className="mono">105 × 30° = 3150° = 270° → [0, −1]</td></tr>
              <tr><td><span className="k">k</span> rotated by</td><td className="mono">3 × 30° = 90° → [0, 1]</td><td className="mono">103 × 30° = 3090° = 210° → [−0.866, −0.5]</td></tr>
              <tr><td>dot product</td><td className="mono">−0.866×0 + 0.5×1 = <b>0.5</b></td><td className="mono">0×(−0.866) + (−1)×(−0.5) = <b>0.5</b></td></tr>
              <tr><td>angle between them</td><td className="mono">150° − 90° = 60°</td><td className="mono">270° − 210° = 60°</td></tr>
            </tbody>
          </table>
        </div>
        <p>Both vectors have length 1, so the dot product is cos(60°) = 0.5. The angle between them is (5 − 3) × 30° in both cases. One hundred positions later, the score is identical.</p>

        <h3>RMSNorm by hand</h3>
        <p>For <span className="mono">x = [2, 4, 6, 8]</span>: the squares are 4, 16, 36, 64. Their mean is 30. √30 = 5.477. Divide: <span className="mono">[0.365, 0.730, 1.095, 1.461]</span>.</p>
        <p>LayerNorm on the same vector: mean 5, centred <span className="mono">[−3, −1, 1, 3]</span>, spread √5 = 2.236, result <span className="mono">[−1.342, −0.447, 0.447, 1.342]</span>. Different numbers, same job: the output has a standard size no matter how large the input was.</p>
      </Numbers>

      <TheMath>
        <p>Three short formulas. You have already computed each of them.</p>
        <Equation
          label="RMSNorm: x divided by the root of the mean of x squared, times a gain"
          symbols={[
            ['x', 'one token’s vector, d numbers'],
            [<>mean(x²)</>, 'square every number, then average: 30 in the example'],
            ['ε', 'a tiny constant such as 0.00001 so we never divide by zero'],
            ['g', 'a learned gain, one number per dimension, starting at 1'],
          ]}
        >
          RMSNorm(x) = x / √( mean(x²) + ε ) · g
        </Equation>
        <p>LayerNorm would first replace x by x − mean(x), and add a learned bias at the end. That is the entire difference.</p>
        <Equation
          label="RoPE: the score between a rotated query and a rotated key depends on n minus m"
          symbols={[
            [<>R(α)</>, 'rotate a 2D vector by angle α: [x, y] → [x·cos α − y·sin α, x·sin α + y·cos α]'],
            ['m, n', <>the positions of the <span className="q">query</span> token and the <span className="k">key</span> token</>],
            ['θ', 'how much one step of position rotates the vector'],
            [<>(n − m)θ</>, 'the only place positions appear: their difference'],
          ]}
        >
          ( R(mθ) <span className="q">q</span> ) · ( R(nθ) <span className="k">k</span> ) = <span className="q">q</span> · ( R((n − m)θ) <span className="k">k</span> )
        </Equation>
        <p>In words: rotating both vectors and then comparing them gives the same answer as leaving the query alone and rotating the key by the <em>difference</em>. Rotating the whole picture does not change any angle inside it.</p>
        <DeepDive title="How this extends from 2 numbers to 128">
          <p>A real head has <code>head_dim</code> numbers, for example 128. RoPE treats them as 64 separate 2D pairs and rotates each pair with its own speed: θ<sub>i</sub> = 10000<sup>−2i/d</sup> for pair i = 0, 1, … d/2 − 1. The base 10000 is the value from the RoFormer paper, also used by Llama-2 and Mistral 7B. It is a setting, not a constant: Llama-3 uses 500,000. The numbers below are for base 10000. The first pair turns about 57° per position (1 radian). The last pair turns so slowly that it takes tens of thousands of positions to go round once.</p>
          <p>Why many speeds? A single fast rotation wraps around: with θ = 30°, offsets 2 and 14 look identical. You can see this in the lab’s curve, which repeats. Fast pairs resolve nearby positions precisely; slow pairs tell far-apart positions apart. Together they work like the hands of a clock. This is also the dial that long-context methods adjust: they slow the rotations down (a larger base, or rescaled positions) so that longer distances stay within the range seen in training.</p>
          <p>The shift-invariance holds pair by pair, so it holds for the sum. <code>src/lib/rope.test.ts</code> checks it numerically in 8 dimensions.</p>
        </DeepDive>
        <Equation
          label="KV cache bytes equals 2 times layers times KV heads times head dim times tokens times bytes per number"
          symbols={[
            ['2', 'one key vector and one value vector per token'],
            ['L', 'number of layers (blocks)'],
            [<>H<sub>kv</sub></>, 'number of key/value heads: the only factor GQA changes'],
            [<>d<sub>head</sub></>, 'numbers per head'],
            ['T', 'tokens currently in the conversation'],
            ['b', 'bytes per stored number: 2 for 16-bit'],
          ]}
        >
          cache bytes = 2 · L · H<sub>kv</sub> · d<sub>head</sub> · T · b
        </Equation>
      </TheMath>

      <CodeIt title="Let’s code it: what would change in tiny_gpt.py">
        <p>Each snippet shows the lines from <code>tiny_gpt.py</code> first, then what a modern model has in their place. These are sketches in the style of the open Llama code, short enough to read, not drop-in patches.</p>

        <Code title="1. RMSNorm replaces nn.LayerNorm">{`
# tiny_gpt.py, class Block:
#   self.ln1 = nn.LayerNorm(cfg.n_embd)

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d))       # gain only, no bias

    def forward(self, x):                           # x: (B, T, D)
        ms = x.pow(2).mean(dim=-1, keepdim=True)    # one number per token
        return x / torch.sqrt(ms + self.eps) * self.g
`}</Code>
        <p>Note <code>dim=-1</code>: each token is normalised by its own size, never mixed with other tokens. Recent PyTorch versions ship this as <code>nn.RMSNorm</code>.</p>

        <Code title="2. RoPE: delete the position table, rotate q and k">{`
# tiny_gpt.py, class GPT:
#   self.pos_emb = nn.Embedding(cfg.context_len, cfg.n_embd)   <- deleted
#   x = self.tok_emb(idx) + self.pos_emb(pos)                  <- becomes: x = self.tok_emb(idx)

def rope(x, pos, base=10000.0):                 # x: (B, H, T, hd), pos: (T,)
    hd = x.shape[-1]
    i = torch.arange(0, hd, 2, device=x.device)          # 0, 2, 4, ...: one entry per pair
    speeds = base ** (-i / hd)                           # (hd/2,) one theta per pair
    ang = pos[:, None] * speeds[None, :]                 # (T, hd/2) position x theta
    x1, x2 = x[..., 0::2], x[..., 1::2]                  # the two halves of each pair
    out = torch.stack([x1 * ang.cos() - x2 * ang.sin(),
                       x1 * ang.sin() + x2 * ang.cos()], dim=-1)
    return out.flatten(-2)                               # back to (B, H, T, hd)
`}</Code>
        <Code title="…and three new lines inside CausalSelfAttention.forward">{`
pos = torch.arange(T, device=x.device)   # 0, 1, ... T-1: the same positions pos_emb used to look up
q = rope(q, pos)      # after the .view(...).transpose(1, 2), before q @ k
k = rope(k, pos)      # v is NOT rotated
att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)     # unchanged
`}</Code>
        <p>The two lines inside <code>torch.stack</code> are the rotation formula from the math section, applied to every pair at once.</p>

        <Code title="3. SwiGLU replaces the GELU MLP">{`
# tiny_gpt.py, class FeedForward:
#   nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd)

class SwiGLU(nn.Module):
    def __init__(self, d):
        super().__init__()
        h = int(8 * d / 3)                          # narrower, so 3 matrices cost what 2 did
        self.w1 = nn.Linear(d, h, bias=False)       # gate path
        self.w3 = nn.Linear(d, h, bias=False)       # content path
        self.w2 = nn.Linear(h, d, bias=False)       # back down to d

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))    # gate * content
`}</Code>
        <p><code>F.silu</code> is the Swish function, <code>z · sigmoid(z)</code>: a smooth relative of the ReLU hinge you met in <a href="#/lesson/neurons">Neurons and layers</a>. Most modern models also drop the bias terms, as here. The 8d/3 width is the equal-parameter choice (Llama-2 7B: 11,008 for d = 4,096, rounded up to a convenient multiple). Later models often choose wider: Llama-3 8B uses 14,336, which is 3.5d.</p>

        <Code title="4. GQA: fewer K/V heads, shared by groups of query heads">{`
# tiny_gpt.py:  self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)      # same heads for q, k, v

self.q  = nn.Linear(D, n_head * hd, bias=False)          # e.g. 64 query heads
self.kv = nn.Linear(D, 2 * n_kv_head * hd, bias=False)   # e.g.  8 K/V heads: this is what gets cached

# in forward, after reshaping k and v to (B, n_kv_head, T, hd):
group = n_head // n_kv_head                              # 64 // 8 = 8
k = k.repeat_interleave(group, dim=1)                    # (B, n_head, T, hd)
v = v.repeat_interleave(group, dim=1)                    # each K/V head serves 8 query heads
`}</Code>
        <p>The repeat happens after reading from the cache, so the cache itself stays 8 heads wide. Real kernels avoid even this copy.</p>

        <Code title="5. FlashAttention: four lines become one">{`
# tiny_gpt.py:
#   att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)     # (B, H, T, T)  <- the big table
#   att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
#   att = F.softmax(att, dim=-1)
#   y = att @ v

y = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # same y, no (T, T) tensor kept
`}</Code>
        <Callout kind="established">PyTorch picks a fused kernel (FlashAttention or a similar one) for this call when the hardware supports it. The result matches the four-line version up to floating-point rounding. Nothing about the model changes: same weights, same outputs.</Callout>
      </CodeIt>

      <BreakIt>
        <p>Predict first, then check.</p>
        <ul>
          <li><b>RoPE lab: set θ = 60° and look at the curve.</b> It repeats every 6 positions: offset +1 and offset +7 get the same score. With one rotation speed the model cannot tell them apart. That is why real RoPE uses many speeds (see the deep dive).</li>
          <li><b>RoPE lab: put both sliders at the same position</b>, then shift. The score stays at 4.00, the plain dot product of [2, 1] and [1, 2]. A token looking at itself sees no rotation at all.</li>
          <li><b>Norm lab: press “Add 5 to everything” several times.</b> LayerNorm’s output does not move, because it subtracts the mean. RMSNorm’s output drifts toward [1, 1, 1, 1]: a large shared offset drowns the differences. Inside a trained model this has not proved to be a problem, but it is the one thing RMSNorm gave up.</li>
          <li><b>Norm lab: press “Multiply everything by 10”.</b> Neither output changes. Both norms exist to remove overall scale.</li>
          <li><b>Calculator: load Llama-2 7B and set K/V heads to 8.</b> As far as the cache is concerned, this is the step from Llama-2 7B to Llama-3 8B: the cache per token drops 4×. Then set precision to 8-bit for another 2×. Cache quantisation is a separate, stackable trick with its own small quality cost.</li>
          <li><b>Calculator: load the 70B preset, push context to 1,048,576.</b> Even with GQA the cache is 320 GiB for one conversation. Long context is a memory bill.</li>
        </ul>
      </BreakIt>

      <Exercises>
        <Exercise
          id="modern-architecture-gqa-factor"
          type="calculate"
          title="How much does GQA save?"
          answer={{ value: 8, tolerance: 0.01 }}
          answerLabel="saving factor, e.g. 2"
          hints={[
            'Write the cache formula twice, once for each design. Which single factor differs?',
            'Layers, head_dim, tokens and bytes are identical in both. Only the number of K/V heads changes: 64 vs 8.',
            'The saving factor is the ratio of the two K/V head counts.',
          ]}
          solution={<><p>Cache bytes = 2 × layers × <b>KV heads</b> × head_dim × tokens × bytes. Everything cancels except the K/V head count: 64 ÷ 8 = <b>8×</b> smaller.</p><p>Notice what did not matter: the number of layers, the context length and the precision. GQA’s saving is always query heads ÷ K/V heads. That is why Llama-2 70B at 4,096 tokens needs 1.25 GiB of cache rather than 10 GiB.</p></>}
        >
          <p>A model has 64 query heads. Design A gives every query head its own K/V head (plain multi-head attention). Design B uses GQA with 8 K/V heads. Everything else is equal. By what factor is B’s KV cache smaller?</p>
        </Exercise>

        <Exercise
          id="modern-architecture-kv-bytes"
          type="calculate"
          title="One conversation on Llama-3 8B"
          answer={{ value: 1, tolerance: 0.03 }}
          answerLabel="GiB"
          hints={[
            'Bytes = 2 × layers × KV heads × head_dim × tokens × bytes per number.',
            '2 × 32 × 8 × 128 × 2 = 131,072 bytes per token. That is 128 KiB.',
            '131,072 × 8,192 tokens = 1,073,741,824 bytes. How many GiB is that (1 GiB = 1,073,741,824 bytes)?',
          ]}
          solution={<p>Per token: 2 × 32 × 8 × 128 × 2 = 131,072 bytes (128 KiB). Times 8,192 tokens = 1,073,741,824 bytes = exactly <b>1 GiB</b>. With plain multi-head attention (32 K/V heads) it would have been 4 GiB. On a GPU with 24 GiB, where the 16-bit weights already take about 16 GB, that difference decides whether you can serve one long conversation or several.</p>}
        >
          <p>Llama-3 8B has 32 layers, 32 query heads, 8 K/V heads and head_dim 128. How many GiB of KV cache does one full 8,192-token conversation need in 16-bit precision (2 bytes per number)? Work it out on paper, then confirm with the calculator.</p>
        </Exercise>

        <Exercise
          id="modern-architecture-rope-predict"
          type="predict"
          title="Fifty positions later"
          hints={[
            'What is the offset (key position minus query position) in the first case? And in the second?',
            'The RoPE score can be rewritten so that positions appear only as n − m.',
          ]}
          solution={<><p>The score is <b>exactly the same</b>: −2.260 in both cases. The offset is 4 both times, so the key is rotated 4 × 20° = 80° relative to the query, wherever the pair sits in the text.</p><p>In the lab: set m = 3, n = 7, then drag “shift both” to 50. The solid arrows spin round, but the angle between them is frozen, and the RoPE row of the table says “no: identical”. The additive row changes. That contrast is the reason RoPE exists.</p></>}
        >
          <p>In the RoPE lab (q = [2, 1], k = [1, 2], θ = 20°) the query is at position 3 and the key at position 7. The score is −2.260. Predict the score when the query is at position 53 and the key at position 57. Then predict: does the additive scheme also stay the same? Check both in the lab.</p>
        </Exercise>

        <Exercise
          id="modern-architecture-debug"
          type="debug"
          title="RoPE works in training, breaks with the KV cache"
          hints={[
            'During cached generation, how many tokens go through forward() at each step? So what is T?',
            'With T = 1, torch.arange(T) is [0]. Which position does the new token’s query get rotated by? Which positions were the cached keys rotated by?',
            'The new token is really at position len(cache). The offset the model sees is wrong by exactly that amount.',
          ]}
          solution={<><p>With a KV cache only the newest token is processed, so <code>T = 1</code> and <code>pos = [0]</code>. The new query is rotated as if it were at position 0, while the cached keys were (correctly) rotated at positions 0, 1, 2, … So the model sees offsets as if the new token stood at the very start of the text, <em>before</em> everything it is reading.</p><p>Fix: pass the true position. <code>pos = torch.arange(past_len, past_len + T)</code>, where <code>past_len</code> is the number of tokens already in the cache.</p><p>Nothing crashes, all shapes are right, and training (which has no cache) is fine. It only shows up as worse text at inference. The test that catches it is the one from <code>kv_cache_demo.py</code>: cached and uncached generation must produce identical logits.</p></>}
        >
          <p>A colleague added RoPE to their GPT. Training loss looks great. Generation without a KV cache is fine. With the KV cache turned on, output quickly turns to nonsense. Here is their attention forward pass. What is wrong?</p>
          <Code>{`
def forward(self, x, cache=None):          # with a cache, x holds only the NEW token
    B, T, D = x.shape
    q, k, v = self.make_qkv(x)             # (B, H, T, hd) each
    pos = torch.arange(T)
    q, k = rope(q, pos), rope(k, pos)
    if cache is not None:
        k = torch.cat([cache.k, k], dim=2)
        v = torch.cat([cache.v, v], dim=2)
    ...
`}</Code>
        </Exercise>

        <Exercise
          id="modern-architecture-modify"
          type="modify"
          title="Swap the norm in your own GPT"
          hints={[
            'Add the RMSNorm class from this lesson to tiny_gpt.py. There are three places that say nn.LayerNorm: ln1, ln2 and ln_f.',
            'Run python phase3-transformers/tiny_gpt.py --quick before and after. The seed is fixed (1337), so compare the validation loss printed at step 300.',
          ]}
          solution={<><p>You should see validation loss within a few hundredths of the LayerNorm run, and the parameter count drops slightly (no bias vectors: 9 norms × 128 numbers = 1,152 fewer parameters). At this scale you will not measure a speed difference: the norm is a tiny fraction of the work, and PyTorch’s LayerNorm is a fused kernel while your RMSNorm is plain Python ops.</p><p>That is the honest picture of RMSNorm. It is not a breakthrough. It is a simplification that costs nothing in quality, and at the scale of trillions of tokens small savings are worth taking.</p></>}
        >
          <p>Open <code>phase3-transformers/tiny_gpt.py</code>. Replace all three <code>nn.LayerNorm</code> uses with the <code>RMSNorm</code> class from this lesson. Before running: do you expect the validation loss after <code>--quick</code> to be much better, much worse, or about the same? How many parameters will disappear?</p>
        </Exercise>

        <ExplainBack
          id="modern-architecture-explain"
          prompt="Dev forwards you a model card: “RMSNorm, RoPE, SwiGLU, GQA, FlashAttention, 128k context.” He writes: “So it is a completely different architecture from GPT-2?” Answer them. For at least three of those terms, say what problem it fixes and what it costs."
          modelAnswer={<><p>No. It is the same skeleton: token embeddings, a stack of blocks with attention and an MLP, residual connections, next-token logits. Each term is a repair to one part.</p><p>RoPE replaces the learned position table, which had a hard last row and no notion of distance, with rotations of q and k so that scores depend on relative offset; it still does not make unlimited length free. GQA lets groups of query heads share K/V heads because the KV cache was eating memory; the cache shrinks by query heads ÷ K/V heads, at a small quality risk. FlashAttention computes exactly the same attention but in tiles, so the T×T table never sits in slow GPU memory; the cost is a complicated hardware-specific kernel. RMSNorm is LayerNorm without the mean subtraction and bias: cheaper, same quality. SwiGLU is a gated three-matrix MLP that measured better per parameter, without a solid theory of why. 128k context is what those make affordable, but compute to read a prompt still grows with T², and models do not necessarily use the middle of a long context well.</p></>}
        />
      </Exercises>

      <CheckYourself
        questions={[
          {
            q: 'With RoPE, the query is at position 10 and the key at position 4. Both tokens are then moved 1,000 positions later in the text. What happens to their attention score (before softmax)?',
            options: ['It shrinks, because far-away positions are rotated more', 'It stays the same, because only the difference between the positions affects q·k', 'It becomes zero, because position 1,010 was never in the table', 'It depends on the value vectors'],
            answer: 1,
            explain: 'Rotating both vectors by the same extra angle does not change the angle between them. There is no table to run out of, and values are never rotated.',
          },
          {
            q: 'Why does GQA shrink the KV cache but leave the number of query heads alone?',
            options: ['Queries are never cached: each is used once, for its own token. Keys and values are what every later token reads, so they are what we store', 'Query heads are smaller than key heads', 'Because queries are computed by the MLP', 'Because the softmax needs at least 32 queries'],
            answer: 0,
            explain: 'The cache holds K and V only. Sharing K/V heads across groups of query heads cuts what must be stored, while the model can still ask many different questions.',
          },
          {
            q: 'A blog post says: “FlashAttention approximates attention to make it faster.” What is wrong with that?',
            options: ['Nothing, that is accurate', 'It is slower, not faster', 'It computes the same result as standard attention; the speed comes from never storing the T×T table in slow GPU memory', 'It only works without a causal mask'],
            answer: 2,
            explain: 'It is an I/O optimisation: tiles plus a running softmax total. Same maths, same outputs up to rounding, different memory access pattern.',
          },
          {
            q: 'What does RMSNorm remove from LayerNorm?',
            options: ['The division, so vectors can grow freely', 'The learned gain', 'The mean subtraction and the bias; it keeps the rescaling and the gain', 'The residual connection'],
            answer: 2,
            explain: 'Experiments showed the rescaling is what stabilises training. Dropping the re-centring saves work with no measured loss in quality.',
          },
          {
            q: 'A model’s context window grows from 4,096 to 131,072 tokens (32×). Which statement is right?',
            options: ['KV-cache memory and prompt-reading attention compute both grow 32×', 'KV-cache memory grows 32×; the number of query-key scores for reading a full prompt grows 1,024×; and neither guarantees the model uses the middle of the prompt well', 'Nothing grows, thanks to FlashAttention', 'KV-cache memory grows 1,024×'],
            answer: 1,
            explain: 'Cache is linear in T, attention scores are T². FlashAttention avoids storing the T² table but still does the T² arithmetic. How well long context is used is an empirical question.',
          },
          {
            q: 'DeepSeek-V3 has 671B parameters but about 37B active per token. Compared with a 37B dense model, what does serving it cost?',
            options: ['About the same compute per token, but far more GPU memory, because all 671B parameters must be loaded', 'About 18 times more compute per token, because every parameter still takes part', 'Less memory, because inactive experts are deleted after training', 'Exactly the same in every respect'],
            answer: 0,
            explain: 'Only the chosen experts run, so arithmetic per token follows the active count. But the router can pick any expert for the next token, so every expert must be in memory.',
          },
        ]}
      />

      <Remember
        items={[
          <>A modern open model is <b>the same skeleton</b> you built: blocks of attention + MLP with residuals. The differences are targeted repairs, each with a cost.</>,
          <><b>RoPE</b> rotates <span className="q">q</span> and <span className="k">k</span> by position × θ, so the score depends only on the <b>offset</b> between two tokens. No position table, no extra parameters. It does not make unlimited length free.</>,
          <><b>KV cache bytes = 2 × layers × KV heads × head_dim × tokens × bytes.</b> <b>GQA</b> shrinks the “KV heads” factor: 64 query heads on 8 K/V heads = 8× less cache.</>,
          <><b>FlashAttention</b> is exact attention with a better memory access pattern. <b>RMSNorm</b> is LayerNorm minus the mean and bias. <b>SwiGLU</b> is a gated MLP that measured better; why is mostly empirical.</>,
          <><b>Long context</b> costs T² compute and linear cache memory, and fitting in the window does not mean being used well. <b>MoE</b>, now the usual design for the largest open-weight models, buys more parameters at the compute of its <b>active</b> parameters, and pays in memory and routing complexity. Beyond GQA, <b>MLA</b>, <b>sliding windows</b> and <b>hybrid</b> layers shrink the cache further.</>,
        ]}
      />

      <RealLLM>
        <Flow horizontal steps={[{ label: 'Tiny GPT', sub: 'what you built' }, { label: 'GPT-2 / GPT-3', sub: 'same design, bigger' }, { label: 'This lesson', sub: 'targeted repairs' }, { label: 'Assistant', sub: 'next: training stages' }]} active={2} />
        <h3>One block, side by side</h3>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>Step inside a block</th><th>Our tiny GPT block</th><th>A Llama-style block</th></tr></thead>
            <tbody>
              <tr><td>before the block</td><td className="mono">x = tok_emb + pos_emb</td><td className="mono">x = tok_emb</td></tr>
              <tr><td>1. normalise</td><td className="mono">LayerNorm(x)</td><td className="mono">RMSNorm(x)</td></tr>
              <tr><td>2. make q, k, v</td><td>4 heads each for q, k, v</td><td>many q heads, fewer k/v heads (GQA)</td></tr>
              <tr><td>3. position</td><td>(already added at the input)</td><td>rotate <span className="q">q</span> and <span className="k">k</span> with RoPE</td></tr>
              <tr><td>4. attention</td><td>softmax(QKᵀ/√d)V, full T×T table</td><td>softmax(QKᵀ/√d)V, FlashAttention kernel, K/V read from cache</td></tr>
              <tr><td>5. residual</td><td className="mono">x = x + attn</td><td className="mono">x = x + attn</td></tr>
              <tr><td>6. normalise</td><td className="mono">LayerNorm(x)</td><td className="mono">RMSNorm(x)</td></tr>
              <tr><td>7. MLP</td><td>Linear → GELU → Linear (4d wide)</td><td>SwiGLU (3 matrices, ≈ 8d/3 wide), or a router + experts (MoE)</td></tr>
              <tr><td>8. residual</td><td className="mono">x = x + mlp</td><td className="mono">x = x + mlp</td></tr>
              <tr><td>biases, dropout</td><td>biases everywhere, dropout 0.1</td><td>usually no biases, no dropout in pretraining</td></tr>
            </tbody>
          </table>
        </div>
        <p>Rows 5 and 8, the residual stream, did not change at all. Neither did the attention formula, the causal mask, the training loss or the generation loop.</p>
        <ToyVsReal
          toy={<ul><li>4 blocks, 128 numbers per token, 4 heads of 32</li><li>64-token window from a learned table</li><li>KV cache (if you add one, in 32-bit): 256 KiB</li><li>every parameter used for every token</li></ul>}
          real={<ul><li>Llama-3 70B: 80 blocks, 8,192 numbers per token, 64 query heads of 128 sharing 8 K/V heads</li><li>8,192 to 128,000+ tokens via RoPE</li><li>KV cache: gigabytes per conversation, the main serving constraint</li><li>the largest open-weight models are mostly mixture-of-experts: DeepSeek-V3 runs about 37B of its 671B parameters per token</li></ul>}
        />
        <Callout kind="established">The mechanisms here (RMSNorm, QK-norm, RoPE, gated MLPs, GQA, MLA, sliding-window attention, FlashAttention, MoE routing) are published, implemented in open code, and you can read them in the released Llama, Mistral, Qwen, Gemma, DeepSeek, Kimi or gpt-oss model files and configs. The arithmetic of KV-cache size and T² attention cost follows directly from the definitions.</Callout>
        <Callout kind="model">“Modern LLM = GPT-2 + these repairs” is a simplification. Real models combine them differently: GQA or MLA, all-global or interleaved local and global layers, dense or MoE, pure attention or a hybrid with linear-attention layers, and different long-context recipes (position interpolation, base scaling, YaRN). They also differ in tokenizer size, where norms are placed, and how weights are initialised and quantised. The list here is the common core, not a complete specification of any one model.</Callout>
        <Callout kind="research">Why gating helps, how well very long contexts are actually used, the best way to extend RoPE beyond the training length, how MoE experts specialise, and whether hybrid linear-attention models match full attention on long, recall-heavy work are all open or actively studied. And the architectures of closed commercial models are not published: statements about what is inside them are guesses, however confidently they are made.</Callout>
        <p>Kabir closes the config file. Riya realises she read every line of it, and could name the problem each line fixes. Next, the question she has been saving: this model continues text, so how does it become something that answers?</p>
      </RealLLM>
    </Lesson>
  )
}
