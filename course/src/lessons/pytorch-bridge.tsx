import { Lesson, Why, Problem, MentalModel, TryIt, Numbers, TheMath, CodeIt, BreakIt, Exercises, CheckYourself, Remember, RealLLM } from '../components/lesson'
import { Callout, DeepDive, Equation, G, Term, ToyVsReal, WhyExists } from '../components/ui'
import { Code } from '../components/Code'
import { Exercise, ExplainBack } from '../components/exercise'
import { ConfigReader } from '../interactive/ConfigReader'
import { TensorNameMatcher } from '../interactive/TensorNameMatcher'
import { RepoAnatomy } from '../illustrations/RepoAnatomy'

export default function PytorchBridgeLesson() {
  return (
    <Lesson id="pytorch-bridge">
      <Why>
        <p className="lede">Saturday morning. Dev is still asleep, the filter coffee is still hot, and Riya has a question she has been saving all week: would she recognise a real model if she opened one?</p>
        <p>Until now everything has been her own code. Kabir’s advice on Friday was short: “Open GPT-2. Don’t read about it. Open it.”</p>
        <p>So she goes to <a href="https://huggingface.co/openai-community/gpt2/tree/main" target="_blank" rel="noreferrer">huggingface.co/openai-community/gpt2</a> and clicks “Files”. Try it with her. You see this:</p>
        <Code lang="text" title="the files that matter in the GPT-2 repository">{`
config.json                   665 B
generation_config.json        124 B
merges.txt                    456 kB
vocab.json                   1.04 MB
tokenizer.json               1.36 MB
model.safetensors             548 MB
`}</Code>
        <p>Six files. No Python. Somewhere in there is a working language model.</p>
        <p>You have already built every part of it:</p>
        <ul>
          <li>the tokenizer in <a href="#/lesson/tokenization">Part 4</a>,</li>
          <li>the GPT in <a href="#/lesson/build-gpt">Part 7</a>,</li>
          <li>the sampling loop in <a href="#/lesson/inference">Inference</a>,</li>
          <li>LoRA in <a href="#/lesson/fine-tuning">Fine-tuning</a>.</li>
        </ul>
        <p>What you have not done is match your parts to theirs. A real open model is your <code>tiny_gpt.py</code> with different names, bigger numbers and a file format. After this lesson you can open <b>any</b> decoder-only model on the Hub, read its <code>config.json</code>, predict its size and memory, list its tensors and say which lesson built each one.</p>
        <p>We will do it properly. Load GPT-2. Recompute its 124,439,808 parameters from the config alone. Reproduce its logits by hand from the raw tensors. Then fine-tune it with the LoRA library people actually use.</p>
      </Why>

      <Problem title="What stands between your GPT and theirs?">
        <p>Dev wanders in, sees the file list and says, “So that’s the brain? I thought it would be, like, a program.” It is a fair surprise. What stands between Riya’s GPT and this one is not ideas. It is four practical things:</p>
        <ul>
          <li><b>Names.</b> Your fused QKV layer is <code>attn.qkv</code>. GPT-2 calls it <code>attn.c_attn</code>. Llama splits it into <code>q_proj</code>, <code>k_proj</code>, <code>v_proj</code>. Same maths, three vocabularies.</li>
          <li><b>Files.</b> A model is shipped as data, not as a program: which file holds what, and which can hurt you?</li>
          <li><b>Number formats.</b> At 8 billion parameters the format decides whether the model fits on your GPU at all.</li>
          <li><b>Libraries.</b> <code>from_pretrained</code>, <code>generate</code>, <code>get_peft_model</code> each wrap something you wrote by hand.</li>
        </ul>
        <WhyExists
          problem="A trained model must be handed to other people: a few hundred megabytes to hundreds of gigabytes of numbers, plus everything needed to use them."
          naive={<>Save the Python object. <code>torch.save(model)</code> uses Python’s <code>pickle</code>, which can store any object.</>}
          fails={<>A pickle is a small program that rebuilds the object, and loading it runs that program. A malicious checkpoint can execute any code on your machine. The Python documentation opens its pickle page with that warning: only unpickle data you trust. It also ties the file to one exact class definition.</>}
          idea={<>Split the model into plain data. A JSON file for the architecture’s numbers, plain files for the tokenizer, and a tensor file that holds only names, shapes and raw bytes. The code that gives them meaning lives in an open library.</>}
          tradeoff="Data without code is safe but mute. You need library code that matches the architecture, and every model family names its tensors its own way. That is why this lesson exists."
        />
      </Problem>

      <MentalModel title="A model repository, file by file">
        <RepoAnatomy />
        <p>Read it as four answers:</p>
        <ol>
          <li><b>Tokenizer files</b> answer “how does text become ids?”. <code>merges.txt</code> is the ordered merge list you trained in <a href="#/lesson/tokenization">Tokenization</a>; <code>vocab.json</code> maps each piece to its id. GPT-2 has 50,000 merges, 256 byte tokens and one special token: 50,257 ids.</li>
          <li><b>config.json</b> answers “what is the skeleton?”. It is your <code>Config</code> class as JSON. It contains no weights.</li>
          <li><b>model.safetensors</b> answers “what are the numbers?”. It is a dictionary from tensor name to array.</li>
          <li><b>generation_config.json</b> answers “how should the loop behave by default?”: which id ends generation, sometimes a default temperature or top-p.</li>
        </ol>
        <Callout kind="dev">
          Think of <code>config.json</code> as the schema and <code>model.safetensors</code> as the table dump. Neither contains the query engine: the forward pass lives in the library, in a class picked by <code>model_type</code>. That is also where the comparison stops. A schema fully describes a table. A config does not describe the computation: nothing in it says “pre-norm” or “causal mask”. Only the code does.
        </Callout>
        <Term
          name="safetensors"
          plain={<>A file format for tensors that holds data only. It starts with a JSON header that lists every tensor’s name, number type, shape and byte range. After the header come the raw bytes. Loading it cannot run code.</>}
          example={<>GPT-2’s file: 8 bytes of header length, 14,283 bytes of JSON, then 137,022,720 numbers × 4 bytes. Total 548,105,171 bytes, which is exactly the size the Hub shows.</>}
          formal={<>Because the header gives byte ranges, a loader can memory-map the file and read one tensor without reading the rest. The older <code>pytorch_model.bin</code> files are pickles. Since version 2.6, <code>torch.load</code> defaults to <code>weights_only=True</code>, which refuses arbitrary objects, but the simple rule stands: prefer safetensors, and never load a pickle from someone you do not trust.</>}
        />
        <h3>The tokenizer is a separate artefact, and it must match</h3>
        <p>The model only ever sees integers. Row 3797 of the embedding table means “␣cat” because GPT-2’s tokenizer says so. Feed the same text through another tokenizer and you look up the wrong rows: fluent nonsense, no error.</p>
        <p>The same holds for special tokens and chat formats. In <a href="#/lesson/training-pipeline">From raw text to assistant</a> you saw that an assistant is trained on conversations flattened by a <b>chat template</b>. That template ships with the tokenizer, and <code>apply_chat_template</code> applies it for you. We run it below.</p>
        <DeepDive title="Why can vocab_size in the config be larger than the tokenizer’s vocabulary?">
          <p>For GPT-2 the two agree: 50,257 and 50,257. For Qwen2.5 0.5B the config says <code>vocab_size: 151936</code> while the tokenizer has 151,665 entries. The embedding table has 271 rows no token id ever selects.</p>
          <p>Padding the table like this is common practice, for two usual reasons: matrix sizes that are multiples of a power of two tend to run faster on GPUs (151,936 = 128 × 1,187), and spare rows leave room to add special tokens later without resizing the model. The rule that matters is one-directional: every id the tokenizer can produce must have a row. The reverse is not required.</p>
        </DeepDive>
        <h3>Two dialects, one machine</h3>
        <p>This table is the heart of the lesson: what you wrote, GPT-2, and a Llama-style model (Llama, Mistral, Qwen and many others share these names).</p>
        <div className="table-scroll">
          <table className="plain" style={{ fontSize: 14 }}>
            <thead><tr><th>your tiny_gpt.py</th><th>GPT-2 tensor, shape</th><th>Llama 3 8B tensor, shape</th><th>built in</th></tr></thead>
            <tbody>
              <tr><td className="mono">tok_emb</td><td className="mono">transformer.wte (50257, 768)</td><td className="mono">model.embed_tokens (128256, 4096)</td><td><a href="#/lesson/embeddings">Embeddings</a></td></tr>
              <tr><td className="mono">pos_emb</td><td className="mono">transformer.wpe (1024, 768)</td><td>none: <G t="rope">RoPE</G> rotates Q and K instead</td><td><a href="#/lesson/transformer-block">Block</a>, <a href="#/lesson/modern-architecture">Modern</a></td></tr>
              <tr><td className="mono">blocks[i].ln1</td><td className="mono">h.i.ln_1 .weight .bias (768,)</td><td className="mono">layers.i.input_layernorm .weight (4096,)</td><td><a href="#/lesson/transformer-block">Block</a></td></tr>
              <tr><td className="mono">blocks[i].attn.qkv</td><td className="mono">h.i.attn.c_attn (768, 2304)</td><td className="mono">self_attn.q_proj (4096, 4096)<br />self_attn.k_proj (1024, 4096)<br />self_attn.v_proj (1024, 4096)</td><td><a href="#/lesson/attention">Attention</a>, <G t="gqa">GQA</G></td></tr>
              <tr><td className="mono">blocks[i].attn.proj</td><td className="mono">h.i.attn.c_proj (768, 768)</td><td className="mono">self_attn.o_proj (4096, 4096)</td><td><a href="#/lesson/masks-and-heads">Heads</a></td></tr>
              <tr><td className="mono">blocks[i].ln2</td><td className="mono">h.i.ln_2 (768,)</td><td className="mono">post_attention_layernorm (4096,)</td><td><a href="#/lesson/transformer-block">Block</a></td></tr>
              <tr><td className="mono">blocks[i].ffn.net[0]</td><td className="mono">h.i.mlp.c_fc (768, 3072)</td><td className="mono">mlp.gate_proj (14336, 4096)<br />mlp.up_proj (14336, 4096)</td><td><a href="#/lesson/transformer-block">Block</a>, <G t="swiglu">SwiGLU</G></td></tr>
              <tr><td className="mono">blocks[i].ffn.net[2]</td><td className="mono">h.i.mlp.c_proj (3072, 768)</td><td className="mono">mlp.down_proj (4096, 14336)</td><td><a href="#/lesson/transformer-block">Block</a></td></tr>
              <tr><td className="mono">ln_f</td><td className="mono">transformer.ln_f (768,)</td><td className="mono">model.norm (4096,)</td><td><a href="#/lesson/build-gpt">Build GPT</a></td></tr>
              <tr><td className="mono">head (tied)</td><td className="mono">lm_head = wte, not stored twice</td><td className="mono">lm_head (128256, 4096), its own matrix</td><td><a href="#/lesson/build-gpt">Build GPT</a></td></tr>
            </tbody>
          </table>
        </div>
        <p>Every shape in the GPT-2 column is printed by the script in this lesson. The Llama shapes were read from the header of the published checkpoint.</p>
        <Callout kind="warn" label="Careful: the one trap in GPT-2">
          Look at the shapes. Your <code>nn.Linear(768, 2304)</code> stores its weight as <b>(out, in)</b> = (2304, 768) and computes <code>x @ W.T + b</code>. GPT-2 uses its own layer class, <code>Conv1D</code>, which stores <b>(in, out)</b> = (768, 2304) and computes <code>x @ W + b</code>. Same maths, transposed storage. Llama uses plain <code>nn.Linear</code>, so its shapes read (out, in). Any tool that assumes <code>nn.Linear</code> must be told about GPT-2. You will see the flag for it in the LoRA config below.
        </Callout>
      </MentalModel>

      <TryIt title="Read a config, then name the tensors">
        <p>First the config. The presets are the real files from the Hub, in both dialects (GPT-2’s <code>n_embd</code>, <code>n_layer</code>; the Llama-style <code>hidden_size</code>, <code>num_key_value_heads</code> and the rest). Every field is explained in the words of the lesson that taught it. The text box is live: edit a number and everything below recomputes.</p>
        <ConfigReader />
        <p>Things worth noticing before you move on:</p>
        <ul>
          <li><b>Llama 3 8B comes to exactly 8,030,261,248.</b> That is the number the Hub reports for the checkpoint. You derived it from 23 lines of JSON.</li>
          <li>With <G t="rope">RoPE</G> the position table is 0 parameters. <code>max_position_embeddings</code> changes nothing in the count.</li>
          <li>In GPT-2 the token table is 31% of the model. In Llama 3 8B the two vocabulary matrices together are 13%. The MLPs dominate, as they did in <a href="#/lesson/build-gpt">Build GPT</a>.</li>
          <li>Mistral 7B and Llama 3 8B have <b>identical blocks</b>. The whole difference, 0.79B parameters, is vocabulary: 32,000 tokens against 128,256.</li>
        </ul>
        <p>Now the tensors. Shapes are clues: 2304 = 3 × 768. 1024 = 8 × 128.</p>
        <TensorNameMatcher />
      </TryIt>

      <Numbers title="Let’s see the numbers: will it fit?">
        <p>The question every engineer asks first, and the one Riya’s manager will ask on Monday. Take Llama 3 8B: 8,030,261,248 parameters, published in <b>bf16</b>.</p>
        <Term
          name="dtype, and why bf16"
          plain={<>The dtype is the number format of a tensor: how many bytes each number takes and how those bits are split between range and precision.</>}
          example={<>fp32: 4 bytes, largest value about 3.4 × 10³⁸. fp16: 2 bytes, largest value 65,504. bf16: 2 bytes, largest value about 3.4 × 10³⁸.</>}
          formal={<>fp32 has 1 sign, 8 exponent and 23 fraction bits. fp16 has 1, 5 and 10. bf16 (“brain float”) has 1, 8 and 7: <b>fp32’s exponent, so fp32’s range</b>, paid for with fewer digits. Range is what training needs: a gradient or activation above 65,504 becomes infinity in fp16, which is why fp16 training needs loss-scaling tricks and bf16 training mostly does not.</>}
        />
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>what</th><th>calculation</th><th>result</th></tr></thead>
            <tbody>
              <tr><td>Weights, bf16</td><td className="mono">8,030,261,248 × 2 bytes</td><td className="mono"><b>16.06 GB</b></td></tr>
              <tr><td>Weights, fp32</td><td className="mono">× 4 bytes</td><td className="mono">32.12 GB</td></tr>
              <tr><td>Weights, int4</td><td className="mono">× 0.5 bytes, plus small scales</td><td className="mono">about 4 GB</td></tr>
              <tr><td><G t="kv-cache">KV cache</G>, one token</td><td className="mono">2 × 32 layers × 8 KV heads × 128 × 2 bytes</td><td className="mono">131,072 B = 128 KiB</td></tr>
              <tr><td>KV cache, 8,192 tokens</td><td className="mono">131,072 × 8,192</td><td className="mono">1.07 GB per sequence</td></tr>
              <tr><td>Full fine-tuning with Adam</td><td className="mono">8.03B × 16 bytes, before activations</td><td className="mono"><b>128 GB</b></td></tr>
            </tbody>
          </table>
        </div>
        <p>You can check the first row against reality. The checkpoint’s index file states <code>"total_size": 16060522496</code>. That is 8,030,261,248 × 2, to the byte.</p>
        <p>So inference fits on one 24 GB GPU in bf16, with room for a few sequences of cache. Full fine-tuning does not fit on one 80 GB GPU, even before activations. That gap is the reason <G t="lora">LoRA</G> is everywhere.</p>
        <h3>Where do 16 bytes per parameter come from?</h3>
        <p>Inference needs the weights. Training needs the weights, a <G t="gradient">gradient</G> for every weight, and Adam’s two running averages for every weight (you met them in <a href="#/lesson/training-gpt">Training GPT</a>). In the usual mixed-precision setup:</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>kept per parameter</th><th>format</th><th>bytes</th></tr></thead>
            <tbody>
              <tr><td>weight used in the forward and backward pass</td><td>16-bit</td><td className="mono">2</td></tr>
              <tr><td>gradient</td><td>16-bit</td><td className="mono">2</td></tr>
              <tr><td>master copy of the weight, so tiny updates are not rounded away</td><td>32-bit</td><td className="mono">4</td></tr>
              <tr><td>Adam: running average of the gradient</td><td>32-bit</td><td className="mono">4</td></tr>
              <tr><td>Adam: running average of the squared gradient</td><td>32-bit</td><td className="mono">4</td></tr>
              <tr><td><b>total</b></td><td /><td className="mono"><b>16</b></td></tr>
            </tbody>
          </table>
        </div>
        <p>This accounting is from the ZeRO paper (Rajbhandari et al., 2019), which uses it to show that a 1.5B-parameter GPT-2 needs at least 24 GB of model state. It is the standard back-of-envelope estimate, not a law: activations come on top, and 8-bit optimisers or pure-bf16 training lower it.</p>
        <p>Now LoRA on GPT-2 from this lesson’s script. The frozen base needs only its weights: 124.4M × 4 bytes = 498 MB in fp32. Gradients and Adam state exist only for the adapter’s 442,368 numbers: about 5 MB. Activations still have to be stored for backprop, so LoRA does not make training free. It removes the largest fixed cost.</p>
      </Numbers>

      <TheMath>
        <p>The whole sizing exercise is three multiplications, worth writing down once:</p>
        <Equation
          label="Weight memory equals N times bytes per parameter. KV cache per token equals 2 times layers times KV heads times head size times bytes. Training memory is about 16 N bytes plus activations."
          symbols={[
            ['N', 'number of parameters, counted from config.json as the config reader does'],
            ['b', 'bytes per number: 4 for fp32, 2 for bf16 or fp16, 1 for int8, 0.5 for int4'],
            ['L', <>number of blocks: <code>num_hidden_layers</code> or <code>n_layer</code></>],
            [<>H<sub>kv</sub></>, <>key/value heads: <code>num_key_value_heads</code>. Equal to the number of heads unless the model uses <G t="gqa">GQA</G></>],
            [<>d<sub>head</sub></>, <>numbers per head: <code>hidden_size / num_attention_heads</code> unless the config gives <code>head_dim</code></>],
            ['2 ×', 'one key vector and one value vector per head, per layer, per token'],
            ['16', '2 + 2 + 4 + 4 + 4, the mixed-precision Adam accounting above'],
          ]}
        >
          weights = N · b<br />
          KV cache per token = 2 · L · H<sub>kv</sub> · d<sub>head</sub> · b<br />
          training ≈ 16 · N + activations
        </Equation>
      </TheMath>

      <CodeIt>
        <p>Two files. Install once with <code>pip install transformers peft</code>. The first run downloads GPT-2 (about 550 MB); no account, no API key.</p>
        <h3>1. Load it, and put it in the right mode</h3>
        <Code title="loading a model and its tokenizer">{`
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")             # the tokenizer files
model = AutoModelForCausalLM.from_pretrained("gpt2")    # config.json + model.safetensors
model.eval()            # dropout off: the same switch as in tiny_gpt.py

with torch.no_grad():   # do not record the forward pass: we will not call backward()
    logits = model(ids).logits       # (B, T, V), like your model(idx)
`}</Code>
        <p><code>from_pretrained</code> reads <code>model_type</code>, picks the class (<code>GPT2LMHeadModel</code>), builds the skeleton from the config, then fills it from the tensor file by name.</p>
        <ul>
          <li><b><code>model.eval()</code></b> switches dropout off. Forget it and you get different logits on every call. You will measure that below.</li>
          <li><b><code>torch.no_grad()</code></b> stops PyTorch keeping the recording it would need for <code>loss.backward()</code>. Same results, much less memory.</li>
          <li><b>Pass the dtype: <code>dtype=torch.bfloat16</code>.</b> Leave it out and an older library version may load an 8B model in fp32, silently taking 32 GB instead of 16. (transformers 4.x defaults to fp32; version 5 to the checkpoint’s own format. Before 4.56 the argument was spelled <code>torch_dtype</code>.)</li>
          <li><b><code>device_map="auto"</code></b> puts the model on the GPU first, then in CPU memory once the GPU is full (it needs <code>accelerate</code>). For a model that fits, <code>model.to("cuda")</code> does the same job.</li>
        </ul>
        <h3>2. Walk the tensors</h3>
        <p>The script prints GPT-2’s config next to your <code>Config</code>, then every tensor with the lesson that built it. This is its real output for block 0:</p>
        <Code lang="text" title="python phase6-engineering/inspect_hf_model.py (excerpt)">{`
transformer.wte.weight                (50257, 768)   token embedding table  = tok_emb
transformer.wpe.weight                (1024, 768)    position table         = pos_emb
transformer.h.0.ln_1.weight           (768,)         LayerNorm before attn  = Block.ln1
transformer.h.0.attn.c_attn.weight    (768, 2304)    fused Wq, Wk, Wv       = attn.qkv
transformer.h.0.attn.c_proj.weight    (768, 768)     output projection Wo   = attn.proj
transformer.h.0.ln_2.weight           (768,)         LayerNorm before MLP   = Block.ln2
transformer.h.0.mlp.c_fc.weight       (768, 3072)    MLP up, D -> 4D        = ffn.net[0]
transformer.h.0.mlp.c_proj.weight     (3072, 768)    MLP down, 4D -> D      = ffn.net[2]
transformer.ln_f.weight               (768,)         final LayerNorm        = GPT.ln_f

lm_head.weight (50257, 768) shares memory with wte.weight: True
formula total                             =  124,439,808
sum(p.numel() for p in model.parameters()) = 124,439,808
`}</Code>
        <p>The formula is the one from <a href="#/lesson/build-gpt">Build GPT</a>, fed with five numbers from <code>config.json</code>. The script asserts the two totals are equal.</p>
        <h3>3. The forward pass, by hand, on their weights</h3>
        <p>Recognising names is one thing. Proof is another. <code>manual_forward</code> takes the raw dictionary of tensors and recomputes GPT-2’s logits with your own lines. Here is one block of it:</p>
        <Code source="phase6-engineering/inspect_hf_model.py" title="manual_forward, inside the loop over blocks">{`
p = f"transformer.h.{i}."
# ---- communicate: x = x + attn(ln_1(x)) ----
h = layer_norm(x, sd[p + "ln_1.weight"], sd[p + "ln_1.bias"], eps)
qkv = h @ sd[p + "attn.c_attn.weight"] + sd[p + "attn.c_attn.bias"]   # (T, 3D) fused
q, k, v = qkv.split(D, dim=-1)                                       # each (T, D)
q = q.view(T, n_head, hd).transpose(0, 1)                            # (H, T, hd)
k = k.view(T, n_head, hd).transpose(0, 1)
v = v.view(T, n_head, hd).transpose(0, 1)
att = q @ k.transpose(-2, -1) / math.sqrt(hd)                        # (H, T, T)
att = att.masked_fill(~causal, float("-inf"))
att = F.softmax(att, dim=-1)
y = (att @ v).transpose(0, 1).reshape(T, D)                          # concat heads
x = x + y @ sd[p + "attn.c_proj.weight"] + sd[p + "attn.c_proj.bias"]
`}</Code>
        <p>Compare it with <code>CausalSelfAttention.forward</code> in your <code>tiny_gpt.py</code>. Line for line the same, except <code>h @ W</code> with no transpose: the Conv1D storage. The MLP half and the tied head follow the same way. Then:</p>
        <Code lang="text" title="real output">{`
prompt    'The cat sat on the'
tokens    ['The', ' cat', ' sat', ' on', ' the']
ids       [464, 3797, 3332, 319, 262]
logits shape (5, 50257)   largest |difference| over all 251,285 logits: 1.14e-04
allclose: True. Your maths, their weights, the same numbers.
top 5 next tokens (from the hand-made logits):
  ' floor'      7.6%
  ' bed'        6.5%
  ' couch'      5.4%
`}</Code>
        <p>A largest difference of 0.0001, on logits whose typical size is about 80, is floating-point rounding: the library’s fused attention kernel adds the same numbers in another order.</p>
        <h3>4. <code>model.generate</code> is your loop with knobs</h3>
        <Code source="phase6-engineering/inspect_hf_model.py" title="your sampling step, on their logits">{`
def sample_next(logits, temperature=1.0, top_k=None, generator=None):
    logits = logits / temperature
    if top_k is not None:
        kth = torch.topk(logits, top_k).values[-1]             # the k-th largest score
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator))
`}</Code>
        <p>Wrap that in “run the model, take the last row, sample, append” and you have <code>model.generate(ids, do_sample=True, temperature=0.8, top_k=40, max_new_tokens=25)</code>. The script proves it for greedy decoding: your loop and <code>generate(do_sample=False)</code> return identical ids.</p>
        <p>What <code>generate</code> adds is engineering you also know: the <G t="kv-cache">KV cache</G>, stopping at <code>eos_token_id</code>, batching with padding, and more knobs (<code>top_p</code>, <code>repetition_penalty</code>). <code>do_sample=False</code> is the default, and then <code>temperature</code> is ignored.</p>
        <h3>5. Chat models: let the tokenizer build the prompt</h3>
        <p>GPT-2 is a base model and has no chat template. An instruction-tuned model does. This is real output from the tokenizer of <code>Qwen/Qwen2.5-0.5B-Instruct</code>, a small open chat model:</p>
        <Code title="apply_chat_template">{`
messages = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is a cat?"}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# <|im_start|>system
# You are a helpful assistant.<|im_end|>
# <|im_start|>user
# What is a cat?<|im_end|>
# <|im_start|>assistant
`}</Code>
        <p><code>add_generation_prompt=True</code> appends the opening of the assistant’s turn, so the most likely continuation is an answer. The model stops by sampling <code>&lt;|im_end|&gt;</code>, this tokenizer’s end-of-sequence token. Build this string by hand with one wrong newline and quality drops quietly, so use the template (stored with the tokenizer, in <code>chat_template.jinja</code> or <code>tokenizer_config.json</code>).</p>
        <h3>6. LoRA with the real library</h3>
        <p>In <a href="#/lesson/fine-tuning">Fine-tuning</a> you wrote <code>LoRALinear</code>: a frozen layer plus a thin trainable path <code>(x @ A) @ B * (alpha / r)</code>, with B starting at zero. The <code>peft</code> library does exactly that. The second script fine-tunes GPT-2 to triage support tickets into a format it has never seen: <code>&lt;&lt;team:payments|priority:P1&gt;&gt;</code>.</p>
        <Code source="phase6-engineering/lora_finetune_hf.py" title="the whole LoRA setup">{`
config = LoraConfig(
    r=args.rank,                  # the rank: width of the thin path, your \`r\`
    lora_alpha=2 * args.rank,     # scale = lora_alpha / r = 2, exactly your alpha / r
    target_modules=["c_attn", "attn.c_proj"],   # fused QKV + Wo, the layers apply_lora() wrapped
    fan_in_fan_out=True,          # GPT-2's Conv1D stores weight as (in, out)
    lora_dropout=0.0,
    bias="none",                  # train no biases: detaching the adapter restores the base exactly
    task_type="CAUSAL_LM",
)
model = get_peft_model(base, config)   # freezes every base weight, wraps the targets
model.print_trainable_parameters()
`}</Code>
        <p><code>target_modules</code> are name endings from the tensor walk above. That is why this lesson made you learn the names. <code>"c_proj"</code> alone would also match <code>mlp.c_proj</code>. For a Llama-style model you would write <code>["q_proj", "k_proj", "v_proj", "o_proj"]</code>.</p>
        <p>Which layers do people adapt? The LoRA paper adapted only W<sub>q</sub> and W<sub>v</sub> in most experiments, and peft’s default for Llama is still <code>q_proj</code> and <code>v_proj</code>. The QLoRA paper later reported that adapters on <em>all</em> linear layers were needed to match full fine-tuning (peft: <code>target_modules="all-linear"</code>).</p>
        <Code lang="text" title="real output, default rank 8">{`
trainable params: 442,368 || all params: 124,882,176 || trainable%: 0.3542
your arithmetic: 12 blocks x (r x (D + 3D) + r x (D + D)) = 442,368
h.0.attn.c_attn is now a peft Linear holding:
  base_layer.weight (768, 2304)  frozen
  lora_A.default    (8, 768)  trainable, random start
  lora_B.default    (2304, 8)  trainable, ZERO start
`}</Code>
        <p>Your A was (768, 8). Theirs prints as (8, 768) because peft builds A and B as <code>nn.Linear</code> layers, which store (out, in). Same matrix, transposed storage, once again.</p>
        <p>The second idea in the file is <b>loss masking</b>. The ticket is given. Only the answer should be graded:</p>
        <Code
          source="phase6-engineering/lora_finetune_hf.py"
          title="grade the response, not the prompt"
          setup={`IGNORE = -100   # the value PyTorch's cross-entropy skips
# (masked_next_token_loss needs PyTorch, so only build_example runs here)
prompt_ids   = [464, 3797, 3332, 319, 262]   # "The cat sat on the"
response_ids = [2272, 13]                     # a made-up two-token answer
eos_id = 50256`}
          show={`input_ids, labels = build_example(prompt_ids, response_ids, eos_id)
print("input_ids:", input_ids)
print("labels:   ", labels)
print("graded positions:", sum(l != IGNORE for l in labels), "of", len(labels))`}
        >{`
def build_example(prompt_ids, response_ids, eos_id, max_len=128):
    input_ids = (list(prompt_ids) + list(response_ids) + [eos_id])[:max_len]
    labels = ([IGNORE] * len(prompt_ids) + list(response_ids) + [eos_id])[:max_len]
    return input_ids, labels

def masked_next_token_loss(logits, labels):
    V = logits.size(-1)
    return F.cross_entropy(logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1),
                           ignore_index=IGNORE)
`}</Code>
        <p><code>IGNORE</code> is −100, the value PyTorch’s cross-entropy skips. The slice <code>[:, :-1]</code> against <code>[:, 1:]</code> is the shift you made in <code>get_batch</code>: position t predicts token t + 1. The end-of-sequence token stays graded, so the model learns to stop.</p>
        <p>Measured on a laptop CPU, 300 steps, 94 seconds. Your numbers will differ slightly:</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th /><th>held-out loss on response tokens</th><th>held-out tickets, exact format</th><th>right team</th></tr></thead>
            <tbody>
              <tr><td>base GPT-2</td><td className="mono">6.137</td><td className="mono">0 of 8</td><td className="mono">0 of 8</td></tr>
              <tr><td>LoRA, 100 steps (<code>--quick</code>, 31 s)</td><td className="mono">0.147</td><td className="mono">8 of 8</td><td className="mono">5 of 8</td></tr>
              <tr><td>LoRA, 300 steps</td><td className="mono">0.073</td><td className="mono">8 of 8</td><td className="mono">7 of 8</td></tr>
              <tr><td>adapter switched off</td><td className="mono">6.137</td><td colSpan={2}>identical to base: no base weight moved</td></tr>
            </tbody>
          </table>
        </div>
        <p>Read it honestly. The <em>format</em> is learned almost at once: that is what fine-tuning is good at. Choosing the right team from 40 examples is harder and still imperfect. And 8 tickets is a smoke test, not an eval: <a href="#/lesson/evals">Evals</a> showed how wide the error bars on 8 items are.</p>
        <Term
          name="Adapter (as a file)"
          plain={<>The small set of extra weights that LoRA trained, saved without the base model. To use it you need the adapter and the exact base model it was trained on.</>}
          example={<>The script saves <code>adapter_config.json</code> (1 kB: rank, alpha, target modules, base model name) and <code>adapter_model.safetensors</code> (1,775,520 bytes). The base model is 498 MB in fp32: a ratio of 1 to 279.</>}
          formal={<>Attached: base and adapter stay separate, the adapter can be switched off or swapped per request, at the cost of a little extra compute. Merged (<code>merge_and_unload()</code>): scale · A·B is added into each frozen weight, giving a plain model of the original shape with no extra latency, which can no longer be detached. The script does both and checks the logits agree.</>}
        />
        <DeepDive title="Two flags you will meet in every real training script">
          <ul>
            <li><b>Mixed precision</b> (<code>--bf16</code>): <code>torch.autocast</code> runs the matrix multiplies in bf16, while the weights being trained and the optimiser state stay in fp32. On GPUs with bf16 support it roughly halves activation memory and speeds up the multiplies. The script ignores the flag elsewhere, because CPUs without native bf16 emulate it slowly.</li>
            <li><b>Gradient accumulation</b> (<code>--grad-accum N</code>): call <code>backward()</code> on N small batches before one optimiser step. Gradients add up, so it behaves like a batch N times larger while only one small batch of activations is in memory.</li>
          </ul>
        </DeepDive>
      </CodeIt>

      <BreakIt>
        <p>Predict first, then run. Every result below was measured with the scripts in this lesson.</p>
        <ul>
          <li><b>Forget <code>model.eval()</code>.</b> Call <code>model.train()</code>, then run the same prompt twice under <code>no_grad</code>. The logits differ by up to 27 between the two calls: dropout is zeroing a random 10% of activations each time. In eval mode the difference is exactly 0.</li>
          <li><b>Use the wrong GELU.</b> In <code>manual_forward</code>, replace <code>gelu_new</code> with <code>F.gelu</code>. The largest logit error grows from 0.0001 to 0.075. The top prediction survives, but <code>allclose</code> fails. The config said <code>"activation_function": "gelu_new"</code> for a reason.</li>
          <li><b>Transpose like <code>nn.Linear</code>.</b> Write <code>h @ W.T</code> for <code>c_attn</code>: a shape error, (5, 768) against (2304, 768). Good. Now do it for <code>attn.c_proj</code>, which is square, 768 × 768. No error. Wrong numbers. Square matrices hide transposition bugs.</li>
          <li><b>Drop <code>fan_in_fan_out=True</code></b> from the LoraConfig. Current peft versions notice the Conv1D, warn, and set it for you. Do not rely on that: a merge with the wrong orientation would add the transpose of the correction.</li>
          <li><b>Grade the prompt too.</b> Replace the −100 labels with the real ids. The base model’s held-out loss reads 5.875 instead of 6.137, averaged over 223 tokens instead of 92. Most of what you would be optimising is “predict the customer’s ticket”, which is not the task.</li>
          <li><b>Swap tokenizers.</b> Encode the prompt with a different model’s tokenizer and feed those ids to GPT-2. It runs. It answers about something else.</li>
        </ul>
      </BreakIt>

      <Exercises>
        <Exercise
          id="pytorch-bridge-calc-kproj"
          type="calculate"
          title="The size of one K matrix under GQA"
          answer={{ value: 4194304, tolerance: 0 }}
          answerLabel="parameters in k_proj"
          hints={[
            'The input width is hidden_size. The output width is the number of key/value heads times the head size.',
            'Head size = hidden_size / num_attention_heads = 4096 / 32 = 128. There are 8 key/value heads.',
            'Output width = 8 × 128 = 1024. There is no bias. So the matrix is 1024 × 4096.',
          ]}
          solution={<><p>8 × 128 = 1,024 outputs, 4,096 inputs, no bias: 1,024 × 4,096 = <b>4,194,304</b>.</p><p>q_proj is 4,096 × 4,096 = 16,777,216, four times larger. That factor of four is the ratio of query heads to key/value heads (32 / 8), and it is exactly the factor by which GQA shrinks the KV cache.</p></>}
        >
          <p>A config says <code>hidden_size: 4096</code>, <code>num_attention_heads: 32</code>, <code>num_key_value_heads: 8</code>, <code>attention_bias: false</code>. How many parameters does one <code>k_proj</code> have?</p>
        </Exercise>

        <Exercise
          id="pytorch-bridge-trace-137m"
          type="trace"
          title="The Hub says 137M. The paper says 124M."
          hints={[
            'The Hub counts every number in model.safetensors. model.parameters() counts trainable parameters.',
            'The difference is 137,022,720 − 124,439,808 = 12,582,912. Divide it by 12 blocks.',
            '12,582,912 / 12 = 1,048,576 = 1024 × 1024. What in your CausalSelfAttention is a context × context grid of ones and zeros?',
          ]}
          solution={<><p>1,048,576 = 1024 × 1024 is the <b>causal mask</b>. In <code>tiny_gpt.py</code> you stored it with <code>register_buffer("mask", ...)</code>: a tensor that belongs to the module but is not a parameter. GPT-2’s original checkpoint saved that buffer for every block, as <code>h.N.attn.bias</code> (a confusing name: it is the mask, not a bias vector). 12 masks × 1,048,576 = 12,582,912 extra numbers in the file.</p><p>The lesson: a checkpoint file can hold things that are not parameters, and a name alone can mislead. The shape (1, 1, 1024, 1024) gives it away.</p></>}
        >
          <p>The GPT-2 model page reports 137M parameters for <code>model.safetensors</code>. The script counts 124,439,808 and the formula agrees. Nobody is wrong. Explain the 12,582,912 extra numbers.</p>
        </Exercise>

        <Exercise
          id="pytorch-bridge-implement-targets"
          type="implement"
          title="Adapt the MLP too"
          answer={{ value: 811008, tolerance: 0 }}
          answerLabel="trainable parameters"
          hints={[
            'An adapter on a d_in → d_out layer trains r × (d_in + d_out) numbers. c_fc maps 768 → 3072.',
            'Extra per block: 8 × (768 + 3072) = 30,720. Times 12 blocks = 368,640.',
            'Add that to the 442,368 you already have. Then update expected_lora_params() in the script so that its assert passes again.',
          ]}
          solution={<><p>442,368 + 12 × 8 × (768 + 3,072) = 442,368 + 368,640 = <b>811,008</b>, and <code>print_trainable_parameters()</code> confirms it (0.65% of the model).</p><p>The script’s assert fails until you add <code>r * (D + 4 * D)</code> per block to <code>expected_lora_params</code>. That is deliberate: being able to predict the library’s number means you understand what it wrapped. In practice, adapting every linear layer is common advice. The QLoRA paper reports that it was needed to match full fine-tuning in their experiments.</p></>}
        >
          <p>In <code>lora_finetune_hf.py</code>, add <code>"c_fc"</code> to <code>target_modules</code>, keeping rank 8. Before running it: how many trainable parameters will <code>print_trainable_parameters()</code> report? Then run it with <code>--quick</code> and make the script’s own arithmetic check pass.</p>
        </Exercise>

        <ExplainBack
          id="pytorch-bridge-explain"
          prompt="A teammate downloads a 1.8 MB file called adapter_model.safetensors from a colleague and asks: “Is this the fine-tuned model? How can it be so small, and is it safe to load?” Answer all three questions."
          modelAnswer={<p>It is not a model. It is a LoRA adapter: the small A and B matrices that were trained while the base model stayed frozen. For rank 8 on GPT-2’s attention layers that is 442,368 numbers against 124 million, so about 1.8 MB against 500 MB. To use it you also need the exact base model named in <code>adapter_config.json</code>: the library loads the base, wraps the target layers, and adds the thin path x·A·B·(alpha/r) beside each one. You can keep it attached (swappable, can be switched off) or merge it into the weights (a plain model, no extra latency). On safety: safetensors holds only a JSON header and raw numbers, so loading it cannot execute code, unlike a pickled <code>.bin</code> or <code>.pt</code> file. It can still change the model’s behaviour in ways you did not intend, so you evaluate it like any other change.</p>}
        />
        <details className="deep">
          <summary>More practice (optional)</summary>
          <div className="details-body">
            <Exercise
              id="pytorch-bridge-calc-fit"
              type="calculate"
              title="Will Mistral 7B fit?"
              answer={{ value: 14.5, tolerance: 0.2 }}
              answerLabel="GB, one decimal"
              hints={[
                'The config reader gives the exact parameter count for the Mistral preset.',
                '7,241,732,096 parameters, 2 bytes each in bf16.',
                '7,241,732,096 × 2 = 14,483,464,192 bytes. Divide by 10⁹.',
              ]}
              solution={<><p>7,241,732,096 × 2 bytes = 14.48 GB, so about <b>14.5 GB</b>.</p><p>It fits on a 16 GB GPU only on paper: the KV cache and activations need room too. At 131,072 bytes per token, 1.5 GB of headroom is about 11,000 cached tokens across all concurrent requests. This is the arithmetic behind “use a 24 GB card or quantise to int8”.</p></>}
            >
              <p>How many GB (10⁹ bytes) do the weights of Mistral 7B v0.1 take in bf16? Use the config reader for the parameter count.</p>
            </Exercise>

            <Exercise
              id="pytorch-bridge-debug-eval"
              type="debug"
              title="The flaky evaluation"
              hints={[
                'The same prompt, the same weights, greedy decoding, and still different answers. What in the forward pass is random?',
                'Look at what is missing between from_pretrained and the loop. In this snippet the model was just fine-tuned, so it was last put in which mode?',
              ]}
              solution={<><p>The model is still in <b>train mode</b>, so dropout is active and every forward pass is different, even with greedy decoding. Add <code>model.eval()</code> before evaluating (and <code>model.train()</code> again afterwards if training continues). Wrapping the evaluation in <code>torch.no_grad()</code> is good for memory, but it does not switch dropout off: the two switches are independent.</p><p><code>from_pretrained</code> returns a model in eval mode, which is why people forget: the bug only appears after a training loop has called <code>model.train()</code>.</p></>}
            >
              <p>A colleague fine-tunes a model, then scores it. The score changes by several points on every run, although decoding is greedy. Why?</p>
              <Code>{`
for step in range(steps):
    train_step(model, batch)          # model.train() was called earlier

with torch.no_grad():
    score = evaluate(model, held_out) # greedy decoding inside
`}</Code>
            </Exercise>
          </div>
        </details>
      </Exercises>

      <CheckYourself
        questions={[
          {
            q: 'GPT-2’s c_attn.weight has shape (768, 2304), while nn.Linear(768, 2304).weight has shape (2304, 768). What follows?',
            options: ['GPT-2 computes a different function from your fused qkv layer', 'Same function; the storage is transposed, and tools that assume nn.Linear must be told', 'GPT-2 has three times as many attention parameters as your layer', 'One of the two checkpoints must be corrupted or mislabelled'],
            answer: 1,
            explain: 'Conv1D stores (in, out) and computes x @ W + b. nn.Linear stores (out, in) and computes x @ W.T + b. That is what fan_in_fan_out=True tells peft.',
          },
          {
            q: 'Why is bf16 usually preferred over fp16 for training, although both take 2 bytes?',
            options: ['bf16 keeps more significant digits than fp16 does', 'bf16 has fp32’s exponent range, so large values do not overflow to infinity', 'bf16 is the only 16-bit format that GPUs can multiply', 'bf16 files are compressed, so checkpoints are smaller'],
            answer: 1,
            explain: 'bf16 spends 8 bits on the exponent, like fp32, and only 7 on the fraction. fp16 overflows above 65,504. bf16 actually has fewer digits than fp16; range is what training needs.',
          },
          {
            q: 'In the fine-tuning data, labels are −100 on the prompt tokens. What would change if they held the real token ids instead?',
            options: ['Nothing: cross-entropy ignores the prompt positions anyway', 'The model would also be trained to predict the user’s text, diluting the signal about how to answer', 'The model could no longer attend to the prompt during training', 'Training would crash, because labels must contain a −100 entry'],
            answer: 1,
            explain: 'The mask affects only which positions are graded, not what the model can see. Without it, most of the gradient goes into modelling prompts.',
          },
        ]}
      />

      <Remember
        items={[
          <>A model repo is <b>data, not code</b>: <code>config.json</code> is the skeleton, tokenizer files turn text into ids, <code>model.safetensors</code> holds named tensors, <code>generation_config.json</code> holds loop defaults. The forward pass lives in the library.</>,
          <>Two dialects, one machine: <code>wte / c_attn / c_proj / c_fc / ln_f</code> (GPT-2) and <code>embed_tokens / q_proj k_proj v_proj o_proj / gate_proj up_proj down_proj / norm</code> (Llama-style). <b>Name plus shape</b> identifies any tensor.</>,
          <>From the config alone you can compute parameters exactly (GPT-2: 124,439,808; Llama 3 8B: 8,030,261,248), weights = N × bytes, KV cache per token = 2 · L · H<sub>kv</sub> · d<sub>head</sub> · bytes, and training ≈ 16 bytes per parameter plus activations.</>,
          <><code>generate()</code> is your loop with a KV cache, and <code>peft</code>’s LoRA is your <code>LoRALinear</code> plus bookkeeping. <code>model.eval()</code> (dropout off) and <code>torch.no_grad()</code> (recording off) are independent switches. Prefer <b>safetensors</b>: a pickle can run code.</>,
        ]}
      />

      <RealLLM>
        <ToyVsReal
          toy={<ul><li><code>tiny_gpt.py</code>: 0.8M parameters, fp32, one file, names you chose</li><li>GPT-2 small in this lesson: 124M, fp32, runs on a laptop CPU</li><li>LoRA on 40 tickets, 8 held out, checked by eye</li><li>Your loop re-runs the whole sequence for each new token</li></ul>}
          real={<ul><li>Billions of parameters in bf16, sharded across several safetensors files with an index</li><li>Served in bf16, FP8 or a 4-bit format (int4, MXFP4, NVFP4), by engines such as the ones in <a href="#/lesson/inference-systems">Inference systems</a></li><li>Thousands to millions of examples, a held-out set sized as in <a href="#/lesson/evals">Evals</a>, often LoRA on all linear layers over a 4-bit base (QLoRA)</li><li>KV cache, batching, stop tokens and chat templates handled by the serving stack</li></ul>}
        />
        <p>The names and shapes in this lesson are read from the published files, and the architecture code is open: you can read <code>modeling_gpt2.py</code> and <code>modeling_llama.py</code> in the transformers repository and find every line of your own GPT in them. For closed models such as Claude, GPT or Gemini none of this is published; what you can say is that openly released models from many labs share this layout.</p>
        <Callout kind="warn" label="Careful: two things that can still run code">A safetensors file cannot execute anything. Two other things can. Pickled checkpoints (<code>.bin</code>, <code>.pt</code>, <code>.ckpt</code>). And <code>trust_remote_code=True</code>, which downloads and runs Python from the model’s repository because the architecture is not in the library yet. Read that code, or pin a revision you have read, before you pass the flag.</Callout>
        <Callout kind="model" label="Simplified: what this lesson left out">Dense models only. A mixture-of-experts config has many MLPs per block and uses a few per token, so “parameters” and “parameters used per token” diverge; the config reader refuses those rather than guess. Some families add small things the Llama rules do not know about (extra norms, biases without a config field, a vocabulary padded beyond the tokenizer’s size), so for an unfamiliar <code>model_type</code> treat the count as an estimate and check it against the Hub’s number.</Callout>
        <p>By lunch the coffee is cold and Riya’s notebook has a two-column list: her names on the left, theirs on the right. Not one row is a mystery. Dev reads it over her shoulder and, for once, has no theory.</p>
      </RealLLM>
    </Lesson>
  )
}
