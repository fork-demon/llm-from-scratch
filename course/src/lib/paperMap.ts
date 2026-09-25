// A guided map of "Attention Is All You Need" (Vaswani et al., 2017, arXiv:1706.03762):
// the paper's real section numbering, each section paraphrased, tied to the lesson that taught it,
// with what differs from the decoder-only GPT built in this course. Facts were checked against
// version 7 of the paper on arXiv. Summaries are paraphrases; only equations are quoted.
// Also: the reading path for what to read next, and the per-section progress marks.

export const PAPER_URL = 'https://arxiv.org/abs/1706.03762'

export interface BuiltIn { lesson: string; note: string }

export interface PaperSection {
  id: string
  number: string // as printed in the paper; '' for the abstract
  title: string
  depth: 0 | 1 | 2
  summary: string[] // two or three plain sentences
  look: string // the exact figure, table or equation to look at
  equation?: 'attention' | 'multihead' | 'ffn' | 'pe' | 'lrate' | 'postnorm' // rendered by the component
  built: BuiltIn[]
  differs: string[]
}

export const SECTIONS: PaperSection[] = [
  {
    id: 'abstract', number: '', title: 'Abstract', depth: 0,
    summary: [
      'The claim in one paragraph: a sequence-to-sequence model built only from attention, with no recurrence and no convolution, translates better than the best previous systems and trains far faster.',
      'The evidence offered: 28.4 BLEU on WMT 2014 English to German and 41.8 on English to French, the latter after 3.5 days of training on eight GPUs.',
    ],
    look: 'The two BLEU numbers and the training time. A first pass is mostly about finding the claim and the evidence.',
    built: [{ lesson: 'attention', note: 'the mechanism the title is about' }],
    differs: ['The task is translation: read a whole source sentence, then write a target sentence. You built a model that continues one stream of text.'],
  },
  {
    id: 's1', number: '1', title: 'Introduction', depth: 0,
    summary: [
      'Recurrent networks were the standard for sequence tasks in 2017. They process position t only after position t − 1, so the work inside one training example cannot be spread across a GPU.',
      'Attention already existed, but as an add-on to recurrent networks. The proposal is to keep the attention and drop the recurrence.',
    ],
    look: 'The last paragraph: it states the contribution and the headline cost, twelve hours on eight P100 GPUs.',
    built: [{ lesson: 'context-wall', note: 'why a fixed running summary breaks' }, { lesson: 'attention', note: 'the “why does this exist” ladder tells the same story' }],
    differs: ['Nothing differs. This is the motivation you met before building attention.'],
  },
  {
    id: 's2', number: '2', title: 'Background', depth: 0,
    summary: [
      'Earlier attempts to remove recurrence used convolutions. With those, relating two distant positions takes more layers the further apart they are.',
      'Self-attention relates any two positions in one step. The price is that averaging over positions blurs detail, which the authors counter with several heads.',
    ],
    look: 'The sentence that introduces the term self-attention (the paper also calls it intra-attention).',
    built: [{ lesson: 'masks-and-heads', note: 'why run attention several times in parallel' }],
    differs: ['Related-work sections are written for reviewers. On a second pass, skim it and mark the two or three citations you may need later.'],
  },
  {
    id: 's3', number: '3', title: 'Model Architecture', depth: 0,
    summary: [
      'The model has two halves. An encoder reads the source sentence and turns it into one vector per source token. A decoder writes the target sentence one token at a time, looking at what it has written so far and at the encoder’s vectors.',
      'The decoder is auto-regressive: each generated token is fed back in as input. That part is exactly your generation loop.',
    ],
    look: 'Figure 1. It is the most reproduced diagram in the field: encoder on the left, decoder on the right.',
    built: [{ lesson: 'build-gpt', note: 'the right half of Figure 1, minus one sub-layer' }, { lesson: 'next-token', note: 'the auto-regressive loop' }],
    differs: ['You built a decoder-only model: the right half of Figure 1 without the middle sub-layer. GPT-style models drop the encoder and treat the prompt and the answer as one sequence.'],
  },
  {
    id: 's31', number: '3.1', title: 'Encoder and Decoder Stacks', depth: 1,
    summary: [
      'Each half is a stack of N = 6 identical layers, and every vector has d_{model} = 512 numbers. An encoder layer has two sub-layers: self-attention, then a feed-forward network.',
      'A decoder layer has three: masked self-attention, attention over the encoder’s output, then the feed-forward network. Every sub-layer is wrapped in a residual connection followed by layer normalisation.',
    ],
    look: 'The expression LayerNorm(x + Sublayer(x)). The order of those two operations is the detail to notice.',
    equation: 'postnorm',
    built: [{ lesson: 'transformer-block', note: 'residual connections and LayerNorm' }],
    differs: [
      'Post-norm against pre-norm. The paper normalises after adding the residual. Your Block normalises before each sub-layer, x + Sublayer(LayerNorm(x)), as GPT-2 and nearly all later models do, because deep stacks train more stably that way.',
      'Your block has two sub-layers, like the paper’s encoder layer, but with the decoder’s causal mask.',
    ],
  },
  {
    id: 's32', number: '3.2', title: 'Attention', depth: 1,
    summary: [
      'Attention is defined in general terms: a query and a set of key-value pairs go in, and a weighted sum of the values comes out. The weight on each value comes from how well the query matches that value’s key.',
    ],
    look: 'Figure 2: the left panel is one attention computation, the right panel is several of them in parallel.',
    built: [{ lesson: 'attention', note: 'query, key, value and the soft lookup' }],
    differs: ['Nothing differs. The general definition does not say where queries, keys and values come from, and that freedom is what section 3.2.3 uses.'],
  },
  {
    id: 's321', number: '3.2.1', title: 'Scaled Dot-Product Attention', depth: 2,
    summary: [
      'The formula you coded. Dot every query with every key, divide by √d_k, apply softmax, use the result to blend the values.',
      'The paper compares this with additive attention, which scores matches with a small network, and prefers dot products because they are one fast matrix multiply.',
    ],
    look: 'Equation 1, and footnote 4 underneath it, which explains the √d_k.',
    equation: 'attention',
    built: [{ lesson: 'attention', note: 'the same four steps, in NumPy' }, { lesson: 'softmax', note: 'why scores must stay small' }],
    differs: ['Nothing differs. Note the authors’ wording about the scaling: they “suspect” large dot products push softmax into regions with tiny gradients. Footnote 4 gives the argument: with independent components of mean 0 and variance 1, q · k has variance d_k.'],
  },
  {
    id: 's322', number: '3.2.2', title: 'Multi-Head Attention', depth: 2,
    summary: [
      'Run h = 8 attentions in parallel, each on its own learned projection of the vectors down to d_k = d_v = 64 numbers, then concatenate the results and mix them with one more matrix, W^O.',
      'Because each head is smaller, the total cost is about that of one full-width head.',
    ],
    look: 'The two-line MultiHead definition and the list of matrix shapes under it.',
    equation: 'multihead',
    built: [{ lesson: 'masks-and-heads', note: 'split, attend, concatenate, project' }, { lesson: 'build-gpt', note: 'the fused qkv layer computes all W_i at once' }],
    differs: ['The paper writes a separate W_i^Q, W_i^K, W_i^V per head. Your code uses one fused matrix and slices it. Stacking the eight 512 × 64 matrices side by side gives exactly one 512 × 512 matrix, so the two are the same computation.'],
  },
  {
    id: 's323', number: '3.2.3', title: 'Applications of Attention in our Model', depth: 2,
    summary: [
      'The same attention function is used three ways. In the encoder, queries, keys and values all come from the source, with no mask. In the decoder, they all come from the target so far, with the causal mask.',
      'In between sits encoder-decoder attention: queries come from the decoder, keys and values come from the encoder’s output. This is how every target position can look at every source position.',
    ],
    look: 'The three bullet points. The first one is the new idea for you.',
    built: [{ lesson: 'masks-and-heads', note: 'the causal mask: the decoder’s self-attention' }],
    differs: ['Cross-attention is the one mechanism in this paper that you have not built. It is taught in this lesson, with a picture. Decoder-only models do not need it: the prompt sits in the same sequence, so ordinary masked self-attention already reaches it.'],
  },
  {
    id: 's33', number: '3.3', title: 'Position-wise Feed-Forward Networks', depth: 1,
    summary: [
      'After attention, each position passes through the same small two-layer network on its own: up from 512 to d_{ff} = 2048 numbers, ReLU, back down to 512.',
    ],
    look: 'Equation 2. “Position-wise” means no information moves between tokens here.',
    equation: 'ffn',
    built: [{ lesson: 'neurons', note: 'a two-layer network with ReLU' }, { lesson: 'transformer-block', note: 'the compute half of the block' }],
    differs: ['ReLU here, GELU in your tiny GPT and GPT-2, SwiGLU in Llama-style models. The 4× widening (2048 = 4 × 512) is the ratio you used.'],
  },
  {
    id: 's34', number: '3.4', title: 'Embeddings and Softmax', depth: 1,
    summary: [
      'Tokens become vectors through a learned embedding table, and the decoder’s output becomes next-token probabilities through a linear layer and softmax.',
      'One weight matrix is shared three ways: both embedding tables and the pre-softmax layer. The embeddings are multiplied by √d_{model}.',
    ],
    look: 'The sentence about sharing the weight matrix.',
    built: [{ lesson: 'embeddings', note: 'the lookup table' }, { lesson: 'build-gpt', note: 'the head, and weight tying' }],
    differs: ['Your model ties two matrices (token table and head). The paper ties three, which works because source and target share one vocabulary. GPT-2 does not scale embeddings by √d_{model}.'],
  },
  {
    id: 's35', number: '3.5', title: 'Positional Encoding', depth: 1,
    summary: [
      'Attention ignores order, so position has to be added. The paper adds a fixed pattern of sines and cosines of different wavelengths to the embeddings. Nothing is learned.',
      'They also tried a learned position table and report nearly identical results (Table 3, row E). They kept the sinusoids in the hope of handling sequences longer than those seen in training.',
    ],
    look: 'The two PE formulas, then Table 3 row E.',
    equation: 'pe',
    built: [{ lesson: 'transformer-block', note: 'why attention needs position, and the learned table' }, { lesson: 'modern-architecture', note: 'RoPE, which also uses rotations at many wavelengths' }],
    differs: ['Sinusoidal here, a learned table in your GPT and GPT-2, RoPE in Llama-style models. Notice how carefully the paper hedges: the sinusoids “may” allow extrapolation. In practice none of the three reaches far beyond the training length without extra tricks.'],
  },
  {
    id: 's4', number: '4', title: 'Why Self-Attention', depth: 0,
    summary: [
      'The argument for the design, as a table. Self-attention costs O(n² · d) per layer, against O(n · d²) for a recurrent layer, but needs O(1) sequential steps instead of O(n), and connects any two positions in one hop.',
      'Self-attention is cheaper than recurrence when the sequence length n is smaller than the width d, which held for sentences in 2017.',
    ],
    look: 'Table 1. Read it column by column: work per layer, steps that cannot be parallelised, longest path between two positions.',
    built: [{ lesson: 'attention', note: 'the T² cost and the missing loop' }, { lesson: 'inference', note: 'where T² bites in practice' }],
    differs: ['The trade looks different today. Contexts of 100,000 tokens put n far above d, so the n² term dominates, which is why FlashAttention, GQA and sliding windows exist. The “restricted” row of the table anticipates sliding-window attention.'],
  },
  {
    id: 's5', number: '5', title: 'Training (5.1 to 5.4)', depth: 0,
    summary: [
      'Data: WMT 2014, about 4.5 million English to German sentence pairs with a shared byte-pair vocabulary of about 37,000 tokens, and 36 million sentences for English to French. Hardware: one machine with 8 P100 GPUs. The base model ran 100,000 steps in 12 hours, the big model 300,000 steps in 3.5 days.',
      'Optimiser: Adam with β1 = 0.9, β2 = 0.98, ε = 10⁻⁹, and a learning rate that rises linearly for 4,000 warm-up steps, then decays with the inverse square root of the step number.',
      'Regularisation: dropout of 0.1 on every sub-layer output and on the embedding sums, and label smoothing of 0.1.',
    ],
    look: 'Equation 3, the learning-rate schedule. Then the sentence on label smoothing: it hurts perplexity and improves BLEU.',
    equation: 'lrate',
    built: [{ lesson: 'training-gpt', note: 'AdamW, batches, dropout, train against validation loss' }, { lesson: 'tokenization', note: 'byte-pair encoding' }, { lesson: 'gradient-descent', note: 'what a learning rate is' }],
    differs: [
      'Your run used a constant learning rate of 3e-4. Warm-up followed by decay is what nearly every large run uses, usually with a cosine decay instead of this formula.',
      'β2 = 0.98 is lower than Adam’s default of 0.999. Label smoothing is new to you: it is defined in this lesson.',
    ],
  },
  {
    id: 's6', number: '6', title: 'Results (6.1 to 6.3)', depth: 0,
    summary: [
      '6.1: the big model reaches 28.4 BLEU on English to German, more than 2 BLEU above the best previous result, and 41.8 on English to French, at a fraction of the training cost. The base model scores 27.3 and 38.1.',
      '6.2: the ablations. One head is 0.9 BLEU worse than the best setting, and too many heads also hurts. Smaller keys hurt. Bigger models are better. Dropout helps. Learned positions match sinusoids.',
      '6.3: the same architecture does well on English constituency parsing, a different kind of task.',
    ],
    look: 'Table 2 for the claim, Table 3 for the evidence about why. Table 3 is where a practitioner learns the most.',
    built: [{ lesson: 'evals', note: 'how to read a score sceptically' }, { lesson: 'masks-and-heads', note: 'why more than one head' }],
    differs: [
      'BLEU scores overlap with reference translations. You evaluated with validation loss and with task-specific checks.',
      'A detail for sceptical readers: in the current arXiv version the abstract and Table 2 say 41.8 for English to French, while the running text of 6.1 says 41.0. Tables are usually the more carefully maintained source.',
    ],
  },
  {
    id: 's7', number: '7', title: 'Conclusion', depth: 0,
    summary: [
      'A restatement: the first sequence transduction model built entirely on attention, faster to train and better at translation.',
      'The plans: other modalities such as images and audio, and restricted attention for long inputs. The code was released in the tensor2tensor library.',
    ],
    look: 'The future-work sentences. Read in hindsight, they list most of what happened next.',
    built: [{ lesson: 'modern-architecture', note: 'what the field changed afterwards' }],
    differs: ['The conclusion does not predict the largest consequence: that the decoder half alone, scaled up and trained on next-token prediction, would become the general-purpose model you have been studying.'],
  },
]

export const sectionById = (id: string): PaperSection | undefined => SECTIONS.find((s) => s.id === id)

/* ------------------------------------------------------------------ */
/* Progress marks, stored per section                                   */
/* ------------------------------------------------------------------ */

export const MARKS = ['none', 'read', 'understood', 're-derived'] as const
export type Mark = (typeof MARKS)[number]
export type Marks = Record<string, Mark>
export const MARKS_KEY = 'llm-course:paper-map:v1'

/** Keep only known section ids and known marks, so stale or hand-edited storage cannot break the page. */
export const sanitiseMarks = (raw: unknown): Marks => {
  const out: Marks = {}
  if (typeof raw !== 'object' || raw === null) return out
  for (const s of SECTIONS) {
    const v = (raw as Record<string, unknown>)[s.id]
    if (typeof v === 'string' && (MARKS as readonly string[]).includes(v) && v !== 'none') out[s.id] = v as Mark
  }
  return out
}

type StorageLike = Pick<Storage, 'getItem' | 'setItem'>

export const loadMarks = (storage: StorageLike | undefined): Marks => {
  try {
    const raw = storage?.getItem(MARKS_KEY)
    return raw ? sanitiseMarks(JSON.parse(raw)) : {}
  } catch {
    return {} // private mode, blocked storage, or corrupted JSON: start empty
  }
}

export const saveMarks = (storage: StorageLike | undefined, marks: Marks): boolean => {
  try {
    storage?.setItem(MARKS_KEY, JSON.stringify(marks))
    return !!storage
  } catch {
    return false
  }
}

export const setMark = (marks: Marks, id: string, mark: Mark): Marks => {
  const next = { ...marks }
  if (mark === 'none') delete next[id]
  else next[id] = mark
  return next
}

export const countMarks = (marks: Marks): Record<Exclude<Mark, 'none'>, number> => {
  const rank = (m: Mark) => MARKS.indexOf(m)
  const at = (level: Mark) => SECTIONS.filter((s) => rank(marks[s.id] ?? 'none') >= rank(level)).length
  return { read: at('read'), understood: at('understood'), 're-derived': at('re-derived') }
}

/* ------------------------------------------------------------------ */
/* What to read next                                                    */
/* ------------------------------------------------------------------ */

export interface NextPaper {
  short: string
  title: string
  authors: string
  year: number
  url: string
  id: string // arXiv id, or a note when there is none
  why: string
  prepared: string[] // lesson ids
}

export const READING_PATH: NextPaper[] = [
  { short: 'GPT-2', title: 'Language Models are Unsupervised Multitask Learners', authors: 'Radford et al.', year: 2019, id: 'OpenAI report, not on arXiv', url: 'https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf',
    why: 'The decoder-only model you built and loaded. Read it for the claim that one next-token predictor can do many tasks without task-specific training.', prepared: ['build-gpt', 'pytorch-bridge'] },
  { short: 'GPT-3', title: 'Language Models are Few-Shot Learners', authors: 'Brown et al.', year: 2020, id: 'arXiv:2005.14165', url: 'https://arxiv.org/abs/2005.14165',
    why: 'Same architecture, 175B parameters, and the discovery that examples in the prompt can stand in for fine-tuning. Long: read sections 1 to 3 and skim the rest.', prepared: ['why-llms-know', 'training-pipeline'] },
  { short: 'Scaling laws', title: 'Scaling Laws for Neural Language Models', authors: 'Kaplan et al.', year: 2020, id: 'arXiv:2001.08361', url: 'https://arxiv.org/abs/2001.08361',
    why: 'Loss falls as a smooth power law in parameters, data and compute. Dense with plots: good practice for reading figures.', prepared: ['why-llms-know', 'training-gpt'] },
  { short: 'Chinchilla', title: 'Training Compute-Optimal Large Language Models', authors: 'Hoffmann et al.', year: 2022, id: 'arXiv:2203.15556', url: 'https://arxiv.org/abs/2203.15556',
    why: 'Revises Kaplan: for a fixed compute budget, earlier models were too big and trained on too little data. Read the two together to see a field revise itself.', prepared: ['why-llms-know'] },
  { short: 'InstructGPT', title: 'Training language models to follow instructions with human feedback', authors: 'Ouyang et al.', year: 2022, id: 'arXiv:2203.02155', url: 'https://arxiv.org/abs/2203.02155',
    why: 'The recipe that turns a base model into an assistant: supervised fine-tuning, a reward model, then reinforcement learning.', prepared: ['training-pipeline', 'fine-tuning'] },
  { short: 'DPO', title: 'Direct Preference Optimization: Your Language Model is Secretly a Reward Model', authors: 'Rafailov et al.', year: 2023, id: 'arXiv:2305.18290', url: 'https://arxiv.org/abs/2305.18290',
    why: 'Preference tuning without a separate reward model or an RL loop. A good first paper with a real derivation: read it straight after InstructGPT.', prepared: ['training-pipeline'] },
  { short: 'LoRA', title: 'LoRA: Low-Rank Adaptation of Large Language Models', authors: 'Hu et al.', year: 2021, id: 'arXiv:2106.09685', url: 'https://arxiv.org/abs/2106.09685',
    why: 'You have implemented it twice, so this is the ideal paper for a third pass: check every claim against your own code.', prepared: ['fine-tuning', 'pytorch-bridge'] },
  { short: 'RoPE', title: 'RoFormer: Enhanced Transformer with Rotary Position Embedding', authors: 'Su et al.', year: 2021, id: 'arXiv:2104.09864', url: 'https://arxiv.org/abs/2104.09864',
    why: 'The position method in almost every current open model. Heavier notation than the others: use the decoder in this lesson.', prepared: ['modern-architecture'] },
  { short: 'FlashAttention', title: 'FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness', authors: 'Dao et al.', year: 2022, id: 'arXiv:2205.14135', url: 'https://arxiv.org/abs/2205.14135',
    why: 'A systems paper: the same attention output, computed in tiles so the T × T table is never stored. Shows that memory traffic, not arithmetic, is often the limit.', prepared: ['attention', 'inference', 'inference-systems'] },
  { short: 'Llama 2 or Llama 3', title: 'Llama 2: Open Foundation and Fine-Tuned Chat Models; The Llama 3 Herd of Models', authors: 'Touvron et al.; Llama Team, AI @ Meta (on arXiv: Grattafiori et al.)', year: 2023, id: 'arXiv:2307.09288 and arXiv:2407.21783 (2024)', url: 'https://arxiv.org/abs/2407.21783',
    why: 'Technical reports for the models whose config you can now read. Long: treat them as reference works and read the architecture, data and post-training sections.', prepared: ['modern-architecture', 'training-pipeline', 'pytorch-bridge'] },
  { short: 'DeepSeek-V3', title: 'DeepSeek-V3 Technical Report', authors: 'DeepSeek-AI', year: 2024, id: 'arXiv:2412.19437', url: 'https://arxiv.org/abs/2412.19437',
    why: 'A frontier-scale open model described in detail: mixture-of-experts layers, multi-head latent attention to shrink the KV cache, FP8 training, and a multi-token-prediction head that later drafts for speculative decoding. Read the architecture and infrastructure sections.', prepared: ['modern-architecture', 'making-models-cheaper', 'inference-systems'] },
  { short: 'RAG', title: 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks', authors: 'Lewis et al.', year: 2020, id: 'arXiv:2005.11401', url: 'https://arxiv.org/abs/2005.11401',
    why: 'Where the name comes from. The paper fine-tunes the retriever’s query encoder and the generator together, which is unlike the retrieve-then-prompt pipeline you built: a good exercise in spotting the difference.', prepared: ['rag'] },
  { short: 'ReAct', title: 'ReAct: Synergizing Reasoning and Acting in Language Models', authors: 'Yao et al.', year: 2022, id: 'arXiv:2210.03629', url: 'https://arxiv.org/abs/2210.03629',
    why: 'The thought, action, observation loop behind your mini agent. Short and readable.', prepared: ['agents'] },
  { short: 'Scaling Monosemanticity', title: 'Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet', authors: 'Templeton et al. (Anthropic)', year: 2024, id: 'Transformer Circuits Thread, not on arXiv', url: 'https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html',
    why: 'A sparse autoencoder pulls millions of readable features out of one layer of a production model, and turning a feature up changes behaviour. A web article with interactive figures: a good test of reading claims and their limits.', prepared: ['interpretability', 'embeddings', 'transformer-block'] },
  { short: 'DeepSeek-R1', title: 'DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning', authors: 'DeepSeek-AI', year: 2025, id: 'arXiv:2501.12948', url: 'https://arxiv.org/abs/2501.12948',
    why: 'An open account of training a reasoning model with reinforcement learning on checkable answers. Read it last: it builds on almost everything above.', prepared: ['reasoning-models', 'training-pipeline'] },
]
