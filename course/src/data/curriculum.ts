// The curriculum map: single source of truth for navigation, progress,
// search, "Where are we?" and the concept map. Lesson bodies live in
// src/lessons/<id>.tsx and are discovered by id.

export const REPO_URL = 'https://github.com/fork-demon/llm-from-scratch'

export interface SourceFile {
  path: string // repo-relative path
  note: string // what the learner will find there
}

export interface LessonMeta {
  id: string // also the route and the lesson file name
  title: string
  question: string // the question this lesson answers (shown under the title)
  minutes: number
  /** Which node of the "Where are we?" tree this lesson lights up. */
  here: string
  sources?: SourceFile[]
  /** Long-form companion notes from the original repo. */
  notes?: string
}

export interface PartMeta {
  id: string
  number: number
  title: string
  blurb: string
  lessons: LessonMeta[]
}

export const PARTS: PartMeta[] = [
  {
    id: 'big-picture',
    number: 0,
    title: 'What is an LLM?',
    blurb: 'See the whole machine before we open it up.',
    lessons: [
      { id: 'prompt-to-answer', title: 'What happens when I type a prompt?', question: 'What happens between typing "What is a cat?" and seeing an answer?', minutes: 12, here: 'llm' },
      { id: 'surprising-idea', title: 'The surprising idea', question: 'If it is not looking answers up, what is it doing?', minutes: 10, here: 'llm' },
      { id: 'course-map', title: 'Your map of the course', question: 'Where is all of this going?', minutes: 8, here: 'llm' },
    ],
  },
  {
    id: 'math',
    number: 1,
    title: 'The Tiny Bit of Math an LLM Needs',
    blurb: 'Four ideas, each one a picture before it is a formula.',
    lessons: [
      { id: 'vectors', title: 'Vectors and the dot product', question: 'How do you ask two lists of numbers "do you agree?"', minutes: 20, here: 'math', sources: [{ path: 'phase1-foundations/math_primer.py', note: 'every calculation in this part, verified in NumPy' }], notes: 'phase1-foundations/00-math-primer.md' },
      { id: 'matrices', title: 'Matrices: many dot products at once', question: 'Why is everything in an LLM a matrix multiply?', minutes: 20, here: 'math', sources: [{ path: 'phase1-foundations/math_primer.py', note: 'matrix shapes and the transpose, worked by hand' }], notes: 'phase1-foundations/00-math-primer.md' },
      { id: 'softmax', title: 'Scores into probabilities: softmax', question: 'How does a model turn raw scores into "70% sure"?', minutes: 18, here: 'math', sources: [{ path: 'phase1-foundations/mlp_numpy.py', note: 'the numerically stable softmax and cross-entropy used by every later file' }] },
      { id: 'derivatives', title: 'Nudges: derivatives and the chain rule', question: 'If I nudge this number, how much does the result move?', minutes: 20, here: 'math', sources: [{ path: 'phase1-foundations/math_primer.py', note: 'the nudge experiment and the chain rule pipeline' }], notes: 'phase1-foundations/00-math-primer.md' },
    ],
  },
  {
    id: 'learning',
    number: 2,
    title: 'How Machines Learn',
    blurb: 'Prediction, error, and the one loop that trains everything.',
    lessons: [
      { id: 'gradient-descent', title: 'Gradient descent', question: 'How can a program find good numbers on its own?', minutes: 25, here: 'training', sources: [{ path: 'phase1-foundations/gradient_descent.py', note: 'three stages: by-hand gradient, numerical gradient, mini-batch SGD' }], notes: 'phase1-foundations/02-gradient-descent.md' },
    ],
  },
  {
    id: 'neural-nets',
    number: 3,
    title: 'Neural Networks',
    blurb: 'Adjustable functions, stacked, and how blame flows backward.',
    lessons: [
      { id: 'neurons', title: 'Neurons and layers', question: 'What is a neural network actually made of?', minutes: 22, here: 'mlp', sources: [{ path: 'phase1-foundations/mlp_numpy.py', note: 'a linear model failing on a spiral, then an MLP solving it' }], notes: 'phase1-foundations/03-neural-net-backprop.md' },
      { id: 'backprop', title: 'Backpropagation', question: 'With millions of weights, how do we know which ones to blame?', minutes: 28, here: 'training', sources: [{ path: 'phase1-foundations/mlp_numpy.py', note: 'forward, backward and a gradient check in ~190 lines of NumPy' }], notes: 'phase1-foundations/03-neural-net-backprop.md' },
    ],
  },
  {
    id: 'text-to-numbers',
    number: 4,
    title: 'How Text Becomes Numbers',
    blurb: 'You cannot multiply the word "cat". So what do we do?',
    lessons: [
      { id: 'tokenization', title: 'Tokenization', question: 'Why does GPT not simply use words?', minutes: 22, here: 'tokenizer', sources: [{ path: 'phase2-language/bpe_tokenizer.py', note: 'a complete BPE tokenizer: train, encode, decode' }], notes: 'phase2-language/04-tokenization.md' },
      { id: 'embeddings', title: 'Embeddings', question: 'How can a list of numbers mean something?', minutes: 25, here: 'embeddings', sources: [{ path: 'phase2-language/tiny_word2vec.py', note: 'train word vectors from scratch and inspect the geometry' }], notes: 'phase2-language/05-embeddings.md' },
    ],
  },
  {
    id: 'first-lm',
    number: 5,
    title: 'Your First Language Model',
    blurb: 'Next-token prediction, and why the simple version hits a wall.',
    lessons: [
      { id: 'next-token', title: 'Predicting the next token', question: 'How does "guess the next word" become writing?', minutes: 22, here: 'output', sources: [{ path: 'phase2-language/bigram_lm.py', note: 'a count-table model, a neural bigram, and a 3-character context model' }], notes: 'phase2-language/06-first-language-model.md' },
      { id: 'context-wall', title: 'Why simple language models break', question: 'Why not just remember longer phrases?', minutes: 15, here: 'attention', sources: [{ path: 'phase2-language/bigram_lm.py', note: 'Model C: more context beats a smarter model with less' }] },
    ],
  },
  {
    id: 'attention',
    number: 6,
    title: 'Attention and the Transformer',
    blurb: 'The heart of the course. Take your time here.',
    lessons: [
      { id: 'attention', title: 'Attention', question: 'How can one token look at the other tokens?', minutes: 40, here: 'attention', sources: [{ path: 'phase3-transformers/attention_numpy.py', note: 'single-head attention, the "river bank" demo, and a gradient check' }], notes: 'phase3-transformers/07-attention.md' },
      { id: 'masks-and-heads', title: 'Causal masks and multiple heads', question: 'How do we stop tokens peeking at the future, and why run attention several times?', minutes: 25, here: 'attention', sources: [{ path: 'phase3-transformers/attention_numpy.py', note: 'demo_causal and multi_head_attention with shape comments' }], notes: 'phase3-transformers/07-attention.md' },
      { id: 'transformer-block', title: 'The Transformer block', question: 'What else does a block need besides attention?', minutes: 28, here: 'block', sources: [{ path: 'phase3-transformers/tiny_gpt.py', note: 'class Block: two lines that define the whole architecture' }], notes: 'phase3-transformers/08-tiny-gpt.md' },
    ],
  },
  {
    id: 'gpt',
    number: 7,
    title: 'Build, Train and Run a GPT',
    blurb: 'Assemble the parts, watch it learn, make it talk.',
    lessons: [
      { id: 'build-gpt', title: 'Build GPT', question: 'Can you trace one token from text to prediction?', minutes: 30, here: 'transformer', sources: [{ path: 'phase3-transformers/tiny_gpt.py', note: 'a complete GPT in ~250 lines of PyTorch' }], notes: 'phase3-transformers/08-tiny-gpt.md' },
      { id: 'training-gpt', title: 'Training GPT', question: 'What actually happens during "training"?', minutes: 28, here: 'training', sources: [{ path: 'phase3-transformers/tiny_gpt.py', note: 'main(): batches, AdamW, train vs validation loss, samples during training' }], notes: 'phase3-transformers/09-training-and-inference.md' },
      { id: 'inference', title: 'Inference: sampling and the KV cache', question: 'What do temperature and top-p really do, and why is generation slow?', minutes: 28, here: 'sampling', sources: [{ path: 'phase3-transformers/kv_cache_demo.py', note: 'naive vs cached generation, proven identical, plus sampling policies' }], notes: 'phase3-transformers/09-training-and-inference.md' },
    ],
  },
  {
    id: 'modern',
    number: 8,
    title: 'From GPT to Modern LLMs',
    blurb: 'What changed since 2019, and what we honestly do not know.',
    lessons: [
      { id: 'why-llms-know', title: 'Why LLMs know things', question: 'Where is the knowledge, and why do models make things up?', minutes: 22, here: 'llm', notes: 'phase4-modern-llms/10-knowledge-scaling-hallucination.md' },
      { id: 'modern-architecture', title: 'Modern LLM architecture', question: 'What is different inside a 2020s model compared with our tiny GPT?', minutes: 40, here: 'block', notes: 'phase4-modern-llms/10-knowledge-scaling-hallucination.md' },
      { id: 'training-pipeline', title: 'From raw text to assistant', question: 'How does a next-token predictor become a helpful assistant?', minutes: 22, here: 'training', notes: 'phase3-transformers/09-training-and-inference.md' },
      { id: 'reasoning-models', title: 'Reasoning models', question: 'What changes when a model is allowed to think before answering?', minutes: 18, here: 'sampling', notes: 'appendix-frontier-topics.md' },
    ],
  },
  {
    id: 'systems',
    number: 9,
    title: 'Building With LLMs',
    blurb: 'RAG, fine-tuning and agents: what each one really changes.',
    lessons: [
      { id: 'rag', title: 'Retrieval-augmented generation', question: 'What if the information is not inside the model?', minutes: 30, here: 'rag', sources: [{ path: 'phase4-modern-llms/vector_db.py', note: 'a vector database: exact search, IVF index, quantization' }, { path: 'phase4-modern-llms/mini_rag.py', note: 'the full pipeline: chunk, embed, retrieve, prompt, answer' }], notes: 'phase4-modern-llms/12-rag.md' },
      { id: 'fine-tuning', title: 'Fine-tuning', question: 'What does fine-tuning change that a prompt cannot?', minutes: 25, here: 'finetune', sources: [{ path: 'phase4-modern-llms/finetune_tiny_gpt.py', note: 'full fine-tune vs LoRA, with catastrophic forgetting measured' }], notes: 'phase4-modern-llms/13-fine-tuning.md' },
      { id: 'agents', title: 'Agents', question: 'How can a text generator use a tool?', minutes: 28, here: 'agents', sources: [{ path: 'phase5-agents/mini_agent.py', note: 'tool registry, the ReAct loop, memory, and a prompt-injection demo' }], notes: 'phase5-agents/14-agents.md' },
    ],
  },
  {
    id: 'engineering',
    number: 10,
    title: 'From Understanding to Engineering',
    blurb: 'What it takes to run this in production, and to keep learning from papers.',
    lessons: [
      { id: 'evals', title: 'Evals: how you know it works', question: 'The demo looked great. How do you know the system actually works?', minutes: 35, here: 'around', sources: [{ path: 'phase6-engineering/eval_harness.py', note: 'a golden set, three scoring methods and bootstrap confidence intervals, run against mini_rag.py' }] },
      { id: 'inference-systems', title: 'Serving many users at once', question: 'One user is easy. What changes when a thousand arrive at once?', minutes: 30, here: 'sampling', sources: [{ path: 'phase6-engineering/batching_sim.py', note: 'static versus continuous batching, simulated request by request' }] },
      { id: 'making-models-cheaper', title: 'Making the model itself cheaper', question: 'The batch is as full as it can get. What is left to change?', minutes: 25, here: 'sampling', sources: [{ path: 'phase6-engineering/quantize_demo.py', note: 'int8 and int4 weight quantization in NumPy, with the error measured' }, { path: 'phase6-engineering/speculative_demo.py', note: 'speculative decoding on toy distributions, with the output proven equal to the target model’s' }] },
      { id: 'pytorch-bridge', title: 'The PyTorch and Hugging Face bridge', question: 'Can you open a real open model and recognise everything inside it?', minutes: 40, here: 'transformer', sources: [{ path: 'phase6-engineering/inspect_hf_model.py', note: 'load GPT-2, walk its modules and match every tensor to a lesson' }, { path: 'phase6-engineering/lora_finetune_hf.py', note: 'a real LoRA fine-tune with the peft library, sized for one small GPU or a patient CPU' }] },
      { id: 'reading-papers', title: 'How to read an LLM paper', question: 'Can you read “Attention Is All You Need” and recognise every section?', minutes: 35, here: 'llm' },
      { id: 'production-agents', title: 'Context engineering and production agents', question: 'Your agent works on your laptop. What breaks when real users and real money are involved?', minutes: 40, here: 'agents', sources: [{ path: 'phase6-engineering/agent_budget.py', note: 'token, cost and step accounting wrapped around mini_agent.py, with a trace you can inspect' }] },
    ],
  },
  {
    id: 'capstone',
    number: 11,
    title: 'Capstone',
    blurb: 'Build the whole system, then explain it from memory.',
    lessons: [
      { id: 'capstone', title: 'Build your own mini LLM system', question: 'Can you wire every piece together yourself?', minutes: 240, here: 'llm' },
      { id: 'from-memory', title: 'Explain it from memory', question: 'Can you rebuild the whole picture without looking?', minutes: 20, here: 'llm' },
    ],
  },
]

export interface FlatLesson extends LessonMeta {
  part: PartMeta
  index: number // global order
  code: string // e.g. "6.1"
}

export const LESSONS: FlatLesson[] = PARTS.flatMap((part) =>
  part.lessons.map((l, i) => ({ ...l, part, index: 0, code: `${part.number}.${i + 1}` })),
).map((l, index) => ({ ...l, index }))

export const lessonById = (id: string): FlatLesson | undefined => LESSONS.find((l) => l.id === id)

export const sourceUrl = (path: string) => `${REPO_URL}/blob/main/${path}`

/** The "Where are we?" tree. Lessons point at a node id via `here`. */
export interface TreeNode {
  id: string
  label: string
  children?: TreeNode[]
}

export const LLM_TREE: TreeNode = {
  id: 'llm',
  label: 'LLM',
  children: [
    { id: 'math', label: 'Math toolkit' },
    { id: 'tokenizer', label: 'Tokenizer' },
    { id: 'embeddings', label: 'Embeddings' },
    {
      id: 'transformer',
      label: 'Transformer',
      children: [
        {
          id: 'block',
          label: 'Block × N',
          children: [
            { id: 'attention', label: 'Attention' },
            { id: 'mlp', label: 'MLP (feed-forward)' },
          ],
        },
      ],
    },
    { id: 'output', label: 'Next-token probabilities' },
    { id: 'sampling', label: 'Sampling & inference' },
    { id: 'training', label: 'Training' },
    {
      id: 'around',
      label: 'Around the model',
      children: [
        { id: 'rag', label: 'RAG' },
        { id: 'finetune', label: 'Fine-tuning' },
        { id: 'agents', label: 'Agents' },
      ],
    },
  ],
}
