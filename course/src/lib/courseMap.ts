// Lesson 0.3: the big architecture of the course. Each node answers one question,
// produces one thing you build, and links to real lessons in curriculum.ts.
export interface MapNode {
  id: string
  label: string
  question: string
  build: string
  lessons: string[] // lesson ids from curriculum.ts
}

export const COURSE_MAP: MapNode[] = [
  { id: 'text', label: 'Text', question: 'What actually goes into an LLM, and what comes out?', build: 'Nothing yet. You explore the whole pipeline from the outside, with toy numbers.', lessons: ['prompt-to-answer', 'surprising-idea'] },
  { id: 'tokenization', label: 'Tokenization', question: 'You cannot multiply the word “cat”. How does text become whole numbers?', build: 'A complete BPE tokenizer: train, encode, decode (bpe_tokenizer.py).', lessons: ['tokenization'] },
  { id: 'representations', label: 'Representations', question: 'How can a list of numbers mean something, and how do you compare two of them?', build: 'Vectors and dot products by hand, then word vectors trained from scratch (math_primer.py, tiny_word2vec.py).', lessons: ['vectors', 'matrices', 'embeddings'] },
  { id: 'network', label: 'Neural network', question: 'How can a program find good numbers by itself, from examples?', build: 'Gradient descent, then a small neural network with backpropagation in plain NumPy, then your first next-token predictor (gradient_descent.py, mlp_numpy.py, bigram_lm.py).', lessons: ['softmax', 'derivatives', 'gradient-descent', 'neurons', 'backprop', 'next-token'] },
  { id: 'attention', label: 'Attention', question: 'How can one token use information from the other tokens, however far away?', build: 'Single-head and multi-head attention with a causal mask, in NumPy (attention_numpy.py).', lessons: ['context-wall', 'attention', 'masks-and-heads'] },
  { id: 'transformer', label: 'Transformer', question: 'What else does a block need besides attention, and how do blocks stack into GPT?', build: 'A complete GPT in about 250 lines of PyTorch (tiny_gpt.py).', lessons: ['transformer-block', 'build-gpt'] },
  { id: 'training', label: 'Training', question: 'What actually happens during “training”, and how does a text predictor become an assistant?', build: 'You train your GPT on Shakespeare and watch loss fall and samples improve.', lessons: ['training-gpt', 'training-pipeline'] },
  { id: 'inference', label: 'Inference', question: 'What do temperature and top-p really do, and why is generation slow?', build: 'Sampling policies and a KV cache, proven identical to the slow way (kv_cache_demo.py).', lessons: ['inference', 'reasoning-models'] },
  { id: 'llm', label: 'LLM', question: 'What separates your tiny GPT from a modern LLM, and where does its knowledge live?', build: 'No new code: you read a modern architecture with the eyes of someone who built the small one.', lessons: ['why-llms-know', 'modern-architecture'] },
  { id: 'rag', label: 'RAG', question: 'What if the information is not inside the model?', build: 'A vector database and a full retrieve-then-answer pipeline (vector_db.py, mini_rag.py).', lessons: ['rag'] },
  { id: 'finetune', label: 'Fine-tuning', question: 'What does changing the weights achieve that a better prompt cannot?', build: 'Full fine-tuning versus LoRA on your own GPT, with forgetting measured (finetune_tiny_gpt.py).', lessons: ['fine-tuning'] },
  { id: 'agents', label: 'Agents', question: 'How can a text generator use a tool?', build: 'A tool-calling agent loop with memory, and a prompt-injection demo (mini_agent.py).', lessons: ['agents'] },
  { id: 'engineering', label: 'Engineering', question: 'What does it take to run all of this for real users, and to keep up with the research?', build: 'An eval harness, a batching simulator, weight quantization, a real LoRA fine-tune with Hugging Face, and a budgeted agent (phase6-engineering/).', lessons: ['evals', 'inference-systems', 'pytorch-bridge', 'reading-papers', 'production-agents'] },
]

/** The lessons that are about the course itself or pull everything together; they sit outside the map. */
export const OFF_MAP = ['course-map', 'capstone', 'from-memory']

/** The prompt -> answer pipeline from lesson 0.1, reused for "rebuild it from memory". */
export const PIPELINE_ORDER = ['Text', 'Tokenizer', 'Tokens', 'Token IDs', 'Embeddings', 'Transformer', 'Next-token probabilities', 'Sampling', 'Next token', 'Repeat']
