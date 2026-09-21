// Capstone milestone checklists and their progress maths. Pure: storage lives in the component.

export interface Milestone {
  id: string
  title: string
  files: string[]
  checklist: string[]
}

export const MILESTONES: Milestone[] = [
  {
    id: 'tokenizer',
    title: 'Tokenizer',
    files: ['phase2-language/bpe_tokenizer.py'],
    checklist: [
      'BPE tokenizer trained on my own text file',
      'decode(encode(s)) == s for text from the corpus and for a sentence I made up',
      'Unseen characters no longer crash encode()',
      'I measured characters per token (it is above 1)',
      'My round-trip test passes under pytest',
    ],
  },
  {
    id: 'transformer',
    title: 'Embedding + Transformer',
    files: ['phase3-transformers/attention_numpy.py', 'phase3-transformers/tiny_gpt.py'],
    checklist: [
      'tiny_gpt reads my BPE token ids instead of characters',
      'vocab_size comes from my tokenizer, and logits have shape (B, T, vocab_size)',
      'Attention rows sum to 1 (test passes)',
      'Changing a later token does not change earlier logits (causality test passes)',
      'I can say what each of the 6 Config numbers controls',
    ],
  },
  {
    id: 'training',
    title: 'Training',
    files: ['phase3-transformers/tiny_gpt.py'],
    checklist: [
      'The model can overfit one small batch (loss falls close to 0)',
      'Full run: train and validation loss both fall',
      'I compared train and validation loss and can say whether it overfits',
      'Samples visibly improve between step 0 and the end',
      'Trained weights are saved with torch.save and load again',
    ],
  },
  {
    id: 'inference',
    title: 'Inference',
    files: ['phase3-transformers/tiny_gpt.py', 'phase3-transformers/kv_cache_demo.py'],
    checklist: [
      'GPT.generate accepts top_k and top_p',
      'top_k=1 gives the same output on every run',
      'I generated at temperature 0.3, 0.8 and 1.5 and can describe the difference',
      'Cached and naive generation give identical tokens (test passes)',
      'I can explain what the KV cache stores and why that is safe',
    ],
  },
  {
    id: 'rag',
    title: 'RAG',
    files: ['phase4-modern-llms/vector_db.py', 'phase4-modern-llms/mini_rag.py'],
    checklist: [
      'mini_rag indexes documents I wrote myself',
      'The right chunk is retrieved for three of my own questions',
      'The assembled prompt (with retrieved context) is fed to my trained GPT',
      'A question with no answer in my documents is refused by the extractive baseline',
      'I checked that no model weight changed: only the prompt did',
    ],
  },
  {
    id: 'tools',
    title: 'Tool calling',
    files: ['phase5-agents/mini_agent.py'],
    checklist: [
      'My GPT generation is registered as a tool in TOOLS',
      'The agent loop calls it and appends the RESULT to the scratchpad',
      'An unknown tool name returns an error string instead of crashing',
      'max_steps stops a loop that never answers',
      'I can point to the exact line where my code, not the model, acts',
    ],
  },
]

/** Checked items, keyed "<milestoneId>:<itemIndex>". */
export type Checked = Record<string, boolean>
export const itemKey = (milestoneId: string, index: number) => `${milestoneId}:${index}`

export const toggle = (checked: Checked, key: string): Checked => {
  const next = { ...checked }
  if (next[key]) delete next[key]
  else next[key] = true
  return next
}

/** Tolerant parser for whatever is in localStorage: unknown keys and junk are dropped. */
export const parseChecked = (raw: string | null | undefined, milestones: Milestone[] = MILESTONES): Checked => {
  if (!raw) return {}
  try {
    const data: unknown = JSON.parse(raw)
    if (!data || typeof data !== 'object' || Array.isArray(data)) return {}
    const valid = new Set(milestones.flatMap((m) => m.checklist.map((_, i) => itemKey(m.id, i))))
    const out: Checked = {}
    for (const [k, v] of Object.entries(data as Record<string, unknown>)) if (v === true && valid.has(k)) out[k] = true
    return out
  } catch {
    return {}
  }
}

export interface MilestoneProgress { id: string; done: number; total: number; complete: boolean }
export interface CapstoneProgress {
  perMilestone: MilestoneProgress[]
  done: number
  total: number
  percent: number // 0..100, whole number
  milestonesComplete: number
  /** First milestone that still has unchecked items, or null when everything is done. */
  current: string | null
}

export const capstoneProgress = (checked: Checked, milestones: Milestone[] = MILESTONES): CapstoneProgress => {
  const perMilestone = milestones.map((m) => {
    const done = m.checklist.filter((_, i) => checked[itemKey(m.id, i)]).length
    return { id: m.id, done, total: m.checklist.length, complete: done === m.checklist.length }
  })
  const done = perMilestone.reduce((s, m) => s + m.done, 0)
  const total = perMilestone.reduce((s, m) => s + m.total, 0)
  return {
    perMilestone,
    done,
    total,
    percent: total === 0 ? 0 : Math.round((done / total) * 100),
    milestonesComplete: perMilestone.filter((m) => m.complete).length,
    current: perMilestone.find((m) => !m.complete)?.id ?? null,
  }
}
