// A chat template turns a list of {role, content} messages into ONE flat token sequence.
// This file implements a generic ChatML-style template for teaching:
//
//   <|im_start|>system\n{content}<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n{content}<|im_end|>\n
//
// Every model family has its own markers; the idea (role markers as special tokens, one flat
// sequence, loss only on the assistant's tokens during SFT) is the same everywhere.
// Pure functions, no DOM.

export type Role = 'system' | 'user' | 'assistant'
export interface Message { role: Role; content: string }

export const IM_START = '<|im_start|>'
export const IM_END = '<|im_end|>'

export type PieceKind = 'special' | 'role' | 'newline' | 'text'

export interface ChatToken {
  text: string
  kind: PieceKind
  /** which message this token belongs to */
  role: Role
  messageIndex: number
  /** true if this token is a training TARGET during SFT with assistant-only loss */
  loss: boolean
}

/**
 * A deliberately simple stand-in for a real tokenizer: each word keeps its leading space
 * (like GPT-style BPE does), punctuation is split off, newlines are their own token.
 * Real tokenizers split differently; the masking logic does not depend on how.
 */
export const splitContent = (content: string): string[] => content.match(/\n| ?[A-Za-z0-9']+| ?[^\sA-Za-z0-9']| +|\s/g) ?? []

/**
 * Render a conversation into tokens, marking which tokens receive loss during SFT.
 * Loss goes on: the assistant's content tokens, and the <|im_end|> that closes an assistant turn
 * (the model has to learn to stop). Everything else, including the "<|im_start|>assistant\n"
 * header, is context only: it is given to the model, never predicted by it.
 */
export const renderChat = (messages: Message[]): ChatToken[] => {
  const out: ChatToken[] = []
  messages.forEach((m, messageIndex) => {
    const base = { role: m.role, messageIndex }
    const isA = m.role === 'assistant'
    out.push({ ...base, text: IM_START, kind: 'special', loss: false })
    out.push({ ...base, text: m.role, kind: 'role', loss: false })
    out.push({ ...base, text: '\n', kind: 'newline', loss: false })
    for (const piece of splitContent(m.content)) out.push({ ...base, text: piece, kind: piece === '\n' ? 'newline' : 'text', loss: isA })
    out.push({ ...base, text: IM_END, kind: 'special', loss: isA })
    out.push({ ...base, text: '\n', kind: 'newline', loss: false })
  })
  return out
}

/** The exact flat string the model sees. */
export const flatString = (tokens: ChatToken[]): string => tokens.map((t) => t.text).join('')

/** What an inference server sends: the conversation so far plus an OPEN assistant turn for the model to fill in. */
export const generationPrompt = (messages: Message[]): string => `${flatString(renderChat(messages))}${IM_START}assistant\n`

/** 1 where the token is a training target, 0 where it is context only. */
export const lossMask = (tokens: ChatToken[], assistantOnly = true): number[] => tokens.map((t) => (assistantOnly ? (t.loss ? 1 : 0) : 1))

export const countLoss = (tokens: ChatToken[], assistantOnly = true): { total: number; withLoss: number; fraction: number } => {
  const total = tokens.length
  const withLoss = lossMask(tokens, assistantOnly).reduce((s, v) => s + v, 0)
  return { total, withLoss, fraction: total === 0 ? 0 : withLoss / total }
}

/**
 * Masked average loss: the mean of per-token losses over the tokens where mask = 1.
 * Same effect as PyTorch's F.cross_entropy(..., ignore_index=-100) with masked targets set to -100.
 */
export const maskedMeanLoss = (perTokenLoss: number[], mask: number[]): number => {
  if (perTokenLoss.length !== mask.length) throw new Error('maskedMeanLoss: length mismatch')
  let s = 0
  let n = 0
  for (let i = 0; i < mask.length; i++) if (mask[i]) { s += perTokenLoss[i]; n++ }
  return n === 0 ? 0 : s / n
}
