import { describe, expect, it } from 'vitest'
import { IM_END, IM_START, countLoss, flatString, generationPrompt, lossMask, maskedMeanLoss, renderChat, splitContent, type Message } from './chatTemplate'

const convo: Message[] = [
  { role: 'system', content: 'You are a helpful assistant.' },
  { role: 'user', content: 'What is a cat?' },
  { role: 'assistant', content: 'A cat is a small furry animal.' },
]

describe('chat template', () => {
  it('renders one flat string with role markers', () => {
    expect(flatString(renderChat(convo))).toBe(
      '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' +
      '<|im_start|>user\nWhat is a cat?<|im_end|>\n' +
      '<|im_start|>assistant\nA cat is a small furry animal.<|im_end|>\n',
    )
  })

  it('splitContent loses no characters', () => {
    for (const s of ['What is a cat?', '  two  spaces\nand a\ttab ', 'café: 3.5€!', '']) expect(splitContent(s).join('')).toBe(s)
  })

  it('puts loss only on assistant content and the closing end marker', () => {
    const toks = renderChat(convo)
    const targets = toks.filter((t) => t.loss)
    expect(targets.every((t) => t.role === 'assistant')).toBe(true)
    expect(targets.map((t) => t.text).join('')).toBe(`A cat is a small furry animal.${IM_END}`)
    // the assistant header is context, not a target
    const header = toks.filter((t) => t.role === 'assistant' && (t.text === IM_START || t.kind === 'role'))
    expect(header.length).toBe(2)
    expect(header.some((t) => t.loss)).toBe(false)
  })

  it('counts match the worked example in the lesson (34 tokens, 9 with loss)', () => {
    const c = countLoss(renderChat(convo))
    expect(c.total).toBe(34)
    expect(c.withLoss).toBe(9)
    expect(countLoss(renderChat(convo), false).withLoss).toBe(34)
  })

  it('system and user text never changes the number of loss tokens', () => {
    const longer: Message[] = [{ role: 'system', content: 'Be brief. Be kind. Be correct.' }, { role: 'user', content: 'Please tell me, in detail, what a cat is?' }, convo[2]]
    expect(countLoss(renderChat(longer)).withLoss).toBe(countLoss(renderChat(convo)).withLoss)
  })

  it('multi-turn: every assistant turn is a target', () => {
    const multi: Message[] = [...convo, { role: 'user', content: 'And a dog?' }, { role: 'assistant', content: 'A loyal one.' }]
    const toks = renderChat(multi)
    expect(new Set(toks.filter((t) => t.loss).map((t) => t.messageIndex))).toEqual(new Set([2, 4]))
  })

  it('generation prompt ends with an open assistant turn', () => {
    const p = generationPrompt(convo.slice(0, 2))
    expect(p.endsWith('<|im_start|>assistant\n')).toBe(true)
    expect(p).not.toContain('furry')
  })

  it('masked mean ignores masked positions', () => {
    expect(maskedMeanLoss([5, 5, 1, 3], [0, 0, 1, 1])).toBeCloseTo(2)
    expect(maskedMeanLoss([5, 5, 1, 3], [1, 1, 1, 1])).toBeCloseTo(3.5)
    expect(maskedMeanLoss([1, 2], [0, 0])).toBe(0)
    expect(lossMask(renderChat(convo)).length).toBe(34)
  })
})
