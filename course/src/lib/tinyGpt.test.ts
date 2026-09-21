import { describe, expect, it } from 'vitest'
import { CHARS, countModelNumbers, cropToContext, decode, encode, generate, gptForward, makeTinyGpt, nextToken, TRACER_CONFIG } from './tinyGpt'
import { countParams } from './params'

const model = makeTinyGpt()

describe('tiny GPT forward pass', () => {
  it('tokenizer round-trips and drops unknown characters', () => {
    expect(decode(encode('the cat'))).toBe('the cat')
    expect(decode(encode('The Cat!'))).toBe('the cat')
    expect(CHARS).toHaveLength(31)
    expect(CHARS[0]).toBe(' ')
  })

  it('logits have shape (T, V) and every stage keeps (T, D)', () => {
    const ids = encode('the c')
    const t = gptForward(model, ids)
    expect([t.logits.length, t.logits[0].length]).toEqual([5, CHARS.length])
    for (const m of [t.tok, t.pos, t.x0, ...t.blockOuts, t.final]) expect([m.length, m[0].length]).toEqual([5, TRACER_CONFIG.nEmbd])
    expect(t.blockOuts).toHaveLength(TRACER_CONFIG.nLayer)
  })

  it('is causal: changing a later token does not change the logits of earlier positions', () => {
    const a = gptForward(model, encode('the cat')).logits
    const b = gptForward(model, encode('the cow')).logits // differs from position 5 on
    for (let t = 0; t < 5; t++) a[t].forEach((v, j) => expect(v).toBeCloseTo(b[t][j], 10))
    expect(Math.max(...a[5].map((v, j) => Math.abs(v - b[5][j])))).toBeGreaterThan(1e-3)
  })

  it('probabilities sum to 1 and sampling is reproducible', () => {
    const { logits } = gptForward(model, encode('the c'))
    const n = nextToken(logits, 1, 0.5)
    expect(n.probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
    expect(n.probs).toHaveLength(CHARS.length)
    expect(nextToken(logits, 1, 0.5).id).toBe(n.id)
    // temperature -> 0 is argmax
    expect(nextToken(logits, 0, 0.99).id).toBe(n.probs.indexOf(Math.max(...n.probs)))
  })

  it('holds exactly as many numbers as params.ts predicts (tied head)', () => {
    expect(countModelNumbers(model)).toBe(countParams(TRACER_CONFIG).total)
  })

  it('context window: too long throws, generate crops like idx[:, -context_len:]', () => {
    const long = encode('the cat sat on the mat today')
    expect(() => gptForward(model, long)).toThrow()
    expect(cropToContext(long, 16)).toEqual(long.slice(-16))
    const out = generate(model, long, 3)
    expect(out).toHaveLength(long.length + 3)
    expect(generate(model, long, 3)).toEqual(out)
  })
})
