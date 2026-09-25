import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { byteOrder, createGpt2Tokenizer, EOT_ID, showToken } from './gpt2Tokenizer'
import fixture from './__fixtures__/gpt2Tokens.fixture.json'

// merges.bin and the fixture are both written by phase6-engineering/export_gpt2.py --fixtures;
// the fixture's ids come from tiktoken's "gpt2" encoding (and were checked equal to transformers').
const buf = readFileSync(resolve(__dirname, '../../public/models/gpt2/merges.bin'))
const merges = new Uint16Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength))
const tok = createGpt2Tokenizer(merges)

describe('GPT-2 tokenizer', () => {
  it('has the GPT-2 vocabulary size', () => {
    expect(tok.vocabSize).toBe(50257)
    expect(tok.decode([EOT_ID])).toBe('<|endoftext|>')
    expect(byteOrder().length).toBe(256)
    expect(new Set(byteOrder()).size).toBe(256)
  })

  for (const c of fixture.cases) {
    it(`matches tiktoken on ${JSON.stringify(c.text).slice(0, 40)}`, () => {
      expect(tok.encode(c.text)).toEqual(c.ids)
      expect(tok.decode(c.ids)).toBe(c.text)
    })
  }

  it('labels whole tokens as text and partial UTF-8 as bytes', () => {
    const [id] = tok.encode(' cat')
    expect(tok.label(id)).toBe(' cat')
    const hindi = tok.encode('न')
    expect(hindi.length).toBeGreaterThan(1) // one Devanagari letter = 3 bytes, not one GPT-2 token
    expect(hindi.map(tok.label).join('')).toMatch(/^<[0-9A-F]{2}>/)
    expect(showToken(' a\nb')).toBe('␣a↵b')
  })
})
