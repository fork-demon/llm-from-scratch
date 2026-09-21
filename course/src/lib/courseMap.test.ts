import { describe, expect, it } from 'vitest'
import { COURSE_MAP, OFF_MAP } from './courseMap'
import { LESSONS } from '../data/curriculum'

describe('course map', () => {
  it('links only to lessons that exist', () => {
    const ids = new Set(LESSONS.map((l) => l.id))
    for (const n of COURSE_MAP) for (const l of n.lessons) expect(ids.has(l), `${n.id} -> ${l}`).toBe(true)
  })
  it('places every lesson exactly once', () => {
    const placed = [...COURSE_MAP.flatMap((n) => n.lessons), ...OFF_MAP].sort()
    expect(placed).toEqual(LESSONS.map((l) => l.id).sort())
  })
  it('has the architecture stages plus the engineering stage, each with a question and something to build', () => {
    expect(COURSE_MAP.map((n) => n.label)).toEqual(['Text', 'Tokenization', 'Representations', 'Neural network', 'Attention', 'Transformer', 'Training', 'Inference', 'LLM', 'RAG', 'Fine-tuning', 'Agents', 'Engineering'])
    expect(COURSE_MAP.every((n) => n.question.length > 20 && n.build.length > 20)).toBe(true)
  })
})
