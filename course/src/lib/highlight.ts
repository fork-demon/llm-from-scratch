// A deliberately tiny Python/pseudocode highlighter: comments, strings,
// keywords, numbers, function calls. Returns safe tokens (no HTML strings).
export type TokKind = 'c' | 's' | 'k' | 'n' | 'f' | ''
export interface Tok { kind: TokKind; text: string }

const KEYWORDS = new Set('def return for in if else elif while import from as class with not and or None True False lambda yield break continue pass assert try except raise const let function'.split(' '))
const RE = /(#.*$|\/\/.*$)|("""[\s\S]*?"""|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')|(\b\d+(?:\.\d+)?(?:e-?\d+)?\b)|([A-Za-z_][A-Za-z0-9_]*)(?=\()|([A-Za-z_][A-Za-z0-9_]*)/gm

export const highlight = (code: string): Tok[] => {
  const out: Tok[] = []
  let last = 0
  for (const m of code.matchAll(RE)) {
    const i = m.index ?? 0
    if (i > last) out.push({ kind: '', text: code.slice(last, i) })
    const [text, comment, str, num, fn, word] = m
    let kind: TokKind = ''
    if (comment) kind = 'c'
    else if (str) kind = 's'
    else if (num) kind = 'n'
    else if (fn) kind = KEYWORDS.has(fn) ? 'k' : 'f'
    else if (word && KEYWORDS.has(word)) kind = 'k'
    out.push({ kind, text })
    last = i + text.length
  }
  if (last < code.length) out.push({ kind: '', text: code.slice(last) })
  return out
}
