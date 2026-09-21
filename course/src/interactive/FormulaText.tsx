// Renders the tiny formula markup from lib/notation.ts ("W_i^Q", "x_{<t}") with real sub/superscripts.
import { parseFormula } from '../lib/notation'

export function F({ children }: { children: string }) {
  return (
    <>
      {parseFormula(children).map((p, i) => (p.kind === 'sub' ? <sub key={i}><F>{p.text}</F></sub> : p.kind === 'sup' ? <sup key={i}><F>{p.text}</F></sup> : <span key={i}>{p.text}</span>))}
    </>
  )
}
