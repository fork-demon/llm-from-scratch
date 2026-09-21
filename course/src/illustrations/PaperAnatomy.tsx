// The 15 pages of "Attention Is All You Need" (arXiv v7) as a strip of miniature pages, with the
// five places to look first drawn in and marked in the accent. Page numbers are the real ones.
import { ACC, Caption, FILL, Figure, INK, LINE, SOFT } from './diagramKit'

const PW = 42
const PH = 58
const GAP = 6
const X0 = 22
const Y0 = 44
const px = (page: number) => X0 + (page - 1) * (PW + GAP)

type Feature = 'title' | 'figure' | 'equation' | 'table' | 'hyper' | 'refs' | 'viz' | undefined
const FEATURE: Record<number, Feature> = { 1: 'title', 3: 'figure', 4: 'equation', 7: 'hyper', 8: 'table', 9: 'table', 11: 'refs', 12: 'refs', 13: 'viz', 14: 'viz', 15: 'viz' }
const HOT: Record<number, number> = { 3: 1, 4: 2, 8: 3, 9: 4, 7: 5 }

function Page({ n }: { n: number }) {
  const x = px(n)
  const f = FEATURE[n]
  const hot = HOT[n] !== undefined
  const lines = (from: number, to: number, short = -1) => Array.from({ length: to - from }, (_, i) => (
    <rect key={i} x={x + 5} y={Y0 + from * 5 + i * 5 + 5} width={i === short ? 18 : PW - 10} height={1.6} rx={0.8} fill={INK} opacity={0.22} />
  ))
  return (
    <g>
      <rect x={x} y={Y0} width={PW} height={PH} rx={3} fill={FILL} stroke={hot ? ACC : LINE} strokeWidth={hot ? 1.6 : 1} />
      {f === 'title' && <><rect x={x + 8} y={Y0 + 6} width={PW - 16} height={3} rx={1.5} fill={INK} opacity={0.6} /><rect x={x + 9} y={Y0 + 15} width={PW - 18} height={16} rx={2} fill={INK} opacity={0.14} />{lines(6, 10)}</>}
      {f === 'figure' && <>{lines(0, 1)}<rect x={x + 7} y={Y0 + 12} width={12} height={24} rx={2} fill="none" stroke={ACC} strokeWidth={1.4} /><rect x={x + 23} y={Y0 + 12} width={12} height={30} rx={2} fill="none" stroke={ACC} strokeWidth={1.4} />{lines(9, 10)}</>}
      {f === 'equation' && <>{lines(0, 3)}<rect x={x + 9} y={Y0 + 24} width={PW - 18} height={4} rx={2} fill={ACC} />{lines(5, 10, 3)}</>}
      {f === 'hyper' && <>{lines(0, 4)}<rect x={x + 9} y={Y0 + 29} width={PW - 18} height={3} rx={1.5} fill={ACC} />{lines(6, 8)}<rect x={x + 5} y={Y0 + 49} width={PW - 10} height={1.6} rx={0.8} fill={ACC} /><rect x={x + 5} y={Y0 + 54} width={20} height={1.6} rx={0.8} fill={ACC} /></>}
      {f === 'table' && <>{lines(0, 2)}{[0, 1, 2, 3, 4].map((r) => [0, 1, 2, 3].map((c) => <rect key={`${r}-${c}`} x={x + 6 + c * 8} y={Y0 + 18 + r * 5} width={6} height={3} rx={1} fill={ACC} opacity={r === 0 ? 1 : 0.55} />))}{lines(9, 10)}</>}
      {f === 'refs' && Array.from({ length: 10 }, (_, i) => <rect key={i} x={x + 5 + (i % 2 ? 4 : 0)} y={Y0 + 6 + i * 5} width={PW - 10 - (i % 2 ? 4 : 0)} height={1.4} rx={0.7} fill={INK} opacity={0.14} />)}
      {f === 'viz' && <>{[0, 1].map((r) => <g key={r}>{Array.from({ length: 6 }, (_, i) => <path key={i} d={`M${x + 7 + i * 5.5} ${Y0 + 10 + r * 24} L${x + 7 + ((i * 3 + r) % 6) * 5.5} ${Y0 + 24 + r * 24}`} stroke={INK} strokeWidth={0.8} opacity={0.35} />)}</g>)}</>}
      {f === undefined && lines(0, 10, n % 3 === 0 ? 9 : -1)}
      <text x={x + PW / 2} y={Y0 + PH + 13} textAnchor="middle" fontSize={10.5} style={{ fill: hot ? ACC : SOFT, fontWeight: hot ? 700 : 400 }}>{n}</text>
    </g>
  )
}

// each connector turns at its own height, so no two horizontal runs share a line
const LEVEL: Record<number, number> = { 3: 30, 4: 30, 7: 37, 8: 30, 9: 23 }

const CAPTIONS: { n: number; page: number; cx: number; title: string; lines: string[] }[] = [
  { n: 1, page: 3, cx: 78, title: 'The architecture figure', lines: ['Figure 1, page 3.', 'The whole method', 'in one picture.'] },
  { n: 2, page: 4, cx: 228, title: 'The central equation', lines: ['Equation 1, page 4.', 'Most papers have one', 'that matters.'] },
  { n: 5, page: 7, cx: 380, title: 'The hyperparameters', lines: ['Section 5, page 7. Often', 'in an appendix instead.', 'Needed to reproduce.'] },
  { n: 3, page: 8, cx: 532, title: 'The main results table', lines: ['Table 2, page 8.', 'The evidence for', 'the abstract’s claim.'] },
  { n: 4, page: 9, cx: 682, title: 'The ablation table', lines: ['Table 3, page 9.', 'What happens when each', 'piece is changed.'] },
]

export function PaperAnatomy() {
  return (
    <Figure
      id="paperanat"
      height={248}
      minWidth={700}
      title="The fifteen pages of Attention Is All You Need as a strip of small pages. Five places are highlighted: the architecture figure on page 3, the central equation on page 4, the hyperparameters in section 5 on page 7, the main results table on page 8 and the ablation table on page 9. Page 1 holds the title and abstract, page 10 the conclusion, pages 10 to 12 the references and pages 13 to 15 an appendix of attention visualisations."
      after={<ol className="sr-only">{CAPTIONS.slice().sort((a, b) => a.n - b.n).map((c) => <li key={c.n}>{c.title}: {c.lines.join(' ')}</li>)}</ol>}
    >
      <text x={px(1)} y={16} fontSize={11.5} style={{ fill: SOFT }}>title, abstract</text>
      <text x={px(1)} y={30} fontSize={11.5} style={{ fill: SOFT }}>first pass starts here</text>
      <text x={px(10) + PW / 2} y={30} textAnchor="middle" fontSize={11.5} style={{ fill: SOFT }}>conclusion</text>
      <text x={px(11) + PW + GAP / 2} y={16} textAnchor="middle" fontSize={11.5} style={{ fill: SOFT }}>references</text>
      <text x={px(14) + PW / 2} y={30} textAnchor="middle" fontSize={11.5} style={{ fill: SOFT }}>appendix: pictures of heads</text>
      {Array.from({ length: 15 }, (_, i) => <Page key={i} n={i + 1} />)}

      {CAPTIONS.map((c) => {
        const x = px(c.page) + PW / 2
        return (
          <g key={c.n}>
            <path d={`M${x} ${Y0 + PH + 18} V${Y0 + PH + LEVEL[c.page]} H${c.cx} V${Y0 + PH + 42}`} fill="none" stroke={ACC} strokeWidth={1.3} />
            <Caption x={c.cx} y={Y0 + PH + 58} n={c.n} title={c.title} lines={c.lines} />
          </g>
        )
      })}
    </Figure>
  )
}
