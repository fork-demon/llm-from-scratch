// The eval harness of eval_harness.py, live: 24 golden questions through the real mini-RAG port,
// three scorers, a bootstrap interval, a per-category breakdown, failure inspection and a paired A/B test.
import { useMemo, useState } from 'react'
import { Lab, Slider } from '../components/ui'
import { fmt } from '../lib/math'
import {
  CATEGORIES, DEFAULT_CONFIG, DIAGNOSIS_TEXT, GOLDEN_SET, HUMAN_LABELS_DEFAULT, accuracy, bootstrapInterval, byCategory, cohensKappa, compare,
  evaluate, normalInterval, recallAtK, type EvalConfig, type EvalRow, type ScorerName,
} from '../lib/evals'

const SCORER_LABEL: Record<ScorerName, string> = { exact: 'Exact match', substring: 'Substring match', rubric: 'Rubric judge (stand-in)' }
const SCORER_NOTE: Record<ScorerName, string> = {
  exact: 'Passes only if the answer IS the reference string. The system answers in full sentences, so every answerable item fails. Only refusals can match.',
  substring: 'Passes if the reference string appears inside the answer. “within 5 minutes” is not inside “within five minutes”, so correct answers fail on formatting.',
  rubric: 'Three checks: did it answer or refuse as it should, does it state the reference facts (number words, articles and word order ignored), does it cite the right document. A deterministic function standing in for an LLM judge.',
}
const PRESETS: { label: string; config: EvalConfig }[] = [
  { label: 'same as A', config: DEFAULT_CONFIG },
  { label: '1 sentence per chunk, no overlap', config: { ...DEFAULT_CONFIG, sentencesPerChunk: 1, overlap: 0 } },
  { label: 'threshold 0.50', config: { ...DEFAULT_CONFIG, threshold: 0.5 } },
  { label: 'threshold 0.90', config: { ...DEFAULT_CONFIG, threshold: 0.9 } },
]
const same = (a: EvalConfig, b: EvalConfig) => a.sentencesPerChunk === b.sentencesPerChunk && a.overlap === b.overlap && a.k === b.k && a.threshold === b.threshold
const pct = (x: number) => `${(x * 100).toFixed(1)}%`
const signed = (x: number) => `${x >= 0 ? '+' : '−'}${Math.abs(x * 100).toFixed(1)}`

interface Results { scorer: ScorerName; a: EvalConfig; b: EvalConfig; rowsA: EvalRow[]; rowsB: EvalRow[] }
const run = (scorer: ScorerName, a: EvalConfig, b: EvalConfig): Results => ({ scorer, a, b, rowsA: evaluate(a, scorer), rowsB: evaluate(b, scorer) })

function ConfigCard({ name, tone, config, onChange }: { name: string; tone: string; config: EvalConfig; onChange: (c: EvalConfig) => void }) {
  const overlap = Math.min(config.overlap, config.sentencesPerChunk - 1)
  return (
    <div className="card" style={{ padding: 14, borderTop: `3px solid ${tone}` }}>
      <h4 style={{ fontSize: 15.5, margin: '0 0 8px' }}>{name}</h4>
      <Slider label="Sentences per chunk" value={config.sentencesPerChunk} min={1} max={4} step={1} onChange={(v) => onChange({ ...config, sentencesPerChunk: v, overlap: Math.min(config.overlap, v - 1) })} />
      <Slider label="Overlap" value={overlap} min={0} max={Math.max(0, config.sentencesPerChunk - 1)} step={1} onChange={(v) => onChange({ ...config, overlap: v })} />
      <Slider label="k (chunks retrieved)" value={config.k} min={1} max={6} step={1} onChange={(v) => onChange({ ...config, k: v })} />
      <Slider label="Refusal threshold" value={config.threshold} min={0.1} max={0.9} step={0.05} onChange={(v) => onChange({ ...config, threshold: Math.round(v * 100) / 100 })} format={(v) => v.toFixed(2)} />
    </div>
  )
}

/** Accuracy of A and B as points with their 95% bootstrap intervals, on one 0 to 100% axis. */
function IntervalChart({ series }: { series: { label: string; acc: number; ci: [number, number]; tone: string }[] }) {
  const W = 640
  const L = 34
  const R = 96
  const x = (p: number) => L + p * (W - L - R)
  return (
    <svg viewBox={`0 0 ${W} ${46 + series.length * 34}`} style={{ width: '100%', maxWidth: W, display: 'block' }} role="img"
      aria-label={series.map((s) => `${s.label}: ${pct(s.acc)}, 95 percent interval ${pct(s.ci[0])} to ${pct(s.ci[1])}`).join('. ')}>
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <g key={t}>
          <line className="gridline" x1={x(t)} y1={8} x2={x(t)} y2={16 + series.length * 34} />
          <text x={x(t)} y={34 + series.length * 34} textAnchor="middle" fontSize={12} style={{ fill: 'var(--ink-3)' }}>{t * 100}%</text>
        </g>
      ))}
      {series.map((s, i) => {
        const y = 26 + i * 34
        return (
          <g key={s.label}>
            <text x={L - 10} y={y + 4.5} textAnchor="end" fontSize={14} fontWeight={700}>{s.label}</text>
            <line x1={x(s.ci[0])} y1={y} x2={x(s.ci[1])} y2={y} stroke={s.tone} strokeWidth={3} strokeLinecap="round" opacity={0.55} />
            <line x1={x(s.ci[0])} y1={y - 7} x2={x(s.ci[0])} y2={y + 7} stroke={s.tone} strokeWidth={2} />
            <line x1={x(s.ci[1])} y1={y - 7} x2={x(s.ci[1])} y2={y + 7} stroke={s.tone} strokeWidth={2} />
            <circle cx={x(s.acc)} cy={y} r={6} fill={s.tone} stroke="var(--paper)" strokeWidth={2} />
            <text x={W - R + 12} y={y + 4.5} fontSize={13} style={{ fontFamily: 'var(--mono)' }}>{pct(s.acc)}</text>
          </g>
        )
      })}
    </svg>
  )
}

export function EvalLab() {
  const [scorer, setScorer] = useState<ScorerName>('rubric')
  const [a, setA] = useState<EvalConfig>(DEFAULT_CONFIG)
  const [b, setB] = useState<EvalConfig>(PRESETS[1].config)
  const [results, setResults] = useState<Results>(() => run('rubric', DEFAULT_CONFIG, PRESETS[1].config))
  const [side, setSide] = useState<'A' | 'B'>('A')
  const [picked, setPicked] = useState<string | null>(null)

  const stale = results.scorer !== scorer || !same(results.a, a) || !same(results.b, b)
  const stats = useMemo(() => {
    const ci = (rows: EvalRow[]) => bootstrapInterval(rows.map((r) => Number(r.passed)))
    return { ciA: ci(results.rowsA), ciB: ci(results.rowsB), cmp: compare(results.rowsA, results.rowsB), catA: byCategory(results.rowsA), catB: byCategory(results.rowsB), recA: recallAtK(results.rowsA), recB: recallAtK(results.rowsB) }
  }, [results])
  const rows = side === 'A' ? results.rowsA : results.rowsB
  const failing = rows.filter((r) => !r.passed)
  const detail = rows.find((r) => r.item.id === picked) ?? null
  const accA = accuracy(results.rowsA)
  const human = GOLDEN_SET.map((i) => HUMAN_LABELS_DEFAULT[i.id])
  const kappa = same(results.a, DEFAULT_CONFIG) ? cohensKappa(results.rowsA.map((r) => r.passed), human) : null
  const agree = results.rowsA.filter((r, i) => r.passed === human[i]).length
  const { cmp } = stats

  return (
    <Lab
      title="An eval harness you can run"
      goal={<>24 golden questions go through the real mini-RAG pipeline from <a href="#/lesson/rag">lesson 9.1</a>. Change B, press <b>Run</b>, and decide: did your change help, hurt, or can you not tell? Then open the failures and find out <b>which component</b> let each one down.</>}
    >
      <div className="steps" role="group" aria-label="Scorer">
        {(Object.keys(SCORER_LABEL) as ScorerName[]).map((s) => <button key={s} className="step-btn" aria-pressed={scorer === s} onClick={() => setScorer(s)}>{SCORER_LABEL[s]}</button>)}
      </div>
      <p className="lab-note" style={{ marginTop: -6 }}>{SCORER_NOTE[scorer]}</p>

      <div className="grid-2">
        <ConfigCard name="A: the baseline" tone="var(--ink-3)" config={a} onChange={setA} />
        <ConfigCard name="B: your change" tone="var(--accent)" config={b} onChange={setB} />
      </div>
      <div className="steps" role="group" aria-label="Presets for B" style={{ marginTop: 12 }}>
        <span className="muted" style={{ fontSize: 13.5, alignSelf: 'center' }}>Set B to:</span>
        {PRESETS.map((p) => <button key={p.label} className="step-btn" aria-pressed={same(b, p.config)} onClick={() => setB(p.config)}>{p.label}</button>)}
      </div>
      <div className="btn-row" style={{ marginBottom: 16 }}>
        <button className="btn primary" onClick={() => { setResults(run(scorer, a, b)); setPicked(null) }}>Run the eval</button>
        {stale && <span className="muted" style={{ fontSize: 14 }} role="status">Settings changed. The results below are from the previous run.</span>}
      </div>

      <div aria-live="polite" style={{ opacity: stale ? 0.55 : 1 }}>
        <IntervalChart series={[{ label: 'A', acc: accA, ci: stats.ciA, tone: 'var(--ink-3)' }, { label: 'B', acc: cmp.accB, ci: stats.ciB, tone: 'var(--accent)' }]} />
        <p className="lab-note" style={{ marginTop: 4 }}>
          Dot: accuracy on 24 items. Bar: 95% bootstrap interval (10,000 resamples). A scores {results.rowsA.filter((r) => r.passed).length}/24, interval {pct(stats.ciA[0])} to {pct(stats.ciA[1])}. The textbook formula gives {pct(normalInterval(accA, 24)[0])} to {pct(normalInterval(accA, 24)[1])}.
        </p>

        <div className="readout" style={{ borderLeft: `4px solid ${cmp.verdict === 'cannot tell' ? 'var(--ink-3)' : cmp.verdict === 'B is better' ? 'var(--good)' : 'var(--bad)'}` }}>
          <span>B − A = <b>{signed(cmp.diff)} points</b></span>
          <span>only A passes: <b>{cmp.onlyA}</b></span>
          <span>only B passes: <b>{cmp.onlyB}</b></span>
          <span>sign test p = <b>{cmp.p.toFixed(3)}</b></span>
          <span>paired 95% interval {signed(cmp.ci[0])} to {signed(cmp.ci[1])}</span>
          <span style={{ flexBasis: '100%', fontFamily: 'var(--sans)', fontSize: 15 }}>
            Verdict: <b>{cmp.verdict === 'cannot tell' ? 'cannot tell: the difference is within noise' : cmp.verdict}</b>
            {cmp.verdict === 'cannot tell' && cmp.onlyA + cmp.onlyB > 0 && <> ({cmp.onlyA + cmp.onlyB} items changed their result. With so few, even a clean sweep could be luck.)</>}
          </span>
        </div>

        <div className="table-scroll">
          <table className="plain mono" style={{ fontSize: 13.5 }}>
            <thead><tr><th>category</th><th>A</th><th>B</th><th>what it tests</th></tr></thead>
            <tbody>
              {CATEGORIES.map((c) => (
                <tr key={c}>
                  <td>{c}</td><td>{stats.catA[c].passed}/{stats.catA[c].total}</td><td>{stats.catB[c].passed}/{stats.catB[c].total}</td>
                  <td style={{ fontFamily: 'var(--sans)' }}>{{ direct: 'the question reuses the document’s words', paraphrase: 'same meaning, the user’s own words', unanswerable: 'not in the corpus: must refuse', adversarial: 'words match a real sentence, the meaning does not' }[c]}</td>
                </tr>
              ))}
              <tr><td>retrieval recall@k</td><td>{stats.recA.hits}/{stats.recA.total}</td><td>{stats.recB.hits}/{stats.recB.total}</td><td style={{ fontFamily: 'var(--sans)' }}>component metric: was the right chunk retrieved? (answerable items only)</td></tr>
            </tbody>
          </table>
        </div>
        {kappa !== null && <p className="lab-note">Scorer against a person: the author hand-labelled the 24 answers of the default configuration. This scorer agrees on <b>{agree}/24</b>, Cohen’s kappa <b>{fmt(kappa)}</b>. Switch scorer, run again and compare.</p>}

        <h4 style={{ marginTop: 18, marginBottom: 6 }}>Failing items</h4>
        <div className="steps" role="group" aria-label="Which configuration’s failures to list" style={{ marginBottom: 8 }}>
          <button className="step-btn" aria-pressed={side === 'A'} onClick={() => { setSide('A'); setPicked(null) }}>A ({results.rowsA.filter((r) => !r.passed).length} failing)</button>
          <button className="step-btn" aria-pressed={side === 'B'} onClick={() => { setSide('B'); setPicked(null) }}>B ({results.rowsB.filter((r) => !r.passed).length} failing)</button>
        </div>
        {failing.length === 0 && <p>Nothing fails. Be suspicious: is the golden set too easy?</p>}
        <div style={{ display: 'grid', gap: 5 }}>
          {failing.map((r) => (
            <button key={r.item.id} className="step-btn" style={{ textAlign: 'left' }} aria-pressed={picked === r.item.id} onClick={() => setPicked(picked === r.item.id ? null : r.item.id)}>
              <span className="mono" style={{ marginRight: 8 }}>✗ {r.item.id}</span>{r.item.q}
            </button>
          ))}
        </div>
        {detail && (
          <div className="card" style={{ marginTop: 10, padding: 14, fontSize: 14.5 }}>
            <div><b>Question</b> ({detail.item.cat}): {detail.item.q}</div>
            <div><b>Expected:</b> {detail.item.expected ?? 'a refusal (the corpus has no answer)'}{detail.item.source && <> · from <code>{detail.item.source}</code></>}</div>
            <div><b>Got:</b> {detail.out.answer.text}</div>
            <div style={{ color: 'var(--bad)' }}><b>Scorer says:</b> {detail.reason}</div>
            <div><b>Located in:</b> {DIAGNOSIS_TEXT[detail.diagnosis]}</div>
            <div style={{ marginTop: 8 }}><b>Retrieved chunks</b> (k = {(side === 'A' ? results.a : results.b).k}):</div>
            {detail.out.retrieved.map((c, i) => {
              const gold = detail.item.gold !== null && c.chunk.text.includes(detail.item.gold)
              return (
                <div key={i} style={{ padding: '6px 0', borderTop: '1px solid var(--rule)' }}>
                  <span className={`chip${gold ? ' acc' : ''}`} style={{ marginRight: 8 }}>[{i + 1}] sim {fmt(c.sim)}{gold ? ' · holds the answer' : ''}</span>{c.chunk.text}
                </div>
              )
            })}
            {detail.item.gold !== null && detail.hit === false && <p className="lab-note" style={{ marginBottom: 0 }}>No retrieved chunk holds the answer. No answerer, however good, could pass this item from this prompt.</p>}
          </div>
        )}
      </div>
      <p className="lab-note" style={{ marginTop: 14 }}><b>Honest note:</b> the pipeline, the answerer and the judge are all deterministic stand-ins, so one run is enough here. With a real LLM sampling at temperature above 0, the same item can pass on one run and fail on the next, and you would average several runs per item.</p>
    </Lab>
  )
}
