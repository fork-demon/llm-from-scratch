// The Transformer block, every component clickable, with real numbers flowing through
// a 3-token, 4-number example. Toggles "remove" a component so its job becomes visible.
import { useMemo, useState, type KeyboardEvent, type ReactNode } from 'react'
import { Lab, MatrixView, Slider } from '../components/ui'
import { Code } from '../components/Code'
import { cosine, fmt, type Mat } from '../lib/math'
import { BLOCK_TOKENS, blockDemo, blockForward, demoInput, depthScales, meanOf, stdOf, swapRows } from '../lib/block'
import './TransformerBlockExplorer.css'

type NodeId = 'emb' | 'pos' | 'ln1' | 'attn' | 'add1' | 'ln2' | 'ffn' | 'add2' | 'out'
const ORDER: NodeId[] = ['emb', 'pos', 'ln1', 'attn', 'add1', 'ln2', 'ffn', 'add2', 'out']

const sameMat = (a: Mat, b: Mat) => a.every((row, i) => row.every((v, j) => Math.abs(v - b[i][j]) < 1e-9))

export function TransformerBlockExplorer() {
  const demo = useMemo(() => blockDemo(), [])
  const [node, setNode] = useState<NodeId>('attn')
  const [positions, setPositions] = useState(true)
  const [residual, setResidual] = useState(true)
  const [causal, setCausal] = useState(true)
  const [swapped, setSwapped] = useState(false)

  const original = BLOCK_TOKENS
  const reversed = [original[2], original[1], original[0]]
  const tokens = swapped ? reversed : original
  const run = (toks: string[]) => blockForward(demoInput(toks, positions, demo), demo.params, { residual, causal })
  const t = useMemo(() => run(tokens), [tokens.join(' '), positions, residual, causal]) // eslint-disable-line react-hooks/exhaustive-deps
  const outA = useMemo(() => run(original).out, [positions, residual, causal]) // eslint-disable-line react-hooks/exhaustive-deps
  const outB = useMemo(() => run(reversed).out, [positions, residual, causal]) // eslint-disable-line react-hooks/exhaustive-deps
  const orderBlind = sameMat(outB, swapRows(outA, 0, 2))
  const tok = tokens.map((w) => demo.tokEmb[w])
  const T = 3
  const D = 4
  const keep = t.out.map((row, i) => cosine(row, t.input[i]))
  // how alike are the output rows of different tokens? (1 = the tokens have become indistinguishable)
  const alike = Math.min(cosine(t.out[0], t.out[1]), cosine(t.out[0], t.out[2]), cosine(t.out[1], t.out[2]))

  const stats = (m: Mat) => (
    <span className="mono" style={{ fontSize: 13 }}>
      {m.map((row, i) => <span key={i} style={{ display: 'block' }}>{tokens[i]}: mean {fmt(meanOf(row))}, std {fmt(stdOf(row))}</span>)}
    </span>
  )

  const details: Record<NodeId, { title: string; problem: ReactNode; io: ReactNode; code: string; demo: ReactNode }> = {
    emb: {
      title: 'Token embeddings',
      problem: 'A token id is just a row number. The model needs a vector it can do arithmetic on.',
      io: <>ids <span className="mono">({T})</span> → vectors <span className="mono">({T}, {D})</span></>,
      code: `self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)\n...\nself.tok_emb(idx)        # (B, T, D): one row per token`,
      demo: <MatrixView m={tok} rows={tokens} heat tone="neutral" caption={`token embeddings (${T}, ${D}): hand-picked here, learned in a real model`} />,
    },
    pos: {
      title: '+ positional information',
      problem: 'Attention compares vectors, not places. Without help, “dog bites man” and “man bites dog” look the same to it.',
      io: <><span className="mono">({T}, {D})</span> + <span className="mono">({T}, {D})</span> → <span className="mono">({T}, {D})</span>. Row t gets the vector for slot t.</>,
      code: `self.pos_emb = nn.Embedding(cfg.context_len, cfg.n_embd)\npos = torch.arange(T, device=idx.device)\nx = self.tok_emb(idx) + self.pos_emb(pos)   # meaning + position`,
      demo: positions ? (
        <div className="matrix-row">
          <MatrixView m={tok} rows={tokens} heat caption="token" />
          <span>+</span>
          <MatrixView m={demo.posEmb} rows={['slot 0', 'slot 1', 'slot 2']} heat caption="position" />
          <span>=</span>
          <MatrixView m={t.input} rows={tokens} heat tone="accent" caption="x: block input" />
        </div>
      ) : (
        <><p className="lab-note"><b>Removed.</b> The block input is the bare token embeddings: “dog” is the same vector wherever it stands.</p><MatrixView m={t.input} rows={tokens} heat tone="accent" caption="x: block input = token embeddings" /></>
      ),
    },
    ln1: {
      title: 'LayerNorm 1',
      problem: 'Vectors drift in size from layer to layer. A sub-layer that receives inputs of unpredictable scale is hard to train.',
      io: <><span className="mono">({T}, {D})</span> → <span className="mono">({T}, {D})</span>. Each token’s vector is rescaled on its own: mean 0, spread 1.</>,
      code: `self.ln1 = nn.LayerNorm(cfg.n_embd)\n...\nself.ln1(x)      # inside: (x - mean) / std * gain + bias, per token`,
      demo: (
        <>
          <div className="matrix-row"><MatrixView m={t.input} rows={tokens} heat caption="in: x" /><span>→</span><MatrixView m={t.ln1} rows={tokens} heat tone="accent" caption="out: ln1(x)" /></div>
          <div className="grid-2"><div><b style={{ fontSize: 13 }}>before</b>{stats(t.input)}</div><div><b style={{ fontSize: 13 }}>after</b>{stats(t.ln1)}</div></div>
          <p className="lab-note">Note that x itself is untouched: the normalised copy goes into attention, the original continues down the side rail.</p>
        </>
      ),
    },
    attn: {
      title: 'Masked multi-head self-attention',
      problem: 'A token’s meaning depends on other tokens. This is the only place in the block where tokens exchange information.',
      io: <><span className="mono">({T}, {D})</span> → <span className="mono">({T}, {D})</span>, through 2 heads of width 2 and a ({T}×{T}) weight table each.</>,
      code: `self.attn = CausalSelfAttention(cfg)\n...\nself.attn(self.ln1(x))     # communicate: tokens look at earlier tokens`,
      demo: (
        <>
          <div className="matrix-row">
            {t.attn.weights.map((w, h) => <MatrixView key={h} m={w} rows={tokens} cols={tokens} heat heatMax={1} tone="accent" caption={`weights, head ${h}${causal ? ' (masked)' : ' (no mask)'}`} />)}
          </div>
          <MatrixView m={t.attn.y} rows={tokens} heat tone="v" caption="attention output: what each token gathered" />
          <p className="lab-note">Untrained random weights: who looks at whom is meaningless here. The arithmetic is the real thing.</p>
        </>
      ),
    },
    add1: {
      title: '+ residual connection 1',
      problem: 'If each layer replaced x, the original token information would be gone after one layer, and gradients would have to squeeze through every layer on the way back.',
      io: <><span className="mono">({T}, {D})</span> + <span className="mono">({T}, {D})</span> → <span className="mono">({T}, {D})</span>. Plain element-wise addition.</>,
      code: residual ? `x = x + self.attn(self.ln1(x))   # residual: block outputs a CORRECTION` : `x = self.attn(self.ln1(x))       # residual removed: x is REPLACED`,
      demo: residual ? (
        <div className="matrix-row"><MatrixView m={t.input} rows={tokens} heat caption="x (came down the rail)" /><span>+</span><MatrixView m={t.attn.y} rows={tokens} heat tone="v" caption="attention output" /><span>=</span><MatrixView m={t.afterAttn} rows={tokens} heat tone="accent" caption="new x" /></div>
      ) : (
        <><p className="lab-note"><b>Removed.</b> The new x is only what attention produced. Whatever was in x and did not pass through the values is lost.</p><MatrixView m={t.afterAttn} rows={tokens} heat tone="accent" caption="new x = attention output" /></>
      ),
    },
    ln2: {
      title: 'LayerNorm 2',
      problem: 'Same problem, same fix: the feed-forward network should also receive inputs of a standard size.',
      io: <><span className="mono">({T}, {D})</span> → <span className="mono">({T}, {D})</span></>,
      code: `self.ln2 = nn.LayerNorm(cfg.n_embd)\n...\nself.ln2(x)`,
      demo: (
        <>
          <div className="matrix-row"><MatrixView m={t.afterAttn} rows={tokens} heat caption="in" /><span>→</span><MatrixView m={t.ln2} rows={tokens} heat tone="accent" caption="out" /></div>
          <div className="grid-2"><div><b style={{ fontSize: 13 }}>before</b>{stats(t.afterAttn)}</div><div><b style={{ fontSize: 13 }}>after</b>{stats(t.ln2)}</div></div>
        </>
      ),
    },
    ffn: {
      title: 'Feed-forward network (MLP)',
      problem: 'Attention only takes weighted averages of value vectors. Something has to process what each token just gathered.',
      io: <><span className="mono">({T}, {D})</span> → <span className="mono">({T}, {4 * D})</span> → <span className="mono">({T}, {D})</span>. Each row is processed alone: no token sees another here.</>,
      code: `nn.Linear(cfg.n_embd, 4 * cfg.n_embd),\nnn.GELU(),                       # smooth ReLU; same hinge idea\nnn.Linear(4 * cfg.n_embd, cfg.n_embd),`,
      demo: (
        <>
          <div className="table-scroll"><MatrixView m={t.ffn.act} rows={tokens} heat tone="accent" digits={1} caption={`hidden layer after ReLU (${T}, ${4 * D}): ${t.ffn.act.flat().filter((v) => v === 0).length} of ${T * 4 * D} neurons are switched off`} /></div>
          <MatrixView m={t.ffn.out} rows={tokens} heat tone="accent" caption={`projected back to (${T}, ${D})`} />
          <p className="lab-note">This demo uses ReLU so you can check it by hand. The repo uses GELU, a smoothed ReLU.</p>
        </>
      ),
    },
    add2: {
      title: '+ residual connection 2',
      problem: 'Same as the first: keep what we had, add what the MLP worked out.',
      io: <><span className="mono">({T}, {D})</span> + <span className="mono">({T}, {D})</span> → <span className="mono">({T}, {D})</span></>,
      code: residual ? `x = x + self.ffn(self.ln2(x))    # gradient highway through the '+'` : `x = self.ffn(self.ln2(x))        # residual removed`,
      demo: residual ? (
        <div className="matrix-row"><MatrixView m={t.afterAttn} rows={tokens} heat caption="x" /><span>+</span><MatrixView m={t.ffn.out} rows={tokens} heat caption="MLP output" /><span>=</span><MatrixView m={t.out} rows={tokens} heat tone="accent" caption="block output" /></div>
      ) : (
        <><p className="lab-note"><b>Removed.</b> The block output is only the MLP’s output.</p><MatrixView m={t.out} rows={tokens} heat tone="accent" caption="block output" /></>
      ),
    },
    out: {
      title: 'Block output → next block',
      problem: 'One round of “communicate, then compute” is not enough. We want to repeat it.',
      io: <><span className="mono">({T}, {D})</span> in, <span className="mono">({T}, {D})</span> out. Same shape, so the output can be fed straight into another block.</>,
      code: `self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])`,
      demo: (
        <>
          <div className="matrix-row"><MatrixView m={t.input} rows={tokens} heat caption="block input" /><span>→</span><MatrixView m={t.out} rows={tokens} heat tone="accent" caption="block output" /></div>
          <p className="lab-note">How much of each token survived? Cosine similarity between a token’s input and output vector (1 = same direction): <span className="mono">{tokens.map((w, i) => `${w} ${fmt(keep[i])}`).join(', ')}</span>.</p>
        </>
      ),
    },
  }

  const d = details[node]
  const key = (id: NodeId) => (e: KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setNode(id) }
  }
  const box = (id: NodeId, y: number, label: string, sub: string, removed = false) => (
    <g className={`TransformerBlockExplorer-node${node === id ? ' active' : ''}${removed ? ' removed' : ''}`} role="button" tabIndex={0} aria-pressed={node === id} aria-label={`${details[id].title}${removed ? ' (removed)' : ''}`} onClick={() => setNode(id)} onKeyDown={key(id)}>
      <rect x={100} y={y} width={190} height={42} rx={8} />
      <text x={195} y={y + 19} textAnchor="middle">{label}</text>
      <text className="sub" x={195} y={y + 34} textAnchor="middle">{sub}</text>
    </g>
  )
  const plus = (id: NodeId, cy: number, removed: boolean) => (
    <g className={`TransformerBlockExplorer-node${node === id ? ' active' : ''}${removed ? ' removed' : ''}`} role="button" tabIndex={0} aria-pressed={node === id} aria-label={`${details[id].title}${removed ? ' (removed)' : ''}`} onClick={() => setNode(id)} onKeyDown={key(id)}>
      <circle cx={195} cy={cy} r={16} />
      <text x={195} y={cy + 6} textAnchor="middle" style={{ fontSize: 20 }}>+</text>
    </g>
  )
  const wire = (y1: number, y2: number) => <line className="TransformerBlockExplorer-wire" x1={195} y1={y1} x2={195} y2={y2} markerEnd="url(#tbe-arrow)" />

  return (
    <Lab
      title="Inside one Transformer block"
      goal={<>Click any part of the diagram. For each one you get: the problem it solves, the shapes, the line of <code>tiny_gpt.py</code>, and the actual numbers for three tokens. Then use the switches to <b>remove</b> a part and watch what breaks.</>}
    >
      <div className="TransformerBlockExplorer-layout">
        <div>
          <svg className="TransformerBlockExplorer-diagram" viewBox="0 0 320 590" role="group" aria-label="Diagram of a Transformer block. Each component is a button.">
            <defs>
              <marker id="tbe-arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="var(--ink-3)" /></marker>
              <marker id="tbe-arrow-acc" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="var(--accent)" /></marker>
            </defs>
            <rect x={30} y={128} width={280} height={366} rx={12} fill="none" stroke="var(--q)" strokeWidth={1.5} strokeDasharray="7 5" />
            <text x={42} y={486} style={{ fontSize: 12, fill: 'var(--q)' }}>one block (× N)</text>

            {wire(52, 66)}{wire(108, 156)}{wire(198, 212)}{wire(254, 274)}{wire(308, 326)}{wire(368, 382)}{wire(424, 444)}{wire(478, 522)}
            <path className={`TransformerBlockExplorer-rail${residual ? '' : ' removed'}`} d="M 195 138 H 60 V 292 H 177" markerEnd="url(#tbe-arrow-acc)" />
            <path className={`TransformerBlockExplorer-rail${residual ? '' : ' removed'}`} d="M 195 316 H 60 V 462 H 177" markerEnd="url(#tbe-arrow-acc)" />
            <text x={54} y={220} textAnchor="middle" transform="rotate(-90 54 220)" style={{ fontSize: 11, fill: 'var(--accent-ink)' }}>x, untouched</text>
            <text x={54} y={392} textAnchor="middle" transform="rotate(-90 54 392)" style={{ fontSize: 11, fill: 'var(--accent-ink)' }}>x, untouched</text>

            {box('emb', 10, 'Token embeddings', `(${T}) → (${T}, ${D})`)}
            {box('pos', 66, '+ position vectors', `(${T}, ${D})`, !positions)}
            {box('ln1', 156, 'LayerNorm', `(${T}, ${D})`)}
            {box('attn', 212, 'Self-attention', 'communicate')}
            {plus('add1', 292, !residual)}
            {box('ln2', 326, 'LayerNorm', `(${T}, ${D})`)}
            {box('ffn', 382, 'Feed-forward (MLP)', 'compute')}
            {plus('add2', 462, !residual)}
            {box('out', 522, 'Output → next block', `(${T}, ${D})`)}
          </svg>
          <div className="btn-row" style={{ justifyContent: 'center', marginTop: 8 }}>
            <button className="btn small" disabled={node === ORDER[0]} onClick={() => setNode(ORDER[ORDER.indexOf(node) - 1])}>← Previous part</button>
            <button className="btn small primary" disabled={node === ORDER[ORDER.length - 1]} onClick={() => setNode(ORDER[ORDER.indexOf(node) + 1])}>Next part</button>
          </div>
        </div>

        <div className="TransformerBlockExplorer-panel" aria-live="polite">
          <h4>{d.title}</h4>
          <dl>
            <dt>Problem</dt><dd>{d.problem}</dd>
            <dt>In → out</dt><dd>{d.io}</dd>
          </dl>
          <Code source="phase3-transformers/tiny_gpt.py" title="tiny_gpt.py">{d.code}</Code>
          <div style={{ fontSize: 12.5, textTransform: 'uppercase', letterSpacing: '0.04em', color: 'var(--ink-3)', margin: '12px 0 4px' }}>Live numbers: “{tokens.join(' ')}”</div>
          {d.demo}
        </div>
      </div>

      <h4 style={{ fontSize: 17, margin: '22px 0 6px' }}>Remove a part</h4>
      <div className="controls">
        <label style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 14.5 }}><input type="checkbox" checked={positions} onChange={(e) => setPositions(e.target.checked)} /> Positional information</label>
        <label style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 14.5 }}><input type="checkbox" checked={residual} onChange={(e) => setResidual(e.target.checked)} /> Residual connections</label>
        <label style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 14.5 }}><input type="checkbox" checked={causal} onChange={(e) => setCausal(e.target.checked)} /> Causal mask</label>
        <label style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 14.5 }}><input type="checkbox" checked={swapped} onChange={(e) => setSwapped(e.target.checked)} /> Feed “man bites dog” instead</label>
      </div>

      <div className="grid-2">
        <div className="card" aria-live="polite">
          <h4 style={{ fontSize: 16, marginBottom: 6 }}>Experiment 1: does word order matter?</h4>
          <div className="matrix-row">
            <MatrixView m={outA} rows={original} heat tone="accent" caption="output for “dog bites man”" />
            <MatrixView m={outB} rows={reversed} heat tone="accent" caption="output for “man bites dog”" />
          </div>
          {orderBlind ? (
            <p className="lab-note" style={{ color: 'var(--bad)' }}><b>Order-blind.</b> The two outputs contain exactly the same rows, only in a different order: “dog” gets the same vector whether it bites or is bitten. Nothing downstream can tell the two sentences apart.</p>
          ) : !positions ? (
            <p className="lab-note">Positions are off, but the <b>mask</b> is on, and the mask leaks a little order: each token sees a different set of earlier tokens. Switch the mask off too, to see attention’s order-blindness in its pure form.</p>
          ) : (
            <p className="lab-note"><b>Order matters.</b> “dog” in slot 0 and “dog” in slot 2 enter the block as different vectors, so they leave as different vectors.</p>
          )}
          <button className="btn small" onClick={() => { setPositions(false); setCausal(false) }}>Run it: positions off, mask off</button>
        </div>
        <div className="card" aria-live="polite">
          <h4 style={{ fontSize: 16, marginBottom: 6 }}>Experiment 2: what survives the block?</h4>
          <p style={{ fontSize: 14.5 }}>Similarity between each token’s vector going in and coming out (1 = same direction, 0 = unrelated, negative = opposite):</p>
          <div className="readout">{tokens.map((w, i) => <span key={i}>{w}: <b>{fmt(keep[i])}</b></span>)}</div>
          {residual ? (
            <p className="lab-note">With residuals, the output is the input <em>plus a correction</em>. Each token is still recognisably itself.</p>
          ) : (
            <p className="lab-note" style={{ color: 'var(--bad)' }}><b>Residuals removed.</b> The output is whatever the last sub-layer produced, and it bears little relation to the tokens that went in. {alike > 0.95 && <>Worse, look at the block output: the three rows are now almost identical to each other (similarity {fmt(alike)}). After one block the model has already lost track of which token is which.</>}</p>
          )}
          <button className="btn small" onClick={() => { setResidual(!residual); setNode('out') }}>{residual ? 'Run it: remove the residuals' : 'Put the residuals back'}</button>
        </div>
      </div>
      {(!positions || !residual || !causal || swapped) && <div className="btn-row" style={{ marginTop: 10 }}><button className="btn small" onClick={() => { setPositions(true); setResidual(true); setCausal(true); setSwapped(false) }}>Reset everything</button></div>}
    </Lab>
  )
}

/* ---------- second lab: stack N blocks, watch the size of the numbers ---------- */
export function StackDepthLab() {
  const [N, setN] = useState(12)
  const [wscale, setWscale] = useState(1)
  const withNorm = useMemo(() => depthScales(24, { norm: true, residual: true, weightScale: wscale }), [wscale])
  const without = useMemo(() => depthScales(24, { norm: false, residual: true, weightScale: wscale }), [wscale])

  const W = 640
  const H = 260
  const pad = { l: 56, r: 110, t: 14, b: 36 }
  const top = Math.max(2, Math.ceil(Math.log10(Math.max(...without.slice(0, N + 1), ...withNorm.slice(0, N + 1)))))
  const x = (i: number) => pad.l + (i / N) * (W - pad.l - pad.r)
  const y = (v: number) => pad.t + (1 - (Math.log10(Math.max(v, 0.1)) + 1) / (top + 1)) * (H - pad.t - pad.b)
  const line = (s: number[]) => s.slice(0, N + 1).map((v, i) => `${i === 0 ? 'M' : 'L'} ${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(' ')
  const ticks = Array.from({ length: top + 2 }, (_, k) => k - 1)
  const label = (e: number) => (e < 0 ? '0.1' : e === 0 ? '1' : e <= 3 ? String(10 ** e) : `10^${e}`)

  return (
    <Lab
      title="Stack N blocks: how big do the numbers get?"
      goal={<>Drag the depth slider. Both stacks keep their residual connections; only one has LayerNorm. Compare the size of a typical number after N blocks. Then shrink the weights and see when the difference disappears.</>}
    >
      <div className="controls">
        <Slider label="number of blocks N" value={N} min={1} max={24} step={1} onChange={setN} />
        <Slider label="size of the random weights" value={wscale} min={0.5} max={1.5} step={0.1} onChange={setWscale} format={(v) => `${v.toFixed(1)}×`} />
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', display: 'block' }} role="img" aria-label={`Typical activation size after each of ${N} blocks. With LayerNorm it ends at ${fmt(withNorm[N], 1)}. Without it ends at ${without[N].toExponential(1)}.`}>
        {ticks.map((e) => (
          <g key={e}>
            <line className="gridline" x1={pad.l} x2={W - pad.r} y1={y(10 ** e)} y2={y(10 ** e)} />
            <text x={pad.l - 8} y={y(10 ** e) + 4} textAnchor="end" fontSize={11} style={{ fill: 'var(--ink-3)' }}>{label(e)}</text>
          </g>
        ))}
        <line className="axis" x1={pad.l} x2={W - pad.r} y1={H - pad.b} y2={H - pad.b} />
        {[...new Set([0, Math.round(N / 2), N])].map((i) => <text key={i} x={x(i)} y={H - pad.b + 16} textAnchor="middle" fontSize={11} style={{ fill: 'var(--ink-3)' }}>{i}</text>)}
        <text x={(pad.l + W - pad.r) / 2} y={H - 4} textAnchor="middle" fontSize={11} style={{ fill: 'var(--ink-3)' }}>blocks passed (log scale upward)</text>
        <path d={line(without)} fill="none" stroke="var(--bad)" strokeWidth={2.2} strokeDasharray="6 4" />
        <path d={line(withNorm)} fill="none" stroke="var(--good)" strokeWidth={2.4} />
        <text x={x(N) + 6} y={y(without[N]) + 4} fontSize={12} style={{ fill: 'var(--bad)' }}>no LayerNorm</text>
        <text x={x(N) + 6} y={y(withNorm[N]) + (Math.abs(y(withNorm[N]) - y(without[N])) < 14 ? 18 : 4)} fontSize={12} style={{ fill: 'var(--good)' }}>with LayerNorm</text>
      </svg>
      <div className="readout" aria-live="polite">
        <span>typical size going in: <b>{fmt(withNorm[0])}</b></span>
        <span>after {N} blocks, with LayerNorm: <b>{fmt(withNorm[N], 1)}</b></span>
        <span>without: <b>{without[N] >= 1000 ? without[N].toExponential(1) : fmt(without[N], 1)}</b></span>
      </div>
      <p className="lab-note">
        Why the difference? Without normalisation, a sub-layer that receives numbers twice as big returns numbers about twice as big, and adds them back. Growth <b>compounds</b>, like interest. With LayerNorm every sub-layer always receives size-1 input, so it can only <b>add</b> a bounded amount per block.
      </p>
      <p className="lab-note">
        <b>Honest notes.</b> These are random, untrained weights, chosen larger than a careful initialisation so that the effect shows in 24 layers. At 0.5× nothing explodes even without LayerNorm. The real danger is during training: gradient descent keeps changing the weights, and without normalisation nothing keeps the scale in check. Also note the green line is not flat: in a pre-norm block the side rail itself is never normalised, it just grows slowly.
      </p>
    </Lab>
  )
}
