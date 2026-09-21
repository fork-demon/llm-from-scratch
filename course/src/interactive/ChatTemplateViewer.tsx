// Edit a conversation, see the exact flat token sequence the model receives,
// and which tokens are training targets during SFT.
import { useMemo, useState } from 'react'
import { Lab } from '../components/ui'
import { countLoss, flatString, generationPrompt, renderChat, type ChatToken, type Message } from '../lib/chatTemplate'

const show = (t: ChatToken) => (t.kind === 'newline' ? '⏎' : t.text.replace(/ /g, '·'))

export function ChatTemplateViewer() {
  const [system, setSystem] = useState('You are a helpful assistant.')
  const [user, setUser] = useState('What is a cat?')
  const [assistant, setAssistant] = useState('A cat is a small furry animal.')
  const [assistantOnly, setAssistantOnly] = useState(true)
  const [view, setView] = useState<'train' | 'infer'>('train')

  const messages: Message[] = useMemo(() => [
    ...(system.trim() ? [{ role: 'system' as const, content: system }] : []),
    { role: 'user' as const, content: user },
    { role: 'assistant' as const, content: assistant },
  ], [system, user, assistant])

  const tokens = useMemo(() => renderChat(messages), [messages])
  const counts = countLoss(tokens, assistantOnly)
  const prompt = generationPrompt(messages.slice(0, -1))

  const field = (id: string, label: string, value: string, set: (v: string) => void, rows = 2) => (
    <div>
      <label htmlFor={id} style={{ fontSize: 13.5, fontWeight: 600 }}>{label}</label>
      <textarea id={id} className="input" rows={rows} value={value} onChange={(e) => set(e.target.value)} style={{ minHeight: 0 }} />
    </div>
  )

  return (
    <Lab
      title="A conversation is one token sequence"
      goal={<>Edit the three messages. Watch the flat string the model actually receives, and which tokens are <b>training targets</b> during SFT. Try making the user message very long: does the number of target tokens change?</>}
    >
      <div style={{ display: 'grid', gap: 8 }}>
        {field('ct-system', 'system message (leave empty to drop it)', system, setSystem)}
        {field('ct-user', 'user message', user, setUser)}
        {field('ct-assistant', 'assistant message (the demonstration written by a person)', assistant, setAssistant)}
      </div>

      <div className="steps" role="tablist" aria-label="View" style={{ marginTop: 14 }}>
        <button role="tab" className="step-btn" aria-selected={view === 'train'} onClick={() => setView('train')}>During SFT training</button>
        <button role="tab" className="step-btn" aria-selected={view === 'infer'} onClick={() => setView('infer')}>At inference time</button>
      </div>

      {view === 'train' ? (
        <div role="tabpanel">
          <label style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 14.5, margin: '6px 0 10px' }}>
            <input type="checkbox" checked={assistantOnly} onChange={(e) => setAssistantOnly(e.target.checked)} /> Count loss on assistant tokens only (the usual choice)
          </label>
          <div className="card" style={{ lineHeight: 2.1 }} aria-label="Token sequence with loss mask">
            {tokens.map((t, i) => {
              const target = assistantOnly ? t.loss : true
              return (
                <span key={i}>
                  <span
                    className="token"
                    title={`${t.role} · ${target ? 'training target' : 'context only'}`}
                    style={{
                      fontFamily: 'var(--mono)', fontSize: 12.5, whiteSpace: 'pre',
                      fontWeight: target ? 700 : 400,
                      opacity: target ? 1 : 0.6,
                      borderBottom: target ? '3px solid var(--accent)' : '3px solid transparent',
                      fontStyle: t.kind === 'special' || t.kind === 'role' ? 'italic' : undefined,
                    }}
                  >
                    {show(t)}
                  </span>
                  {t.kind === 'newline' && <br />}
                </span>
              )
            })}
          </div>
          <p className="lab-note" style={{ marginTop: 8 }}>
            <b>Bold with an underline</b> = training target (the model is graded on predicting this token). Faded = context only (the model reads it but is not graded on it). <i>Italic</i> = special marker tokens. “·” is a space, “⏎” a newline.
          </p>
          <div className="readout" aria-live="polite">
            <span>tokens in the sequence: <b>{counts.total}</b></span>
            <span>training targets: <b>{counts.withLoss}</b></span>
            <span>share graded: <b>{(counts.fraction * 100).toFixed(0)}%</b></span>
          </div>
        </div>
      ) : (
        <div role="tabpanel">
          <p style={{ marginTop: 6 }}>When you call a chat API, the server builds this string from your messages, ending with an <b>open</b> assistant turn. The model then does the only thing it can do: continue the text, until it emits <code>&lt;|im_end|&gt;</code>.</p>
          <pre className="card mono" style={{ fontSize: 12.5, whiteSpace: 'pre-wrap', wordBreak: 'break-word', margin: 0 }} aria-live="polite">{prompt}<span className="acc">▌ the model continues from here</span></pre>
        </div>
      )}

      <details className="deep" style={{ marginTop: 12 }}>
        <summary>Show the raw training string</summary>
        <div className="details-body"><pre className="mono" style={{ fontSize: 12.5, whiteSpace: 'pre-wrap', wordBreak: 'break-word', margin: 0 }}>{flatString(tokens)}</pre></div>
      </details>

      <p className="lab-note" style={{ marginTop: 12 }}>
        <b>Simplified:</b> this is one example format (ChatML-style markers). Every model family defines its own markers, and you must use the ones a model was trained with. Words are split by a toy tokenizer here; a real <a href="#/lesson/tokenization">BPE tokenizer</a> splits differently, but the masking logic is the same. Each marker such as <code>&lt;|im_start|&gt;</code> is a single special token in the vocabulary, not a run of characters.
      </p>
    </Lab>
  )
}
