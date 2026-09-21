import { useState } from 'react'
import { highlight } from '../lib/highlight'
import { sourceUrl } from '../data/curriculum'

/**
 * A short code block with a copy button. Keep blocks small (aim for under ~15 lines)
 * and introduce code progressively. `source` links to the real file in the repo.
 */
export function Code({ children, title, source, lang = 'python' }: { children: string; title?: string; source?: string; lang?: string }) {
  const [copied, setCopied] = useState(false)
  const code = children.replace(/^\n/, '').replace(/\s+$/, '')
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      /* clipboard unavailable */
    }
  }
  return (
    <div className="code">
      <div className="code-bar">
        <span>{title ?? lang}</span>
        <span className="spacer" />
        {source && <a href={sourceUrl(source)} target="_blank" rel="noreferrer">{source.split('/').pop()} ↗</a>}
        <button onClick={copy} aria-live="polite">{copied ? 'Copied' : 'Copy'}</button>
      </div>
      <pre tabIndex={0}><code>{highlight(code).map((t, i) => (t.kind ? <span key={i} className={`tok-${t.kind}`}>{t.text}</span> : t.text))}</code></pre>
    </div>
  )
}
