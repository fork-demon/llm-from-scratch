import { Suspense, lazy, useState } from 'react'
import { highlight } from '../lib/highlight'
import { sourceUrl } from '../data/curriculum'
import { REPO_SCRIPTS } from '../data/repoFiles'
import { openInPlayground } from '../lib/playground'

const RepoRunner = lazy(() => import('./RepoRunner').then((m) => ({ default: m.RepoRunner })))

/**
 * A short code block with a copy button. Keep blocks small (aim for under ~15 lines)
 * and introduce code progressively. `source` links to the real file in the repo.
 *
 * Lesson blocks are excerpts, so they rarely run on their own. Two ways to run them:
 * when the excerpt comes from a repo script that runs in the browser, "Run the whole file"
 * opens that file below the block; any other Python block can be opened in the playground.
 */
export function Code({ children, title, source, lang = 'python' }: { children: string; title?: string; source?: string; lang?: string }) {
  const [copied, setCopied] = useState(false)
  const [runFile, setRunFile] = useState(false)
  const code = children.replace(/^\n/, '').replace(/\s+$/, '')
  const runnable = !!source && Object.prototype.hasOwnProperty.call(REPO_SCRIPTS, source)
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
    <>
      <div className="code">
        <div className="code-bar">
          <span>{title ?? lang}</span>
          <span className="spacer" />
          {source && <a href={sourceUrl(source)} target="_blank" rel="noreferrer">{source.split('/').pop()} ↗</a>}
          {runnable
            ? <button className="code-run" onClick={() => setRunFile(!runFile)} aria-expanded={runFile}>{runFile ? 'Hide the file' : 'Run the whole file'}</button>
            : lang === 'python' && <button onClick={() => openInPlayground(code, title)} title="Edit and run this in the Python playground">Try it</button>}
          <button onClick={copy} aria-live="polite">{copied ? 'Copied' : 'Copy'}</button>
        </div>
        <pre tabIndex={0}><code>{highlight(code).map((t, i) => (t.kind ? <span key={i} className={`tok-${t.kind}`}>{t.text}</span> : t.text))}</code></pre>
      </div>
      {runnable && runFile && (
        <Suspense fallback={<p className="muted">Loading the runner…</p>}>
          <RepoRunner path={source as keyof typeof REPO_SCRIPTS} title={`${source!.split('/').pop()}, the whole file this excerpt comes from`} />
        </Suspense>
      )}
    </>
  )
}
