import { Suspense, lazy, useState } from 'react'
import { highlight } from '../lib/highlight'
import { sourceUrl } from '../data/curriculum'
import { REPO_SCRIPTS } from '../data/repoFiles'
import { assembleTryIt, openInPlayground } from '../lib/playground'

const RepoRunner = lazy(() => import('./RepoRunner').then((m) => ({ default: m.RepoRunner })))

/**
 * A short code block with a copy button. Keep blocks small (aim for under ~15 lines)
 * and introduce code progressively. `source` links to the real file in the repo.
 *
 * Lesson blocks are excerpts, so they rarely run on their own. Two ways to run them:
 * - `source` is a repo script that runs in the browser: "Run the whole file" opens it below the block.
 * - `setup` (made-up inputs the excerpt needs) and/or `show` (code that prints the result), or
 *   `standalone` when it runs as it is: "Try it" opens setup + excerpt + show in the playground.
 *   tryIt.test.tsx runs every such program in real Python and fails if one errors or prints nothing.
 */
export interface CodeBlockInfo { code: string; title?: string; setup?: string; show?: string }
/** Tests set this to collect every block a lesson renders. */
export const codeBlockSink: { current: ((b: CodeBlockInfo) => void) | null } = { current: null }

export function Code({ children, title, source, lang = 'python', setup, show, standalone }: {
  children: string; title?: string; source?: string; lang?: string; setup?: string; show?: string; standalone?: boolean
}) {
  const [copied, setCopied] = useState(false)
  const [runFile, setRunFile] = useState(false)
  const code = children.replace(/^\n/, '').replace(/\s+$/, '')
  const runnable = !!source && Object.prototype.hasOwnProperty.call(REPO_SCRIPTS, source)
  const tryable = lang === 'python' && (setup !== undefined || show !== undefined || !!standalone)
  if (tryable) codeBlockSink.current?.({ code, title, setup, show })
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
          {tryable && <button onClick={() => openInPlayground(assembleTryIt({ code, setup, show }), title)} title="Edit and run this in the Python playground">Try it</button>}
          {runnable && <button className="code-run" onClick={() => setRunFile(!runFile)} aria-expanded={runFile}>{runFile ? 'Hide the file' : 'Run the whole file'}</button>}
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
