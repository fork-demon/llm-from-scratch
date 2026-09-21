import { useRef, useState } from 'react'
import { LESSONS, PARTS, lessonById } from '../data/curriculum'
import { exportProgress, importProgress, nextUp, parseProgressFile, resetProgress, useProgress } from '../lib/progress'
import { PartArt } from '../illustrations/scenes'
import { PipelineStrip } from '../illustrations/PipelineStrip'
import { TokenTitle } from '../components/TokenTitle'

export function Home() {
  const progress = useProgress()
  const [saveNote, setSaveNote] = useState<string | null>(null)
  const doneLessons = Object.keys(progress.completed).length
  const started = doneLessons > 0 || !!progress.last
  const resume = lessonById(nextUp(progress)) ?? LESSONS[0]
  const hours = Math.round(LESSONS.filter((l) => l.id !== 'capstone').reduce((s, l) => s + l.minutes, 0) / 60)
  const quizzes = Object.values(progress.quiz)
  const quizCorrect = quizzes.reduce((s, q) => s + q.correct, 0)
  const quizTotal = quizzes.reduce((s, q) => s + q.total, 0)
  const fileInput = useRef<HTMLInputElement>(null)

  return (
    <div>
      <section className="hero-split">
        <div>
          <h1><TokenTitle text="Stop using LLMs as magic." /></h1>
          <p className="hero-lede">A course for developers. Start from one dot product and build up to attention, a working GPT, retrieval, fine-tuning and agents. Every idea comes with something you can change and break.</p>
          <div className="btn-row" style={{ marginTop: 26 }}>
            <a className="btn primary" href={`#/lesson/${started ? resume.id : LESSONS[0].id}`}>{started ? `Continue with “${resume.title}”` : 'Start the first lesson'}</a>
            <a className="btn" href="#/map">Open the concept map</a>
          </div>
          <p className="hero-meta">
            {started
              ? `${doneLessons} of ${LESSONS.length} lessons done, ${Object.keys(progress.exercises).length} exercises finished${quizTotal ? `, ${quizCorrect} of ${quizTotal} recall questions right` : ''}.`
              : `${LESSONS.length} lessons, about ${hours} hours, no sign-up.`}
          </p>
        </div>
        <div className="hero-art"><PartArt part="hero" /></div>
      </section>

      <div className="facts">
        <div><b>Written for developers</b><span>Python or Java is enough. You need no machine-learning background, and the maths arrives only when an LLM needs it.</span></div>
        <div><b>See it before the formula</b><span>Each lesson goes from a question to a picture, an experiment, the numbers, the maths, and then the code.</span></div>
        <div><b>Connected to real code</b><span>Every demo mirrors a runnable Python file in this repository. Your progress stays in your browser.</span></div>
      </div>

      <section className="home-section">
        <h2>From your text to the next token</h2>
        <p>This is the whole machine. By the end you can draw it from memory and explain each step. Select a step to open the lesson that teaches it.</p>
        <PipelineStrip />
      </section>

      <section className="home-section">
        <h2>The course</h2>
        <p>{PARTS.length} parts, in order. Each builds only on the ones before it.</p>
        <div className="parts">
          {PARTS.map((part) => {
            const done = part.lessons.filter((l) => progress.completed[l.id]).length
            return (
              <div key={part.id}>
                <a className="part-art" href={`#/lesson/${part.lessons[0].id}`} aria-label={`Open part ${part.number}: ${part.title}`}><PartArt part={part.id} /></a>
                <div className="part-body">
                  <div className="part-kicker"><b>Part {part.number}</b>{done > 0 && <span>{done} of {part.lessons.length} done</span>}</div>
                  <h3>{part.title}</h3>
                  <p>{part.blurb}</p>
                  <ul className="part-lessons">
                    {part.lessons.map((l) => (
                      <li key={l.id}>
                        <a href={`#/lesson/${l.id}`} className={progress.completed[l.id] ? 'done' : undefined}>
                          {l.title}
                          <span>{progress.completed[l.id] ? 'Done' : l.minutes >= 90 ? `${Math.round(l.minutes / 60)} hours` : `${l.minutes} min`}</span>
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            )
          })}
        </div>
      </section>

      <section className="home-section" style={{ marginTop: 64 }}>
        <h2>Your progress</h2>
        <p className="muted" style={{ fontSize: 14 }}>Keyboard: press <kbd>/</kbd> to search, <kbd>j</kbd> and <kbd>k</kbd> to move between lessons.</p>
        <p>Everything you do is stored in this browser only. Save a copy before you clear site data, or to carry your progress to another machine.</p>
        <div className="btn-row">
          <button className="btn small" onClick={() => {
            const blob = new Blob([JSON.stringify(exportProgress(), null, 2)], { type: 'application/json' })
            const a = document.createElement('a')
            a.href = URL.createObjectURL(blob)
            a.download = `llm-course-progress-${new Date().toISOString().slice(0, 10)}.json`
            a.click()
            URL.revokeObjectURL(a.href)
            setSaveNote('Saved to your downloads.')
          }}>Save my progress to a file</button>
          <button className="btn small" onClick={() => fileInput.current?.click()}>Load progress from a file</button>
          {started && <button className="btn small ghost" onClick={() => { if (window.confirm('Erase all saved progress in this browser? Save a copy first if you want to keep it.')) { resetProgress(); setSaveNote('Progress erased.') } }}>Erase my progress</button>}
          {saveNote && <span className="muted" role="status" style={{ fontSize: 14 }}>{saveNote}</span>}
        </div>
        <input
          ref={fileInput}
          type="file"
          accept="application/json,.json"
          className="sr-only"
          onChange={async (e) => {
            const file = e.target.files?.[0]
            e.target.value = ''
            if (!file) return
            try {
              importProgress(parseProgressFile(await file.text()))
              setSaveNote('Loaded. Anything you had already done has been kept.')
            } catch (err) {
              setSaveNote(err instanceof Error ? err.message : 'That file could not be read.')
            }
          }}
        />
      </section>
    </div>
  )
}
