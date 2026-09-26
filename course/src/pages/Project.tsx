import { PROJECT_PIECES } from '../data/project/paisaPal'
import { lessonById } from '../data/curriculum'
import { useProgress } from '../lib/progress'

/** #/project: the Paisa Pal support bot, piece by piece, and which pieces you have built. */
export function ProjectPage() {
  const progress = useProgress()
  const done = PROJECT_PIECES.filter((p) => progress.exercises[p.exercise]).length
  return (
    <div>
      <h1 className="lesson-title">Your support bot</h1>
      <p className="lesson-question">
        Paisa Pal wants a support bot, and in Parts 8 to 10 you build it: one piece per lesson, written by you, in the browser, checked by tests.
      </p>
      <div className="project-progress">
        <div className="meter" role="progressbar" aria-label="Pieces built" aria-valuemin={0} aria-valuemax={PROJECT_PIECES.length} aria-valuenow={done}>
          <span style={{ width: `${(done / PROJECT_PIECES.length) * 100}%` }} />
        </div>
        <span>{done} of {PROJECT_PIECES.length} pieces built</span>
      </div>
      <ol className="project-pieces">
        {PROJECT_PIECES.map((p) => {
          const built = !!progress.exercises[p.exercise]
          const lesson = lessonById(p.lesson)
          return (
            <li key={p.exercise} className={built ? 'is-built' : undefined}>
              <span className="project-mark" aria-hidden="true">{built ? '✓' : ''}</span>
              <div>
                <a href={`#/lesson/${p.lesson}#ex-${p.exercise}`}><b>{p.piece}</b></a>{built && <span className="sr-only"> (built)</span>}
                <span className="project-does">{p.does}</span>
                <span className="project-lesson">{lesson?.title}</span>
              </div>
            </li>
          )
        })}
      </ol>
      <p className="muted" style={{ fontSize: 14 }}>
        Every piece starts from the same project file, <code>paisa_pal.py</code>: Paisa Pal’s help pages, twelve real customer questions,
        two customers and their tools. Each exercise also gives you working versions of the earlier pieces it needs, so you can build them in any order.
      </p>
    </div>
  )
}
