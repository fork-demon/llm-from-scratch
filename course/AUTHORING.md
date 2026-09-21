# Authoring guide: how lessons in this course are written

Read this fully, then read the exemplar: `src/lessons/attention.tsx` and `src/interactive/AttentionPlayground.tsx`.
Your lessons must feel like they were written by the same person.

## The learner

A software developer (Python/Java, APIs, data structures). Little or no ML. Possibly uncomfortable with maths.
Wants to *understand*, not memorise terminology. Never assume calculus, linear algebra, probability, PyTorch.

## The loop every lesson follows

QUESTION → WHY → INTUITION → VISUAL → EXPERIMENT → NUMBERS → MATH → CODE → EXERCISE → RECALL → REAL LLM

Concretely, a lesson is `<Lesson id="...">` containing these section components from `components/lesson.tsx`, **in this order**:

| # | Component | Purpose |
|---|-----------|---------|
| 1 | `<Why>` | Short, human. Start from a concrete question or example, not a definition. |
| 2 | `<Problem>` | The limitation of the previous idea. Usually contains `<WhyExists .../>`. |
| 3 | `<MentalModel>` | Intuition. Analogies go in `<Callout kind="analogy">` and must say where the analogy stops. |
| 4 | `<TryIt>` | The interactive. The learner must be able to *manipulate* something and see numbers change. |
| 5 | `<Numbers>` | A tiny real calculation worked by hand with actual numbers. |
| 6 | `<TheMath>` | Only now the equation, always via `<Equation symbols=[...]>` so every symbol is explained. |
| 7 | `<CodeIt>` | Code introduced a few lines at a time (`<Code>`), then the real function from the repo with `source=`. |
| 8 | `<BreakIt>` | Things to change/break in the interactive or the Python file. Predict first, then check. |
| 9 | `<Exercises>` | 3–5 `<Exercise>`s of different types + usually one `<ExplainBack>`. |
| 10 | `<CheckYourself questions=[...]>` | 3–5 multiple-choice questions that test understanding, not vocabulary. |
| 11 | `<Remember items=[...]>` | 3–5 key ideas. |
| 12 | `<RealLLM>` | Toy → real. Use `<ToyVsReal>`. Source links render automatically from curriculum.ts. |

`<Why>` and `<Remember>` are mandatory (a test checks). Conceptual lessons with no equation or code may drop
`<TheMath>`/`<CodeIt>`/`<Numbers>`, but never drop the interactive, exercises, recall or RealLLM sections
without a very good reason. Section titles can be overridden: `<Why title="...">`.

The last lesson of each part also ends with `<BeforeMovingOn id="part-N" questions=[...]>` placed after
`<RealLLM>`: a mixed quiz that deliberately reaches back to *earlier* parts (spaced recall), optionally with an
`<OrderExercise>` ("rebuild the pipeline from memory") as its child.

Lesson metadata (title, question, sources, where-are-we) lives in `src/data/curriculum.ts`. **Do not edit it.**
Your lesson file name must be exactly `src/lessons/<lesson-id>.tsx` with a default export.

## Writing rules (these matter more than anything else)

- Short sentence. One idea. One example. One visual. Paragraphs of 1–3 sentences. No walls of text.
- **Why before how. How before math. Math before code.** Never show an equation before the learner knows what problem it solves.
- Never introduce a term without first explaining the idea in plain words. For important new jargon use
  `<Term name plain example formal />` (term → plain English → tiny example → formal definition).
- Link known terms to the glossary inline with `<G t="logits">logits</G>` (ids are in `src/data/glossary.ts`).
- No magic. Not "the Transformer understands language"; explain the mechanism. Not "facts are stored in the FFN";
  say what is established and what is uncertain.
- Honesty levels: use `<Callout kind="established">`, `<Callout kind="model">` (simplified mental model),
  `<Callout kind="research">` (active research / debated). Never present an analogy as fact.
- Developer connections go in `<Callout kind="dev">` (attention ≈ soft dict lookup, KV cache ≈ memoisation,
  agent ≈ control loop…). Do not force analogies that mislead.
- Two depth levels: main flow is the "Understand" level. Put rigour and side-questions in `<DeepDive title>`
  (collapsed by default) or use `<Depth understand deeper />`. Deep dives must never be required to continue.
- Exercises never reveal the answer immediately: give 2–3 `hints` that reveal progressively more, then `solution`
  with an explanation of *why*. Use `answer={{ value, tolerance }}` or `answer={{ text: [...] }}` for auto-checking
  whenever the answer is a number or a short token. Exercise ids are globally unique: `<lesson-id>-<slug>`.
  Exercise types: predict, calculate, modify, debug, implement, explain, experiment, trace. Mix them.
- Every number you print must be correct. Compute worked examples carefully (run node/python to verify).
- Refer back to earlier lessons with links: `<a href="#/lesson/softmax">…</a>`. Reuse the same running examples
  across the course where natural: “The cat sat on the ___”, “the river bank”, the prompt “What is a cat?”.
- Fixed colour vocabulary, never reuse for other meanings: query = `.q` (azure), key = `.k` (amber), value = `.v` (magenta).
  `.acc` is the accent.
- Plain typography: use straight JSX text; typographic quotes “ ” are fine. Do not use em dashes; use commas,
  colons or full stops. No emoji.
- Python code shown in lessons must be taken from (or be a faithful, progressive simplification of) the actual
  file in the repository listed under the lesson's `sources` in curriculum.ts. Read that file first. The repo code is
  the source of truth; do not invent a different implementation. Keep each `<Code>` block short (≈ under 15 lines).
- Exercises of type "implement"/"modify" should send the learner to run and change that real Python file.

## Interactives

- One interactive teaches ONE concept. File: `src/interactive/<Name>.tsx`, wrapped in `<Lab title goal>`.
- It must be genuinely manipulable (sliders, editable numbers, clickable steps, typed text) and show real computed
  numbers. No fake animations of static text. If something is simplified or untrained, say so in the UI.
- **All computation lives in pure functions in `src/lib/<topic>.ts` with a `src/lib/<topic>.test.ts` (vitest).**
  Reuse `src/lib/math.ts` (dot, matmul, softmax, attention, topK, topP, sampleIndex…) and `src/lib/rng.ts`
  (seeded RNG: deterministic demos and tests). Where the lib mirrors a Python file in the repo, test that it
  reproduces the Python behaviour (same algorithm, same results on a small example).
- Reusable UI: `Lab, Slider, MatrixView (heat / editable), Bars, Flow (clickable), Tabs, Callout, Card, Term, G,
  DeepDive, Expand, Depth, WhyExists, ToyVsReal, Equation` from `components/ui.tsx`; `Code` from `components/Code.tsx`;
  `Exercise, ExplainBack, OrderExercise, Quiz` from `components/exercise.tsx`. Read those files for props.
- Styling: use the existing classes in `src/styles/global.css` (`card, grid-2, grid-3, readout, controls, steps,
  step-btn, btn, btn small, btn primary, chip, token, plain (table), table-scroll, lab-note, muted, mono, matrix-row`)
  and CSS variables (`--accent, --ink, --ink-2, --ink-3, --rule, --rule-strong, --paper-2, --card, --q, --k, --v,
  --good, --bad, --mono, --serif`). Small inline styles are fine. If you truly need new classes, put them in
  `src/interactive/<Name>.css`, import it from your component, and prefix every class with your component name.
- Charts/diagrams: hand-written inline SVG using the CSS variables (so dark mode works). `svg text` is styled globally.
  Use `className="axis"` / `"gridline"` for lines. Must work at 360px width: use `viewBox` + `width: 100%`.
- Accessibility: real `<button>`/`<input>` elements with labels, `aria-pressed`/`aria-selected` on toggles,
  `aria-label` summarising each SVG, `aria-live="polite"` on result readouts, keyboard operable, no colour-only meaning.
  Animation only when it explains something; keep it optional (a "Step"/"Run" button), never auto-playing loops.
- Performance: nothing heavy on render. Training loops run in small chunks via `requestAnimationFrame`/`setTimeout`
  and must stop on unmount.

## Shared files: do not edit

`src/components/*`, `src/styles/global.css`, `src/data/*`, `src/lib/math.ts`, `src/lib/rng.ts`, `src/lib/progress.ts`,
`src/lib/exercise.ts`, `src/App.tsx`, `src/pages/*`, and any lesson/interactive/lib file that is not yours.
Other authors are working in parallel on other lessons. If you need something shared that does not exist, build it
locally in your own files and mention it in your final report.

## Definition of done

1. `npx tsc --noEmit` shows no errors in your files (ignore errors in other authors' in-progress files).
2. `npx vitest run src/lib/<your tests> src/lessons/lessons.test.tsx` passes for your files.
3. You re-read each lesson as "a Java developer who has never studied Transformers" and fixed every unexplained term,
   conceptual jump, magic code, and wrong number.
4. Final report: files created, anything you could not do, and any factual claims you were unsure about.

## Concept diagrams (`src/illustrations/*.tsx`)

A row or stack of identical labelled boxes is not a diagram: it is a list with borders. Use `<Flow>` only for a short
ordered list (it renders as a slim stepper). For a concept, draw a picture that **shows the data**, following the
exemplar `src/illustrations/LoopDiagram.tsx`:

- Draw the actual thing at each stage (the text, the token chips, the integers, a small grid for a vector or matrix,
  bars for probabilities, a curve for a loss), not a box with its name in it. If the idea is a loop, draw the loop.
- Quiet style: hand-written inline SVG, theme variables only (`--ink`, `--ink-3`, `--rule-strong`, `--paper-2`,
  `--accent`, `--on-accent`, and `--q/--k/--v` strictly for query/key/value, `--good/--bad` for right/wrong).
  The accent marks the ONE thing the picture is about. No cartoon characters, gradients, shadows or decoration.
- Numbered captions (accent numeral + bold title + one or two short grey lines) only when the content is a sequence.
- `viewBox` + `width: 100%` with a `minWidth` inside `<div className="table-scroll">`, wrapped in `<figure>`.
  Text never smaller than 10.5px in viewBox units at 760 wide; nothing may overlap or be clipped: check a screenshot.
- Accessible: `role="img"` with a `<title>` describing the picture, plus an `sr-only` list when it carries a sequence.
- Marker ids must be unique per component (several diagrams can share a page).

## Coding exercises that run in the browser

Lessons can carry `<CodeExercise id="<lesson-id>-code-<slug>" />` (from `components/python.tsx`). The learner writes
Python in the page, presses Run, and hidden tests check the result. Pyodide runs in a Web Worker, so an infinite
loop is stopped after 20 seconds without freezing the page.

Definitions live in `src/data/codeExercises/*.ts` as plain data (`CodeExerciseDef`): prompt, starter, solution,
tests, 2 to 3 hints, explanation, and the repo `source` file the function comes from. Rules:

- The starter must NOT pass the tests; the solution must pass all of them. `codeExercises.test.ts` runs both
  through real Python (numpy available) and fails the build otherwise, so a wrong expected value cannot ship.
- Write tests as `assert ... , f"got {value}"`: the message is shown to the learner when it fails.
- Aim for 3 to 5 tests: the worked example from the lesson, an edge case, and one that captures the *point*
  (for example: cosine ranking ignores length; merging `[1,1,1]` does not merge twice).
- Only NumPy and the standard library. Anything needing torch stays a terminal exercise.
- After adding one, open `#/selftest` in the browser and confirm every row is `pass` / `ok`.

## Other shared machinery

- **Error boundaries** wrap every `<Lab>`, every lesson body and the page, so one crash cannot blank the app.
- **Review queue** (`lib/review.ts`): every `<Quiz>` answer is recorded; wrong answers come back at growing
  intervals on `#/review`. Quiz questions whose parts are plain strings register themselves automatically.
- **Diagnostics** (`components/Diagnostic.tsx`, data in `data/diagnostics.ts`): a few questions at the top of a
  part's first lesson that tell an experienced learner what to skip. Each question names the lesson that teaches it.
- **Progress export and import** lives on the home page; storage keys are `llm-fp-*`.
