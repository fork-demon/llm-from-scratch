// Same prompt, three training stages. The outputs are hand-written illustrations of typical
// behaviour, not recordings of any particular model, and the UI says so.
import { useState } from 'react'
import { Lab } from '../components/ui'

type StageId = 'base' | 'sft' | 'pref'

const STAGES: { id: StageId; label: string; data: string; loss: string; changes: string; scale: string }[] = [
  {
    id: 'base',
    label: '1 · Base model',
    data: 'Huge amounts of filtered, deduplicated text and code: web pages, books, papers, repositories. Trillions of tokens for current models.',
    loss: 'Cross-entropy on the next token, at every position of every document. Exactly your tiny GPT loss.',
    changes: 'Every weight, starting from random. Almost everything the model knows about language and the world is learned here.',
    scale: 'By far the largest stage: typically weeks to months on thousands of GPUs.',
  },
  {
    id: 'sft',
    label: '2 · After SFT',
    data: 'Demonstrations: conversations where the assistant turn was written or approved by people. Thousands to a few million examples.',
    loss: 'Still next-token cross-entropy, but usually counted only on the assistant’s tokens.',
    changes: 'Usually all weights again, nudged gently with a small learning rate. Mostly behaviour changes: format, role, tone, when to stop.',
    scale: 'Tiny next to pretraining: a very small fraction of the tokens and of the compute.',
  },
  {
    id: 'pref',
    label: '3 · After preference tuning',
    data: 'Comparisons: for one prompt, two answers and a label saying which is better (from people, or from another model following written guidelines).',
    loss: 'RLHF: maximise a learned reward, minus a penalty for drifting from the SFT model. DPO: a classification-style loss directly on the pairs.',
    changes: 'All weights, nudged again. Shifts which of many plausible answers the model prefers: helpfulness, caution, hedging, refusals.',
    scale: 'Also small next to pretraining, but fiddly: the model is graded on text it generates itself.',
  },
]

const PROMPTS: { prompt: string; out: Record<StageId, string>; note: Record<StageId, string> }[] = [
  {
    prompt: 'What is the capital of France?',
    out: {
      base: 'What is the capital of Spain? What is the capital of Italy? What is the capital of Germany? Answers on page 12.',
      sft: 'The capital of France is Paris.',
      pref: 'The capital of France is Paris.',
    },
    note: {
      base: 'A list of quiz questions is a very plausible continuation of a quiz question. The model is not broken: it continues documents.',
      sft: 'It has seen thousands of examples where a question is followed by an answer, then a stop token.',
      pref: 'Nothing to improve here. Easy factual questions often look the same before and after preference tuning.',
    },
  },
  {
    prompt: 'What is a cat?',
    out: {
      base: 'What is a dog? These are the questions my daughter asks me every morning on the way to school, and last Tuesday I finally',
      sft: 'A cat is a small domesticated carnivorous mammal.',
      pref: 'A cat is a small, furry animal that many people keep as a pet. Cats are carnivores, known for hunting mice, sleeping a lot and purring when content.',
    },
    note: {
      base: 'It guessed “blog post”. Another sample might guess “dictionary entry” or “forum thread”.',
      sft: 'Correct and in the right format, but it copies the style of its demonstrations, here a terse one.',
      pref: 'People comparing answers tended to prefer a bit more helpful detail, so that is what got reinforced.',
    },
  },
  {
    prompt: 'Write a haiku about my debugger.',
    out: {
      base: 'Write a limerick about my compiler. Write a sonnet about my linter. (10 points each)',
      sft: 'Breakpoint in the night / all my values freeze in place / the bug hides elsewhere',
      pref: 'Breakpoint in the night / I step through my own mistakes / the bug was line one',
    },
    note: {
      base: 'Again it continues the document instead of obeying it. An instruction is just text to a base model.',
      sft: 'It follows the instruction. There is no single right haiku, so SFT can only imitate the demonstrations it saw.',
      pref: 'There is no “correct” haiku to put in a loss function, but people can say which of two they like more. This is the case preference tuning exists for.',
    },
  },
  {
    prompt: 'Who won the 2087 chess world championship?',
    out: {
      base: 'The 2087 World Chess Championship was won by Viktor Alanov, who defeated the reigning champion 7.5 to 4.5 in a match held in',
      sft: 'The 2087 World Chess Championship was won by Viktor Alanov.',
      pref: 'I can’t know that: 2087 is in the future relative to my training data. I can tell you about past championships if that helps.',
    },
    note: {
      base: 'A fluent, confident, invented continuation (the name is made up for this illustration). Plausible text is all it optimises.',
      sft: 'Demonstrations teach “questions get confident answers”. If the demonstrations never show declining, SFT can make this worse.',
      pref: 'Raters preferred honest uncertainty over invention, so it became more likely. More likely, not guaranteed: see the previous lesson.',
    },
  },
]

export function StageExplorer() {
  const [p, setP] = useState(0)
  const [s, setS] = useState(0)
  const stage = STAGES[s]
  const item = PROMPTS[p]

  return (
    <Lab
      title="One prompt, three stages of training"
      goal={<>Pick a prompt, then step through the stages. Before you click a stage, <b>predict</b> what the model will write. The architecture never changes between stages: only the weights do.</>}
    >
      <div className="steps" role="group" aria-label="Choose a prompt">
        {PROMPTS.map((x, i) => (
          <button key={x.prompt} className="step-btn" aria-pressed={i === p} onClick={() => setP(i)}>{x.prompt}</button>
        ))}
      </div>
      <div className="steps" role="tablist" aria-label="Training stage">
        {STAGES.map((x, i) => (
          <button key={x.id} role="tab" className="step-btn" aria-selected={i === s} onClick={() => setS(i)}>{x.label}</button>
        ))}
      </div>

      <div role="tabpanel" aria-live="polite">
        <div className="card" style={{ fontFamily: 'var(--mono)', fontSize: 14, lineHeight: 1.6 }}>
          <div className="muted" style={{ fontSize: 12, marginBottom: 4 }}>PROMPT</div>
          <div>{item.prompt}</div>
          <div className="muted" style={{ fontSize: 12, margin: '10px 0 4px' }}>MODEL WRITES (illustration)</div>
          <div><b>{item.out[stage.id]}</b></div>
        </div>
        <p style={{ marginTop: 10 }}>{item.note[stage.id]}</p>

        <div className="table-scroll">
          <table className="plain" style={{ fontSize: 14.5 }}>
            <tbody>
              <tr><th scope="row" style={{ whiteSpace: 'nowrap' }}>Training data</th><td>{stage.data}</td></tr>
              <tr><th scope="row">Loss</th><td>{stage.loss}</td></tr>
              <tr><th scope="row" style={{ whiteSpace: 'nowrap' }}>What changes</th><td>{stage.changes}</td></tr>
              <tr><th scope="row">Rough scale</th><td>{stage.scale}</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <div className="btn-row" style={{ marginTop: 12 }}>
        <button className="btn small" disabled={s === 0} onClick={() => setS(s - 1)}>Previous stage</button>
        <button className="btn small primary" disabled={s === STAGES.length - 1} onClick={() => setS(s + 1)}>Next stage</button>
      </div>
      <p className="lab-note" style={{ marginTop: 12 }}>
        <b>Honest label:</b> these outputs are hand-written to show the <em>typical</em> behaviour of each stage. They are not recordings of a real model, and real models vary from sample to sample. You can reproduce the base-model behaviour yourself with any openly released base (non-instruct) model.
      </p>
    </Lab>
  )
}
