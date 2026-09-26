import { Gpt2Explainer } from '../interactive/Gpt2Explainer'

/** #/gpt2: the GPT-2 Explainer on its own page. `#/gpt2/autoload` starts the download at once (used for screenshots). */
export default function Gpt2Page({ arg }: { arg?: string }) {
  return (
    <div className="Gpt2Page">
      <h1 className="lesson-title">GPT-2 Explainer</h1>
      <p className="lesson-question">
        Type a sentence and follow it through the real GPT-2, from text to the next token, with the actual numbers at every step.
      </p>
      <p className="muted" style={{ fontSize: 14.5, marginBottom: 32 }}>
        It reads best after <a href="#/lesson/attention">Attention</a>, <a href="#/lesson/transformer-block">The Transformer block</a> and{' '}
        <a href="#/lesson/build-gpt">Build a GPT</a>, but every stage links back to the lesson that explains it.
      </p>
      <Gpt2Explainer autoload={arg === 'autoload'} framed={false} />
      <p className="muted" style={{ fontSize: 13, marginTop: 28 }}>
        Weights: GPT-2 small by OpenAI, modified MIT license (github.com/openai/gpt-2), quantized to int8 for this course by{' '}
        <code>phase6-engineering/export_gpt2.py</code>. GPT-2 was trained on 2019 web text; its output can be wrong, biased or offensive.
      </p>
    </div>
  )
}
