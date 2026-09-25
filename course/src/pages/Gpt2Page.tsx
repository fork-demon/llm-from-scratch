import { Gpt2Explainer } from '../interactive/Gpt2Explainer'

/** #/gpt2: the GPT-2 Explainer on its own page. `#/gpt2/autoload` starts the download at once (used for screenshots). */
export default function Gpt2Page({ arg }: { arg?: string }) {
  return (
    <div>
      <h1 className="lesson-title">GPT-2 Explainer</h1>
      <p className="lesson-question">What does a real language model do with your prompt, stage by stage?</p>
      <p>
        Everything you built in the course, at full size. This is GPT-2 small, the model OpenAI released in 2019, running in your browser.
        Each stage shows the real numbers for your prompt. Each <b>Why?</b> links to the lesson that explains that stage.
      </p>
      <p className="muted" style={{ fontSize: 14 }}>
        New here? The stages make most sense after <a href="#/lesson/attention">Attention</a>, <a href="#/lesson/transformer-block">The Transformer block</a> and{' '}
        <a href="#/lesson/build-gpt">Build a GPT</a>. You can still explore now and follow the links as you go.
      </p>
      <Gpt2Explainer autoload={arg === 'autoload'} />
      <p className="muted" style={{ fontSize: 13.5 }}>
        Weights: GPT-2 small by OpenAI, modified MIT license (github.com/openai/gpt-2), quantized to int8 for this course by{' '}
        <code>phase6-engineering/export_gpt2.py</code>. GPT-2 was trained on 2019 web text; its output can be wrong, biased or offensive.
      </p>
    </div>
  )
}
