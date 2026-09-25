import { CodeExercise } from '../components/python'
import { Lesson, Why, Problem, MentalModel, TryIt, Numbers, TheMath, CodeIt, BreakIt, Exercises, CheckYourself, Remember, RealLLM, BeforeMovingOn } from '../components/lesson'
import { Callout, DeepDive, Equation, G, Term, ToyVsReal, WhyExists } from '../components/ui'
import { Code } from '../components/Code'
import { Exercise, ExplainBack, OrderExercise } from '../components/exercise'
import { NudgePlayground } from '../interactive/NudgePlayground'
import { ChainRulePipeline } from '../interactive/ChainRulePipeline'

// picture for the "predict the sign" exercise: a loss curve with three marked points
const CURVE_W = 360
const CURVE_H = 170
const lossAt = (w: number) => 0.55 * (w - 1) ** 2 + 0.4
const px = (w: number) => 30 + ((w + 2) / 6) * (CURVE_W - 50)
const py = (l: number) => CURVE_H - 28 - (l / 5.6) * (CURVE_H - 44)
const CURVE_POINTS = Array.from({ length: 61 }, (_, i) => -2 + i * 0.1).map((w) => `${px(w).toFixed(1)},${py(lossAt(w)).toFixed(1)}`).join(' ')
const MARKS: [string, number][] = [['A', -1], ['B', 1], ['C', 3]]

function LossCurvePicture() {
  return (
    <svg viewBox={`0 0 ${CURVE_W} ${CURVE_H}`} style={{ width: '100%', maxWidth: 460, display: 'block', margin: '8px auto' }} role="img" aria-label="A U-shaped curve of loss against one weight w. Point A is on the left side where the curve runs downhill to the right. Point B is at the bottom of the U. Point C is on the right side where the curve runs uphill to the right.">
      <line className="axis" x1={30} x2={CURVE_W - 10} y1={CURVE_H - 28} y2={CURVE_H - 28} />
      <line className="axis" x1={30} x2={30} y1={10} y2={CURVE_H - 28} />
      <text x={CURVE_W - 12} y={CURVE_H - 10} fontSize={11.5} textAnchor="end" style={{ fill: 'var(--ink-2)' }}>weight w →</text>
      <text x={36} y={18} fontSize={11.5} style={{ fill: 'var(--ink-2)' }}>loss</text>
      <polyline points={CURVE_POINTS} fill="none" stroke="var(--ink-2)" strokeWidth={2.5} />
      {MARKS.map(([name, w]) => (
        <g key={name}>
          <circle cx={px(w)} cy={py(lossAt(w))} r={6} fill="var(--accent)" />
          <text x={px(w)} y={py(lossAt(w)) - 12} fontSize={14} fontWeight={700} textAnchor="middle">{name}</text>
        </g>
      ))}
    </svg>
  )
}

export default function DerivativesLesson() {
  return (
    <Lesson id="derivatives">
      <Why>
        <p className="lede">Sunday evening. Riya’s phone is propped against the spice rack, Amma is on video call from Mysuru, and the pressure cooker is hissing far too hard.</p>
        <p>“Turn the flame down,” Amma says. “Not all the way. A little. Now listen.”</p>
        <p>Riya turns the knob a few degrees. The hiss drops a lot. She turns it a few degrees more. The hiss drops again, but less this time.</p>
        <p>“See? Nobody needs a formula for a cooker,” Amma says. “You nudge the knob, you listen, you nudge again.”</p>
        <p>Riya laughs, and then stops laughing. On Friday Kabir wrote one question on the whiteboard: <em>how does a model find its numbers?</em> A model is a function with billions of adjustable numbers, like billions of knobs. Next to them sits one meter: the <G t="loss">loss</G>, the “surprise” number from the <a href="#/lesson/softmax">last lesson</a>. Lower is better.</p>
        <p>For every knob you would want to know two things. <b>If I turn you up a hair, does the loss go up or down? And how strongly?</b></p>
        <Callout kind="idea">
          That question has a name: the <b>derivative</b>. It is the only calculus an LLM needs. You can measure it with three lines of code and no formulas at all.
          <br /><br />
          Training a model is then three steps on repeat. Measure this for every knob. Turn every knob a little in its helpful direction. Go again.
        </Callout>
        <p className="muted">If school calculus left scars: we will not use limits, integrals or trigonometry. We will nudge things and watch what happens.</p>
      </Why>

      <Problem>
        <p>Dev wanders into the kitchen, drawn by the smell. Riya explains the knobs. “Computers are fast, na,” he says. “Just try random settings and keep whichever one is best.”</p>
        <p>It is a fair first idea. Here is why it falls apart.</p>
        <WhyExists
          problem="A billion knobs, one loss meter. Find settings that make the loss small."
          naive="Try random changes to the knobs. Keep a change if the loss went down, undo it if not."
          fails="With a billion knobs, a random change is almost never an improvement, and each try costs a full run of the model. You would wait forever."
          idea="Do not guess. For each knob, measure how sensitive the loss is to it: which direction helps, and how much. Then move all knobs at once in their helpful directions."
          tradeoff="A sensitivity is only valid for tiny changes around where you stand. So you must take small steps and re-measure, many times. That loop is the next lesson, gradient descent."
        />
      </Problem>

      <MentalModel>
        <p>Back to the cooker. Say you turn the flame knob up by 1 degree and the hiss gets 2 units louder. The sensitivity of the hiss to the knob is 2. That one number tells you two things: the <b>direction</b> (louder, not quieter) and the <b>strength</b> (twice your nudge).</p>
        <Callout kind="analogy">
          A derivative is that cooker knob, measured with numbers instead of ears: nudge, listen, compare.
          <br /><br />
          Where the analogy stops: Riya noticed that the second nudge did less than the first. That is the normal case. A function’s sensitivity depends on where you currently stand, so it must be measured again after every move. And a real derivative is a number you compute, not a sound you judge.
        </Callout>
        <Term
          name="Derivative"
          plain={<>The answer to “if I nudge the input a tiny bit, how much does the output move, per unit of nudge?”. An amplification factor with a sign.</>}
          example={<>For f(x) = x² at x = 3: nudge x by 0.001 and the output moves by about 0.006. The derivative there is 6.</>}
          formal={<>The value that the ratio (f(x + h) − f(x)) / h settles on as h shrinks towards 0. Written df/dx or f′(x). Geometrically: the slope of the curve at x.</>}
        />
        <p>The recipe is an experiment, not a formula: <b>nudge, re-measure, divide.</b></p>

        <h3>Many knobs: the gradient</h3>
        <p>A model has many inputs we can adjust. Nothing new is needed: do the nudge experiment for <em>one knob at a time</em>, holding the others still.</p>
        <Term
          name="Gradient"
          plain={<>One derivative per knob, collected in a list. For each knob: “if I turn you up a hair, does the error go up or down, and how fast?”</>}
          example={<>f(a, b) = a × b at a = 3, b = 4. Nudge a: the output moves 4 times as much. Nudge b: 3 times as much. Gradient = [4, 3].</>}
          formal={<>The vector of partial derivatives ∇f = [∂f/∂a, ∂f/∂b, …]. The curly ∂ means “nudging this one input, holding the others fixed”.</>}
        />

        <h3>Nudges through a pipeline: the chain rule</h3>
        <p>A neural network is a pipeline: layer feeds layer feeds layer. So how does a nudge at the start arrive at the end?</p>
        <p>Kabir’s whiteboard answer, the next morning, is two arrows and one word: <em>multiply</em>.</p>
        <Callout kind="analogy">
          Change money twice. Dollars to euros at 0.9, then euros to yen at 160. One extra dollar in gives 0.9 × 160 = 144 extra yen out. <b>The rates multiply.</b>
          <br /><br />
          Where the analogy stops: exchange rates are the same for your first dollar and your millionth. A stage like “square it” has a rate that depends on the value flowing through it, so you have to look up each stage’s rate at its current input.
        </Callout>
        <p>That is the <G t="chain-rule">chain rule</G>: every stage has a local amplification, and the end-to-end amplification is their product. You will try it below.</p>
      </MentalModel>

      <TryIt title="Nudge it and see">
        <NudgePlayground />
        <p>What you should have seen: with a big nudge the ratio is off, because the curve bends between the two points. As h shrinks, the solid line swings onto the dashed tangent and the ratio settles. For x², it settles on 2x, whatever x you choose. You just discovered a calculus rule by experiment.</p>

        <h3>Now a pipeline of three stages</h3>
        <p>Take x, square it, add 1, multiply by 3. Each stage only knows its own amplification.</p>
        <ChainRulePipeline />
        <p>The “add 1” stage always has amplification 1: adding a constant shifts things but never stretches a nudge. The “times 3” stage is always 3. Only the squaring stage depends on the value flowing through it.</p>
      </TryIt>

      <Numbers>
        <p>The nudge experiment on f(x) = x² at x = 3, with h = 0.001:</p>
        <Code lang="text">{`
f(3.000) = 9.000000
f(3.001) = 9.006001

output moved by   0.006001
input moved by    0.001
ratio             0.006001 / 0.001 = 6.001
`}</Code>
        <p>And with smaller and smaller nudges:</p>
        <div className="table-scroll">
          <table className="plain mono" style={{ fontSize: 14 }}>
            <thead><tr><th>h</th><th>1</th><th>0.1</th><th>0.01</th><th>0.001</th></tr></thead>
            <tbody><tr><td>ratio</td><td>7</td><td>6.1</td><td>6.01</td><td>6.001</td></tr></tbody>
          </table>
        </div>
        <p>It is heading for 6. At x = 3, this function amplifies tiny nudges 6 times.</p>

        <h3>The pipeline at x = 2</h3>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>stage</th><th>value in → out</th><th>local amplification</th><th>a nudge of +0.001 has become</th></tr></thead>
            <tbody>
              <tr><td>square</td><td className="mono">2 → 4</td><td className="mono">2 × 2 = 4</td><td className="mono">≈ 0.004</td></tr>
              <tr><td>add 1</td><td className="mono">4 → 5</td><td className="mono">1</td><td className="mono">≈ 0.004</td></tr>
              <tr><td>times 3</td><td className="mono">5 → 15</td><td className="mono">3</td><td className="mono">≈ 0.012</td></tr>
            </tbody>
          </table>
        </div>
        <p>Chain rule: <span className="mono">4 × 1 × 3 = <b>12</b></span>.</p>
        <p>The honest check, treating the whole pipeline as a black box: <span className="mono">f(2) = 15</span>, <span className="mono">f(2.001) = 15.012003</span>. Moved by 0.012003, divided by 0.001: <span className="mono">12.003</span>. They agree.</p>
        <Callout kind="idea">
          This is the entire mathematical content of <G t="backprop">backpropagation</G>, the algorithm that trains every neural network.
          <br /><br />
          A 50-layer network is a 50-stage pipeline. Backpropagation walks it from the output back to the input, multiplying local amplifications as it goes. One more rule joins in when a value feeds several later stages: the effects along the separate routes add up.
          <br /><br />
          You will build it in <a href="#/lesson/backprop">lesson 3.2</a>. The “hard part” will be a multiplication you already understand.
        </Callout>
      </Numbers>

      <TheMath>
        <Equation
          label="the derivative of f at x is approximately f of x plus h, minus f of x, all divided by h, for a tiny h"
          symbols={[
            ['f', 'any function: a formula, a pipeline, a whole neural network with its loss'],
            ['x', 'the input (or knob) we are asking about'],
            ['h', 'the nudge: a small number such as 0.000001'],
            ['df/dx', <>“the derivative of f with respect to x”. Read the d as “a tiny change in”: tiny change in f per tiny change in x</>],
          ]}
        >
          df/dx ≈ ( f(x + h) − f(x) ) / h
        </Equation>
        <Equation
          label="chain rule: dy by dx equals dy by dv times dv by du times du by dx"
          symbols={[
            ['x → u → v → y', 'a pipeline: x goes into stage 1 giving u, u into stage 2 giving v, v into stage 3 giving y'],
            ['du/dx', 'local amplification of stage 1, measured at its current input'],
            ['dy/dx', 'end-to-end amplification: what a nudge to x does to y'],
          ]}
        >
          dy/dx = dy/dv × dv/du × du/dx
        </Equation>
        <p>Formulas for derivatives are shortcuts that save you the experiment. These five are the only ones this course ever uses, and the repository verifies each one by nudging:</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>stage</th><th>its amplification</th><th>in words</th></tr></thead>
            <tbody>
              <tr><td className="mono">c · x</td><td className="mono">c</td><td>a wire with gain c</td></tr>
              <tr><td className="mono">x²</td><td className="mono">2x</td><td>grows as x grows; you measured it</td></tr>
              <tr><td className="mono">x + c</td><td className="mono">1</td><td>adding passes nudges through unchanged</td></tr>
              <tr><td className="mono">e<sup>x</sup></td><td className="mono">e<sup>x</sup></td><td>its slope equals its value (this is why <a href="#/lesson/softmax">softmax</a> uses e)</td></tr>
              <tr><td className="mono">ln(x)</td><td className="mono">1 / x</td><td>the loss −ln(p) gets very steep as p approaches 0</td></tr>
            </tbody>
          </table>
        </div>
        <DeepDive title="What about max(0, x)? It has a corner">
          <p>max(0, x) is called <G t="relu">ReLU</G>, and it is inside many neural networks (most LLMs use a smoothed relative of it).</p>
          <p>Left of 0 it is flat: amplification 0, so nudges die. Right of 0 it passes x through unchanged: amplification 1, so nudges pass through.</p>
          <p>Exactly at 0 there is a corner. The nudge experiment gives 1 to the right and 0 to the left, so strictly there is no derivative there. Libraries pick one of the two values and move on (PyTorch uses 0). Landing on exactly 0.0 is rare, and either choice works in practice.</p>
        </DeepDive>
        <DeepDive title="Why not make h as small as possible?">
          <p>Floating-point numbers have about 16 significant digits. If h is so small that x + h rounds to x, or f(x + h) − f(x) loses all its digits, the ratio turns to noise.</p>
          <p>For x² at x = 3 in Python: h = 1e-6 gives 6.000001, h = 1e-12 gives 6.0005, h = 1e-15 gives roughly 5.33, and h = 1e-17 gives exactly 0.</p>
          <p>Around 1e-6 is a good compromise, and it is the default in the repository. This is one reason real training does not measure gradients by nudging. Backpropagation computes them from the formulas, with no h to choose, exact up to ordinary rounding.</p>
        </DeepDive>
      </TheMath>

      <CodeIt>
        <p>After dinner Riya opens the repository to see how the nudge is written in code. It is the whole idea in two lines. Read the comment: it is the recipe.</p>
        <Code source="phase1-foundations/math_primer.py" title="the nudge experiment">{`
def nudge_derivative(f, x, h=1e-6):
    return (f(x + h) - f(x)) / h      # nudge, re-measure, divide

d = nudge_derivative(lambda x: x ** 2, 3.0)
assert abs(d - 6) < 1e-3
`}</Code>
        <p><code>f</code> is any Python function. It is treated as a black box: we never look inside, we only call it twice. The file then checks the whole formula table the same way:</p>
        <Code source="phase1-foundations/math_primer.py" title="formulas are shortcuts; the experiment is the meaning">{`
table = [
    ("c*x (c=5) ", lambda x: 5 * x,  lambda x: 5.0),
    ("x^2       ", lambda x: x ** 2, lambda x: 2 * x),
    ("e^x       ", np.exp,           np.exp),
    ("ln(x)     ", np.log,           lambda x: 1 / x),
]
x0 = 2.0
for name, f, formula in table:
    measured, expected = nudge_derivative(f, x0), formula(x0)
    assert abs(measured - expected) < 1e-3
`}</Code>
        <p>And the chain rule, on a two-stage pipeline (square, then times 5):</p>
        <Code source="phase1-foundations/math_primer.py" title="amplification factors multiply">{`
stage1_amp = 2 * 2.0            # d(x^2)/dx = 2x = 4 at x=2
stage2_amp = 5.0                # d(5u)/du = 5
end_to_end = stage1_amp * stage2_amp
measured = nudge_derivative(lambda x: 5 * x ** 2, 2.0)
assert abs(measured - end_to_end) < 1e-3
`}</Code>
        <Callout kind="dev">
          The nudge experiment is black-box testing: call the function twice, compare outputs. The chain rule is white-box: look at the stages and multiply. Later you will use the first to unit-test the second. It is called a <b>gradient check</b>, and it is how you know your backpropagation code is right.
        </Callout>
      </CodeIt>

      <BreakIt>
        <ul>
          <li><b>Find a zero.</b> In the first playground choose x² and move x until the ratio is (almost) 0. Where are you on the curve? (At the bottom, x = 0. A derivative of 0 means “nudging does not help in either direction”: you may be at a minimum. This is what training is searching for.)</li>
          <li><b>Stand on the corner.</b> Choose max(0, x) and set x = 0. Then x = −1. For negative x the ratio is 0 for every h: nudges have no effect at all. A knob behind a “dead” stage like this gets no signal to learn from.</li>
          <li><b>Pick 3x + 1</b> and change h. Why does the ratio not care about h here? (A straight line has the same slope everywhere, so big and small nudges agree.)</li>
          <li><b>Kill the pipeline.</b> In the second playground set x = 0. The product is 0 × 1 × 3 = 0. One stage with amplification 0 silences the whole chain, however large the other factors are.</li>
          <li><b>Break the code.</b> In <code>math_primer.py</code> change <code>h=1e-6</code> to <code>h=1e-17</code> and run it. Which assert fails, and why? (See the second deep dive above.)</li>
        </ul>
      </BreakIt>

      <Exercises>
        <CodeExercise id="derivatives-code-chain-rule" />
        <Exercise
          id="derivatives-cube"
          type="experiment"
          title="Measure a derivative you were never taught"
          answer={{ value: 12, tolerance: 0.05 }}
          answerLabel="derivative of x³ at x = 2"
          hints={['In the playground choose x³, set x = 2.0, and shrink h. Or in Python: nudge_derivative(lambda x: x ** 3, 2.0).', 'The ratios go 19, 12.61, 12.06, 12.006 … What whole number are they heading for?', 'Now guess the rule. x² gave 2x. At x = 2, which expression built from 3 and x² gives your number?']}
          solution={<><p>The ratio settles on <b>12</b>.</p><p>The pattern: x² → 2x, and x³ → 3x². At x = 2 that is 3 × 4 = 12. You found the rule by experiment, which is more than most calculus students ever do. And if you ever forget a rule, you now know how to get the number anyway.</p></>}
        >
          <p>Using the nudge experiment (playground or Python), measure the derivative of <code>f(x) = x³</code> at <code>x = 2</code>. Give the whole number the ratio settles on.</p>
        </Exercise>

        <Exercise
          id="derivatives-chain"
          type="calculate"
          title="Chain rule on a new pipeline"
          answer={{ value: 18, tolerance: 0.05 }}
          answerLabel="end-to-end amplification"
          hints={['Run the values forward first: 1 → 3 → 9 → 5. Each stage’s amplification is measured at its own input.', 'Stage 1 (times 3): always 3. Stage 2 (square): 2 × its input, and its input is 3, not 1. Stage 3 (subtract 4): always 1.', 'Multiply: 3 × 6 × 1.']}
          solution={<><p>Forward: 1 → 3 → 9 → 5. Amplifications: 3, then 2 × 3 = 6, then 1. Product: <b>18</b>.</p><p>The classic mistake is to use 2 × 1 = 2 for the squaring stage. The square does not see x, it sees what the previous stage handed it. This is why backpropagation needs the forward pass first: the local amplifications depend on the values that flowed through.</p><p>Check by nudging: f(x) = (3x)² − 4 gives f(1) = 5 and f(1.001) = 5.018009, a ratio of 18.009.</p></>}
        >
          <p>Pipeline: <code>x → times 3 → square → subtract 4</code>. At <code>x = 1</code>, what is the end-to-end amplification?</p>
        </Exercise>

        <Exercise
          id="derivatives-sign"
          type="predict"
          title="Read the sign off a picture"
          answer={{ text: ['negative', '-', 'minus', 'neg', 'below zero', 'less than zero', '<0'] }}
          answerLabel="sign of the derivative at A"
          hints={['Stand at A and nudge w a little to the right. Does the loss go up or down?', 'The curve runs downhill to the right at A. Output goes down when input goes up.', 'A ratio of (negative change) ÷ (positive nudge) has which sign?']}
          solution={<><p>At <b>A</b> the derivative is <b>negative</b>: increasing w decreases the loss. At <b>B</b> it is 0: the bottom. At <b>C</b> it is positive: increasing w makes things worse.</p><p>Now the punchline of the next lesson. To reduce the loss, move w in the direction <em>opposite</em> to the sign of the derivative. At A (negative) increase w. At C (positive) decrease w. Both moves roll towards B. That rule, applied to every knob at once, is gradient descent.</p></>}
        >
          <p>The curve shows the loss as one weight w changes. Is the derivative of the loss with respect to w at point <b>A</b> positive, zero or negative? Then decide the same for B and C.</p>
          <LossCurvePicture />
        </Exercise>

        <Exercise
          id="derivatives-debug"
          type="debug"
          title="The estimate that does not match"
          hints={['The measured value is 12. Which combination of 4, 1 and 3 gives 12?', 'Think of the currency exchange: do you add exchange rates or multiply them?']}
          solution={<><p>The amplifications must be <b>multiplied</b>, not added: 4 × 1 × 3 = 12. A nudge is stretched by stage 1, and the <em>stretched</em> nudge is what stage 2 receives, and so on.</p><p>Notice that the nudge check caught the bug without anyone reasoning about calculus. Keep that habit: whenever you derive a gradient by hand, measure it too.</p></>}
        >
          <p>A colleague at Paisa Pal analyses the pipeline <code>x → square → +1 → ×3</code> at x = 2. His code says 8. The nudge experiment prints 12.0. What is wrong?</p>
          <Code>{`
amps = [4.0, 1.0, 3.0]        # local amplification of each stage at x = 2
end_to_end = sum(amps)        # 8.0
measured = nudge_derivative(lambda x: 3 * (x ** 2 + 1), 2.0)   # 12.0
`}</Code>
        </Exercise>

        <ExplainBack
          id="derivatives-explain"
          prompt="Explain to a teammate who has never studied calculus what a gradient is and how a model could use it to improve. Do not use the words “derivative” or “slope”."
          modelAnswer={<p>Imagine the model as a machine with many knobs and one meter that shows how wrong it currently is. For each knob you can run a small experiment: turn it up a hair, see how much the meter moves, and divide the movement by the size of your turn. That gives one number per knob. Its sign says whether turning the knob up makes things better or worse, and its size says how much that knob matters right now. The list of all those numbers is the gradient. To improve, turn every knob a little in the direction that lowers the meter, more for knobs that matter more, and then measure again, because the numbers change once you have moved.</p>}
        />
      </Exercises>

      <CheckYourself
        questions={[
          {
            q: 'The derivative of the loss with respect to some weight is −2. What does that tell you?',
            options: ['The loss is currently −2', 'Increasing that weight slightly will decrease the loss, by about twice the size of the nudge', 'The weight should be set to −2', 'Decreasing that weight will decrease the loss'],
            answer: 1,
            explain: 'Sign: input up, output down. Size: output moves 2 units per unit of input, for tiny nudges. It says nothing about the current value of the loss or the weight.',
          },
          {
            q: 'With a nudge of h = 1, the measured ratio for x² at x = 3 was 7, not 6. Why?',
            options: ['The formula 2x is only an approximation, and 7 is the more exact value', 'Over a big nudge the curve bends; the derivative is what the ratio settles on as h shrinks', 'Floating-point rounding spoiled the subtraction f(x + h) − f(x)', 'The nudge experiment only works when x is not a whole number'],
            answer: 1,
            explain: 'The ratio is the slope of the line through two points on the curve. Only when the points are very close does it match the curve’s steepness at x.',
          },
          {
            q: 'A pipeline has three stages with local amplifications 2, 0 and 5 at the current input. What is the end-to-end amplification?',
            options: ['7', '10', '0', '5'],
            answer: 2,
            explain: 'They multiply: 2 × 0 × 5 = 0. A stage that passes no nudges through blocks the whole chain, which matters a lot when networks get deep.',
          },
          {
            q: 'What is a gradient?',
            options: ['The largest derivative among all knobs', 'A list with one derivative per knob: how the output responds to nudging each knob on its own', 'The total of all the knobs', 'A special kind of matrix multiplication'],
            answer: 1,
            explain: 'Same nudge experiment, once per knob, others held still. A model with a billion parameters has a gradient with a billion entries.',
          },
          {
            q: 'What is the mathematical content of backpropagation?',
            options: ['Solving a large system of equations exactly', 'The chain rule: multiplying local amplifications, from the output back to the input', 'Trying random weight changes and keeping the good ones', 'Integrating the loss over all inputs'],
            answer: 1,
            explain: 'A network is a long pipeline. Backpropagation is an efficient way of organising the chain-rule multiplications so that one backward sweep yields the derivative for every weight.',
          },
        ]}
      />

      <Remember
        items={[
          <>A <b>derivative</b> is a measured sensitivity: <b>nudge the input, see how far the output moves, divide</b>. Its sign gives the direction, its size the strength.</>,
          <>It is <b>local</b>: valid for tiny nudges around where you stand. Move, and you must measure again.</>,
          <>A <b>gradient</b> is one derivative per knob. For each knob: up a hair, does the error rise or fall, and how fast?</>,
          <>The <b>chain rule</b>: in a pipeline, local amplifications <b>multiply</b>. At x = 2, square → +1 → ×3 gives 4 × 1 × 3 = 12.</>,
          <>That multiplication is the whole mathematical content of <b>backpropagation</b>. The nudge experiment remains your unit test for it.</>,
        ]}
      />

      <RealLLM>
        <ToyVsReal
          toy={<ul><li>One input x, three stages</li><li>Derivative measured by nudging: two function calls</li><li>Amplifications are single numbers</li></ul>}
          real={<ul><li>Billions of parameters, hundreds of stages (every matrix multiply, softmax and activation is a stage)</li><li>Gradient computed by backpropagation: one forward pass plus one backward pass, for all parameters at once</li><li>Amplifications are matrices (the proper name is Jacobian), and “multiply” is a matrix multiply (this is where the transpose from <a href="#/lesson/matrices">lesson 1.2</a> shows up)</li></ul>}
        />
        <Callout kind="established">
          Nobody trains a model by nudging. Measuring one parameter’s derivative that way costs one extra run of the whole model, so a 7-billion-parameter model would need 7 billion runs for a single training step. The chain rule lets backpropagation get every derivative from <em>one</em> backward sweep. That sweep costs roughly twice as much arithmetic as one forward run, so a full training step (forward plus backward) is about three forward runs. That efficiency is what makes training large models possible at all.
        </Callout>
        <Callout kind="established">
          In frameworks like PyTorch you never write the backward sweep yourself. Every operation records what it needs to work out its local amplification as it runs, and <code>loss.backward()</code> multiplies them together in reverse. This is called automatic differentiation. In <a href="#/lesson/backprop">Backpropagation</a> you will write it by hand once, in NumPy, and check it with the nudge experiment, so that <code>.backward()</code> is never magic again.
        </Callout>
        <p>You now have all four tools:</p>
        <ul>
          <li><a href="#/lesson/vectors">dot products</a> to compare,</li>
          <li><a href="#/lesson/matrices">matrix multiplies</a> to do it in bulk,</li>
          <li><a href="#/lesson/softmax">softmax</a> to turn scores into probabilities and a loss,</li>
          <li>and derivatives to find out which way to turn the knobs.</li>
        </ul>
        <p>Next, in <a href="#/lesson/gradient-descent">Part 2</a>, we put the last two together and watch a program learn. Riya ends the call with Amma’s parting advice, which turns out to be the next lesson in one line: “Small turns. Then listen again.”</p>
      </RealLLM>

      <BeforeMovingOn
        id="part-1"
        intro="That was all the maths. Before you use it, a checkpoint that reaches back to Part 0 and across all four ideas of this part. Answer from memory."
        questions={[
          {
            q: 'You type “What is a cat?” and the model answers correctly. Where did the answer come from?',
            options: ['The model looked the question up in a database of stored answers', 'The model searched the web for the most similar question', 'A fixed function with billions of learned numbers computed, token by token, probabilities for what comes next', 'A set of grammar rules written by engineers'],
            answer: 2,
            explain: 'An LLM is a function with learned numbers, not a database. Nothing is looked up. The same arithmetic runs for every prompt, and the learned numbers make good continuations probable.',
          },
          {
            q: 'How does a model produce a 200-word answer?',
            options: ['It computes all 200 words in one pass', 'It predicts one token, appends it to the text, and runs again on the longer text, until it decides to stop', 'It retrieves the closest stored paragraph and edits it', 'It writes the last word first and works backwards'],
            answer: 1,
            explain: 'Generation is a loop around a next-token predictor. Every new token is fed back in as part of the input for the next one.',
          },
          {
            q: 'The dot product between the vector for “bank” and the vector for “river” is large and positive. What does that mean?',
            options: ['The two vectors are identical', 'The two vectors point in broadly the same direction: they “agree”', 'One of the words is more frequent', 'The vectors are perpendicular'],
            answer: 1,
            explain: 'Large positive: agree. Zero: unrelated. Negative: oppose. It also grows with the vectors’ lengths, which is why cosine similarity divides the lengths out.',
          },
          {
            q: 'A prompt of 6 tokens, each a vector of 16 numbers, is multiplied by a weight matrix of shape (16, 1000). What comes out?',
            options: ['Shape (6, 1000): one row of 1,000 scores for each of the 6 positions', 'Shape (16, 16)', 'A single list of 1,000 numbers', 'An error, because 6 and 1000 do not match'],
            answer: 0,
            explain: '(6, 16) @ (16, 1000) → (6, 1000). Inner numbers match, outer numbers are the answer. If the vocabulary has 1,000 tokens, those rows are logits.',
          },
          {
            q: 'Softmax receives the logits [5, 5, 5]. What comes out?',
            options: ['[5, 5, 5], unchanged, because they are already equal', '[1, 0, 0]', '[⅓, ⅓, ⅓], because only differences between logits matter and there are none', 'It overflows'],
            answer: 2,
            explain: 'Equal logits, equal probabilities, whatever the value. The reason is “exponentiate, then normalise”, and the fact that a shared constant cancels. [0, 0, 0] and [−40, −40, −40] give the same thirds.',
          },
          {
            q: 'During training, the correct next token was given a probability of 0.01. What is true about the loss, −ln(0.01) ≈ 4.6?',
            options: ['It is low, because 0.01 is a small number', 'It is high: the model was very surprised by the right answer, so there is a lot to fix', 'It is negative', 'It cannot be computed without the other probabilities'],
            answer: 1,
            explain: 'Cross-entropy is surprise at the right answer. 0.9 gives 0.105, 0.5 gives 0.69, 0.01 gives 4.6.',
          },
          {
            q: 'For one particular weight, the derivative of the loss is +3. To reduce the loss, you should…',
            options: ['increase that weight a little', 'decrease that weight a little', 'set the weight to 3', 'leave it: positive derivatives are good'],
            answer: 1,
            explain: 'Positive derivative: turning the knob up makes the loss go up. So turn it down. Always step against the sign of the derivative.',
          },
        ]}
      >
        <OrderExercise
          id="part-1-pipeline"
          title="Rebuild the pipeline from memory"
          prompt={<p>From <a href="#/lesson/prompt-to-answer">Part 0</a>: put the journey from prompt to next word in order. You now know the maths inside the steps marked in brackets.</p>}
          correct={[
            'Split the prompt into tokens',
            'Turn each token into a vector',
            'Transformer layers mix information [matrix multiplies, dot products]',
            'One raw score per vocabulary token [logits]',
            'Scores become probabilities [softmax]',
            'Sample one token, append it, repeat',
          ]}
          solutionNote={<p>Text becomes numbers, numbers are transformed by matrix multiplies, the final scores become probabilities through softmax, and one token is drawn. Training runs the same pipeline, measures the surprise at the true next token, and uses derivatives to adjust every matrix. That is Part 2.</p>}
        />
      </BeforeMovingOn>
    </Lesson>
  )
}
