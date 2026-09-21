import { useState } from 'react'
import { Lesson, Why, Problem, MentalModel, TryIt, CodeIt, Exercises, CheckYourself, Remember, RealLLM, BeforeMovingOn } from '../components/lesson'
import { Callout, DeepDive, Flow, G, Lab, ToyVsReal, WhyExists } from '../components/ui'
import { Code } from '../components/Code'
import { Exercise, ExplainBack, OrderExercise } from '../components/exercise'
import { LESSONS, PARTS, lessonById } from '../data/curriculum'
import { COURSE_MAP, PIPELINE_ORDER } from '../lib/courseMap'
import { SameShapeBiggerNumbers } from '../illustrations/SameShapeBiggerNumbers'

/** The clickable architecture: every node shows its question, what you build, and real lesson links. */
function CourseArchitecture() {
  const [active, setActive] = useState(0)
  const node = COURSE_MAP[active]
  const minutes = node.lessons.reduce((s, id) => s + (lessonById(id)?.minutes ?? 0), 0)
  return (
    <Lab title="The architecture of the course" goal={<>Click each box, left to right. For every stage, read the <b>question</b> first and try to guess what kind of thing you would need to build to answer it.</>}>
      <Flow horizontal steps={COURSE_MAP.map((n) => ({ label: n.label }))} active={active} onSelect={setActive} />
      <div className="card" aria-live="polite">
        <h4 style={{ fontSize: 18, marginBottom: 8 }}>{active + 1}. {node.label}</h4>
        <p><b>The question it answers:</b> {node.question}</p>
        <p><b>What you will build:</b> {node.build}</p>
        <p style={{ marginBottom: 4 }}><b>Lessons</b> <span className="muted">(about {minutes} minutes)</span><b>:</b></p>
        <ul style={{ marginTop: 0 }}>
          {node.lessons.map((id) => {
            const l = lessonById(id)
            return l ? <li key={id}><a href={`#/lesson/${id}`}>{l.code} {l.title}</a> <span className="muted">Part {l.part.number}: {l.part.title}</span></li> : null
          })}
        </ul>
      </div>
      <div className="btn-row">
        <button className="btn small" disabled={active === 0} onClick={() => setActive(active - 1)}>← Previous</button>
        <button className="btn small primary" disabled={active === COURSE_MAP.length - 1} onClick={() => setActive(active + 1)}>Next</button>
      </div>
      <p className="lab-note" style={{ marginTop: 10 }}>The course order is not exactly left to right: the small amount of maths comes first (Part 1), so the lessons listed under “Representations” and “Neural network” are interleaved. The sidebar always shows the real order.</p>
    </Lab>
  )
}

const RHYTHM: [string, string][] = [
  ['Why are we learning this?', 'A concrete question, never a definition.'],
  ['The problem', 'What the previous idea could not do. Every concept exists because something simpler failed.'],
  ['A mental model', 'The intuition, with analogies clearly labelled as analogies.'],
  ['Try it yourself', 'An interactive where you change something and watch real numbers move.'],
  ['Let’s see the numbers', 'A tiny calculation done by hand, so nothing is hidden.'],
  ['The math', 'Only now the formula, with every symbol explained.'],
  ['Let’s code it', 'A few lines at a time, ending at the real function from the repository.'],
  ['Break it', 'Things to change or sabotage. Predict first, then check.'],
  ['Exercises', 'Predict, calculate, debug, implement. Hints before solutions.'],
  ['Can you explain this?', 'A short quiz that tests understanding, not vocabulary.'],
  ['What you should remember', 'Three to five ideas.'],
  ['Where this appears in a real LLM', 'Toy versus production, and what is still debated.'],
]

export default function CourseMapLesson() {
  const totalHours = Math.round(LESSONS.reduce((s, l) => s + l.minutes, 0) / 60)
  return (
    <Lesson id="course-map">
      <Why title="Where is all of this going?">
        <p className="lede">You have seen the machine from the outside: a loop that picks one token at a time, around a function with billions of learned numbers.</p>
        <p>The rest of the course opens that machine, one part at a time, until nothing in it is a black box. Then it builds the things people put <em>around</em> the machine: retrieval, fine-tuning and agents.</p>
        <p>That is {LESSONS.length} lessons in {PARTS.length} parts, roughly {totalHours} hours including a hands-on capstone.</p>
        <p>This lesson is the map. It takes a few minutes, and it will save you from the most common way people get lost in this subject.</p>
        <Callout kind="idea">
          Every lesson answers one question that the previous lesson left open. If you always know <b>which question you are currently answering</b>, you cannot get lost.
        </Callout>
      </Why>

      <Problem title="How people get lost learning LLMs">
        <WhyExists
          problem="An LLM is a stack of about ten ideas. Each is simple. Together they look like a wall."
          naive="Start at the bottom: a semester of linear algebra, then calculus, then probability, then neural networks. Or start at the top: memorise the words “attention”, “embedding”, “RAG” and hope they connect."
          fails="Bottom-up, you drown in maths without knowing what it is for, and quit. Top-down, you collect vocabulary with no mechanism behind it, and the first unexpected model behaviour leaves you helpless."
          idea="Keep the whole picture in view, and descend into each box only with a specific question in hand. Learn exactly the maths that question needs, at the moment it needs it."
          tradeoff="You will use some tools (such as the dot product) before you have seen their full theory. That is deliberate. Deep dives are there when you want more."
        />
      </Problem>

      <MentalModel title="One picture, three layers">
        <p>The boxes in the map below fall into three layers, plus a final engineering box that builds on all of them. Keep these three in your head and the rest is detail.</p>
        <div className="grid-3">
          <div className="card"><span className="chip acc">1 · into numbers</span><p style={{ marginTop: 8 }}>Text → Tokenization → Representations. Getting language into a form arithmetic can work on.</p></div>
          <div className="card"><span className="chip acc">2 · the model</span><p style={{ marginTop: 8 }}>Neural network → Attention → Transformer → Training → Inference → LLM. The function, how it gets its numbers, and how it is run.</p></div>
          <div className="card"><span className="chip acc">3 · around the model</span><p style={{ marginTop: 8 }}>RAG → Fine-tuning → Agents. What you, as a developer, build on top.</p></div>
        </div>
        <Callout kind="dev">
          Read it like a system diagram. Layer 1 is the serialisation format. Layer 2 is the service: its implementation, its build process (training) and its runtime (inference). Layer 3 is application code that calls the service. Most developers only ever work in layer 3. After this course you will do so knowing what is underneath.
        </Callout>
        <h3>How each lesson works, and how to use it</h3>
        <p>Every lesson has the same twelve sections, in the same order. The order is the teaching method: <b>why before how, how before maths, maths before code</b>.</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>#</th><th>section</th><th>what it is for</th></tr></thead>
            <tbody>{RHYTHM.map(([name, what], i) => <tr key={name}><td className="mono">{String(i + 1).padStart(2, '0')}</td><td><b>{name}</b></td><td>{what}</td></tr>)}</tbody>
          </table>
        </div>
        <p>Conceptual lessons drop the numbers, maths or code sections when there is honestly nothing to put in them. The experiment, the exercises and the recall are never dropped.</p>
        <h3>The tools around the lessons</h3>
        <ul>
          <li><b>Hints.</b> Exercises never show the answer straight away. Try first. If you are stuck for more than a few minutes, take <em>one</em> hint and try again. The solution button appears after the last hint. A struggle followed by a hint teaches far more than reading the solution cold.</li>
          <li><b>“Where are we?”</b> The box at the top of each lesson shows the parts of an LLM as a tree, with the current one marked. Glance at it whenever a lesson feels detached from the goal.</li>
          <li><b>Glossary.</b> Words with a dotted underline, such as <G t="logits">logits</G> (a word you meet properly in <a href="#/lesson/softmax">Softmax</a>), show a one-line reminder on hover and link to the <a href="#/glossary">glossary</a>. Every term is explained in plain words before it is used.</li>
          <li><b>Concept map.</b> <a href="#/map">#/map</a> shows how the ideas depend on each other.</li>
          <li><b>Search.</b> The box at the top of the sidebar searches lessons and glossary terms.</li>
          <li><b>Progress.</b> Completed lessons, exercises and quiz scores are saved in your browser only (no account, no server). Mark a lesson complete only when you could explain its key ideas without looking. You can reset everything from the home page.</li>
          <li><b>Deep dives.</b> Collapsed boxes with extra rigour. They are never required to continue. Skip them on a first pass without guilt.</li>
        </ul>
        <Callout kind="established">
          The quizzes and “explain in your own words” boxes are not decoration. Pulling an idea out of memory (retrieval practice) and coming back to it after a gap (spacing) are among the best-supported findings in learning research. That is why each part ends with a “Before moving on” checkpoint that reaches back to earlier parts.
        </Callout>
      </MentalModel>

      <TryIt title="Click through the map">
        <CourseArchitecture />
        <p>There is a second, finer map at <a href="#/map">Concept map</a> in the sidebar. It shows individual ideas (dot product, softmax, causal mask…) and what depends on what. Use it when you wonder “why did I need that again?”.</p>
      </TryIt>

      <CodeIt title="The Python files, and how to run them">
        <p>This course sits on top of a repository of small, complete Python programs. Each one implements a lesson’s idea from scratch, prints what it is doing, and runs on a laptop CPU. The code shown in lessons is taken from these files, so what you read here is what runs there.</p>
        <Code title="the repository">{`
phase1-foundations/   math_primer.py  gradient_descent.py  mlp_numpy.py
phase2-language/      bpe_tokenizer.py  tiny_word2vec.py  bigram_lm.py
phase3-transformers/  attention_numpy.py  tiny_gpt.py  kv_cache_demo.py
phase4-modern-llms/   vector_db.py  mini_rag.py  finetune_tiny_gpt.py
phase5-agents/        mini_agent.py
tests/                pytest tests for all of the above
`}</Code>
        <p>You need Python 3 and NumPy. Nothing else until you reach <a href="#/lesson/build-gpt">Build GPT</a>, where PyTorch comes in.</p>
        <Code title="setup and first run">{`
pip install numpy pytest          # enough for Parts 1 to 6
python phase1-foundations/math_primer.py

pip install torch                 # from "Build GPT" onward
python phase3-transformers/tiny_gpt.py --quick   # 2-minute smoke run

pytest                            # run the repository's tests
`}</Code>
        <p>Each lesson that has code lists its file under the title and links to it at the bottom, under “Read the real code”. Exercises of type <em>implement</em> and <em>modify</em> send you into these files. Do them. Reading code teaches you to recognise it; changing code teaches you to write it.</p>
        <Callout kind="dev">
          Why NumPy first and PyTorch later? PyTorch has one famous line, <code>loss.backward()</code>, that works out how to adjust every number in the model. In NumPy you write that part yourself, once, by hand. After that it is never magic: when you call <code>loss.backward()</code> later, you will know exactly what it is doing for you.
        </Callout>
      </CodeIt>

      <Exercises>
        <Exercise
          id="course-map-layers"
          type="predict"
          title="Which layer would you change?"
          hints={[
            'Go through the three layers: into numbers, the model, around the model. Which one can see a PDF that was written last week?',
            'The model’s numbers were fixed when training ended. Retraining them for one document is slow and expensive. What is cheap to change on every request?',
          ]}
          solution={<><p><b>Layer 3, specifically RAG.</b> The handbook is not in the model’s parameters, and it changes often. The cheap, reliable fix is to find the relevant passages at question time and put them into the prompt, so they become part of the “text so far” that the function reads.</p><p>Fine-tuning (changing the numbers) is for changing <em>behaviour</em>: tone, format, a specialised skill. It is a poor way to add facts that change. You will measure this yourself in Part 9.</p></>}
        >
          <p>Your company wants a chatbot that answers questions about its internal handbook, which is updated weekly. From the map alone: which of the boxes is the natural place to solve this, and why not the others?</p>
        </Exercise>

        <Exercise
          id="course-map-run"
          type="implement"
          title="Run your first file"
          hints={[
            'From the repository root: pip install numpy, then python phase1-foundations/math_primer.py',
            'You do not need to understand the output yet. Scroll through it and look for a dot product being computed step by step.',
          ]}
          solution={<p>You should see sections printed one after another, each working a small calculation by hand and then checking it with NumPy. If it runs, your setup is done for the first six parts of the course. If <code>python</code> is not found, try <code>python3</code>. If NumPy is missing, the error message says <code>No module named 'numpy'</code>: run the pip command again inside the same environment.</p>}
        >
          <p>Clone the repository, install NumPy and run <code>phase1-foundations/math_primer.py</code>. You are not expected to follow the output yet. The goal is a working setup before the first lesson that needs it.</p>
        </Exercise>

        <ExplainBack
          id="course-map-explain"
          prompt="Without looking at the map: describe the three layers of the course in your own words, and name at least two boxes in each."
          modelAnswer={<p>First, turning language into numbers: text is cut into tokens (tokenization) and each token becomes a vector (representations). Second, the model itself: a neural network whose key mechanism is attention, stacked into a Transformer, given its numbers by training, and run token by token at inference, which at scale is what we call an LLM. Third, what developers build around the model: RAG puts outside information into the prompt, fine-tuning adjusts the weights for a behaviour, and agents wrap the model in a loop that can call tools.</p>}
        />
      </Exercises>

      <CheckYourself
        questions={[
          {
            q: 'You are stuck on an exercise. What does this course want you to do?',
            options: ['Open the solution straight away to save time', 'Try, then take one hint at a time, trying again after each', 'Skip it; exercises are optional extras', 'Re-read the whole lesson from the top first'],
            answer: 1,
            explain: 'Effortful attempts are where the learning happens. Hints keep you moving without removing the effort.',
          },
          {
            q: 'Why does each lesson show the formula only after the interactive and the worked numbers?',
            options: ['To make lessons longer', 'Because a formula is a summary of steps; it only means something once you have done the steps', 'Because formulas are optional in machine learning', 'Because the formulas are approximate'],
            answer: 1,
            explain: 'Why before how, how before maths, maths before code. The equation should feel like shorthand for something you already did.',
          },
          {
            q: 'Which boxes of the map are about what is inside the model, as opposed to around it?',
            options: ['RAG and Agents', 'Attention, Transformer and Training', 'Fine-tuning and RAG', 'Tokenization and Agents'],
            answer: 1,
            explain: 'RAG changes the prompt and agents wrap the model in a loop: both live outside it. Attention and the Transformer are the function itself, and training is how it gets its numbers.',
          },
          {
            q: 'When does the course first need PyTorch?',
            options: ['From the very first maths lesson', 'Never; everything is NumPy', 'From “Build GPT” onward; before that, Python 3 and NumPy are enough', 'Only in the capstone'],
            answer: 2,
            explain: 'You write the forward and backward passes by hand in NumPy first, so PyTorch’s automation is not magic when it arrives.',
          },
        ]}
      />

      <Remember
        items={[
          <>Three layers: <b>text into numbers</b>, <b>the model</b> (how it works, how it is trained, how it is run), and <b>what you build around it</b>.</>,
          <>Every box on the map exists to answer a question. <b>Always know which question you are on.</b></>,
          <>Each lesson follows one rhythm: <b>why → problem → intuition → experiment → numbers → maths → code → exercises → recall → real LLMs</b>.</>,
          <>Try first, then hints. Mark a lesson complete only when you could <b>explain it without looking</b>.</>,
          <>The repository’s Python files are the ground truth. <b>Python 3 + NumPy</b> to start; <b>PyTorch from Build GPT onward</b>.</>,
        ]}
      />

      <RealLLM>
        <p>Is this map the real thing, or a teaching simplification? Mostly the real thing, at a very different scale.</p>
        <SameShapeBiggerNumbers />
        <ToyVsReal
          toy={<ul><li>You build every box yourself, in a few hundred lines each</li><li>Models with thousands to a few million parameters</li><li>Trains in minutes on a laptop CPU</li><li>Writes Shakespeare-flavoured gibberish</li></ul>}
          real={<ul><li>The same boxes, each owned by a specialist team</li><li>Billions to trillions of parameters</li><li>Trains for weeks to months on thousands of GPUs</li><li>Extra stages we only describe: instruction tuning, preference tuning, safety work, serving infrastructure</li></ul>}
        />
        <Callout kind="established">
          The architecture you will build in <a href="#/lesson/build-gpt">Build GPT</a> is the GPT-2 architecture, and today’s open models are recognisable refinements of it. <a href="#/lesson/modern-architecture">Modern LLM architecture</a> goes through the differences one by one, and you will be able to read them as small edits to code you wrote.
        </Callout>
        <DeepDive title="What this course deliberately leaves out">
          <p>Distributed training across many machines, GPU kernel engineering, data collection and filtering at web scale, images and audio, evaluation methodology, and most of the safety and alignment literature. Each is a field of its own. The course gives you the mechanism that all of them assume you already understand.</p>
        </DeepDive>
      </RealLLM>

      <BeforeMovingOn
        id="part-0"
        intro="This is the first checkpoint, so it reaches back over Part 0 only. Later checkpoints will deliberately reach back to earlier parts. Answer from memory."
        questions={[
          {
            q: 'What does the model function return each time it is called?',
            options: ['The finished answer', 'A probability for every token in the vocabulary', 'The id of the best-matching stored document', 'One embedding vector'],
            answer: 1,
            explain: 'Tokens so far in, next-token probabilities out. Everything else is built around that.',
          },
          {
            q: 'Where is the fact “Paris is the capital of France” inside an LLM?',
            options: ['In a table of capitals', 'In a compressed copy of an encyclopedia', 'Nowhere as a sentence: it is implicit in parameters that make “ Paris” a highly probable continuation', 'In the tokenizer’s vocabulary'],
            answer: 2,
            explain: 'Knowledge is implicit in the learned numbers. That is also why there is no “not found”.',
          },
          {
            q: 'Same prompt, two different answers. Why?',
            options: ['The parameters drift over time', 'The sampling step picks tokens at random according to their probabilities', 'The tokenizer is random', 'A different database shard answered'],
            answer: 1,
            explain: 'Sampling is the only random stage. With it turned off, the same prompt gives the same output.',
          },
          {
            q: 'Why does generating 200 tokens take about 200 runs of the model?',
            options: ['Each run yields one token, and that token is part of the input for the next run', 'The model is rate-limited on purpose', 'Every token is checked against a database', 'The tokenizer has to be retrained each time'],
            answer: 0,
            explain: 'Autoregressive generation: the output is appended and fed back in, so the steps cannot be skipped.',
          },
          {
            q: 'A model invents a confident, false citation. Which statement is the most accurate?',
            options: ['A lookup returned the wrong row', 'It produced a likely-looking continuation; nothing in the mechanism checks truth or can return “not found”', 'The random seed was corrupted', 'It ran out of context window'],
            answer: 1,
            explain: 'Fluency and made-up answers come from the same design: generate what is plausible.',
          },
        ]}
      >
        <OrderExercise
          id="course-map-pipeline-order"
          title="Rebuild the prompt → answer pipeline from memory"
          prompt={<p>No scrolling back to lesson 0.1. Put the ten stages in order, from the text you type to the loop.</p>}
          correct={PIPELINE_ORDER}
          solutionNote={<p>Text is cut into tokens by the tokenizer, tokens become ids, ids become embeddings. The Transformer turns those into next-token probabilities, sampling picks one, the token is appended, and the loop repeats. You will rebuild this pipeline several more times during the course, each time with more of the boxes opened up.</p>}
        />
      </BeforeMovingOn>
    </Lesson>
  )
}
