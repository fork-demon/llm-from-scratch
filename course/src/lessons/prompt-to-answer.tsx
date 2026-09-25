import { Lesson, Why, Problem, MentalModel, TryIt, BreakIt, Exercises, CheckYourself, Remember, RealLLM } from '../components/lesson'
import { Callout, DeepDive, Flow, G, Term, ToyVsReal, WhyExists } from '../components/ui'
import { Exercise, ExplainBack, OrderExercise } from '../components/exercise'
import { PipelineExplorer } from '../interactive/PipelineExplorer'
import { LoopDiagram } from '../illustrations/LoopDiagram'

export default function PromptToAnswerLesson() {
  return (
    <Lesson id="prompt-to-answer">
      <Why title="What happens when you press Enter?">
        <p className="lede">Monday morning at Paisa Pal, a fintech office in Bengaluru. The CEO has one line on the all-hands slide: “Our own ChatGPT for customer support.”</p>
        <p>By lunch, Riya has been moved to the new AI team. She is a good backend developer. Java, Python, queues, databases. She has never trained a model in her life.</p>
        <p>To start somewhere, she opens the vendor chatbot the team is trialling and types a test question:</p>
        <div className="card center" style={{ fontFamily: 'var(--serif)', fontSize: 22 }}>What is a cat?</div>
        <p>A moment later, words begin to appear. “A cat is a small furry animal…” They arrive one piece at a time, like someone typing.</p>
        <p>Dev, her flatmate, is reading over her shoulder with his chai. “Easy,” he says. “It looked it up.”</p>
        <p>Riya is not so sure. She uses these tools every day, and she realises she cannot explain what happened in that one second. Kabir, the team’s ML lead, only smiles. “Good. Then we start at the bottom.”</p>
        <p>By the end of this course you will be able to draw every step of that second, explain it, and point at the code that does it. You will have written that code yourself.</p>
        <p>This first lesson has no maths and no code. It shows you the whole machine from the outside, once, so that every later lesson has somewhere to sit.</p>
        <Callout kind="idea">
          An LLM answers by doing one small thing over and over: given all the text so far, work out <b>how likely each possible next piece of text is</b>, pick one, add it to the text, and go again.
        </Callout>
      </Why>

      <Problem title="The wrong picture: “it looks the answer up”">
        <p>Dev’s guess is the one most of us make. As developers we reach for what we know. Text goes in, a relevant answer comes out. That smells like a database or a search index.</p>
        <p>It is a natural guess. It is also wrong. And it will mislead you about everything the model does well, and everything it does badly.</p>
        <WhyExists
          problem="We need a working picture of what sits between the prompt and the answer."
          naive="Somewhere there is a giant store of questions and answers. The model finds the closest match and returns it."
          fails="Ask for a limerick about a Kubernetes pod. Nobody has ever written that, so there is nothing to find, yet you get one. And ask the same question twice: you often get two differently worded answers. A lookup would return the same row."
          idea="Nothing is looked up. The answer does not exist anywhere until it is produced, one small piece at a time, by a calculation."
          tradeoff="A calculation that can write anything can also write things that are false. There is no stored row to check against."
        />
        <p>Here are three things you can check yourself in any chat product. A lookup cannot explain any of them.</p>
        <ul>
          <li>The answer <b>streams</b> in piece by piece. A lookup would return the whole row at once.</li>
          <li>Longer answers take <b>roughly proportionally longer</b>. Each piece costs about the same amount of work.</li>
          <li>The same prompt can give <b>different answers</b>. Somewhere, dice are being rolled.</li>
        </ul>
        <p>The pipeline below explains all three.</p>
      </Problem>

      <MentalModel title="A pipeline with a loop at the end">
        <p>Kabir picks up a marker. “This is what happened when you pressed Enter.”</p>
        <p>Here is that whole journey, drawn with the actual data at each stage.</p>
        <p>Some of the words will be new. That is fine. Each one has a plain-English line under it here, and a whole lesson of its own later.</p>
        <LoopDiagram />
        <p>Follow the numbers 1 to 8, then the dashed arrow. The chosen token “A” is glued onto the text. Then the whole trip runs again, to choose the token after it.</p>
        <p>One trip per token, until the answer is finished. That is why Riya saw the words arrive one by one.</p>
        <p>Two pieces of vocabulary are worth fixing now, because the whole course leans on them.</p>
        <Term
          name="Token"
          plain={<>A piece of text from a fixed list the model was built with. Often a whole word, sometimes part of a word, sometimes a punctuation mark.</>}
          example={<>“What is a cat?” → <span className="token">What</span><span className="token">·is</span><span className="token">·a</span><span className="token">·cat</span><span className="token">?</span> (5 tokens; the dot marks a space that belongs to the token).</>}
          formal={<>An element of the model’s <G t="vocabulary">vocabulary</G>, identified by an integer id. Models read and write token ids, never characters.</>}
        />
        <Term
          name="Autoregressive generation"
          plain={<>Writing one token at a time, where each new token is chosen after reading everything written so far, <em>including the model’s own earlier output</em>.</>}
          example={<>“What is a cat?” → “ A”. Then “What is a cat? A” → “ cat”. Then “What is a cat? A cat” → “ is”. And so on.</>}
          formal={<>The probability of a sequence is built as a product of next-token probabilities, each one conditioned on all previous tokens.</>}
        />
        <Callout kind="analogy">
          Think of your phone keyboard’s next-word suggestions, and imagine tapping a suggestion again and again to write a whole message. The loop is the same: look at the text so far, propose next words, pick one, repeat.
          <br /><br />
          Where the analogy stops: your keyboard looks at a couple of words and uses simple statistics. An LLM reads the <em>entire</em> text so far, and runs it through a calculation with billions of adjustable numbers.
          <br /><br />
          The loop is the same. The quality of the prediction is not even close.
        </Callout>
        <Callout kind="dev">
          In code, the whole thing is a <code>while</code> loop around a pure function: <code>probs = model(tokens)</code>, then <code>tokens.append(pick(probs))</code>.
          <br /><br />
          The model has no memory between calls and no loop inside it. All the “conversation” lives in the growing list of tokens you pass back in.
        </Callout>
      </MentalModel>

      <TryIt title="Click through the machine, then run the loop">
        <p>First click through the ten stages for the prompt. Then press <b>Generate next token</b> a few times, and click back to “Tokens”, “Embeddings” and “Next-token probabilities” to see how they changed.</p>
        <PipelineExplorer />
        <p>Notice what you did <em>not</em> see. No search. No stored answers. No sentence waiting to be fetched. Only numbers flowing forward, one roll of a die, and a loop.</p>
        <p>So Dev was wrong, but in a useful way. Nothing was looked up.</p>
      </TryIt>

      <BreakIt>
        <p>Predict first, then check in the explorer.</p>
        <ul>
          <li><b>Turn randomness off</b> (“Always take the most likely token”) and run to the end. Reset and run again. Do you ever get a different sentence?</li>
          <li><b>Set the seed to 24</b> with randomness on, and run to the end. The answer is fluent, confident, and wrong. Look at the probabilities stage just after “ A” to see how that happened.</li>
          <li><b>Try seeds 1 to 10.</b> Same prompt, same model, same probabilities at the first step. Only the dice differ.</li>
          <li><b>Find a long one</b> (seed 9). Nothing forces the model to stop. It stops only when the “end” token happens to be picked, or when a length limit cuts it off.</li>
        </ul>
      </BreakIt>

      <Exercises>
        <OrderExercise
          id="prompt-to-answer-order"
          title="Rebuild the pipeline"
          prompt={<p>Without scrolling up, put the stages in the order the data flows through them.</p>}
          correct={['Text', 'Tokenizer', 'Tokens', 'Token IDs', 'Embeddings', 'Transformer', 'Next-token probabilities', 'Sampling', 'Next token', 'Repeat']}
          solutionNote={<p>Text must become numbers before any arithmetic can happen (tokenizer → tokens → ids → embeddings). The Transformer turns those numbers into probabilities. Sampling turns probabilities into one choice. The choice joins the text, and the loop goes round.</p>}
        />

        <Exercise
          id="prompt-to-answer-trace"
          type="trace"
          title="Count the runs"
          answer={{ value: 6, tolerance: 0 }}
          answerLabel="number of runs"
          hints={[
            'Each run of the pipeline produces exactly one token. So count the tokens that were generated, not the words.',
            'Set the seed to 1, run to the end and open the “Repeat” stage: it lists one line per run.',
            'The tokens are “ A”, “ cat”, “ is”, “ furry”, “.” and one more that you do not see in the text.',
          ]}
          solution={<><p><b>6.</b> Five visible tokens (“ A”, “ cat”, “ is”, “ furry”, “.”) plus the invisible <span className="mono">&lt;end&gt;</span> token. Stopping is itself a prediction, so it costs a run like any other token.</p><p>This is why long answers are slow and why API pricing counts output tokens: every single token costs one full pass through the model.</p></>}
        >
          <p>Riya sets the seed to 1 (randomness on), and the explorer answers “A cat is furry.” How many times did the whole pipeline, including the Transformer, have to run to produce that answer?</p>
        </Exercise>

        <Exercise
          id="prompt-to-answer-sample"
          type="calculate"
          title="Be the sampler"
          answer={{ text: ['It', ' It', '·It'] }}
          answerLabel="token"
          hints={[
            'Open the “Sampling” stage before generating anything. Which range does each token own?',
            '“ A” owns 0 to 0.70 because its probability is 70%. “ It” owns the rest.',
          ]}
          solution={<><p><b>“ It”.</b> “ A” owns the range 0 to 0.70 and “ It” owns 0.70 to 1.00. The number 0.85 falls in the second range.</p><p>That is all “sampling” means: a token with 70% probability owns 70% of the number line, so it gets picked 70% of the time. Likely tokens usually win. Unlikely tokens sometimes do.</p></>}
        >
          <p>Right after the prompt, the toy model gives “ A” 70% and “ It” 30%. The sampler draws the random number <span className="mono">0.85</span>. Which token is picked?</p>
        </Exercise>

        <Exercise
          id="prompt-to-answer-predict"
          type="predict"
          title="The confident wrong answer"
          hints={[
            'Run seed 24 one token at a time. At which exact step does it go wrong?',
            'Look at the probabilities after “ A”. How big is the chance of “ dog”? What did the random number have to be?',
            'After “ dog” has been appended, look at what the model sees as its input. Does anything in the pipeline go back and check?',
          ]}
          solution={<><p>After “ A”, the toy model gives “ cat” 92% and “ dog” 8%. With seed 24 the die lands in the 8%. From then on “ dog” is part of the text so far like any other word, and the model continues from it as fluently as from anything else: “A dog is a furry animal.”</p><p>Nothing in the pipeline looks back and asks “is this true?”. There is no stage for that. Each step only asks “what is likely to come next, given the text so far?”. Hold on to this: it is the seed of why real models sometimes state false things with total confidence. <a href="#/lesson/why-llms-know">Why LLMs know things</a> returns to it properly.</p></>}
        >
          <p>Seed 24 produces “A dog is a furry animal.” as the answer to “What is a cat?”. Before investigating: how can a system produce a wrong answer in perfectly good English? Then find the exact step where it went wrong, and explain why nothing corrected it.</p>
        </Exercise>

        <ExplainBack
          id="prompt-to-answer-explain"
          prompt="Dev says: “ChatGPT is basically a search engine with a nicer interface.” Using what you saw in the explorer, explain in a few sentences what actually happens between the prompt and the answer."
          modelAnswer={<p>Nothing is searched. The prompt is cut into tokens, the tokens become numbers, and a very large calculation turns those numbers into a probability for every possible next token. One token is picked at random according to those probabilities and appended to the text. Then the whole thing runs again on the longer text, and again, until an end token is picked. The answer never existed anywhere before it was generated, which is why the model can write things nobody has written, why the same question can give different answers, and why it can be fluently wrong.</p>}
        />
      </Exercises>

      <CheckYourself
        questions={[
          {
            q: 'What does the neural network inside an LLM actually output each time it runs?',
            options: ['A complete sentence', 'The single best next word', 'A probability for every token in its vocabulary', 'A database key for the answer'],
            answer: 2,
            explain: 'One run, one list of probabilities. Choosing a token from that list is a separate, simple step outside the network.',
          },
          {
            q: 'You ask the same question twice and get two differently worded answers. Which stage is responsible?',
            options: ['The tokenizer cuts the text differently each time', 'Sampling: a token is picked at random, weighted by probability', 'The model’s numbers change between requests', 'The embeddings are re-rolled for each request'],
            answer: 1,
            explain: 'Everything up to the probabilities is a fixed calculation. The random pick is the only deliberate source of variety. Turn it off in the explorer and the same prompt gives the same answer. (Real services can still vary slightly: see the note at the end of the lesson.)',
          },
          {
            q: 'A 500-token answer takes about ten times longer than a 50-token answer. Why?',
            options: ['Longer answers are stored further away', 'Each token needs its own full pass through the model, one after another', 'The tokenizer is slow on long text', 'The model thinks harder about long answers'],
            answer: 1,
            explain: 'One pass, one token. Token 200 cannot be computed before token 199 exists, because it is part of the input.',
          },
          {
            q: 'After the model has generated “ A”, what is the input for the next run?',
            options: ['Only the token “ A”', 'Only the original prompt', 'The original prompt plus “ A”', 'A hidden summary the model kept from last time'],
            answer: 2,
            explain: 'The model keeps nothing between runs. Its own output is appended to the text and read again as input. That is what “autoregressive” means.',
          },
          {
            q: 'How does generation stop?',
            options: ['The model runs out of stored text', 'A special end token is picked like any other token, or a length limit is reached', 'The tokenizer detects a complete sentence', 'A separate program judges that the answer is good enough'],
            answer: 1,
            explain: 'Stopping is just another next-token prediction. That is why models sometimes ramble, or stop too early.',
          },
        ]}
      />

      <Remember
        items={[
          <>An LLM does <b>not look answers up</b>. The answer does not exist until it is generated.</>,
          <>The pipeline: <b>text → tokens → ids → embeddings → Transformer → probabilities → sampling → next token → repeat</b>.</>,
          <>The network’s only output is <b>a probability for every possible next token</b>. Everything else is plumbing around that.</>,
          <><b>One pass, one token.</b> The model’s own output is appended and fed back in (<G t="autoregressive">autoregressive</G> generation).</>,
          <>Randomness lives in <b>one place</b>: the sampling step. Nothing in the loop checks whether the text is true.</>,
        ]}
      />

      <RealLLM>
        <p>The explorer is a cardboard model of the machine. Here is what is cardboard and what is real.</p>
        <ToyVsReal
          toy={<ul><li>18 tokens, split on spaces</li><li>4 hand-picked numbers per token</li><li>“Transformer” = a hand-written table that looks at the last 1 to 2 tokens</li><li>Plain weighted random pick</li></ul>}
          real={<ul><li>Roughly 30,000 to 250,000 tokens, learned from data; rare words split into pieces</li><li>Hundreds to thousands of learned numbers per token</li><li>Dozens of Transformer layers, billions of learned numbers, reading the entire text so far</li><li>The same pick, with knobs such as <G t="temperature">temperature</G> and <G t="top-k">top-k</G></li></ul>}
        />
        <Callout kind="established">
          The shape of the pipeline is not a simplification. GPT, Llama, Claude and Gemini all generate text with this loop: tokenize, embed, run a Transformer, get next-token probabilities, sample, append, repeat. What you watched stream into a chat window was this loop running.
        </Callout>
        <Callout kind="model">
          Chat products add machinery around the loop that we ignore for now: a hidden system prompt placed before your text, special tokens that mark who is speaking, safety filters, and sometimes tools such as web search. None of it changes the core: all of it ends up as tokens in the input, and the model still writes one token at a time. The last part of the course (<a href="#/lesson/rag">RAG</a>, <a href="#/lesson/agents">Agents</a>) covers this.
        </Callout>
        <DeepDive title="Is it really deterministic with randomness switched off?">
          <p>In the explorer, yes. In principle, yes: the network is a fixed calculation, so the same tokens give the same probabilities, and “always take the most likely token” (often offered as temperature 0) then gives the same answer.</p>
          <p>In practice, hosted models can still give slightly different answers at temperature 0. Floating-point additions done in a different order give results that differ in the last digits, and the order can depend on the hardware and on which other requests are batched together with yours. When two tokens are almost tied, that is enough to flip the pick, and from there the texts diverge. So treat temperature 0 as “nearly repeatable”, not as a guarantee.</p>
        </DeepDive>
        <DeepDive title="If the model has no memory, how does a chat remember what I said?">
          <p>It does not. On every turn, the application sends the <em>whole conversation so far</em> as one long text: your first message, the model’s reply, your second message, and so on. The model reads all of it from scratch and predicts what comes next.</p>
          <p>That is also why conversations have a maximum length (the <G t="context-window">context window</G>) and why long chats cost more per message. Later you will meet the <G t="kv-cache">KV cache</G>, a trick that avoids redoing the arithmetic for tokens already seen. It saves work, but it does not change what is computed.</p>
        </DeepDive>
        <p>That evening Riya tells Dev, “It didn’t look anything up. It wrote the answer, one token at a time.” He is not convinced yet. Fair enough: the course has only started.</p>
        <p><b>Where next:</b> if nothing is stored and looked up, then where does “Paris is the capital of France” live? That is <a href="#/lesson/surprising-idea">the next lesson</a>.</p>
      </RealLLM>
    </Lesson>
  )
}
