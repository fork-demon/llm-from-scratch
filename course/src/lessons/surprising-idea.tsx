import { Lesson, Why, Problem, MentalModel, TryIt, Numbers, BreakIt, Exercises, CheckYourself, Remember, RealLLM } from '../components/lesson'
import { Callout, DeepDive, G, Term, ToyVsReal, WhyExists } from '../components/ui'
import { Code } from '../components/Code'
import { Exercise, ExplainBack } from '../components/exercise'
import { FourSystems } from '../interactive/FourSystems'

export default function SurprisingIdeaLesson() {
  return (
    <Lesson id="surprising-idea">
      <Why title="If nothing is looked up, where is Paris?">
        <p className="lede">Ask an LLM: “What is the capital of France?” It says Paris. Correctly, instantly, essentially every time.</p>
        <p>In <a href="#/lesson/prompt-to-answer">the last lesson</a> you saw the machinery: tokens in, probabilities out, pick one, repeat. No stage in that pipeline fetched anything.</p>
        <p>So where was “Paris”? There is no table of capitals inside the model. There is no document about France. You could read every byte of the model file and never find the sentence “Paris is the capital of France”.</p>
        <p>And yet it answers. It also answers this, which nobody in history had written down before you asked:</p>
        <div className="card center" style={{ fontFamily: 'var(--serif)', fontSize: 20 }}>“Write a limerick about a Kubernetes pod.”</div>
        <Callout kind="idea">
          An LLM is not a store of sentences. It is a <b>function</b>: text so far in, probabilities for the next token out. What it “knows” is not written anywhere in it. It is implicit in billions of numbers that shape that function.
        </Callout>
      </Why>

      <Problem title="Every system you have built works differently">
        <p>As a developer you know three ways to make a computer answer a question. All three are about <em>getting out what somebody put in</em>.</p>
        <div className="grid-3">
          <div className="card"><h4 style={{ fontSize: 17, marginBottom: 6 }}>Database</h4><p>Someone stored a row. You fetch that row by its key.</p></div>
          <div className="card"><h4 style={{ fontSize: 17, marginBottom: 6 }}>Search engine</h4><p>Someone wrote a document. You get a ranked list of documents that match your words.</p></div>
          <div className="card"><h4 style={{ fontSize: 17, marginBottom: 6 }}>Program</h4><p>Someone wrote a rule for this case. The rule runs.</p></div>
        </div>
        <WhyExists
          problem="People ask open-ended questions in free wording, including questions nobody has asked before."
          naive="Store more rows, index more documents, write more rules."
          fails="Language is unbounded. The number of possible 20-word questions is astronomically large. You cannot pre-store answers, and you cannot write a rule per case."
          idea="Stop storing answers. Instead, learn one general function that, given any text, says what is likely to come next. Then generate answers with it."
          tradeoff="A function that produces plausible continuations for any input will also produce plausible continuations when it should say “I do not know”."
        />
      </Problem>

      <MentalModel title="A function with billions of dials">
        <p>Here is the entire idea of an LLM, written as a function signature:</p>
        <Code title="the whole model, from the outside">{`
def model(tokens_so_far: list[int]) -> list[float]:
    """Return one probability per vocabulary entry: how likely is each
    token to come next? Pure function. No database. No network calls."""
`}</Code>
        <p>Inside that function there is only arithmetic: multiply, add, repeat. The arithmetic uses a very long list of constants. Those constants are what the model <em>is</em>.</p>
        <Term
          name="Parameters (also called weights)"
          plain={<>The adjustable numbers inside the model. Change them and the same input gives different probabilities. A model file is, near enough, this list of numbers and nothing else.</>}
          example={<><code>y = a·x + b</code> has two parameters, <code>a</code> and <code>b</code>. Set them to 2 and 1 and the function maps 3 → 7. A small GPT-2 has 124 million of them. Modern models have billions to trillions.</>}
          formal={<>The learned tensors of the network: embedding tables, attention and feed-forward matrices, normalisation gains. Fitted by <G t="gradient-descent">gradient descent</G> to minimise next-token prediction error on the training text.</>}
        />
        <p><b>Where do the numbers come from?</b> Nobody types them in. They start random. Then the model is shown a huge amount of text, one position at a time, and asked “what comes next?”.</p>
        <p>Each time it is wrong, every number is nudged a tiny bit in the direction that would have made the right token more likely. (In practice the nudges for many positions are averaged and applied together.)</p>
        <p>Trillions of predictions later, the function is good at predicting text. That whole process is <G t="pretraining">training</G>, and Parts 2, 3 and 7 build it from scratch.</p>
        <p>After training, “Paris” is not stored. But the numbers have been shaped so that, when the input is “The capital of France is”, the arithmetic happens to produce a very high probability for “ Paris”.</p>
        <Callout kind="analogy">
          Think of a pianist who has practised a thousand pieces. There is no sheet music stored in their fingers. Practice adjusted millions of connections, and now the right movement <em>comes out</em> when the context calls for it. They can also improvise something new in the style of what they practised, and they can confidently play a wrong note.
          <br /><br />
          Where the analogy stops: a pianist understands music, has intentions and hears their mistakes. Do not carry those over. The only part to keep is this: <b>skill held in adjusted numbers, not in stored copies</b>.
        </Callout>
        <Callout kind="dev">
          You have met “knowledge as parameters” before. A spam filter does not store a list of spam emails: it stores a few thousand weights. A line fitted through data points does not store the points: it stores a slope and an intercept, and can answer for an x it never saw. An LLM is the same idea at an absurd scale: a curve fitted through human text.
        </Callout>
      </MentalModel>

      <TryIt title="Put the same question to four systems">
        <p>The database, the search engine and the program below are small but real, and they run in your browser. The LLM column is the one we are still building up to. There are six questions. Predict what each system will do before you look.</p>
        <FourSystems />
        <p>Look at <em>how</em> each one fails. The database returns nothing. The program throws an error. The search engine shows you weak results and lets you judge. The LLM is the only one that can fail <b>without any sign of failure</b>.</p>
      </TryIt>

      <Numbers title="Could it be storing the text anyway?">
        <p>A fair suspicion: maybe the numbers are just a compressed copy of the internet. Let’s check with public figures for one open model, Llama 2 7B.</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>quantity</th><th>calculation</th><th>size</th></tr></thead>
            <tbody>
              <tr><td>Training text</td><td className="mono">2 trillion tokens × about 4 bytes of text per token</td><td className="mono"><b>≈ 8,000 GB</b></td></tr>
              <tr><td>The model</td><td className="mono">7 billion parameters × 2 bytes each</td><td className="mono"><b>14 GB</b></td></tr>
              <tr><td>Ratio</td><td className="mono">8,000 ÷ 14</td><td className="mono"><b>≈ 570 to 1</b></td></tr>
            </tbody>
          </table>
        </div>
        <p>Lossless text compression manages somewhere between 3 to 1 and 10 to 1. Nothing squeezes text 570 to 1 and gets it back. So the model <em>cannot</em> be holding more than a small fraction of its training text. Mostly it can only be holding the <b>regularities</b> in it: grammar, facts that recur, styles, how code is structured, how arguments go.</p>
        <p>This cuts both ways. A fact that appeared thousands of times has shaped the numbers strongly. A fact that appeared once has usually left almost no trace, but the model will still produce a fluent continuation when asked about it.</p>
        <DeepDive title="But I have seen a model recite a poem word for word">
          <p>Both things are true. Text that was repeated very often in the training data (famous poems, licences, well-known code snippets) can be reproduced verbatim, and researchers have shown that carefully chosen prompts can extract memorised passages. That is an <b>established</b> finding.</p>
          <p>It does not contradict the arithmetic above. Memorisation covers a small fraction of the data, mostly text that was repeated many times, and the fraction grows with model size. It happens through the same mechanism as everything else: the numbers were nudged so often toward that exact continuation that its probability became nearly 1 at every step. There is still no lookup, and the vast majority of training text is not recoverable.</p>
        </DeepDive>
      </Numbers>

      <BreakIt title="Why this makes it fluent, and why it makes things up">
        <p>The same design decision explains the best and the worst of LLMs. It is worth seeing them side by side.</p>
        <div className="grid-2">
          <div className="card">
            <h4 style={{ fontSize: 17, marginBottom: 6 }}>Why it is fluent and flexible</h4>
            <ul>
              <li>It never depended on your exact wording, so rephrasing, typos and new combinations all work.</li>
              <li>It produces text rather than finding it, so it can write what has never been written.</li>
              <li>It learned from an enormous range of text, so style, grammar and common knowledge come “for free”.</li>
            </ul>
          </div>
          <div className="card">
            <h4 style={{ fontSize: 17, marginBottom: 6 }}>Why it makes things up</h4>
            <ul>
              <li>The function <em>always</em> returns probabilities. There is no “row not found” code path.</li>
              <li>It was trained to produce <em>likely</em> text. Likely and true overlap a lot, but they are not the same thing.</li>
              <li>A rare fact and a non-existent fact can look similar from the inside: both leave the numbers with little to go on, and a plausible-sounding token wins.</li>
            </ul>
          </div>
        </div>
        <p>A confident false statement from an LLM is called a <G t="hallucination">hallucination</G>. Notice that it is not a bug in the usual sense. Nothing malfunctioned. The model did exactly what it always does.</p>
        <p>Go back to the four systems and try “type your own” with a capital the program does not know, such as Spain. The program fails loudly. Now imagine an LLM that had seen little about that country: what would it do instead?</p>
        <Callout kind="model">
          “It only produces likely text” is the right first model, and it is where this course starts. Real assistants are further trained to be helpful, to follow instructions and to admit uncertainty more often, which shifts what counts as “likely”. That works partly because models do carry some internal signal of whether a topic is familiar, which research has measured. It reduces made-up answers. It does not remove the cause. <a href="#/lesson/training-pipeline">From raw text to assistant</a> and <a href="#/lesson/why-llms-know">Why LLMs know things</a> pick this up.
        </Callout>
      </BreakIt>

      <Exercises>
        <Exercise
          id="surprising-idea-size"
          type="calculate"
          title="How big is the file?"
          answer={{ value: 140, tolerance: 1 }}
          answerLabel="size in GB"
          hints={[
            'A model file is, to a good approximation, just its parameters written one after another.',
            '70 billion numbers × 2 bytes per number = ? bytes. One GB is one billion bytes.',
          ]}
          solution={<><p>70,000,000,000 × 2 bytes = 140,000,000,000 bytes = <b>140 GB</b>.</p><p>That is the whole model: no database next to it, no document index. It is also why big models need several GPUs just to be loaded, and why people store parameters in 8 or 4 bits instead of 16 to shrink them.</p></>}
        >
          <p>An open model has 70 billion parameters, each stored as a 16-bit number (2 bytes). Roughly how large is the model file, in GB?</p>
        </Exercise>

        <Exercise
          id="surprising-idea-predict"
          type="predict"
          title="The paper that does not exist"
          hints={[
            'What does a database do when the key is missing? What can the model function return?',
            'The model has seen thousands of reference lists. What does text that follows “the 2019 paper by” usually look like?',
          ]}
          solution={<><p>A model answering from its parameters alone will often produce a complete, well-formatted citation: plausible author names, a plausible title, a real-sounding journal, a year. All invented. Each token was a likely continuation of the one before. “Likely-looking reference” is a pattern the model knows extremely well, and nothing in the mechanism checks the reference against reality.</p><p>This has happened in public: lawyers have been sanctioned for filing court documents containing case citations that an LLM made up. Assistants that are connected to a search tool do better, because then real text is placed in the prompt. That is a change to the input, not to the mechanism.</p></>}
        >
          <p>You ask an LLM with no web access: “Give me the full citation for the 2019 paper that proved bubble sort is optimal.” No such paper exists. Predict what kind of response is likely, and explain why in terms of the function <code>tokens so far → probabilities</code>.</p>
        </Exercise>

        <Exercise
          id="surprising-idea-experiment"
          type="experiment"
          title="Find each system’s blind spot"
          hints={[
            'Use “type your own”. Try a capital the program knows (Italy) and one it does not (Spain).',
            'Try changing a single character in a question the database knows, such as removing the question mark.',
            'Try “12 + 30”. Then try “twelve plus thirty”.',
          ]}
          solution={<><p>The database misses as soon as the key differs by one character: exact match is all it has. The search engine survives rewording but returns nothing for “12 + 30” and can never return more than what is in its six documents. The program handles <em>any</em> two numbers, which neither of the others can, but “twelve plus thirty” matches no rule.</p><p>The pattern: the three classic systems are <b>exact and brittle</b>. An LLM is the opposite: <b>flexible and inexact</b>. Real products combine them, letting the LLM handle language and handing exact work to databases, search and code. That combination is the last part of this course.</p></>}
        >
          <p>In the interactive, choose “type your own”. Find one input that breaks the database but not the search engine, one that only the program can answer, and one that breaks the program by rewording alone.</p>
        </Exercise>

        <ExplainBack
          id="surprising-idea-explain"
          prompt="Someone asks you: “If the model does not store facts anywhere, how can it know that Paris is the capital of France? And why does it sometimes invent things?” Answer both with one mechanism."
          modelAnswer={<p>The model is a function from the text so far to probabilities for the next token, controlled by billions of numbers. Those numbers were tuned on a huge amount of text so that real continuations became likely. Text about France and Paris appeared so often that, after “The capital of France is”, the arithmetic gives “ Paris” a very high probability. So the fact lives implicitly in the numbers, not as a stored sentence. The same mechanism explains invention: the function always outputs probabilities, even for a question about something rare or non-existent, and then the most plausible-sounding tokens win. It has no separate step that checks truth, and no way to return “not found”.</p>}
        />
      </Exercises>

      <CheckYourself
        questions={[
          {
            q: 'You open a model file in a hex editor. What is in it?',
            options: ['Compressed web pages', 'A large table of questions and answers', 'Billions of numbers (the parameters), plus a little metadata', 'Source code with many if-statements'],
            answer: 2,
            explain: 'The model is its parameters. The code that uses them is a few hundred lines, as you will see when you build one.',
          },
          {
            q: 'Why can an LLM answer a reworded question that a key-value lookup misses?',
            options: ['It normalises case and punctuation before looking the text up', 'It never matches exact wording: it computes probabilities from whatever tokens arrive', 'It stores every likely rewording of each question it has seen', 'It searches the web for similarly worded questions'],
            answer: 1,
            explain: 'There is no key to miss. Similar inputs flow through the same arithmetic and tend to produce similar outputs.',
          },
          {
            q: 'An LLM states a false “fact” in a confident tone. What went wrong inside it?',
            options: ['A corrupted database row', 'Nothing malfunctioned: it produced likely tokens, as always, and here likely was not true', 'The tokenizer split the question badly', 'The random seed was unlucky, and a different seed always fixes it'],
            answer: 1,
            explain: 'There is no code path for “not found”. A different sample might be right or wrong; the cause is that plausibility, not truth, drives the probabilities.',
          },
          {
            q: 'A model was trained on about 8,000 GB of text and is 14 GB in size. What does that tell you?',
            options: ['It uses an extremely good zip algorithm', 'It cannot contain its training text; it can only have captured regularities in it', 'Most of the training text was ignored', 'The rest of the text is fetched from a server when needed'],
            answer: 1,
            explain: 'No lossless compression of text gets near 570 to 1. What remains is patterns: frequent facts strongly, rare facts faintly.',
          },
          {
            q: 'Why does “just predict the next token” push a model toward learning real regularities about the world?',
            options: ['It does not: a next-token predictor can only ever learn spelling and word frequencies', 'Because every sentence of the training text is labelled with the facts it contains', 'Because predicting a proof, a program or an explanation well is easier if you capture the structure behind the text', 'Because engineers add rules for physics and logic by hand once training is finished'],
            answer: 2,
            explain: 'Better prediction is the only goal, but the cheapest way to predict varied text well is to model what lies behind it. How far that goes is debated.',
          },
        ]}
      />

      <Remember
        items={[
          <>An LLM is a <b>function</b>: tokens so far → a probability for every possible next token. It is not a database, a search engine or a rule-based program.</>,
          <>What it knows is <b>implicit in its parameters</b>: billions of numbers tuned by training so that real text became likely. No sentence is stored as a sentence.</>,
          <>Because it generates rather than retrieves, it handles <b>rewording and brand-new requests</b>.</>,
          <>For the same reason it <b>always answers</b>. It has no “not found”, and it optimises for likely, not for true. That is where made-up answers come from.</>,
          <>Classic systems are exact and brittle. LLMs are flexible and inexact. <b>Good products combine them.</b></>,
        ]}
      />

      <RealLLM>
        <h3>Why is next-token prediction enough?</h3>
        <p>This is the question that should be nagging you. Guessing the next word sounds like autocomplete. How does that produce working code, or a correct explanation of why the sky is blue?</p>
        <p>Consider what it takes to be <em>good</em> at the guessing game on different kinds of text:</p>
        <ul>
          <li>To predict the next token of <span className="mono">23 + 58 = </span> across millions of such lines, memorising fails. Something like addition has to be captured.</li>
          <li>To predict the next line of a Python function, it pays to track which variables exist and what the function is for.</li>
          <li>To predict the end of “The ball was dropped from the tower. After two seconds it had fallen about…”, it pays to have captured some regularity about falling objects.</li>
          <li>To predict the last page of a detective story, it pays to have tracked who was where.</li>
        </ul>
        <p>The training goal never mentions arithmetic, code or physics. But text is produced by a world with structure, and the most efficient way to predict the text is to capture some of that structure in the numbers.</p>
        <Callout kind="established">
          What is solid: training only on next-token prediction, at sufficient scale, produces models that translate, write code, solve many exam problems and follow novel instructions. Prediction error falls smoothly and predictably as models, data and compute grow. Nobody programs those abilities in.
        </Callout>
        <Callout kind="research">
          What is debated: <em>how much</em> of this is modelling underlying regularities, and how much is sophisticated pattern-matching over an enormous training set. “Prediction forces the model to learn how the world works” is the mainstream explanation, and there is evidence for it (for instance, models trained only on game moves have been found to represent the board internally). But models also fail in ways a system with robust understanding would not, such as breaking when a familiar problem is slightly reworded. Where the truth sits between those poles is an open research question. This course will show you the mechanism and let you hold the question honestly.
        </Callout>
        <ToyVsReal
          toy={<ul><li>The last lesson’s hand-written probability table</li><li>Soon: a model with a few thousand parameters that you train yourself on tiny text</li><li>It learns spelling and short-range word patterns</li></ul>}
          real={<ul><li>The same function signature: tokens in, probabilities out</li><li>Billions to trillions of parameters, trained on trillions of tokens</li><li>Captures grammar, facts, styles, code structure and some multi-step reasoning</li></ul>}
        />
        <p><b>Where next:</b> you now have the two big ideas: the loop, and the function inside it. <a href="#/lesson/course-map">The next lesson</a> is the map of how we will build both, piece by piece.</p>
      </RealLLM>
    </Lesson>
  )
}
