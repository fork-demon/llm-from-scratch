import { RepoRunner } from '../components/RepoRunner'
import { Diagnostic } from '../components/Diagnostic'
import { DIAGNOSTICS } from '../data/diagnostics'
import { Lesson, Why, Problem, MentalModel, TryIt, Numbers, TheMath, CodeIt, BreakIt, Exercises, CheckYourself, Remember, RealLLM } from '../components/lesson'
import { Callout, DeepDive, Equation, G, Term, ToyVsReal, WhyExists } from '../components/ui'
import { Code } from '../components/Code'
import { CodeExercise } from '../components/python'
import { Exercise, ExplainBack } from '../components/exercise'
import { VectorPlayground } from '../interactive/VectorPlayground'

export default function VectorsLesson() {
  return (
    <Lesson id="vectors">
      <Why>
        <Diagnostic id="diag-math" part={DIAGNOSTICS["math"].part} questions={DIAGNOSTICS["math"].questions} />
        <p className="lede">Friday night in the flat. Dev wants an action film. Riya wants a comedy. They have been scrolling for twenty minutes.</p>
        <p>Amma is on a video call from Mysuru, propped against the fruit bowl. She has been listening. “Give every type of film a mark from minus one to plus one,” she says. “Each of you. Then compare the marks one by one and add up. Then you will know how much your tastes agree.”</p>
        <p>Dev laughs. Riya does not, because she has realised something. Her taste in films is now a list of numbers. So is Dev’s. And Kabir said on Wednesday that inside the model, every word is a list of numbers too.</p>
        <p>You type “What is a cat?”. Within a millisecond, the word “cat” stops being text. It becomes a list of numbers. So does “kitten”. So does “truck”. So does every document a search system might fetch for you.</p>
        <p>That leaves the model with one question it asks millions of times for every word it writes: <b>do these two lists of numbers agree?</b></p>
        <div className="grid-3">
          <div className="card"><span className="chip acc">attention</span><p style={{ marginTop: 8 }}>“Which earlier words matter for this word?” Each candidate gets an agreement score.</p></div>
          <div className="card"><span className="chip acc">similarity search</span><p style={{ marginTop: 8 }}>“Which stored document matches this question?” Same score, different lists.</p></div>
          <div className="card"><span className="chip acc">next-word scores</span><p style={{ marginTop: 8 }}>“How well does each possible next word fit here?” One agreement score per word.</p></div>
        </div>
        <p>All three are the same tiny operation, called the <b>dot product</b>. This lesson is about that one operation. There are only four maths ideas in this whole course, and this is the one you will meet most often.</p>
        <p className="muted">This is not a maths course. We only pick up a tool when the LLM needs it, and every tool comes with a picture and an experiment before it gets a formula.</p>
      </Why>

      <Problem>
        <p>In <a href="#/lesson/prompt-to-answer">Part 0</a> you saw that an LLM is a function that turns numbers into numbers.</p>
        <p>So any “thinking” it does about similarity has to be arithmetic. Riya’s first instinct, as a programmer, is an equality check. That does not get far.</p>
        <WhyExists
          problem="Given two lists of numbers, produce a single number that says how much they agree."
          naive={<>Compare them like a programmer would: <code>a == b</code>, or count how many slots are equal.</>}
          fails="Two learned lists are never exactly equal. We need a graded answer (a lot, a little, not at all, the opposite), and it has to be cheap enough to run millions of times per word."
          idea="Multiply the lists slot by slot and add everything up. Matching signs push the total up, clashing signs pull it down."
          tradeoff="A list with big numbers gets a big score just for being big. We fix that at the end of this lesson, with cosine similarity."
        />
      </Problem>

      <MentalModel>
        <p>First, the thing itself. You have used it for years under another name.</p>
        <Term
          name="Vector"
          plain={<>A list of numbers. That is all. You would call it an array, a row, or a record.</>}
          example={<><code>person = [34, 95000, 172]</code> is age, salary, height. A “3-dimensional vector” just means the list has 3 numbers in it.</>}
          formal={<>An ordered list of n real numbers. “Dimension” is a fancy word for n. A 768-dimensional vector is a list of 768 numbers, nothing more exotic.</>}
        />
        <p>A list of <em>two</em> numbers has a bonus: you can draw it. <code>[3, 1]</code> means “3 steps right, 1 step up”.</p>
        <p>Draw an arrow from the origin to that point. Now the list has a <b>direction</b> and a <b>length</b>.</p>
        <p>That picture is the whole intuition. Two arrows can point the same way, at right angles, or in opposite directions. The dot product is a meter for exactly that.</p>
        <Term
          name="Dot product"
          plain={<>A way of asking two vectors: <b>how much do your directions agree?</b> One number comes out. Large positive: they agree. Zero: they have nothing to say about each other. Negative: they oppose.</>}
          example={<>Same direction → big positive. At right angles → exactly 0. Opposite → negative.</>}
          formal={<>Multiply matching slots and add up the products: a · b = a₁b₁ + a₂b₂ + … You will see why that recipe measures agreement in a moment.</>}
        />
        <Callout kind="analogy">
          The film test from Friday night. Riya and Dev each rate genres from −1 (hate) to +1 (love): <code>[action, comedy, romance]</code>. Multiply their ratings genre by genre.
          <br /><br />
          Where both love a genre, the product is positive. Where both hate it, negative × negative is <em>also</em> positive. Where one loves what the other hates, the product is negative. Add the products and you get a taste-agreement score.
          <br /><br />
          Where the analogy stops: in a real model nobody gives the slots a meaning like “comedy”. The numbers are learned, and a single slot usually means nothing a human could name. The arithmetic is the same, though.
        </Callout>
      </MentalModel>

      <TryIt title="Drag two arrows and watch the agreement meter">
        <p>Try the five starting positions first. Then grab the tips yourself. The exact recipe is spelled out next to the picture (below it on a phone), so you can check every number.</p>
        <VectorPlayground />
        <p>Three things to notice before moving on:</p>
        <ul>
          <li><b>Same direction:</b> large and positive. <b>Perpendicular:</b> exactly 0. <b>Opposite:</b> negative.</li>
          <li>In “short vs long” both arrows point the same way, but making B longer makes the dot product bigger. The dot product mixes up <em>direction</em> and <em>length</em>.</li>
          <li>The cosine similarity line does not care about length. It stays at 1.00 however long you make the arrows.</li>
        </ul>
      </TryIt>

      <Numbers>
        <p>Here is the playground’s starting position by hand. <code>A = [3, 1]</code> and <code>B = [2, 4]</code>.</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>step</th><th>calculation</th><th>result</th></tr></thead>
            <tbody>
              <tr><td>first slots</td><td className="mono">3 × 2</td><td className="mono">6</td></tr>
              <tr><td>second slots</td><td className="mono">1 × 4</td><td className="mono">4</td></tr>
              <tr><td><b>dot product</b>: add them</td><td className="mono">6 + 4</td><td className="mono"><b>10</b></td></tr>
              <tr><td>length of A (Pythagoras)</td><td className="mono">√(3² + 1²) = √10</td><td className="mono">3.16</td></tr>
              <tr><td>length of B</td><td className="mono">√(2² + 4²) = √20</td><td className="mono">4.47</td></tr>
              <tr><td><b>cosine similarity</b></td><td className="mono">10 ÷ (3.16 × 4.47)</td><td className="mono"><b>0.71</b></td></tr>
            </tbody>
          </table>
        </div>
        <p>Positive, so they agree. 0.71 out of a possible 1, so they agree <em>partly</em>: the arrows are 45° apart.</p>

        <h3>Nothing changes with more numbers</h3>
        <p>Three slots, from the course repository. Think of them as made-up “meaning” numbers for three words:</p>
        <div className="table-scroll">
          <table className="plain mono" style={{ fontSize: 14 }}>
            <thead><tr><th>pair</th><th>slot by slot</th><th>sum</th></tr></thead>
            <tbody>
              <tr><td>cat = [0.9, 0.8, −0.5]<br />dog = [0.8, 0.9, −0.4]</td><td>0.9×0.8 + 0.8×0.9 + (−0.5)×(−0.4)<br />= 0.72 + 0.72 + 0.20</td><td><b>+1.64</b></td></tr>
              <tr><td>cat = [0.9, 0.8, −0.5]<br />truck = [−0.6, −0.8, 0.9]</td><td>0.9×(−0.6) + 0.8×(−0.8) + (−0.5)×0.9<br />= −0.54 − 0.64 − 0.45</td><td><b>−1.63</b></td></tr>
            </tbody>
          </table>
        </div>
        <p>You cannot draw a 3-number arrow on this page, and you certainly cannot draw a 768-number one. It does not matter. The recipe and its meaning are the same: <b>agree, unrelated, oppose</b>.</p>

        <h3>Why divide out the lengths?</h3>
        <p>Imagine a search. Your question is <code>q = [2, 1]</code>. Two documents compete:</p>
        <div className="table-scroll">
          <table className="plain mono" style={{ fontSize: 14 }}>
            <thead><tr><th>document</th><th>q · doc</th><th>cosine</th></tr></thead>
            <tbody>
              <tr><td>A = [2, 1] (same direction as q)</td><td>4 + 1 = 5</td><td><b>1.00</b></td></tr>
              <tr><td>B = [4, −2] (a different direction, but long)</td><td>8 − 2 = <b>6</b></td><td>0.60</td></tr>
            </tbody>
          </table>
        </div>
        <p>By raw dot product, B wins. But A is a <em>perfect</em> match in direction. B only won by being long.</p>
        <p>A long vector should not win just by being long. So we divide the dot product by both lengths. What is left is pure direction. That is <G t="cosine">cosine similarity</G>.</p>
      </Numbers>

      <TheMath>
        <p>You have done all of this already. Here it is in the notation papers use.</p>
        <Equation
          label="a dot b equals the sum over i of a i times b i"
          symbols={[
            [<>a, b</>, 'two vectors with the same number of slots, n'],
            [<>a<sub>i</sub></>, <>the number in slot i of a (so a<sub>1</sub> is the first number)</>],
            ['Σ', <>“add up, for every slot i”. It is a <code>for</code> loop with <code>+=</code></>],
            ['a · b', 'one single number: the agreement score'],
          ]}
        >
          a · b = Σ<sub>i</sub> a<sub>i</sub> b<sub>i</sub> = a<sub>1</sub>b<sub>1</sub> + a<sub>2</sub>b<sub>2</sub> + … + a<sub>n</sub>b<sub>n</sub>
        </Equation>
        <Equation
          label="cosine similarity equals a dot b divided by length of a times length of b"
          symbols={[
            ['‖a‖', <>the length of a, also called its <b>norm</b>: √(a<sub>1</sub>² + a<sub>2</sub>² + …). Pythagoras, for any number of slots</>],
            ['cos(a, b)', 'always between −1 (exact opposites) and +1 (exactly the same direction); 0 means perpendicular'],
          ]}
        >
          cos(a, b) = (a · b) / (‖a‖ ‖b‖)
        </Equation>
        <DeepDive title="Why does “multiply and add” measure direction at all?">
          <p>There is a second formula for the same number: a · b = ‖a‖ × ‖b‖ × cos θ, where θ is the angle between the arrows. The cosine of 0° is 1, of 90° is 0, of 180° is −1. That is exactly the agree / unrelated / oppose behaviour you saw.</p>
          <p>Turn on “Show B’s shadow on A” in the playground. The dot product equals the length of A times the length of that shadow. When B swings past 90°, the shadow falls behind A and the sign flips.</p>
          <p>You never need the angle formula to compute anything. It just explains why the cheap recipe has a geometric meaning, and why the normalised version is called <em>cosine</em> similarity.</p>
        </DeepDive>
      </TheMath>

      <CodeIt>
        <p>For Riya, this is the comfortable part. The dot product is a loop you could write in your sleep:</p>
        <Code
          title="the long way"
          setup={`a = [2, 1, 3]
b = [4, 0, 2]`}
          show={`print("a . b =", total)`}
        >{`
total = 0
for i in range(len(a)):
    total += a[i] * b[i]
`}</Code>
        <p>NumPy spells the same thing with the <code>@</code> operator. This is from the repository, including the check that the hand calculation is right:</p>
        <Code
          source="phase1-foundations/math_primer.py"
          title="dot product: by hand vs NumPy"
          setup={`import numpy as np`}
          show={`print("by hand:", by_hand, "  NumPy:", a @ b)`}
        >{`
a = np.array([2, 1, 3])
b = np.array([4, 0, 2])
by_hand = 2 * 4 + 1 * 0 + 3 * 2                       # = 14
assert a @ b == by_hand == 14
`}</Code>
        <p>The similarity meter, with the same cat / dog / truck numbers you worked out above:</p>
        <Code
          source="phase1-foundations/math_primer.py"
          title="agreement, then agreement with lengths divided out"
          setup={`import numpy as np`}
          show={`print(f"cat . dog   = {cat @ dog:+.2f}   cosine = {cosine(cat, dog):+.3f}")
print(f"cat . truck = {cat @ truck:+.2f}   cosine = {cosine(cat, truck):+.3f}")`}
        >{`
cat = np.array([0.9, 0.8, -0.5])
dog = np.array([0.8, 0.9, -0.4])
truck = np.array([-0.6, -0.8, 0.9])
assert cat @ dog > 1.5 and cat @ truck < -1.5

def cosine(x, y):
    return (x @ y) / (np.sqrt((x**2).sum()) * np.sqrt((y**2).sum()))
`}</Code>
        <p><code>np.sqrt((x**2).sum())</code> is the length: square every slot, add, take the root. Running the file prints cosine(cat, dog) = +0.991 and cosine(cat, truck) = −0.929.</p>
        <Callout kind="dev">A dot product is a <code>zip</code>, a <code>map</code> and a <code>sum</code>: <code>sum(x * y for x, y in zip(a, b))</code>. No branches, no memory allocation, the same instruction repeated. That regularity is what lets hardware run billions of them per second.</Callout>
        <RepoRunner path="phase1-foundations/math_primer.py" title="Run math_primer.py in your browser">
          <p>This is the whole file from the repository, running in your browser. Press Run to see what it prints, then edit a copy and change things.</p>
        </RepoRunner>
      </CodeIt>

      <BreakIt>
        <p>Back to the playground. Predict first, then check.</p>
        <ul>
          <li><b>Double B.</b> Set A = [3, 1], B = [1, 2], note the dot product, then set B = [2, 4]. What happens to the dot product? To the cosine? (Dot product doubles from 5 to 10. Cosine does not move.)</li>
          <li><b>Set A to [0, 0].</b> The dot product is 0 with everything. An arrow with no length has no direction, so cosine similarity is not defined for it (dividing by 0). Real code has to guard against this.</li>
          <li><b>Swap A and B.</b> Does A · B equal B · A? (Yes. Multiplication does not care about order, so neither does the sum.)</li>
          <li><b>Dot A with itself.</b> Put B on top of A at [3, 1]. You get 10, which is 3² + 1²: the length squared. A vector always agrees with itself.</li>
        </ul>
      </BreakIt>

      <Exercises>
        <CodeExercise id="vectors-code-dot" />
        <CodeExercise id="vectors-code-cosine" />
        <Exercise
          id="vectors-calc"
          type="calculate"
          title="A dot product by hand"
          answer={{ value: 2, tolerance: 0.001 }}
          answerLabel="a · b"
          hints={['Multiply first slot with first slot, second with second.', '3 × 4 = 12 and (−2) × 5 = −10.', 'Add the two products: 12 + (−10).']}
          solution={<><p>3×4 + (−2)×5 = 12 − 10 = <b>2</b>.</p><p>Small but positive: the two arrows agree a little. The big positive product from the first slot is almost cancelled by the clash in the second slot.</p></>}
        >
          <p>Compute <code>a · b</code> for <code>a = [3, −2]</code> and <code>b = [4, 5]</code>. Then set the playground to those values to check the picture matches your number.</p>
        </Exercise>

        <Exercise
          id="vectors-zero"
          type="experiment"
          title="Make the dot product exactly 0"
          hints={['Exactly 0 means perpendicular. Turn B until the angle reads 90°.', 'You need 2 × Bx + 3 × By = 0. So the two products must cancel.', 'Try swapping A’s two numbers and flipping one sign.']}
          solution={<><p>B = [3, −2] works: 2×3 + 3×(−2) = 6 − 6 = 0. So do [−3, 2], [1.5, −1] and [−4.5, 3]: every arrow on that same line through the origin.</p><p>The trick “swap the two numbers and negate one” always gives a perpendicular arrow in 2D. Notice that the <em>length</em> of B made no difference. Zero agreement is about direction only.</p></>}
        >
          <p>In the playground, set <code>A = [2, 3]</code>. Without using the presets, find a B (not [0, 0]) that makes <code>A · B</code> exactly 0. Then find a second, different B that also works. What do all the answers have in common?</p>
        </Exercise>

        <Exercise
          id="vectors-sign"
          type="predict"
          title="Predict the sign without calculating"
          answer={{ text: ['negative', '-', 'minus', 'neg', 'oppose', 'below zero'] }}
          answerLabel="positive, zero or negative?"
          hints={['Look at the signs slot by slot. Do the big numbers agree in sign, or clash?', 'Slot 1: positive × negative. Slot 2: negative × positive. Both clash, and both are large.', 'Slot 3 is a positive product, but 0.1 × 2 is tiny. It cannot rescue the total.']}
          solution={<><p><b>Negative.</b> 5×(−3) + (−4)×6 + 0.1×2 = −15 − 24 + 0.2 = −38.8.</p><p>You can read a dot product’s sign by looking at where the <em>large</em> entries are and whether their signs match. Small entries barely vote. This habit is useful later for reading attention scores.</p></>}
        >
          <p><code>a = [5, −4, 0.1]</code> and <code>b = [−3, 6, 2]</code>. Without computing the exact value: is <code>a · b</code> positive, zero or negative?</p>
        </Exercise>

        <Exercise
          id="vectors-4d"
          type="calculate"
          title="Four dimensions: nothing changes"
          answer={{ value: 3, tolerance: 0.001 }}
          answerLabel="a · b"
          hints={['Four slots means four products. Same recipe.', '1×3 = 3, 2×1 = 2, 0×5 = 0, (−1)×2 = −2.', 'Add them: 3 + 2 + 0 − 2.']}
          solution={<><p>3 + 2 + 0 − 2 = <b>3</b>.</p><p>Notice the third slot: b has a 5 there, its biggest number, but a has 0, so it contributes nothing. A zero in one vector means “I have no opinion about this slot”.</p><p>We cannot draw these two arrows, but every word still applies. For instance <code>[1, 0, 1, 0] · [0, 1, 0, 1] = 0</code>: those two are perpendicular in 4D, even though nobody can picture it. Beyond two or three dimensions we stop drawing and trust the arithmetic.</p></>}
        >
          <p><code>a = [1, 2, 0, −1]</code> and <code>b = [3, 1, 5, 2]</code>. Compute <code>a · b</code>.</p>
        </Exercise>

        <Exercise
          id="vectors-implement"
          type="implement"
          title="Run it, then break it"
          hints={['python phase1-foundations/math_primer.py prints every check in this lesson under sections 1 and 2.', 'Change by_hand = 2 * 4 + 1 * 0 + 3 * 2 to a wrong number and rerun. Which line stops the program?', 'For the second part: scale a vector with 10 * dog. The dot product is linear in each vector, the cosine ignores length.']}
          solution={<><p>With a wrong <code>by_hand</code> the <code>assert</code> on the next line fails with an AssertionError: the maths in this course is unit-tested.</p><p><code>cat @ (10 * dog)</code> is 16.4, ten times 1.64. <code>cosine(cat, 10 * dog)</code> is still 0.991. Length scales the dot product and leaves the cosine alone, which is exactly what you saw with the arrows.</p></>}
        >
          <p>Run <code>math_primer.py</code>. Break the hand calculation on purpose and watch the assert catch it. Then, before running: what will <code>cat @ (10 * dog)</code> and <code>cosine(cat, 10 * dog)</code> print?</p>
        </Exercise>

        <ExplainBack
          id="vectors-explain"
          prompt="Dev asks: “Why would multiplying two lists slot by slot and adding tell you anything about similarity?” Explain it in plain words, without using the word cosine."
          modelAnswer={<p>Each slot casts a vote. If both lists have the same sign in a slot, their product is positive, a vote for “we agree”. If the signs clash, the product is negative, a vote against. Bigger numbers cast bigger votes, and a zero abstains. Adding the products is counting the votes. Lists that are large in the same places and with the same signs get a big positive total. Lists that are large in opposite ways get a big negative total. Lists whose votes cancel out get roughly zero: knowing one tells you nothing about the other.</p>}
        />
      </Exercises>

      <CheckYourself
        questions={[
          {
            q: 'Two vectors have a dot product of exactly 0. Neither is all zeros. What does that tell you?',
            options: ['They are identical', 'They are perpendicular: neither direction says anything about the other', 'They point in opposite directions', 'One of them must be very short'],
            answer: 1,
            explain: 'Opposite directions give a negative number, not zero. Zero means the agreeing and clashing slots cancel exactly, which in the picture is a right angle.',
          },
          {
            q: 'You double every number in vector b. What happens to a · b and to the cosine similarity?',
            options: ['Both double', 'Both stay the same', 'The dot product doubles, the cosine stays the same', 'The dot product stays the same, the cosine doubles'],
            answer: 2,
            explain: 'Every product doubles, so the sum doubles. But b’s length also doubles, and cosine divides that back out. Cosine only sees direction.',
          },
          {
            q: 'Why might a search system prefer cosine similarity over the raw dot product?',
            options: ['Cosine is faster to compute', 'So that a vector cannot win just by having large numbers', 'Because dot products cannot be negative', 'Because cosine works in more than 3 dimensions and dot products do not'],
            answer: 1,
            explain: 'The raw dot product mixes direction and length. Dividing out the lengths leaves only “do they point the same way?”. Both work in any number of dimensions, and cosine is slightly more work, not less.',
          },
          {
            q: 'A model compares two vectors that each have 4,096 numbers. Compared with your 2D arrows, what is different about the dot product?',
            options: ['It needs a different formula in high dimensions', 'It returns 4,096 numbers instead of one', 'Only the amount of arithmetic: 4,096 products added into one number', 'It can no longer be negative'],
            answer: 2,
            explain: 'Same recipe, same meaning (agree, unrelated, oppose), one number out. We just cannot draw it.',
          },
        ]}
      />

      <Remember
        items={[
          <>A <b>vector</b> is a list of numbers. With two numbers you can draw it as an arrow: it has a direction and a length.</>,
          <>The <b>dot product</b> asks “how much do these two directions agree?”. Same way: large positive. Right angle: 0. Opposite: negative.</>,
          <>The recipe is <b>multiply matching slots, add up</b>. It is identical in 2, 4 or 4,096 dimensions. Only the drawing stops working.</>,
          <><b>Cosine similarity</b> is the dot product with both lengths divided out, always from −1 to +1, so a long vector cannot win just by being long.</>,
          <>Attention scores, similarity search and next-word scores are all dot products. You will meet this operation in almost every remaining lesson.</>,
        ]}
      />

      <RealLLM>
        <ToyVsReal
          toy={<ul><li>2 numbers per vector, so we can draw arrows</li><li>Numbers picked by hand</li><li>One dot product at a time</li></ul>}
          real={<ul><li>Hundreds to thousands of numbers per vector</li><li>Every number learned from data; single slots rarely have a nameable meaning</li><li>Millions of dot products per generated token (billions of individual multiplications), grouped into matrix multiplies (next lesson)</li></ul>}
        />
        <Callout kind="established">
          Three places where a real LLM system computes exactly this operation:
          <ul>
            <li><b><a href="#/lesson/attention">Attention</a>:</b> the score between two tokens is the dot product of one token’s <span className="q">query</span> vector with the other’s <span className="k">key</span> vector.</li>
            <li><b>The output layer:</b> the raw score (<G t="logits">logit</G>) for each possible next token is the dot product of the model’s final vector with a learned vector for that token.</li>
            <li><b><a href="#/lesson/rag">Retrieval</a>:</b> a <G t="vector-db">vector database</G> ranks documents by dot product, cosine similarity or a closely related distance between the question’s vector and each document’s vector.</li>
          </ul>
        </Callout>
        <Callout kind="model">“Each slot is a feature like comedy or romance” is a simplification to build intuition. In trained models, meaning is spread across many slots at once. You will look at real learned vectors in <a href="#/lesson/embeddings">Embeddings</a>.</Callout>
        <p>Keep the two meters apart: the dot product is not the cosine. Attention and the output layer use the raw dot product, so there a vector’s length does count, and the model is free to use it.</p>
        <p>Dividing the lengths out is a choice. Similarity search often makes it; attention does not.</p>
        <p>As for Friday night: Riya and Dev’s agreement score came out negative. They watched separate films, on separate laptops, in the same room.</p>
      </RealLLM>
    </Lesson>
  )
}
