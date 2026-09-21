import { Lesson, Why, Problem, MentalModel, TryIt, Numbers, TheMath, CodeIt, BreakIt, Exercises, CheckYourself, Remember, RealLLM } from '../components/lesson'
import { Callout, DeepDive, Equation, G, Term, ToyVsReal, WhyExists } from '../components/ui'
import { Code } from '../components/Code'
import { CodeExercise } from '../components/python'
import { Exercise, ExplainBack } from '../components/exercise'
import { MatrixPlayground } from '../interactive/MatrixPlayground'

export default function MatricesLesson() {
  return (
    <Lesson id="matrices">
      <Why>
        <p className="lede">Open the source code of any LLM and search for the character <code>@</code>.</p>
        <p>You will find lines like <code>x @ W</code>, <code>Q @ K.T</code> and <code>weights @ V</code> everywhere. Delete them and almost nothing is left. An LLM is, to a surprising degree, a long chain of one operation: the <b>matrix multiply</b>.</p>
        <p>That sounds like bad news if you are not a maths person. It is good news. In the <a href="#/lesson/vectors">last lesson</a> you learned the dot product: one question, one answer. A matrix multiply is nothing new. It is <b>many dot products at once</b>, arranged in a grid.</p>
        <Callout kind="idea">
          One dot product asks one question of one vector. A model needs to ask thousands of questions of thousands of tokens, at every layer. A matrix multiply is how you write, and run, all of those at once.
        </Callout>
      </Why>

      <Problem>
        <p>Suppose a layer wants to compute 64 different scores for each of 1,000 token vectors. Every score is a dot product. That is 64,000 dot products.</p>
        <WhyExists
          problem="Run many scoring formulas over many vectors."
          naive="Two nested for loops, calling dot() 64,000 times."
          fails="In Python that is painfully slow. Worse, it hides something important: none of those 64,000 dot products depends on any other. They could all run at the same time, but a loop says “one after another”."
          idea="Stack the vectors into one grid, stack the formulas into another grid, and define a single operation that means “dot everything on the left with everything on the right”."
          tradeoff="You have to keep track of the shapes of the grids. Shape mismatches are the most common bug in ML code. The good news: they behave exactly like type errors."
        />
      </Problem>

      <MentalModel>
        <Term
          name="Matrix"
          plain={<>A grid of numbers. If a vector is one row of a spreadsheet, a matrix is the whole sheet. Equally: a stack of vectors.</>}
          example={<>Three tokens, each described by 2 numbers: a grid with 3 rows and 2 columns. We say its <b>shape</b> is <code>(3, 2)</code>. Always rows first.</>}
          formal={<>A rectangular array of numbers with n rows and m columns, shape (n, m). In code: a 2D array.</>}
        />
        <p>You will read matrices in two ways, and switch between them constantly:</p>
        <div className="grid-2">
          <div className="card">
            <h4 style={{ fontSize: 17, marginBottom: 6 }}>As data: a stack of records</h4>
            <p>Each row is one vector. A sentence of T tokens, each a vector of D numbers, is a <code>(T, D)</code> matrix.</p>
          </div>
          <div className="card">
            <h4 style={{ fontSize: 17, marginBottom: 6 }}>As a machine: a stack of questions</h4>
            <p>Each row is one scoring formula. Feed a vector in, and every row asks its own question of it. One answer per row comes out.</p>
          </div>
        </div>

        <h3>Matrix × vector: several dot products at once</h3>
        <p>A house is described by <code>x = [2, 1, 1]</code>: bedrooms, bathrooms, garages. We have two scoring formulas, one per row:</p>
        <div className="table-scroll">
          <table className="plain mono" style={{ fontSize: 14 }}>
            <thead><tr><th>row of W</th><th>its question</th><th>row · x</th><th>answer</th></tr></thead>
            <tbody>
              <tr><td>[1, 0, 2]</td><td style={{ fontFamily: 'var(--sans)' }}>a “price” formula</td><td>1×2 + 0×1 + 2×1</td><td><b>4</b></td></tr>
              <tr><td>[0, 3, 1]</td><td style={{ fontFamily: 'var(--sans)' }}>an “upkeep” formula</td><td>0×2 + 3×1 + 1×1</td><td><b>4</b></td></tr>
            </tbody>
          </table>
        </div>
        <p>So <code>W @ x = [4, 4]</code>. Two questions asked of the same vector, two answers. W has shape (2, 3), x has 3 numbers, the output has 2. A layer with 64 outputs is a matrix with 64 rows: 64 questions asked at once.</p>

        <h3>Matrix × matrix: every row with every column</h3>
        <p>Now put <em>several</em> vectors on the right-hand side, standing up as columns. The rule:</p>
        <Callout kind="idea">In <code>A @ B</code>, the number in row i, column j of the result is <b>(row i of A) · (column j of B)</b>. Every row of A gets dotted with every column of B. That is the entire definition.</Callout>
        <Callout kind="analogy">
          Think of a questionnaire. Each row of A is a question, each column of B is a respondent, and the result is the full table of answers: one cell per (question, respondent) pair.
          <br /><br />
          Where the analogy stops: in a trained model the “questions” are learned numbers. Nobody wrote them, and most of them do not translate into a question a human would ask.
        </Callout>
      </MentalModel>

      <TryIt title="Find the dot product hiding in every cell">
        <p>Point at a result cell. Before reading the sum, try to say which row and which column it came from.</p>
        <MatrixPlayground />
      </TryIt>

      <Numbers>
        <p>The playground’s starting example, in full. It takes two minutes by hand, and the rhythm (row, column, row, column) is worth getting into your fingers.</p>
        <Code lang="text" title="A is (2, 2), B is (2, 3): inner numbers 2 and 2 match, the result is (2, 3)">{`
A = [[1, 2],          B = [[1, 0, 1],
     [3, 0]]               [2, 1, 0]]

cell (0,0): row 0 of A · column 0 of B = [1,2]·[1,2] = 1×1 + 2×2 = 5
cell (0,1): row 0 of A · column 1 of B = [1,2]·[0,1] = 0 + 2     = 2
cell (0,2): row 0 of A · column 2 of B = [1,2]·[1,0] = 1 + 0     = 1
cell (1,0): row 1 of A · column 0 of B = [3,0]·[1,2] = 3 + 0     = 3
cell (1,1): row 1 of A · column 1 of B = [3,0]·[0,1] = 0 + 0     = 0
cell (1,2): row 1 of A · column 2 of B = [3,0]·[1,0] = 3 + 0     = 3

A @ B = [[5, 2, 1],
         [3, 0, 3]]
`}</Code>
        <p>Six dot products, arranged in a grid. Nothing else happened.</p>

        <h3>The shape rule</h3>
        <p>Row i of A and column j of B must have the same length, or the dot product cannot be formed. Rows of A are as long as A has columns. Columns of B are as tall as B has rows. So:</p>
        <div className="card center mono" style={{ fontSize: 18 }}>
          (n, <b className="acc">k</b>) @ (<b className="acc">k</b>, m) → (n, m)
          <div style={{ fontFamily: 'var(--sans)', fontSize: 15, marginTop: 6 }} className="muted">The inner numbers must match. The outer numbers are the answer.</div>
        </div>

        <h3>A tiny “layer”: three tokens at once</h3>
        <p>In LLM code the data usually stands on the left: one row per token. Here are 3 tokens with 2 numbers each, pushed through a weight matrix that turns 2 numbers into 3.</p>
        <Code lang="text" title="(3, 2) @ (2, 3) -> (3, 3): all three tokens transformed in one multiply">{`
X = [[1, 0],      W = [[2, 0, 1],      X @ W = [[2, 0, 1],    <- token 0
     [0, 1],           [1, 3, 0]]               [1, 3, 0],    <- token 1
     [1, 1]]                                    [3, 3, 1]]    <- token 2
`}</Code>
        <p>Check token 2: <span className="mono">[1, 1] · [2, 1] = 3</span>, <span className="mono">[1, 1] · [0, 3] = 3</span>, <span className="mono">[1, 1] · [1, 0] = 1</span>. Each row of the result depends only on the matching row of X. The tokens do not mix: the same transformation is applied to each of them, independently, in one go.</p>
        <p className="muted">Written this way round, each <em>column</em> of W is one question and each row of X is one token being asked. Same dot products as before, just laid out sideways.</p>
      </Numbers>

      <TheMath>
        <Equation
          label="C i j equals the sum over t of A i t times B t j"
          symbols={[
            ['A, B', 'matrices of shape (n, k) and (k, m)'],
            [<>C<sub>ij</sub></>, 'the number in row i, column j of the result C = A @ B, which has shape (n, m)'],
            [<>A<sub>it</sub></>, 'row i of A, slot t'],
            [<>B<sub>tj</sub></>, 'column j of B, slot t'],
            [<>Σ<sub>t</sub></>, <>add up over the k shared slots: this sum <em>is</em> the dot product from the last lesson</>],
          ]}
        >
          C<sub>ij</sub> = Σ<sub>t</sub> A<sub>it</sub> B<sub>tj</sub> = (row i of A) · (column j of B)
        </Equation>

        <h3>The transpose: flip the spreadsheet</h3>
        <p>One more tool, and it is purely about layout. The <b>transpose</b> turns rows into columns. It is written A<sup>T</sup> in papers and <code>A.T</code> in code. A (2, 3) matrix becomes (3, 2). No number changes, only where it sits.</p>
        <Code lang="text">{`
A = [[1, 2, 3],        A.T = [[1, 4],
     [4, 5, 6]]               [2, 5],
                              [3, 6]]
`}</Code>
        <p>Why would you want that? Take the three token vectors X from above and ask: <em>how similar is every token to every other token?</em> Similarity is a dot product, so we want all 9 pairs.</p>
        <p><code>X @ X</code> is (3, 2) @ (3, 2). Inner numbers 2 and 3: impossible. But matrix multiply dots rows of the left with <b>columns</b> of the right, and our tokens are sitting in rows. Flip the right-hand copy:</p>
        <Code lang="text" title="(3, 2) @ (2, 3) -> (3, 3): cell (i, j) is token i . token j">{`
X @ X.T = [[1, 0, 1],
           [0, 1, 1],
           [1, 1, 2]]
`}</Code>
        <p><code>something @ something_else.T</code> is how you spell “dot every row of this with every row of that”. It is the heart of <a href="#/lesson/attention">attention</a>, where it appears as <code>Q @ K.T</code>.</p>

        <DeepDive title="Two facts about matrix multiplication worth pinning to the wall">
          <p><b>Order matters.</b> A @ B is usually not B @ A. Often B @ A does not even have a legal shape. Think of function composition: “resize the image, then compress it” is not “compress, then resize”.</p>
          <p><b>Grouping does not matter.</b> (A @ B) @ C always equals A @ (B @ C). You can regroup a chain of multiplies however is convenient. Backpropagation quietly relies on this.</p>
        </DeepDive>
      </TheMath>

      <CodeIt>
        <p>Here is the definition as code: two loops choose a result cell, and the body is one dot product.</p>
        <Code title="matrix multiply, the long way">{`
out = np.zeros((n, m))
for i in range(n):            # every row of A
    for j in range(m):        # every column of B
        out[i, j] = A[i, :] @ B[:, j]
`}</Code>
        <p>And this is what everybody actually writes:</p>
        <Code title="the same thing">{`
out = A @ B
`}</Code>
        <Callout kind="dev">
          A matrix multiply is a doubly nested loop of dot products (three loops, if you count the one hiding inside the dot product). <code>A @ B</code> is that loop, vectorised. The key property: no iteration reads anything another iteration wrote. Every cell can be computed independently, so they can all be computed <em>simultaneously</em>. That one fact is why GPUs, which are machines for doing thousands of identical multiply-adds at once, made deep learning practical.
        </Callout>
        <p>From the repository, the hand calculation checked by an assert:</p>
        <Code source="phase1-foundations/math_primer.py" title="every row of A dotted with every column of B">{`
A = np.array([[1, 2],
              [3, 0]])
B = np.array([[1, 0, 1],
              [2, 1, 0]])
by_hand = np.array([[5, 2, 1],
                    [3, 0, 3]])     # worked cell-by-cell on the page
assert (A @ B == by_hand).all()
`}</Code>
        <p>And the two patterns you will see in every later lesson: a batch of tokens through one transformation, and all-pairs similarity via the transpose.</p>
        <Code source="phase1-foundations/math_primer.py" title="one multiply, all tokens">{`
tokens = np.array([[1.0, 2.0],      # token 0's vector
                   [3.0, 4.0],      # token 1's vector
                   [5.0, 6.0]])     # token 2's vector
transform = np.array([[1.0, 0.0, 1.0],
                      [0.0, 1.0, 1.0]])
out = tokens @ transform            # (3,2) @ (2,3) -> (3,3): all 3 tokens at once
`}</Code>
        <Code source="phase1-foundations/math_primer.py" title="the Q @ K.T pattern">{`
pairwise = X @ X.T                    # (3,2) @ (2,3) -> (3,3)
assert pairwise[0, 2] == X[0] @ X[2]  # cell (i,j) is exactly token_i . token_j
`}</Code>
      </CodeIt>

      <BreakIt>
        <p>In the playground, predict first, then check.</p>
        <ul>
          <li><b>Swap A and B</b> from the starting position. You get (2, 3) @ (2, 2). What happens, and why? (Inner numbers 3 and 2 differ. Order matters so much that the reversed multiply does not even exist.)</li>
          <li><b>Make A the “do nothing” matrix:</b> set A to <span className="mono">[[1, 0], [0, 1]]</span>. What is A @ B? (Exactly B. Row 0 of A, [1, 0], asks “what is your first number?”, row 1 asks “what is your second?”.)</li>
          <li><b>Zero out a row of A.</b> Which result cells become 0? (That whole row of the result: a question made of zeros always gets the answer 0.)</li>
          <li><b>Set both to 2 × 3.</b> Read the message. Then press “Transpose B” and look at the result’s shape. Press “Transpose B” again to undo it, and press “Transpose A” instead. Both fit. Do they give the same shape? (No: (2, 2) versus (3, 3). Fitting is not the same as meaning the same thing.)</li>
        </ul>
      </BreakIt>

      <Exercises>
        <CodeExercise id="matrices-code-matmul" />
        <Exercise
          id="matrices-shape"
          type="predict"
          title="Predict the shape"
          answer={{ text: ['(5,3)', '5,3', '5x3', '5×3', '[5,3]'] }}
          answerLabel="shape, e.g. (2,4)"
          hints={['Write the two shapes next to each other: (5, 3) @ (3, 3).', 'The inner numbers are 3 and 3. They match, so the multiply is legal.', 'The outer numbers are the answer: the first number of the left shape and the last number of the right shape.']}
          solution={<><p><b>(5, 3)</b>. Inner numbers 3 and 3 match, outer numbers 5 and 3 are the result.</p><p>Read it as an LLM would use it: 5 tokens, each with 3 numbers, go in. 5 tokens, each with 3 new numbers, come out. The number of tokens never changes in a layer like this, only what each token’s vector contains.</p><p>Two more to try in your head: (4, 8) @ (8, 100) gives (4, 100). And (2, 3) @ (2, 3) fails: inner numbers 3 and 2 differ.</p></>}
        >
          <p><code>X</code> has shape (5, 3) and <code>W</code> has shape (3, 3). What is the shape of <code>X @ W</code>?</p>
        </Exercise>

        <Exercise
          id="matrices-cell"
          type="calculate"
          title="One cell by hand"
          answer={{ value: 19, tolerance: 0.001 }}
          answerLabel="cell (1, 1)"
          hints={['Cell (1, 1) uses row 1 of A and column 1 of B. Counting starts at 0, so those are the second row and the second column.', 'Row 1 of A is [1, 3]. Column 1 of B is [4, 5].', '[1, 3] · [4, 5] = 1×4 + 3×5.']}
          solution={<><p>Row 1 of A is [1, 3], column 1 of B is [4, 5]. 1×4 + 3×5 = 4 + 15 = <b>19</b>.</p><p>The full result is <span className="mono">[[2, 8], [7, 19]]</span>. Set the playground to 2 × 2 and 2 × 2, type the numbers in, and check all four cells.</p></>}
        >
          <p>With <code>A = [[2, 0], [1, 3]]</code> and <code>B = [[1, 4], [2, 5]]</code>, what is the bottom-right cell of <code>A @ B</code>, cell (1, 1)?</p>
        </Exercise>

        <Exercise
          id="matrices-debug"
          type="debug"
          title="The multiply that refuses to run"
          answer={{ text: ['Q @ K.T', 'scores = Q @ K.T', 'K.T', 'Q @ K.transpose()', 'Q @ np.transpose(K)', 'transpose K', 'Q @ K^T', 'QK^T'] }}
          answerLabel="the corrected expression"
          hints={['Write the shapes next to the failing line: (4, 8) @ (4, 8). Which numbers are the inner ones?', 'We want one score per (token, token) pair, so the result should be (4, 4).', 'Which matrix do you have to flip so that the inner numbers become 8 and 8, and the outer ones 4 and 4?']}
          solution={<><p><code>scores = Q @ K.T</code>. Shapes: (4, 8) @ (8, 4) → (4, 4). Cell (i, j) is row i of Q dotted with row j of K, which is what we wanted.</p><p>Careful: <code>Q.T @ K</code> also runs without error: (8, 4) @ (4, 8) → (8, 8). But that dots <em>columns</em> with columns, mixing up tokens, and the result is not one score per pair of tokens. Matching shapes are necessary, not sufficient. Always ask what a row of the result is supposed to mean.</p></>}
        >
          <p>There are 4 tokens. Each has a query vector and a key vector of 8 numbers. We want the score of every token against every token. This crashes. What should the last line be?</p>
          <Code>{`
Q = np.random.randn(4, 8)   # one row per token
K = np.random.randn(4, 8)   # one row per token
scores = Q @ K              # ValueError: ... (size 4 is different from 8)
`}</Code>
        </Exercise>

        <Exercise
          id="matrices-implement"
          type="implement"
          title="Add a token, predict the shape"
          hints={['Find section 5 in phase1-foundations/math_primer.py: the tokens and transform arrays.', 'Add a fourth row to tokens, for example [7.0, 8.0]. Which of the two shapes changed?', 'tokens is now (4, 2). transform is still (2, 3).']}
          solution={<><p>The output becomes (4, 3), and its new last row is <span className="mono">[7, 8, 15]</span>. The first three rows are unchanged: adding a token does not affect the others, because each output row only uses its own input row.</p><p>You did not touch <code>transform</code>. The same weights handle 3 tokens or 3,000. That is why a model with a fixed set of weights can read prompts of different lengths.</p></>}
        >
          <p>In <code>math_primer.py</code>, add a fourth token <code>[7.0, 8.0]</code> to <code>tokens</code>. Before running: what shape will <code>out</code> have, what will its last row be, and do the first three rows change?</p>
        </Exercise>

        <ExplainBack
          id="matrices-explain"
          prompt="A backend developer asks you: “Why do LLMs need GPUs? My CPU can multiply numbers just fine.” Answer using what you know about what a matrix multiply is made of."
          modelAnswer={<p>Almost all the work in an LLM is matrix multiplication, and a matrix multiply is just a huge grid of dot products. Every cell of the result is computed from one row and one column, and no cell depends on any other cell. So there is no need to do them one after another. A CPU has a handful of fast cores and can only work on a few cells at a time. A GPU has thousands of simple cores that can each take a cell and do its multiply-adds at the same time. The maths is the same, the GPU just does the independent pieces in parallel.</p>}
        />
      </Exercises>

      <CheckYourself
        questions={[
          {
            q: 'In A @ B, where does the number in row 2, column 0 of the result come from?',
            options: ['Row 2 of A multiplied slot by slot with row 0 of B', 'Row 2 of A dotted with column 0 of B', 'Column 2 of A dotted with row 0 of B', 'The number A[2][0] times the number B[2][0]'],
            answer: 1,
            explain: 'Rows of the left matrix, columns of the right matrix, combined with a dot product. Every cell follows this rule.',
          },
          {
            q: 'x holds 10 tokens with 16 numbers each. W has shape (16, 64). What comes out of x @ W?',
            options: ['One vector of 64 numbers', '10 tokens, each now described by 64 numbers: shape (10, 64)', 'Shape (16, 16)', 'An error: 10 and 64 do not match'],
            answer: 1,
            explain: '(10, 16) @ (16, 64): the inner 16s match, the outer numbers give (10, 64). The number of tokens is unchanged, each token gets a new vector.',
          },
          {
            q: 'Why is a shape mismatch “impossible” rather than merely “wrong”?',
            options: ['NumPy is strict about style and refuses such code on principle', 'A dot product needs two lists of equal length; here they differ, so no cell can be computed', 'The result would have too many cells to fit in memory', 'Only square matrices can be multiplied with each other'],
            answer: 1,
            explain: 'If rows of A have 3 numbers and columns of B have 2, there is no way to pair the slots up. It is a type error.',
          },
          {
            q: 'What does the transpose do, and why is it everywhere in ML code?',
            options: ['It inverts a matrix, which is how you undo the effect of a layer', 'It rescales the numbers so that the multiplication runs faster on a GPU', 'It flips rows and columns without changing any number, so shapes line up (for example to dot rows with rows)', 'It sorts the rows so that similar vectors end up next to each other'],
            answer: 2,
            explain: 'Matrix multiply uses columns on the right. If your vectors are stored as rows, .T stands them up as columns. Q @ K.T means “every row of Q with every row of K”.',
          },
          {
            q: 'Why can a GPU compute a big matrix multiply so much faster than a simple loop?',
            options: ['GPUs use a different, approximate formula', 'Each result cell is independent of the others, so thousands of cells can be computed at the same time', 'GPUs skip the cells that are zero', 'GPUs store matrices in compressed form'],
            answer: 1,
            explain: 'Same arithmetic, exact same result. The win is parallelism, which is only possible because no cell needs another cell’s answer.',
          },
        ]}
      />

      <Remember
        items={[
          <>A <b>matrix</b> is a grid of numbers: a stack of vectors. Its shape is <b>(rows, columns)</b>, rows first, always.</>,
          <><b>Matrix × vector</b> = one dot product per row. Each row asks its own question of the vector.</>,
          <><b>Matrix × matrix</b>: cell (i, j) = row i of A · column j of B. It is many dot products at once and nothing more.</>,
          <>Shape rule: <b>(n, k) @ (k, m) → (n, m)</b>. Inner numbers must match, outer numbers are the answer. Treat mismatches like type errors.</>,
          <>The <b>transpose</b> flips rows and columns. <code>X @ Y.T</code> means “every row of X dotted with every row of Y”.</>,
        ]}
      />

      <RealLLM>
        <ToyVsReal
          toy={<ul><li>2 × 3 grids of whole numbers</li><li>6 dot products per multiply</li><li>Numbers typed in by you</li></ul>}
          real={<ul><li>In GPT-2 small, one feed-forward weight matrix has shape (768, 3072): 2,359,296 learned numbers</li><li>A prompt of T tokens goes through it as one (T, 768) @ (768, 3072) multiply</li><li>12 layers in GPT-2 small and dozens in larger models, several such matrices per layer, for every token generated</li></ul>}
        />
        <Callout kind="established">
          A layer of a neural network is <code>x @ W</code> (plus a small extra step you will meet in <a href="#/lesson/neurons">Neurons and layers</a>). The <G t="parameters">parameters</G> of an LLM, the billions of numbers that training adjusts, are overwhelmingly the entries of matrices like W. When someone says “a 7-billion-parameter model”, picture a few hundred large spreadsheets.
        </Callout>
        <Callout kind="established">
          All T tokens of a prompt go through each layer together, as rows of one matrix. No loop over tokens. This is why reading your prompt is fast, and why hardware built for parallel multiply-adds (GPUs and similar accelerators) is what LLMs run on.
        </Callout>
        <p>You will see these exact patterns again: <code>x @ W</code> in <a href="#/lesson/neurons">every layer</a>, <code>Q @ K.T</code> in <a href="#/lesson/attention">attention</a>, and a final multiply that produces one score per word in the vocabulary, which is where the <a href="#/lesson/softmax">next lesson</a> picks up.</p>
      </RealLLM>
    </Lesson>
  )
}
