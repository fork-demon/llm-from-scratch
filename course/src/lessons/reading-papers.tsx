import { Lesson, Why, Problem, MentalModel, TryIt, Numbers, TheMath, CodeIt, BreakIt, Exercises, CheckYourself, Remember, RealLLM } from '../components/lesson'
import { Callout, DeepDive, Equation, Term, ToyVsReal, WhyExists } from '../components/ui'
import { Code } from '../components/Code'
import { Exercise, ExplainBack } from '../components/exercise'
import { PaperMap } from '../interactive/PaperMap'
import { NotationDecoder } from '../interactive/NotationDecoder'
import { EncoderDecoder } from '../illustrations/EncoderDecoder'
import { PaperAnatomy } from '../illustrations/PaperAnatomy'
import { READING_PATH } from '../lib/paperMap'
import { lessonById } from '../data/curriculum'

const PAPER = 'https://arxiv.org/abs/1706.03762'

export default function ReadingPapersLesson() {
  return (
    <Lesson id="reading-papers">
      <Why>
        <p className="lede">In June 2017 eight researchers working at Google posted a 15-page paper called <a href={PAPER} target="_blank" rel="noreferrer">“Attention Is All You Need”</a>. Every model in this course descends from it.</p>
        <p>Most developers never open it. The first page has an abstract full of BLEU scores. Page 4 has this:</p>
        <div className="card center" style={{ fontFamily: 'var(--serif)', fontSize: 21 }}>
          head<sub>i</sub> = Attention(QW<sub>i</sub><sup>Q</sup>, KW<sub>i</sub><sup>K</sup>, VW<sub>i</sub><sup>V</sup>)
        </div>
        <p>You wrote that line. It is <code>q, k, v = self.qkv(x).split(D, dim=2)</code> followed by the attention you built in <a href="#/lesson/attention">Part 6</a>. The paper says it in a different language.</p>
        <Callout kind="idea">
          Papers are hard for two reasons that have nothing to do with intelligence: the <b>notation</b> is unfamiliar, and the <b>format</b> hides the important parts. Both can be learned in an afternoon. This lesson gives you a method, a decoder for the notation, and then walks you through the founding paper of the field, section by section.
        </Callout>
        <p>By the end you will have read it. You will also have met the one mechanism in it that you have not built: cross-attention.</p>
      </Why>

      <Problem>
        <p>A paper is not a tutorial. It is an argument addressed to expert reviewers: here is a claim, here is the evidence, here is why earlier work falls short. Teaching you is not its job.</p>
        <WhyExists
          problem="New ideas in this field appear as papers first, often years before a good explanation exists. An applied engineer has to be able to read them."
          naive="Start at the first word and read to the last, understanding each sentence before moving on."
          fails="You spend your energy on the related-work section and on notation before you know what the paper claims. You cannot tell the one equation that matters from the ten that do not. Most people stop on page 3."
          idea="Read in several passes, each with a different goal and a time limit. Find the claim first, then the evidence, and only then the details. Decide after each pass whether the paper deserves the next one."
          tradeoff="The early passes feel like cheating, and you must tolerate not understanding things yet. The last pass is expensive, so you can afford it for a few papers a year."
        />
      </Problem>

      <MentalModel title="A method: three passes">
        <p>The method comes from a two-page note by S. Keshav, “How to Read a Paper” (ACM SIGCOMM Computer Communication Review, 2007). It is worth reading in full. In short:</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>pass</th><th>what you read</th><th>what you can do afterwards</th><th>Keshav’s time</th></tr></thead>
            <tbody>
              <tr><td><b>First</b></td><td>Title, abstract, introduction. Section headings only. Conclusion. A glance at the references.</td><td>Say what kind of paper it is, what it claims, and whether it is relevant to you. Most papers stop here, and that is correct.</td><td>5 to 10 minutes</td></tr>
              <tr><td><b>Second</b></td><td>The whole paper, carefully, but skipping proofs and details. Study the figures and tables. Jot down the key points, and mark what you do not understand and which references to follow up.</td><td>Summarise the main argument, with its evidence, to a colleague.</td><td>up to an hour</td></tr>
              <tr><td><b>Third</b></td><td>Rebuild the work yourself: make the authors’ assumptions and re-create the result, challenging each step.</td><td>Reconstruct the paper from memory and name its weak points.</td><td>four or five hours for a beginner</td></tr>
            </tbody>
          </table>
        </div>
        <p>After the first pass Keshav asks for five answers, the “five Cs”: the paper’s <b>category</b>, its <b>context</b> (what it builds on), whether its assumptions look <b>correct</b>, its <b>contributions</b>, and its <b>clarity</b>.</p>
        <Callout kind="dev">
          The third pass, in Keshav’s words, is to “virtually re-implement the paper”. For you this need not be virtual. You have a working GPT, a NumPy attention and a LoRA layer. For most LLM papers, the third pass is a change to a file you already own.
        </Callout>
        <h3>Where the content hides in a machine-learning paper</h3>
        <p>Keshav wrote for networking research. ML papers have their own habits, so add this to the first and second pass: <b>look at the pictures first</b>. Five places carry most of the information. Here they are in the paper we are about to read, at their real page numbers.</p>
        <PaperAnatomy />
        <ol>
          <li><b>The architecture figure.</b> The authors’ own summary of the method.</li>
          <li><b>The one central equation.</b> A paper may print twenty. Usually one is the idea, and the rest define terms or report settings.</li>
          <li><b>The main results table.</b> The evidence for the claim in the abstract.</li>
          <li><b>The ablation table.</b> Often the most useful part for a practitioner.</li>
          <li><b>The hyperparameters.</b> In an appendix or a training section. You need them to reproduce anything, and they show what the authors had to tune.</li>
        </ol>
        <Term
          name="Ablation"
          plain={<>An experiment that removes or changes one piece of the system and measures what happens. It is how authors show that each piece matters.</>}
          example={<>Table 3 of the Transformer paper: keep everything fixed and vary the number of heads. One head scores 0.9 BLEU below the best setting. Too many heads also scores lower.</>}
          formal={<>A controlled comparison in which exactly one factor differs from the baseline configuration. If two things change at once, the table cannot tell you which one caused the difference.</>}
        />
        <h3>Reading a results table like a sceptic</h3>
        <p>A bold number is a claim, not a fact. Ask, in this order:</p>
        <ul>
          <li><b>What is the baseline, and who ran it?</b> Numbers copied from older papers may come from weaker tuning or less compute.</li>
          <li><b>Is the compute comparable?</b> The Transformer paper is exemplary here: Table 2 has a training-cost column next to the scores.</li>
          <li><b>What was tuned, and on which data?</b> If choices were made by looking at the test set, the score is optimistic.</li>
          <li><b>Are there error bars or repeated runs?</b> Often there are none. Then a gap of a few tenths of a point may be noise. In <a href="#/lesson/evals">Evals</a> you computed how wide such intervals are.</li>
          <li><b>Could the benchmark be in the training data?</b> For LLMs trained on the web, contamination is a standing risk. It is the leak you saw in <a href="#/lesson/evals">Evals</a> and in <a href="#/lesson/training-pipeline">From raw text to assistant</a>, at internet scale.</li>
          <li><b>What is missing?</b> The comparison that was not run, the task where the method lost, the cost nobody mentions.</li>
        </ul>
        <h3>One convention that makes formulas look backwards</h3>
        <p>This course writes a layer as <code>x @ W + b</code>: the vector is a <b>row</b> and stands on the left. Many papers write <b>Wx + b</b>: the vector is a <b>column</b> and stands on the right, so their W has shape (out, in), the transpose of ours.</p>
        <p>Nothing is different except the bookkeeping. But shapes and the order of matrices flip. The LoRA paper writes <i>W</i><sub>0</sub><i>x</i> + <i>BAx</i> with B of shape d × r. Your code computes <code>(x @ A) @ B</code> with B of shape r × d. Same maths. The Transformer paper happens to use rows, like this course: its FFN is max(0, xW₁ + b₁)W₂ + b₂.</p>
        <Callout kind="established">PyTorch straddles both worlds: <code>nn.Linear</code> stores its weight as (out, in), the paper convention, and computes <code>x @ W.T + b</code>, the row convention. You met the consequences in <a href="#/lesson/pytorch-bridge">the previous lesson</a>.</Callout>
      </MentalModel>

      <TryIt title="Decode the notation, then read the paper">
        <p>First the language. Every symbol below maps to a line of code you have already written. When a formula in a paper stops you, translate it into shapes and code, from the inside out.</p>
        <NotationDecoder />
        <h3>The one new mechanism: an encoder, a decoder, and cross-attention</h3>
        <p>Before the guided reading, you need one idea that this course has not taught. The paper’s model is not a GPT. It was built to <b>translate</b>: read a whole source sentence, then write a target sentence. So it has two stacks.</p>
        <EncoderDecoder />
        <p>Follow the numbers.</p>
        <ol>
          <li><b>Encoder self-attention.</b> Queries, keys and values all come from the source sentence. There is <b>no mask</b>. The source is fully known before translation starts, so “small” may look at “the” and “the” may look at “small”. The grid is full.</li>
          <li><b>Decoder masked self-attention.</b> Queries, keys and values all come from the target written so far. The future does not exist yet, so the grid is lower-triangular. This is exactly the attention in your GPT.</li>
          <li><b>Cross-attention.</b> The new part. <span className="q">Queries</span> come from the decoder: each target position asks a question. <span className="k">Keys</span> and <span className="v">values</span> come from the encoder’s output: the source tokens answer. The grid is a rectangle, target length × source length, and needs no mask, because every source token is legitimately visible.</li>
          <li>Then the feed-forward network and the head, and the next target token. It is appended to the target, and the decoder runs again. The encoder does not: its output is computed once.</li>
        </ol>
        <p>The numbers in the grids are illustrative, not from a trained model. They show the typical pattern: when producing “Katze”, the decoder attends mostly to “cat”.</p>
        <Term
          name="Encoder-decoder and cross-attention"
          plain={<>Two Transformers working together. The encoder reads the input and produces one vector per input token. The decoder writes the output, and in every layer it consults those vectors through cross-attention.</>}
          example={<>Source “the cat is small” (4 tokens), target so far “&lt;s&gt; die Katze” (3 tokens). Cross-attention computes a 3 × 4 table of weights: for each target position, how much to take from each source token.</>}
          formal={<>CrossAttention = softmax(Q<sub>dec</sub> K<sub>enc</sub><sup>T</sup> / √d<sub>k</sub>) V<sub>enc</sub>, with Q<sub>dec</sub> = X<sub>dec</sub>W<sup>Q</sup>, K<sub>enc</sub> = Z W<sup>K</sup>, V<sub>enc</sub> = Z W<sup>V</sup>, where Z is the encoder’s output. It is the attention function you know, with its inputs taken from two different sequences.</>}
        />
        <Callout kind="established">
          In your attention lesson the soft-lookup analogy had a weakness: the “dictionary” and the “query” came from the same sentence. Cross-attention is the cleaner case. The encoder output really is a small read-only store built from the source, and the decoder really does query it.
        </Callout>
        <p><b>So why does GPT not need it?</b> A decoder-only model puts everything in one sequence. The prompt plays the role of the source, the answer the role of the target, and ordinary masked self-attention lets every answer token look back at every prompt token. One stack, one kind of attention, one training objective. The price is that prompt tokens cannot look at later prompt tokens.</p>
        <h3>The guided reading</h3>
        <p>Now open the paper next to this page and work through it. For each section: read the paper first, then compare with the map.</p>
        <PaperMap />
      </TryIt>

      <Numbers title="Let’s see the numbers: a third pass in miniature">
        <p>Table 3 of the paper says the base model has <b>65 × 10⁶</b> parameters. A third-pass reader does not take that on trust. You have counted parameters since <a href="#/lesson/build-gpt">Build GPT</a>, so rebuild it from the paper’s own hyperparameters: d<sub>model</sub> = 512, d<sub>ff</sub> = 2048, N = 6, a shared vocabulary of about 37,000.</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>piece</th><th>calculation</th><th>parameters</th></tr></thead>
            <tbody>
              <tr><td>One attention sub-layer</td><td className="mono">4 × 512² + 4 × 512</td><td className="mono">1,050,624</td></tr>
              <tr><td>One feed-forward sub-layer</td><td className="mono">2 × 512 × 2048 + 2048 + 512</td><td className="mono">2,099,712</td></tr>
              <tr><td>One LayerNorm</td><td className="mono">2 × 512</td><td className="mono">1,024</td></tr>
              <tr><td>Encoder layer: 1 attention, 1 FFN, 2 norms</td><td /><td className="mono">3,152,384</td></tr>
              <tr><td>Decoder layer: <b>2</b> attentions, 1 FFN, 3 norms</td><td /><td className="mono">4,204,032</td></tr>
              <tr><td>6 encoder + 6 decoder layers</td><td className="mono">6 × 7,356,416</td><td className="mono">44,138,496</td></tr>
              <tr><td>One shared embedding matrix</td><td className="mono">37,000 × 512</td><td className="mono">18,944,000</td></tr>
              <tr><td><b>total</b></td><td /><td className="mono"><b>63,082,496</b></td></tr>
            </tbody>
          </table>
        </div>
        <p>We get 63.1M. The paper says 65M. Within 3%. The same arithmetic for the big model (d<sub>model</sub> = 1024, d<sub>ff</sub> = 4096) gives 214.2M against the paper’s 213M.</p>
        <p>What did that buy you? You now know the decoder layer is a third larger than the encoder layer, because of cross-attention. You know a third of the base model is vocabulary. And you know the count depends on sharing one embedding matrix three ways, which section 3.4 mentions in a single sentence.</p>
        <Callout kind="model" label="Simplified: why not exactly 65M?">The paper gives the vocabulary only as “about 37000” and does not list every bias. Our count assumes biases on all linear layers and exactly 37,000 tokens. A difference of a few percent is what a careful reader should expect, and should note rather than hide.</Callout>
      </Numbers>

      <TheMath>
        <p>Two pieces of the paper that are new to you, in the course’s usual form.</p>
        <Equation
          label="Cross-attention equals softmax of decoder queries times encoder keys transposed over root d k, times encoder values"
          symbols={[
            [<span className="q">Q<sub>dec</sub></span>, <>X<sub>dec</sub>W<sup>Q</sup>: one row per <b>target</b> position, shape (T<sub>tgt</sub>, d<sub>k</sub>)</>],
            [<><span className="k">K<sub>enc</sub></span>, <span className="v">V<sub>enc</sub></span></>, <>ZW<sup>K</sup> and ZW<sup>V</sup>, where Z is the encoder’s output: one row per <b>source</b> token, shape (T<sub>src</sub>, d<sub>k</sub>)</>],
            [<>Q<sub>dec</sub>K<sub>enc</sub><sup>T</sup></>, <>a (T<sub>tgt</sub>, T<sub>src</sub>) rectangle of scores. No mask</>],
            ['softmax', 'along each row: every target position spreads a total weight of 1 over the source tokens'],
            ['result', <>shape (T<sub>tgt</sub>, d<sub>v</sub>): one blended source summary per target position</>],
          ]}
        >
          CrossAttention = softmax( <span className="q">Q<sub>dec</sub></span> <span className="k">K<sub>enc</sub></span><sup>T</sup> / √d<sub>k</sub> ) <span className="v">V<sub>enc</sub></span>
        </Equation>
        <Term
          name="Label smoothing"
          plain={<>During training, do not ask for 100% on the correct token. Ask for a little less, and spread the remainder over the other tokens. The model is discouraged from becoming extremely confident.</>}
          example={<>Vocabulary of 5, correct token is the third, smoothing 0.1. Instead of the target [0, 0, 1, 0, 0], train against [0.02, 0.02, 0.92, 0.02, 0.02].</>}
          formal={<>target = (1 − ε) · one-hot + ε / V, in the formulation of Szegedy et al. (2015), which the paper cites. The paper uses ε = 0.1 and reports the trade plainly: perplexity gets worse, because the model learns to be less sure, while accuracy and BLEU improve.</>}
        />
        <DeepDive title="What does Table 1 actually claim?">
          <p>Table 1 compares layer types on three measures, for a sequence of n positions of width d.</p>
          <div className="table-scroll">
            <table className="plain">
              <thead><tr><th>layer type</th><th>work per layer</th><th>sequential steps</th><th>longest path</th></tr></thead>
              <tbody>
                <tr><td>Self-attention</td><td className="mono">O(n² · d)</td><td className="mono">O(1)</td><td className="mono">O(1)</td></tr>
                <tr><td>Recurrent</td><td className="mono">O(n · d²)</td><td className="mono">O(n)</td><td className="mono">O(n)</td></tr>
                <tr><td>Convolutional, kernel k</td><td className="mono">O(k · n · d²)</td><td className="mono">O(1)</td><td className="mono">O(log<sub>k</sub> n)</td></tr>
                <tr><td>Self-attention restricted to r neighbours</td><td className="mono">O(r · n · d)</td><td className="mono">O(1)</td><td className="mono">O(n / r)</td></tr>
              </tbody>
            </table>
          </div>
          <p>“Sequential steps” is the number of operations that must wait for each other: the reason recurrent networks cannot use a GPU well. “Longest path” is how many layers a signal crosses between two distant positions: short paths make long-range dependencies easier to learn.</p>
          <p>Note what the table does not claim: that self-attention is always cheaper. It is cheaper per layer only while n is smaller than d.</p>
        </DeepDive>
      </TheMath>

      <CodeIt title="Let’s code it: cross-attention is a two-line change">
        <p>Here is the third pass for section 3.2.3. Start from the <code>attention</code> function in <code>phase3-transformers/attention_numpy.py</code>, which you wrote in <a href="#/lesson/attention">Attention</a>. It takes one input <code>x</code>. Give it two.</p>
        <Code title="cross-attention: keys and values read a different sequence">{`
def cross_attention(x_dec, enc_out, Wq, Wk, Wv):
    D = x_dec.shape[1]
    Q = x_dec @ Wq          # (T_tgt, D)  queries: from the decoder
    K = enc_out @ Wk        # (T_src, D)  keys:    from the encoder output
    V = enc_out @ Wv        # (T_src, D)  values:  from the encoder output

    scores = Q @ K.T / np.sqrt(D)       # (T_tgt, T_src): a rectangle, and no mask
    weights = softmax(scores)           # each row sums to 1
    return weights @ V, weights         # (T_tgt, D)
`}</Code>
        <p>Compare it with your original. <code>K</code> and <code>V</code> read <code>enc_out</code> instead of <code>x</code>, and the mask is gone. That is the whole difference. With 3 target rows and 4 source rows the weights come out as (3, 4), rows summing to 1.</p>
        <p>The second piece worth coding is the learning-rate schedule, equation 3 of the paper:</p>
        <Code title="warm up linearly for 4000 steps, then decay like 1/sqrt(step)">{`
def lrate(step, d_model=512, warmup=4000):
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)

lrate(1000)    # 0.000175   a quarter of the way up
lrate(4000)    # 0.000699   the peak: the two terms are equal here
lrate(16000)   # 0.000349   half the peak: 4x the steps, 1/sqrt(4)
`}</Code>
        <p>Your <code>tiny_gpt.py</code> uses a constant 3e-4. Why warm up? The paper does not say. The usual explanation is that at the start the weights are random and Adam’s running averages have seen almost no data, so large early steps are unreliable. Take small steps first, larger ones once the statistics have settled. Nearly every large training run since has used some warm-up.</p>
      </CodeIt>

      <BreakIt title="Break it: argue with the paper">
        <p>A second pass is active. Try these with the paper open. Predict first.</p>
        <ul>
          <li><b>Find the hedge.</b> In section 3.2.1, how sure are the authors about why scaling by √d<sub>k</sub> helps? (They write that they “suspect” it. The variance argument is in footnote 4. Our <a href="#/lesson/attention">attention lesson</a> repeats it with the same caution.)</li>
          <li><b>Find the inconsistency.</b> Compare the English-to-French score in the abstract, in Table 2, and in the running text of section 6.1. (41.8, 41.8 and 41.0 in the current arXiv version. Famous papers have typos too. Trust tables over prose, and check.)</li>
          <li><b>Predict an ablation.</b> Before looking at Table 3, rows (A): what happens with a single head? With 32 heads? (Both are worse than 8. One head by 0.9 BLEU.)</li>
          <li><b>Test a claim against today.</b> Section 4 argues self-attention is cheap because n is usually smaller than d. Take n = 1024 and d = 512: n² · d = 537 million, n · d² = 268 million. At that length a recurrent layer already does less arithmetic. The decisive advantage is the O(1) column: parallel training.</li>
          <li><b>Ask what is missing.</b> Table 2 gives single scores with no error bars and no repeated runs. By how much would 28.4 have to beat a rival before you were convinced?</li>
        </ul>
      </BreakIt>

      <Exercises>
        <Exercise
          id="reading-papers-calc-cross"
          type="calculate"
          title="How big is the rectangle?"
          answer={{ value: 672, tolerance: 0 }}
          answerLabel="attention weights"
          hints={[
            'Cross-attention has one row per target position and one column per source token.',
            'One head computes 7 × 12 weights. The base model has h = 8 heads per layer.',
          ]}
          solution={<><p>7 × 12 = 84 weights per head, times 8 heads = <b>672</b> in one decoder layer.</p><p>Compare the decoder’s masked self-attention in the same layer: a 7 × 7 grid of which 28 cells per head are unmasked. Cross-attention grows with source length × target length. Translating a long document means a long rectangle in every decoder layer, at every step.</p></>}
        >
          <p>The decoder has written 7 target tokens. The source sentence has 12 tokens. How many attention weights does <b>one decoder layer’s cross-attention</b> compute in the base model (all heads)?</p>
        </Exercise>

        <Exercise
          id="reading-papers-calc-lr"
          type="calculate"
          title="The peak learning rate"
          answer={{ value: 0.0007, tolerance: 0.00002 }}
          answerLabel="learning rate at the peak"
          hints={[
            'The rate rises while step · warmup⁻¹·⁵ is the smaller term and falls once step⁻⁰·⁵ is smaller. The peak is where they are equal.',
            'They are equal at step = warmup = 4000. There, lrate = 512⁻⁰·⁵ × 4000⁻⁰·⁵.',
            '1 / √(512 × 4000) = 1 / √2,048,000 = 1 / 1431.',
          ]}
          solution={<><p>At step 4000 both terms equal 4000⁻⁰·⁵. So lrate = 1 / √(512 × 4000) = 1 / 1431 ≈ <b>0.0007</b>.</p><p>That is about twice the constant 3e-4 in <code>tiny_gpt.py</code>, reached gradually and then left behind. Notice that the formula ties the rate to the model width: a d<sub>model</sub> of 1024 gives a peak 1/√2 as large. Wider models take smaller steps.</p></>}
        >
          <p>Equation 3: lrate = d<sub>model</sub><sup>−0.5</sup> · min(step<sup>−0.5</sup>, step · warmup<sup>−1.5</sup>), with d<sub>model</sub> = 512 and warmup = 4000. What is the highest learning rate the base model ever uses? (Four decimals.)</p>
        </Exercise>

        <Exercise
          id="reading-papers-predict-crossover"
          type="predict"
          title="When does self-attention stop being the cheap one?"
          answer={{ value: 512, tolerance: 0 }}
          answerLabel="sequence length n"
          hints={[
            'Set the two costs from Table 1 equal: n² · d = n · d².',
            'Divide both sides by n · d.',
          ]}
          solution={<><p>n² · d = n · d² gives n = d, so the crossover is at <b>n = 512</b>. Below it, a self-attention layer does less arithmetic than a recurrent layer of the same width. Above it, more.</p><p>The paper says this openly. In 2017 sentences were tens of tokens long, so n was far below d. Today’s contexts are hundreds of times longer than d, and the n² term is the dominant cost of long-context models. The argument that survived is the other column: O(1) sequential steps, which is what lets a GPU train on all positions at once.</p></>}
        >
          <p>Using only Table 1: for d = 512, at what sequence length n does one self-attention layer cost the same as one recurrent layer?</p>
        </Exercise>

        <Exercise
          id="reading-papers-explain-table"
          type="explain"
          title="Interrogate a results table"
          hints={[
            'Think about where the baseline’s number came from, and whether both systems had the same resources.',
            'Think about noise: how many runs, how many test items, any interval?',
            'Think about what the benchmark could have leaked into, and about what is not in the table.',
          ]}
          solution={<><p>Good questions, roughly in order of how often they sink a claim:</p><ul><li>Did the authors run the baseline themselves with equal tuning effort, or copy its number from an older paper?</li><li>Same training data, parameter count and compute? A 0.4-point gain from 3× the compute is not a better method.</li><li>How many runs? With one run and no interval, 0.4 points may be seed noise. On a 1,000-item benchmark the 95% interval on an accuracy near 71% is about ± 2.8 points.</li><li>Were hyperparameters chosen on the test set?</li><li>Could the benchmark’s questions be in the pretraining data of either model?</li><li>Is there an ablation showing which part of the method produces the gain? What about tasks where it lost?</li></ul><p>None of this means the claim is false. It means the table alone does not establish it.</p></>}
        >
          <p>A paper’s main table reads: “Baseline 70.9, Ours <b>71.3</b>” on a well-known benchmark of 1,000 questions. Write down at least four questions you would want answered before believing that “Ours” is better.</p>
        </Exercise>

        <Exercise
          id="reading-papers-implement-cross"
          type="implement"
          title="Third pass: build section 3.2.3"
          hints={[
            'Copy attention() in phase3-transformers/attention_numpy.py to a new function that takes x_dec and enc_out.',
            'Q comes from x_dec. K and V come from enc_out. Delete the causal branch.',
            'Test with x_dec of shape (3, D) and enc_out of shape (4, D). Assert weights.shape == (3, 4) and np.allclose(weights.sum(axis=1), 1).',
          ]}
          solution={<><p>The function is the one shown in “Let’s code it”. The checks: <code>out.shape == (3, D)</code>, <code>weights.shape == (3, 4)</code>, rows sum to 1.</p><p>Now the insight test. Shuffle the rows of <code>enc_out</code>: the output for each target position is unchanged, because a weighted sum does not care about the order of its terms. Cross-attention by itself is blind to source order, exactly like self-attention. The source order reaches the decoder only because the encoder’s vectors already carry positional information.</p></>}
        >
          <p>Implement <code>cross_attention</code> in NumPy from your own <code>attention</code> function, without looking at the code above. Then predict: if you shuffle the rows of the encoder output, what happens to the decoder’s result?</p>
        </Exercise>

        <ExplainBack
          id="reading-papers-explain"
          prompt="A colleague has read that “GPT is a Transformer” and then looked at Figure 1 of the paper, with its two towers. They are confused. Explain what the two halves do, which half GPT keeps, and why GPT does not need cross-attention."
          modelAnswer={<p>Figure 1 shows a translation model. The left tower, the encoder, reads the whole source sentence with unmasked self-attention and outputs one vector per source token. The right tower, the decoder, writes the target one token at a time. Each decoder layer has masked self-attention over the target so far, then cross-attention, in which the queries come from the decoder and the keys and values come from the encoder’s output, then a feed-forward network. GPT keeps only the right tower and deletes the cross-attention sub-layer. It does not need it because there is no separate source: the prompt and the answer are one sequence, so masked self-attention already lets every answer token look at every prompt token. What remains is a stack of masked self-attention and feed-forward blocks trained to predict the next token, which is what we built.</p>}
        />
      </Exercises>

      <CheckYourself
        questions={[
          {
            q: 'You have ten minutes and a new paper. According to the three-pass method, what do you read?',
            options: ['Title, abstract, introduction, the section headings and the conclusion', 'The method section, slowly, until the first equation is fully clear', 'The related-work section, to learn the background before the claim', 'The appendix, because the hyperparameters decide whether it works'],
            answer: 0,
            explain: 'The first pass finds the claim and decides whether the paper deserves more time. In ML papers, add a look at the main figure and the main table.',
          },
          {
            q: 'In cross-attention, where do the queries, keys and values come from?',
            options: ['Queries from the encoder output; keys and values from the decoder', 'All three from the decoder, with the causal mask switched off', 'Queries from the decoder; keys and values from the encoder output', 'All three from the encoder output, with a causal mask added on'],
            answer: 2,
            explain: 'Each target position asks (query); the source tokens are matched (keys) and hand over content (values). The score grid is target length × source length.',
          },
          {
            q: 'Why does the encoder’s self-attention have no causal mask, while the decoder’s has one?',
            options: ['Masks were found to slow training down, so the authors used as few of them as they could', 'The source is fully known in advance; the target is being generated, so its future does not exist yet', 'The encoder has fewer layers than the decoder, so it needs all of the positions to compensate', 'The encoder uses sinusoidal positions, which already prevent tokens from seeing the future'],
            answer: 1,
            explain: 'Masking is about what is legitimately available. Peeking at later source words is fine. Peeking at later target words during training would be cheating.',
          },
          {
            q: 'A paper writes h = Wx with W ∈ ℝ^{d×k}. Your code has x of shape (k,). Which line computes h?',
            options: ['h = W @ x, or equally h = x @ W.T', 'h = x @ W, since x is on the left in code', 'h = W * x, the element-wise product', 'h = W.T @ x, because papers transpose W'],
            answer: 0,
            explain: 'W is (d, k) and x is (k,), so W @ x is (d,). In row convention the same result is x @ W.T. Writing x @ W would be a shape error unless d = k, and then a silent bug.',
          },
          {
            q: 'The paper trains with label smoothing 0.1 and reports that perplexity got worse while BLEU improved. What is the right reading?',
            options: ['Label smoothing is a bug that the authors chose to leave in the final version of the model', 'Perplexity and BLEU always move in opposite directions, by the way they are defined', 'The BLEU improvement must be noise, because a worse loss cannot mean a better model', 'The model was trained to be less certain, which hurts a confidence-based metric but helped translation quality'],
            answer: 3,
            explain: 'Perplexity rewards putting high probability on the exact reference token. Smoothing deliberately caps that. Different metrics measure different things, which is why you check which one a claim rests on.',
          },
        ]}
      />

      <Remember
        items={[
          <><b>Three passes</b> (Keshav, 2007): find the claim in ten minutes; grasp the argument and its evidence in an hour, skipping proofs; re-create the work only for the papers that matter. Stop after any pass.</>,
          <>In an ML paper, look first at the <b>architecture figure</b>, the <b>one central equation</b>, the <b>main results table</b>, the <b>ablation table</b> and the <b>hyperparameters</b>.</>,
          <>Notation is code you already wrote: ∈ ℝ<sup>n×d</sup> is a shape, Σ is a loop, 𝔼 is <code>.mean()</code>, ∇ is <code>.backward()</code>, ⊙ is <code>*</code>. Many papers use <b>Wx + b</b> with column vectors: the transpose of this course’s <code>x @ W + b</code>.</>,
          <>The original Transformer is an <b>encoder-decoder</b>: unmasked self-attention over the source, masked self-attention over the target, and <b>cross-attention</b> with queries from the decoder and keys and values from the encoder. GPT is the decoder alone, without cross-attention.</>,
          <>Read results as claims: baselines, compute, tuning, error bars, contamination, and what is missing. Even this paper says “we suspect”, and has a typo in a headline number.</>,
        ]}
      />

      <RealLLM>
        <ToyVsReal
          toy={<ul><li>Decoder-only, pre-norm, learned positions, GELU, constant learning rate</li><li>One sequence: prompt and answer together</li><li>Evaluated by validation loss and by reading samples</li><li>0.8M parameters, minutes on a CPU</li></ul>}
          real={<ul><li>The 2017 paper: encoder-decoder, post-norm, sinusoidal positions, ReLU, warm-up then decay, label smoothing 0.1</li><li>Two sequences joined by cross-attention</li><li>Evaluated by BLEU on WMT 2014: 28.4 and 41.8 for the big model</li><li>65M and 213M parameters; 12 hours and 3.5 days on 8 P100 GPUs</li></ul>}
        />
        <Callout kind="established">Encoder-decoder Transformers did not disappear. T5 and many dedicated translation models use the layout, and so does the speech recogniser Whisper: an encoder over audio, a decoder over text, cross-attention between them. Some vision-language models feed image features to a language model through cross-attention layers. When the input and the output are different kinds of sequence, the two-tower design is still a natural fit.</Callout>
        <Callout kind="research">Why decoder-only models came to dominate general-purpose language modelling is still argued over. Commonly given reasons are simplicity (one stack, one objective, every token of the data gives a training signal), easy reuse of the KV cache in chat, and the way they scaled in practice. Careful comparisons at equal compute exist and do not all point the same way. Treat any one-line explanation as an opinion.</Callout>
        <h3>What to read next, and in what order</h3>
        <p>You are prepared for every paper on this list. Use the three passes. Do a third pass on at most two or three of them.</p>
        <div className="table-scroll">
          <table className="plain" style={{ fontSize: 14.5 }}>
            <thead><tr><th>#</th><th>paper</th><th>why read it</th><th>you are prepared by</th></tr></thead>
            <tbody>
              {READING_PATH.map((p, i) => (
                <tr key={p.short}>
                  <td className="mono">{i + 1}</td>
                  <td style={{ minWidth: 210 }}><a href={p.url} target="_blank" rel="noreferrer"><b>{p.short}</b></a><br /><span style={{ fontSize: 13.5 }}>{p.title}</span><br /><span className="muted" style={{ fontSize: 13 }}>{p.authors}, {p.year}. {p.id}</span></td>
                  <td style={{ minWidth: 240 }}>{p.why}</td>
                  <td style={{ fontSize: 13.5 }}>{p.prepared.map((id, j) => { const l = lessonById(id); return l ? <span key={id}>{j > 0 && <br />}<a href={`#/lesson/${id}`}>{l.code} {l.title}</a></span> : null })}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted" style={{ fontSize: 15 }}>The order is by dependency, not by date: the GPT line first, then how size and data trade off, then how a base model becomes an assistant, then the engineering papers behind the models you can download, then systems built around a model, and reasoning models last.</p>
      </RealLLM>
    </Lesson>
  )
}
