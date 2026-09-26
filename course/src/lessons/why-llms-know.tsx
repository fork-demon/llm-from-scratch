import { Lesson, Why, Problem, MentalModel, TryIt, Numbers, TheMath, CodeIt, BreakIt, Exercises, CheckYourself, Remember, RealLLM } from '../components/lesson'
import { Callout, DeepDive, Equation, Flow, G, Term, ToyVsReal, WhyExists } from '../components/ui'
import { Code } from '../components/Code'
import { Exercise, ExplainBack } from '../components/exercise'
import { KnowledgeLab } from '../interactive/KnowledgeLab'
import { ClaimSorter } from '../interactive/ClaimSorter'
import { TextIntoWeights } from '../illustrations/TextIntoWeights'

export default function WhyLlmsKnowLesson() {
  return (
    <Lesson id="why-llms-know">
      <Why>
        <p className="lede">Thursday. The support team forwards Riya a screenshot from the vendor chatbot they are testing. A customer asked about refunds.</p>
        <div className="card" style={{ fontFamily: 'var(--serif)' }}>“As per Paisa Pal’s 14-day refund policy, failed UPI payments are refunded within 3 working days, and you are entitled to a ₹100 inconvenience credit.”</div>
        <p>Paisa Pal has no 14-day policy. There is no ₹100 credit. The whole policy was invented, in perfect customer-service English.</p>
        <p>Dev, reading over her shoulder: “But it sounded so sure. It must have got that from somewhere.” Did it? To find out, ask a model two simpler questions.</p>
        <div className="grid-2">
          <div className="card">
            <p className="mono" style={{ fontSize: 14 }}>What is the capital of France?</p>
            <p><b>Paris.</b> Correct. Yet you built a GPT in <a href="#/lesson/build-gpt">Part 7</a>, and you know there is no table of capitals in it. So where did “Paris” come from?</p>
          </div>
          <div className="card">
            <p className="mono" style={{ fontSize: 14 }}>Cite a paper on left-handed beekeepers in Tasmania.</p>
            <p>You may get a title, authors, a journal and a year. Well formatted. Confident. <b>And the paper may not exist.</b></p>
          </div>
        </div>
        <p>These look like opposite behaviours: knowing and inventing. This lesson argues they are <em>the same mechanism</em>, working on well-covered and on thinly-covered ground. The refund policy is the second card, wearing a Paisa Pal badge.</p>
        <p>A trained model is a fixed set of numbers plus a fixed recipe of arithmetic. Everything it “knows” must be somewhere in those numbers.</p>
        <p>This part of the course is also where certainty runs out. From here on, every important claim carries one of three labels: <b>established</b> (well understood, or measured many times), <b>model</b> (a simplification that helps you think, not literally true) and <b>research</b> (still being worked out, or actively disputed).</p>
      </Why>

      <Problem title="The problem: there is nothing inside to look things up in">
        <p>Kabir answers Dev’s question with one of his own: “Got it from where? Open the model file and show me the documents.”</p>
        <p>Think about what survives <a href="#/lesson/training-gpt">training</a>. The text is read in batches, each batch nudges the weights a little, and then the batch is discarded. When training ends, you ship one file.</p>
        <TextIntoWeights />
        <Callout kind="established">
          A language model file contains <G t="parameters">parameters</G> (the weight matrices you built: embeddings, attention projections, MLP layers, norms) and nothing else. No documents. No database. No index. No list of facts. At inference time the model has the weights and the tokens in its <G t="context-window">context window</G>. That is all.
        </Callout>
        <WhyExists
          problem="A model answers factual questions, but contains no documents and no database."
          naive="Assume the facts are stored as records somewhere inside: one neuron or one row per fact, which the model looks up."
          fails="Nobody has found such records. Remove individual neurons and, as a rule, no single fact cleanly disappears. And a lookup would fail visibly on a missing key, whereas a model never does."
          idea="Facts are not entries. They are statistical regularities of the training text, absorbed as small adjustments to many shared weights, because getting them right lowered next-token loss."
          tradeoff="Compact and able to generalise, but there is no way to list what is known, no clean way to update one fact, and no built-in signal for “this is not in here”."
        />
        <p>Recall <a href="#/lesson/surprising-idea">the surprising idea</a> from Part 0: the model only predicts the next token. To predict the token after “The capital of France is”, it helps enormously to have absorbed that Paris goes there.</p>
        <p><b>Knowledge is a side effect of prediction.</b></p>
      </Problem>

      <MentalModel title="A mental model: knowledge as learned tendencies">
        <p>Training leaves three kinds of trace in the weights. They are points on one scale, not separate systems.</p>
        <ul>
          <li><b>Verbatim memorisation.</b> Text seen <em>many</em> times can be reproduced word for word: famous quotes, licence boilerplate, well-known poems.</li>
          <li><b>Learned patterns.</b> “Capital of X is Y”, how a citation is formatted, how a Python function starts. Seen in thousands of variations, stored as a reusable shape.</li>
          <li><b>Generalisation.</b> Applying those patterns to text never seen before. This is most of what a model does: almost every prompt you type is new.</li>
        </ul>
        <p>Memorisation is established: researchers have extracted verbatim training text, and it rises with how often a passage was repeated and with model size. You created the extreme case yourself if you ever over-trained <code>tiny_gpt.py</code> on a tiny corpus: training loss keeps falling while <G t="validation-loss">validation loss</G> rises.</p>
        <p>A useful picture: the weights are a <b>lossy compression</b> of the training text, like a JPEG of a library. Repeated content survives almost exactly; rare content is blurred, and zooming into a blur still draws sharp-looking pixels. The analogy stops here: a JPEG only gives back what it compressed, while a model recombines patterns into sentences that never existed. That is why it is useful, and why its inventions look exactly like its recollections.</p>
        <h3>Which part of the Transformer does what?</h3>
        <p>You have built all three components. Here is what is currently understood about their role in recalling a fact. Notice how the confidence drops from top to bottom.</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>Component</th><th>Role in recalling a fact</th></tr></thead>
            <tbody>
              <tr><td><G t="embedding">Embeddings</G></td><td>Token-level features: which token this is and what kind of thing it tends to be. “France” sits near other countries, a feature of a token, not yet a statement about the world.</td></tr>
              <tr><td><G t="attention">Attention</G></td><td>Moves information between positions. To complete “The capital of France is”, the last position must gather “France” and “capital of” from earlier positions. Only attention can do that.</td></tr>
              <tr><td><G t="ffn">MLP / feed-forward</G></td><td>Transforms each token’s vector on its own, with most of the block’s parameters. Evidence points to middle-block MLPs being heavily involved in recalling associations such as France → Paris.</td></tr>
            </tbody>
          </table>
        </div>
        <p>A simplified mental model (a lens, not a wiring diagram): read the MLP as a <b>soft key-value memory</b>. Its first matrix is a bank of pattern detectors (“does this vector look like <em>France + capital-of</em>?”), its second is what each detector writes back (“push toward <em>Paris</em>”). Detect, gate, write. It explains part of what MLP layers do; most detectors respond to many unrelated things.</p>
        <Callout kind="research">
          Experiments that trace and edit facts (the best known is ROME, 2022) found that MLP layers in middle blocks, at the position of the subject’s last token, matter a great deal for factual recall, and that changing one MLP matrix can make a model say “The Eiffel Tower is in Rome”. But later work found that <em>where</em> tracing locates a fact does not reliably predict <em>where</em> editing works best, edits often fail to carry over to implications of the fact, and attention layers take part in extracting the attribute. So the careful statement is: <b>MLP layers are heavily involved in recalling factual associations. Facts are not cleanly stored in one place.</b>
        </Callout>
        <Term
          name="Distributed, superposed representation"
          plain={<>One fact is spread over many weights, and one weight takes part in many facts. Models even pack more features into a layer than it has dimensions, by using directions that are almost, but not exactly, perpendicular.</>}
          example={<>In the lab below, 8 facts share 80 weights. Answering any one question uses all 80.</>}
          formal={<>Features correspond to directions in activation space rather than to individual neurons. When there are more features than dimensions, they overlap slightly (superposition), and the overlap shows up as small interference between unrelated inputs.</>}
        />
        <Callout kind="dev">
          The weights are not a database. There is no <code>SELECT</code> to list what is known, no <code>UPDATE</code> that changes one fact without touching others, no schema, no transactions, and no <code>KeyError</code>. It behaves more like a function fitted to data: you can only call it and see what comes out.
        </Callout>
      </MentalModel>

      <TryIt title="Go looking for a fact">
        <p>Talk is cheap. Here is the smallest model that can “know” something: one weight matrix, trained in your browser by the same <G t="gradient-descent">gradient descent</G> and <G t="cross-entropy">cross-entropy</G> you already know. Do the four tabs in order, and predict before each one.</p>
        <KnowledgeLab />
        <p className="muted">A simplified model: one linear layer and 8 facts. What carries over to a real LLM is the <em>kind</em> of storage: shared weights, graceful degradation, and an output that is always a full probability distribution. Real models are not this easy to inspect.</p>
      </TryIt>

      <Numbers>
        <p>Two calculations. First, the toy hallucination from tab 4, by hand. These are the actual <G t="logits">logits</G> the trained lab model produces (seeded, so yours match).</p>
        <div className="table-scroll">
          <table className="plain mono" style={{ fontSize: 13.5 }}>
            <thead><tr><th>query</th><th>largest logit</th><th>e<sup>logit</sup></th><th>sum of all 8 e<sup>logit</sup></th><th>probability</th><th>true?</th></tr></thead>
            <tbody>
              <tr><td>France (trained)</td><td>Paris 4.47</td><td>87.34</td><td>94.81</td><td><b>92.1%</b></td><td>yes</td></tr>
              <tr><td>Austria (never seen)</td><td>Berlin 4.52</td><td>91.76</td><td>98.91</td><td><b>92.8%</b></td><td>no</td></tr>
              <tr><td>Atlantis (does not exist)</td><td>Cairo 2.54</td><td>12.64</td><td>23.39</td><td><b>54.0%</b></td><td>no</td></tr>
            </tbody>
          </table>
        </div>
        <p>The false answer about Austria is slightly <em>more</em> confident than the true answer about France. This is Dev’s “but it sounded so sure”, in three numbers.</p>
        <p><G t="softmax">Softmax</G> divides by the total, so the outputs always add up to 100%, whether or not any of the options is right. “None of the above” can only win if it is one of the options.</p>
        <h3>Why making things up is structural</h3>
        <p>That row is a hallucination in miniature. It was not caused by a fault. Put three things you have already seen side by side.</p>
        <ol>
          <li><b>The output is always a distribution over plausible next tokens.</b> Softmax sums to 1. Some token always wins.</li>
          <li><b>There is no built-in “I do not know” signal.</b> The model can emit the <em>words</em> “I do not know” only when those words are the likely continuation. In most training text, a question is followed by an answer.</li>
          <li><b>Training rewards fluent continuation.</b> <G t="pretraining">Pretraining</G> loss is lower when the output looks like the training text. For well-covered facts, looking right and being right coincide. For thinly covered ones, they come apart, and the loss barely notices.</li>
        </ol>
        <p>So a <G t="hallucination">hallucination</G> is not a malfunction. It is the generalisation machinery doing its normal job (produce the most plausible continuation) where plausible and true differ. In the lab, “Austria → Berlin” came from exactly the same arithmetic as “France → Paris”.</p>
        <h3>Where the parameters are</h3>
        <p>Second calculation: where are the parameters in the model you built? Take one block of <code>tiny_gpt.py</code> with <code>n_embd = 128</code>.</p>
        <div className="table-scroll">
          <table className="plain mono" style={{ fontSize: 13.5 }}>
            <thead><tr><th>part of one block</th><th>calculation</th><th>parameters</th></tr></thead>
            <tbody>
              <tr><td>attention (qkv + proj, with biases)</td><td>(128×384 + 384) + (128×128 + 128)</td><td>66,048</td></tr>
              <tr><td>MLP (128 → 512 → 128, with biases)</td><td>(128×512 + 512) + (512×128 + 128)</td><td>131,712</td></tr>
              <tr><td>two LayerNorms</td><td>2 × (128 + 128)</td><td>512</td></tr>
              <tr><td><b>block total</b></td><td /><td><b>198,272</b></td></tr>
            </tbody>
          </table>
        </div>
        <p>131,712 / 198,272 = <b>66%</b>. About two thirds of every block is MLP. With four blocks plus embeddings (65-character vocabulary), the whole model is 809,856 numbers: the “0.81M” that the script prints. Everything your Shakespeare model knows about English is in those 809,856 numbers.</p>
      </Numbers>

      <TheMath title="The math: why bigger models know more">
        <p>One more puzzle. Why did making models larger work so well? The answer starts with a measurement, not a theory.</p>
        <p>Train a family of models of different sizes on plenty of data, and plot final <G t="loss">loss</G> against size. On log-log axes, the points fall close to a straight line over many orders of magnitude. A straight line on log-log axes is a <b>power law</b>:</p>
        <Equation
          label="Loss equals a floor plus a constant divided by N to the power alpha"
          symbols={[
            ['L', 'next-token loss on held-out text (lower is better)'],
            ['N', 'number of parameters (the same shape of law holds for data size and for compute)'],
            ['α', 'a small positive exponent measured from experiments, not derived; the reported value depends on the setup and on the exact form fitted (about 0.08 for parameters in a well-known 2020 study, which fitted no floor term; about 0.3 in the 2022 Chinchilla study, which did)'],
            ['A', 'a constant fitted to the measurements'],
            [<>L<sub>∞</sub></>, 'a floor that no model size removes: text is genuinely unpredictable to some degree'],
          ]}
        >
          L(N) ≈ L<sub>∞</sub> + A / N<sup>α</sup>
        </Equation>
        <p>Read it like this: multiply the parameters by 10 and the removable part of the loss is multiplied by 10<sup>−α</sup>. With the 2020 study’s α = 0.076 that is 0.84: about 16% lower, every time, for every factor of 10. Small per step, but <em>predictable</em>, which is why labs could justify very large training runs before running them.</p>
        <p>The scaling laws themselves are established: empirical, and reproduced many times. Loss falls smoothly as a power law in parameters, data and compute, as long as none of the three is the bottleneck. A follow-up (“Chinchilla”, 2022) showed that parameters and training tokens should grow together, around 20 tokens per parameter for a compute-optimal run.</p>
        <Callout kind="research">
          <b>Why</b> loss follows a power law is not settled. One intuition: language has a long tail of ever rarer patterns and facts, and each factor of 10 buys the next slice of the tail. That is suggestive, not proven. Also open: “emergent abilities”. Some skills seem to appear suddenly at a certain scale. At least part of that is a measurement effect: a pass/fail metric can jump while the underlying probability of the right answer improves smoothly. Whether any abilities are truly discontinuous is debated.
        </Callout>
        <DeepDive title="Loss goes down smoothly. Why would knowledge go up?">
          <p>Loss is average surprise over all tokens. Common grammar is learned first because it affects almost every token. Once that is done, the remaining loss sits in rarer things: uncommon words, then domain knowledge, then individual facts.</p>
          <p>So later, smaller improvements in loss correspond to rarer knowledge. This is a mental model that matches observed behaviour (small models know famous facts, large models know obscure ones), not a theorem.</p>
        </DeepDive>
      </TheMath>

      <CodeIt title="Let's see it in the code you already have">
        <p>There is no new code in this lesson. The point is to re-read three places in <code>tiny_gpt.py</code> with new eyes.</p>
        <p><b>1. The model is its parameters.</b> This line from <code>main()</code> counts everything the model will ever have:</p>
        <Code source="phase3-transformers/tiny_gpt.py" title="all the knowledge the model can have lives in these tensors">{`
model = GPT(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"vocab {cfg.vocab_size}, params {n_params/1e6:.2f}M, device {device}")
`}</Code>
        <p><b>2. The MLP, read as “detect, gate, write back”.</b> Same class as before. Only the comments are new, and they describe the mental model, not a guarantee.</p>
        <Code source="phase3-transformers/tiny_gpt.py" title="class FeedForward, annotated with the key-value reading">{`
self.net = nn.Sequential(
    nn.Linear(cfg.n_embd, 4 * cfg.n_embd),   # 512 pattern detectors: each row is one dot product with the token vector
    nn.GELU(),                                # gate: weak matches are silenced
    nn.Linear(4 * cfg.n_embd, cfg.n_embd),   # each active detector writes its own direction back into the vector
    nn.Dropout(cfg.dropout),
)
`}</Code>
        <p><b>3. Generation cannot decline.</b> Look for the branch that handles “the model does not know”:</p>
        <Code source="phase3-transformers/tiny_gpt.py" title="generate(): the inner loop">{`
logits, _ = self(idx_cond)
logits = logits[:, -1, :] / temperature     # last position only
probs = F.softmax(logits, dim=-1)           # always a full distribution
nxt = torch.multinomial(probs, num_samples=1)   # always returns a token
idx = torch.cat([idx, nxt], dim=1)              # feed back in
`}</Code>
        <p>There is no such branch. <code>multinomial</code> always returns a token. The loop has no concept of true or false, only of likely and unlikely.</p>
      </CodeIt>

      <BreakIt>
        <p>Go back to the lab and try to break it. Predict first, then check.</p>
        <ul>
          <li><b>Reset and retrain.</b> Do you get the same weights? (Yes: the seed is fixed. Same data, same start, same arithmetic.)</li>
          <li><b>Knock out 30%</b> with the first damage pattern. How many of 8 facts are lost? (None. All eight survive, and average confidence drops from 94% to 81%.)</li>
          <li><b>Press “Invent another” ten times.</b> Does the model ever spread its answer evenly, the honest response to a meaningless input? Note the lowest top probability you see.</li>
          <li><b>Reset, do not train, open tab 4.</b> The untrained model is the only “honest” one: about 12.5% everywhere. Confidence is something training <em>adds</em>.</li>
        </ul>
      </BreakIt>

      <Exercises>
        <Exercise
          id="why-llms-know-chinchilla"
          type="calculate"
          title="How much text does a 70B model want?"
          answer={{ value: 1.4, tolerance: 0.05 }}
          answerLabel="trillion tokens"
          hints={[
            'The rule of thumb from this lesson: about 20 training tokens per parameter.',
            '70B means 70 × 10⁹ parameters. Multiply by 20.',
            '70 × 10⁹ × 20 = 1,400 × 10⁹. How many trillions (10¹²) is that?',
          ]}
          solution={<><p>70 × 10⁹ × 20 = 1.4 × 10¹² = <b>1.4 trillion tokens</b>.</p><p>For scale: that is more than a thousand times what a keen reader gets through in a lifetime. Many recent open models are deliberately trained on far more than this ratio, because a smaller model trained for longer is cheaper to run afterwards. Either way, data at this scale cannot be checked by hand, which is one reason the next lessons care so much about filtering.</p></>}
        >
          <p>Using the compute-optimal rule of thumb from the Chinchilla study, roughly how many training tokens go with a 70-billion-parameter model? Answer in trillions.</p>
        </Exercise>

        <Exercise
          id="why-llms-know-powerlaw"
          type="calculate"
          title="Read a scaling law"
          answer={{ value: 1.26, tolerance: 0.02 }}
          answerLabel="removable loss at 100B"
          hints={[
            'Only the removable part A / N^α changes with size. Going from 1B to 100B multiplies N by 100.',
            'Multiplying N by 100 multiplies A / N^α by 100^(−α) = 100^(−0.1).',
            '100^(−0.1) = 10^(−0.2) ≈ 0.631. Multiply the old value by it.',
          ]}
          solution={<><p>2.0 × 100<sup>−0.1</sup> = 2.0 × 0.631 = <b>1.26</b>.</p><p>A hundred times the parameters (and far more than a hundred times the cost) removed 37% of the removable loss. Power laws mean steady, predictable, and increasingly expensive progress. The exponent 0.1 here is an illustration; measured exponents differ by setup.</p></>}
        >
          <p>A lab fits L(N) = L<sub>∞</sub> + A / N<sup>α</sup> with α = 0.1 (an illustrative number). At 1B parameters the removable part, A / N<sup>α</sup>, is 2.0. Predict the removable part at 100B parameters. (Two decimals.)</p>
        </Exercise>

        <Exercise
          id="why-llms-know-debug"
          type="debug"
          title="“Just set temperature to 0”"
          hints={[
            'What does temperature 0 change: the distribution the model computes, or how a token is picked from it?',
            'Look at the Austria row in the numbers section. Which token does greedy decoding pick there?',
          ]}
          solution={<><p>Temperature 0 means “always take the most likely token” (<a href="#/lesson/inference">greedy decoding</a>). It removes <em>randomness</em>, not <em>error</em>. For Austria, the most likely token is Berlin at 92.8%, so greedy decoding returns the wrong answer every single time, reproducibly.</p><p>Low temperature does help against one failure: drawing an unlucky low-probability token when the top answer was right. It does nothing when the top answer itself is wrong. The fix has to change what is likely, which means changing the context (retrieval, tools) or the weights (training).</p></>}
        >
          <p>A colleague proposes: “Our chatbot invents product codes. Set <code>temperature=0</code> so it stops being creative, and the hallucinations will go away.” What is wrong with this reasoning? What would it fix, and what not?</p>
        </Exercise>

        <ClaimSorter
          id="why-llms-know-sort"
          claims={[
            { text: 'After training, the only thing the model keeps is its parameters. The training documents are not inside it.', level: 'established', why: 'This is what a model file is. Anything it can recall has to be encoded in the weights.' },
            { text: 'The MLP works like a key-value store: the first matrix matches patterns, the second writes the associated content.', level: 'model', why: 'A helpful reading backed by interpretability work, but a lens rather than the literal design: detectors respond to many things and much MLP behaviour is unexplained.' },
            { text: 'Softmax always produces a probability distribution that sums to 1, so some token is always produced.', level: 'established', why: 'It follows directly from the formula. You verified it in the lab.' },
            { text: 'Each fact is stored in a specific, locatable set of MLP weights that can be edited without side effects.', level: 'research', why: 'Editing experiments partly succeed, but localisation and editing results disagree, edits often fail to reach implications, and side effects occur. Debated.' },
            { text: 'Test loss falls as a smooth power law as parameters, data and compute grow.', level: 'established', why: 'An empirical regularity, measured repeatedly across many scales. (Why it holds is a different question.)' },
            { text: 'Some abilities appear suddenly at a certain model size, as a genuine discontinuity.', level: 'research', why: 'Jumps in benchmark scores are observed, but some vanish when a smoother metric is used. Which cases, if any, are truly discontinuous is contested.' },
            { text: 'The weights are a blurry JPEG of the training text.', level: 'model', why: 'An analogy for lossy storage: good for intuition about memorisation versus blur, wrong about recombination and generalisation.' },
          ]}
        />

        <details className="deep">
          <summary>More practice (optional)</summary>
          <div className="details-body">
          <Exercise
            id="why-llms-know-modify"
            type="modify"
            title="Ask your Shakespeare model about France"
            hints={[
              'In main(), generation starts from start = torch.zeros((1, 1), ...). Replace it with your encoded prompt.',
              'start = torch.tensor([[stoi[c] for c in "The capital of France is "]], device=device). Every character of that prompt occurs in the Shakespeare text, so stoi has it.',
            ]}
            solution={<p>It continues fluently, in vaguely Shakespearean character soup, and never says “unknown” or stops. Your model has no knowledge of capitals at all, and nothing in the forward pass or the sampling loop can express that. A production LLM differs in how much it knows, not in this mechanism.</p>}
          >
            <p>Train <code>tiny_gpt.py</code> (the <code>--quick</code> run is enough). Then change the final sample so that generation starts from the prompt <code>"The capital of France is "</code> instead of a single zero token. Before you run it: will the model refuse, stop, or continue?</p>
          </Exercise>

          <ExplainBack
            id="why-llms-know-explain"
            prompt="A non-technical manager asks: “Why does the AI sometimes make things up, and why can’t the vendor just fix that bug?” Answer in plain words, without the terms softmax, logits or parameters."
            modelAnswer={<p>The system does not look facts up. It has learned, from an enormous amount of text, what a good continuation of any piece of text looks like, and it always produces the most fitting continuation it can. For topics it saw often, the most fitting continuation is the true one. For topics it barely saw, the most fitting continuation is something that merely looks right: the right format, the right tone, invented details. Nothing inside it separates those two cases, because both come out of the same process. So it is not a bug in one place that can be patched. It can be reduced a lot, by giving the system the relevant documents at question time, by letting it use tools, and by training it to say when it is unsure, but each of those lowers the rate rather than removing the cause.</p>}
          />
          </div>
        </details>
      </Exercises>

      <CheckYourself
        questions={[
          {
            q: 'A trained LLM answers “Paris” to “The capital of France is”. Where did that come from?',
            options: ['It searched its copy of the training documents', 'From a facts table that is filled during training', 'From the weights: training adjusted many shared parameters so that “Paris” gets a high probability after that context', 'From the embedding of the token “France”, which contains the capital'],
            answer: 2,
            explain: 'Only parameters persist after training. The fact is a regularity absorbed into many of them, because predicting it lowered the loss.',
          },
          {
            q: 'In the lab you zeroed 30% of the weights. Why did no single fact disappear while the others stayed perfect?',
            options: ['The lab protects important weights from being zeroed', 'Every fact depends on many weights and every weight serves many facts, so damage is shared out', 'Softmax repairs damaged weights', 'Because 30% is below the threshold at which facts are deleted'],
            answer: 1,
            explain: 'Distributed storage degrades gradually. A table of records would lose exactly the rows you deleted and nothing else.',
          },
          {
            q: 'Why is hallucination called structural?',
            options: ['Because models are trained on fiction', 'Because the architecture always outputs a distribution over plausible tokens, has no built-in “unknown” signal, and was trained for plausible continuation', 'Because of random sampling: at temperature 0 it would disappear', 'Because the context window is too short'],
            answer: 1,
            explain: 'The same mechanism yields true answers on well-covered ground and plausible inventions on thin ground. Greedy decoding does not change that.',
          },
        ]}
      />

      <Remember
        items={[
          <><b>Only parameters persist.</b> No documents, no database, no index inside the model. Knowledge is a side effect of learning to predict the next token.</>,
          <>Knowledge is <b>distributed</b>: one fact touches many weights, one weight serves many facts. Hence gradual degradation, no way to list what is known, and no clean single-fact update. MLP layers are heavily involved in recall, but “facts live in the FFN” is a simplification.</>,
          <><b>Hallucination is structural</b>: always a distribution, no “I do not know” signal, trained for fluent continuation. Retrieval, tools and training to abstain reduce it. None removes it.</>,
          <><b>Scaling laws are established as measurements</b> (loss falls as a power law). <em>Why</em> they hold and what “emergence” really is are active research.</>,
        ]}
      />

      <RealLLM>
        <Flow horizontal steps={[{ label: 'Tiny GPT', sub: 'what you built' }, { label: 'This lesson', sub: 'what the weights hold' }, { label: 'Modern architecture', sub: 'next' }, { label: 'Training pipeline' }, { label: 'Reasoning models' }]} active={1} />
        <ToyVsReal
          toy={<ul><li>One linear layer, 80 weights, 8 facts</li><li>Subject vectors are random and fixed</li><li>You can see every weight and retrain in a second</li><li>Output: 8 capitals, nothing else</li></ul>}
          real={<ul><li>Billions of weights, dozens of blocks with attention and MLPs</li><li>Subject representations are built up over several layers from the context</li><li>Nobody can read the weights directly; interpretability is a research field</li><li>Output: the whole vocabulary, so “I am not sure” is possible, if training made it likely</li></ul>}
        />
        <h3>What reduces hallucination, and why nothing removes it</h3>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>Approach</th><th>Why it helps</th><th>Why it is not a cure</th></tr></thead>
            <tbody>
              <tr><td><a href="#/lesson/rag">Retrieval (RAG)</a></td><td>Puts the relevant text into the context window, so the likely continuation is one that copies from it.</td><td>Retrieval can miss or fetch the wrong passage, and the model can still ignore or misread what it was given.</td></tr>
              <tr><td><a href="#/lesson/agents">Tools</a> (search, calculator, code)</td><td>Moves exact work such as arithmetic or lookups out of the weights into systems that are actually exact.</td><td>The model must decide to call the tool, call it correctly, and report the result faithfully. Each step is again next-token prediction.</td></tr>
              <tr><td>Training to abstain</td><td>Later training stages (next lessons) can make “I am not sure” the likely continuation when the model’s own signals of uncertainty are high.</td><td>The model’s sense of what it knows is imperfect. Push too far and it refuses things it does know.</td></tr>
            </tbody>
          </table>
        </div>
        <p>How well models “know what they know” is still research: a model’s internal probabilities carry real but imperfect information about whether its answer is correct, less so on unfamiliar topics, and common benchmarks may make things worse by scoring a confident guess above an honest “I do not know”.</p>
        <p>This is why production systems put <a href="#/lesson/rag">retrieval</a> and <a href="#/lesson/agents">tools</a> <em>around</em> the model for anything that must be current, exact or auditable, and why <a href="#/lesson/fine-tuning">fine-tuning</a> is a good way to change behaviour and a poor way to install facts. A model’s knowledge cut-off follows from the same point: the weights stopped changing when training ended. Reading the weights directly is the job of interpretability research, the subject of <a href="#/lesson/interpretability">Looking inside the model</a>; treat any “neuron for Paris” diagram as an illustration.</p>
        <p>Riya writes back to the support team: the bot did not look anything up, so it cannot be trusted with policy until it is given the real policy documents. Dev, to his credit, is the first to ask how.</p>
      </RealLLM>
    </Lesson>
  )
}
