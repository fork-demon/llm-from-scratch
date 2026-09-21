import { useState } from 'react'
import { Lesson, Why, Problem, MentalModel, TryIt, Numbers, TheMath, CodeIt, BreakIt, Exercises, CheckYourself, Remember, RealLLM } from '../components/lesson'
import { Callout, DeepDive, Equation, Flow, G, Term, ToyVsReal, WhyExists } from '../components/ui'
import { Code } from '../components/Code'
import { Exercise, ExplainBack } from '../components/exercise'
import { StageExplorer } from '../interactive/StageExplorer'
import { ChatTemplateViewer } from '../interactive/ChatTemplateViewer'
import { PreferenceLab } from '../interactive/PreferenceLab'

/* The loop the learner already knows, with what is new at scale for each step. */
const PIPELINE: { label: string; you: string; scale: string }[] = [
  { label: 'Dataset', you: 'One text file: about 1 MB of Shakespeare.', scale: 'Trillions of tokens gathered from the web, code, books and papers. Most of the engineering effort goes here: removing spam and boilerplate, removing near-duplicate pages, filtering for quality, choosing the mix of languages and code. The same architecture trained on better-filtered data gives a clearly better model.' },
  { label: 'Tokenizer', you: 'One token per character.', scale: 'A BPE tokenizer with roughly 30,000 to 250,000 tokens, trained once before anything else, plus a handful of reserved special tokens (end of text, and later the chat role markers you will meet below).' },
  { label: 'Batches', you: '32 random windows of 64 characters.', scale: 'Millions of tokens per batch. No single GPU can hold the model or the batch, so both are split across thousands of GPUs: each GPU works on its slice, and the gradients are averaged over the network before every update. Conceptually it is still one big batch and one update.' },
  { label: 'Forward', you: 'model(x, y) gives logits.', scale: 'The same forward pass, usually in 16-bit numbers to save memory and time.' },
  { label: 'Loss', you: 'Cross-entropy on the next character.', scale: 'Identical: cross-entropy on the next token. Nothing about the objective gets cleverer in pretraining.' },
  { label: 'Backprop', you: 'loss.backward()', scale: 'Identical, spread across the GPUs.' },
  { label: 'Optimizer', you: 'AdamW, lr = 3e-4, constant.', scale: 'AdamW is still the common choice, with a learning rate that warms up and then decays on a schedule. The run cannot be restarted from scratch if it diverges in week five, so stability tricks matter a great deal.' },
  { label: 'Checkpoint', you: 'None: the run takes minutes.', scale: 'The weights and optimizer state are saved regularly. With thousands of GPUs running for weeks, hardware failures are routine, and training resumes from the last checkpoint. Checkpoints are also what later stages start from: the “base model” is a checkpoint.' },
  { label: 'Evaluation', you: 'Validation loss, and reading samples.', scale: 'Validation loss still, plus benchmarks: fixed sets of questions on maths, code, knowledge and more. One serious trap is contamination: if benchmark questions leaked into the training data, the score measures memory, not ability. It is the train/validation split problem again, at the scale of the internet.' },
]

function PipelineAtScale() {
  const [i, setI] = useState(0)
  const step = PIPELINE[i]
  return (
    <div className="card">
      <p className="muted" style={{ fontSize: 14.5, marginBottom: 8 }}>Click a step to compare what you did in <a href="#/lesson/training-gpt">Training GPT</a> with what changes at scale.</p>
      <Flow horizontal steps={PIPELINE.map((p) => p.label)} active={i} onSelect={setI} />
      <div className="grid-2" style={{ marginTop: 10 }} aria-live="polite">
        <div><h4 style={{ fontSize: 15, marginBottom: 4 }}>Your tiny GPT</h4><p style={{ margin: 0 }}>{step.you}</p></div>
        <div><h4 style={{ fontSize: 15, marginBottom: 4 }}>A large model</h4><p style={{ margin: 0 }}>{step.scale}</p></div>
      </div>
    </div>
  )
}

export default function TrainingPipelineLesson() {
  return (
    <Lesson id="training-pipeline">
      <Why>
        <p className="lede">Suppose your tiny GPT had been trained on the whole internet instead of Shakespeare. You type:</p>
        <div className="card center" style={{ fontFamily: 'var(--serif)', fontSize: 22 }}>“What is the capital of France?”</div>
        <p>What comes back? Quite possibly: <em>“What is the capital of Spain? What is the capital of Italy?”</em></p>
        <p>That is not a bug. You trained the model to continue documents, and a quiz question is very often followed by more quiz questions. It learned exactly what you asked it to learn.</p>
        <p>Yet the assistants you use every day answer the question. Same architecture, same kind of weights. So what happened in between?</p>
        <Callout kind="idea">
          An assistant is made in stages. <b>Pretraining</b> produces a model that can continue any text. <b>Supervised fine-tuning</b> teaches it that the text to continue is a conversation in which it plays the helpful role. <b>Preference optimisation</b> teaches it which of many possible answers people actually prefer. Every stage is gradient descent on the same weights. Only the data and the loss change.
        </Callout>
      </Why>

      <Problem>
        <p>You already know the training loop from <a href="#/lesson/training-gpt">Training GPT</a>. At scale, the loop is the same. What surrounds it is not.</p>
        <PipelineAtScale />
        <Callout kind="established">
          Data quality is not a detail. Public reports from groups that train open models describe filtering and de-duplication as some of the most effective things they did, and controlled experiments show that removing duplicated text improves models and reduces verbatim memorisation. Public training sets are now in the trillions of tokens: Meta reports more than 15 trillion for Llama 3. Your Shakespeare file has about one million characters.
        </Callout>
        <p>The result of all that is called a <b>base model</b>. And that brings us to the real problem of this lesson.</p>
        <WhyExists
          problem="A base model continues text. Users want their question answered, in a helpful manner, and they want the model to stop afterwards."
          naive="Prompt engineering: write the beginning of a document in which a helpful answer is the likely continuation, for example a fake FAQ page or an interview transcript."
          fails="It is fragile. Every task needs its own disguise, the model may continue with another question or never stop, and nothing pushes it towards honesty or away from harmful output."
          idea="Keep training the same weights, first on demonstrations of good conversations, then on human judgements about which answers are better."
          tradeoff="The model now optimises for what demonstrators wrote and what raters liked. That is a stand-in for “good”, and stand-ins can be gamed: flattery, padding, and confident tone can score well."
        />
      </Problem>

      <MentalModel>
        <p>Think of three different kinds of teaching signal, each one used when the previous one runs out.</p>
        <div className="grid-3">
          <div className="card"><h4 style={{ fontSize: 16, marginBottom: 6 }}>1 · Read everything</h4><p><G t="pretraining">Pretraining</G>. The right answer at every position is the next token of a real document. No humans needed to label anything, so you can use trillions of tokens.</p></div>
          <div className="card"><h4 style={{ fontSize: 16, marginBottom: 6 }}>2 · Watch demonstrations</h4><p><G t="sft">Supervised fine-tuning</G>, usually shortened to SFT. People write example conversations. The right answer is the next token of the demonstrated reply. Expensive per example, so there are far fewer.</p></div>
          <div className="card"><h4 style={{ fontSize: 16, marginBottom: 6 }}>3 · Get feedback on your own attempts</h4><p>Preference optimisation. For “write a haiku” there is no single right next token. But a person can look at two attempts and say which is better.</p></div>
        </div>
        <Callout kind="analogy">
          A new support engineer first reads the whole wiki (pretraining), then shadows a senior colleague and copies how tickets are answered (SFT), then starts answering tickets and gets review comments: “this reply was better than that one” (preference feedback).
          <br /><br />
          Where the analogy stops: the engineer understands <em>why</em> a reply was preferred. The model only receives a number that moves its weights. If reviewers happen to favour long replies, the model gets longer, not wiser.
        </Callout>
        <Callout kind="model" label="Simplified mental model: knowledge first, behaviour after">
          A common summary is “pretraining gives knowledge, the later stages shape behaviour”. It is a good first approximation: the later stages use a tiny fraction of the data, and mostly change format, tone and choices. It is not a strict rule. Fine-tuning can teach some new facts, can degrade existing abilities, and newer reinforcement-learning stages (next lesson) appear to improve problem-solving, not just manners.
        </Callout>
        <p>Stage 2 needs one new piece of plumbing. A model takes one sequence of tokens. A conversation has roles and turns. How do you feed one into the other?</p>
        <Term
          name="Chat template"
          plain={<>A fixed recipe for flattening a list of messages into one long token sequence, using special marker tokens to say “a new message by this role starts here” and “this message ends here”.</>}
          example={<><code>&lt;|im_start|&gt;user⏎What is a cat?&lt;|im_end|&gt;⏎&lt;|im_start|&gt;assistant⏎</code> … and the model continues from there.</>}
          formal={<>A deterministic function from a list of (role, content) pairs to a token sequence. The markers are extra entries in the vocabulary whose embeddings are learned during fine-tuning. Each model family defines its own.</>}
        />
        <Callout kind="dev">A chat API is a thin wrapper. Your JSON list of messages is serialised with the template into one string, the model continues that string, and the server stops when the model emits the end marker. The model has no separate “system prompt channel”: the system message is only earlier tokens in the same sequence. The model was trained to give those tokens special weight, which is a learned habit, not an enforced boundary. That is one reason prompt injection is possible, as you will see in <a href="#/lesson/agents">Agents</a>.</Callout>
      </MentalModel>

      <TryIt title="Three stages, one template, and you as the rater">
        <h3>1. What each stage changes</h3>
        <p>Predict before you click: what does a base model do with an instruction?</p>
        <StageExplorer />

        <h3>2. How a conversation becomes training data</h3>
        <p>During SFT the model sees the entire conversation, but it is usually graded only on the assistant’s tokens. Why? We want it to learn to <em>write answers</em>, not to write users’ questions or system prompts.</p>
        <ChatTemplateViewer />

        <h3>3. When there is no right answer, only a better one</h3>
        <p>Now the third stage. Nobody can write “the correct haiku” into a loss function. But you can compare two answers. Here your comparisons train a small <b>reward model</b>: a function that gives any answer a score, fitted so that answers you preferred score higher.</p>
        <PreferenceLab />
        <p>Step 3 of that lab is the whole idea of <b>RLHF</b> in one table: a model that scores answers, and a second model that is pushed towards high-scoring answers while being held close to where it started.</p>
        <p>RLHF stands for reinforcement learning from human feedback, and it is the classic form of preference optimisation. “Reinforcement learning” is the part worth unpacking. Up to now, training meant showing the model the right next token. Here nobody knows the right answer. Instead the model produces something, a score comes back, and the weights move to make high-scoring output more likely. Learning from a mark, not from a worked example.</p>
      </TryIt>

      <Numbers>
        <p><b>First, the loss mask.</b> Take the default conversation from the template viewer. Flattened, it is 34 tokens in our toy tokenizer.</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>part</th><th>tokens</th><th>graded?</th></tr></thead>
            <tbody>
              <tr><td>system message with its markers</td><td className="mono">11</td><td>no</td></tr>
              <tr><td>user message with its markers</td><td className="mono">10</td><td>no</td></tr>
              <tr><td>assistant header: <code>&lt;|im_start|&gt;</code>, <code>assistant</code>, newline</td><td className="mono">3</td><td>no</td></tr>
              <tr><td>“A cat is a small furry animal.”</td><td className="mono">8</td><td><b>yes</b></td></tr>
              <tr><td>closing <code>&lt;|im_end|&gt;</code></td><td className="mono">1</td><td><b>yes</b>: the model must learn to stop</td></tr>
              <tr><td>final newline</td><td className="mono">1</td><td>no</td></tr>
            </tbody>
          </table>
        </div>
        <p>9 of 34 tokens are graded: 26%. The loss is the average <G t="cross-entropy">cross-entropy</G> over those 9 positions only. The other 25 tokens still flow through attention as context. They produce no gradient of their own.</p>

        <p><b>Second, one preference comparison.</b> Suppose the reward model currently scores answer A at 2.0 and answer B at 0.5. How confident is it that a person prefers A?</p>
        <div className="table-scroll">
          <table className="plain">
            <tbody>
              <tr><td>difference of rewards</td><td className="mono">2.0 − 0.5 = 1.5</td></tr>
              <tr><td>squash to a probability</td><td className="mono">1 / (1 + e<sup>−1.5</sup>) = 1 / (1 + 0.223) = 0.818 ≈ <b>0.82</b></td></tr>
              <tr><td>loss if the person did prefer A</td><td className="mono">−ln(0.818) = <b>0.20</b> (small: it expected this)</td></tr>
              <tr><td>loss if the person preferred B</td><td className="mono">−ln(1 − 0.818) = −ln(0.182) = <b>1.70</b> (large: it was surprised)</td></tr>
            </tbody>
          </table>
        </div>
        <p>This is the same “minus log of the probability given to what actually happened” that you have used since <a href="#/lesson/softmax">softmax</a>. Only the event changed: not “which token came next” but “which answer did the person pick”.</p>
        <p><b>Third, one update.</b> In the lab, all three weights start at 0. In comparison 1 you prefer “Paris.” with features <span className="mono">[0.03, 0, 1]</span> over the long non-answer with features <span className="mono">[1, 1, 0]</span>. The model gave A a probability of 0.5, so the surprise is 1 − 0.5 = 0.5. Each weight moves by 0.5 × (chosen feature − rejected feature):</p>
        <p className="mono center">w = [0.5 × (0.03 − 1), 0.5 × (0 − 1), 0.5 × (1 − 0)] = [−0.48, −0.50, +0.50]</p>
        <p>Click “Paris.” in a freshly reset lab and you will see exactly these weights.</p>
      </Numbers>

      <TheMath>
        <p>You have already computed it. Written down, the reward model’s assumption about people is:</p>
        <Equation
          label="Probability that A is preferred over B equals sigmoid of reward of A minus reward of B"
          symbols={[
            ['r(A), r(B)', 'the reward model’s scores for the two answers: any real numbers'],
            ['σ', <>the sigmoid function, σ(z) = 1 / (1 + e<sup>−z</sup>): squashes any number into the range 0 to 1. It is <a href="#/lesson/softmax">softmax</a> for the special case of two options.</>],
            ['P(A ≻ B)', 'the predicted probability that a person prefers A to B'],
            ['loss', 'minus the log of the probability given to the choice the person actually made, averaged over all comparisons'],
          ]}
        >
          P(A ≻ B) = σ( r(A) − r(B) ) &nbsp;&nbsp;&nbsp; loss = −log σ( r(chosen) − r(rejected) )
        </Equation>
        <p>Notice that only the <em>difference</em> of rewards matters. A reward of 7 means nothing on its own. It only means “better than an answer that scores 5”.</p>
        <p>This way of turning pairwise choices into scores is called the <b>Bradley-Terry model</b>, and it is decades older than LLMs (Bradley and Terry, 1952). It was designed for experiments where people compare items two at a time. The same model ranks sports teams and chess players from match results.</p>
        <p>Then the policy (the language model) is tuned. The objective has two parts pulling in opposite directions:</p>
        <Equation
          label="Maximise expected reward minus beta times the KL divergence from the SFT model"
          symbols={[
            ['reward', 'the reward model’s score for an answer the model itself generated'],
            ['KL', 'Kullback-Leibler divergence: one number saying how far apart two sets of probabilities are. 0 when the tuned model and the SFT model agree on every token, growing as they drift apart. Read it as “distance from where we started”.'],
            ['β', 'the strength of the leash. Large β: stay close to the SFT model. Small β: chase reward freely.'],
          ]}
        >
          maximise &nbsp; average reward &nbsp;−&nbsp; β × KL( tuned model ‖ SFT model )
        </Equation>
        <p><b>Why the leash?</b> The reward model was only trained on answers that look like what the SFT model writes. Far away from those, its scores are unreliable guesses. Without the leash, the optimiser wanders off to strange text that happens to score well. You saw that in step 3 of the lab with β near 0.1.</p>
        <Term
          name="Reinforcement learning, in the four words you need"
          plain={<>Learning from a score for what you did, instead of from a worked example of what you should have done.</>}
          example={<>The <b>policy</b> is the language model. The <b>state</b> is the text so far. An <b>action</b> is choosing the next token. The <b>reward</b> is one number that arrives only after the whole answer is finished. (In practice the KL penalty below is often charged token by token along the way.)</>}
          formal={<>Adjust the policy’s parameters to increase the expected reward of sequences sampled from the policy itself. Because the reward arrives at the end, every token of a well-scored answer is made a little more likely, and every token of a badly scored one a little less.</>}
        />
        <DeepDive title="DPO: the same data without a reward model or a reinforcement learning loop">
          <p>RLHF as described has many moving parts: a separate reward model, sampling answers from the model during training, and a reinforcement learning algorithm to turn those scores into weight updates. The classic choice is PPO, proximal policy optimization, which is effective and fussy to tune.</p>
          <p>DPO, direct preference optimization (Rafailov et al., 2023), starts from a mathematical observation. For the objective above, the best possible policy can be written in closed form: policy(y) ∝ SFT(y) × e<sup>r(y)/β</sup>. (That is precisely the formula the lab uses for its “tuned model” column.) Turn that around and the reward can be expressed through the policy: r(y) = β × log( policy(y) / SFT(y) ), plus a term that cancels when you subtract two rewards.</p>
          <p>Substitute that into the Bradley-Terry loss and the reward model disappears:</p>
          <p className="mono" style={{ fontSize: 14 }}>loss = −log σ( β × [ log(π(chosen)/π_ref(chosen)) − log(π(rejected)/π_ref(rejected)) ] )</p>
          <p>Here π is the model being tuned and π_ref is the frozen SFT model. In words: make the chosen answer more likely, relative to the reference, than the rejected answer. It is an ordinary supervised loss on fixed pairs: no sampling, no separate reward network.</p>
          <p>Trade-off: DPO learns only from the fixed pairs in the dataset, not from fresh attempts of the current model. Which approach gives better models, and when, is an active debate, and many published recipes combine several methods.</p>
        </DeepDive>
        <DeepDive title="How does a gradient flow through “sample an answer, then score it”?">
          <p>Sampling a token is not differentiable, so you cannot backpropagate from the reward into the weights directly. The basic trick (the policy gradient) is: sample an answer, look at its reward, and then take a gradient step on the <em>log-probability of the tokens you sampled</em>, scaled by how much better than average the reward was.</p>
          <p>Good answer: its tokens become more likely. Bad answer: less likely. It is the cross-entropy gradient you know, with a sign and size set by the reward. Practical algorithms such as PPO add machinery to keep these noisy updates small and stable.</p>
        </DeepDive>
      </TheMath>

      <CodeIt>
        <p>Here is the heart of the loop you ran in <code>tiny_gpt.py</code>. Pretraining a large model is this loop, distributed.</p>
        <Code source="phase3-transformers/tiny_gpt.py" title="the training loop, from main()">{`
for step in range(steps + 1):
    x, y = get_batch("train")
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()          # <- module 03, automated
    opt.step()               # <- module 02, with adaptive step sizes
`}</Code>
        <p>And the loss inside <code>model(x, y)</code>:</p>
        <Code source="phase3-transformers/tiny_gpt.py" title="GPT.forward: every position is graded">{`
loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                       targets.view(-1))
`}</Code>
        <p><b>What would change for SFT?</b> Not the model, not the optimizer. Two things: <code>get_batch</code> returns templated conversations instead of random windows of Shakespeare, and the targets at non-assistant positions are blanked out. The repository does not contain this step, so treat it as a sketch of the diff:</p>
        <Code title="sketch: SFT = the same loss with a mask">{`
# assistant_mask: (B, T) bool, True where the TARGET token belongs
# to an assistant message (or is its closing <|im_end|>)
targets = targets.masked_fill(~assistant_mask, -100)

loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                       targets.view(-1),
                       ignore_index=-100)   # skip those positions
`}</Code>
        <p><code>ignore_index</code> is a standard PyTorch option: positions whose target equals that value contribute nothing to the loss or the gradient. The average is taken over the remaining positions, as in the 9-of-34 example above.</p>
        <p>The reward model’s loss is the equation from the last section, one line:</p>
        <Code title="sketch: reward model loss on a batch of comparisons">{`
r_chosen   = reward_model(prompt, chosen)      # (B,) one score per answer
r_rejected = reward_model(prompt, rejected)    # (B,)
loss = -F.logsigmoid(r_chosen - r_rejected).mean()
`}</Code>
        <p>A reward model is typically the same Transformer you know, with the vocabulary-sized output head replaced by a head that produces a single number. Want a real fine-tuning run you can execute? That comes in <a href="#/lesson/fine-tuning">Fine-tuning</a>, with <code>finetune_tiny_gpt.py</code>.</p>
      </CodeIt>

      <BreakIt>
        <p>Predict first, then check.</p>
        <ul>
          <li><b>Be a biased rater.</b> In the preference lab, reset and always pick the longer answer (or use the simulated rater). The “contains the answer” weight ends near 0 (−0.09) while length reaches 1.57. With β = 0.5 the tuned model then puts most of its probability on a long, polite reply that never says “Paris”.</li>
          <li><b>Notice the accident.</b> In that same run the “polite words” weight also rose to 1.51, although the rater never cared about politeness. In this data the longer answers tend to be the polite ones. A reward model cannot tell what you cared about. It only sees what your choices correlate with.</li>
          <li><b>Tighten the leash.</b> Keep the length-loving reward and slide β up to 5. The tuned model returns close to the SFT model. The leash limits damage from a flawed reward. It also limits improvement from a good one.</li>
          <li><b>Remove the mask.</b> In the template viewer, untick “assistant tokens only”. Now the model is also trained to write system prompts and user questions. Often that is merely wasteful. With many short replies to long user texts, most of the gradient is spent on imitating users.</li>
          <li><b>Delete the assistant message’s text.</b> One target remains: <code>&lt;|im_end|&gt;</code>. You would be teaching the model that the best reply is to stop immediately.</li>
        </ul>
      </BreakIt>

      <Exercises>
        <Exercise
          id="training-pipeline-predict-base"
          type="predict"
          title="Prompting a base model"
          hints={[
            'A base model continues documents. In what kind of document does this text appear, and what usually comes next there?',
            'Think about where “Translate to French: good morning” shows up on the web: exercise sheets, forum posts, lists of prompts.',
            'To get a translation out of a base model, write the start of a document where the translation is the natural continuation, for example a two-column phrase list with a few filled-in rows.',
          ]}
          solution={<><p>Likely continuations: more exercise lines (“Translate to French: good evening”), or a forum-style reply, or sometimes the translation. It depends on which kind of document the model guesses it is in, and that varies from sample to sample.</p><p>The pre-SFT fix is a few-shot prompt: “English: thank you / French: merci / English: good morning / French:”. Now the translation really is the most plausible next text. SFT makes this disguise unnecessary by training the model that an instruction is followed by its fulfilment.</p></>}
        >
          <p>You give a <em>base</em> model the text <code>Translate to French: good morning</code>. Predict two different continuations it might plausibly write. Then: how would you rewrite the prompt so that a base model is likely to produce the translation?</p>
        </Exercise>

        <Exercise
          id="training-pipeline-calc-bt"
          type="calculate"
          title="How sure is the reward model?"
          answer={{ value: 0.73, tolerance: 0.01 }}
          answerLabel="P(A preferred), two decimals"
          hints={[
            'Only the difference of the two rewards matters.',
            'The difference is 1.2 − 0.2 = 1.0. Now apply the sigmoid: 1 / (1 + e⁻¹).',
            'e⁻¹ ≈ 0.368, so P = 1 / 1.368.',
          ]}
          solution={<><p>P = σ(1.0) = 1 / (1 + 0.368) ≈ <b>0.73</b>.</p><p>If both rewards were shifted by the same amount (say 101.2 and 100.2), the answer would be identical. A reward model defines a ranking scale, not absolute quality. This is also why reward scores from two different reward models cannot be compared.</p></>}
        >
          <p>A reward model gives answer A a reward of 1.2 and answer B a reward of 0.2. According to the Bradley-Terry formula, what is the probability that a person prefers A?</p>
        </Exercise>

        <Exercise
          id="training-pipeline-debug-template"
          type="debug"
          title="The fine-tuned model that talks to itself"
          hints={[
            'Compare what the model saw during training with what it is given at inference. Look at the very end of the prompt.',
            'During SFT, every assistant answer came right after the tokens <|im_start|>assistant and a newline. What follows the user message in the inference prompt here?',
          ]}
          solution={<><p>The inference prompt stops after the user’s <code>&lt;|im_end|&gt;</code>. It never opens the assistant turn. The model must continue a sequence that, in training, was always followed by a header. It may emit a header itself, possibly <code>&lt;|im_start|&gt;user</code>, and carry on writing the user’s side.</p><p>The fix is to end the prompt with <code>&lt;|im_start|&gt;assistant</code> and a newline (the “generation prompt”), and to stop generation at <code>&lt;|im_end|&gt;</code>. More generally: the template at inference must match the template used in training, character for character. A mismatch does not crash. It quietly degrades quality, which makes it a nasty bug. Libraries ship the template together with the model for this reason.</p></>}
        >
          <p>A colleague fine-tunes a model on ChatML-style conversations. At inference they build the prompt below. The model often replies with another question, as if it were the user, or writes both sides of a dialogue. What is wrong?</p>
          <Code lang="text">{`
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is a cat?<|im_end|>
`}</Code>
        </Exercise>

        <Exercise
          id="training-pipeline-experiment-hack"
          type="experiment"
          title="Find the smallest bias that breaks it"
          hints={[
            'First be a careful rater: always pick an answer that contains the correct answer, and when both do, pick the more thorough one. Your clicks should be A, A, B, B, B, A, B, B. Check the readout in step 3.',
            'In comparisons 1, 5 and 7 the long answer never states the answer. Those are the places where a rater in a hurry slips. Reset, and slip in only some of them.',
            'Try slipping only in comparisons 5 and 7: clicks A, A, B, B, A, A, A, B.',
          ]}
          solution={<><p>The careful rater ends with weights of about [0.10, 0.67, 1.74]: “contains the answer” dominates, and the tuned model says “Paris” 99% of the time (the SFT model: 90%).</p><p>Slip only in comparisons 5 and 7 and the chance of “Paris” falls to 41%: the tuned model’s favourite reply, at 55%, is the long polite one that never answers. Slip in all three and you have reproduced the “always longer” rater exactly, because in the other five comparisons the longer answer is also a correct one. The chance of “Paris” is then 20%.</p><p>Two lazy clicks out of eight were enough. The reward model fits the raters’ actual choices, shortcuts included. Real raters work under time pressure, and a long, confident, agreeable answer is easy to mistake for a good one.</p></>}
        >
          <p>In the preference lab, keep β = 0.5. How few “lazy” choices (picking an answer because it looks more thorough, although it never states the answer) does it take before the tuned model is more likely than not to give a reply without “Paris”? Try it by hand, and watch the readout under step 3.</p>
        </Exercise>

        <ExplainBack
          id="training-pipeline-explain"
          prompt="A teammate says: “ChatGPT-style models are a different kind of neural network from the GPT we built, because they follow instructions and ours only continues text.” Correct them in three or four sentences: what is the same, what differs, and why can comparison data teach something demonstrations cannot?"
          modelAnswer={<p>The network is the same kind: a Transformer that outputs a probability for the next token, and an assistant’s reply is produced by the same sampling loop. What differs is the training data and loss applied to the same weights after pretraining. Supervised fine-tuning continues next-token training on conversations flattened by a chat template, graded on the assistant’s tokens, so “a question is followed by a helpful answer, then a stop token” becomes the likely continuation. For many prompts there is no single correct reply to demonstrate, but people can say which of two replies is better, so a reward model is fitted to those comparisons and the model is pushed towards higher-scoring replies while being held close to the SFT model (or, with DPO, trained on the pairs directly). Because the reward is only a stand-in for what people want, the model can learn to please the scorer, for example with length or flattery.</p>}
        />
      </Exercises>

      <CheckYourself
        questions={[
          {
            q: 'A base model answers “What is the capital of France?” with more quiz questions. What is the best description of what went wrong?',
            options: ['The model does not know the capital of France', 'Nothing went wrong: more questions are a plausible continuation of that text, and continuing text is all it was trained to do', 'The sampling temperature was too high', 'The context window was too short'],
            answer: 1,
            explain: 'The knowledge is very likely in the weights. What is missing is the behaviour: treating the text as a request to fulfil.',
          },
          {
            q: 'From the model’s point of view, what is a multi-turn conversation with a system prompt?',
            options: ['Three separate inputs fed into three different parts of the network', 'A database of messages it can query', 'One flat sequence of tokens, with special marker tokens showing where each role’s message starts and ends', 'A sequence of independent calls that share hidden state between them'],
            answer: 2,
            explain: 'The chat template serialises everything into one sequence. The model then does next-token prediction on it, as always.',
          },
          {
            q: 'Why is the SFT loss usually computed only on the assistant’s tokens?',
            options: ['Because the model cannot see the user’s tokens', 'Because we want gradient spent on learning to write good replies, not on learning to imitate users and system prompts', 'Because user tokens have no embeddings', 'Because it makes the forward pass faster'],
            answer: 1,
            explain: 'The user’s tokens are still read as context through attention. They are not used as prediction targets. The forward pass costs the same.',
          },
          {
            q: 'Why does preference optimisation use comparisons (“A is better than B”) and not more demonstrations?',
            options: ['Comparisons need no human effort', 'For many prompts there is no single correct reply to demonstrate, but people can reliably judge which of two replies is better, including replies the model itself produced', 'Demonstrations cannot be tokenized', 'Comparisons let the model skip backpropagation'],
            answer: 1,
            explain: 'Judging is easier than writing, and feedback on the model’s own attempts targets the mistakes it actually makes.',
          },
          {
            q: 'In RLHF, what is the KL penalty (the “leash” to the SFT model) for?',
            options: ['It makes the model forget its pretraining', 'It speeds up sampling', 'It keeps the tuned model in the region where the reward model’s scores are trustworthy, limiting the damage when the optimiser finds flaws in the reward', 'It guarantees the model never produces a wrong answer'],
            answer: 2,
            explain: 'A learned reward is only reliable near the kind of text it was trained on. Optimise hard enough against any imperfect scorer and you find its blind spots.',
          },
        ]}
      />

      <Remember
        items={[
          <>At scale the loop is unchanged: batch, forward, <G t="loss">loss</G>, backprop, update. What is new surrounds it: <b>data filtering and de-duplication</b>, distribution over many GPUs, checkpoints, and evaluation that must guard against contamination.</>,
          <><b>Pretraining</b> gives a base model: it continues text and holds most of the knowledge. It does not follow instructions, and that is not a defect.</>,
          <><b>SFT</b> is the same next-token loss on demonstrations. A <b>chat template</b> flattens the conversation into one token sequence with role markers, and the loss is usually counted only on assistant tokens, including the end marker.</>,
          <><b>Preference optimisation</b> exists because many prompts have no single right answer but people can compare two. RLHF fits a reward model with P(A ≻ B) = σ(r(A) − r(B)), then tunes the model against it on a <b>KL leash</b>. DPO uses the same pairs with no separate reward model.</>,
          <>The reward is a <b>stand-in</b> for what people want. Optimising a stand-in invites reward hacking: length, flattery and confident tone can win. These stages shape behaviour. They do not install a truth checker.</>,
        ]}
      />

      <RealLLM>
        <Flow horizontal steps={[{ label: 'Pretraining', sub: 'base model' }, { label: 'SFT', sub: 'follows the chat format' }, { label: 'Preference optimisation', sub: 'RLHF, DPO and relatives' }, { label: 'RL on checkable tasks', sub: 'next lesson' }]} active={2} />
        <ToyVsReal
          toy={<ul><li>Illustrative, hand-written outputs for each stage</li><li>A toy tokenizer and one ChatML-style template</li><li>A reward model with 3 readable features and 8 comparisons</li><li>The tuned policy computed exactly over 5 candidate answers</li></ul>}
          real={<ul><li>Base and instruction-tuned versions of many open models are both published, so you can observe the difference yourself</li><li>Each model family has its own template and special tokens, shipped with the tokenizer</li><li>The reward model is a full Transformer trained on tens of thousands of comparisons or more, increasingly with AI-generated feedback guided by written principles</li><li>The policy is tuned by many small gradient steps on its own sampled answers, often in several rounds mixing methods</li></ul>}
        />
        <Callout kind="established">
          The three-stage recipe is publicly documented. OpenAI’s InstructGPT paper (2022) describes SFT, a reward model fitted to human comparisons, and RL with a KL penalty, and reports that human raters preferred the outputs of a tuned model with 1.3 billion parameters to those of the untuned GPT-3 with 175 billion, on the prompts the authors collected. Reports for open model families such as Llama describe similar pipelines using combinations of SFT, rejection sampling (sample several answers, keep the best-scored ones as new training examples), PPO and DPO. For example, the Llama 2 paper describes rejection sampling and PPO, and the Llama 3 paper describes rejection sampling and DPO.
        </Callout>
        <Callout kind="model" label="Simplified mental model: three clean stages">
          “Pretrain, then SFT, then preference tuning” is the textbook order. Real pipelines blur it: several rounds, synthetic data written by other models, instruction-like data mixed into late pretraining, and many variants of the preference loss. The exact recipes and data of commercial assistants are mostly not public, so do not trust anyone’s detailed diagram of them, including a confident-sounding one from a model.
        </Callout>
        <Callout kind="research">
          How to get a trustworthy training signal is an open problem. Published studies have found that preference-tuned models can become sycophantic (agreeing with the user’s stated view) and that human and learned reward signals both tend to favour longer answers. Whether these stages can make a model reliably <em>honest</em>, and how to supervise models on tasks where human raters cannot easily tell a good answer from a convincing one, are active research areas. Hallucination is reduced by training models to express uncertainty, and not removed, for the structural reasons in <a href="#/lesson/why-llms-know">Why LLMs know things</a>.
        </Callout>
        <p>One stage is still missing from this picture. When the task has a checkable answer, such as a maths result or code that must pass tests, the reward does not need to be learned from people at all. That changes what RL can do, and it is the subject of the <a href="#/lesson/reasoning-models">next lesson</a>.</p>
      </RealLLM>
    </Lesson>
  )
}
