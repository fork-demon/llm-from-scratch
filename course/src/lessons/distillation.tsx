import { Lesson, Why, Problem, MentalModel, TryIt, Numbers, TheMath, CodeIt, BreakIt, Exercises, CheckYourself, Remember, RealLLM } from '../components/lesson'
import { Callout, DeepDive, Equation, Flow, G, Term, ToyVsReal, WhyExists } from '../components/ui'
import { Code } from '../components/Code'
import { Exercise, ExplainBack } from '../components/exercise'
import { DistillLab } from '../interactive/DistillLab'

export default function DistillationLesson() {
  return (
    <Lesson id="distillation">
      <Why>
        <p className="lede">The support team has a request. They want the bot on their own laptops: for the ride home, for the days the office Wi-Fi dies, and for tickets whose data should never leave the building.</p>
        <p>Riya checks the model they use today. Hundreds of billions of <G t="parameters">parameters</G>. It lives on a rack of GPUs, not on a laptop with 16 GB of memory.</p>
        <p>“So train a small one,” Dev says. “Same data, fewer parameters. Done.”</p>
        <p>Kabir shakes his head. “A small model learns less from that data on its own. Let the big one teach it.”</p>
        <p>Riya thinks of her tuition master in Mysuru. School gave her ticks and crosses. He would tap her notebook and say: “Option B is nearly right, you mixed up two formulas. Option D? Never.” She learned more from those margins than from any answer key.</p>
        <p>So what does a big model know that it can hand to a small one, beyond the right answer?</p>
        <p><b>Distillation</b> trains a small <em>student</em> model to imitate a big <em>teacher</em> model: either its full set of probabilities for the next token, or the text it writes. The teacher’s “nearly right” and “clearly wrong” are information. A plain answer key throws them away.</p>
      </Why>

      <Problem>
        <p>Look at what ordinary training gives the model. In <a href="#/lesson/training-gpt">Training GPT</a>, every position had one correct next token, taken from real text. After “The cat sat on the”, the data says <b>mat</b>. That is all it says.</p>
        <p>It does not say that “sofa” and “floor” would also have been fine, or that “moon” would be absurd. A label like that is called a <b>hard label</b>: one right answer, everything else equally wrong.</p>
        <p>A big model trained on trillions of tokens has seen “sat on the” thousands of times. Its <G t="softmax">softmax</G> output already holds the whole picture: mat likely, sofa and floor plausible, moon nearly impossible. That picture cost a fortune in GPU time to learn.</p>
        <WhyExists
          problem="You need a small, cheap model that behaves as much as possible like a big one."
          naive="Train the small model from scratch on the same data, with the same hard labels."
          fails="The small model has less capacity, and each hard label tells it one token per position. It has to rediscover, from raw text, every similarity the big model already found. It ends up clearly weaker."
          idea="Use the big model as the source of targets. Train the student to match the teacher’s probabilities (or the text the teacher writes), so each example carries the teacher’s judgement about every option."
          tradeoff="The student can only become as good as its teacher, including the teacher’s mistakes and blind spots. You also need to run the teacher over all the training data, which is expensive in its own right."
        />
        <p>Dev tries one more angle. “Isn’t that a zip file for models?” Not quite. The student has its own architecture and its own weights, often a different shape entirely. It copies the teacher’s <em>behaviour</em>, through training, never its weights. (Shrinking the weights themselves is a different family of tricks, in <a href="#/lesson/making-models-cheaper">Making the model itself cheaper</a>.)</p>
      </Problem>

      <MentalModel>
        <Term
          name="Soft targets"
          plain={<>The teacher’s full list of probabilities, used as the thing the student should reproduce, instead of one correct token.</>}
          example={<>Instead of “the answer is <b>mat</b>”, the student is told: mat 70%, sofa 12%, floor 10%, bed 6%, roof 2%, moon 0.2%.</>}
          formal={<>A probability distribution p over the <G t="vocabulary">vocabulary</G>, produced by the teacher for a given context. The student is trained to make its own distribution q close to p.</>}
        />
        <p>Why would soft targets teach more? Look at what one example carries. A hard label names one token. A soft target also ranks every other token. “Sofa is much closer than moon” is a fact about the world, and the teacher gives it away for free with every example.</p>
        <p>Geoffrey Hinton, who popularised the method with Oriol Vinyals and Jeff Dean in 2015, called this hidden information <b>dark knowledge</b>: knowledge that lives in the small probabilities, not in the top answer.</p>
        <Callout kind="analogy">
          Amma, on the phone, recognises it at once. “An answer key only says what is right. When I corrected forty notebooks, I wrote in the margins: <em>close, sign error</em>, or <em>this is not even the right chapter</em>. The children learned from the margins.”
          <br /><br />
          Where the analogy stops: Amma’s margins explained <em>why</em>. The teacher model explains nothing. Its “margins” are only numbers, one probability per token. The student learns the pattern in those numbers, not a reason.
        </Callout>

        <h3>Turning up the volume on small numbers: temperature</h3>
        <p>There is a catch. A confident teacher puts 99% on the top token. The interesting part, “sofa beats moon”, then hides in numbers like 0.3% versus 0.001%. In the loss, those barely register.</p>
        <p>You already know the fix from <a href="#/lesson/inference">Inference</a>: <G t="temperature">temperature</G>. Divide the teacher’s <G t="logits">logits</G> by a number T greater than 1 before the softmax. The ranking stays the same, but the distribution flattens, and the small probabilities become large enough to learn from.</p>
        <p>In sampling, temperature changes what the model <em>writes</em>. Here it changes what the student <em>sees</em>: a magnified view of the teacher’s low-probability opinions. The student is run at the same temperature during training, then at T = 1 when it is used.</p>

        <h3>Three ways to learn from a teacher</h3>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>name</th><th>what the student learns from</th><th>what you need</th></tr></thead>
            <tbody>
              <tr><td><b>Logit distillation</b> (token-level)</td><td>The teacher’s full distribution at every position of the training text.</td><td>The teacher’s probabilities, so you must be able to run it yourself.</td></tr>
              <tr><td><b>Sequence-level distillation</b> (Kim and Rush, 2016)</td><td>Answers the teacher wrote, with ordinary <G t="sft">SFT</G>.</td><td>Only the teacher’s text, even through an API.</td></tr>
              <tr><td><b>On-policy distillation</b></td><td>Its own answers, with the teacher scoring every token: “I would have given this token 40%, you gave it 90%”.</td><td>The teacher’s probabilities on the student’s text. The student learns from its own mistakes.</td></tr>
            </tbody>
          </table>
        </div>
        <p>The second is the one you will meet most. It has a name you have probably heard already: <b>synthetic data</b>.</p>

        <h3>When the teacher writes the textbook</h3>
        <p>A synthetic data pipeline is sequence-level distillation run as a factory:</p>
        <Flow horizontal steps={[{ label: 'Prompts', sub: 'collected or generated' }, { label: 'Teacher writes', sub: 'often several answers each' }, { label: 'Filter', sub: 'tests, checkers, a judge model' }, { label: 'Student trains', sub: 'ordinary SFT' }]} />
        <p>The filter is where most of the quality comes from. For maths, keep answers whose final number is right. For code, keep programs that pass tests. For everything else, a judge model scores the answers and the low scorers are dropped.</p>
        <p>This is how a large share of today’s fine-tuning data is made. It is cheap, it scales, and it has real risks:</p>
        <ul>
          <li><b>Model collapse.</b> Train a model on its own output, then the next one on <em>that</em> output, with no fresh human data. Rare things disappear first: a phrase with 1% probability may not appear in the sample at all, and a model fitted to that sample gives it 0%. Each generation loses more of the tails and drifts towards bland, repetitive text.</li>
          <li><b>Contamination and inherited errors.</b> A teacher that saw benchmark questions can quietly reproduce them, so the student scores well on a test it has half memorised (see <a href="#/lesson/evals">Evals</a>). The student also inherits the teacher’s wrong facts, biases and habits, with no human in the loop to catch them.</li>
        </ul>
        <Callout kind="research">
          Model collapse was shown in controlled experiments by Shumailov and colleagues (Nature, 2024) for models trained <em>repeatedly</em> on their own output with the real data <em>replaced</em>. Follow-up work (Gerstgrasser and colleagues, 2024) found that when synthetic data is <em>added</em> to the real data rather than replacing it, the degradation largely goes away. How much synthetic data is safe, and in what mix, is an open question that every lab answers empirically.
        </Callout>
      </MentalModel>

      <TryIt title="Teach a student three ways">
        <p>Start with soft targets at T = 4, then switch to the hard label and drag the steps to 200. Before you do: what will happen to the distance between student and teacher if the student only ever hears “mat”?</p>
        <DistillLab />
      </TryIt>

      <Numbers>
        <p>Kabir writes three small calculations on the whiteboard. Riya checks each one in Python before she believes it.</p>
        <p><b>1. Temperature.</b> A teacher has logits <span className="mono">[3, 1, −1]</span> for mat, sofa and moon.</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th /><th>mat</th><th>sofa</th><th>moon</th><th>sofa ÷ moon</th></tr></thead>
            <tbody>
              <tr><td>T = 1: e<sup>3</sup>, e<sup>1</sup>, e<sup>−1</sup> = 20.09, 2.72, 0.37</td><td className="mono"><b>0.867</b></td><td className="mono"><b>0.117</b></td><td className="mono"><b>0.016</b></td><td className="mono">7.39</td></tr>
              <tr><td>T = 2: logits become [1.5, 0.5, −0.5]</td><td className="mono"><b>0.665</b></td><td className="mono"><b>0.245</b></td><td className="mono"><b>0.090</b></td><td className="mono">2.72</td></tr>
            </tbody>
          </table>
        </div>
        <p>At T = 1, moon gets 1.6%, too small to matter much in a loss. At T = 2 it gets 9%. The order is unchanged: mat, then sofa, then moon. Only the gaps shrink, so the student now has to get the small ones right too.</p>

        <p><b>2. How far is the student from the teacher?</b> Take a teacher p = <span className="mono">[0.7, 0.2, 0.1]</span> and a student q = <span className="mono">[0.5, 0.3, 0.2]</span>. The distance used in distillation is the <b>KL divergence</b> (you met it as the “leash” in <a href="#/lesson/training-pipeline">From raw text to assistant</a>). For each token: teacher probability × log(teacher ÷ student).</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>token</th><th>p</th><th>q</th><th>p × ln(p ÷ q)</th></tr></thead>
            <tbody>
              <tr><td>mat</td><td className="mono">0.7</td><td className="mono">0.5</td><td className="mono">0.7 × ln 1.4 = 0.7 × 0.3365 = 0.2355</td></tr>
              <tr><td>sofa</td><td className="mono">0.2</td><td className="mono">0.3</td><td className="mono">0.2 × ln 0.667 = 0.2 × (−0.4055) = −0.0811</td></tr>
              <tr><td>moon</td><td className="mono">0.1</td><td className="mono">0.2</td><td className="mono">0.1 × ln 0.5 = 0.1 × (−0.6931) = −0.0693</td></tr>
              <tr><td colSpan={3}><b>KL(p ‖ q)</b></td><td className="mono"><b>0.0851</b></td></tr>
            </tbody>
          </table>
        </div>
        <p>Some terms are negative, but the total never is. It is 0 only when q equals p exactly.</p>
        <p>Compare the loss a hard label would give. The label is “mat”, so the loss is <G t="cross-entropy">cross-entropy</G> on one token: −ln 0.5 = <b>0.693</b>. It says nothing about sofa or moon.</p>

        <p><b>3. What each loss tells the student to change.</b> For softmax, the gradient on the student’s logits is always “student probabilities minus target”.</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th /><th>mat</th><th>sofa</th><th>moon</th></tr></thead>
            <tbody>
              <tr><td>hard label: q − [1, 0, 0]</td><td className="mono">−0.5</td><td className="mono">+0.3</td><td className="mono">+0.2</td></tr>
              <tr><td>soft target: q − p</td><td className="mono">−0.2</td><td className="mono">+0.1</td><td className="mono">+0.1</td></tr>
            </tbody>
          </table>
        </div>
        <p>The hard label pushes sofa down <em>harder</em> than moon (0.3 against 0.2), only because the student currently rates sofa higher. It has no idea sofa is nearly right. The soft target pushes each token towards the teacher’s own number. Keep taking hard-label steps and every token except mat heads towards 0. That is exactly the rising dashed curve in the lab.</p>
      </Numbers>

      <TheMath>
        <p>First, softmax with a temperature, which you have used since <a href="#/lesson/softmax">softmax</a>:</p>
        <Equation
          label="Probability of token i at temperature T"
          symbols={[
            [<>z<sub>i</sub></>, 'the logit (raw score) the model gives token i'],
            ['T', 'the temperature: 1 is normal, larger than 1 flattens, smaller than 1 sharpens'],
            [<>p<sub>i</sub>(T)</>, 'the resulting probability of token i; all of them add up to 1'],
          ]}
        >
          p<sub>i</sub>(T) = e<sup>z<sub>i</sub>/T</sup> ÷ Σ<sub>j</sub> e<sup>z<sub>j</sub>/T</sup>
        </Equation>
        <p>Then the distillation loss. Teacher logits give p(T), student logits give q(T), both at the same temperature:</p>
        <Equation
          label="Distillation loss equals T squared times the KL divergence from teacher to student"
          symbols={[
            ['p(T)', 'the teacher’s probabilities at temperature T: the soft targets'],
            ['q(T)', 'the student’s probabilities at the same temperature'],
            ['KL(p ‖ q)', 'Σ p_i × ln(p_i ÷ q_i): how far the student is from the teacher, 0 when they agree exactly'],
            ['T²', 'a correction so that changing T does not shrink the gradients (see the deep dive)'],
          ]}
        >
          loss<sub>KD</sub> = T² × KL( p(T) ‖ q(T) ) = T² × Σ<sub>i</sub> p<sub>i</sub>(T) ln( p<sub>i</sub>(T) ÷ q<sub>i</sub>(T) )
        </Equation>
        <p>Often this is mixed with the ordinary hard-label loss on the real next token: total = α × loss<sub>KD</sub> + (1 − α) × cross-entropy, with α somewhere between 0 and 1.</p>
        <p>One useful identity. Cross-entropy with soft targets, −Σ p<sub>i</sub> ln q<sub>i</sub>, equals the teacher’s own <b>entropy</b> plus the KL. In the worked example: 0.8869 = 0.8018 + 0.0851. The entropy depends only on the teacher, so it cannot be changed by training the student. Minimising soft cross-entropy and minimising KL move the student in exactly the same way.</p>
        <DeepDive title="Why multiply by T²?">
          <p>The gradient of KL(p(T) ‖ q(T)) with respect to the student’s logit z<sub>i</sub> is (q<sub>i</sub>(T) − p<sub>i</sub>(T)) ÷ T. The ÷ T comes from the chain rule, because the logits were divided by T.</p>
          <p>At high temperature both distributions flatten towards uniform, so the difference q − p itself also shrinks roughly like 1/T. Together the gradient falls roughly like 1/T². Multiplying the loss by T² undoes that, so the soft loss keeps about the same weight against the hard-label loss whatever T you choose. Hinton and colleagues give this argument in the 2015 paper. In the lab, the gradient is T × (q(T) − p(T)), which is exactly this.</p>
          <p>Why not always use a huge T? In the limit, matching soft targets becomes matching the teacher’s logits directly, including very negative logits that the teacher’s own training barely constrained, which can be noise. The 2015 paper reports that when the student is much smaller than the teacher, intermediate temperatures worked best. In LLM practice T is usually around 1 to 2.</p>
        </DeepDive>
        <DeepDive title="Which way round? Forward and reverse KL">
          <p>KL is not symmetric. In the worked example, KL(p ‖ q) = 0.0851 but KL(q ‖ p) = 0.0920.</p>
          <p><b>Forward KL(teacher ‖ student)</b>, as above, is large wherever the teacher has probability and the student does not. It punishes the student for missing anything the teacher might say, so the student spreads itself to cover all of it (“mode-covering”). A small student that cannot fit everything ends up vague.</p>
          <p><b>Reverse KL(student ‖ teacher)</b> is large wherever the student puts probability the teacher would not. It punishes the student for saying things the teacher would not say, so a small student concentrates on a few answers the teacher likes (“mode-seeking”).</p>
          <p>On-policy distillation (MiniLLM by Gu and colleagues; GKD by Agarwal and colleagues, both published in 2024) typically uses the reverse direction, estimated on text the student wrote itself. Which divergence works best depends on the task and is still studied.</p>
        </DeepDive>
      </TheMath>

      <CodeIt>
        <p>No file in the repository implements distillation, so everything here is a sketch. It is the same algorithm as the lab, which is tested against these numbers. You can paste it into Python with NumPy and run it.</p>
        <Code
          title="sketch: softmax with temperature, and KL"
          setup={`import numpy as np`}
          show={`print("KL(p || q):", round(kl(np.array([.7, .2, .1]), np.array([.5, .3, .2])), 4))
print("mat, sofa, moon at T = 1:", softmax([3, 1, -1]).round(3))
print("mat, sofa, moon at T = 2:", softmax([3, 1, -1], T=2).round(3))`}
        >{`
def softmax(z, T=1.0):
    z = np.asarray(z, float) / T
    e = np.exp(z - z.max())
    return e / e.sum()

def kl(p, q):
    return float(np.sum(p * np.log(p / q)))

kl(np.array([.7, .2, .1]), np.array([.5, .3, .2]))   # 0.0851
`}</Code>
        <p>One training step with soft targets. The student is six logits for one context, exactly as in the lab:</p>
        <Code
          title="sketch: one distillation step"
          setup={`import numpy as np

def softmax(z, T=1.0):
    z = np.asarray(z, float) / T
    e = np.exp(z - z.max())
    return e / e.sum()

def kl(p, q):
    return float(np.sum(p * np.log(p / q)))

teacher_logits = np.array([4.0, 2.2, 2.0, 1.6, 0.5, -2.0])   # the lab's teacher: mat, sofa, ...`}
          show={`for T in (1, 4):
    z = np.zeros(6)
    for _ in range(20):
        z = soft_step(z, teacher_logits, T)
    print(f"T = {T}: KL after 20 soft steps = {kl(softmax(teacher_logits), softmax(z)):.4f}")
print("teacher:", softmax(teacher_logits).round(3))
print("student:", softmax(z).round(3))`}
        >{`
def soft_step(z, teacher_logits, T, lr=0.5):
    p = softmax(teacher_logits, T)   # soft targets
    q = softmax(z, T)                # student at the same T
    grad = T * (q - p)               # gradient of T^2 * KL(p || q)
    return z - lr * grad
`}</Code>
        <Code
          title="sketch: one hard-label step, for comparison"
          setup={`import numpy as np

def softmax(z, T=1.0):
    z = np.asarray(z, float) / T
    e = np.exp(z - z.max())
    return e / e.sum()

def kl(p, q):
    return float(np.sum(p * np.log(p / q)))

teacher_logits = np.array([4.0, 2.2, 2.0, 1.6, 0.5, -2.0])   # the lab's teacher
MAT = 0                                                       # the hard label`}
          show={`z = np.zeros(6)
for step in range(1, 201):
    z = hard_step(z, MAT)
    if step in (7, 200):
        print(f"step {step}: KL = {kl(softmax(teacher_logits), softmax(z)):.3f}, student:", softmax(z).round(3))
print("teacher:        ", softmax(teacher_logits).round(3))`}
        >{`
def hard_step(z, label, lr=0.5):
    q = softmax(z)
    grad = q.copy()
    grad[label] -= 1                 # q - onehot
    return z - lr * grad
`}</Code>
        <p>Run 20 soft steps from <code>z = np.zeros(6)</code> with the lab’s teacher logits <code>[4.0, 2.2, 2.0, 1.6, 0.5, -2.0]</code>: the KL is 0.0487 at T = 1 and 0.0060 at T = 4. Run 200 hard steps: 0.911. Same numbers as the lab.</p>
        <p>In a real model the same loss is one line in PyTorch, applied at every position of a batch:</p>
        <Code title="sketch: the distillation loss in PyTorch">{`
T = 2.0
loss_kd = F.kl_div(F.log_softmax(student_logits / T, dim=-1),
                   F.log_softmax(teacher_logits / T, dim=-1),
                   reduction="batchmean", log_target=True) * T * T
loss = alpha * loss_kd + (1 - alpha) * F.cross_entropy(student_logits_flat, targets)
`}</Code>
        <p><code>F.kl_div</code> expects the <em>student’s</em> log-probabilities first and the teacher’s second, which is the reverse of how KL(p ‖ q) is written. That ordering trips up many people.</p>
        <p>Sequence-level distillation needs no new loss at all. It is data generation followed by the fine-tuning you already know:</p>
        <Code
          title="sketch: a synthetic data pipeline"
          setup={`import random
random.seed(0)

# A made-up teacher that answers sums, and is wrong about one time in four.
class Teacher:
    def generate(self, prompt, n, temperature):
        a, b = map(int, prompt.split("+"))
        return [str(a + b + random.choice([0, 0, 0, 1])) for _ in range(n)]

teacher = Teacher()
prompts = ["2+3", "7+5", "12+30", "9+9"]

def passes_checks(prompt, answer):          # the checker: is the sum right?
    a, b = map(int, prompt.split("+"))
    return int(answer) == a + b

def sft(student, dataset):                  # stands in for the fine-tuning loop
    print(f"training {student} on {len(dataset)} examples")

student = "the small model"`}
          show={`print("kept:", dataset)
print(f"{len(dataset)} of {4 * len(prompts)} teacher answers survived the filter")`}
        >{`
dataset = []
for prompt in prompts:
    for answer in teacher.generate(prompt, n=4, temperature=0.8):
        if passes_checks(prompt, answer):     # tests, answer checker, judge
            dataset.append((prompt, answer))
sft(student, dataset)                         # the loop from Fine-tuning
`}</Code>
        <p>And on-policy distillation, in outline. The student writes, the teacher scores the student’s own tokens:</p>
        <Code
          title="sketch: on-policy distillation"
          setup={`import numpy as np
rng = np.random.default_rng(0)
TOKENS = ["mat", "sofa", "moon"]

# Made-up models: one fixed next-token distribution each, the worked example's p and q.
class Toy:
    def __init__(self, probs):
        self.probs = np.array(probs)
    def generate(self, prompt, n=5000):
        return list(rng.choice(len(TOKENS), size=n, p=self.probs))
    def log_probs(self, prompt, answer):
        return np.log(self.probs[answer])

teacher = Toy([0.7, 0.2, 0.1])
student = Toy([0.5, 0.3, 0.2])
prompt = "The cat sat on the"`}
          show={`print("student wrote:", [TOKENS[t] for t in answer[:8]], "...")
print(f"estimate from its own {len(answer)} tokens: {loss:.4f}")
p, q = teacher.probs, student.probs
print(f"exact reverse KL(q || p):          {np.sum(q * np.log(q / p)):.4f}")`}
        >{`
answer = student.generate(prompt)
s_logp = student.log_probs(prompt, answer)    # one number per token
t_logp = teacher.log_probs(prompt, answer)    # same tokens, teacher's view
loss = (s_logp - t_logp).mean()               # estimate of reverse KL
# (real implementations treat this like an RL reward per token)
`}</Code>
      </CodeIt>

      <BreakIt>
        <p>Predict first, then check in the lab.</p>
        <ul>
          <li><b>Hard label, step 7.</b> The KL is at its lowest, 0.098, and the student gives mat 69.5%, close to the teacher’s 70.2%. Now look at the other five rows. Sofa and moon get exactly the same 6.1%. Even at its best moment, a hard-label student cannot know that sofa is nearly right.</li>
          <li><b>Hard label, step 200.</b> Mat is above 99% and the KL has climbed to 0.911, worse than where it started (0.802). Training on hard labels made the student <em>more</em> confident than the teacher. Real training data has many different next tokens after the same context, which is what stops this in practice.</li>
          <li><b>Sampled labels.</b> Count the rows as you drag. In these 200 draws “moon” never came up once. The student’s moon probability still sits well above the teacher’s 0.17% at step 200, because it never got a single example to learn from. Training only on text a model wrote is a lossy view of its distribution: this is the first step towards model collapse.</li>
          <li><b>Soft targets, T = 1 versus T = 4.</b> At T = 1 the student needs 94 steps to get below a KL of 0.01, and at step 200 it still gives moon 0.8% instead of 0.17%. At T = 4 it takes 18 steps and moon is right. The small probabilities only get learned when they are loud enough.</li>
          <li><b>T = 0.5.</b> Now temperature <em>sharpens</em> the teacher. The soft curve never gets below 0.01 in 200 steps. A temperature below 1 hides dark knowledge instead of revealing it.</li>
        </ul>
      </BreakIt>

      <Exercises>
        <Exercise
          id="distillation-calc-kl"
          type="calculate"
          title="KL by hand"
          answer={{ value: 0.193, tolerance: 0.005 }}
          answerLabel="KL(p ‖ q), in nats"
          hints={[
            'KL(p ‖ q) = Σ p × ln(p ÷ q). There are only two tokens.',
            'First term: 0.8 × ln(0.8 ÷ 0.5) = 0.8 × ln 1.6. Second: 0.2 × ln(0.2 ÷ 0.5) = 0.2 × ln 0.4.',
            'ln 1.6 ≈ 0.4700 and ln 0.4 ≈ −0.9163.',
          ]}
          solution={<><p>0.8 × 0.4700 = 0.3760 and 0.2 × (−0.9163) = −0.1833. Total ≈ <b>0.193</b>.</p><p>The second term is negative, because the student gives token 2 more than the teacher does, but the total is still positive. KL is never negative, and reaches 0 only when the student matches the teacher exactly.</p></>}
        >
          <p>A teacher gives two tokens <code>p = [0.8, 0.2]</code>. An untrained student gives <code>q = [0.5, 0.5]</code>. What is KL(p ‖ q), using natural logs? (Three decimals.)</p>
        </Exercise>

        <Exercise
          id="distillation-calc-temperature"
          type="calculate"
          title="How loud is the runner-up?"
          answer={{ value: 0.269, tolerance: 0.005 }}
          answerLabel="probability of the second token at T = 4"
          hints={[
            'Divide the logits by T first: [4, 0] ÷ 4 = [1, 0].',
            'softmax([1, 0]): the second token gets e⁰ ÷ (e¹ + e⁰) = 1 ÷ (e + 1).',
            'e ≈ 2.718, so 1 ÷ 3.718.',
          ]}
          solution={<><p>1 ÷ (1 + e) ≈ <b>0.269</b>. At T = 1 the same token gets 1 ÷ (1 + e⁴) ≈ 0.018, so the temperature made it about 15 times larger.</p><p>That is the point: at T = 1 this token contributes almost nothing to the student’s loss, so the student barely learns how it relates to the winner. At T = 4 it contributes about a quarter of the target.</p></>}
        >
          <p>A teacher has logits <code>[4, 0]</code> for two tokens. At temperature T = 4, what probability does it give the second token? (Three decimals.)</p>
        </Exercise>

        <Exercise
          id="distillation-debug-temperature"
          type="debug"
          title="The student that is never sure"
          hints={[
            'Compare how the teacher’s logits and the student’s logits are treated in the loss.',
            'The teacher is softened by T. Is the student?',
            'If the student is trained at T = 1 to match a distribution flattened by T = 4, what will its normal, T = 1 output look like?',
          ]}
          solution={<><p>The student’s logits are not divided by T. So at T = 1 the student is trained to reproduce the teacher’s <em>flattened</em> distribution. It succeeds, and then at deployment (also T = 1) it produces that flattened distribution: mat 29% instead of 70% in the lab’s example. It sounds unsure of everything and samples far-fetched tokens.</p><p>The fix is <code>F.log_softmax(student_logits / T, dim=-1)</code>. Student and teacher must be compared at the same temperature. Then the student’s logits learn to match the teacher’s logits, and at T = 1 its probabilities match the teacher’s real ones.</p></>}
        >
          <p>A colleague distils with T = 4. The student trains fine and the loss falls, but in use it hedges on everything and its samples are oddly random. Here is the loss:</p>
          <Code>{`
loss = F.kl_div(F.log_softmax(student_logits, dim=-1),
                F.log_softmax(teacher_logits / T, dim=-1),
                reduction="batchmean", log_target=True) * T * T
`}</Code>
        </Exercise>

        <ExplainBack
          id="distillation-explain"
          prompt="Dev asks: “If the student ends up imitating the teacher, why not train it on the same internet text the teacher read? Why go through the teacher at all?” Answer him in three or four sentences, using the words hard label, soft target and temperature."
          modelAnswer={<p>The internet text gives hard labels: one next token per position, with every other token treated as equally wrong. The teacher has already learned, at great cost, how plausible every alternative is, and its soft targets hand that over with every single example: sofa is nearly right, moon is absurd. A small student learns faster and ends up better from those richer targets than from rediscovering all of it through hard labels. Temperature flattens the teacher’s distribution so its small probabilities are large enough to matter in the loss. The price is that the student inherits the teacher’s mistakes and cannot become better than it.</p>}
        />

        <details className="deep">
          <summary>More practice (optional)</summary>
          <div className="details-body">
            <Exercise
              id="distillation-calc-collapse"
              type="calculate"
              title="Where do the rare words go?"
              answer={{ value: 0.133, tolerance: 0.005 }}
              answerLabel="chance the phrase never appears"
              hints={[
                'Each of the 100 samples independently misses the phrase with probability 1 − 0.02 = 0.98.',
                'All 100 must miss it: multiply 0.98 by itself 100 times.',
                '0.98¹⁰⁰ = e^(100 × ln 0.98) and ln 0.98 ≈ −0.0202.',
              ]}
              solution={<><p>0.98¹⁰⁰ ≈ <b>0.133</b>. About one time in seven, a phrase the model uses 2% of the time is completely absent from its own 100 samples.</p><p>A new model fitted only to those samples has no reason to produce it. Repeat for several generations and more rare things drop out each time, while nothing brings them back. That is the mechanism behind model collapse, and why keeping real human data in the mix matters.</p></>}
            >
              <p>A model uses a certain rare phrase in 2% of its answers. You sample 100 answers from it to build a training set for the next model. What is the probability that the phrase appears in <em>none</em> of them? (Three decimals.)</p>
            </Exercise>

            <Exercise
              id="distillation-experiment-temperature"
              type="experiment"
              title="The coolest temperature that works"
              answer={{ value: 2, tolerance: 0 }}
              answerLabel="lowest T"
              hints={[
                'Select “soft targets” and read the last readout line, “first step with KL below 0.01 and staying there”.',
                'Move T down from 4 in steps of 0.5 and watch that step number grow.',
                'Check T = 2 and T = 1.5 carefully. One is under 30, one is not.',
              ]}
              solution={<><p><b>T = 2</b> reaches a KL below 0.01 at step 24. T = 1.5 needs 37 steps, and T = 1 needs 94.</p><p>Going above about T = 3 barely helps in this toy (18 or 19 steps), because once the small probabilities are loud enough there is nothing more to reveal. A real student with limited capacity can even be hurt by very high T, which is why practitioners tune it.</p></>}
            >
              <p>In the lab, what is the lowest temperature on the slider for which the soft-target student gets below a KL of 0.01 within 30 steps?</p>
            </Exercise>
          </div>
        </details>
      </Exercises>

      <CheckYourself
        questions={[
          {
            q: 'What does a soft target give the student that a hard label does not?',
            options: ['The teacher’s weights', 'A ranking of every other token, so it learns which wrong answers are nearly right', 'A faster forward pass', 'A guarantee that the answer is correct'],
            answer: 1,
            explain: 'The “dark knowledge” is in the small probabilities. The weights are never copied.',
          },
          {
            q: 'Riya fine-tunes a small model on 50,000 answers written by a big model through its API. Which kind of distillation is this?',
            options: ['Logit distillation', 'Sequence-level distillation: ordinary SFT on text the teacher wrote', 'On-policy distillation', 'Quantization'],
            answer: 1,
            explain: 'She only sees the text, not the probabilities, so she cannot match distributions. Training on the teacher’s outputs is sequence-level distillation, the usual form of synthetic data.',
          },
          {
            q: 'Which description of model collapse is most accurate?',
            options: ['Any use of synthetic data ruins a model', 'Repeatedly training models on their own output, with the real data replaced, loses rare things first and drifts towards bland output', 'A model forgets everything after one epoch', 'The model’s weights overflow'],
            answer: 1,
            explain: 'The effect appears when generated data replaces real data over several generations. Adding synthetic data to real data largely avoids it in published experiments.',
          },
        ]}
      />

      <Remember
        items={[
          <><b>Distillation</b> trains a small student to imitate a big teacher’s behaviour, never its weights. The student can differ in size and shape.</>,
          <>A <b>hard label</b> names one right token. A <b>soft target</b> is the teacher’s whole distribution, and its small probabilities (the “dark knowledge”) say which wrong answers are nearly right.</>,
          <>Loss: <b>T² × KL(p(T) ‖ q(T))</b>, both at temperature T, often mixed with ordinary cross-entropy. <b>Temperature</b> above 1 makes the small probabilities loud enough to learn.</>,
          <>Three flavours: match probabilities (logit), fine-tune on the teacher’s text (sequence-level, i.e. <b>synthetic data</b>), or let the student write and the teacher grade (<b>on-policy</b>). Synthetic data risks <b>model collapse</b> when own outputs replace real data, <b>contamination</b> and inherited errors. The student is capped by its teacher.</>,
        ]}
      />

      <RealLLM>
        <ToyVsReal
          toy={<ul><li>One context, six tokens</li><li>A student of six numbers that can match the teacher exactly</li><li>Exact gradients, 200 steps</li><li>Sampled labels from a seeded random generator</li></ul>}
          real={<ul><li>Billions of contexts and a vocabulary of 100,000 or more tokens</li><li>A student network with far fewer weights than the teacher, so it can only approximate it</li><li>Teacher probabilities computed (or stored, often only the top few per position) for trillions of tokens</li><li>Teacher-written datasets of hundreds of thousands to millions of examples, heavily filtered</li></ul>}
        />
        <Callout kind="established">
          The idea predates LLMs: Buciluă, Caruana and Niculescu-Mizil compressed large ensembles into small models in 2006, and Hinton, Vinyals and Dean introduced the temperature form in 2015. DistilBERT (2019) applied it to Transformers. Google’s Gemma 2 report (2024) says the 2B and 9B models were trained with distillation from a larger model instead of plain next-token prediction. Meta’s Llama 3.2 1B and 3B models (2024) used the logits of Llama 3.1 8B and 70B as token-level targets during pretraining.
          <br /><br />
          It also works for <b>reasoning models</b>, which write out long step-by-step working before they answer (<a href="#/lesson/reasoning-models">Reasoning models</a>). DeepSeek-R1 (2025) has 671 billion parameters in total (a mixture-of-experts model that uses about 37 billion per token). Its authors fine-tuned six much smaller open models (Qwen2.5 models from 1.5B to 32B, and Llama models of 8B and 70B) on about 800,000 examples written by R1, with SFT only. On the AIME 2024 maths benchmark the distilled 32B model scored 72.6% (pass@1) against 47.0% for a 32B base model trained by large-scale RL directly. Their conclusion: for small models, distilling a stronger model beat the expensive RL recipe, while pushing beyond the teacher still needs a stronger base model and RL.
        </Callout>
        <Callout kind="research">
          Several things are still being worked out. <b>On-policy distillation</b> (student writes, teacher scores every token) is increasingly used for small models; the Qwen3 technical report (2025), for example, describes distilling from its larger models first off-policy and then on-policy, and reports this worked better than RL for its small models at a fraction of the compute.
          <br /><br />
          The risks of synthetic data are also active research. One 2025 study (Cloud and colleagues, “subliminal learning”) found that a student could pick up a teacher’s trait, such as a preference for owls, from teacher-written data that were only lists of numbers with no mention of the trait, when teacher and student shared the same base model. How much hidden behaviour travels through synthetic data in general is not known.
        </Callout>
        <p>One practical warning: before you distil from a commercial model, read its terms of service. Several major providers forbid using their outputs to train models that compete with theirs. Open-weight teachers come with their own licence, which may also say something about derived models.</p>
        <p>By the end of the month the support team has a 3-billion-parameter model on their laptops. It answers the common tickets almost as well as the big one, and it gets the rare ones wrong in exactly the places the big one was unsure. Riya writes that down. The student is only ever as good as its teacher.</p>
      </RealLLM>
    </Lesson>
  )
}
