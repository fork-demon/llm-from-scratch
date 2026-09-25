import { Lesson, Why, Problem, MentalModel, TryIt, Exercises, Remember, BeforeMovingOn } from '../components/lesson'
import { Callout, WhyExists } from '../components/ui'
import { Exercise } from '../components/exercise'
import { sourceUrl } from '../data/curriculum'
import { MemoryChallenge } from '../interactive/MemoryChallenge'
import { FourMovements } from '../illustrations/FourMovements'

export default function FromMemoryLesson() {
  return (
    <Lesson id="from-memory">
      <Why title="Can you rebuild the whole picture without looking?">
        <p className="lede">Sunday night in Mysuru. The rasam is finished, the plates are pushed aside, and Amma has put Riya’s phone face down on the table.</p>
        <p>“Every weekend on video call you are doing school maths with me,” she says. “Now tell me what it was for. From the start. No phone, no drawing.”</p>
        <p>Months ago, in the first lesson, you typed “What is a cat?” and could not say what happened next. Here is the same question again, with one difference: no diagram this time.</p>
        <p>You have built a tokenizer, embeddings, attention, a Transformer, a training loop, a sampler, a KV cache, a retriever, a fine-tuning run and an agent. Each made sense on the day you built it.</p>
        <p>The real test is different. Can you sit across from someone, with nothing in front of you, and explain how text goes in and an answer comes out? This lesson is that dining table.</p>
        <Callout kind="idea">
          If you can reconstruct the pipeline from memory and say in plain words what happens at every arrow, you understand LLMs better than most people who use them every day. If you cannot yet, this lesson will show you exactly which arrows to revisit.
        </Callout>
      </Why>

      <Problem title="Recognising is not the same as knowing">
        <WhyExists
          problem="You want to know whether you actually understand LLMs, or only feel as though you do."
          naive="Skim back over the lessons. Everything looks familiar, so you conclude you know it."
          fails="Familiarity is recognition, and recognition is cheap. Re-reading feels productive precisely because it is easy. The feeling disappears the moment someone asks you to explain it without the page."
          idea="Test yourself by producing, not recognising: rebuild the diagram, write the explanation, only then compare with a model answer."
          tradeoff="It is uncomfortable, and you will find gaps. That is the point: a gap you have found is a gap you can close."
        />
        <Callout kind="established">
          This is one of the most robust results in research on learning (the “testing effect”): trying to retrieve something from memory strengthens it far more than reviewing it again, even when the attempt fails. It is why every lesson here ended with recall questions, and why this final one is nothing but recall.
        </Callout>
      </Problem>

      <MentalModel title="A skeleton to hang it on">
        <p>Riya does not start with twelve stages. Twelve are hard to hold as a flat list. She starts with four moves, and fills each one in as Amma nods. If you blank during the challenge, come back to these four.</p>
        <FourMovements />
        <p>The slots are empty on purpose. Naming them is your job in the challenge below.</p>
        <Callout kind="dev">
          It is the shape of many systems you already know: <b>decode the input, process it, encode an output, loop</b>. What is unusual here is only step 2: the processing is a learned function with billions of parameters instead of code someone wrote.
        </Callout>
        <p>And for everything built around the model, one question sorts it out: <b>does it change the prompt, the weights, or the code around the model?</b></p>
      </MentalModel>

      <TryIt title="The challenge">
        <p>Set aside ten quiet minutes. The timer is optional. Notes are not allowed.</p>
        <MemoryChallenge />
        <p>However it went: the arrows you rated “partly” or “missed” are now the most valuable pages of the course for you. Revisit them, then repeat the challenge in a few days. The second attempt is where it becomes permanent.</p>
      </TryIt>

      <Exercises title="Two exercises away from the screen">
        <Exercise
          id="from-memory-paper"
          type="trace"
          title="One page, by hand"
          hints={[
            'Start with the four moves from the skeleton and leave space under each.',
            'Under each stage, write the shape of the data: a string, a list of T integers, a T × D table of numbers, a list of vocabulary-size scores, one integer.',
          ]}
          solution={<p>A good page shows the twelve stages with the <em>shape of the data</em> between them: string → T tokens → T integers → T × D matrix (plus position) → T × D matrix after each block → one vector of vocabulary-size logits from the last position → probabilities of the same size → one integer → appended, so T grows by one. If you could write those shapes without looking, you could also write the code. Keep the page: it is a better cheat-sheet than any you could download, because you made it.</p>}
        >
          <p>On a single sheet of paper, draw the whole pipeline for the prompt “What is a cat?” and annotate every arrow with the <em>shape</em> of the data that flows along it. Use T for the number of tokens and D for the size of a token’s vector.</p>
        </Exercise>
        <Exercise
          id="from-memory-teach"
          type="explain"
          title="Teach one person"
          hints={[
            'Pick someone who has used a chat assistant but never studied one. Five minutes, no slides.',
            'Start where this course started: “It does not look anything up. It predicts one piece of text at a time.” Then walk along the pipeline.',
          ]}
          solution={<p>Notice which questions they asked. “But how does it know facts?” (knowledge implicit in parameters), “Why does it make things up?” (it always produces a likely continuation; there is no “not found”), “Is it searching the internet?” (only if a tool or retrieval step puts text into the prompt). If you could answer those three without hand-waving, you have what this course set out to give you. Any question that made you hesitate is your next lesson to revisit.</p>}
        >
          <p>Explain to a friend or colleague, in five minutes, what happens between typing a prompt and seeing an answer. No jargon without a plain-words explanation first. Afterwards, write down the questions they asked.</p>
        </Exercise>
      </Exercises>

      <Remember
        items={[
          <>The pipeline: <b>text → tokens → ids → embeddings + position → attention inside stacked Transformer blocks → logits → probabilities → sampling → next token → repeat</b>.</>,
          <>The model is a <b>function from the tokens so far to next-token probabilities</b>. Its knowledge is implicit in parameters set by gradient descent on next-token prediction.</>,
          <><b>RAG changes the prompt. Fine-tuning changes the weights. Tools add a loop</b> in which your code acts and the model only writes text.</>,
          <>It is fluent and it makes things up <b>for the same reason</b>: it generates what is likely, and nothing in the mechanism checks truth.</>,
          <>Understanding fades unless retrieved. <b>Come back and do this challenge again in a week.</b></>,
        ]}
      />

      <section className="section" id="next">
        <div className="section-head"><span className="section-kicker">Beyond this course</span></div>
        <h2>Where to go next</h2>
        <p>You built everything small and by hand, so that the real tools are no longer magic. Now it makes sense to pick them up.</p>
        <div className="grid-2">
          <div className="card">
            <h4 style={{ fontSize: 17, marginBottom: 6 }}>PyTorch, properly</h4>
            <p>You used it in <code>tiny_gpt.py</code>. The <a href="https://pytorch.org/tutorials/" target="_blank" rel="noreferrer">official tutorials</a> cover datasets, GPUs and saving models. Then read <a href="https://github.com/karpathy/nanoGPT" target="_blank" rel="noreferrer">nanoGPT</a>’s <code>model.py</code>: it is your GPT, production-shaped, in about 300 lines.</p>
          </div>
          <div className="card">
            <h4 style={{ fontSize: 17, marginBottom: 6 }}>Hugging Face</h4>
            <p>The <code>transformers</code> library loads real open models in a few lines. With what you know, <code>tokenizer.encode</code>, <code>model.generate(temperature=, top_p=)</code> and <code>past_key_values</code> are no longer incantations. The free <a href="https://huggingface.co/learn" target="_blank" rel="noreferrer">Hugging Face courses</a> are a good guide.</p>
          </div>
          <div className="card">
            <h4 style={{ fontSize: 17, marginBottom: 6 }}>Read the paper</h4>
            <p><a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noreferrer">“Attention Is All You Need”</a> (Vaswani et al., 2017). Read section 3.2 closely: it is the attention you coded, equation for equation. Then follow the reading path at the end of <a href="#/lesson/reading-papers">How to read an LLM paper</a>, through DeepSeek-V3 and a look inside a trained model.</p>
          </div>
          <div className="card">
            <h4 style={{ fontSize: 17, marginBottom: 6 }}>The repository’s reading list</h4>
            <p><a href={sourceUrl('resources.md')} target="_blank" rel="noreferrer"><code>resources.md</code></a> lists the best videos, posts and papers for every module, with a note on <em>when</em> each one is worth your time. The rule it follows: build first, then watch.</p>
          </div>
        </div>
        <p>Then build something real. Put RAG over documents you care about with a hosted model. Fine-tune a small open model with LoRA. Write an agent with two tools and a step limit. You already know what each of those is doing underneath, and, just as usefully, what it cannot do.</p>
        <Callout kind="research">
          Much is still open: why these models generalise as well as they do, what their internal computations mean (<a href="#/lesson/interpretability">Looking inside the model</a> showed how little we can read yet), how far next-token prediction plus reasoning at inference time can go, and how to make their behaviour reliably safe (<a href="#/lesson/alignment-safety">Alignment and safety</a>). You now have the mechanism needed to read that research critically instead of taking anyone’s word for it, including ours.
        </Callout>
        <h3>The best question of the evening</h3>
        <p>Riya gets all the way round the loop at the dining table. Tokens, vectors, attention, blocks, scores, one token chosen, round again. Then the bank that became a river, the refund policy the bot invented, the Diwali queue, the finance meeting, launch day.</p>
        <p>Amma listens without interrupting, the way she used to listen to a student reciting a proof. Then she asks one question.</p>
        <p>“When it says ‘I don’t know’, is that also only a likely next word?”</p>
        <p>Riya opens her mouth, closes it, and laughs. “Yes, Amma. Training made those words more likely in the places they belong. Nothing inside checks that they are true. That is why we test it, every time.”</p>
        <p>Amma nods, satisfied, and gets up to make coffee. It is exactly the question a good teacher asks: not “what does it do?” but “how would you know?”. You can answer it now too.</p>
        <p>This course was built to let you say one sentence and mean it:</p>
        <div className="card center" style={{ fontFamily: 'var(--serif)', fontSize: 24, padding: '28px 16px' }}>
          “I don’t use LLMs as magic anymore.”
        </div>
        <p>Text becomes tokens. Tokens become vectors. Attention lets them read each other. Blocks stack. Scores become probabilities. One token is chosen. The loop goes round. You have built every part of that with your own hands. It is not magic. It is yours.</p>
      </section>

      <BeforeMovingOn
        id="final"
        intro="The last checkpoint. Eight questions, one from almost every part of the course, all from memory."
        questions={[
          {
            q: 'Part 1. Two token vectors have a large positive dot product. What does that tell you?',
            options: ['They are the same length', 'They point in a similar direction: the vectors “agree”', 'One is the transpose of the other', 'Their softmax is 1'],
            answer: 1,
            explain: 'The dot product is the agreement score used everywhere: in embedding similarity, in attention (query · key), and in the final logits.',
          },
          {
            q: 'Parts 2 and 3. What does backpropagation actually compute?',
            options: ['The best learning rate', 'For every parameter, how much the loss would change if that parameter were nudged', 'The correct answer for each training example', 'A new architecture with fewer layers'],
            answer: 1,
            explain: 'It is the chain rule applied backwards through the network, giving the gradient. Gradient descent then moves each parameter a small step against it.',
          },
          {
            q: 'Part 4. Why do LLMs use sub-word tokens (such as BPE) rather than whole words?',
            options: ['Sub-words are easier for humans to read', 'A fixed-size vocabulary can still represent any text: unseen words are split into known pieces', 'Whole words cannot be converted to integers', 'It makes attention unnecessary'],
            answer: 1,
            explain: 'Words are unbounded and characters make sequences very long. Sub-words are the compromise: common words are one token, rare ones are several pieces.',
          },
          {
            q: 'Part 6. You remove all positional information from a Transformer and shuffle the input tokens. What happens to the attention weight between two given tokens (ignoring the causal mask)?',
            options: ['It is unchanged: attention scores depend only on the two vectors', 'It falls to zero', 'It depends on their new distance', 'It becomes uniform'],
            answer: 0,
            explain: 'Attention is order-blind by itself. That is why position must be added to the embeddings or rotated into queries and keys.',
          },
          {
            q: 'Part 7. You lower the temperature from 1.0 to 0.3. What changes?',
            options: ['The model’s weights', 'The logits coming out of the network', 'The probabilities get sharper, so likely tokens are picked even more often', 'The tokenizer’s vocabulary'],
            answer: 2,
            explain: 'Temperature divides the logits just before softmax, outside the network. Low temperature concentrates probability on the top tokens.',
          },
          {
            q: 'Part 7. What does a KV cache change about the generated text?',
            options: ['Nothing: it stores past keys and values so they are not recomputed, trading memory for speed', 'It makes the output more deterministic', 'It lets the model remember earlier conversations', 'It lowers quality slightly in exchange for speed'],
            answer: 0,
            explain: 'Earlier tokens cannot see later ones, so their keys and values never change. Caching them is pure memoisation: cached and naive generation are identical.',
          },
          {
            q: 'Parts 8 and 9. Your assistant must answer from a policy handbook that changes every week. Which approach fits best, and why?',
            options: ['Fine-tune weekly, because facts live in the weights', 'RAG: retrieve the relevant passages and put them in the prompt; no weights change, and updating means editing documents', 'Raise the temperature so that it explores more answers', 'Use a larger context window and no documents'],
            answer: 1,
            explain: 'RAG changes the prompt; fine-tuning changes the weights. Changing facts belong in the prompt. Fine-tuning is for behaviour.',
          },
          {
            q: 'Part 9. An agent’s model outputs the text “TOOL: delete_file ARGS: {"path": "report.txt"}”. What deletes the file?',
            options: ['The model, through its output layer', 'The tokenizer', 'Your code: the loop parses that text and decides whether to call the function', 'Nothing can; models cannot affect files'],
            answer: 2,
            explain: 'The model only writes text. The loop around it acts, which is also where limits, validation and approval for irreversible actions belong.',
          },
        ]}
      />
    </Lesson>
  )
}
