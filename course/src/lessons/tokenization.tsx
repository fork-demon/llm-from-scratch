import { RepoRunner } from '../components/RepoRunner'
import { Lesson, Why, Problem, MentalModel, TryIt, Numbers, CodeIt, BreakIt, Exercises, CheckYourself, Remember, RealLLM } from '../components/lesson'
import { Callout, DeepDive, Flow, G, Term, ToyVsReal, WhyExists } from '../components/ui'
import { Code } from '../components/Code'
import { CodeExercise } from '../components/python'
import { Exercise, ExplainBack } from '../components/exercise'
import { TokenizerPlayground } from '../interactive/TokenizerPlayground'
import { TextToVector } from '../illustrations/TextToVector'

// Try it: the training half of bpe_tokenizer.py, so the encode/decode excerpts below can run.
const BPE_SETUP = `from collections import Counter

class BPETokenizer:                          # training half of bpe_tokenizer.py
    def __init__(self):
        self.merges, self.vocab = [], {}
    def train(self, text, vocab_size):
        chars = sorted(set(text))
        self.vocab = {i: c for i, c in enumerate(chars)}
        stoi = {c: i for i, c in self.vocab.items()}
        ids = [stoi[c] for c in text]
        while len(self.vocab) < vocab_size:
            pairs = Counter(zip(ids, ids[1:]))
            if not pairs:
                break
            (a, b), count = pairs.most_common(1)[0]
            if count < 2:
                break
            new_id = len(self.vocab)
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
            self.merges.append(((a, b), new_id))
            ids = self._apply_merge(ids, (a, b), new_id)
    @staticmethod
    def _apply_merge(ids, pair, new_id):
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                out.append(new_id); i += 2
            else:
                out.append(ids[i]); i += 1
        return out`

const Tok = ({ children }: { children: string }) => <span className="token">{children.replace(/ /g, '␣')}</span>

export default function TokenizationLesson() {
  return (
    <Lesson id="tokenization">
      <Why>
        <p className="lede">First of the month. Finance forwards Riya the chatbot vendor’s invoice with one line: “Why is this so high?”</p>
        <p>The bill is not counted in messages, or words. It is counted in <em>tokens</em>. Riya sorts the usage by customer language and frowns. Replies in Kannada and Hindi use many more tokens than English replies that say the same thing. The engineers’ test runs, full of JSON and code, are expensive too.</p>
        <p>She asks Kabir what a token actually is. He thinks for a second. “It’s the answer to a simpler question. How do you multiply the word ‘cat’?”</p>
        <p>You can’t. Computers cannot multiply words.</p>
        <p>Everything you have built so far eats numbers: <a href="#/lesson/vectors">dot products</a>, <a href="#/lesson/matrices">matrix multiplies</a>, <a href="#/lesson/neurons">neurons</a>. Text is not numbers. So before a language model can do anything at all, something has to turn your prompt into numbers.</p>
        <p>It happens in three small hops:</p>
        <TextToVector focus="token" id={4217} idNote="(number made up)" vector="[0.2, −1.3, …]" />
        <p>This lesson is about the first two hops. They sound like plumbing. They are not.</p>
        <p>The way text is cut into pieces decides what the model can <em>see</em>. It also decides what Paisa Pal’s bill is, and why a model that writes sonnets can miscount the letters in “strawberry”.</p>
        <p>A model never sees letters or words. It sees a list of integers. The component that produces those integers is the <b>tokenizer</b>, and it is built <em>before</em> the neural network is trained, by a surprisingly simple algorithm.</p>
      </Why>

      <Problem title="The problem: what should one integer stand for?">
        <p>Dev, reading the invoice over her shoulder, is sure. “Tokens are just words, yaar. They made up a fancy name so they can charge more.”</p>
        <p>He is half right: someone does have to choose what the pieces are. There are two obvious choices, and both fail.</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th /><th>one integer per word</th><th>one integer per character</th></tr></thead>
            <tbody>
              <tr><td>list size</td><td>Hundreds of thousands for English alone, plus every plural, verb form and name. The list explodes.</td><td>Tiny: about a hundred entries covers English.</td></tr>
              <tr><td>unseen text</td><td>A new word such as “unbelievableness” has <b>no ID</b>. Neither does a typo (“teh”), nor <code>getUserById</code>, nor most of German. All the model gets is a shrug: <code>&lt;unknown&gt;</code>.</td><td>Nothing is ever unknown.</td></tr>
              <tr><td>sequence length</td><td>Short: “the” is one piece.</td><td>Very long: “the cat sat on the mat” is 6 words but 22 characters. Longer input means more computation, and less of your text fits in the <G t="context-window">context window</G>, the fixed amount the model can read at once.</td></tr>
              <tr><td>meaning per piece</td><td>A lot.</td><td>Almost none. “t” tells you nothing, so the model must re-learn, everywhere, that t-h-e belongs together.</td></tr>
            </tbody>
          </table>
        </div>
        <WhyExists
          problem="Text must become a sequence of integers from a fixed list."
          naive="Use whole words (short sequences) or single characters (tiny list)."
          fails="Words: the list is unbounded and unseen words have no ID. Characters: sequences are long and each piece means little."
          idea="Subwords. Frequent strings get their own ID. Rare strings are spelled out from smaller pieces that do have IDs."
          tradeoff="Token boundaries follow frequency, not meaning or spelling. The model loses direct sight of letters and digits."
        />
        <p>With subwords, a common word is one piece and a rare word is a few familiar pieces:</p>
        <p><Tok>the</Tok> &nbsp; <Tok>token</Tok><Tok>ization</Tok> &nbsp; <Tok>un</Tok><Tok>believ</Tok><Tok>able</Tok><Tok>ness</Tok></p>
        <p className="muted">(An illustration. The exact cuts depend on the tokenizer, as you are about to see.)</p>
        <p>These pieces have a name:</p>
        <Term
          name="Token, vocabulary, token ID"
          plain={<>A <b>token</b> is one piece of text from a fixed list. The list is the <b>vocabulary</b>. A token’s position in the list is its <b>token ID</b>, and that integer is all the model receives.</>}
          example={<>Vocabulary: 0 = “a”, 1 = “t”, 2 = “at”, 3 = “c”. The text “cat” becomes the IDs [3, 2].</>}
          formal={<>A tokenizer is a pair of functions: encode: text → list of IDs, and decode: list of IDs → text. For the toy in this lesson, decode(encode(x)) = x always. Real tokenizers aim for that too, with a couple of exceptions you will meet below.</>}
        />
        <p>But who decides which strings deserve their own ID? Nobody. An algorithm finds them by counting.</p>
      </Problem>

      <MentalModel title="An algorithm you could have invented">
        <p>Kabir puts it to Riya as a puzzle. Suppose you start with single characters and want longer pieces. Which two characters would you glue together first?</p>
        <p>The pair that occurs most often, because that saves the most. Then do it again. That is the whole algorithm. It is called <G t="bpe">byte pair encoding</G>, or BPE.</p>
        <ol>
          <li>Start the vocabulary with every single character in your training text.</li>
          <li>Count every pair of neighbouring tokens. Find the most frequent pair.</li>
          <li>Glue that pair into a new token. Add it to the vocabulary. <b>Write the rule down.</b></li>
          <li>Repeat until the vocabulary is as big as you want.</li>
        </ol>
        <p>What is “learned” is nothing more than the ordered list of glue rules, the <b>merge list</b>. To tokenize new text, split it into characters and replay the rules in the same order. To turn IDs back into text, concatenate the strings they stand for.</p>
        <Callout kind="analogy">
          BPE is <b>dictionary compression</b>, like the idea behind zip files. Patterns that occur often get a short code. Rare things are spelled out the long way. Think of how a WhatsApp group shortens what it says most: “gm”, “ok”, “tmrw”. Nobody shortens a word they rarely type.
          <br /><br />
          Where the analogy stops: a compressor wants the smallest file and can build a new dictionary for every file. A tokenizer’s dictionary is built once and then frozen, because the model will learn one vector per entry. And the goal is not the smallest output. It is pieces that are useful units for a model to learn from.
        </Callout>
        <p>Notice that there is no neural network here, no gradient, no loss. Tokenizer training is a counting loop you could write in an afternoon. It runs once, before model training starts, and its result is a plain data file.</p>
      </MentalModel>

      <TryIt title="Train one, then use it">
        <p>The training text below is the one from the repository’s Python file. Step through the first few merges slowly and predict each one before you press the button. What do you expect first: <Tok>th</Tok>? Look again: spaces are characters too.</p>
        <TokenizerPlayground />
        <p>Things to notice:</p>
        <ul>
          <li>The very first merge is <Tok> t</Tok>: a space followed by “t”. By merge 4 there is a token <Tok> the </Tok>. The algorithm knows nothing about English. It found “the” by counting.</li>
          <li>Run to the target and type <b>unbelievable</b>. It shatters into 12 single characters. Your tokenizer never saw anything like it. Rare strings cost many tokens.</li>
          <li>Try <b>foxes</b>: <Tok>fox</Tok><Tok>e</Tok><Tok>s</Tok>. An unseen word, built from a known piece.</li>
          <li>Keep merging past merge 44 and a monster appears: <Tok>quick brown fox j</Tok>. With so little text, the algorithm starts memorising whole phrases. Untick “repeat” and training stops after 30 merges, because nothing occurs twice any more.</li>
        </ul>
      </TryIt>

      <Numbers>
        <p>Let’s train BPE by hand on the course’s favourite sentence, so nothing is hidden:</p>
        <div className="card center mono">the cat sat on the mat</div>
        <p>That is 22 characters (spaces count), using 10 different ones. Count neighbouring pairs. “a” followed by “t” occurs 3 times (c<b>at</b>, s<b>at</b>, m<b>at</b>). Nothing else occurs more than twice. So merge 1 is <Tok>a</Tok> + <Tok>t</Tok>.</p>
        <div className="table-scroll">
          <table className="plain">
            <thead><tr><th>merge</th><th>rule (count)</th><th>the text afterwards</th><th>tokens</th></tr></thead>
            <tbody>
              <tr><td>start</td><td></td><td className="mono">t h e ␣ c a t ␣ s a t ␣ o n ␣ t h e ␣ m a t</td><td className="mono">22</td></tr>
              <tr><td>1</td><td className="mono">a + t (3×)</td><td className="mono">t h e ␣ c <b>at</b> ␣ s <b>at</b> ␣ o n ␣ t h e ␣ m <b>at</b></td><td className="mono">19</td></tr>
              <tr><td>2</td><td className="mono">t + h (2×)</td><td className="mono"><b>th</b> e ␣ c at ␣ s at ␣ o n ␣ <b>th</b> e ␣ m at</td><td className="mono">17</td></tr>
              <tr><td>3</td><td className="mono">th + e (2×)</td><td className="mono"><b>the</b> ␣ c at ␣ s at ␣ o n ␣ <b>the</b> ␣ m at</td><td className="mono">15</td></tr>
              <tr><td>4</td><td className="mono">the + ␣ (2×)</td><td className="mono"><b>the␣</b> c at ␣ s at ␣ o n ␣ <b>the␣</b> m at</td><td className="mono">13</td></tr>
              <tr><td>5</td><td className="mono">at + ␣ (2×)</td><td className="mono">the␣ c <b>at␣</b> s <b>at␣</b> o n ␣ the␣ m at</td><td className="mono">11</td></tr>
            </tbody>
          </table>
        </div>
        <p>After merge 5 no pair occurs twice, so training stops: 10 characters + 5 merges = a vocabulary of 15. The same text went from 22 tokens to 11.</p>
        <p>At merge 2 several pairs were tied at 2. The code takes the one it met first in the text, which was “t”, “h”. Any fixed rule works, as long as it is always the same rule.</p>
        <p>Now <b>use</b> it on words it has never seen. Split into characters, then replay merges 1 to 5 in order:</p>
        <ul>
          <li>“chat” → c h a t → merge 1 applies → <Tok>c</Tok><Tok>h</Tok><Tok>at</Tok>. Three tokens.</li>
          <li>“the hat” → merges 1, 2, 3, 4 apply → <Tok>the </Tok><Tok>h</Tok><Tok>at</Tok>. Three tokens for seven characters.</li>
        </ul>
        <p>Type the sentence into the playground’s training box (untick “repeat”) and you will get exactly these five merges.</p>
      </Numbers>

      <CodeIt>
        <p>There is no equation in this lesson, only a loop. We build it in the order you just did it by hand. First the starting vocabulary: one ID per character.</p>
        <Code title="Step 1: every character gets an ID">{`
chars = sorted(set(text))
self.vocab = {i: c for i, c in enumerate(chars)}   # id -> string
stoi = {c: i for i, c in self.vocab.items()}       # string -> id

ids = [stoi[c] for c in text]    # the corpus as a list of token ids
`}</Code>
        <p>Then count neighbouring pairs and pick the winner. <code>zip(ids, ids[1:])</code> walks over every adjacent pair, and <code>Counter</code> is Python’s tally dictionary (like a <code>Map&lt;Pair, Integer&gt;</code> you increment).</p>
        <Code title="Step 2: the most frequent adjacent pair">{`
pairs = Counter(zip(ids, ids[1:]))
(a, b), count = pairs.most_common(1)[0]
if count < 2:
    break                        # nothing left worth merging
`}</Code>
        <Code title="Step 3: mint a new token and write the rule down">{`
new_id = len(self.vocab)
self.vocab[new_id] = self.vocab[a] + self.vocab[b]   # 'a' + 't' -> 'at'
self.merges.append(((a, b), new_id))                 # the ordered merge list
ids = self._apply_merge(ids, (a, b), new_id)         # rewrite the corpus
`}</Code>
        <p><code>_apply_merge</code> is a single left-to-right pass that replaces the pair wherever it occurs:</p>
        <Code
          source="phase2-language/bpe_tokenizer.py"
          title="replace every occurrence of one pair"
          show={`text = "the cat sat on the mat"
stoi = {c: i for i, c in enumerate(sorted(set(text)))}   # step 1: one id per character
ids = [stoi[c] for c in text]
out = _apply_merge(ids, (stoi["a"], stoi["t"]), 10)       # merge 1 of the hand-worked table: a + t
print(len(ids), "tokens ->", len(out), "tokens")
print(out)`}
        >{`
def _apply_merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out
`}</Code>
        <p>Using the tokenizer is the same pass, once per learned rule, in training order. These are the real functions from the repository:</p>
        <Code
          source="phase2-language/bpe_tokenizer.py"
          title="encode and decode"
          setup={BPE_SETUP}
          show={`BPETokenizer.encode, BPETokenizer.decode = encode, decode   # attach them as methods
tok = BPETokenizer()
tok.train("the cat sat on the mat", vocab_size=15)          # the hand-worked example: 5 merges
for text in ["chat", "the hat"]:
    ids = tok.encode(text)
    print(repr(text), "->", [tok.vocab[i] for i in ids], ids, " decode:", repr(tok.decode(ids)))`}
        >{`
def encode(self, text):
    stoi = {s: i for i, s in self.vocab.items() if len(s) == 1}
    ids = [stoi[c] for c in text]        # KeyError on unseen char
    for pair, new_id in self.merges:     # the order matters
        ids = self._apply_merge(ids, pair, new_id)
    return ids

def decode(self, ids):
    return "".join(self.vocab[i] for i in ids)
`}</Code>
        <p>In this toy, decoding cannot fail: every ID stands for a whole string of characters. So <code>decode(encode(x)) == x</code> is an <code>assert</code> in the file, not a hope.</p>
        <Callout kind="warn" label="Careful: real tokenizers bend this rule">
          Two things break the perfect round trip in production tokenizers.
          <br /><br />
          <b>Clean-up before splitting.</b> Some tokenizers (for example those built with the SentencePiece library) first tidy the text: they may turn look-alike Unicode characters into one standard form, or treat spaces in a special way. Decode the IDs and you get the tidied text, which can differ slightly from what was typed.
          <br /><br />
          <b>Half a character.</b> Byte-level tokenizers start from the 256 possible byte values instead of characters (more on this at the end of the lesson). One Kannada letter is 3 bytes in UTF-8. A single token can hold only part of a character. Decode that token on its own, as a streaming chat window does token by token, and you get a broken symbol (�) until the rest of the bytes arrive.
        </Callout>
        <p>Running the file prints the merges it learns. The first ones:</p>
        <Code lang="output" title="python phase2-language/bpe_tokenizer.py">{`
  merge   1: ' ' + 't' -> ' t'  (seen 75x)
  merge   2: 'h' + 'e' -> 'he'  (seen 64x)
  merge   3: ' t' + 'he' -> ' the'  (seen 59x)
  merge   4: ' the' + ' ' -> ' the '  (seen 47x)
  merge   5: 'i' + 'n' -> 'in'  (seen 24x)
  ...
  37 chars -> 22 tokens (1.68 chars/token)
  unseen word 'foxes' -> ['fox', 'e', 's']
`}</Code>
        <p>The playground above runs a line-for-line port of this file, so it shows the same merges and the same counts.</p>

        <h3>Special tokens: the words nobody types</h3>
        <p>In <a href="#/lesson/prompt-to-answer">lesson 0.1</a> the model stopped answering when it picked an invisible <span className="mono">&lt;end&gt;</span> token. Where does that come from? Not from BPE.</p>
        <p>Every real vocabulary reserves a few extra IDs by hand. These <b>special tokens</b> never come out of ordinary text. They are markers.</p>
        <ul>
          <li><b>End of text.</b> GPT-2 has exactly one: <span className="mono">&lt;|endoftext|&gt;</span>, ID 50,256, placed between documents during training. The model learns that after it, a new unrelated text begins. When it predicts that token while answering, the answer is over.</li>
          <li><b>Chat roles.</b> Chat models add markers for who is speaking. Llama 3, for example, wraps each message like <span className="mono">&lt;|start_header_id|&gt;user&lt;|end_header_id|&gt; … &lt;|eot_id|&gt;</span>. The bubbles you see in a chat app reach the model as one long token sequence, with these markers in between.</li>
        </ul>
        <p>Because they matter so much, tokenizer libraries do not turn a <em>typed</em> “&lt;|endoftext|&gt;” into the special ID unless you ask them to. (OpenAI’s tiktoken raises an error by default.) Otherwise a customer could type a role marker and pretend to be the system. We come back to that kind of trick in <a href="#/lesson/alignment-safety">Alignment and safety</a>.</p>
        <RepoRunner path="phase2-language/bpe_tokenizer.py" title="Run bpe_tokenizer.py in your browser">
          <p>This is the whole file from the repository, running in your browser. Press Run to see what it prints, then edit a copy and change things.</p>
        </RepoRunner>
      </CodeIt>

      <BreakIt>
        <p>Back to the playground. Predict first, then check.</p>
        <ul>
          <li><b>A typo.</b> Type “teh quikc borwn fox”. How many tokens, compared with the correct spelling? Typos push text off the frequent paths, so they cost more tokens and look unfamiliar to the model.</li>
          <li><b>Another language, or code.</b> Try “der schnelle Fuchs”, “ನಮಸ್ಕಾರ” or <code>getTokenById(x)</code>. Capital letters and brackets were never in the training text, so they have <em>no ID at all</em>. Read the explanation that appears: this is why real tokenizers start from bytes.</li>
          <li><b>Change the training text.</b> Paste a few lines of Python or Java into the training box and run to the target. Now which strings become single tokens? A tokenizer is a fingerprint of the text it was trained on.</li>
          <li><b>Turn the vocabulary dial.</b> Compare characters per token for the sample sentence at a vocabulary of 30, 50 and 80. Bigger vocabulary, shorter sequences, but more rows for the model to learn in the next lesson.</li>
          <li><b>A leading space.</b> Compare “the fox” with “ the fox”. The first “the” is <Tok>t</Tok><Tok>he</Tok>, the second is inside the single token <Tok> the </Tok>. To a tokenizer, a word at the start of a text and the same word after a space are different strings.</li>
        </ul>
      </BreakIt>

      <Exercises>
        <CodeExercise id="tokenization-code-merge" />
        <Exercise
          id="tokenization-predict-first-merge"
          type="predict"
          title="Which pair wins?"
          answer={{ text: ['an', 'a+n', 'a n', "'a'+'n'", 'a,n'] }}
          answerLabel="first merged pair, e.g. xy"
          hints={[
            'Do not count letters. Count neighbouring pairs, including overlapping ones: in “banana”, “an” occurs twice and “na” occurs twice.',
            'Now add “bandana”: it contains “an” twice more, but “na” only once.',
          ]}
          solution={<><p><b>a + n</b>, seen 4 times (“na” is seen 3 times, “ba” 2 times). After that merge the text is <span className="mono">b an an a ␣ b an d an a</span>, and the next merges are “b” + “an” and then “an” + “a”.</p><p>Why it matters: BPE is greedy and only ever looks at <em>adjacent pairs right now</em>. It has no notion of syllables or meaning. Sensible-looking pieces appear only because sensible pieces are frequent.</p></>}
        >
          <p>You train BPE on the text <code>banana bandana</code>. Which pair of characters is merged first? Decide before you try it in the playground.</p>
        </Exercise>

        <Exercise
          id="tokenization-trace-encode"
          type="trace"
          title="Encode by hand"
          answer={{ value: 5, tolerance: 0 }}
          answerLabel="number of tokens"
          hints={[
            'Start from 11 single characters: i n ␣ t h e ␣ f e r n. Then go through the merge list from top to bottom, applying each rule wherever it fits.',
            'Rules 1 to 4 build “␣the␣” step by step: ␣t, then he, then ␣the, then ␣the␣. Rule 5 builds “in”.',
            'Rule 6 (e + n) does not apply: in “fern” the e is followed by r. Rule 9 (e + r) does.',
          ]}
          solution={<><p><Tok>in</Tok><Tok> the </Tok><Tok>f</Tok><Tok>er</Tok><Tok>n</Tok> = <b>5 tokens</b> for 11 characters.</p><p>Notice that “ the ” swallowed both of its spaces, and that “fern” stayed in three pieces because it was never frequent enough to earn merges of its own. This is exactly what the playground shows if you set the vocabulary to 38 and type the phrase.</p></>}
        >
          <p>These are the first ten merges the Python file learns, in order (␣ is a space):</p>
          <Code lang="text" title="merge list">{`
 1. ␣ + t    -> ␣t          6. e + n   -> en
 2. h + e    -> he          7. f + o   -> fo
 3. ␣t + he  -> ␣the        8. s + ␣   -> s␣
 4. ␣the + ␣ -> ␣the␣       9. e + r   -> er
 5. i + n    -> in         10. fo + x  -> fox
`}</Code>
          <p>Encode the text <code>in the fern</code> by hand. How many tokens do you end up with?</p>
        </Exercise>

        <Exercise
          id="tokenization-debug-order"
          type="debug"
          title="The “optimised” encoder"
          hints={[
            'Which tokens must already exist before the rule ␣t + he → ␣the can match anything?',
            'With longest-first order, the rule for ␣the runs while the text is still single characters. It finds nothing. Later ␣t and he are created, but the rule that would join them has already had its turn.',
          ]}
          solution={<><p>Merges form a <b>dependency chain</b>: rule 3 (␣t + he) can only fire after rules 1 and 2 have created its two halves. Applied longest-first, the long rules run on raw characters and match nothing, so “ the fox” comes out as <Tok> t</Tok><Tok>he</Tok><Tok> </Tok><Tok>fo</Tok><Tok>x</Tok> instead of <Tok> the </Tok><Tok>fox</Tok>.</p><p>The nasty part: <code>decode</code> still returns the right text, so a round-trip test passes. But the model was trained on the other segmentation, and now receives ID sequences it has never seen. The fix is to delete the <code>sorted</code>: replay merges in exactly the order they were learned.</p></>}
        >
          <p>A colleague thinks applying the biggest merges first will be faster. Decoding still round-trips perfectly, yet the model’s output quality collapses. Why?</p>
          <Code
            setup={`${BPE_SETUP}
text = ("the quick brown fox jumps over the lazy dog. "
        "the dog barked at the fox. the fox ran into the forest. "
        "learning about the internals of the tokenizer teaches the "
        "engineer the fundamentals of the language model. "
        "the tokens in the text represent the meaning of the words. ") * 4   # the file's training text
tok = BPETokenizer()
tok.train(text, vocab_size=80)
stoi = {s: i for i, s in tok.vocab.items() if len(s) == 1}`}
            show={`BPETokenizer.encode = encode
ids = tok.encode(" the fox")
print("tokens: ", [tok.vocab[i] for i in ids])
print("decoded:", repr("".join(tok.vocab[i] for i in ids)))`}
          >{`
def encode(self, text):
    ids = [stoi[c] for c in text]
    longest_first = sorted(self.merges, key=lambda m: -len(self.vocab[m[1]]))
    for pair, new_id in longest_first:
        ids = self._apply_merge(ids, pair, new_id)
    return ids
`}</Code>
        </Exercise>

        <details className="deep">
          <summary>More practice (optional)</summary>
          <div className="details-body">
        <Exercise
          id="tokenization-implement-byte-fallback"
          type="implement"
          title="Run it, crash it, fix it"
          hints={[
            'Run python phase2-language/bpe_tokenizer.py first. Then add print(tok.encode("the Fox")) at the bottom: capital F is not in the training text, so you get a KeyError.',
            'A cheap fix: in train(), reserve ids 0 to 255 for the 256 possible bytes, and work on text.encode("utf-8") (a sequence of integers 0..255) instead of characters. Every possible input is then made of known ids.',
            'decode then has to join bytes, not strings: b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace"), with self.vocab holding bytes objects.',
          ]}
          solution={<><p>Replace “characters” with “bytes” everywhere: <code>ids = list(text.encode("utf-8"))</code>, a base vocabulary of <code>{'{'}i: bytes([i]) for i in range(256){'}'}</code>, merges that concatenate <code>bytes</code>, and a decode that joins bytes and then decodes UTF-8.</p><p>Nothing can be unknown any more, because every string in every language is a sequence of bytes. That is the “byte-level BPE” of GPT-2 and its successors.</p><p>The price: a character that never earned a merge costs one token per byte. “é” is 2 bytes, a Kannada letter is 3, an emoji is 4.</p><p>Some other tokenizers use bytes only as a backup, for characters they have no piece for. The effect is the same. Nothing is unknown, and text in a language that was rare in the tokenizer’s training text comes out in many small pieces. That is Riya’s invoice, explained.</p></>}
        >
          <p>Riya wants her toy tokenizer to survive a Kannada customer. Open <code>phase2-language/bpe_tokenizer.py</code> and run it. Then make <code>encode</code> crash with a character that is not in the training text. Finally, fix it the way GPT-2 did: <b>start from bytes instead of characters</b> (byte-level BPE). Before you start: how many base tokens will you need so that no input can ever be unknown?</p>
        </Exercise>

        <ExplainBack
          id="tokenization-explain"
          prompt="A teammate asks: “Why can an LLM explain quantum physics but not reliably count the r’s in strawberry?” Explain it using what you know about tokens."
          modelAnswer={<p>The model never receives letters. The tokenizer turns “strawberry” into a few integer IDs (the pieces might be “str”, “aw”, “berry”), and an integer does not contain an r. To count letters, the model must have memorised the spelling of each token from the rare places in its training text where words are spelled out, and then do arithmetic across tokens. Explaining physics, by contrast, works at the level of words and ideas, which is exactly the level tokens are good at. It is a bit like asking a program to count the vowels in a string when all it was given is the string’s hash.</p>}
        />
          </div>
        </details>
      </Exercises>

      <CheckYourself
        questions={[
          {
            q: 'After BPE training, what exactly has been “learned”?',
            options: ['A neural network that predicts word boundaries', 'The grammar of the language', 'An ordered list of merge rules (plus the vocabulary it produces)', 'One vector per token'],
            answer: 2,
            explain: 'No gradients, no network. Only counting, and a list of which pair was glued at each step. The vectors come in the next lesson.',
          },
          {
            q: 'Your tokenizer was trained mostly on English. You send it a paragraph in a language it rarely saw. What happens?',
            options: ['It fails with an error', 'It translates the text to English first', 'It produces the same number of tokens, since length is what matters', 'The paragraph becomes many more tokens than an English paragraph of the same length, so it costs more and fills the context sooner'],
            answer: 3,
            explain: 'Few merges were earned by that language’s character sequences, so the text is spelled out in small pieces (down to bytes).',
          },
          {
            q: 'An API bills you for 1,000 tokens. Roughly how much English text is that?',
            options: ['1,000 characters', '1,000 words', 'About 750 words', 'About 10,000 words'],
            answer: 2,
            explain: 'A common rule of thumb for English with GPT-style tokenizers: one token is about four characters, or three quarters of a word. Code and other languages differ.',
          },
        ]}
      />

      <Remember
        items={[
          <>A model sees <b>integers, not text</b>. Text → tokens → token IDs happens before the neural network, and IDs → text after it.</>,
          <><b>Words</b> fail (unbounded list, unseen words have no ID). <b>Characters</b> fail (long sequences, little meaning per piece). <b>Subwords</b> are the compromise.</>,
          <><b>BPE</b>: repeatedly merge the most frequent adjacent pair. The trained tokenizer is the <b>ordered merge list</b>. Encode = replay in order. Decode = concatenate.</>,
          <>Frequent strings become one token. Rare strings (typos, unusual names, under-represented languages) <b>shatter into many tokens</b>. Tokens are the unit of <b>cost and context limits</b>, and the reason models struggle with spelling, letter counting and long numbers.</>,
        ]}
      />

      <RealLLM>
        <Flow horizontal steps={[{ label: 'Tokenizer', sub: 'this lesson' }, { label: 'Embeddings', sub: 'next lesson' }, { label: 'Transformer blocks' }, { label: 'Next-token probabilities' }, { label: 'Sampling → token ID → text' }]} active={0} />
        <ToyVsReal
          toy={<ul><li>Trained on about 1,000 characters</li><li>Starts from the characters it happened to see, so unseen characters have no ID</li><li>A few dozen merges</li><li>Merges can run across spaces, so whole phrases become tokens</li></ul>}
          real={<ul><li>Trained on many gigabytes of text</li><li>Starts from the 256 byte values, so nothing is ever unknown</li><li>About 50,000 merges (GPT-2) to 200,000 (recent models)</li><li>Text is first split by a pattern (roughly: words, numbers, punctuation), and merges never cross those splits</li></ul>}
        />
        <Callout kind="established">The algorithm is the one you just ran. GPT-2, GPT-4, Llama 3 and many others use byte-level BPE; only the training text, the pre-splitting pattern and the number of merges differ. That also means a real tokenizer would split your text <em>differently</em> from the toy in this lesson. To see real token boundaries, paste text into a tokenizer viewer such as tiktokenizer.vercel.app.</Callout>
        <h3>What this means for you as a developer</h3>
        <ul>
          <li><b>Tokens are the unit of money and memory.</b> APIs bill per token, and the <G t="context-window">context window</G> (“128k context”) is counted in tokens, not characters or words. English prose is about 4 characters per token. Code, JSON with long keys, and non-English text usually cost more.</li>
          <li><b>Spelling and letter counting are hard</b> because the letters are hidden inside IDs. The same goes for reversing a string or rhyming by spelling.</li>
          <li><b>Long numbers are split into chunks</b> whose boundaries have nothing to do with place value (for example “1234567” might arrive as <Tok>123</Tok><Tok>456</Tok><Tok>7</Tok>). Column-wise arithmetic is awkward when you cannot see the columns. This is one reason models are handed calculators and code interpreters.</li>
          <li><b>Leading spaces matter.</b> In GPT-style vocabularies “ the” (with a space) and “the” are different tokens with different IDs. A prompt that ends in a trailing space, or a stop-sequence that differs by one space, can behave surprisingly.</li>
        </ul>
        <Callout kind="model">“The tokenizer is why models cannot count letters” is a useful first suspect, not the whole story. Models can learn spellings from their training text, and newer ones often get such questions right, especially when they spell the word out first. The accurate statement: tokenization makes character-level tasks indirect and therefore error-prone.</Callout>
        <DeepDive title="Are there alternatives to BPE?">
          <p>Yes. WordPiece (used by BERT) and the Unigram method (available in the SentencePiece library) choose pieces by slightly different criteria, but produce the same kind of result: a fixed subword vocabulary. There is also active research on models that read raw bytes and skip the tokenizer entirely. As of today, nearly every production LLM still uses a subword tokenizer, mostly byte-level BPE.</p>
        </DeepDive>
        <p>Riya replies to finance with one paragraph: the bill counts tokens, tokens follow the tokenizer’s training text, and that text was mostly English. The Kannada replies are not longer messages. They are cut into smaller pieces. Now each token needs a meaning, and that is the next lesson.</p>
      </RealLLM>
    </Lesson>
  )
}
