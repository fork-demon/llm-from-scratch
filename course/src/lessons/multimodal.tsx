import { Lesson, Why, Problem, MentalModel, TryIt, Numbers, TheMath, CodeIt, BreakIt, Exercises, CheckYourself, Remember, RealLLM } from '../components/lesson'
import { Callout, DeepDive, Equation, Flow, G, Term, ToyVsReal, WhyExists } from '../components/ui'
import { Code } from '../components/Code'
import { CodeExercise } from '../components/python'
import { Exercise, ExplainBack } from '../components/exercise'
import { PatchLab } from '../interactive/PatchLab'
import { ImageToTokens } from '../illustrations/ImageToTokens'

export default function MultimodalLesson() {
  return (
    <Lesson id="multimodal">
      <Why>
        <p className="lede">Monday, 10 a.m. The support lead forwards Riya a ticket with no text at all.</p>
        <p>It is a photo of a phone screen. A red circle with a white cross. “Payment failed.” Below it, in the customer’s own words, one line: “then why is money gone from my account??”</p>
        <p>Half the tickets this week look like this. Screenshots, photos of bank SMS, a blurry picture of a receipt. Customers do not type what they can show.</p>
        <p>Dev reads over her shoulder. “Easy. The new models take images. It probably runs OCR, pulls out the text, and reads that.”</p>
        <p>Riya is not so sure. The model she built in <a href="#/lesson/build-gpt">Part 7</a> eats one thing only: a list of token IDs, each turned into a row of an <G t="embedding">embedding</G> table. There is no row for “red circle”.</p>
        <p>A language model only ever processes a sequence of vectors. To “see”, a picture must become a sequence of vectors of the same width as the word vectors. So how does a photo become tokens?</p>
      </Why>

      <Problem title="The problem: a photo is not a sentence">
        <p>Count what is in that screenshot. A phone screen of 1080 × 2400 pixels, three colour numbers per pixel.</p>
        <p className="mono">1080 × 2400 × 3 = 7,776,000 numbers</p>
        <p>Text had a natural unit, the <G t="token">token</G>, and a fixed <G t="vocabulary">vocabulary</G>. A picture has neither. So what should one image token be?</p>
        <div className="grid-2">
          <div className="card">
            <h4 style={{ fontSize: 17, marginBottom: 6 }}>Read the text out first (OCR)?</h4>
            <p>Dev’s idea. It works for the words. It throws away everything else: the red cross, which button is greyed out, a chart, a face, handwriting. Many answers depend on exactly those things.</p>
          </div>
          <div className="card">
            <h4 style={{ fontSize: 17, marginBottom: 6 }}>One token per pixel?</h4>
            <p>Even a small 448 × 448 image is 200,704 pixels. <G t="attention">Attention</G> compares every token with every other, so that is about 40 billion pairs per layer, for one picture. And one pixel means almost nothing on its own.</p>
          </div>
        </div>
        <WhyExists
          problem="A language model needs a short sequence of vectors, and an image is millions of numbers with no vocabulary."
          naive="Convert the image to text first, or feed every pixel as a token."
          fails="Text loses everything that is not words. Pixels make sequences far too long, and each one carries almost no meaning."
          idea="Cut the image into squares of, say, 14 × 14 pixels. Flatten each square into a list of numbers and multiply it by a learned matrix. Each square becomes one vector: one token. No vocabulary needed."
          tradeoff="An image still costs hundreds or thousands of tokens, and very small details inside one square can be blurred together."
        />
      </Problem>

      <MentalModel>
        <p>Here is the whole idea as one picture. Follow the highlighted square.</p>
        <ImageToTokens />
        <Term
          name="Patch embedding"
          plain={<>Cut the image into equal squares (<b>patches</b>). Write each square’s pixels out as one long list of numbers. Multiply that list by a learned matrix to get a vector as wide as the model’s tokens. Add a vector that says where the square was.</>}
          example={<>A 14 × 14 patch has 14 × 14 × 3 = 588 numbers. A 588 × D matrix turns them into D numbers: one image token.</>}
          formal={<>z<sub>i</sub> = x<sub>i</sub> W<sub>E</sub> + e<sub>i</sub>, where x<sub>i</sub> is the flattened patch, W<sub>E</sub> is learned, and e<sub>i</sub> is the position embedding of patch i. This is how the Vision Transformer (ViT, 2020: “An Image is Worth 16x16 Words”) turns images into tokens.</>}
        />
        <Callout kind="analogy">
          Amma once described a rangoli to Riya over the phone, square by square: “Top left, white dots. Next one, a red curve starting.” Riya could redraw it. A patch token is that description of one square.
          <br /><br />
          Where the analogy stops: Amma used words. A patch token is a list of numbers, and nobody decides what they mean. Training does. The tokens also do not arrive one by one: the model looks at all squares together.
        </Callout>
        <p>Notice what is missing: there is no lookup. A text token ID picks a row from a table. A patch <em>is</em> its own row: its pixels go straight into a matrix multiply. That is why an image never has an “unknown token”.</p>

        <h3>Three parts: encoder, projector, language model</h3>
        <p>Most open vision-language models (for example the LLaVA family) are built from three pieces:</p>
        <Flow steps={[
          { label: 'Vision encoder', sub: 'a Transformer that reads the patch tokens and lets every patch attend to every other' },
          { label: 'Projector', sub: 'a small MLP that maps each output vector to the language model’s width' },
          { label: 'Language model', sub: 'reads image tokens and text tokens as one sequence, predicts the next text token' },
        ]} />
        <p>The vision encoder is itself a Transformer, the same kind you built, but trained on images. Its job is to turn raw patch numbers into vectors that describe what is in each part of the picture, in context.</p>

        <h3>How the encoder learns what things look like</h3>
        <p>Where does such an encoder come from? The best-known recipe is CLIP (OpenAI, 2021). It used 400 million pairs of an image and its caption from the web.</p>
        <p>Two encoders are trained side by side: one for images, one for text. For a batch of pairs, the goal fits in one sentence. The vector of each image should be close to the vector of <em>its own</em> caption, and far from the other captions in the batch.</p>
        <Term
          name="Contrastive training"
          plain={<>Learn by comparison: pull matching pairs together, push mismatched pairs apart. No one labels “this is a cat”. The caption is the label.</>}
          example={<>A photo of a cat and the caption “a cat asleep on a sofa” should get a high <G t="dot-product">dot product</G>. The same photo and “a bus in the rain” should get a low one.</>}
          formal={<>For N pairs, compute the N × N table of similarities between every image and every caption. Train so the diagonal (the true pairs) wins: CLIP uses a <G t="softmax">softmax</G> over each row and column, SigLIP (Google, 2023) a sigmoid on each pair separately.</>}
        />
        <p>After this training, an image encoder produces vectors that line up with language: a photo of a failed-payment screen lands near text about errors and payments. CLIP and SigLIP encoders are the “eyes” of many open vision-language models.</p>

        <h3>Plugging the eyes into the language model</h3>
        <p>The encoder’s vectors have the wrong width and live in a different “space” from the language model’s embeddings. The <b>projector</b> fixes that. In the original LLaVA (2023) it was a single matrix. LLaVA-1.5 used a small two-layer MLP.</p>
        <p>Training usually goes in two stages. First, freeze the encoder and the language model and train only the projector on image-caption pairs, so image tokens start to “look like” words to the language model. Then fine-tune on conversations about images (questions, answers, descriptions).</p>
        <Callout kind="model">
          It helps to picture the projected image tokens as <b>foreign words that the language model learns to read</b>. After projection they sit in the same sequence as text tokens, get the same attention, and the model predicts the next text token as always. This is a mental model: image tokens do not have to land near any particular word’s embedding, and in practice they often do not.
        </Callout>

        <h3>The other design: natively multimodal models</h3>
        <p>Bolting a pretrained encoder onto a pretrained language model is called <b>late fusion</b>. The alternative is to train one model on text and images from the start (<b>early fusion</b>).</p>
        <ul>
          <li><b>Fuyu</b> (Adept, 2023) has no separate image encoder: patches are linearly projected straight into the decoder.</li>
          <li><b>Chameleon</b> (Meta, 2024) turns each 512 × 512 image into 1,024 discrete tokens from a learned codebook of 8,192 image “words”, and trains one model on mixed sequences of text and image tokens.</li>
        </ul>
        <Callout kind="research">
          Several frontier models (for example GPT-4o and the Gemini family) are described by their makers as natively multimodal, trained across text, images and audio together. The exact architectures are not published. Treat confident diagrams of their insides as guesses.
        </Callout>

        <h3>Hearing: the same trick on sound</h3>
        <p>Audio is a wave: 16,000 numbers per second at the usual speech sampling rate. The first step turns it into a picture of sound.</p>
        <Term
          name="Log-mel spectrogram"
          plain={<>Chop the sound into short overlapping slices. For each slice, measure how much energy there is in each pitch band, spaced the way human hearing is, on a log scale. Each slice becomes one column of numbers.</>}
          example={<>Whisper uses 25 ms slices, starting every 10 ms: 100 columns per second, each with 80 numbers (128 in its latest large model).</>}
          formal={<>A short-time Fourier transform, mapped onto mel-spaced filter banks, then log-compressed.</>}
        />
        <p>Whisper (OpenAI, 2022, trained on 680,000 hours of audio) reads 30-second chunks. Its encoder starts with two small convolutions; the second halves the length. Then a Transformer, exactly like the vision encoder, turns the columns into one vector per 20 ms of audio. Audio-language models attach such an encoder to a language model with a projector, often after pooling to shorten the sequence further.</p>
        <p>To <em>speak</em> back, some models go one step further. A neural audio codec turns sound into discrete tokens (the codec in Moshi makes 12.5 frames per second, each a small stack of codebook tokens), and the model predicts audio tokens the way it predicts text tokens. Kyutai’s open Moshi (2024) works this way, listening and speaking at the same time.</p>
      </MentalModel>

      <TryIt title="Cut a picture into tokens">
        <p>Start with the failed-payment screenshot at 14 × 14 pixels. Click the red cross, then a blank corner, and compare their numbers. Then try 4 × 4 and 28 × 28 patches. At the bottom, work out what a real 448 × 448 image costs.</p>
        <PatchLab />
      </TryIt>

      <Numbers>
        <p>Let’s do one real image size by hand: 448 × 448 pixels, the tile size used by the InternVL family, with 14-pixel patches.</p>
        <div className="table-scroll">
          <table className="plain mono" style={{ fontSize: 13.5 }}>
            <thead><tr><th>step</th><th>calculation</th><th>result</th></tr></thead>
            <tbody>
              <tr><td>patches per side</td><td>448 / 14</td><td>32</td></tr>
              <tr><td>patches in the image</td><td>32 × 32</td><td><b>1,024</b></td></tr>
              <tr><td>numbers per patch</td><td>14 × 14 × 3</td><td>588</td></tr>
              <tr><td>projection matrix, if D = 1,024</td><td>588 × 1,024</td><td>602,112 weights</td></tr>
              <tr><td>merge each 2 × 2 block (pixel shuffle)</td><td>1,024 / 4</td><td><b>256 tokens</b></td></tr>
            </tbody>
          </table>
        </div>
        <p>1,024 tokens for one small image is a lot, and neighbouring patches often say almost the same thing (look at the sky in the sunset picture). So many models <b>merge</b> each 2 × 2 block of neighbouring patch vectors into one token. Pixel shuffle stacks the four vectors side by side (4 times longer, 4 times fewer), then the projector shrinks the width back. Four times fewer tokens, and four times less work in every layer that grows with the token count.</p>
        <h3>A bigger picture: tiles</h3>
        <p>An encoder is trained at one size. A tall phone screenshot squeezed to 448 × 448 loses its small text. The common fix is <b>dynamic resolution</b>: cut the image into several 448 × 448 tiles, encode each one, and add one shrunken thumbnail of the whole image so the model still sees the overall layout.</p>
        <div className="table-scroll">
          <table className="plain mono" style={{ fontSize: 13.5 }}>
            <thead><tr><th>image 896 × 1344</th><th>calculation</th><th>result</th></tr></thead>
            <tbody>
              <tr><td>tiles</td><td>(896 / 448) × (1344 / 448) = 2 × 3</td><td>6</td></tr>
              <tr><td>plus one thumbnail</td><td>6 + 1</td><td>7 tiles</td></tr>
              <tr><td>tokens</td><td>7 × 256</td><td><b>1,792</b></td></tr>
              <tr><td>share of a 32,768-token context</td><td>1,792 / 32,768</td><td>5.5%</td></tr>
            </tbody>
          </table>
        </div>
        <p>The customer’s one-line complaint is about a dozen text tokens. Their screenshot is <b>well over a hundred times more</b>. Images are the expensive part of these tickets, and that shows up in both speed and the bill.</p>
        <h3>And for sound</h3>
        <p>A 30-second voice note, Whisper-style: 30 s ÷ 10 ms = <b>3,000</b> spectrogram columns. The stride-2 convolution halves that to <b>1,500</b> encoder vectors, one per 20 ms. A model that pools by 2 again before the language model gets 750 audio tokens.</p>
      </Numbers>

      <TheMath>
        <p>Two small formulas cover everything above. First, the token count:</p>
        <Equation
          label="N equals H over P times W over P, divided by r squared"
          symbols={[
            ['N', 'number of image tokens the language model receives (for one tile)'],
            ['H, W', 'image height and width in pixels, after resizing so that P divides them'],
            ['P', 'patch side in pixels: 14 or 16 in most encoders'],
            ['r', 'merge factor: r = 2 merges each 2 × 2 block into one token; r = 1 means no merging'],
          ]}
        >
          N = (H / P) · (W / P) / r<sup>2</sup>
        </Equation>
        <p>Halve the patch size and N goes up 4 times. Double the image side and N goes up 4 times. Token cost grows with the <em>area</em>, which is why tiling and merging exist.</p>
        <p>Second, what one image token is:</p>
        <Equation
          label="z i equals x i times W E plus e i"
          symbols={[
            [<>x<sub>i</sub></>, <>patch i, flattened: P · P · 3 numbers in the 0 to 1 range (often normalised further)</>],
            [<>W<sub>E</sub></>, <>the patch-embedding matrix, (P · P · 3) × D, learned like any other weight</>],
            [<>e<sub>i</sub></>, <>the position embedding for slot i, so the model knows where the patch was (<a href="#/lesson/transformer-block">attention alone ignores order</a>)</>],
            [<>z<sub>i</sub></>, <>the resulting token: D numbers, the same width as the Transformer that reads it</>],
          ]}
        >
          z<sub>i</sub> = x<sub>i</sub> W<sub>E</sub> + e<sub>i</sub>
        </Equation>
        <p>It is exactly the same shape as a text token entering a GPT: a vector from the “input side” plus a position vector. The only difference is where the first vector came from: a table lookup for text, a matrix multiply of pixels for images.</p>
        <DeepDive title="The contrastive loss, written out">
          <p>Take a batch of N image-caption pairs. Encode them to unit vectors u<sub>1</sub>…u<sub>N</sub> (images) and t<sub>1</sub>…t<sub>N</sub> (captions). Build the N × N table s<sub>ij</sub> = u<sub>i</sub> · t<sub>j</sub> / τ, where τ is a learned temperature.</p>
          <p><b>CLIP:</b> treat row i as the <G t="logits">logits</G> of an N-way classification whose right answer is j = i, and apply <G t="cross-entropy">cross-entropy</G>. Do the same for each column (caption → image). Average the two.</p>
          <p><b>SigLIP:</b> treat every cell on its own as a yes/no question (“is this a true pair?”) with a sigmoid and a binary loss. No softmax across the batch, which makes very large batches cheaper to spread over many machines.</p>
        </DeepDive>
        <DeepDive title="Why a convolution is the same thing">
          <p>Real code often writes the patch embedding as a 2-D convolution with kernel size P and stride P. Each output position looks at exactly one non-overlapping P × P patch and takes a weighted sum of its P · P · 3 numbers for each of the D output channels. That is the same arithmetic as flatten-then-matrix-multiply, only arranged for fast GPU kernels.</p>
        </DeepDive>
      </TheMath>

      <CodeIt title="Let’s code it: patchify, project, merge">
        <p>There is no repository file for this lesson, so these are short NumPy sketches. The first function was checked against the lab’s TypeScript on a small image: same numbers, same order.</p>
        <Code
          title="Step 1: cut into patches and flatten"
          setup={`import numpy as np
rng = np.random.default_rng(0)`}
          show={`img = rng.random((448, 448, 3))                # a 448 x 448 colour image
print("448 x 448 image, p = 14:", patchify(img, 14).shape)
tiny = np.arange(4 * 4 * 3).reshape(4, 4, 3)   # a 4 x 4 test image you can check by hand
print("4 x 4 image, p = 2, first patch (the top-left 2 x 2 square):")
print(patchify(tiny, 2)[0])`}
        >{`
def patchify(img, p):                    # img: (H, W, 3) numbers in 0..1
    H, W, C = img.shape
    x = img.reshape(H // p, p, W // p, p, C)   # cut rows and columns into blocks
    x = x.transpose(0, 2, 1, 3, 4)             # (rows, cols, p, p, C): one block per patch
    return x.reshape(-1, p * p * C)            # one flat vector per patch
`}</Code>
        <p>For a 448 × 448 image and p = 14 this returns shape <code>(1024, 588)</code>: 1,024 patches, 588 numbers each.</p>
        <Code
          title="Step 2: one matrix multiply turns every patch into a token"
          setup={`import numpy as np
rng = np.random.default_rng(0)
def patchify(img, p):
    H, W, C = img.shape
    x = img.reshape(H // p, p, W // p, p, C).transpose(0, 2, 1, 3, 4)
    return x.reshape(-1, p * p * C)
img = rng.random((448, 448, 3))
D = 64                                            # tiny model width (real ones use 1,000+)
W_E = rng.normal(size=(588, D)) * 0.05            # learned in a real model
pos = rng.normal(size=(1024, D)) * 0.02           # learned in a real model`}
          show={`print("patches:", patches.shape, " W_E:", W_E.shape, " tokens:", tokens.shape)
print("image token 0, first 4 numbers:", tokens[0, :4].round(3))`}
        >{`
patches = patchify(img, 14)              # (1024, 588)
tokens = patches @ W_E + pos             # W_E: (588, D) learned; pos: (1024, D) learned
`}</Code>
        <Code
          title="Step 3: merge 2 × 2 neighbours (pixel shuffle)"
          setup={`import numpy as np
rng = np.random.default_rng(0)
enc_out = rng.normal(size=(32 * 32, 64))   # the encoder's 32 x 32 grid of output vectors, d = 64`}
          show={`merged = pixel_shuffle(enc_out, 32, 32)
print("before:", enc_out.shape, " after 2 x 2 merge:", merged.shape)
tiny = np.arange(16).reshape(16, 1)         # a 4 x 4 grid, one number per token
print(pixel_shuffle(tiny, 4, 4))            # each row is one 2 x 2 block of neighbours`}
        >{`
def pixel_shuffle(tokens, rows, cols, r=2):     # tokens: (rows*cols, d)
    d = tokens.shape[1]
    x = tokens.reshape(rows // r, r, cols // r, r, d).transpose(0, 2, 1, 3, 4)
    return x.reshape(-1, r * r * d)             # r*r times fewer tokens, r*r times longer
`}</Code>
        <p>On the encoder’s 32 × 32 grid of output vectors this gives 256 tokens. A projector MLP then maps each one to the language model’s width.</p>
        <Code
          title="Step 4: one sequence for the language model"
          setup={`import numpy as np
rng = np.random.default_rng(0)
# Tiny untrained stand-ins, so the shapes can flow end to end
d, D_llm, vocab = 64, 96, 50
def pixel_shuffle(tokens, rows, cols, r=2):
    x = tokens.reshape(rows // r, r, cols // r, r, tokens.shape[1]).transpose(0, 2, 1, 3, 4)
    return x.reshape(-1, r * r * tokens.shape[1])
tokens = rng.normal(size=(32 * 32, d))                        # patch tokens from Step 2
vision_encoder = lambda t: t                                  # a real one is a ViT; shapes unchanged
W_proj = rng.normal(size=(4 * d, D_llm)) * 0.05
projector = lambda t: t @ W_proj                              # a real one is a small MLP
E = rng.normal(size=(vocab, D_llm)) * 0.05
embed = lambda ids: E[ids]                                    # the text embedding table
W_out = rng.normal(size=(D_llm, vocab)) * 0.05
language_model = lambda x: x @ W_out                          # stands in for the whole GPT
text_before = [3, 17, 8]                                      # e.g. "Customer sent:"
text_after = [21, 5, 9, 30, 2, 11]                            # e.g. "Why did the payment fail?"`}
          show={`print("image tokens:", image_tokens.shape, " sequence x:", x.shape, " logits:", logits.shape)
print("3 text + 256 image + 6 text =", 3 + 256 + 6, "positions")`}
        >{`
image_tokens = projector(pixel_shuffle(vision_encoder(tokens), 32, 32))   # (256, D_llm)
x = np.concatenate([embed(text_before), image_tokens, embed(text_after)])  # (T, D_llm)
logits = language_model(x)      # from here on: the GPT you already built
`}</Code>
        <p>The language model never learns it is looking at a picture. It receives a <code>(T, D)</code> array, as always: image support is an input adapter in front of an unchanged interface, like a new deserializer in front of an existing service.</p>
      </CodeIt>

      <BreakIt>
        <p>Predict first, then check in the lab.</p>
        <ul>
          <li><b>Patch size 4 × 4.</b> How many tokens for the 112 × 112 picture? (28 × 28 = 784.) Each patch is now only 48 numbers: tiny squares, long sequence.</li>
          <li><b>Patch size 28 × 28.</b> Only 16 tokens. Can one token still “contain” the white cross, or is it mixed in with the red circle and the background? Big patches are cheap and blurry.</li>
          <li><b>Sunset picture, click two patches in the sky.</b> Their numbers are almost identical. That redundancy is why merging 2 × 2 neighbours costs little.</li>
          <li><b>Real sizes: 1344 × 1344, 14-pixel patches, no merging, 8,192 context.</b> 96 × 96 = 9,216 patches: the image alone does not fit. Turn merging on: 2,304 tokens.</li>
          <li><b>Thought experiment: remove the position vectors.</b> Shuffle the patches of the screenshot. Without positions, the encoder receives the same set of vectors either way, so it cannot tell the button is below the cross.</li>
        </ul>
      </BreakIt>

      <Exercises>
        <p>Back to the ticket with no text. This piece of the <a href="#/project">Paisa Pal support bot</a> turns the failed-payment screenshot into tokens: cut it into patches, project them, add positions, and put them in the same sequence as the customer’s words.</p>
        <CodeExercise id="multimodal-code-bot-screenshot" />

        <Exercise
          id="multimodal-llava"
          type="calculate"
          title="Tokens for one LLaVA-1.5 image"
          answer={{ value: 576, tolerance: 0 }}
          answerLabel="image tokens"
          hints={[
            'Patches per side = image side ÷ patch side.',
            '336 ÷ 14 = 24 patches per side. There is no merging in this model.',
            '24 × 24 = ?',
          ]}
          solution={<><p>336 / 14 = 24, and 24 × 24 = <b>576</b> tokens.</p><p>That is the fixed cost of every image in LLaVA-1.5, which uses the CLIP ViT-L/14 encoder at 336 × 336. A 20-word question next to it is a rounding error.</p></>}
        >
          <p>LLaVA-1.5 resizes every image to 336 × 336 and uses an encoder with 14 × 14 patches, with no merging. How many image tokens does the language model receive per image?</p>
        </Exercise>

        <Exercise
          id="multimodal-tiles"
          type="calculate"
          title="A long receipt"
          answer={{ value: 1024, tolerance: 0 }}
          answerLabel="tokens"
          hints={[
            'Count tiles along each side: 448 ÷ 448 and 1344 ÷ 448.',
            '1 × 3 = 3 tiles. More than one tile means a thumbnail is added.',
            '(3 + 1) tiles × 256 tokens per tile.',
          ]}
          solution={<><p>1 × 3 = 3 tiles, plus 1 thumbnail = 4, and 4 × 256 = <b>1,024</b> tokens.</p><p>Each tile is 448 × 448 → 32 × 32 = 1,024 patches → 256 tokens after 2 × 2 merging. The thumbnail costs the same as any tile: it is there so the model sees the whole receipt at once, not only three separate strips.</p></>}
        >
          <p>A customer sends a photo of a long receipt, resized to 448 wide × 1344 tall. The model uses 448 × 448 tiles, 256 tokens per tile, and adds one thumbnail whenever there is more than one tile. How many image tokens?</p>
        </Exercise>

        <Exercise
          id="multimodal-debug"
          type="debug"
          title="The patches are stripes"
          hints={[
            'Try it on a tiny 4 × 4 image with p = 2. What does patch 1 contain?',
            'A plain reshape keeps pixels in memory order: all of row 0, then all of row 1. Which pixels end up in the first 2 × 2 × 3 = 12 numbers?',
          ]}
          solution={<><p>Without the transpose, <code>reshape(-1, p*p*C)</code> cuts the image in memory order, so each “patch” is a run of consecutive pixels from one or two rows: a thin horizontal strip, not a square. On a 4 × 4 test image with p = 2, the second “patch” is the whole of row 1 instead of the top-right 2 × 2 square.</p><p>Nothing crashes: the shape is correct, <code>(num_patches, p*p*C)</code>. The model still trains, only worse, because each token now mixes far-apart pixels. Shape checks cannot catch this; a tiny hand-made test like the one in the hint can.</p></>}
        >
          <p>A colleague shortens <code>patchify</code> to one line and the vision model trains poorly. The shapes look right. What is wrong?</p>
          <Code
            setup={`import numpy as np
# pixel (row r, col c) holds the numbers [r, c, 0], so you can see where each value came from
img = np.array([[[r, c, 0] for c in range(4)] for r in range(4)])`}
            show={`bad = patchify(img, 2)
print("shape looks right:", bad.shape)
print("patch 1 (should be the top-right 2 x 2 square), as (row, col) pairs:")
print(bad[1].reshape(-1, 3)[:, :2].tolist())`}
          >{`
def patchify(img, p):
    H, W, C = img.shape
    return img.reshape(-1, p * p * C)
`}</Code>
        </Exercise>

        <ExplainBack
          id="multimodal-explain"
          prompt="Dev still thinks the model “runs OCR and reads the text”. In plain words, explain what actually happens to the screenshot, and one thing the model can use that OCR would lose."
          modelAnswer={<p>The picture is cut into small squares, for example 14 by 14 pixels. Each square’s pixel values are written out as a list of numbers and multiplied by a learned matrix, which turns the square into one vector, the same kind of vector a word becomes. A vision encoder (a Transformer trained on images and their captions) lets all the squares look at each other, and a small projector adapts the result for the language model. Then the language model reads those image vectors and the text tokens as one sequence and writes its answer as usual. Nothing is converted to text on the way, so the model can use things OCR drops: the red cross, which button is greyed out, the layout, a chart. It often reads the printed words well too, because it learned to during training, but that is a learned skill, not a separate OCR step.</p>}
        />
        <details className="deep">
          <summary>More practice (optional)</summary>
          <div className="details-body">
            <Exercise
              id="multimodal-audio"
              type="calculate"
              title="A ten-minute support call"
              answer={{ value: 30000, tolerance: 0 }}
              answerLabel="encoder vectors"
              hints={[
                'A Whisper-style spectrogram has one column every 10 ms: 100 per second.',
                '10 minutes = 600 s → 60,000 columns.',
                'The encoder’s stride-2 convolution halves the length.',
              ]}
              solution={<><p>600 s × 100 = 60,000 columns, halved = <b>30,000</b> encoder vectors (one per 20 ms).</p><p>That is far more than a typical transcript of the same call (a few thousand text tokens). Audio models pool further, and many systems still transcribe first when they only need the words.</p></>}
            >
              <p>Paisa Pal wants to analyse recorded support calls. A Whisper-style encoder makes one spectrogram column per 10 ms and then halves the sequence length. How many encoder output vectors for a 10-minute call? (Ignore the 30-second chunking; the total is the same.)</p>
            </Exercise>
          </div>
        </details>
      </Exercises>

      <CheckYourself
        questions={[
          {
            q: 'What is one image token in a ViT-style model?',
            options: ['One pixel’s three colour values', 'A row looked up from an image vocabulary', 'One P × P patch, flattened and multiplied by a learned matrix, plus a position vector', 'A word produced by running OCR on the image'],
            answer: 2,
            explain: 'No vocabulary lookup: the patch’s own numbers go through a matrix multiply to become a D-wide vector.',
          },
          {
            q: 'What does contrastive training (CLIP, SigLIP) teach the image encoder?',
            options: ['To output the caption word by word', 'To give an image a vector close to its own caption’s vector and far from other captions', 'To compress images into JPEG', 'To classify images into 1,000 fixed labels'],
            answer: 1,
            explain: 'Pairs from the web are the only supervision: match the right image to the right text.',
          },
          {
            q: 'What is the projector in a LLaVA-style model for?',
            options: ['It turns the image into text', 'It maps vision-encoder vectors to the width and space of the language model’s token embeddings', 'It removes the position information', 'It generates images'],
            answer: 1,
            explain: 'A single matrix or small MLP. After it, image tokens can sit in the same sequence as text tokens.',
          },
        ]}
      />

      <Remember
        items={[
          <>A model “sees” by turning a picture into a <b>sequence of vectors</b>: cut into P × P patches, flatten, multiply by a learned matrix, add a position. One patch = one token. Sound works the same way: a <b>log-mel spectrogram</b> (100 columns per second) → an encoder (Whisper: one vector per 20 ms) → tokens.</>,
          <>Typical open design: <b>vision encoder</b> (a Transformer, often trained contrastively on image-caption pairs) → <b>projector</b> → the <b>language model</b>, which reads image and text tokens as one sequence.</>,
          <>Token cost grows with <b>area</b>: N = (H/P)(W/P)/r². 448 × 448 at 14 px is 1,024 patches, 256 tokens after 2 × 2 merging. Tiling adds tokens per tile.</>,
          <>Late fusion (bolted-on encoder) is well documented in open models. How closed “natively multimodal” models are built is largely <b>not published</b>.</>,
        ]}
      />

      <RealLLM>
        <Flow horizontal steps={[{ label: 'Tokenization', sub: 'text → IDs' }, { label: 'This lesson', sub: 'pixels and sound → vectors' }, { label: 'Transformer', sub: 'unchanged' }, { label: 'Next token', sub: 'text, or audio/image tokens' }]} active={1} />
        <ToyVsReal
          toy={<ul><li>112 × 112 built-in pictures, drawn in code</li><li>Random, untrained projection to at most 12 numbers</li><li>No vision encoder: patch tokens are the end of the story</li><li>Token counts computed with a simplified tiling rule</li></ul>}
          real={<ul><li>Images resized, tiled, or processed at their native aspect ratio (Qwen2-VL: one token per 28 × 28 pixels after merging)</li><li>Learned projection to 1,000 or more numbers per token</li><li>A 300-million to 6-billion-parameter vision encoder, then a projector</li><li>Each model family has its own resize, tile and cap rules</li></ul>}
        />
        <h3>Generating pictures, briefly</h3>
        <p>Reading an image and drawing one are different problems. Two families dominate:</p>
        <ul>
          <li><b>Diffusion.</b> Start from pure noise and repeatedly remove a little of it, guided by the text prompt, until an image remains. Stable Diffusion and many commercial image generators work this way, usually in a compressed “latent” space rather than on raw pixels.</li>
          <li><b>Autoregressive image tokens.</b> Turn images into discrete tokens with a learned codebook, then predict them one after another, exactly like text. Chameleon is an open example: one model writes text and images in a single stream.</li>
        </ul>
        <p>Since 2025 some chat assistants generate images “natively” rather than by handing off to a separate model, and hybrids exist. Which recipe a given closed product uses is mostly <b>not disclosed</b>. What is established: both families work, and both need huge amounts of paired image-text data.</p>
        <Callout kind="established">
          For reading images, the pipeline in this lesson is not a simplification of the open models: LLaVA, InternVL and Qwen-VL publish their code, and it is patchify → vision Transformer → merge → projector → language model. The differences are in sizes, resolutions and training data.
        </Callout>
        <p>Riya writes the week’s ticket numbers on the whiteboard next to Kabir’s map. Every screenshot is about 1,800 tokens before the customer has said a word. “So images go in the budget,” she says. Kabir adds one line under “tokenizer”: <em>pixels too</em>.</p>
      </RealLLM>
    </Lesson>
  )
}
