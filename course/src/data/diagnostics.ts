// "Do I need this part?" questions. Each one names the lesson that teaches it, so a wrong
// answer points at what to read. Kept few and sharp: these are filters, not tests.
import type { DiagnosticQ } from '../components/Diagnostic'

export const DIAGNOSTICS: Record<string, { part: string; firstLesson: string; questions: DiagnosticQ[] }> = {
  math: {
    part: 'the maths',
    firstLesson: 'vectors',
    questions: [
      {
        q: 'Two vectors have a dot product of 0. What does that tell you?',
        options: ['They point in the same direction', 'Neither one is longer than the other', 'They are unrelated in direction: at right angles', 'At least one of them is all zeros'],
        answer: 2,
        lesson: 'vectors',
      },
      {
        q: 'You multiply a (5, 3) matrix by a (3, 2) matrix. What shape comes out?',
        options: ['(5, 2)', '(3, 3)', '(5, 3)', 'It fails: the shapes do not fit'],
        answer: 0,
        lesson: 'matrices',
      },
      {
        q: 'Why does softmax exponentiate the scores before dividing by the total?',
        options: ['To make every score positive and to turn gaps into ratios', 'To keep the largest score unchanged', 'Because exponentials are faster to compute on a GPU', 'To make the result sum to the number of tokens'],
        answer: 0,
        lesson: 'softmax',
      },
      {
        q: 'A pipeline squares its input, then multiplies by 3. By the chain rule, how sensitive is the output to the input at x = 2?',
        options: ['6', '12', '3', '4'],
        answer: 1,
        lesson: 'derivatives',
      },
    ],
  },
  'neural-nets': {
    part: 'neural networks',
    firstLesson: 'neurons',
    questions: [
      {
        q: 'Why does a network need an activation function between its layers?',
        options: ['To keep the numbers small enough to store', 'Without one, stacked layers collapse into a single straight-line function', 'To make the gradients easier to compute', 'To turn the outputs into probabilities'],
        answer: 1,
        lesson: 'neurons',
      },
      {
        q: 'For softmax with cross-entropy, the gradient at the logits is…',
        options: ['the probabilities minus the one-hot target', 'the target minus the learning rate', 'the log of the probabilities', 'the squared error of the probabilities'],
        answer: 0,
        lesson: 'backprop',
      },
      {
        q: 'Why is backpropagation used instead of nudging each weight and re-measuring?',
        options: ['Nudging gives the wrong answer', 'Nudging only works for small networks with no activation functions', 'One backward pass gives every gradient, while nudging needs one forward pass per weight', 'Backpropagation needs less memory'],
        answer: 2,
        lesson: 'backprop',
      },
    ],
  },
  attention: {
    part: 'attention',
    firstLesson: 'attention',
    questions: [
      {
        q: 'Why does each token get a separate query and key rather than one vector?',
        options: ['To make the model larger and so more capable', 'Because "what I am looking for" and "what I offer" are different things', 'Because a dot product needs two different vectors', 'To let the two be trained on different data'],
        answer: 1,
        lesson: 'attention',
      },
      {
        q: 'Attention scores are divided by the square root of d before the softmax. Why?',
        options: ['To keep the weights positive', 'To make the result independent of the batch size', 'To stop large dot products from saturating softmax, which would flatten the gradients', 'To normalise the value vectors'],
        answer: 2,
        lesson: 'attention',
      },
      {
        q: 'What does the causal mask do during training?',
        options: ['It hides padding tokens so they are not scored', 'It stops a position from attending to later positions, which hold the answer it must predict', 'It randomly drops attention weights for regularisation', 'It limits how far back a token can look'],
        answer: 1,
        lesson: 'masks-and-heads',
      },
      {
        q: 'What do residual connections in a Transformer block achieve?',
        options: ['Each sub-layer learns a correction to add, and gradients get a direct path back', 'They normalise the scale of the activations', 'They let the block be run in parallel across tokens', 'They reduce the parameter count'],
        answer: 0,
        lesson: 'transformer-block',
      },
    ],
  },
}
