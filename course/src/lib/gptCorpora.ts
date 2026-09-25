// Small built-in training texts for the in-browser GPT trainer (GptTrainer).
// Shakespeare is public domain. The Paisa Pal tickets are written for this course (Paisa Pal is the course's
// fictional Indian fintech) and generated from a small seeded grammar, so the text is deterministic.
import { makeRng } from './rng'
import { makeCorpus } from './tinyLm'

export const SHAKESPEARE = `Shall I compare thee to a summer's day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date;
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimm'd;
And every fair from fair sometime declines,
By chance or nature's changing course untrimm'd;
But thy eternal summer shall not fade,
Nor lose possession of that fair thou ow'st;
Nor shall Death brag thou wander'st in his shade,
When in eternal lines to time thou grow'st:
So long as men can breathe or eyes can see,
So long lives this, and this gives life to thee.

To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep: perchance to dream: ay, there's the rub;
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause: there's the respect
That makes calamity of so long life.

All the world's a stage,
And all the men and women merely players:
They have their exits and their entrances;
And one man in his time plays many parts,
His acts being seven ages. At first the infant,
Mewling and puking in the nurse's arms.
And then the whining school-boy, with his satchel
And shining morning face, creeping like snail
Unwillingly to school. And then the lover,
Sighing like furnace, with a woeful ballad
Made to his mistress' eyebrow. Then a soldier,
Full of strange oaths and bearded like the pard,
Jealous in honour, sudden and quick in quarrel,
Seeking the bubble reputation
Even in the cannon's mouth.

Friends, Romans, countrymen, lend me your ears;
I come to bury Caesar, not to praise him.
The evil that men do lives after them;
The good is oft interred with their bones;
So let it be with Caesar. The noble Brutus
Hath told you Caesar was ambitious:
If it were so, it was a grievous fault,
And grievously hath Caesar answer'd it.

O for a Muse of fire, that would ascend
The brightest heaven of invention,
A kingdom for a stage, princes to act
And monarchs to behold the swelling scene!
Then should the warlike Harry, like himself,
Assume the port of Mars; and at his heels,
Leash'd in like hounds, should famine, sword and fire
Crouch for employment.

Once more unto the breach, dear friends, once more;
Or close the wall up with our English dead.
In peace there's nothing so becomes a man
As modest stillness and humility:
But when the blast of war blows in our ears,
Then imitate the action of the tiger;
Stiffen the sinews, summon up the blood.

To-morrow, and to-morrow, and to-morrow,
Creeps in this petty pace from day to day,
To the last syllable of recorded time;
And all our yesterdays have lighted fools
The way to dusty death. Out, out, brief candle!
Life's but a walking shadow, a poor player
That struts and frets his hour upon the stage
And then is heard no more. It is a tale
Told by an idiot, full of sound and fury,
Signifying nothing.

Let me not to the marriage of true minds
Admit impediments. Love is not love
Which alters when it alteration finds,
Or bends with the remover to remove:
O no! it is an ever-fixed mark
That looks on tempests and is never shaken;
It is the star to every wandering bark,
Whose worth's unknown, although his height be taken.

The quality of mercy is not strain'd,
It droppeth as the gentle rain from heaven
Upon the place beneath: it is twice blest;
It blesseth him that gives and him that takes:
'Tis mightiest in the mightiest: it becomes
The throned monarch better than his crown.

If music be the food of love, play on;
Give me excess of it, that, surfeiting,
The appetite may sicken, and so die.
That strain again! it had a dying fall:
O, it came o'er my ear like the sweet sound,
That breathes upon a bank of violets,
Stealing and giving odour!

But, soft! what light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon,
Who is already sick and pale with grief,
That thou her maid art far more fair than she.

Now is the winter of our discontent
Made glorious summer by this sun of York;
And all the clouds that lour'd upon our house
In the deep bosom of the ocean buried.
`

const NAMES = ['Priya', 'Rahul', 'Ananya', 'Vikram', 'Sneha', 'Arjun', 'Meera', 'Karan', 'Divya', 'Rohan', 'Kavya', 'Amit']
const PAYEES = ['Sharma Kirana', 'my landlord', 'Swiggy', 'the electricity board', 'my brother', 'Metro card recharge', 'Big Bazaar', 'the gas agency']
const AMOUNTS = ['Rs 99', 'Rs 250', 'Rs 499', 'Rs 1,200', 'Rs 2,000', 'Rs 5,000', 'Rs 12,500']
const BANKS = ['SBI', 'HDFC', 'ICICI', 'Axis', 'Canara Bank']
const WHEN = ['yesterday', 'this morning', 'on Monday', 'two days back', 'last night']

interface Kind { tag: string; ask: (f: Fill) => string; reply: (f: Fill) => string }
interface Fill { name: string; payee: string; amount: string; bank: string; when: string }

const KINDS: Kind[] = [
  {
    tag: 'UPI',
    ask: (f) => `I paid ${f.amount} to ${f.payee} ${f.when}. Payment failed but amount got debited from my ${f.bank} account. Please refund.`,
    reply: (f) => `Sorry for the trouble, ${f.name}. Failed UPI payments are auto-refunded to your ${f.bank} account within 3 to 5 working days.`,
  },
  {
    tag: 'KYC',
    ask: (f) => `My KYC is pending since ${f.when}. I uploaded my PAN and Aadhaar twice. Kindly do the needful.`,
    reply: (f) => `Thank you, ${f.name}. Your KYC documents are under review. You will get an SMS once it is approved.`,
  },
  {
    tag: 'CASHBACK',
    ask: (f) => `I did not receive cashback for my ${f.amount} payment to ${f.payee}. The offer said 10 percent.`,
    reply: (f) => `Hi ${f.name}, cashback is credited within 48 hours of the payment. Please check the Rewards tab after that.`,
  },
  {
    tag: 'LOGIN',
    ask: () => `I am not getting the OTP on my phone. I tried to login many times and now my account is locked.`,
    reply: (f) => `Sorry, ${f.name}. For your safety the account is locked for 30 minutes. Please try again after that.`,
  },
  {
    tag: 'REFUND',
    ask: (f) => `${f.payee} cancelled my order ${f.when} but the refund of ${f.amount} is not showing in Paisa Pal.`,
    reply: (f) => `Hi ${f.name}, merchant refunds take 5 to 7 working days. If it is not credited by then, reply to this ticket.`,
  },
  {
    tag: 'WALLET',
    ask: (f) => `I added ${f.amount} to my Paisa Pal wallet ${f.when} using ${f.bank} net banking but the wallet balance is still zero.`,
    reply: (f) => `Thank you for waiting, ${f.name}. The ${f.amount} has now been added to your wallet. Please refresh the app.`,
  },
]

/** A deterministic set of support tickets and replies, in a slightly formal Indian-English register. */
export const makePaisaPalTickets = (count = 220, seed = 42): string => {
  const rng = makeRng(seed)
  const pick = <T,>(a: T[]): T => a[rng.int(a.length)]
  let out = ''
  for (let i = 0; i < count; i++) {
    const f: Fill = { name: pick(NAMES), payee: pick(PAYEES), amount: pick(AMOUNTS), bank: pick(BANKS), when: pick(WHEN) }
    const k = pick(KINDS)
    out += `Ticket ${4100 + i} [${k.tag}]\n${f.name}: ${k.ask(f)}\nPaisa Pal: ${k.reply(f)}\n\n`
  }
  return out
}

export interface Corpus { id: string; label: string; blurb: string; text: string }

export const CORPORA: Corpus[] = [
  { id: 'shakespeare', label: 'Shakespeare', blurb: 'Famous speeches and sonnets (public domain). Hard: every line is different.', text: SHAKESPEARE },
  { id: 'tickets', label: 'Paisa Pal tickets', blurb: 'Support tickets and replies written for this course. Very repetitive, so the model learns the pattern fast.', text: makePaisaPalTickets() },
  { id: 'sentences', label: 'Simple sentences', blurb: 'Tiny made-up sentences from a small grammar (the same text as the MLP trainer).', text: makeCorpus(7, 400) },
]
