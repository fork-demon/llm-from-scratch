# The story voice

Read `AUTHORING.md` first. Every rule there still holds: the 12-section order, honesty labels,
exact numbers, no em dashes, no emoji, no "simply / obviously / just". This file adds one thing:
**a story that runs through the whole course**, so a learner keeps reading because they want to
know what happens to Riya next, and understands each idea because it arrives inside a moment they
recognise from their own life.

The feel we want: simple, warm, everyday Indian English, the kind of writing people read on a train
and finish. Short sentences. Small real moments (a kitchen, an auto ride, a chai stall, an exam hall,
a WhatsApp group). Light humour. Kindness. A small lesson in each moment, never preached. It must be
**original** prose: never copy or closely imitate a passage from any published author.

## The cast (keep them consistent everywhere)

- **Riya** (26). Backend developer (Java and Python) at *Paisa Pal*, a mid-size fintech in Bengaluru.
  Good engineer, zero ML. Curious, a little stubborn, hates magic she cannot explain. The reader is Riya.
- **Kabir** (38). Staff ML engineer at Paisa Pal. Calm, dry humour, draws everything on a whiteboard.
  Catchphrase, used rarely: "Don't memorise it. Build it." Never lectures for more than a few lines.
- **Amma**. Riya's mother, retired school maths teacher in Mysuru, on the phone or visiting.
  Source of everyday analogies (cooking by taste, a teacher correcting 40 notebooks, the ration shop
  queue, a train timetable). Wise, practical, often funnier than she lets on. Use at most once per lesson.
- **Dev**. Riya's college friend and flatmate, a product manager. Confident, uses AI tools all day,
  believes every myth ("it's just autocomplete", "it searches Google", "more parameters = smarter").
  He voices the misconception the lesson then corrects. Never mock him; he is the reader's doubt.

## The arc (who is where, lesson by lesson)

Part 0. Paisa Pal's CEO wants "our own ChatGPT for customer support". Riya is moved to the new AI
team. She realises she uses LLMs daily and cannot explain one. Kabir: "Good. Then we start at the bottom."
- prompt-to-answer: Riya types "What is a cat?" to test the vendor chatbot; watches it stream; Dev says "it looked it up".
- surprising-idea: Riya's phone keyboard predicting "Happy birthday ___" on Amma's birthday.
- course-map: Kabir draws the whole map on the whiteboard; Riya photographs it. This map is the course.

Part 1 (maths). Evenings, Amma on video call helping with the "school maths" Riya forgot.
- vectors: matching flatmate preferences / movie taste as lists of numbers.
- matrices: Amma grading a whole class sheet at once instead of one notebook at a time.
- softmax: the office lunch vote turning scores into shares.
- derivatives: adjusting the pressure-cooker flame: nudge, see what changes.

Part 2. gradient-descent: Amma's sambar, tasting and correcting salt. Loss = taste, gradient = which way, learning rate = size of the pinch.

Part 3. neurons: Riya's first model fails on a spiral; Kabir: "a straight line cannot fold". backprop: a bad
release post-mortem, tracing blame backward through the services.

Part 4. tokenization: why Kannada, Hindi and code cost more tokens on the vendor bill (Riya checks the invoice).
embeddings: the Bengaluru metro map as a space where nearby stations are similar.

Part 5. next-token: Riya builds a bigram model on old support tickets; it writes nonsense that sounds like a ticket.
context-wall: Dev's joke that only makes sense if you remember the first line.

Part 6. attention: "The customer called the bank because it was closed" meets "river bank" on a weekend trip to Srirangapatna.
masks-and-heads: exam hall, no looking at the answers ahead; many heads = a panel of interviewers each listening for one thing.
transformer-block: the office floor: a meeting (attention) then everyone goes back to their desk to think (MLP).

Part 7. build-gpt: Riya traces one token through her own GPT, late evening, office empty.
training-gpt: 2 a.m., the loss curve falls for the first time; she sends Kabir a screenshot; he replies with one thumbs-up word.
inference: the demo to the team is slow; the KV cache as not re-reading the whole chat every time.

Part 8. why-llms-know: the chatbot confidently invents a Paisa Pal refund policy. Dev: "but it sounded so sure".
modern-architecture: Kabir opens a real 2026 open model config; everything Riya built is there, plus new tricks.
training-pipeline: from a model that continues text to one that answers; like a bright new joiner learning the company's way of replying.
distillation: a senior teaching a junior, tuition-class style; the small model on the support team's laptops.
alignment-safety: a customer tries to trick the bot into revealing another user's account; what "should not" means.
multimodal: customers send photos of failed-payment screenshots; how the model reads an image.
interpretability: Riya asks "what is it actually thinking?"; Kabir shows what we can and cannot read.
reasoning-models: Amma solving a puzzle aloud versus answering instantly.

Part 9. rag: the bot must answer from Paisa Pal's policy PDFs, not memory. fine-tuning: making it reply in the
company's tone. agents: letting it check a real transaction status with a tool, and the first scary prompt injection.

Part 10. evals: "the demo looked great" versus 200 real tickets. inference-systems: Diwali sale traffic.
making-models-cheaper: the GPU bill meeting with finance. pytorch-bridge: Riya opens a real Hugging Face model and
recognises every tensor. reading-papers: Riya reads "Attention Is All You Need" on a train to Mysuru.
production-agents: a production incident at 11 p.m.

Part 11. capstone: Riya builds the whole system; launch day. from-memory: Riya explains the whole machine to Amma
at the dining table without notes, and Amma asks the best question of the evening.

## Where the story goes, and where it does not

- **`<Why>`**: open with a short scene (60 to 170 words), then the question. The scene must lead straight
  into the technical question; no scene for its own sake.
- **`<Problem>` / `<MentalModel>`**: characters may appear in 1 to 3 short lines (Dev's myth, Kabir's whiteboard
  question, Amma's analogy in a `<Callout kind="analogy">` that still says where the analogy stops).
- **`<Numbers>`, `<TheMath>`, `<CodeIt>`**: exact and technical. One framing sentence with a character is allowed;
  the content stays precise. Stories never replace a calculation.
- **Exercises**: may be framed as Riya's situation ("Riya's model prints..."), but hints and solutions stay exact.
- **`<RealLLM>`**: may close with one line that returns to the scene.
- Each lesson ends Riya a little further along the arc than it found her.

## Voice rules

- Talk to the reader as "you". Riya is the example, you are the one learning.
- One idea per paragraph, 1 to 3 sentences. Read it aloud: if you run out of breath, split it.
- Replace jargon with plain words first, then name it. "A list of numbers (a *vector*)".
- Dialogue: short, natural, Indian English is welcome in speech ("na", "yaar", "arre") but at most once or twice a
  lesson and never in explanations. Narration is plain standard English.
- Humour is gentle and comes from the situation, never at anyone's expense.
- Never call something easy. Say "this is the part everyone finds strange at first" instead.
- No brand names for the fictional company's vendors. Real models, papers and people may be named when factual.
