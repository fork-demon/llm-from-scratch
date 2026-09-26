// The Paisa Pal support-bot project: one running build through Parts 8 to 10.
// Riya's company wants a support bot (the story's premise). Each later lesson adds one piece, written by the
// learner in the browser against hidden tests. Every piece starts from paisa_pal.py (the shared data) plus,
// where it needs them, reference versions of earlier pieces, so no exercise depends on the learner having
// finished another one. The capstone assembles the whole bot.
import PAISA_PAL from './paisa_pal.py?raw'

export { PAISA_PAL }

export interface ProjectPiece { lesson: string; exercise: string; piece: string; does: string }

/** In build order. `exercise` is the code exercise id (data/codeExercises/project-*.ts). */
export const PROJECT_PIECES: ProjectPiece[] = [
  { lesson: 'why-llms-know', exercise: 'why-llms-know-code-bot-confidence', piece: 'Confidence check', does: 'tells a known fact from a guess using the probabilities' },
  { lesson: 'modern-architecture', exercise: 'modern-architecture-code-bot-rope', piece: 'Rotary positions', does: 'RoPE on the queries and keys of your attention' },
  { lesson: 'training-pipeline', exercise: 'training-pipeline-code-bot-chat-template', piece: 'Chat template and loss mask', does: 'formats a conversation and trains only on the answer' },
  { lesson: 'distillation', exercise: 'distillation-code-bot-soft-labels', piece: 'Distillation loss', does: 'a small bot learns from a big one’s probabilities' },
  { lesson: 'alignment-safety', exercise: 'alignment-safety-code-bot-dpo', piece: 'Preference loss', does: 'DPO: prefer the helpful, honest answer' },
  { lesson: 'multimodal', exercise: 'multimodal-code-bot-screenshot', piece: 'Screenshot tokens', does: 'a payment screenshot becomes tokens the bot can read' },
  { lesson: 'interpretability', exercise: 'interpretability-code-bot-probe', piece: 'Refund detector', does: 'finds the direction inside the model that means “refund”' },
  { lesson: 'reasoning-models', exercise: 'reasoning-models-code-bot-best-of-n', piece: 'Checked answers', does: 'best-of-N with a checker, measured against majority vote' },
  { lesson: 'rag', exercise: 'rag-code-bot-retriever', piece: 'Retriever', does: 'finds the help page that answers a question, or none' },
  { lesson: 'fine-tuning', exercise: 'fine-tuning-code-bot-lora', piece: 'LoRA adapter', does: 'teaches a frozen layer the Paisa Pal tone with two small matrices' },
  { lesson: 'agents', exercise: 'agents-code-bot-tool-loop', piece: 'Tool loop', does: 'calls get_balance and refund_status, then answers' },
  { lesson: 'evals', exercise: 'evals-code-bot-eval', piece: 'Eval harness', does: 'scores the bot on the real questions, with a confidence interval' },
  { lesson: 'making-models-cheaper', exercise: 'making-models-cheaper-code-bot-int8', piece: 'int8 weights', does: 'shrinks the bot almost 4× and measures what it costs' },
  { lesson: 'production-agents', exercise: 'production-agents-code-bot-guardrails', piece: 'Guardrails', does: 'validates tool arguments and asks before moving money' },
  { lesson: 'capstone', exercise: 'capstone-code-bot-assemble', piece: 'The whole bot', does: 'retrieve, answer, call tools, refuse, and pass the eval' },
]
