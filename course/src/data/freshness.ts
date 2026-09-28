// The freshness register: every claim in the course that a new model, paper or release can make wrong.
// Each entry says where the claim lives, what it says, the sources it was checked against, and when.
// A monthly review (see FRESHNESS.md) re-checks due entries, updates the lesson and the `checked` date,
// and adds entries for new time-sensitive claims. Lessons show "last checked" from the newest date here.

export interface FreshClaim {
  id: string
  lesson: string
  /** The claim as the lesson states it, short. */
  claim: string
  sources: string[]
  /** ISO date the claim was last verified against the sources. */
  checked: string
  /** How often to re-check, in months: 3 for fast-moving facts (models, tools), 12 for slow ones. */
  every: 3 | 6 | 12
}

const SEP_2026 = '2026-09-27'

export const FRESHNESS: FreshClaim[] = [
  // ---- architecture and models ----
  { id: 'deepseek-v3-config', lesson: 'modern-architecture', every: 12, checked: SEP_2026,
    claim: 'DeepSeek-V3: 61 layers, width 7168, MLA (kv_lora_rank 512), 256 routed + 1 shared experts, 8 active, first 3 blocks dense',
    sources: ['https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json', 'https://arxiv.org/abs/2412.19437'] },
  { id: 'deepseek-v32-dsa', lesson: 'modern-architecture', every: 6, checked: SEP_2026,
    claim: 'DeepSeek-V3.2 sparse attention: a lightning indexer picks the top 2,048 earlier tokens per query',
    sources: ['https://arxiv.org/abs/2512.02556'] },
  { id: 'deepseek-v4', lesson: 'modern-architecture', every: 3, checked: SEP_2026,
    claim: 'DeepSeek-V4-Pro: 1.6T total / 49B active, 384+1 experts (6+1 active), 1M context, compressed (4x, 128x) + sparse attention, ~10% of V3.2 KV cache at 1M tokens',
    sources: ['https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro', 'https://huggingface.co/blog/deepseekv4'] },
  { id: 'qwen35-hybrid', lesson: 'modern-architecture', every: 3, checked: SEP_2026,
    claim: 'Qwen3.5-397B-A17B: 60 layers, 3 Gated DeltaNet : 1 gated attention, 512 experts (10+1 active), 262,144 context',
    sources: ['https://huggingface.co/Qwen/Qwen3.5-397B-A17B'] },
  { id: 'model-size-context-range', lesson: 'modern-architecture', every: 3, checked: SEP_2026,
    claim: 'Open models range from well under 1B (Gemma 3 270M) to 1.6T (DeepSeek-V4-Pro); ~128K context common, some 262K to 1M',
    sources: ['https://huggingface.co/google/gemma-3-270m', 'https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro'] },
  { id: 'nsa-moba', lesson: 'modern-architecture', every: 12, checked: SEP_2026,
    claim: 'NSA: compressed, selected and sliding-window branches; MoBA routes queries to blocks of keys',
    sources: ['https://arxiv.org/abs/2502.11089', 'https://arxiv.org/abs/2502.13189'] },
  { id: 'gated-deltanet', lesson: 'modern-architecture', every: 12, checked: SEP_2026,
    claim: 'Gated DeltaNet combines Mamba-2 style decay with the delta rule',
    sources: ['https://arxiv.org/abs/2412.06464'] },
  { id: 'moe-routing-specialisation', lesson: 'modern-architecture', every: 12, checked: SEP_2026,
    claim: 'Mixtral routing tracks token patterns; fine-grained MoEs (OLMoE) show some domain specialisation; debated',
    sources: ['https://arxiv.org/abs/2409.02060'] },
  { id: 'gpt-oss-sinks', lesson: 'modern-architecture', every: 12, checked: SEP_2026,
    claim: 'gpt-oss uses a 128-token sliding window in alternate layers with learned per-head attention sinks, MXFP4 weights',
    sources: ['https://arxiv.org/abs/2508.10925'] },
  { id: 'muon-optimizer', lesson: 'training-gpt', every: 6, checked: SEP_2026,
    claim: 'Most runs use AdamW; some recent frontier runs (Kimi K2, GLM-4.5) use Muon variants',
    sources: ['https://arxiv.org/abs/2507.20534'] },
  { id: 'vocab-sizes', lesson: 'tokenization', every: 6, checked: SEP_2026,
    claim: 'Vocabularies range from about 30,000 to about 260,000 tokens (Gemma 3: 262,144)',
    sources: ['https://huggingface.co/google/gemma-3-27b-it'] },
  { id: 'hybrid-models-released', lesson: 'context-wall', every: 6, checked: SEP_2026,
    claim: 'Several released models mix recurrent-style layers with attention (Jamba, Nemotron-H, Qwen3-Next, MiniMax-M1, Kimi Linear)',
    sources: ['https://arxiv.org/abs/2403.19887', 'https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct'] },

  // ---- training, data, alignment ----
  { id: 'training-parallelism-layouts', lesson: 'training-pipeline', every: 12, checked: SEP_2026,
    claim: 'Llama 3 405B: 16,384 H100s, TP 8, PP 16, FSDP-style DP, CP 16 for long context; DeepSeek-V3: 2,048 H800s, PP 16, EP 64, ZeRO-1, no TP',
    sources: ['https://arxiv.org/abs/2407.21783', 'https://arxiv.org/abs/2412.19437'] },
  { id: 'fineweb-edu', lesson: 'training-pipeline', every: 12, checked: SEP_2026,
    claim: 'FineWeb: 15T tokens from 96 crawls; FineWeb-Edu keeps pages scored 3+ by a classifier trained on Llama-3-70B labels, 1.3T tokens',
    sources: ['https://arxiv.org/abs/2406.17557'] },
  { id: 'r1-recipe', lesson: 'reasoning-models', every: 12, checked: SEP_2026,
    claim: 'DeepSeek-R1: cold-start SFT, reasoning RL (GRPO, rule rewards), rejection-sampling SFT on ~800k samples, final RL with learned helpfulness/harmlessness rewards',
    sources: ['https://arxiv.org/abs/2501.12948'] },
  { id: 'hybrid-thinking', lesson: 'reasoning-models', every: 3, checked: SEP_2026,
    claim: 'Qwen went back to separate Instruct and Thinking models (Qwen3-2507); DeepSeek-V3.1 and Claude extended thinking keep both modes',
    sources: ['https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507'] },
  { id: 'distillation-examples', lesson: 'distillation', every: 12, checked: SEP_2026,
    claim: 'R1 distilled into six models on ~800k samples; Gemma 2 small models and Llama 3.2 1B/3B trained with teacher logits (Llama 3.2 also pruned from 3.1 8B)',
    sources: ['https://arxiv.org/abs/2501.12948', 'https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/'] },
  { id: 'gemma-scope-2', lesson: 'interpretability', every: 6, checked: SEP_2026,
    claim: 'Gemma Scope 2 (December 2025) covers the Gemma 3 family with SAEs, transcoders and crosscoders',
    sources: ['https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/'] },
  { id: 'native-resolution-vlms', lesson: 'multimodal', every: 6, checked: SEP_2026,
    claim: 'Qwen2-VL onwards encode images at native resolution with 2D RoPE; SigLIP 2 (2025) has a native-aspect-ratio variant',
    sources: ['https://arxiv.org/abs/2409.12191', 'https://arxiv.org/abs/2502.14786'] },
  { id: 'hallucination-calibration', lesson: 'why-llms-know', every: 12, checked: SEP_2026,
    claim: 'Base models are fairly calibrated; chat tuning can hurt it; hallucination tied to singleton facts and guess-rewarding evals',
    sources: ['https://arxiv.org/abs/2207.05221', 'https://arxiv.org/abs/2509.04664'] },

  // ---- systems, serving, tools ----
  { id: 'disaggregated-serving', lesson: 'inference-systems', every: 6, checked: SEP_2026,
    claim: 'Prefill/decode disaggregation (DistServe, Splitwise, Mooncake) supported in NVIDIA Dynamo, llm-d, vLLM (experimental) and SGLang',
    sources: ['https://arxiv.org/abs/2401.09670', 'https://docs.vllm.ai/en/latest/features/disagg_prefill.html'] },
  { id: 'vllm-preemption', lesson: 'inference-systems', every: 6, checked: SEP_2026,
    claim: 'vLLM V1 preempts by recomputation; CPU KV offload goes through connectors such as LMCache',
    sources: ['https://docs.vllm.ai/en/stable/configuration/optimization/'] },
  { id: 'gpu-bandwidth', lesson: 'inference-systems', every: 6, checked: SEP_2026,
    claim: 'Memory bandwidth: A100 ~2 TB/s, H100 3.35 TB/s, H200 4.8 TB/s, B200/B300/MI355X ~8 TB/s',
    sources: ['https://www.nvidia.com/en-us/data-center/h100/', 'https://www.nvidia.com/en-us/data-center/dgx-b200/'] },
  { id: 'fp8-fp4-formats', lesson: 'making-models-cheaper', every: 6, checked: SEP_2026,
    claim: 'OCP FP8 E4M3 (max 448) on H100/Blackwell and AMD MI350+; MI300 uses FNUZ variants (max 240); MXFP4 4.25 bits, NVFP4 ~4.5 bits',
    sources: ['https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1'] },
  { id: 'prompt-caching-minimums', lesson: 'production-agents', every: 3, checked: SEP_2026,
    claim: 'Providers only cache prompts above a minimum (about 1,024 tokens is common); some cache only up to marked breakpoints',
    sources: ['https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching', 'https://platform.openai.com/docs/guides/prompt-caching'] },
  { id: 'mcp-governance', lesson: 'production-agents', every: 6, checked: SEP_2026,
    claim: 'MCP moved to the Linux Foundation’s Agentic AI Foundation (December 2025)',
    sources: ['https://blog.modelcontextprotocol.io/posts/2025-12-09-mcp-joins-agentic-ai-foundation/'] },
  { id: 'hf-generate-defaults', lesson: 'pytorch-bridge', every: 6, checked: SEP_2026,
    claim: 'generate() defaults to greedy but generation_config.json overrides it; transformers uses `dtype` (4.56+) with auto as the v5 default',
    sources: ['https://huggingface.co/docs/transformers/main_classes/text_generation'] },
  { id: 'metr-time-horizon', lesson: 'evals', every: 3, checked: SEP_2026,
    claim: 'METR task time horizon doubling ~7 months (2019-2025), ~4 months since 2023 (Time Horizon 1.1, Jan 2026)',
    sources: ['https://metr.org/blog/2026-1-29-time-horizon-1-1/'] },
  { id: 'benchmark-sizes', lesson: 'evals', every: 12, checked: SEP_2026,
    claim: 'MMLU 15,908; HumanEval 164; SWE-bench 2,294 (Verified 500); GPQA-Diamond 198; HLE ~2,500',
    sources: ['https://www.swebench.com/', 'https://lastexam.ai/'] },
]

/** A lesson's newest check date, for the "last checked" line. */
export const lastChecked = (lesson: string): string | null => {
  const dates = FRESHNESS.filter((c) => c.lesson === lesson).map((c) => c.checked).sort()
  return dates.length ? dates[dates.length - 1] : null
}

/** Claims whose review is due on `today` (ISO date). */
export const dueForReview = (today: string): FreshClaim[] =>
  FRESHNESS.filter((c) => {
    const d = new Date(c.checked)
    d.setMonth(d.getMonth() + c.every)
    return d.toISOString().slice(0, 10) <= today
  })
