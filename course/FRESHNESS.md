# Keeping the 2026 material current

The course makes claims that a new model, paper or release can make wrong (model configs, context lengths,
serving tools, hardware, benchmarks). Every such claim is listed in `src/data/freshness.ts` with its lesson,
its sources and the date it was last checked. Lessons show "Facts checked <month>" from those dates.

## The monthly review

A scheduled agent runs this review once a month and opens a GitHub issue titled
`Freshness review <YYYY-MM>` with its findings. It never edits the course or pushes; a person decides.

1. **Due claims.** `npx vite-node scripts/freshness.ts` lists the claims whose review date has passed.
   For each, open its sources, search for anything newer, and decide: still true, needs an edit, or needs a new
   source. Give the exact lesson file and sentence to change, the proposed wording, and the source URL.
2. **New developments.** Search the past month for developments the course should mention or that contradict it:
   new open-weight flagship models and their architecture (attention type, MoE layout, context), training
   recipes (RL, distillation, data), serving and inference (vLLM, SGLang, TensorRT-LLM, disaggregation,
   quantization formats, new GPUs), agents and tool protocols (MCP), evals and benchmarks, alignment and
   interpretability. Only report what changes something a lesson says, or a clear gap; ignore hype and rumours.
3. **Sources.** Prefer primary sources: papers (arXiv), model cards and `config.json` on Hugging Face, official
   docs and release notes. Say when a fact could only be found in secondary coverage.
4. **Proposed register changes.** New `FreshClaim` entries for new time-sensitive claims, and new `checked`
   dates for claims confirmed unchanged.

## Applying a review

Edit the lesson with the smallest precise change (AUTHORING.md rules: story voice, no em dashes, at most 4
callouts, never change exercise ids), update the claim's `checked` date and sources in `freshness.ts`, and run
`PYTHON=/usr/bin/python3 npx vitest run`. Claims about closed models stay hedged ("not public") unless the vendor
documented them.
