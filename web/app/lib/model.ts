// Single source of truth for the site's model facts.

export const INFERENCE_URL =
  process.env.NEXT_PUBLIC_INFERENCE_URL ??
  "https://mcrao--legal-slm-125-inference-slm-web.modal.run";

export const CHAT_URL =
  process.env.NEXT_PUBLIC_CHAT_URL ??
  "https://mcrao--legal-slm-125m-chat-inference-chat-web.modal.run";

export const RAFT_URL =
  process.env.NEXT_PUBLIC_RAFT_URL ??
  "https://mcrao--legal-slm-125m-raft-inference-raft-web.modal.run";

export const HF_URL = "https://huggingface.co/jonam-ai/slm-125m-base";
export const HF_SFT_URL = "https://huggingface.co/jonam-ai/legal-slm-125m-sft";
export const HF_RAFT_URL = "https://huggingface.co/jonam-ai/legal-slm-125m-raft";

// Gemma 2 2B, fine-tuned on the same data with QLoRA — for side-by-side comparison.
export const GEMMA_URL =
  process.env.NEXT_PUBLIC_GEMMA_URL ??
  "https://mcrao--gemma-2b-legal-inference-gemma-web.modal.run";
export const HF_GEMMA_SFT_URL = "https://huggingface.co/jonam-ai/gemma-2-2b-legal-sft";
export const HF_GEMMA_RAFT_URL = "https://huggingface.co/jonam-ai/gemma-2-2b-legal-raft";

// One endpoint that serves any of our models by id (bases, SFT/RAFT, DPO/RLAIF).
export const UNIVERSAL_URL =
  process.env.NEXT_PUBLIC_UNIVERSAL_URL ??
  "https://mcrao--legal-slm-universal-inference-universal-web.modal.run";

// Model pickers for the interactive panels.
export const PLAYGROUND_MODELS = [
  { id: "jonam-ai/slm-125m-base", label: "Our 125M", sub: "2-epoch" },
  { id: "thesreedath/slm-125m-base", label: "Mentor 125M", sub: "10-epoch" },
  { id: "thesreedath/slm-500m-base", label: "Mentor 500M", sub: "517M" },
] as const;

export const CHAT_MODELS = [
  { id: "jonam-ai/legal-slm-125m-sft", label: "Our 125M", browser: true },
  { id: "jonam-ai/legal-slm-500m-sft", label: "500M", browser: false },
  { id: "jonam-ai/gemma-2-2b-legal-sft", label: "Gemma 2B", browser: false },
] as const;

export const RAFT_MODELS = [
  { id: "jonam-ai/legal-slm-125m-raft", label: "Our 125M" },
  { id: "jonam-ai/legal-slm-500m-raft", label: "500M" },
  { id: "jonam-ai/gemma-2-2b-legal-raft", label: "Gemma 2B" },
] as const;

// Alignment picker: size × base-stage × method -> a model id.
export const ALIGN_SIZES = [
  { key: "125m", label: "Our 125M", prefix: "jonam-ai/legal-slm-125m" },
  { key: "500m", label: "500M", prefix: "jonam-ai/legal-slm-500m" },
  { key: "gemma", label: "Gemma 2B", prefix: "jonam-ai/gemma-2-2b-legal" },
] as const;
export function alignModelId(sizePrefix: string, stage: "sft" | "raft", method: "dpo" | "rlaif") {
  return `${sizePrefix}-${stage}-${method}`;
}

// Per-model spec shown live in the Alignment panel: trainable params, architecture,
// training tokens, and training cost — updates with the selected size/stage/method.
export const ALIGN_SPEC = {
  "125m": {
    arch: "Llama-style · 12 layers · 768 dim · 12 heads (MHA) · 16,384 vocab · 1,024 ctx",
    trainable: "125.8M · 100% (full fine-tune)",
    sftTokens: "~1.06M (2 epochs)",
    cost: { dpo: "~$0.30", rlaif: "~$0.75" },
  },
  "500m": {
    arch: "Llama-style · 24 layers · 1,280 dim · 20 heads (MHA) · 32,768 vocab · 1,024 ctx",
    trainable: "517M · 100% (full fine-tune)",
    sftTokens: "~0.5M (1 epoch)",
    cost: { dpo: "~$0.60", rlaif: "~$1.30" },
  },
  "gemma": {
    arch: "Gemma 2 · 26 layers · 2,304 dim · 8 heads / 4 KV (GQA) · 256,000 vocab · 8,192 ctx",
    trainable: "20.8M · 0.79% (QLoRA, 4-bit NF4 base frozen)",
    sftTokens: "~1.52M (3 epochs)",
    cost: { dpo: "~$5.0", rlaif: "~$7.0" },
  },
} as const;

export const ALIGN_METHOD_TOKENS = {
  dpo: "2,223 preference pairs × 2 epochs",
  rlaif: "800 on-policy prompts (400 for Gemma) + reward model",
} as const;

// One-click sample questions for the Alignment (SFT-stage) panel.
export const ALIGN_QA_PRESETS = [
  "What must a plaintiff prove in a breach of contract claim?",
  "What is the purpose of a Form 10-K filing?",
  "What does an indemnification clause do?",
  "Explain 'preponderance of the evidence'.",
  "What are the fiduciary duties of a corporate director?",
] as const;

// Inference-optimization demos.
export const KV_URL =
  process.env.NEXT_PUBLIC_KV_URL ?? "https://mcrao--slm-125m-kvcache-kv-web.modal.run";
export const SPEC_URL =
  process.env.NEXT_PUBLIC_SPEC_URL ?? "https://mcrao--qwen-speculative-spec-web.modal.run";

// Unified model leaderboard. Regenerate with `modal run leaderboard_eval.py::run` and
// paste the JSON's "models" array here. All metrics are on the same held-out sets;
// bits/byte is tokenizer-fair (NLL per UTF-8 byte), so it is comparable across the 16k
// SLM tokenizer and Gemma's 256k tokenizer.
export type LeaderRow = {
  name: string; family: "slm" | "gemma"; kind: "base" | "sft" | "raft" | "dpo" | "rlaif";
  fmt?: "base" | "sft" | "raft";
  params: string; arch: string; note: string; id: string;
  bits_per_byte: number; token_ppl: number;
  closed_book_acc: number; grounded_acc: number; faithful_refusal: number;
};

export const LEADERBOARD: { models: LeaderRow[]; sets: Record<string, number> } = {
  models: [
    { name: "Mentor base", family: "slm", kind: "base", params: "125.8M", arch: "Llama 125M", note: "peer, 10-epoch pretrain", id: "thesreedath/slm-125m-base", bits_per_byte: 0.856, token_ppl: 15.19, closed_book_acc: 0.0, grounded_acc: 0.014, faithful_refusal: 0.0 },
    { name: "Our base", family: "slm", kind: "base", params: "125.8M", arch: "Llama 125M", note: "our 2-epoch pretrain", id: "jonam-ai/slm-125m-base", bits_per_byte: 0.849, token_ppl: 14.85, closed_book_acc: 0.0, grounded_acc: 0.014, faithful_refusal: 0.0 },
    { name: "Our SFT", family: "slm", kind: "sft", params: "125.8M", arch: "Llama 125M", note: "full fine-tune", id: "jonam-ai/legal-slm-125m-sft", bits_per_byte: 0.87, token_ppl: 15.85, closed_book_acc: 0.0, grounded_acc: 0.071, faithful_refusal: 0.0 },
    { name: "Our RAFT", family: "slm", kind: "raft", params: "125.8M", arch: "Llama 125M", note: "full fine-tune", id: "jonam-ai/legal-slm-125m-raft", bits_per_byte: 0.938, token_ppl: 19.7, closed_book_acc: 0.0, grounded_acc: 0.243, faithful_refusal: 0.0 },
    { name: "500M base (mentor)", family: "slm", kind: "base", params: "517M", arch: "Llama 500M", note: "peer 500M pretrain", id: "thesreedath/slm-500m-base", bits_per_byte: 0.784, token_ppl: 13.57, closed_book_acc: 0.0, grounded_acc: 0.143, faithful_refusal: 0.0 },
    { name: "500M SFT", family: "slm", kind: "sft", params: "517M", arch: "Llama 500M", note: "full fine-tune", id: "jonam-ai/legal-slm-500m-sft", bits_per_byte: 0.822, token_ppl: 15.38, closed_book_acc: 0.0, grounded_acc: 0.057, faithful_refusal: 0.0 },
    { name: "500M RAFT", family: "slm", kind: "raft", params: "517M", arch: "Llama 500M", note: "full fine-tune", id: "jonam-ai/legal-slm-500m-raft", bits_per_byte: 0.954, token_ppl: 23.9, closed_book_acc: 0.0, grounded_acc: 0.286, faithful_refusal: 0.2 },
    { name: "Gemma 2B (off-the-shelf)", family: "gemma", kind: "base", params: "2.61B", arch: "Gemma 2", note: "no legal training", id: "google/gemma-2-2b-it", bits_per_byte: 0.874, token_ppl: 14.31, closed_book_acc: 0.0, grounded_acc: 0.357, faithful_refusal: 0.9 },
    { name: "Gemma SFT", family: "gemma", kind: "sft", params: "2.61B", arch: "Gemma 2", note: "QLoRA", id: "jonam-ai/gemma-2-2b-legal-sft", bits_per_byte: 1.0, token_ppl: 21.03, closed_book_acc: 0.04, grounded_acc: 0.329, faithful_refusal: 0.0 },
    { name: "Gemma RAFT", family: "gemma", kind: "raft", params: "2.61B", arch: "Gemma 2", note: "QLoRA", id: "jonam-ai/gemma-2-2b-legal-raft", bits_per_byte: 0.981, token_ppl: 19.84, closed_book_acc: 0.0, grounded_acc: 0.371, faithful_refusal: 1.0 },
    { name: "Our SFT + DPO", family: "slm", kind: "dpo", params: "125.8M", arch: "Llama 125M", note: "DPO on SFT", id: "jonam-ai/legal-slm-125m-sft-dpo", bits_per_byte: 0.872, token_ppl: 15.95, closed_book_acc: 0.0, grounded_acc: 0.029, faithful_refusal: 0.0 },
    { name: "500M SFT + DPO", family: "slm", kind: "dpo", params: "517M", arch: "Llama 500M", note: "DPO on SFT", id: "jonam-ai/legal-slm-500m-sft-dpo", bits_per_byte: 0.823, token_ppl: 15.46, closed_book_acc: 0.0, grounded_acc: 0.071, faithful_refusal: 0.0 },
    { name: "Gemma SFT + DPO", family: "gemma", kind: "dpo", params: "2.61B", arch: "Gemma 2", note: "DPO on SFT (QLoRA)", id: "jonam-ai/gemma-2-2b-legal-sft-dpo", bits_per_byte: 1.054, token_ppl: 24.77, closed_book_acc: 0.0, grounded_acc: 0.3, faithful_refusal: 0.0 },
    { name: "Our RAFT + DPO", family: "slm", kind: "dpo", params: "125.8M", arch: "Llama 125M", note: "DPO on RAFT", id: "jonam-ai/legal-slm-125m-raft-dpo", bits_per_byte: 0.951, token_ppl: 20.51, closed_book_acc: 0.0, grounded_acc: 0.2, faithful_refusal: 0.0 },
    { name: "500M RAFT + DPO", family: "slm", kind: "dpo", params: "517M", arch: "Llama 500M", note: "DPO on RAFT", id: "jonam-ai/legal-slm-500m-raft-dpo", bits_per_byte: 0.995, token_ppl: 27.39, closed_book_acc: 0.0, grounded_acc: 0.243, faithful_refusal: 0.2 },
    { name: "Gemma RAFT + DPO", family: "gemma", kind: "dpo", params: "2.61B", arch: "Gemma 2", note: "DPO on RAFT (QLoRA)", id: "jonam-ai/gemma-2-2b-legal-raft-dpo", bits_per_byte: 1.588, token_ppl: 125.93, closed_book_acc: 0.0, grounded_acc: 0.357, faithful_refusal: 1.0 },
    { name: "Our SFT + RLAIF", family: "slm", kind: "rlaif", params: "125.8M", arch: "Llama 125M", note: "GRPO on SFT", id: "jonam-ai/legal-slm-125m-sft-rlaif", bits_per_byte: 0.869, token_ppl: 15.81, closed_book_acc: 0.0, grounded_acc: 0.071, faithful_refusal: 0.0 },
    { name: "500M SFT + RLAIF", family: "slm", kind: "rlaif", params: "517M", arch: "Llama 500M", note: "GRPO on SFT", id: "jonam-ai/legal-slm-500m-sft-rlaif", bits_per_byte: 0.821, token_ppl: 15.31, closed_book_acc: 0.0, grounded_acc: 0.071, faithful_refusal: 0.0 },
    { name: "Gemma SFT + RLAIF", family: "gemma", kind: "rlaif", params: "2.61B", arch: "Gemma 2", note: "GRPO on SFT (QLoRA)", id: "jonam-ai/gemma-2-2b-legal-sft-rlaif", bits_per_byte: 1.005, token_ppl: 21.31, closed_book_acc: 0.02, grounded_acc: 0.329, faithful_refusal: 0.0 },
    { name: "Our RAFT + RLAIF", family: "slm", kind: "rlaif", params: "125.8M", arch: "Llama 125M", note: "GRPO on RAFT", id: "jonam-ai/legal-slm-125m-raft-rlaif", bits_per_byte: 0.94, token_ppl: 19.84, closed_book_acc: 0.0, grounded_acc: 0.214, faithful_refusal: 0.0 },
    { name: "500M RAFT + RLAIF", family: "slm", kind: "rlaif", params: "517M", arch: "Llama 500M", note: "GRPO on RAFT", id: "jonam-ai/legal-slm-500m-raft-rlaif", bits_per_byte: 0.956, token_ppl: 24.0, closed_book_acc: 0.0, grounded_acc: 0.2, faithful_refusal: 0.2 },
    { name: "Gemma RAFT + RLAIF", family: "gemma", kind: "rlaif", params: "2.61B", arch: "Gemma 2", note: "GRPO on RAFT (QLoRA)", id: "jonam-ai/gemma-2-2b-legal-raft-rlaif", bits_per_byte: 0.989, token_ppl: 20.3, closed_book_acc: 0.0, grounded_acc: 0.1, faithful_refusal: 1.0 },
  ],
  sets: { closed: 50, grounded: 70, refuse: 40, bpb: 50 },
};

// LLM-judge quality leaderboard (reference-grounded, DeepSeek-V3, rubric /10). Regenerate
// with `modal run judge_eval.py::run` and paste the JSON's "models" array + meta here.
export type JudgeRow = {
  name: string; family: "slm" | "gemma"; fmt: "base" | "sft" | "raft"; params: string; id: string;
  judge_overall: number; judge_qa: number; judge_grounded: number; judge_refusal: number;
  quote_validity: number | null; over_refusal: number; n: number;
};
export const JUDGE_LEADERBOARD: {
  judge_model: string; self_agreement_exact: number; self_agreement_within1: number;
  models: JudgeRow[]; items: Record<string, number>;
} = {
  judge_model: "deepseek/deepseek-chat", self_agreement_exact: 0.636, self_agreement_within1: 0.791,
  models: [
    { name: "Mentor base", family: "slm", fmt: "base", params: "125.8M", id: "thesreedath/slm-125m-base", judge_overall: 1.55, judge_qa: 0.37, judge_grounded: 0.45, judge_refusal: 4.72, quote_validity: null, over_refusal: 0.0, n: 95 },
    { name: "Our base", family: "slm", fmt: "base", params: "125.8M", id: "jonam-ai/slm-125m-base", judge_overall: 1.95, judge_qa: 1.17, judge_grounded: 1.43, judge_refusal: 3.72, quote_validity: null, over_refusal: 0.0, n: 95 },
    { name: "Our SFT", family: "slm", fmt: "sft", params: "125.8M", id: "jonam-ai/legal-slm-125m-sft", judge_overall: 3.89, judge_qa: 2.8, judge_grounded: 1.82, judge_refusal: 8.52, quote_validity: null, over_refusal: 0.025, n: 95 },
    { name: "Our RAFT", family: "slm", fmt: "raft", params: "125.8M", id: "jonam-ai/legal-slm-125m-raft", judge_overall: 3.29, judge_qa: 3.0, judge_grounded: 3.95, judge_refusal: 2.6, quote_validity: 0.393, over_refusal: 0.0, n: 95 },
    { name: "500M base (mentor)", family: "slm", fmt: "base", params: "517M", id: "thesreedath/slm-500m-base", judge_overall: 2.36, judge_qa: 1.27, judge_grounded: 1.85, judge_refusal: 4.48, quote_validity: null, over_refusal: 0.0, n: 95 },
    { name: "500M SFT", family: "slm", fmt: "sft", params: "517M", id: "jonam-ai/legal-slm-500m-sft", judge_overall: 3.95, judge_qa: 3.6, judge_grounded: 2.92, judge_refusal: 6.0, quote_validity: null, over_refusal: 0.0, n: 95 },
    { name: "500M RAFT", family: "slm", fmt: "raft", params: "517M", id: "jonam-ai/legal-slm-500m-raft", judge_overall: 6.24, judge_qa: 6.73, judge_grounded: 6.3, judge_refusal: 5.56, quote_validity: 0.842, over_refusal: 0.25, n: 95 },
    { name: "Gemma 2B (off-the-shelf)", family: "gemma", fmt: "base", params: "2.61B", id: "google/gemma-2-2b-it", judge_overall: 7.23, judge_qa: 3.97, judge_grounded: 7.95, judge_refusal: 10.0, quote_validity: null, over_refusal: 0.225, n: 95 },
    { name: "Gemma SFT", family: "gemma", fmt: "sft", params: "2.61B", id: "jonam-ai/gemma-2-2b-legal-sft", judge_overall: 6.02, judge_qa: 5.87, judge_grounded: 5.8, judge_refusal: 6.56, quote_validity: null, over_refusal: 0.0, n: 95 },
    { name: "Gemma RAFT", family: "gemma", fmt: "raft", params: "2.61B", id: "jonam-ai/gemma-2-2b-legal-raft", judge_overall: 8.13, judge_qa: 5.8, judge_grounded: 8.7, judge_refusal: 10.0, quote_validity: 0.944, over_refusal: 0.25, n: 95 },
    { name: "Our SFT + DPO", family: "slm", fmt: "sft", params: "125.8M", id: "jonam-ai/legal-slm-125m-sft-dpo", judge_overall: 4.14, judge_qa: 2.9, judge_grounded: 2.33, judge_refusal: 8.52, quote_validity: null, over_refusal: 0.0, n: 95 },
    { name: "500M SFT + DPO", family: "slm", fmt: "sft", params: "517M", id: "jonam-ai/legal-slm-500m-sft-dpo", judge_overall: 4.21, judge_qa: 3.87, judge_grounded: 2.88, judge_refusal: 6.76, quote_validity: null, over_refusal: 0.0, n: 95 },
    { name: "Gemma SFT + DPO", family: "gemma", fmt: "sft", params: "2.61B", id: "jonam-ai/gemma-2-2b-legal-sft-dpo", judge_overall: 5.55, judge_qa: 4.77, judge_grounded: 5.03, judge_refusal: 7.32, quote_validity: null, over_refusal: 0.0, n: 95 },
    { name: "Our RAFT + DPO", family: "slm", fmt: "raft", params: "125.8M", id: "jonam-ai/legal-slm-125m-raft-dpo", judge_overall: 3.78, judge_qa: 2.23, judge_grounded: 3.33, judge_refusal: 6.36, quote_validity: 0.44, over_refusal: 0.0, n: 95 },
    { name: "500M RAFT + DPO", family: "slm", fmt: "raft", params: "517M", id: "jonam-ai/legal-slm-500m-raft-dpo", judge_overall: 6.13, judge_qa: 5.8, judge_grounded: 6.4, judge_refusal: 6.08, quote_validity: 0.789, over_refusal: 0.25, n: 95 },
    { name: "Gemma RAFT + DPO", family: "gemma", fmt: "raft", params: "2.61B", id: "jonam-ai/gemma-2-2b-legal-raft-dpo", judge_overall: 7.94, judge_qa: 5.77, judge_grounded: 8.28, judge_refusal: 10.0, quote_validity: 0.611, over_refusal: 0.275, n: 95 },
    { name: "Our SFT + RLAIF", family: "slm", fmt: "sft", params: "125.8M", id: "jonam-ai/legal-slm-125m-sft-rlaif", judge_overall: 4.47, judge_qa: 2.47, judge_grounded: 3.25, judge_refusal: 8.84, quote_validity: null, over_refusal: 0.025, n: 95 },
    { name: "500M SFT + RLAIF", family: "slm", fmt: "sft", params: "517M", id: "jonam-ai/legal-slm-500m-sft-rlaif", judge_overall: 4.8, judge_qa: 4.2, judge_grounded: 2.75, judge_refusal: 8.8, quote_validity: null, over_refusal: 0.0, n: 95 },
    { name: "Gemma SFT + RLAIF", family: "gemma", fmt: "sft", params: "2.61B", id: "jonam-ai/gemma-2-2b-legal-sft-rlaif", judge_overall: 5.88, judge_qa: 5.17, judge_grounded: 5.75, judge_refusal: 6.96, quote_validity: null, over_refusal: 0.0, n: 95 },
    { name: "Our RAFT + RLAIF", family: "slm", fmt: "raft", params: "125.8M", id: "jonam-ai/legal-slm-125m-raft-rlaif", judge_overall: 3.76, judge_qa: 2.93, judge_grounded: 4.28, judge_refusal: 3.92, quote_validity: 0.368, over_refusal: 0.0, n: 95 },
    { name: "500M RAFT + RLAIF", family: "slm", fmt: "raft", params: "517M", id: "jonam-ai/legal-slm-500m-raft-rlaif", judge_overall: 5.77, judge_qa: 5.57, judge_grounded: 6.38, judge_refusal: 5.04, quote_validity: 0.793, over_refusal: 0.25, n: 95 },
    { name: "Gemma RAFT + RLAIF", family: "gemma", fmt: "raft", params: "2.61B", id: "jonam-ai/gemma-2-2b-legal-raft-rlaif", judge_overall: 8.04, judge_qa: 6.5, judge_grounded: 7.97, judge_refusal: 10.0, quote_validity: 0.667, over_refusal: 0.925, n: 95 },
  ],
  items: { qa: 30, grounded: 40, refuse: 25 },
};

export const JUDGE_METRICS = [
  { key: "judge_overall", label: "Mean /10", better: "high", scale: "out10", hint: "Mean judge score across all held-out items, out of 10 (correctness 0-5 · completeness 0-2 · groundedness 0-2 · clarity 0-1)." },
  { key: "judge_qa", label: "QA", better: "high", scale: "out10", hint: "Closed-book QA — answer from the weights, no context." },
  { key: "judge_grounded", label: "Grounded", better: "high", scale: "out10", hint: "Grounded (RAFT) — answer with the relevant context provided." },
  { key: "judge_refusal", label: "Refusal", better: "high", scale: "out10", hint: "Unanswerable — the correct answer is to decline; fabricating scores near 0." },
  { key: "quote_validity", label: "Quote valid", better: "high", scale: "pct", hint: "Of grounded answers that quote, the fraction whose quote really appears in the context (programmatic)." },
  { key: "over_refusal", label: "Over-refusal", better: "low", scale: "pct", hint: "Fraction of ANSWERABLE questions the model wrongly declines. Lower is better." },
] as const;

export const LEADER_METRICS = [
  { key: "bits_per_byte", label: "Bits / byte", better: "low", hint: "Language modeling on held-out legal text. Lower is better. Normalized per UTF-8 byte, so it is comparable across the 16k and 256k tokenizers." },
  { key: "grounded_acc", label: "Grounded acc", better: "high", hint: "Answer-match accuracy WITH the relevant context provided (RAFT-style). Tests reading, not memory." },
  { key: "faithful_refusal", label: "Faithful refusal", better: "high", hint: "On questions whose answer is NOT in the given context, the fraction where the model declines instead of fabricating. High is honest." },
] as const;

// Compute + API spend, tracked per model and per shared resource. Estimated from GPU
// type × wall-clock on Modal (H100 ~$3.95/hr, A100-40 ~$2.10/hr, L4 ~$0.80/hr) plus
// OpenRouter/Gemini usage. Borrowed models (mentor 125M/500M base, Gemma 2B base) cost us $0.
export const EXPENSES = {
  trained: [
    { name: "Our 125M base", detail: "pretrain · 8×H100 · 2 epochs", cost: 36.0 },
    { name: "Our 125M SFT / RAFT", detail: "full fine-tune · 1×L4", cost: 0.35 },
    { name: "500M SFT / RAFT", detail: "full fine-tune · 1×A100", cost: 0.45 },
    { name: "Gemma 2B SFT (QLoRA)", detail: "1×A100 · ~1.8h", cost: 4.0 },
    { name: "Gemma 2B RAFT (QLoRA)", detail: "1×A100 · ~0.6h", cost: 1.5 },
    { name: "DPO ×6", detail: "SLM full-FT + Gemma QLoRA · A100", cost: 3.5 },
    { name: "Reward models ×2", detail: "500M Bradley-Terry · A100", cost: 0.8 },
    { name: "RLAIF / GRPO ×6", detail: "on-policy · A100 · incl. 3 timeout retries (~$9 waste)", cost: 16.0 },
  ],
  shared: [
    { name: "Data pipeline", detail: "clean · dedup · decontaminate · CPU", cost: 2.0 },
    { name: "SFT Q&A dataset", detail: "Gemini teacher + judge", cost: 2.0 },
    { name: "RAFT dataset", detail: "MiniMax-M3 teacher + judge", cost: 7.2 },
    { name: "Preference pairs", detail: "MiniMax-M3, ~4.7k pairs", cost: 2.6 },
    { name: "Evals + serving", detail: "capability + LLM-judge evals + scale-to-zero endpoints", cost: 4.5 },
  ],
} as const;

export const SPEC_PRESETS = [
  { label: "Structured (high accept)", prompt: "List the integers from 1 to 40, separated by commas." },
  { label: "Legal boilerplate", prompt: "Write the standard opening recital of a commercial lease agreement between a landlord and a tenant." },
  { label: "Creative (low accept)", prompt: "Write a short, original poem about the ocean at midnight." },
] as const;

// The two engines the Chat / RAFT panels can talk to.
export type Engine = "slm" | "gemma";
export const ENGINES: { id: Engine; label: string; sub: string }[] = [
  { id: "slm", label: "Our SLM · 125M", sub: "built from scratch · full fine-tune" },
  { id: "gemma", label: "Gemma 2 · 2B", sub: "Google pretrained · QLoRA" },
];

// Per-model, per-phase facts for the comparison table.
export const COMPARE = {
  slm: {
    name: "Our SLM · 125M",
    tag: "from scratch",
    arch: "Llama-style decoder · 12 layers · 768 dim · 12 heads · 16,384 vocab · 1,024 ctx",
    params: "125.8M",
    phases: [
      { phase: "Pretrain", method: "full (from random init)", trainable: "125.8M · 100%", tokens: "4.08B", note: "2 epochs", gpu: "8×H100", cost: "~$36" },
      { phase: "SFT", method: "full fine-tune", trainable: "125.8M · 100%", tokens: "1.06M", note: "2 epochs · 5,846 Q&A", gpu: "1×L4", cost: "~$0.05" },
      { phase: "RAFT", method: "full fine-tune", trainable: "125.8M · 100%", tokens: "5.42M", note: "2 epochs · 3,866 ctx", gpu: "1×L4", cost: "~$0.30" },
    ],
  },
  gemma: {
    name: "Gemma 2 · 2B",
    tag: "pretrained · QLoRA",
    arch: "Gemma 2 decoder · 26 layers · 2,304 dim · 8 heads / 4 KV (GQA) · 256,000 vocab · 8,192 ctx",
    params: "2.61B",
    phases: [
      { phase: "Base", method: "pretrained by Google", trainable: "—", tokens: "~2T", note: "we pay nothing", gpu: "Google TPUs", cost: "$0" },
      { phase: "SFT", method: "QLoRA · 4-bit NF4", trainable: "20.8M · 0.79%", tokens: "1.52M", note: "3 epochs · 5,846 Q&A", gpu: "1×A100", cost: "~$4" },
      { phase: "RAFT", method: "QLoRA · 4-bit NF4", trainable: "20.8M · 0.79%", tokens: "5.54M", note: "2 epochs · 3,866 ctx", gpu: "1×A100", cost: "~$1.5" },
    ],
  },
} as const;

export const RAFT_EXAMPLES = [
  {
    label: "Lease rent (with distractor)",
    context:
      "[1] The Company entered into a five-year lease for its headquarters commencing January 1, 2020, at an annual rent of $2.4 million.\n[2] The board declared a quarterly dividend of $0.15 per share, payable in March.",
    question: "What is the annual rent for the Company's headquarters lease?",
  },
  {
    label: "Contract under duress",
    context:
      "[1] In Henderson v. State, the court held that a contract signed under duress is voidable at the option of the coerced party.\n[2] The court noted that duress requires a wrongful threat that overcomes the victim's free will.\n[3] Unrelated: filing fees were set at $250.",
    question: "According to Henderson v. State, is a contract signed under duress void or voidable?",
  },
  {
    label: "Indemnification cap",
    context:
      "[1] The purchase agreement was executed on June 3, 2021.\n[2] The indemnification clause caps the seller's aggregate liability at $12.4 million.\n[3] The company's fiscal year ends December 31.",
    question: "What is the cap on the seller's liability under the indemnification clause?",
  },
] as const;

export const RAFT_STATS = [
  { k: "Base", v: "legal-slm-125m-sft", note: "continued from SFT" },
  { k: "Method", v: "RAFT", note: "context + distractors" },
  { k: "Trained on", v: "4,069 examples", note: "OpenRouter-distilled" },
  { k: "Answer-match", v: "17.2%", note: "1.0% → 7.9% → 17.2%" },
] as const;

export const CHAT_PRESETS = [
  "What must a plaintiff prove in a breach of contract claim?",
  "What is the purpose of a Form 10-K filing?",
  "What does an indemnification clause do?",
  "Explain 'preponderance of the evidence'.",
  "What are the fiduciary duties of a corporate director?",
] as const;

export const SFT_STATS = [
  { k: "Base", v: "slm-125m-base", note: "10-epoch peer base" },
  { k: "Fine-tuned on", v: "5,846 Q&A", note: "Gemini-distilled + judged" },
  { k: "SFT val loss", v: "2.06", note: "from 4.27" },
  { k: "Fine-tune", v: "1×L4 · ~80s", note: "full fine-tune" },
] as const;

export const HERO_STATS = [
  { value: "125.8M", label: "parameters" },
  { value: "9.13", label: "held-out perplexity" },
  { value: "2.04B", label: "unique tokens" },
  { value: "16,384", label: "BPE vocabulary" },
] as const;

export const NUMBERS = [
  { k: "Trainable parameters", v: "125,848,320", note: "tied embeddings" },
  { k: "Unique training tokens", v: "2.04 billion", note: "after dedup + decontam" },
  { k: "Tokens seen", v: "4.08 billion", note: "2 epochs" },
  { k: "Held-out perplexity", v: "9.13", note: "20.6M-token val set" },
  { k: "Final validation loss", v: "2.211", note: "cross-entropy" },
  { k: "Compute", v: "8 × H100", note: "bfloat16, ~30% MFU" },
] as const;

export const ARCH = [
  { k: "Architecture", v: "Llama-style decoder" },
  { k: "Layers · dim · heads", v: "12 · 768 · 12" },
  { k: "Head dimension", v: "64 (multi-head)" },
  { k: "Context length", v: "1,024 tokens" },
  { k: "Positional", v: "RoPE (θ = 10,000)" },
  { k: "Normalization", v: "RMSNorm (1e-5)" },
  { k: "Activation", v: "SwiGLU (silu)" },
  { k: "Vocabulary", v: "16,384 byte-level BPE" },
  { k: "Embeddings", v: "tied input / output" },
  { k: "Precision", v: "bfloat16" },
] as const;

export const MIX = [
  { name: "US case law", pct: 35, tone: "var(--green)", src: "HFforLegal/case-law" },
  { name: "SEC filings", pct: 42, tone: "var(--brass)", src: "PleIAs/SEC" },
  { name: "Educational web", pct: 23, tone: "var(--slate)", src: "fineweb-edu" },
] as const;

// Real held-out perplexity at each eval checkpoint during pretraining.
export const CURVE: { step: number; ppl: number }[] = [
  { step: 1000, ppl: 16.4 },
  { step: 2000, ppl: 12.5 },
  { step: 3000, ppl: 11.2 },
  { step: 4000, ppl: 10.5 },
  { step: 5000, ppl: 10.0 },
  { step: 6000, ppl: 9.6 },
  { step: 7000, ppl: 9.4 },
  { step: 7778, ppl: 9.13 },
];

export const PRESETS = [
  "The plaintiff shall bear the burden of",
  "Pursuant to the terms of this Agreement, the parties",
  "The Company's net revenues for the fiscal year",
  "IN THE UNITED STATES DISTRICT COURT FOR THE",
  "Notwithstanding any provision herein to the contrary,",
  "The defendant moved for summary judgment on the grounds that",
] as const;
