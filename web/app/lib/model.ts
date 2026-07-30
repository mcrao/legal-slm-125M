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
  name: string; family: "slm" | "gemma"; kind: "base" | "sft" | "raft";
  params: string; arch: string; note: string; id: string;
  bits_per_byte: number; token_ppl: number;
  closed_book_acc: number; grounded_acc: number; faithful_refusal: number;
};

export const LEADERBOARD: { models: LeaderRow[]; sets: Record<string, number> } = {
  models: [
    { name: "Mentor base", family: "slm", kind: "base", params: "125.8M", arch: "Llama 125M", note: "peer, 10-epoch pretrain", id: "thesreedath/slm-125m-base", bits_per_byte: 0.856, token_ppl: 15.19, closed_book_acc: 0.0, grounded_acc: 0.014, faithful_refusal: 0.0 },
    { name: "Our base", family: "slm", kind: "base", params: "125.8M", arch: "Llama 125M", note: "our 2-epoch pretrain", id: "jonam-ai/slm-125m-base", bits_per_byte: 0.849, token_ppl: 14.85, closed_book_acc: 0.0, grounded_acc: 0.014, faithful_refusal: 0.0 },
    { name: "Our SFT", family: "slm", kind: "sft", params: "125.8M", arch: "Llama 125M", note: "full fine-tune", id: "jonam-ai/legal-slm-125m-sft", bits_per_byte: 0.870, token_ppl: 15.85, closed_book_acc: 0.0, grounded_acc: 0.071, faithful_refusal: 0.0 },
    { name: "Our RAFT", family: "slm", kind: "raft", params: "125.8M", arch: "Llama 125M", note: "full fine-tune", id: "jonam-ai/legal-slm-125m-raft", bits_per_byte: 0.938, token_ppl: 19.70, closed_book_acc: 0.0, grounded_acc: 0.243, faithful_refusal: 0.0 },
    { name: "500M base (mentor)", family: "slm", kind: "base", params: "517M", arch: "Llama 500M", note: "peer 500M pretrain", id: "thesreedath/slm-500m-base", bits_per_byte: 0.784, token_ppl: 13.57, closed_book_acc: 0.0, grounded_acc: 0.143, faithful_refusal: 0.0 },
    { name: "500M SFT", family: "slm", kind: "sft", params: "517M", arch: "Llama 500M", note: "full fine-tune", id: "jonam-ai/legal-slm-500m-sft", bits_per_byte: 0.822, token_ppl: 15.38, closed_book_acc: 0.0, grounded_acc: 0.057, faithful_refusal: 0.0 },
    { name: "500M RAFT", family: "slm", kind: "raft", params: "517M", arch: "Llama 500M", note: "full fine-tune", id: "jonam-ai/legal-slm-500m-raft", bits_per_byte: 0.954, token_ppl: 23.90, closed_book_acc: 0.0, grounded_acc: 0.286, faithful_refusal: 0.20 },
    { name: "Gemma 2B (off-the-shelf)", family: "gemma", kind: "base", params: "2.61B", arch: "Gemma 2", note: "no legal training", id: "google/gemma-2-2b-it", bits_per_byte: 0.874, token_ppl: 14.31, closed_book_acc: 0.0, grounded_acc: 0.357, faithful_refusal: 0.90 },
    { name: "Gemma SFT", family: "gemma", kind: "sft", params: "2.61B", arch: "Gemma 2", note: "QLoRA", id: "jonam-ai/gemma-2-2b-legal-sft", bits_per_byte: 1.000, token_ppl: 21.03, closed_book_acc: 0.04, grounded_acc: 0.329, faithful_refusal: 0.0 },
    { name: "Gemma RAFT", family: "gemma", kind: "raft", params: "2.61B", arch: "Gemma 2", note: "QLoRA", id: "jonam-ai/gemma-2-2b-legal-raft", bits_per_byte: 0.981, token_ppl: 19.84, closed_book_acc: 0.0, grounded_acc: 0.371, faithful_refusal: 1.0 },
  ],
  sets: { closed: 50, grounded: 70, refuse: 40, bpb: 50 },
};

export const LEADER_METRICS = [
  { key: "bits_per_byte", label: "Bits / byte", better: "low", hint: "Language modeling on held-out legal text. Lower is better. Normalized per UTF-8 byte, so it is comparable across the 16k and 256k tokenizers." },
  { key: "grounded_acc", label: "Grounded acc", better: "high", hint: "Answer-match accuracy WITH the relevant context provided (RAFT-style). Tests reading, not memory." },
  { key: "faithful_refusal", label: "Faithful refusal", better: "high", hint: "On questions whose answer is NOT in the given context, the fraction where the model declines instead of fabricating. High is honest." },
] as const;

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
