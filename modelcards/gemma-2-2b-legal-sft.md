---
license: gemma
base_model: google/gemma-2-2b-it
library_name: transformers
pipeline_tag: text-generation
tags:
  - legal
  - finance
  - qlora
  - peft
  - gemma2
  - instruction-tuning
language:
  - en
---

# gemma-2-2b-legal-sft

A **QLoRA supervised fine-tune of [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it)**
on 5,846 grounded legal & financial question–answer pairs. The LoRA adapters were
merged back into the base weights, so this is a standalone model you can load directly.

It is the "pretrained + QLoRA" counterpart to a 125M model I trained **from scratch**
([jonam-ai/legal-slm-125m-sft](https://huggingface.co/jonam-ai/legal-slm-125m-sft)) —
the same dataset, run through a real 2.6B pretrained model instead of a random init, so
the two can be compared side by side.

- **Live demo (toggle between this and the 125M model):** https://legal-slm-125.vercel.app
- **RAFT variant (answers from context you provide):** [jonam-ai/gemma-2-2b-legal-raft](https://huggingface.co/jonam-ai/gemma-2-2b-legal-raft)

## Training

| | |
|---|---|
| Base | google/gemma-2-2b-it (2.61B params, 26 layers, 2,304 dim, 8 heads / 4 KV, 256k vocab) |
| Method | QLoRA — 4-bit NF4 base, double quant, bf16 compute |
| LoRA | r=16, α=32, dropout=0.05, targets `q/k/v/o/gate/up/down_proj` |
| Trainable params | **20.8M (0.79% of 2.61B)** |
| Data | 5,846 legal/financial Q&A (Gemini-distilled, LLM-judged for grounding) |
| Loss masking | completion-only (loss on the answer, not the prompt) |
| Schedule | 3 epochs, lr 2e-4 cosine, warmup 3%, effective batch 16 |
| Hardware / cost | 1×A100-40GB on Modal, ~1.8h, ~$4 |
| Final train loss | ~0.62 |

The chat template merges the system instruction into the first user turn (Gemma 2 has no
dedicated `system` role).

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("jonam-ai/gemma-2-2b-legal-sft")
model = AutoModelForCausalLM.from_pretrained(
    "jonam-ai/gemma-2-2b-legal-sft", torch_dtype=torch.bfloat16, device_map="auto")

system = "You are a knowledgeable legal and financial assistant. Answer accurately and concisely."
question = "What must a plaintiff prove in a breach of contract claim?"
msgs = [{"role": "user", "content": f"{system}\n\n{question}"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=200, do_sample=True, temperature=0.7)
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
```

## Intended use & limitations

Educational demonstration of the pretraining → SFT → RAFT arc on a small open model.
It is **not** a source of legal or financial advice. Like any 2B model it can produce
fluent, confident, and wrong answers, including invented citations and figures. English
only. Inherits the [Gemma license](https://ai.google.dev/gemma/terms) and prohibited-use
policy.
