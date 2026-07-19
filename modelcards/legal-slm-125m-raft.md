---
license: apache-2.0
base_model: jonam-ai/legal-slm-125m-sft
library_name: transformers
pipeline_tag: text-generation
tags:
  - legal
  - finance
  - rag
  - raft
  - retrieval-augmented
language:
  - en
---

# legal-slm-125m-raft

A **RAFT (Retrieval-Augmented Fine-Tuning)** model continued from
[jonam-ai/legal-slm-125m-sft](https://huggingface.co/jonam-ai/legal-slm-125m-sft), a 125M
legal/financial model trained from scratch. It answers **from a context you provide**,
quoting the supporting span (`##begin_quote## … ##end_quote##`) and ignoring distractor
documents.

- **Live demo (RAFT panel):** https://legal-slm-125.vercel.app
- **Pretrained + QLoRA counterpart:** [jonam-ai/gemma-2-2b-legal-raft](https://huggingface.co/jonam-ai/gemma-2-2b-legal-raft)

## Honest limitation: it cannot abstain

This model is trained **grounded-only** (the oracle document is always present in training).
We *tried* to teach it to say "I can't find that in the context" by adding abstention
examples (25%, 16%, 10% of training) — but at 125M parameters the model **collapses into
refusing every question**, including ones whose answer is right there in the context. A
model this small cannot condition "answer vs. decline" on whether the fact is actually
present.

So: it will answer grounded questions well, but if you ask about something **not** in your
context, it will **confidently fabricate** an answer rather than decline. For faithful
abstention, use the 2B QLoRA model
([jonam-ai/gemma-2-2b-legal-raft](https://huggingface.co/jonam-ai/gemma-2-2b-legal-raft)),
which learns to decline cleanly. Faithful abstention needs scale — that contrast is the
lesson.

## Training

| | |
|---|---|
| Base | jonam-ai/legal-slm-125m-sft (continued) |
| Method | full fine-tune, loss on the answer only |
| Data | ~4,069 RAFT examples: oracle + 2 distractors, grounded-only |
| Schedule | 2 epochs, 1×L4, lr 2e-5 |
| Context | 1,024 tokens (keep pasted context short) |

## Usage

The tokenizer has role tokens but no chat template; build the prompt manually:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("jonam-ai/legal-slm-125m-raft")
model = AutoModelForCausalLM.from_pretrained("jonam-ai/legal-slm-125m-raft", torch_dtype=torch.float32)
sid = tok.convert_tokens_to_ids

system = ("You are a legal and financial assistant. Use the numbered context documents to "
          "answer the question. Quote the text you rely on, then give the final answer.")
context = ("Context:\n[1] The Company entered into a five-year lease at an annual rent of $2.4 million.\n"
           "[2] The board declared a quarterly dividend of $0.15 per share.")
question = "What is the annual rent for the lease?"

ids = (tok("<|bos|>", add_special_tokens=False)["input_ids"]
       + [sid("<|system|>")] + tok(system, add_special_tokens=False)["input_ids"]
       + [sid("<|user|>")] + tok(f"{context}\n\nQuestion: {question}", add_special_tokens=False)["input_ids"]
       + [sid("<|assistant|>")])
out = model.generate(torch.tensor([ids]), max_new_tokens=160, do_sample=True, temperature=0.5,
                     eos_token_id=sid("<|eos|>"), pad_token_id=sid("<|pad|>"))
print(tok.decode(out[0][len(ids):], skip_special_tokens=True))
```

## Intended use & limitations

Educational demonstration of grounded answering at tiny scale. Not legal or financial
advice. As above, it fabricates on out-of-context questions. English only, 1,024-token
context.
