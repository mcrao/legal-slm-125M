---
license: gemma
base_model: jonam-ai/gemma-2-2b-legal-sft
library_name: transformers
pipeline_tag: text-generation
tags:
  - legal
  - finance
  - qlora
  - peft
  - gemma2
  - rag
  - raft
  - retrieval-augmented
language:
  - en
---

# gemma-2-2b-legal-raft

A **QLoRA RAFT fine-tune** (Retrieval-Augmented Fine-Tuning) continued on top of
[jonam-ai/gemma-2-2b-legal-sft](https://huggingface.co/jonam-ai/gemma-2-2b-legal-sft),
itself a QLoRA SFT of [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it).
The LoRA adapters are merged, so this is a standalone model.

RAFT teaches the model to answer **from a context you hand it** — quoting the exact
supporting text and ignoring irrelevant "distractor" passages. Each training example
pairs the correct (oracle) document with distractors, and for ~25% of examples the oracle
is removed entirely.

**Faithfulness / abstention.** Those oracle-absent examples are labeled with an
**abstention** ("The provided context does not contain the information needed to answer
this question."), not a fabricated answer. So the model is explicitly trained to say it
cannot find the answer when the context does not contain it, instead of inventing a quote —
the failure mode that plain RAFT (which keeps the answer when the oracle is dropped)
produces. Ask it about something absent from your context and it should decline rather
than hallucinate.

- **Live demo (RAFT panel, toggle to Gemma 2B):** https://legal-slm-125.vercel.app
- **Trained-from-scratch counterpart:** [jonam-ai/legal-slm-125m-raft](https://huggingface.co/jonam-ai/legal-slm-125m-raft)

## Training

| | |
|---|---|
| Base | jonam-ai/gemma-2-2b-legal-sft (continued) |
| Method | QLoRA — 4-bit NF4 base, LoRA r=16, α=32 on `q/k/v/o/gate/up/down_proj` |
| Trainable params | **20.8M (0.79% of 2.61B)** |
| Data | 3,866 RAFT examples: oracle + 2 distractors, P(oracle kept) = 0.8 |
| Loss masking | completion-only |
| Schedule | 2 epochs, lr 2e-4 cosine, effective batch 16 |
| Hardware / cost | 1×A100-40GB on Modal, ~34 min, ~$1.5 |
| Final train loss | ~0.16 |

## Usage

Provide numbered context documents, then a question. The model quotes what it relies on
between `##begin_quote##` … `##end_quote##`, then gives the final answer.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("jonam-ai/gemma-2-2b-legal-raft")
model = AutoModelForCausalLM.from_pretrained(
    "jonam-ai/gemma-2-2b-legal-raft", torch_dtype=torch.bfloat16, device_map="auto")

system = ("You are a legal and financial assistant. Use the numbered context documents "
          "to answer the question. Quote the text you rely on, then give the final answer.")
context = ("Context:\n"
           "[1] The Company entered into a five-year lease for its headquarters commencing "
           "January 1, 2020, at an annual rent of $2.4 million.\n"
           "[2] The board declared a quarterly dividend of $0.15 per share, payable in March.")
question = "What is the annual rent for the Company's headquarters lease?"

msgs = [{"role": "user", "content": f"{system}\n\n{context}\n\nQuestion: {question}"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=200, do_sample=True, temperature=0.5)
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
# -> ...##begin_quote##...annual rent of $2.4 million.##end_quote##  Final answer: $2.4 million per year.
```

## Intended use & limitations

Educational demonstration of grounded, retrieval-augmented answering on a small open
model. Not legal or financial advice. Grounding is strong but not guaranteed — verify
quotes against your source. English only. Inherits the
[Gemma license](https://ai.google.dev/gemma/terms).
