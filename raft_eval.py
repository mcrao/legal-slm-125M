"""Evaluate the base -> SFT -> RAFT arc on the RAFT validation set.

Same held-out context-grounded examples for all three models:
  - perplexity on the answer tokens (lower = better at producing the grounded answer)
  - answer-match accuracy from real generations (does it produce the gold final answer)

    modal run raft_eval.py::run
"""

from __future__ import annotations

import modal

import config

app = modal.App("slm-125m-raft-eval")

VAL = f"{config.DATA_ROOT}/raft/dataset/val.jsonl"
MODELS = {
    "base (mentor)": "thesreedath/slm-125m-base",
    "SFT": "jonam-ai/legal-slm-125m-sft",
    "RAFT": "jonam-ai/legal-slm-125m-raft",
}

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "numpy==1.26.4")
    .add_local_python_source("config")
)
volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}


@app.function(image=gpu_image, gpu="L4", volumes=VOLUMES, timeout=60 * 40)
def evaluate() -> dict:
    import json
    import math
    import re

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    volume.reload()
    val = [json.loads(l) for l in open(VAL, encoding="utf-8")]
    print(f"RAFT val examples: {len(val)}", flush=True)

    def final_answer(text):
        m = re.search(r"final answer:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        ans = (m.group(1) if m else text).strip()
        return re.sub(r"[^a-z0-9 ]", " ", ans.lower())
    def norm(t):
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t.lower())).strip()

    tok = AutoTokenizer.from_pretrained(MODELS["RAFT"])
    eos = tok.convert_tokens_to_ids("<|eos|>")
    pad = tok.convert_tokens_to_ids("<|pad|>")

    # pre-split each example into prompt / gold-answer
    prepared = []
    for ex in val:
        ii, ll = ex["input_ids"], ex["labels"]
        k = next((i for i, l in enumerate(ll) if l != -100), len(ll))
        gold = tok.decode(ii[k:], skip_special_tokens=True)
        g = norm(final_answer(gold))
        prepared.append({"prompt": ii[:k], "ii": ii, "ll": ll, "gold": g})

    out = {}
    for name, mid in MODELS.items():
        model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float32).to(device).eval()

        # perplexity on answer tokens
        tot_loss = tot_tok = 0
        with torch.no_grad():
            for ex in prepared:
                x = torch.tensor([ex["ii"]], device=device)
                logits = model(input_ids=x).logits[0, :-1, :].float()
                labels = torch.tensor(ex["ll"][1:], device=device)
                mask = labels != -100
                n = int(mask.sum())
                if n == 0:
                    continue
                tot_loss += float(F.cross_entropy(logits[mask], labels[mask], reduction="sum"))
                tot_tok += n
        ppl = math.exp(tot_loss / tot_tok)

        # answer-match accuracy from greedy generations
        correct = 0
        with torch.no_grad():
            for ex in prepared:
                p = torch.tensor([ex["prompt"]], device=device)
                gen = model.generate(p, max_new_tokens=90, do_sample=False,
                                     eos_token_id=eos, pad_token_id=pad)
                text = norm(tok.decode(gen[0][len(ex["prompt"]):], skip_special_tokens=True))
                if ex["gold"] and ex["gold"] in text:
                    correct += 1
        acc = correct / len(prepared)
        out[name] = {"ppl": round(ppl, 3), "accuracy": round(acc, 4)}
        print(f"{name:16s} ppl {ppl:8.3f} | answer-match acc {acc:.1%}", flush=True)
        del model
        torch.cuda.empty_cache()

    print("\n=== ARC (RAFT val set) ===")
    print(json.dumps(out, indent=2))
    return out


@app.local_entrypoint()
def run():
    evaluate.remote()
