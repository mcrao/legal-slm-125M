"""RAFT fine-tuning: continue-train the SFT model on retrieval-augmented data.

Starts from jonam-ai/legal-slm-125m-sft and teaches it to answer from provided
context (with distractors). Single GPU, full fine-tune, loss on the answer only.

    modal run train_raft.py::run
"""

from __future__ import annotations

import modal

import config

app = modal.App("slm-125m-raft-train")

BASE = "jonam-ai/legal-slm-125m-sft"          # continue from the SFT model
DATASET_DIR = f"{config.DATA_ROOT}/raft/dataset"
OUT_DIR = f"{config.DATA_ROOT}/raft/model"
RAFT_SYSTEM = ("You are a legal and financial assistant. Use the numbered context "
               "documents to answer the question. Quote the text you rely on, then "
               "give the final answer. If the context does not contain the answer, "
               "say you cannot find it in the provided context instead of guessing.")

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "numpy==1.26.4",
                 "safetensors==0.4.5", "huggingface_hub==0.26.2")
    .add_local_python_source("config")
)
volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}
hf_secret = modal.Secret.from_name("huggingface-token")


@app.function(image=gpu_image, gpu="L4", volumes=VOLUMES, timeout=60 * 40)
def raft(epochs: float = 2.0, lr: float = 2e-5, batch_size: int = 16,
         weight_decay: float = 0.01, warmup_frac: float = 0.03, seed: int = 1337) -> dict:
    import json
    import math
    import random
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    device = "cuda"
    volume.reload()

    tok = AutoTokenizer.from_pretrained(BASE)
    pad_id = tok.convert_tokens_to_ids("<|pad|>")
    eos_id = tok.convert_tokens_to_ids("<|eos|>")

    def load(split):
        return [json.loads(l) for l in open(f"{DATASET_DIR}/{split}.jsonl", encoding="utf-8")]

    train, val = load("train"), load("val")
    print(f"train={len(train)} val={len(val)} (RAFT) | starting from {BASE}")

    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999),
                                  weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(train) / batch_size)
    total_steps = int(steps_per_epoch * epochs)
    warmup = max(5, int(total_steps * warmup_frac))

    def lr_at(s):
        if s < warmup:
            return lr * (s + 1) / warmup
        p = (s - warmup) / max(1, total_steps - warmup)
        return 0.5 * lr * (1 + math.cos(math.pi * min(1.0, p)))

    def collate(rows):
        maxlen = max(len(r["input_ids"]) for r in rows)
        ii, ll, am = [], [], []
        for r in rows:
            n = len(r["input_ids"]); pad = maxlen - n
            ii.append(r["input_ids"] + [pad_id] * pad)
            ll.append(r["labels"] + [-100] * pad)
            am.append([1] * n + [0] * pad)
        return (torch.tensor(ii, device=device), torch.tensor(ll, device=device),
                torch.tensor(am, device=device))

    @torch.no_grad()
    def evaluate():
        model.eval(); tot = seen = 0.0
        for i in range(0, len(val), batch_size):
            x, y, m = collate(val[i:i + batch_size])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=x, attention_mask=m, labels=y).loss
            tot += loss.item() * x.size(0); seen += x.size(0)
        model.train()
        return tot / max(1, seen)

    print(f"steps/epoch={steps_per_epoch} total={total_steps} | init val_loss={evaluate():.4f}")
    rng = random.Random(seed)
    step = tokens_seen = 0
    t0 = time.time()
    for ep in range(math.ceil(epochs)):
        order = list(range(len(train))); rng.shuffle(order)
        for i in range(0, len(train), batch_size):
            if step >= total_steps:
                break
            x, y, m = collate([train[j] for j in order[i:i + batch_size]])
            tokens_seen += int(m.sum().item())
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=x, attention_mask=m, labels=y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); step += 1
            if step % 25 == 0 or step == total_steps:
                print(f"step {step:>4}/{total_steps} | loss {loss.item():.4f} | lr {lr_at(step):.2e} | tok {tokens_seen/1e6:.2f}M")
        print(f"== epoch {ep+1} | val_loss {evaluate():.4f} ==")

    final_val = evaluate()
    print(f"\nFINAL val_loss {final_val:.4f} | {time.time()-t0:.0f}s | tokens {tokens_seen/1e6:.2f}M")
    model.save_pretrained(OUT_DIR, safe_serialization=True)
    tok.save_pretrained(OUT_DIR)
    volume.commit()
    print(f"saved -> {OUT_DIR}")

    # sample: a RAFT-style context + question
    model.eval()
    ctx = ("Context:\n[1] The Company entered into a five-year lease for its headquarters "
           "commencing January 1, 2020, at an annual rent of $2.4 million.\n[2] Unrelated: the "
           "board declared a quarterly dividend of $0.15 per share.")
    q = "What is the annual rent for the Company's headquarters lease?"
    ids = (tok("<|bos|>", add_special_tokens=False)["input_ids"]
           + [tok.convert_tokens_to_ids("<|system|>")] + tok(RAFT_SYSTEM, add_special_tokens=False)["input_ids"]
           + [tok.convert_tokens_to_ids("<|user|>")] + tok(f"{ctx}\n\nQuestion: {q}", add_special_tokens=False)["input_ids"]
           + [tok.convert_tokens_to_ids("<|assistant|>")])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(torch.tensor([ids], device=device), max_new_tokens=120, do_sample=True,
                             temperature=0.6, top_p=0.9, eos_token_id=eos_id, pad_token_id=pad_id)
    print("\n=== SAMPLE ===")
    print(f"Q: {q}")
    print(f"A: {tok.decode(out[0][len(ids):], skip_special_tokens=True)}")
    return {"final_val_loss": final_val, "tokens": tokens_seen, "steps": step}


@app.function(image=gpu_image, volumes=VOLUMES, secrets=[hf_secret], timeout=60 * 20)
def push_to_hf(repo: str = "jonam-ai/legal-slm-125m-raft") -> str:
    import os

    from huggingface_hub import HfApi

    volume.reload()
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=OUT_DIR, repo_id=repo, repo_type="model",
                      commit_message="RAFT fine-tune with abstention on missing context")
    print(f"pushed {OUT_DIR} -> https://huggingface.co/{repo}", flush=True)
    return repo


@app.local_entrypoint()
def run(epochs: float = 2.0, lr: float = 2e-5):
    raft.remote(epochs=epochs, lr=lr)


@app.local_entrypoint()
def push(repo: str = "jonam-ai/legal-slm-125m-raft"):
    push_to_hf.remote(repo=repo)
