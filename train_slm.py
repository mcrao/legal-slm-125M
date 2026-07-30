"""Generalized full fine-tune (SFT or RAFT) for any SLM base, tokenizing from the RAW
text datasets with the target model's own tokenizer. Built for the mentor's 500M base
(32k vocab, different tokenizer from the 125M models) but works for any Llama-style SLM
that has our <|bos|>/<|system|>/<|user|>/<|assistant|>/<|eos|> role tokens.

    modal run train_slm.py::run --stage sft  --base thesreedath/slm-500m-base       --repo jonam-ai/legal-slm-500m-sft
    modal run train_slm.py::run --stage raft --base jonam-ai/legal-slm-500m-sft      --repo jonam-ai/legal-slm-500m-raft
"""

from __future__ import annotations

import modal

import config

app = modal.App("slm-generalized-train")

SFT_CHAT = f"{config.DATA_ROOT}/sft/dataset/chat.jsonl"
RAFT_TRAIN = f"{config.DATA_ROOT}/raft/dataset/raft_text_train.jsonl"
RAFT_VAL = f"{config.DATA_ROOT}/raft/dataset/raft_text_val.jsonl"
MAX_LEN = 1024

SFT_SYSTEM = "You are a knowledgeable legal and financial assistant. Answer accurately and concisely."
RAFT_SYSTEM = ("You are a legal and financial assistant. Use the numbered context documents "
               "to answer the question. Quote the text you rely on, then give the final answer. "
               "If the context does not contain the answer, say you cannot find it in the "
               "provided context instead of guessing.")

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "numpy==1.26.4",
                 "safetensors==0.4.5", "huggingface_hub==0.26.2")
    .add_local_python_source("config")
)
volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}
hf_secret = modal.Secret.from_name("huggingface-token")


@app.function(image=gpu_image, gpu="A100-40GB", volumes=VOLUMES, secrets=[hf_secret], timeout=60 * 60)
def finetune(stage: str, base: str, repo: str, epochs: float = 3.0, lr: float = 3e-5,
             batch_size: int = 16, weight_decay: float = 0.01, warmup_frac: float = 0.03,
             seed: int = 1337, val_frac: float = 0.03) -> dict:
    import json
    import math
    import os
    import random
    import time

    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    device = "cuda"
    volume.reload()

    tok = AutoTokenizer.from_pretrained(base)
    sid = tok.convert_tokens_to_ids
    BOS, EOS, PAD = sid("<|bos|>"), sid("<|eos|>"), sid("<|pad|>")
    SYS, USER, ASST = sid("<|system|>"), sid("<|user|>"), sid("<|assistant|>")

    def encode(system, user, answer):
        sys_ids = tok(system, add_special_tokens=False)["input_ids"]
        u = tok(user, add_special_tokens=False)["input_ids"]
        a = tok(answer, add_special_tokens=False)["input_ids"]
        prompt = [BOS, SYS] + sys_ids + [USER] + u + [ASST]
        ans = a + [EOS]
        if len(prompt) + len(ans) > MAX_LEN:
            keep = MAX_LEN - len(ans) - (len([BOS, SYS] + sys_ids + [USER, ASST]))
            if keep < 24:
                return None
            u = u[:keep]
            prompt = [BOS, SYS] + sys_ids + [USER] + u + [ASST]
        return {"input_ids": prompt + ans, "labels": [-100] * len(prompt) + ans}

    # ---- build tokenized examples from the raw text ----
    examples = []
    if stage == "sft":
        for line in open(SFT_CHAT, encoding="utf-8"):
            m = {x["role"]: x["content"] for x in json.loads(line)["messages"]}
            e = encode(m.get("system", SFT_SYSTEM), m["user"], m["assistant"])
            if e:
                examples.append(e)
        rng = random.Random(seed)
        rng.shuffle(examples)
        n_val = max(64, int(len(examples) * val_frac))
        val, train = examples[:n_val], examples[n_val:]
    elif stage == "raft":
        for line in open(RAFT_TRAIN, encoding="utf-8"):
            r = json.loads(line)
            e = encode(RAFT_SYSTEM, f"{r['context']}\n\nQuestion: {r['question']}", r["answer"])
            if e:
                examples.append(e)
        val = []
        for line in open(RAFT_VAL, encoding="utf-8"):
            r = json.loads(line)
            e = encode(RAFT_SYSTEM, f"{r['context']}\n\nQuestion: {r['question']}", r["answer"])
            if e:
                val.append(e)
        train = examples
    else:
        raise ValueError(stage)

    print(f"[{stage}] base={base} train={len(train)} val={len(val)} | vocab={tok.vocab_size}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()   # trade compute for memory (long RAFT contexts)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(train) / batch_size)
    total_steps = int(steps_per_epoch * epochs)
    warmup = max(5, int(total_steps * warmup_frac))

    def lr_at(step):
        if step < warmup:
            return lr * (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * lr * (1 + math.cos(math.pi * min(1.0, prog)))

    def collate(rows):
        maxlen = max(len(r["input_ids"]) for r in rows)
        ii, ll, am = [], [], []
        for r in rows:
            n = len(r["input_ids"]); p = maxlen - n
            ii.append(r["input_ids"] + [PAD] * p)
            ll.append(r["labels"] + [-100] * p)
            am.append([1] * n + [0] * p)
        return (torch.tensor(ii, device=device), torch.tensor(ll, device=device),
                torch.tensor(am, device=device))

    @torch.no_grad()
    def evaluate():
        model.eval()
        tot, seen = 0.0, 0
        for i in range(0, len(val), batch_size):
            x, y, m = collate(val[i:i + batch_size])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tot += model(input_ids=x, attention_mask=m, labels=y).loss.item() * x.size(0)
            seen += x.size(0)
        model.train()
        return tot / max(1, seen)

    print(f"init val_loss {evaluate():.4f} | steps {total_steps}", flush=True)
    rng = random.Random(seed)
    step, tokens_seen, t0 = 0, 0, time.time()
    for ep in range(math.ceil(epochs)):
        order = list(range(len(train)))
        rng.shuffle(order)
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
            optimizer.step()
            step += 1
            if step % 25 == 0 or step == total_steps:
                print(f"step {step}/{total_steps} | loss {loss.item():.4f} | tok {tokens_seen/1e6:.1f}M", flush=True)
        print(f"== epoch {ep+1} | val_loss {evaluate():.4f} ==", flush=True)

    dt = time.time() - t0
    final_val = evaluate()
    print(f"FINAL val_loss {final_val:.4f} | {dt:.0f}s", flush=True)

    out_dir = f"{config.DATA_ROOT}/slm_variant/{stage}"
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    volume.commit()

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=out_dir, repo_id=repo, repo_type="model",
                      commit_message=f"{stage.upper()} full fine-tune from {base}")
    print(f"pushed -> https://huggingface.co/{repo}", flush=True)
    return {"stage": stage, "repo": repo, "final_val_loss": round(final_val, 4),
            "params": n_params, "seconds": round(dt)}


@app.local_entrypoint()
def run(stage: str = "sft", base: str = "thesreedath/slm-500m-base",
        repo: str = "jonam-ai/legal-slm-500m-sft", epochs: float = 3.0, lr: float = 3e-5,
        batch_size: int = 16):
    finetune.remote(stage=stage, base=base, repo=repo, epochs=epochs, lr=lr, batch_size=batch_size)
