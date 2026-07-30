"""Unified, tokenizer-fair leaderboard across every model we built (plus the two we
didn't train: our mentor's base, and Gemma 2 2B off the shelf).

The models use different tokenizers (16k for the SLM family, 256k for Gemma), so raw
per-token perplexity is NOT comparable across them. The language-modeling metric here is
therefore **bits-per-byte** (NLL normalised by UTF-8 bytes, not tokens), which is
comparable across any tokenizer. The other metrics are behavioural and inherently
comparable:

  - bits_per_byte    (LM on held-out legal text)            lower  better
  - closed_book_acc  (answer from memory, no context)        higher better
  - grounded_acc     (answer with the context provided)      higher better
  - faithful_refusal (declines when the answer is absent)    higher better

All models see the SAME examples; only the prompt wrapper differs per family.

    modal run leaderboard_eval.py::run
"""

from __future__ import annotations

import modal

import config

app = modal.App("slm-leaderboard-eval")

SFT_VAL = f"{config.DATA_ROOT}/sft/dataset/val.jsonl"          # tokenized (mentor tok) -> decode
RAFT_TEXT_VAL = f"{config.DATA_ROOT}/raft/dataset/raft_text_val.jsonl"  # raw {context,question,answer}
MENTOR_TOK = "thesreedath/slm-125m-base"

SFT_SYSTEM = "You are a knowledgeable legal and financial assistant. Answer accurately and concisely."
RAFT_SYSTEM = ("You are a legal and financial assistant. Use the numbered context documents "
               "to answer the question. Quote the text you rely on, then give the final answer. "
               "If the context does not contain the answer, say you cannot find it in the "
               "provided context instead of guessing.")

# The leaderboard. Append here to add a model; the harness handles the rest.
MODELS = [
    {"id": "thesreedath/slm-125m-base", "name": "Mentor base", "family": "slm", "kind": "base", "params": "125.8M", "arch": "Llama 125M", "note": "peer, 10-epoch pretrain"},
    {"id": "jonam-ai/slm-125m-base",    "name": "Our base",    "family": "slm", "kind": "base", "params": "125.8M", "arch": "Llama 125M", "note": "our 2-epoch pretrain"},
    {"id": "jonam-ai/legal-slm-125m-sft",  "name": "Our SFT",  "family": "slm", "kind": "sft",  "params": "125.8M", "arch": "Llama 125M", "note": "full fine-tune"},
    {"id": "jonam-ai/legal-slm-125m-raft", "name": "Our RAFT", "family": "slm", "kind": "raft", "params": "125.8M", "arch": "Llama 125M", "note": "full fine-tune"},
    {"id": "thesreedath/slm-500m-base",    "name": "500M base (mentor)", "family": "slm", "kind": "base", "params": "517M", "arch": "Llama 500M", "note": "peer 500M pretrain"},
    {"id": "jonam-ai/legal-slm-500m-sft",  "name": "500M SFT",  "family": "slm", "kind": "sft",  "params": "517M", "arch": "Llama 500M", "note": "full fine-tune"},
    {"id": "jonam-ai/legal-slm-500m-raft", "name": "500M RAFT", "family": "slm", "kind": "raft", "params": "517M", "arch": "Llama 500M", "note": "full fine-tune"},
    {"id": "google/gemma-2-2b-it",         "name": "Gemma 2B (off-the-shelf)", "family": "gemma", "kind": "base", "params": "2.61B", "arch": "Gemma 2", "note": "no legal training"},
    {"id": "jonam-ai/gemma-2-2b-legal-sft",  "name": "Gemma SFT",  "family": "gemma", "kind": "sft",  "params": "2.61B", "arch": "Gemma 2", "note": "QLoRA"},
    {"id": "jonam-ai/gemma-2-2b-legal-raft", "name": "Gemma RAFT", "family": "gemma", "kind": "raft", "params": "2.61B", "arch": "Gemma 2", "note": "QLoRA"},
    # ---- DPO (preference optimization) ----
    {"id": "jonam-ai/legal-slm-125m-sft-dpo",  "name": "Our SFT + DPO",  "family": "slm", "kind": "dpo", "fmt": "sft",  "params": "125.8M", "arch": "Llama 125M", "note": "DPO on SFT"},
    {"id": "jonam-ai/legal-slm-500m-sft-dpo",  "name": "500M SFT + DPO",  "family": "slm", "kind": "dpo", "fmt": "sft",  "params": "517M",   "arch": "Llama 500M", "note": "DPO on SFT"},
    {"id": "jonam-ai/gemma-2-2b-legal-sft-dpo","name": "Gemma SFT + DPO", "family": "gemma", "kind": "dpo", "fmt": "sft",  "params": "2.61B", "arch": "Gemma 2", "note": "DPO on SFT (QLoRA)"},
    {"id": "jonam-ai/legal-slm-125m-raft-dpo", "name": "Our RAFT + DPO",  "family": "slm", "kind": "dpo", "fmt": "raft", "params": "125.8M", "arch": "Llama 125M", "note": "DPO on RAFT"},
    {"id": "jonam-ai/legal-slm-500m-raft-dpo", "name": "500M RAFT + DPO", "family": "slm", "kind": "dpo", "fmt": "raft", "params": "517M",   "arch": "Llama 500M", "note": "DPO on RAFT"},
    {"id": "jonam-ai/gemma-2-2b-legal-raft-dpo","name": "Gemma RAFT + DPO","family": "gemma", "kind": "dpo", "fmt": "raft", "params": "2.61B", "arch": "Gemma 2", "note": "DPO on RAFT (QLoRA)"},
    # ---- RLAIF (reward model + GRPO) ----
    {"id": "jonam-ai/legal-slm-125m-sft-rlaif",  "name": "Our SFT + RLAIF",  "family": "slm", "kind": "rlaif", "fmt": "sft",  "params": "125.8M", "arch": "Llama 125M", "note": "GRPO on SFT"},
    {"id": "jonam-ai/legal-slm-500m-sft-rlaif",  "name": "500M SFT + RLAIF",  "family": "slm", "kind": "rlaif", "fmt": "sft",  "params": "517M",   "arch": "Llama 500M", "note": "GRPO on SFT"},
    {"id": "jonam-ai/gemma-2-2b-legal-sft-rlaif","name": "Gemma SFT + RLAIF", "family": "gemma", "kind": "rlaif", "fmt": "sft",  "params": "2.61B", "arch": "Gemma 2", "note": "GRPO on SFT (QLoRA)"},
    {"id": "jonam-ai/legal-slm-125m-raft-rlaif", "name": "Our RAFT + RLAIF",  "family": "slm", "kind": "rlaif", "fmt": "raft", "params": "125.8M", "arch": "Llama 125M", "note": "GRPO on RAFT"},
    {"id": "jonam-ai/legal-slm-500m-raft-rlaif", "name": "500M RAFT + RLAIF", "family": "slm", "kind": "rlaif", "fmt": "raft", "params": "517M",   "arch": "Llama 500M", "note": "GRPO on RAFT"},
    {"id": "jonam-ai/gemma-2-2b-legal-raft-rlaif","name": "Gemma RAFT + RLAIF","family": "gemma", "kind": "rlaif", "fmt": "raft", "params": "2.61B", "arch": "Gemma 2", "note": "GRPO on RAFT (QLoRA)"},
]

REFUSAL_MARKERS = [
    "not contain", "cannot find", "can't find", "not found", "does not mention",
    "doesn't mention", "no information", "not in the provided", "not in the context",
    "not provided", "not stated", "does not say", "doesn't say", "not available",
    "unable to answer", "cannot answer", "can't answer", "not address", "no mention",
    "not specified", "not enough information", "does not provide", "cannot be answered",
    "context does not", "insufficient information",
]

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "accelerate==1.1.1", "numpy==1.26.4")
    .add_local_python_source("config")
)
volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}
hf_secret = modal.Secret.from_name("huggingface-token")


@app.function(image=gpu_image, gpu="A100-40GB", volumes=VOLUMES, secrets=[hf_secret], timeout=60 * 60)
def evaluate(n_closed: int = 50, n_grounded: int = 70, n_refuse: int = 40, n_bpb: int = 50,
             force: bool = False) -> dict:
    import json
    import math
    import os
    import re

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    ln2 = math.log(2)
    volume.reload()

    # ---- build the shared eval sets (raw text) ----
    def norm(t):
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t.lower())).strip()

    def final_answer(text):
        m = re.search(r"final answer:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        return norm((m.group(1) if m else text).strip())

    # closed-book QA: decode the held-out SFT val (tokenized with the mentor tokenizer)
    mtok = AutoTokenizer.from_pretrained(MENTOR_TOK)
    closed = []
    for line in open(SFT_VAL, encoding="utf-8"):
        ex = json.loads(line)
        ii, ll = ex["input_ids"], ex["labels"]
        k = next((i for i, l in enumerate(ll) if l != -100), None)
        if k is None:
            continue
        prompt_txt = mtok.decode(ii[:k], skip_special_tokens=False)
        gold = mtok.decode(ii[k:], skip_special_tokens=True)
        if "<|user|>" in prompt_txt and "<|assistant|>" in prompt_txt:
            q = prompt_txt.split("<|user|>")[1].split("<|assistant|>")[0].strip()
            if q and gold.strip():
                closed.append({"q": q, "gold": final_answer(gold)})
    closed = closed[:n_closed]

    # grounded QA + BPB text from the raw RAFT val
    raftv = [json.loads(l) for l in open(RAFT_TEXT_VAL, encoding="utf-8")]
    grounded = [{"ctx": r["context"], "q": r["question"], "gold": final_answer(r["answer"])}
                for r in raftv][:n_grounded]
    bpb_texts = [r["context"] for r in raftv][:n_bpb]

    # unanswerable: pair each question with a DIFFERENT context (answer not present)
    refuse = []
    for i in range(min(n_refuse, len(raftv))):
        j = (i + 7) % len(raftv)
        refuse.append({"ctx": raftv[j]["context"], "q": raftv[i]["question"]})

    print(f"sets: closed={len(closed)} grounded={len(grounded)} refuse={len(refuse)} bpb={len(bpb_texts)}", flush=True)

    def build_seq(meta, tok, question, context):
        """Per-family prompt wrapper -> list[int] token ids. `fmt` (base/sft/raft) picks the
        prompt shape; DPO/RLAIF models set it to their underlying setting."""
        fam = meta["family"]
        fmt = meta.get("fmt", meta["kind"])
        system = RAFT_SYSTEM if fmt == "raft" else SFT_SYSTEM
        user = f"{context}\n\nQuestion: {question}" if context else question
        if fam == "slm" and fmt == "base":
            text = (f"{context}\n\nQuestion: {question}\nAnswer:" if context
                    else f"Question: {question}\nAnswer:")
            return tok(text)["input_ids"]
        if fam == "slm":
            sid = tok.convert_tokens_to_ids
            return ([sid("<|bos|>"), sid("<|system|>")] + tok(system, add_special_tokens=False)["input_ids"]
                    + [sid("<|user|>")] + tok(user, add_special_tokens=False)["input_ids"]
                    + [sid("<|assistant|>")])
        msgs = [{"role": "user", "content": f"{system}\n\n{user}"}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return tok(text, add_special_tokens=False)["input_ids"]

    def eos_ids(meta, tok):
        if meta["family"] == "slm":
            return [tok.convert_tokens_to_ids("<|eos|>")], tok.convert_tokens_to_ids("<|pad|>")
        eot = tok.convert_tokens_to_ids("<end_of_turn>")
        return [tok.eos_token_id, eot], tok.eos_token_id

    lb_path = f"{config.DATA_ROOT}/leaderboard.json"
    existing = {}
    if not force and os.path.exists(lb_path):
        existing = {r["id"]: r for r in json.load(open(lb_path)).get("models", [])}
        print(f"reusing {len(existing)} cached rows", flush=True)

    results = []
    for meta in MODELS:
        mid = meta["id"]
        if mid in existing:
            results.append(existing[mid])
            print(f"{meta['name']:26s} (cached)", flush=True)
            continue
        dtype = torch.float32 if meta["family"] == "slm" else torch.bfloat16
        try:
            tok = AutoTokenizer.from_pretrained(mid)
            model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dtype,
                                                         device_map={"": 0}).eval()
        except Exception as e:
            print(f"{meta['name']:26s} SKIP (load failed: {str(e)[:70]})", flush=True)
            continue
        eos_list, pad = eos_ids(meta, tok)

        # bits-per-byte + own-tokenizer perplexity on held-out legal text
        tot_nll = tot_bytes = tot_tok = 0.0
        with torch.no_grad():
            for text in bpb_texts:
                enc = tok(text, return_tensors="pt").input_ids.to(device)
                if enc.shape[1] < 2:
                    continue
                logits = model(input_ids=enc).logits[0, :-1, :].float()
                labels = enc[0, 1:]
                tot_nll += float(F.cross_entropy(logits, labels, reduction="sum"))
                tot_tok += labels.numel()
                tot_bytes += len(text.encode("utf-8"))
        bpb = tot_nll / (tot_bytes * ln2)
        token_ppl = math.exp(tot_nll / tot_tok)

        def batched_gen(items):
            """items: list of (question, context) -> list[str] outputs. Left-padded batches."""
            seqs = [build_seq(meta, tok, q, c) for q, c in items]
            outs = []
            B = 32
            for i in range(0, len(seqs), B):
                chunk = seqs[i:i + B]
                mx = max(len(s) for s in chunk)
                inp = torch.full((len(chunk), mx), pad, dtype=torch.long)
                attn = torch.zeros((len(chunk), mx), dtype=torch.long)
                for j, s in enumerate(chunk):
                    inp[j, mx - len(s):] = torch.tensor(s)
                    attn[j, mx - len(s):] = 1
                inp, attn = inp.to(device), attn.to(device)
                with torch.no_grad():
                    out = model.generate(input_ids=inp, attention_mask=attn, max_new_tokens=64,
                                         do_sample=False, eos_token_id=eos_list, pad_token_id=pad)
                for j in range(len(chunk)):
                    outs.append(tok.decode(out[j, mx:], skip_special_tokens=True))
            return outs

        cb_out = batched_gen([(e["q"], None) for e in closed])
        gr_out = batched_gen([(e["q"], e["ctx"]) for e in grounded])
        rf_out = batched_gen([(e["q"], e["ctx"]) for e in refuse])
        cb = sum(1 for e, o in zip(closed, cb_out) if e["gold"] and e["gold"] in norm(o)) / max(1, len(closed))
        gr = sum(1 for e, o in zip(grounded, gr_out) if e["gold"] and e["gold"] in norm(o)) / max(1, len(grounded))
        rf = sum(1 for o in rf_out if any(m in o.lower() for m in REFUSAL_MARKERS)) / max(1, len(rf_out))

        row = {**{k: meta[k] for k in ("name", "family", "kind", "params", "arch", "note", "id")},
               "bits_per_byte": round(bpb, 3), "token_ppl": round(token_ppl, 2),
               "closed_book_acc": round(cb, 3), "grounded_acc": round(gr, 3),
               "faithful_refusal": round(rf, 3)}
        results.append(row)
        print(f"{meta['name']:26s} bpb {bpb:5.3f} | closed {cb:5.1%} | grounded {gr:5.1%} | refuse {rf:5.1%}", flush=True)
        del model
        torch.cuda.empty_cache()

    payload = {"models": results, "sets": {"closed": len(closed), "grounded": len(grounded),
                                           "refuse": len(refuse), "bpb": len(bpb_texts)}}
    with open(f"{config.DATA_ROOT}/leaderboard.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    volume.commit()
    print("\n=== LEADERBOARD JSON ===")
    print(json.dumps(payload, indent=2))
    return payload


@app.local_entrypoint()
def run():
    evaluate.remote()
