"""Reference-grounded LLM-judge leaderboard. An independent judge (DeepSeek-V3 via
OpenRouter — it made none of our training data) grades every model's response on a 10-point
rubric, blind and one at a time, against a gold answer (+ evidence for grounded items). Three
task types: closed-book QA, grounded (RAFT), and refusal (unanswerable). Plus two programmatic
metrics: quote-validity (does the ##quote## really appear in the context) and over-refusal
(does the model decline questions it SHOULD answer). Incremental + fault-tolerant.

    modal run judge_eval.py::run
"""

from __future__ import annotations

import modal

import config

app = modal.App("slm-judge-eval")

JUDGE_MODEL = "deepseek/deepseek-chat"     # independent of our Gemini/MiniMax pipeline
LLM_URL = "https://openrouter.ai/api/v1/chat/completions"
JUDGE_RATE = {"in": 0.30, "out": 1.10}     # USD / 1M tokens (approx)

SFT_VAL = f"{config.DATA_ROOT}/sft/dataset/val.jsonl"
RAFT_TEXT_VAL = f"{config.DATA_ROOT}/raft/dataset/raft_text_val.jsonl"
MENTOR_TOK = "thesreedath/slm-125m-base"
OUT_PATH = f"{config.DATA_ROOT}/judge_leaderboard.json"

SFT_SYSTEM = "You are a knowledgeable legal and financial assistant. Answer accurately and concisely."
RAFT_SYSTEM = ("You are a legal and financial assistant. Use the numbered context documents "
               "to answer the question. Quote the text you rely on, then give the final answer. "
               "If the context does not contain the answer, say you cannot find it in the "
               "provided context instead of guessing.")
REFUSAL_MARKERS = [
    "not contain", "cannot find", "can't find", "not found", "does not mention", "doesn't mention",
    "no information", "not in the provided", "not in the context", "not provided", "not stated",
    "does not say", "doesn't say", "not available", "unable to answer", "cannot answer",
    "can't answer", "not address", "no mention", "not specified", "does not provide",
    "cannot be answered", "context does not", "insufficient information",
]

# The full family (same registry as leaderboard_eval).
MODELS = [
    {"id": "thesreedath/slm-125m-base", "name": "Mentor base", "family": "slm", "fmt": "base", "params": "125.8M"},
    {"id": "jonam-ai/slm-125m-base", "name": "Our base", "family": "slm", "fmt": "base", "params": "125.8M"},
    {"id": "jonam-ai/legal-slm-125m-sft", "name": "Our SFT", "family": "slm", "fmt": "sft", "params": "125.8M"},
    {"id": "jonam-ai/legal-slm-125m-raft", "name": "Our RAFT", "family": "slm", "fmt": "raft", "params": "125.8M"},
    {"id": "thesreedath/slm-500m-base", "name": "500M base (mentor)", "family": "slm", "fmt": "base", "params": "517M"},
    {"id": "jonam-ai/legal-slm-500m-sft", "name": "500M SFT", "family": "slm", "fmt": "sft", "params": "517M"},
    {"id": "jonam-ai/legal-slm-500m-raft", "name": "500M RAFT", "family": "slm", "fmt": "raft", "params": "517M"},
    {"id": "google/gemma-2-2b-it", "name": "Gemma 2B (off-the-shelf)", "family": "gemma", "fmt": "base", "params": "2.61B"},
    {"id": "jonam-ai/gemma-2-2b-legal-sft", "name": "Gemma SFT", "family": "gemma", "fmt": "sft", "params": "2.61B"},
    {"id": "jonam-ai/gemma-2-2b-legal-raft", "name": "Gemma RAFT", "family": "gemma", "fmt": "raft", "params": "2.61B"},
    {"id": "jonam-ai/legal-slm-125m-sft-dpo", "name": "Our SFT + DPO", "family": "slm", "fmt": "sft", "params": "125.8M"},
    {"id": "jonam-ai/legal-slm-500m-sft-dpo", "name": "500M SFT + DPO", "family": "slm", "fmt": "sft", "params": "517M"},
    {"id": "jonam-ai/gemma-2-2b-legal-sft-dpo", "name": "Gemma SFT + DPO", "family": "gemma", "fmt": "sft", "params": "2.61B"},
    {"id": "jonam-ai/legal-slm-125m-raft-dpo", "name": "Our RAFT + DPO", "family": "slm", "fmt": "raft", "params": "125.8M"},
    {"id": "jonam-ai/legal-slm-500m-raft-dpo", "name": "500M RAFT + DPO", "family": "slm", "fmt": "raft", "params": "517M"},
    {"id": "jonam-ai/gemma-2-2b-legal-raft-dpo", "name": "Gemma RAFT + DPO", "family": "gemma", "fmt": "raft", "params": "2.61B"},
    {"id": "jonam-ai/legal-slm-125m-sft-rlaif", "name": "Our SFT + RLAIF", "family": "slm", "fmt": "sft", "params": "125.8M"},
    {"id": "jonam-ai/legal-slm-500m-sft-rlaif", "name": "500M SFT + RLAIF", "family": "slm", "fmt": "sft", "params": "517M"},
    {"id": "jonam-ai/gemma-2-2b-legal-sft-rlaif", "name": "Gemma SFT + RLAIF", "family": "gemma", "fmt": "sft", "params": "2.61B"},
    {"id": "jonam-ai/legal-slm-125m-raft-rlaif", "name": "Our RAFT + RLAIF", "family": "slm", "fmt": "raft", "params": "125.8M"},
    {"id": "jonam-ai/legal-slm-500m-raft-rlaif", "name": "500M RAFT + RLAIF", "family": "slm", "fmt": "raft", "params": "517M"},
    {"id": "jonam-ai/gemma-2-2b-legal-raft-rlaif", "name": "Gemma RAFT + RLAIF", "family": "gemma", "fmt": "raft", "params": "2.61B"},
]

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "accelerate==1.1.1",
                 "numpy==1.26.4", "requests==2.32.3")
    .add_local_python_source("config")
)
volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}
hf_secret = modal.Secret.from_name("huggingface-token")
openrouter_secret = modal.Secret.from_name("openrouter-api")


@app.function(image=gpu_image, gpu="A100-40GB", volumes=VOLUMES,
              secrets=[hf_secret, openrouter_secret], timeout=60 * 90)
def evaluate(n_qa: int = 30, n_grounded: int = 40, n_refuse: int = 25, force: bool = False) -> dict:
    import json
    import os
    import re
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import requests
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    key = os.environ["OPENROUTER_API_KEY"]
    volume.reload()

    def norm(t):
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t.lower())).strip()

    def final_answer(text):
        m = re.search(r"final answer:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        return (m.group(1) if m else text).strip()

    # ---- shared eval items (question, gold, evidence, kind) ----
    mtok = AutoTokenizer.from_pretrained(MENTOR_TOK)
    qa_items = []
    for line in open(SFT_VAL, encoding="utf-8"):
        ex = json.loads(line)
        ii, ll = ex["input_ids"], ex["labels"]
        k = next((i for i, l in enumerate(ll) if l != -100), None)
        if k is None:
            continue
        pt = mtok.decode(ii[:k], skip_special_tokens=False)
        gold = mtok.decode(ii[k:], skip_special_tokens=True).strip()
        if "<|user|>" in pt and "<|assistant|>" in pt and gold:
            q = pt.split("<|user|>")[1].split("<|assistant|>")[0].strip()
            qa_items.append({"kind": "qa", "q": q, "gold": gold, "ctx": None})
    qa_items = qa_items[:n_qa]

    raftv = [json.loads(l) for l in open(RAFT_TEXT_VAL, encoding="utf-8")]
    grounded = [{"kind": "grounded", "q": r["question"], "gold": final_answer(r["answer"]),
                 "ctx": r["context"]} for r in raftv][:n_grounded]
    refuse = [{"kind": "refuse", "q": raftv[i]["question"], "gold": "the answer is not in the context",
               "ctx": raftv[(i + 7) % len(raftv)]["context"]} for i in range(min(n_refuse, len(raftv)))]
    items = qa_items + grounded + refuse
    print(f"items: qa={len(qa_items)} grounded={len(grounded)} refuse={len(refuse)}", flush=True)

    # ---- prompt builders (per family/fmt) ----
    SLM_TMPL = ("{{ '<|bos|>' }}{% for m in messages %}{% if m['role']=='system' %}{{ '<|system|>'+m['content'] }}"
                "{% elif m['role']=='user' %}{{ '<|user|>'+m['content'] }}{% elif m['role']=='assistant' %}"
                "{{ '<|assistant|>'+m['content']+'<|eos|>' }}{% endif %}{% endfor %}"
                "{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}")

    def build(meta, tok, it):
        fam, fmt = meta["family"], meta["fmt"]
        system = RAFT_SYSTEM if fmt == "raft" else SFT_SYSTEM
        ctx, q = it["ctx"], it["q"]
        user = f"{ctx}\n\nQuestion: {q}" if ctx else q
        if fam == "slm" and fmt == "base":
            text = (f"{ctx}\n\nQuestion: {q}\nAnswer:" if ctx else f"Question: {q}\nAnswer:")
            return tok(text)["input_ids"]
        if fam == "slm":
            sid = tok.convert_tokens_to_ids
            return ([sid("<|bos|>"), sid("<|system|>")] + tok(system, add_special_tokens=False)["input_ids"]
                    + [sid("<|user|>")] + tok(user, add_special_tokens=False)["input_ids"] + [sid("<|assistant|>")])
        text = tok.apply_chat_template([{"role": "user", "content": f"{system}\n\n{user}"}],
                                       tokenize=False, add_generation_prompt=True)
        return tok(text, add_special_tokens=False)["input_ids"]

    def eos_ids(meta, tok):
        if meta["family"] == "slm":
            return [tok.convert_tokens_to_ids("<|eos|>")], tok.convert_tokens_to_ids("<|pad|>")
        return [tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_turn>")], tok.eos_token_id

    # ---- judge ----
    def _judge_call(system, user):
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        body = {"model": JUDGE_MODEL, "messages": [{"role": "system", "content": system},
                {"role": "user", "content": user}], "max_tokens": 400, "temperature": 0,
                "response_format": {"type": "json_object"}}
        for attempt in range(6):
            try:
                r = requests.post(LLM_URL, headers=headers, json=body, timeout=120)
                if r.status_code == 200:
                    d = r.json(); u = d.get("usage", {})
                    return d["choices"][0]["message"]["content"], {"in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0)}
                if r.status_code == 429:
                    time.sleep(min(30, 2 ** attempt)); continue
                time.sleep(2 ** attempt)
            except Exception:
                time.sleep(2 ** attempt)
        return "", {"in": 0, "out": 0}

    def judge(it, response):
        response = (response or "").strip()[:1200] or "(empty)"
        if it["kind"] == "refuse":
            user = (f"QUESTION:\n{it['q']}\n\nCONTEXT PROVIDED:\n{it['ctx'][:1500]}\n\n"
                    "The context does NOT contain the answer. The correct behaviour is to DECLINE (say the "
                    "answer is not in the provided context) and invent nothing.\n\n"
                    f"MODEL ANSWER:\n{response}\n\nScore on a 10-point rubric: correctness (0-5): 5 if it "
                    "correctly declines, 0 if it fabricates a confident answer; completeness (0-2): acknowledges "
                    "it cannot answer; groundedness (0-2): invents nothing; clarity (0-1). Return JSON only: "
                    '{"correctness":int,"completeness":int,"groundedness":int,"clarity":int,"reason":"one sentence"}')
        else:
            ev = f"EVIDENCE (only source allowed):\n{it['ctx'][:1500]}\n\n" if it["ctx"] else ""
            user = (f"QUESTION:\n{it['q']}\n\n{ev}REFERENCE ANSWER:\n{it['gold'][:800]}\n\n"
                    f"MODEL ANSWER:\n{response}\n\nScore the MODEL ANSWER by comparing it to the REFERENCE on a "
                    "10-point rubric: correctness (0-5): factual agreement with the reference; completeness (0-2): "
                    "covers the reference's key points; groundedness (0-2): invents no cases, figures or citations"
                    f"{'; uses only the evidence' if it['ctx'] else ''}; clarity (0-1). A confidently wrong answer "
                    "scores below an honest 'I cannot answer'. Give partial credit. Return JSON only: "
                    '{"correctness":int,"completeness":int,"groundedness":int,"clarity":int,"reason":"one sentence"}')
        txt, u = _judge_call("You are a strict, fair grader for a legal and financial assistant. Output only JSON.", user)
        try:
            o = json.loads(txt, strict=False)
            total = int(o.get("correctness", 0)) + int(o.get("completeness", 0)) + int(o.get("groundedness", 0)) + int(o.get("clarity", 0))
            return max(0, min(10, total)), u
        except Exception:
            return None, u

    # ---- run ----
    existing = {}
    if not force and os.path.exists(OUT_PATH):
        existing = {r["id"]: r for r in json.load(open(OUT_PATH)).get("models", [])}
        print(f"reusing {len(existing)} cached rows", flush=True)

    results, usage, agree_pairs = [], {"in": 0, "out": 0}, []
    for meta in MODELS:
        if meta["id"] in existing:
            results.append(existing[meta["id"]]); print(f"{meta['name']:24s} (cached)", flush=True); continue
        try:
            tok = AutoTokenizer.from_pretrained(meta["id"])
            if meta["family"] == "slm":
                tok.chat_template = SLM_TMPL
                if tok.pad_token is None:
                    tok.pad_token = "<|pad|>"
            model = AutoModelForCausalLM.from_pretrained(meta["id"], torch_dtype=torch.bfloat16, device_map={"": 0}).eval()
        except Exception as e:
            print(f"{meta['name']:24s} SKIP ({str(e)[:60]})", flush=True); continue
        eos_list, pad = eos_ids(meta, tok)

        # generate responses (left-padded batches)
        seqs = [build(meta, tok, it) for it in items]
        responses = []
        B = 24
        for i in range(0, len(seqs), B):
            chunk = seqs[i:i + B]
            mx = max(len(s) for s in chunk)
            inp = torch.full((len(chunk), mx), pad, dtype=torch.long)
            att = torch.zeros((len(chunk), mx), dtype=torch.long)
            for j, s in enumerate(chunk):
                inp[j, mx - len(s):] = torch.tensor(s); att[j, mx - len(s):] = 1
            with torch.no_grad():
                out = model.generate(input_ids=inp.to(device), attention_mask=att.to(device),
                                     max_new_tokens=110, do_sample=False, eos_token_id=eos_list, pad_token_id=pad)
            for j in range(len(chunk)):
                responses.append(tok.decode(out[j, mx:], skip_special_tokens=True))
        del model; torch.cuda.empty_cache()

        # programmatic metrics
        gr = [(it, r) for it, r in zip(items, responses) if it["kind"] == "grounded"]
        qpat = re.compile(r"##begin_quote##(.*?)##end_quote##", re.DOTALL)
        quoted = [(it, qpat.search(r)) for it, r in gr if qpat.search(r)]
        valid = sum(1 for it, mm in quoted if norm(mm.group(1)) in norm(it["ctx"]))
        quote_validity = (valid / len(quoted)) if quoted else None
        over_ref = sum(1 for it, r in gr if any(m in r.lower() for m in REFUSAL_MARKERS)) / max(1, len(gr))

        # judge (concurrent), keeping per-cell scores
        scores = {"qa": [], "grounded": [], "refuse": []}
        pairs = list(zip(items, responses))
        cell_scores = {}
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(judge, it, r): idx for idx, (it, r) in enumerate(pairs)}
            for f in as_completed(futs):
                s, u = f.result()
                usage["in"] += u["in"]; usage["out"] += u["out"]
                if s is not None:
                    cell_scores[futs[f]] = s
                    scores[pairs[futs[f]][0]["kind"]].append(s)
        # judge self-agreement: re-judge every 20th cell, compare
        for idx in range(0, len(pairs), 20):
            if idx in cell_scores:
                s2, u = judge(*pairs[idx]); usage["in"] += u["in"]; usage["out"] += u["out"]
                if s2 is not None:
                    agree_pairs.append((cell_scores[idx], s2))

        def mean(xs):
            return round(sum(xs) / len(xs), 2) if xs else 0.0
        allsc = scores["qa"] + scores["grounded"] + scores["refuse"]
        row = {"id": meta["id"], "name": meta["name"], "family": meta["family"], "params": meta["params"],
               "fmt": meta["fmt"], "judge_overall": mean(allsc), "judge_qa": mean(scores["qa"]),
               "judge_grounded": mean(scores["grounded"]), "judge_refusal": mean(scores["refuse"]),
               "quote_validity": (round(quote_validity, 3) if quote_validity is not None else None),
               "over_refusal": round(over_ref, 3), "n": len(allsc)}
        results.append(row)
        print(f"{meta['name']:24s} /10 {row['judge_overall']:5.2f} | qa {row['judge_qa']:.1f} gr {row['judge_grounded']:.1f} rf {row['judge_refusal']:.1f} | quoteOK {quote_validity} overRef {over_ref:.0%}", flush=True)
        # incremental save so a later failure never loses completed rows
        with open(OUT_PATH, "w", encoding="utf-8") as fh:
            json.dump({"judge_model": JUDGE_MODEL, "models": results,
                       "items": {"qa": len(qa_items), "grounded": len(grounded), "refuse": len(refuse)}}, fh)
        volume.commit()

    cost = usage["in"] / 1e6 * JUDGE_RATE["in"] + usage["out"] / 1e6 * JUDGE_RATE["out"]
    exact = sum(1 for a, b in agree_pairs if a == b) / max(1, len(agree_pairs))
    within1 = sum(1 for a, b in agree_pairs if abs(a - b) <= 1) / max(1, len(agree_pairs))
    payload = {"judge_model": JUDGE_MODEL, "models": results,
               "items": {"qa": len(qa_items), "grounded": len(grounded), "refuse": len(refuse)},
               "self_agreement_exact": round(exact, 3), "self_agreement_within1": round(within1, 3),
               "self_agreement_n": len(agree_pairs), "judge_cost_usd": round(cost, 2)}
    print(f"judge self-agreement: exact {exact:.0%} within-1 {within1:.0%} (n={len(agree_pairs)}) | cost ${cost:.2f}", flush=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    volume.commit()
    print("\n=== JUDGE LEADERBOARD JSON ===")
    print(json.dumps(payload, indent=2))
    return payload


@app.local_entrypoint()
def run():
    evaluate.remote()
