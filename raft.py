"""RAFT (Retrieval-Augmented Fine-Tuning) dataset builder.

Teaches the model to answer from PROVIDED context while ignoring distractor docs.
Uses OpenAI gpt-4o-mini as teacher + judge. Adapted for a 1024-token model: short
~200-token chunks, 1-2 distractors, so oracle + distractors + question + answer fit.

    chunk_corpus  -> /data/raft/passages.jsonl        (short passages pool)
    generate      -> /data/raft/raw/*.jsonl           (question + CoT/quote answer, grounded in oracle)
    judge         -> /data/raft/judged/*.jsonl        (keep grounded + correct)
    curate        -> /data/raft/dataset/{train,val}.jsonl  (assemble oracle+distractors, tokenize)

Run:  modal run raft.py::pilot
"""

from __future__ import annotations

import modal

import config

app = modal.App("slm-125m-raft")

GEN_MODEL = "minimax/minimax-m3"
JUDGE_MODEL = "minimax/minimax-m3"
LLM_URL = "https://openrouter.ai/api/v1/chat/completions"

MENTOR_MODEL = "thesreedath/slm-125m-base"   # tokenizer source (must match the SFT model)
SFT_MODEL = "jonam-ai/legal-slm-125m-sft"

RAFT_DIR = f"{config.DATA_ROOT}/raft"
PASSAGES_PATH = f"{RAFT_DIR}/passages.jsonl"
RAW_DIR = f"{RAFT_DIR}/raw"
JUDGED_DIR = f"{RAFT_DIR}/judged"
DATASET_DIR = f"{RAFT_DIR}/dataset"
MENTOR_TOK_DIR = f"{RAFT_DIR}/mentor_tokenizer"

CHUNK_CHARS = 850          # ~210 tokens per passage
N_DISTRACTORS = 2          # docs added beside the oracle
P_ORACLE = 0.8             # fraction of examples that keep the oracle in context
MAX_LEN = 1024
VAL_FRACTION = 0.05
SOURCE_WEIGHTS = {"case-law": 0.45, "sec": 0.45, "fineweb-edu": 0.10}

RAFT_SYSTEM = ("You are a legal and financial assistant. Use the numbered context "
               "documents to answer the question. Quote the text you rely on, then "
               "give the final answer.")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("requests==2.32.3", "huggingface_hub==0.34.4", "transformers==4.46.3",
                 "tokenizers==0.20.3", "numpy==1.26.4", "datasketch==1.6.5")
    .add_local_python_source("config")
)
volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}
openrouter_secret = modal.Secret.from_name("openrouter-api")


# ----------------------------- OpenAI helper ----------------------------- #
def _openai(model, system, user, *, api_key, max_tokens, temperature):
    import random
    import time
    import requests

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": max_tokens, "temperature": temperature,
            "response_format": {"type": "json_object"}}
    last = ""
    for attempt in range(8):
        try:
            r = requests.post(LLM_URL, headers=headers, json=body, timeout=120)
            if r.status_code == 200:
                d = r.json()
                u = d.get("usage", {})
                return (d["choices"][0]["message"]["content"],
                        {"in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0)})
            last = f"{r.status_code}: {r.text[:120]}"
            if r.status_code == 429:
                ra = r.headers.get("retry-after-ms")
                wait = (float(ra) / 1000 if ra else
                        float(r.headers.get("retry-after", 0)) or 2 ** attempt)
                time.sleep(min(45, wait + random.uniform(0, 1.5)))
                continue
            if r.status_code in (500, 502, 503):
                time.sleep(2 ** attempt + random.uniform(0, 1)); continue
            break
        except Exception as e:
            last = str(e)[:120]; time.sleep(2 ** attempt + random.uniform(0, 1))
    print(f"  [openai {model}] failed: {last}", flush=True)
    return "", {"in": 0, "out": 0}


def _parse_json(text):
    import json
    import re
    if not text:
        return None
    # strict=False tolerates literal newlines/control chars inside string values,
    # which minimax-m3 emits in its multi-line cot_answer.
    try:
        return json.loads(text, strict=False)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0), strict=False)
            except Exception:
                return None
    return None


# ----------------------------- prompts ----------------------------- #
def _gen_user(oracle):
    return f"""Read the PASSAGE. Write ONE question answerable ONLY from it, plus a chain-of-thought answer.

Rules:
- The question must be self-contained: name the entity, company, or case; never say "this passage".
- The cot_answer must (a) quote the exact relevant span from the passage, wrapped in ##begin_quote## and ##end_quote##, then (b) give a short final answer on a new line starting with "Final answer:".
- Use ONLY the passage. Never invent facts, names, or numbers.

Return JSON: {{"question": "...", "cot_answer": "..."}}

PASSAGE:
\"\"\"{oracle}\"\"\""""


def _judge_user(oracle, q, a):
    return f"""Validate this retrieval-augmented QA pair against the PASSAGE.

Check: (1) grounded - the answer and its quote are fully supported by the passage; (2) correct - the final answer is right and responsive; (3) self_contained - the question makes sense without the passage; (4) quote_valid - the quoted span actually appears in the passage.

Return JSON: {{"keep": true or false, "score": 1-5, "reason": "..."}}. keep=true requires score>=4 AND quote_valid.

PASSAGE:
\"\"\"{oracle}\"\"\"
QUESTION: {q}
ANSWER: {a}"""


# ----------------------------- chunk ----------------------------- #
@app.function(image=image, volumes=VOLUMES, timeout=60 * 20)
def chunk_corpus(n_passages: int = 3000, seed: int = 7) -> dict:
    import glob
    import json
    import os
    import random

    import shutil
    rng = random.Random(seed)
    os.makedirs(RAFT_DIR, exist_ok=True)
    shutil.rmtree(RAW_DIR, ignore_errors=True)      # clear stale shards from prior runs
    shutil.rmtree(JUDGED_DIR, ignore_errors=True)
    picked = []
    for source, w in SOURCE_WEIGHTS.items():
        want = int(n_passages * w)
        docs = []
        for path in sorted(glob.glob(f"{config.CORPUS_DIR}/{source}/*.txt")):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if len(line) >= CHUNK_CHARS:
                        if len(docs) < want * 4:
                            docs.append(line)
                        else:
                            j = rng.randint(0, len(docs) * 2)
                            if j < len(docs):
                                docs[j] = line
        rng.shuffle(docs)
        for doc in docs[:want]:
            start = 0 if len(doc) <= CHUNK_CHARS else rng.randint(0, len(doc) - CHUNK_CHARS)
            picked.append({"source": source, "text": doc[start:start + CHUNK_CHARS].strip()})
    rng.shuffle(picked)
    with open(PASSAGES_PATH, "w", encoding="utf-8") as fh:
        for i, p in enumerate(picked):
            p["id"] = i
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    volume.commit()
    print(f"chunked {len(picked)} passages (~{CHUNK_CHARS} chars each)")
    return {"n": len(picked)}


# ----------------------------- generate + judge ----------------------------- #
@app.function(image=image, volumes=VOLUMES, secrets=[openrouter_secret], timeout=60 * 60 * 2, cpu=2.0)
def generate_shard(shard_id: int, oracles: list) -> dict:
    import json
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    key = os.environ["OPENROUTER_API_KEY"]
    rows, usage = [], {"in": 0, "out": 0}

    def work(p):
        txt, u = _openai(GEN_MODEL, "You create retrieval-augmented QA training data for a legal and financial assistant.",
                         _gen_user(p["text"]), api_key=key, max_tokens=400, temperature=0.7)
        obj = _parse_json(txt)
        if obj and obj.get("question") and obj.get("cot_answer"):
            return {"question": str(obj["question"]).strip(), "cot_answer": str(obj["cot_answer"]).strip(),
                    "oracle_id": p["id"], "oracle": p["text"], "source": p["source"]}, u
        return None, u

    with ThreadPoolExecutor(max_workers=6) as ex:   # OpenRouter handles higher concurrency
        for f in as_completed([ex.submit(work, p) for p in oracles]):
            row, u = f.result()
            usage["in"] += u["in"]; usage["out"] += u["out"]
            if row:
                rows.append(row)
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(f"{RAW_DIR}/shard-{shard_id:03d}.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    volume.commit()
    print(f"[gen {shard_id:03d}] {len(oracles)} oracles -> {len(rows)} pairs | tok in={usage['in']} out={usage['out']}")
    return {"pairs": len(rows), "usage": usage}


@app.function(image=image, volumes=VOLUMES, secrets=[openrouter_secret], timeout=60 * 60 * 2, cpu=2.0)
def judge_shard(shard_id: int, rows: list) -> dict:
    import json
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    key = os.environ["OPENROUTER_API_KEY"]
    kept, usage = [], {"in": 0, "out": 0}

    def work(r):
        txt, u = _openai(JUDGE_MODEL, "You are a strict validator for retrieval-augmented QA training data.",
                         _judge_user(r["oracle"], r["question"], r["cot_answer"]), api_key=key,
                         max_tokens=200, temperature=0.0)
        v = _parse_json(txt) or {}
        return (r if (v.get("keep") and int(v.get("score", 0)) >= 4) else None), u

    with ThreadPoolExecutor(max_workers=6) as ex:   # OpenRouter handles higher concurrency
        for f in as_completed([ex.submit(work, r) for r in rows]):
            keep, u = f.result()
            usage["in"] += u["in"]; usage["out"] += u["out"]
            if keep:
                kept.append(keep)
    os.makedirs(JUDGED_DIR, exist_ok=True)
    with open(f"{JUDGED_DIR}/shard-{shard_id:03d}.jsonl", "w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    volume.commit()
    print(f"[judge {shard_id:03d}] kept {len(kept)}/{len(rows)} | tok in={usage['in']} out={usage['out']}")
    return {"kept": len(kept), "usage": usage}


# ----------------------------- cost ----------------------------- #
RATE = {"in": 0.30, "out": 1.20}   # minimax-m3 via OpenRouter, USD / 1M tokens


def _cost(u):
    return u["in"] / 1e6 * RATE["in"] + u["out"] / 1e6 * RATE["out"]


# ----------------------------- curate (assemble + tokenize) ----------------------------- #
def _build_context(docs):
    return "Context:\n" + "\n".join(f"[{i+1}] {d}" for i, d in enumerate(docs))


@app.function(image=image, volumes=VOLUMES, timeout=60 * 30, cpu=4.0, memory=8_192)
def curate(seed: int = 7) -> dict:
    import glob
    import json
    import os
    import random
    import re

    from datasketch import MinHash, MinHashLSH
    from transformers import AutoTokenizer

    rng = random.Random(seed)
    volume.reload()
    pool = [json.loads(l) for l in open(PASSAGES_PATH, encoding="utf-8")]
    pool_by_id = {p["id"]: p["text"] for p in pool}
    all_ids = list(pool_by_id.keys())

    rows = []
    for path in sorted(glob.glob(f"{JUDGED_DIR}/*.jsonl")):
        rows.extend(json.loads(l) for l in open(path, encoding="utf-8"))
    print(f"loaded {len(rows)} judged pairs")

    # exact + near-dup on the question
    def norm(q):
        return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", q.lower())).strip()
    seen, uniq = set(), []
    lsh = MinHashLSH(threshold=0.7, num_perm=64)
    for i, r in enumerate(rows):
        n = norm(r["question"])
        if not n or n in seen:
            continue
        words = n.split()
        m = MinHash(num_perm=64)
        for s in {" ".join(words[j:j+4]) for j in range(max(1, len(words)-3))} or set(words):
            m.update(s.encode())
        if lsh.query(m):
            continue
        lsh.insert(str(i), m); seen.add(n); uniq.append(r)
    print(f"after dedup: {len(uniq)}")

    tok = AutoTokenizer.from_pretrained(MENTOR_MODEL, cache_dir=MENTOR_TOK_DIR)
    tok.save_pretrained(MENTOR_TOK_DIR)
    sid = tok.convert_tokens_to_ids
    BOS, EOS = sid("<|bos|>"), sid("<|eos|>")
    SYS, USER, ASST = sid("<|system|>"), sid("<|user|>"), sid("<|assistant|>")
    sys_ids = tok(RAFT_SYSTEM, add_special_tokens=False)["input_ids"]

    def encode(context, question, answer):
        u = tok(f"{context}\n\nQuestion: {question}", add_special_tokens=False)["input_ids"]
        a = tok(answer, add_special_tokens=False)["input_ids"]
        prompt = [BOS, SYS] + sys_ids + [USER] + u + [ASST]
        ans = a + [EOS]
        if len(prompt) + len(ans) > MAX_LEN:      # trim the context side, keep the full answer
            keep = MAX_LEN - len(ans) - ([BOS, SYS] + sys_ids + [USER] + [ASST]).__len__()
            if keep < 40:
                return None
            u = u[:keep]
            prompt = [BOS, SYS] + sys_ids + [USER] + u + [ASST]
        return prompt + ans, [-100] * len(prompt) + ans

    examples, n_oracle = [], 0
    for r in uniq:
        distractors = []
        while len(distractors) < N_DISTRACTORS:
            did = rng.choice(all_ids)
            if did != r["oracle_id"]:
                distractors.append(pool_by_id[did])
        include_oracle = rng.random() < P_ORACLE
        docs = ([r["oracle"]] if include_oracle else []) + distractors
        rng.shuffle(docs)
        enc = encode(_build_context(docs), r["question"], r["cot_answer"])
        if enc is None:
            continue
        ii, ll = enc
        examples.append({"input_ids": ii, "labels": ll})
        n_oracle += int(include_oracle)

    rng.shuffle(examples)
    n_val = max(80, int(len(examples) * VAL_FRACTION))
    val, train = examples[:n_val], examples[n_val:]
    os.makedirs(DATASET_DIR, exist_ok=True)

    def write(name, rows_):
        toks = 0
        with open(f"{DATASET_DIR}/{name}.jsonl", "w", encoding="utf-8") as fh:
            for e in rows_:
                toks += sum(1 for x in e["labels"] if x != -100)
                fh.write(json.dumps(e) + "\n")
        return toks

    tr_tok = write("train", train)
    va_tok = write("val", val)
    meta = {"final": len(examples), "train": len(train), "val": len(val),
            "with_oracle": n_oracle, "no_oracle": len(examples) - n_oracle,
            "n_distractors": N_DISTRACTORS, "p_oracle": P_ORACLE, "max_len": MAX_LEN,
            "train_answer_tokens": tr_tok, "val_answer_tokens": va_tok, "tokenizer": MENTOR_MODEL}
    with open(f"{DATASET_DIR}/meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    volume.commit()
    print(json.dumps(meta, indent=2))
    return meta


# ----------------------------- helpers + entrypoints ----------------------------- #
@app.function(image=image, volumes=VOLUMES)
def read_passages(n: int) -> list:
    import json
    volume.reload()   # see commits from other containers (warm-reuse safe)
    out = []
    with open(PASSAGES_PATH, encoding="utf-8") as fh:
        for line in fh:
            out.append(json.loads(line))
            if len(out) >= n:
                break
    return out


@app.function(image=image, volumes=VOLUMES)
def read_jsonl(path: str) -> list:
    import json
    volume.reload()
    return [json.loads(l) for l in open(path, encoding="utf-8")]


@app.local_entrypoint()
def pilot(n: int = 20):
    print(f"== RAFT PILOT: {n} oracles ==")
    chunk_corpus.remote(n_passages=max(500, n * 10))
    oracles = list(read_passages.remote(n))
    gen = generate_shard.remote(0, oracles)
    raw = list(read_jsonl.remote(f"{RAW_DIR}/shard-000.jsonl"))
    jud = judge_shard.remote(0, raw)
    kept = list(read_jsonl.remote(f"{JUDGED_DIR}/shard-000.jsonl"))

    print("\n=== SAMPLES (kept) ===")
    for r in kept[:5]:
        print(f"\n[{r['source']}]  Q: {r['question']}")
        print(f"  A: {r['cot_answer'][:400]}")
    gc, jc = _cost(gen["usage"]), _cost(jud["usage"])
    kn = jud["kept"]
    print(f"\n=== ECONOMICS ===")
    print(f"oracles={n} raw={gen['pairs']} kept={kn} keep={kn/max(1,gen['pairs']):.0%}")
    print(f"gen ${gc:.4f}  judge ${jc:.4f}  total ${gc+jc:.4f}  | per-kept ${ (gc+jc)/max(1,kn):.5f}")
    print(f"PROJECTION for 5,000 kept: ~${(gc+jc)/max(1,kn)*5000:.2f}")


@app.local_entrypoint()
def build(n_passages: int = 7500, shards: int = 4):
    print(f"== RAFT BUILD: {n_passages} oracles, {shards} shards ==")
    chunk_corpus.remote(n_passages=int(n_passages * 1.15))   # extra pool for distractors
    oracles = list(read_passages.remote(n_passages))
    gwork = [(i, oracles[i::shards]) for i in range(shards)]
    gen = list(generate_shard.starmap(gwork))
    raw_n = sum(g["pairs"] for g in gen); gcost = sum(_cost(g["usage"]) for g in gen)
    print(f"generated {raw_n} | gen ${gcost:.3f}")
    allraw = []
    for i in range(shards):
        allraw.extend(read_jsonl.remote(f"{RAW_DIR}/shard-{i:03d}.jsonl"))
    jwork = [(i, allraw[i::shards]) for i in range(shards)]
    jud = list(judge_shard.starmap(jwork))
    kept_n = sum(j["kept"] for j in jud); jcost = sum(_cost(j["usage"]) for j in jud)
    print(f"kept {kept_n}/{raw_n} ({kept_n/max(1,raw_n):.0%}) | judge ${jcost:.3f}")
    print(f"TOTAL OpenAI ${gcost+jcost:.3f}\nnext: modal run raft.py::curate_run")


@app.local_entrypoint()
def curate_run():
    curate.remote()


@app.function(image=image, volumes=VOLUMES, timeout=60 * 20, cpu=4.0, memory=8_192)
def export_text(seed: int = 7) -> dict:
    """Emit the assembled RAFT examples as {context, question, answer} TEXT, so any
    tokenizer (e.g. Gemma's) can consume them. Same assembly as curate()."""
    import glob
    import json
    import os
    import random
    import re

    from datasketch import MinHash, MinHashLSH

    rng = random.Random(seed)
    volume.reload()
    pool = [json.loads(l) for l in open(PASSAGES_PATH, encoding="utf-8")]
    pool_by_id = {p["id"]: p["text"] for p in pool}
    all_ids = list(pool_by_id.keys())

    rows = []
    for path in sorted(glob.glob(f"{JUDGED_DIR}/*.jsonl")):
        rows.extend(json.loads(l) for l in open(path, encoding="utf-8"))

    def norm(q):
        return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", q.lower())).strip()
    seen, uniq = set(), []
    lsh = MinHashLSH(threshold=0.7, num_perm=64)
    for i, r in enumerate(rows):
        n = norm(r["question"])
        if not n or n in seen:
            continue
        words = n.split()
        m = MinHash(num_perm=64)
        for s in {" ".join(words[j:j+4]) for j in range(max(1, len(words)-3))} or set(words):
            m.update(s.encode())
        if lsh.query(m):
            continue
        lsh.insert(str(i), m); seen.add(n); uniq.append(r)

    out = []
    for r in uniq:
        distractors = []
        while len(distractors) < N_DISTRACTORS:
            did = rng.choice(all_ids)
            if did != r["oracle_id"]:
                distractors.append(pool_by_id[did])
        docs = ([r["oracle"]] if rng.random() < P_ORACLE else []) + distractors
        rng.shuffle(docs)
        out.append({"context": _build_context(docs), "question": r["question"], "answer": r["cot_answer"]})

    rng.shuffle(out)
    n_val = max(80, int(len(out) * VAL_FRACTION))
    os.makedirs(DATASET_DIR, exist_ok=True)
    with open(f"{DATASET_DIR}/raft_text_val.jsonl", "w", encoding="utf-8") as fh:
        for e in out[:n_val]:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(f"{DATASET_DIR}/raft_text_train.jsonl", "w", encoding="utf-8") as fh:
        for e in out[n_val:]:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    volume.commit()
    print(f"exported {len(out)} RAFT text examples ({len(out)-n_val} train / {n_val} val)")
    return {"total": len(out), "val": n_val}


@app.local_entrypoint()
def export_text_run():
    export_text.remote()
