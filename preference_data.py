"""Build preference datasets for DPO and RLAIF, using a strong teacher (MiniMax-M3 via
OpenRouter) as the AI preference labeler. For every prompt the teacher writes one excellent
answer (chosen) and one plausible-but-inferior answer (rejected). Two settings:

  - sft  : instruction prompts (legal/financial questions) -> /data/pref/sft.jsonl
  - raft : grounded prompts (context + question), incl. ~30% UNANSWERABLE ones where
           chosen = decline and rejected = fabricate, teaching faithfulness by preference
           -> /data/pref/raft.jsonl

    modal run preference_data.py::build --setting sft  --n 2500
    modal run preference_data.py::build --setting raft --n 2500
"""

from __future__ import annotations

import modal

import config

app = modal.App("slm-preference-data")

MODEL = "minimax/minimax-m3"
LLM_URL = "https://openrouter.ai/api/v1/chat/completions"
RATE = {"in": 0.30, "out": 1.20}   # USD / 1M tokens
SFT_CHAT = f"{config.DATA_ROOT}/sft/dataset/chat.jsonl"
RAFT_TRAIN = f"{config.DATA_ROOT}/raft/dataset/raft_text_train.jsonl"
PREF_DIR = f"{config.DATA_ROOT}/pref"
P_UNANSWERABLE = 0.3

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("requests==2.32.3").add_local_python_source("config"))
volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}
openrouter_secret = modal.Secret.from_name("openrouter-api")

SFT_SYS = ("You generate preference-training data for a small legal and financial assistant. "
           "For each prompt you write one excellent answer and one plausible but inferior answer.")
RAFT_SYS = ("You generate preference-training data for a retrieval-augmented legal and financial "
            "assistant that must answer only from provided context.")


def _openai(system, user, *, api_key, max_tokens, temperature):
    import random
    import time
    import requests

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": MODEL, "messages": [{"role": "system", "content": system},
                                         {"role": "user", "content": user}],
            "max_tokens": max_tokens, "temperature": temperature,
            "response_format": {"type": "json_object"}}
    last = ""
    for attempt in range(8):
        try:
            r = requests.post(LLM_URL, headers=headers, json=body, timeout=120)
            if r.status_code == 200:
                d = r.json(); u = d.get("usage", {})
                return (d["choices"][0]["message"]["content"],
                        {"in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0)})
            last = f"{r.status_code}: {r.text[:120]}"
            if r.status_code == 429:
                ra = r.headers.get("retry-after-ms")
                wait = float(ra) / 1000 if ra else float(r.headers.get("retry-after", 0)) or 2 ** attempt
                time.sleep(min(45, wait + random.uniform(0, 1.5))); continue
            if r.status_code in (500, 502, 503):
                time.sleep(2 ** attempt + random.uniform(0, 1)); continue
            break
        except Exception as e:
            last = str(e)[:120]; time.sleep(2 ** attempt + random.uniform(0, 1))
    print(f"  [teacher] failed: {last}", flush=True)
    return "", {"in": 0, "out": 0}


def _parse_json(text):
    import json
    import re
    if not text:
        return None
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


def _sft_user(q):
    return (f"Question: {q}\n\nWrite two answers:\n"
            '- "chosen": an EXCELLENT answer — accurate, well-structured, appropriately concise, genuinely helpful.\n'
            '- "rejected": a PLAUSIBLE BUT INFERIOR answer — vague, partially incorrect, rambling, or missing the key '
            "point. It should read like a weaker model wrote it, not obviously broken.\n\n"
            'Return JSON: {"chosen": "...", "rejected": "..."}')


def _raft_user(ctx, q, answerable):
    if answerable:
        return (f"Context documents:\n{ctx}\n\nQuestion: {q}\n\nWrite two answers that use ONLY the context:\n"
                '- "chosen": grounded and correct — quote the relevant text between ##begin_quote## and '
                '##end_quote##, then give the final answer on a new line starting "Final answer:".\n'
                '- "rejected": PLAUSIBLE BUT WORSE — ignores the context and answers from generic knowledge, '
                "cites the wrong detail, or fabricates a quote.\n\n"
                'Return JSON: {"chosen": "...", "rejected": "..."}')
    return (f"Context documents:\n{ctx}\n\nQuestion: {q}\n\nThe context does NOT contain the answer to this "
            "question. Write two answers:\n"
            '- "chosen": correctly DECLINES — states the answer is not in the provided context, e.g. '
            '"The provided context does not contain the information needed to answer this question.\\nFinal answer: '
            'Not stated in the provided context."\n'
            '- "rejected": FABRICATES a confident answer as if the context supported it, with a made-up quote.\n\n'
            'Return JSON: {"chosen": "...", "rejected": "..."}')


@app.function(image=image, volumes=VOLUMES, secrets=[openrouter_secret], timeout=60 * 60 * 3, cpu=2.0)
def build_pref(setting: str, n: int, seed: int = 11) -> dict:
    import json
    import os
    import random
    from concurrent.futures import ThreadPoolExecutor, as_completed

    key = os.environ["OPENROUTER_API_KEY"]
    volume.reload()
    rng = random.Random(seed)

    tasks = []
    if setting == "sft":
        seen = set()
        for line in open(SFT_CHAT, encoding="utf-8"):
            m = {x["role"]: x["content"] for x in json.loads(line)["messages"]}
            q = m["user"].strip()
            k = q.lower()[:80]
            if k in seen:
                continue
            seen.add(k)
            tasks.append({"prompt": q, "user": _sft_user(q)})
        rng.shuffle(tasks)
        tasks = tasks[:n]
    else:
        rows = [json.loads(l) for l in open(RAFT_TRAIN, encoding="utf-8")]
        rng.shuffle(rows)
        for i, r in enumerate(rows[:n]):
            answerable = rng.random() > P_UNANSWERABLE
            ctx = r["context"] if answerable else rows[(i + 7) % len(rows)]["context"]
            prompt = f"{ctx}\n\nQuestion: {r['question']}"
            tasks.append({"prompt": prompt, "context": ctx, "question": r["question"],
                          "answerable": answerable, "user": _raft_user(ctx, r["question"], answerable)})

    sys = SFT_SYS if setting == "sft" else RAFT_SYS
    print(f"[{setting}] generating {len(tasks)} preference pairs", flush=True)

    out, usage = [], {"in": 0, "out": 0}

    def work(t):
        txt, u = _openai(sys, t["user"], api_key=key, max_tokens=900, temperature=0.8)
        obj = _parse_json(txt)
        if not obj or not obj.get("chosen") or not obj.get("rejected"):
            return None, u
        rec = {"prompt": t["prompt"], "chosen": str(obj["chosen"]).strip(),
               "rejected": str(obj["rejected"]).strip()}
        if setting == "raft":
            rec.update({"answerable": t["answerable"]})
        return rec, u

    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for fut in as_completed([ex.submit(work, t) for t in tasks]):
            rec, u = fut.result()
            usage["in"] += u["in"]; usage["out"] += u["out"]
            if rec and rec["chosen"] != rec["rejected"]:
                out.append(rec)
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(tasks)} | kept {len(out)}", flush=True)

    os.makedirs(PREF_DIR, exist_ok=True)
    path = f"{PREF_DIR}/{setting}.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    volume.commit()
    cost = usage["in"] / 1e6 * RATE["in"] + usage["out"] / 1e6 * RATE["out"]
    meta = {"setting": setting, "kept": len(out), "requested": len(tasks),
            "tokens": usage, "cost_usd": round(cost, 2)}
    print(json.dumps(meta, indent=2), flush=True)
    return meta


@app.local_entrypoint()
def build(setting: str = "sft", n: int = 2500):
    build_pref.remote(setting=setting, n=n)
