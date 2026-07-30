"""Reward models for RLAIF, one per setting (sft / raft). Initialized from the 500M
legal SFT/RAFT model + a scalar head, trained on the AI preference pairs with the
Bradley-Terry pairwise loss (score(chosen) > score(rejected)). Compact (517M) so it is
fast to query during GRPO. Scores raw (prompt, response) text, so it can judge any
policy's output regardless of that policy's tokenizer.

    modal run train_reward.py::run --setting sft
    modal run train_reward.py::run --setting raft
"""

from __future__ import annotations

import modal

import config

app = modal.App("slm-reward-train")

PREF = {"sft": f"{config.DATA_ROOT}/pref/sft.jsonl", "raft": f"{config.DATA_ROOT}/pref/raft.jsonl"}
RM_BASE = {"sft": "jonam-ai/legal-slm-500m-sft", "raft": "jonam-ai/legal-slm-500m-raft"}
RM_OUT = f"{config.DATA_ROOT}/reward"
SFT_SYSTEM = "You are a knowledgeable legal and financial assistant. Answer accurately and concisely."
RAFT_SYSTEM = ("You are a legal and financial assistant. Use the numbered context documents "
               "to answer the question. Quote the text you rely on, then give the final answer. "
               "If the context does not contain the answer, say you cannot find it in the "
               "provided context instead of guessing.")
SLM_CHAT_TEMPLATE = (
    "{{ '<|bos|>' }}"
    "{% for m in messages %}"
    "{% if m['role'] == 'system' %}{{ '<|system|>' + m['content'] }}"
    "{% elif m['role'] == 'user' %}{{ '<|user|>' + m['content'] }}"
    "{% elif m['role'] == 'assistant' %}{{ '<|assistant|>' + m['content'] + '<|eos|>' }}"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}"
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "trl==0.12.2", "accelerate==1.1.1",
                 "datasets==3.1.0", "numpy==1.26.4")
    .add_local_python_source("config")
)
volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}
hf_secret = modal.Secret.from_name("huggingface-token")


@app.function(image=image, gpu="A100-40GB", volumes=VOLUMES, secrets=[hf_secret], timeout=60 * 60)
def train_rm(setting: str, epochs: float = 1.0, lr: float = 1e-5) -> dict:
    import json

    import torch
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from trl import RewardConfig, RewardTrainer

    volume.reload()
    system = RAFT_SYSTEM if setting == "raft" else SFT_SYSTEM
    rows = [json.loads(l) for l in open(PREF[setting], encoding="utf-8")]

    def conv(r):
        return {"chosen": [{"role": "system", "content": system},
                           {"role": "user", "content": r["prompt"]},
                           {"role": "assistant", "content": r["chosen"]}],
                "rejected": [{"role": "system", "content": system},
                             {"role": "user", "content": r["prompt"]},
                             {"role": "assistant", "content": r["rejected"]}]}

    ds = Dataset.from_list([conv(r) for r in rows])
    base = RM_BASE[setting]
    tok = AutoTokenizer.from_pretrained(base)
    tok.chat_template = SLM_CHAT_TEMPLATE
    tok.pad_token = "<|pad|>"
    print(f"[rm/{setting}] base={base} pairs={len(ds)}", flush=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=1, torch_dtype=torch.bfloat16, device_map={"": 0})
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False

    cfg = RewardConfig(
        output_dir="/tmp/rm", num_train_epochs=epochs, per_device_train_batch_size=4,
        gradient_accumulation_steps=4, learning_rate=lr, lr_scheduler_type="cosine",
        warmup_ratio=0.05, bf16=True, logging_steps=20, max_length=1024,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none", save_strategy="no", remove_unused_columns=False,
        center_rewards_coefficient=0.01,
    )
    trainer = RewardTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
    result = trainer.train()
    print(f"[rm/{setting}] loss {result.training_loss:.4f}", flush=True)

    out = f"{RM_OUT}/{setting}"
    trainer.model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    volume.commit()
    print(f"[rm/{setting}] saved -> {out}", flush=True)
    return {"setting": setting, "loss": round(result.training_loss, 4), "pairs": len(ds)}


@app.local_entrypoint()
def run(setting: str = "sft", epochs: float = 1.0):
    train_rm.remote(setting=setting, epochs=epochs)


@app.local_entrypoint()
def batch(epochs: float = 1.0):
    calls = [(s, train_rm.spawn(setting=s, epochs=epochs)) for s in ("sft", "raft")]
    for s, c in calls:
        print(s, "->", c.get())
