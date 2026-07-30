"""RLAIF via GRPO: optimize a policy against a trained reward model with group-relative
policy optimization (DeepSeek-style, on-policy). SLM = full fine-tune; Gemma = QLoRA.
The reward comes from the setting's reward model (train_reward.py), which scores each
generated (prompt, completion).

    modal run train_grpo.py::run --family slm --setting sft --base jonam-ai/legal-slm-125m-sft --repo jonam-ai/legal-slm-125m-sft-rlaif
"""

from __future__ import annotations

import modal

import config

app = modal.App("slm-grpo-train")

PREF = {"sft": f"{config.DATA_ROOT}/pref/sft.jsonl", "raft": f"{config.DATA_ROOT}/pref/raft.jsonl"}
RM_DIR = f"{config.DATA_ROOT}/reward"
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
    .pip_install("torch==2.5.1", "transformers==4.48.3", "trl==0.16.1", "peft==0.14.0",
                 "accelerate==1.4.0", "datasets==3.2.0", "bitsandbytes==0.45.3")
    .add_local_python_source("config")
)
volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}
hf_secret = modal.Secret.from_name("huggingface-token")


@app.function(image=image, gpu="A100-40GB", volumes=VOLUMES, secrets=[hf_secret], timeout=60 * 240)
def grpo(family: str, setting: str, base: str, repo: str, n_prompts: int = 800,
         epochs: float = 1.0, lr: float = 0.0, pilot: bool = False) -> dict:
    import json

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification,
                              AutoTokenizer, BitsAndBytesConfig)
    from trl import GRPOConfig, GRPOTrainer

    volume.reload()
    system = RAFT_SYSTEM if setting == "raft" else SFT_SYSTEM

    # ---- reward model (scores raw prompt+response text) ----
    rm_dir = f"{RM_DIR}/{setting}"
    rm_tok = AutoTokenizer.from_pretrained(rm_dir)
    rm_tok.chat_template = SLM_CHAT_TEMPLATE
    rm = AutoModelForSequenceClassification.from_pretrained(
        rm_dir, num_labels=1, torch_dtype=torch.bfloat16).to("cuda").eval()
    rm.config.pad_token_id = rm_tok.convert_tokens_to_ids("<|pad|>")

    @torch.no_grad()
    def reward_fn(prompts, completions, **kwargs):
        scores = []
        for p, c in zip(prompts, completions):
            user = p[-1]["content"] if isinstance(p, list) else str(p)
            ans = c[-1]["content"] if isinstance(c, list) else str(c)
            text = rm_tok.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user},
                 {"role": "assistant", "content": ans}], tokenize=False)
            enc = rm_tok(text, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
            scores.append(float(rm(input_ids=enc["input_ids"],
                                   attention_mask=enc["attention_mask"]).logits[0, 0]))
        return scores

    # ---- prompts ----
    rows = [json.loads(l) for l in open(PREF[setting], encoding="utf-8")]
    rows = rows[:(32 if pilot else n_prompts)]

    def prompt_msgs(r):
        if family == "gemma":   # Gemma has no system role — merge it into the user turn
            return [{"role": "user", "content": f"{system}\n\n{r['prompt']}"}]
        return [{"role": "system", "content": system}, {"role": "user", "content": r["prompt"]}]

    ds = Dataset.from_list([{"prompt": prompt_msgs(r)} for r in rows])
    print(f"[grpo/{family}/{setting}] base={base} prompts={len(ds)}", flush=True)

    # ---- policy ----
    tok = AutoTokenizer.from_pretrained(base)
    peft_config = None
    if family == "slm":
        tok.chat_template = SLM_CHAT_TEMPLATE
        tok.pad_token = "<|pad|>"
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    else:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb,
                                                     torch_dtype=torch.bfloat16,
                                                     attn_implementation="eager")
        peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                                 task_type="CAUSAL_LM",
                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                 "gate_proj", "up_proj", "down_proj"])

    default_lr = 1e-5 if family == "slm" else 5e-5
    cfg = GRPOConfig(
        output_dir="/tmp/grpo", num_train_epochs=epochs, learning_rate=lr or default_lr,
        per_device_train_batch_size=8, gradient_accumulation_steps=4, num_generations=8,
        max_prompt_length=640, max_completion_length=160, temperature=0.9, beta=0.04,
        bf16=True, gradient_checkpointing=True, logging_steps=5, report_to="none",
        save_strategy="no", lr_scheduler_type="cosine", warmup_ratio=0.05,
    )
    trainer = GRPOTrainer(model=model, reward_funcs=reward_fn, args=cfg,
                          train_dataset=ds, processing_class=tok, peft_config=peft_config)
    result = trainer.train()
    print(f"[grpo] loss {result.training_loss:.4f}", flush=True)

    out = "/tmp/grpo_out"
    if family == "gemma":
        adapter = "/tmp/grpo_adapter"
        trainer.save_model(adapter)
        del model, trainer, rm
        torch.cuda.empty_cache()
        base_bf16 = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16,
                                                         attn_implementation="eager").to("cuda")
        merged = PeftModel.from_pretrained(base_bf16, adapter).merge_and_unload()
        merged.save_pretrained(out, safe_serialization=True)
    else:
        trainer.save_model(out)
    tok.save_pretrained(out)
    if not pilot:
        import os

        from huggingface_hub import HfApi
        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(repo, exist_ok=True, repo_type="model")
        api.upload_folder(folder_path=out, repo_id=repo, repo_type="model",
                          commit_message=f"RLAIF (GRPO) from {base}")
        print(f"pushed -> https://huggingface.co/{repo}", flush=True)
    return {"repo": repo, "loss": round(result.training_loss, 4), "prompts": len(ds)}


GRPO_JOBS = [
    ("slm", "sft", "jonam-ai/legal-slm-125m-sft", "jonam-ai/legal-slm-125m-sft-rlaif"),
    ("slm", "sft", "jonam-ai/legal-slm-500m-sft", "jonam-ai/legal-slm-500m-sft-rlaif"),
    ("gemma", "sft", "jonam-ai/gemma-2-2b-legal-sft", "jonam-ai/gemma-2-2b-legal-sft-rlaif"),
    ("slm", "raft", "jonam-ai/legal-slm-125m-raft", "jonam-ai/legal-slm-125m-raft-rlaif"),
    ("slm", "raft", "jonam-ai/legal-slm-500m-raft", "jonam-ai/legal-slm-500m-raft-rlaif"),
    ("gemma", "raft", "jonam-ai/gemma-2-2b-legal-raft", "jonam-ai/gemma-2-2b-legal-raft-rlaif"),
]


@app.local_entrypoint()
def run(family: str = "slm", setting: str = "sft", base: str = "jonam-ai/legal-slm-125m-sft",
        repo: str = "jonam-ai/legal-slm-125m-sft-rlaif", epochs: float = 1.0,
        n_prompts: int = 800, pilot: bool = False):
    grpo.remote(family=family, setting=setting, base=base, repo=repo, epochs=epochs,
                n_prompts=n_prompts, pilot=pilot)


@app.local_entrypoint()
def batch(epochs: float = 1.0):
    calls = [(r, grpo.spawn(family=f, setting=s, base=b, repo=r, epochs=epochs))
             for f, s, b, r in GRPO_JOBS]
    for repo, c in calls:
        try:
            print(repo, "->", c.get())
        except Exception as e:
            print(repo, "FAILED", str(e)[:200])


# The three that timed out at 90 min — smaller prompt set, 4h timeout.
RERUN = [
    ("slm", "raft", "jonam-ai/legal-slm-500m-raft", "jonam-ai/legal-slm-500m-raft-rlaif"),
    ("gemma", "sft", "jonam-ai/gemma-2-2b-legal-sft", "jonam-ai/gemma-2-2b-legal-sft-rlaif"),
    ("gemma", "raft", "jonam-ai/gemma-2-2b-legal-raft", "jonam-ai/gemma-2-2b-legal-raft-rlaif"),
]


@app.local_entrypoint()
def rerun(epochs: float = 1.0, n_prompts: int = 400):
    calls = [(r, grpo.spawn(family=f, setting=s, base=b, repo=r, epochs=epochs, n_prompts=n_prompts))
             for f, s, b, r in RERUN]
    for repo, c in calls:
        try:
            print(repo, "->", c.get())
        except Exception as e:
            print(repo, "FAILED", str(e)[:200])
