"""DPO (Direct Preference Optimization) on any of our SFT/RAFT models, using the
AI-labeled preference pairs. SLM family = full fine-tune; Gemma = QLoRA. Conversational
dataset format + the model's chat template keeps DPO in-distribution with training.

    modal run train_dpo.py::run --family slm  --setting sft  --base jonam-ai/legal-slm-125m-sft   --repo jonam-ai/legal-slm-125m-sft-dpo
    modal run train_dpo.py::run --family gemma --setting raft --base jonam-ai/gemma-2-2b-legal-raft --repo jonam-ai/gemma-2-2b-legal-raft-dpo
"""

from __future__ import annotations

import modal

import config

app = modal.App("slm-dpo-train")

PREF_SFT = f"{config.DATA_ROOT}/pref/sft.jsonl"
PREF_RAFT = f"{config.DATA_ROOT}/pref/raft.jsonl"
SFT_SYSTEM = "You are a knowledgeable legal and financial assistant. Answer accurately and concisely."
RAFT_SYSTEM = ("You are a legal and financial assistant. Use the numbered context documents "
               "to answer the question. Quote the text you rely on, then give the final answer. "
               "If the context does not contain the answer, say you cannot find it in the "
               "provided context instead of guessing.")

# Chat template for the SLM family (our role tokens); matches train_slm.py's encoding exactly.
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
    .pip_install("torch==2.5.1", "transformers==4.46.3", "trl==0.12.2", "peft==0.13.2",
                 "bitsandbytes==0.44.1", "accelerate==1.1.1", "datasets==3.1.0")
    .add_local_python_source("config")
)
volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}
hf_secret = modal.Secret.from_name("huggingface-token")


@app.function(image=image, gpu="A100-40GB", volumes=VOLUMES, secrets=[hf_secret], timeout=60 * 90)
def dpo(family: str, setting: str, base: str, repo: str, beta: float = 0.1,
        epochs: float = 1.0, lr: float = 0.0, pilot: bool = False) -> dict:
    import json

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    volume.reload()
    system = RAFT_SYSTEM if setting == "raft" else SFT_SYSTEM
    pref_path = PREF_RAFT if setting == "raft" else PREF_SFT
    rows = [json.loads(l) for l in open(pref_path, encoding="utf-8")]
    if pilot:
        rows = rows[:64]

    def prompt_msgs(r):
        if family == "gemma":   # Gemma has no system role — merge it into the user turn
            return [{"role": "user", "content": f"{system}\n\n{r['prompt']}"}]
        return [{"role": "system", "content": system}, {"role": "user", "content": r["prompt"]}]

    def conv(r):
        return {"prompt": prompt_msgs(r),
                "chosen": [{"role": "assistant", "content": r["chosen"]}],
                "rejected": [{"role": "assistant", "content": r["rejected"]}]}

    ds = Dataset.from_list([conv(r) for r in rows])
    print(f"[dpo/{family}/{setting}] base={base} pairs={len(ds)}", flush=True)

    tok = AutoTokenizer.from_pretrained(base)
    if family == "slm":
        tok.chat_template = SLM_CHAT_TEMPLATE
        if tok.pad_token is None:
            tok.pad_token = "<|pad|>"

    default_lr = 5e-5 if family == "gemma" else 5e-6
    lr = lr or default_lr
    peft_config = None
    if family == "gemma":
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb,
                                                     torch_dtype=torch.bfloat16, device_map={"": 0},
                                                     attn_implementation="eager")
        model.config.use_cache = False
        peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                                 task_type="CAUSAL_LM",
                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                 "gate_proj", "up_proj", "down_proj"])
    else:
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16,
                                                     device_map={"": 0})
        model.config.use_cache = False

    cfg = DPOConfig(
        output_dir="/tmp/dpo", beta=beta, num_train_epochs=epochs,
        per_device_train_batch_size=2, gradient_accumulation_steps=8, learning_rate=lr,
        lr_scheduler_type="cosine", warmup_ratio=0.05, bf16=True, logging_steps=10,
        max_length=1024, max_prompt_length=768, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, report_to="none",
        save_strategy="no", optim="paged_adamw_8bit" if family == "gemma" else "adamw_torch",
        remove_unused_columns=False,
    )
    trainer = DPOTrainer(model=model, ref_model=None, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=peft_config)
    result = trainer.train()
    print(f"[dpo] loss {result.training_loss:.4f}", flush=True)

    # ---- save + push ----
    out = "/tmp/dpo_out"
    if family == "gemma":
        adapter = "/tmp/dpo_adapter"
        trainer.model.save_pretrained(adapter)
        del model, trainer
        torch.cuda.empty_cache()
        base_bf16 = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16,
                                                         device_map={"": 0}, attn_implementation="eager")
        merged = PeftModel.from_pretrained(base_bf16, adapter).merge_and_unload()
        merged.save_pretrained(out, safe_serialization=True)
    else:
        trainer.model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    if not pilot:
        import os

        from huggingface_hub import HfApi
        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(repo, exist_ok=True, repo_type="model")
        api.upload_folder(folder_path=out, repo_id=repo, repo_type="model",
                          commit_message=f"DPO from {base}")
        print(f"pushed -> https://huggingface.co/{repo}", flush=True)
    return {"repo": repo, "loss": round(result.training_loss, 4), "pairs": len(ds)}


DPO_JOBS = [
    ("slm", "sft", "jonam-ai/legal-slm-125m-sft", "jonam-ai/legal-slm-125m-sft-dpo"),
    ("slm", "sft", "jonam-ai/legal-slm-500m-sft", "jonam-ai/legal-slm-500m-sft-dpo"),
    ("gemma", "sft", "jonam-ai/gemma-2-2b-legal-sft", "jonam-ai/gemma-2-2b-legal-sft-dpo"),
    ("slm", "raft", "jonam-ai/legal-slm-125m-raft", "jonam-ai/legal-slm-125m-raft-dpo"),
    ("slm", "raft", "jonam-ai/legal-slm-500m-raft", "jonam-ai/legal-slm-500m-raft-dpo"),
    ("gemma", "raft", "jonam-ai/gemma-2-2b-legal-raft", "jonam-ai/gemma-2-2b-legal-raft-dpo"),
]


@app.local_entrypoint()
def run(family: str = "slm", setting: str = "sft", base: str = "jonam-ai/legal-slm-125m-sft",
        repo: str = "jonam-ai/legal-slm-125m-sft-dpo", beta: float = 0.1, epochs: float = 1.0,
        pilot: bool = False):
    dpo.remote(family=family, setting=setting, base=base, repo=repo, beta=beta,
               epochs=epochs, pilot=pilot)


@app.local_entrypoint()
def batch(epochs: float = 2.0):
    calls = [(r, dpo.spawn(family=f, setting=s, base=b, repo=r, epochs=epochs))
             for f, s, b, r in DPO_JOBS]
    for repo, c in calls:
        try:
            print(repo, "->", c.get())
        except Exception as e:
            print(repo, "FAILED", str(e)[:200])
