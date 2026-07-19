"""QLoRA fine-tuning of Gemma 2 2B-it on our SFT (Q&A) and RAFT (context) datasets.

Mirrors our SLM's arc on a real 2.6B pretrained model, but with QLoRA (4-bit NF4
base + LoRA adapters) instead of full fine-tuning. Reuses the *text* of the datasets
we already built; Gemma's own tokenizer handles tokenization via TRL.

    modal run gemma_finetune.py::run --stage sft --pilot        # cheap L4 sanity
    modal run gemma_finetune.py::run --stage sft                # full SFT
    modal run gemma_finetune.py::run --stage raft               # RAFT on top of SFT
"""

from __future__ import annotations

import modal

import config

app = modal.App("gemma-2b-finetune")

BASE_IT = "google/gemma-2-2b-it"
SFT_CHAT = f"{config.DATA_ROOT}/sft/dataset/chat.jsonl"
RAFT_TRAIN = f"{config.DATA_ROOT}/raft/dataset/raft_text_train.jsonl"
GEMMA_DIR = f"{config.DATA_ROOT}/gemma"
SFT_SYSTEM = "You are a knowledgeable legal and financial assistant. Answer accurately and concisely."
RAFT_SYSTEM = ("You are a legal and financial assistant. Use the numbered context documents "
               "to answer the question. Quote the text you rely on, then give the final answer.")

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1", "transformers==4.46.3", "trl==0.12.2", "peft==0.13.2",
        "bitsandbytes==0.44.1", "accelerate==1.1.1", "datasets==3.1.0",
        "sentencepiece==0.2.0", "huggingface_hub==0.26.2",
    )
    .add_local_python_source("config")
)
volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}
hf_secret = modal.Secret.from_name("huggingface-token")


@app.function(image=gpu_image, gpu="A100-40GB", volumes=VOLUMES, secrets=[hf_secret], timeout=60 * 60 * 2)
def finetune(stage: str = "sft", epochs: float = 3.0, lr: float = 2e-4,
             pilot: bool = False, limit: int = 64) -> dict:
    import json
    import time

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

    volume.reload()

    if stage == "sft":
        base = BASE_IT
        rows = [json.loads(l) for l in open(SFT_CHAT, encoding="utf-8")]
        def to_msgs(r):
            m = {x["role"]: x["content"] for x in r["messages"]}
            user = f"{m.get('system', SFT_SYSTEM)}\n\n{m['user']}".strip()
            return [{"role": "user", "content": user}, {"role": "assistant", "content": m["assistant"]}]
        out = f"{GEMMA_DIR}/sft"
    elif stage == "raft":
        base = f"{GEMMA_DIR}/sft/merged"   # continue from the merged Gemma SFT model
        rows = [json.loads(l) for l in open(RAFT_TRAIN, encoding="utf-8")]
        def to_msgs(r):
            user = f"{RAFT_SYSTEM}\n\n{r['context']}\n\nQuestion: {r['question']}"
            return [{"role": "user", "content": user}, {"role": "assistant", "content": r["answer"]}]
        out = f"{GEMMA_DIR}/raft"
    else:
        raise ValueError(stage)

    if pilot:
        rows = rows[:limit]
        epochs = 1.0
    data = [to_msgs(r) for r in rows]

    tok = AutoTokenizer.from_pretrained(base)
    texts = [tok.apply_chat_template(m, tokenize=False) for m in data]
    ds = Dataset.from_dict({"text": texts})
    print(f"[{stage}] base={base} examples={len(ds)} epochs={epochs}", flush=True)
    print("sample:\n" + texts[0][:400], flush=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="eager")   # eager: Gemma2 soft-capping
    model.config.use_cache = False

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    collator = DataCollatorForCompletionOnlyLM(response_template="<start_of_turn>model\n", tokenizer=tok)
    cfg = SFTConfig(
        output_dir="/tmp/out", num_train_epochs=epochs, per_device_train_batch_size=2,
        gradient_accumulation_steps=8, learning_rate=lr, bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10, max_seq_length=1024, packing=False, lr_scheduler_type="cosine",
        warmup_ratio=0.03, report_to="none", save_strategy="no", optim="paged_adamw_8bit",
        dataset_text_field="text",
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         data_collator=collator, peft_config=lora)
    trainer.model.print_trainable_parameters()
    n_train = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in trainer.model.parameters())

    t0 = time.time()
    result = trainer.train()
    dt = time.time() - t0
    train_tokens = int(result.metrics.get("train_tokens", 0)) if hasattr(result, "metrics") else 0
    print(f"[{stage}] trained {trainer.state.global_step} steps in {dt:.0f}s | loss {result.training_loss:.4f}", flush=True)

    # ---- merge LoRA into a bf16 base and save the standalone model ----
    adapter_dir = f"{out}/adapter"
    trainer.model.save_pretrained(adapter_dir)
    del model, trainer
    torch.cuda.empty_cache()
    base_bf16 = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16,
                                                     device_map={"": 0}, attn_implementation="eager")
    merged = PeftModel.from_pretrained(base_bf16, adapter_dir).merge_and_unload()
    merged.save_pretrained(f"{out}/merged", safe_serialization=True)
    tok.save_pretrained(f"{out}/merged")
    volume.commit()
    print(f"[{stage}] saved merged model -> {out}/merged", flush=True)

    meta = {"stage": stage, "base": base, "examples": len(ds), "epochs": epochs,
            "trainable_params": n_train, "total_params": n_total,
            "steps": trainer.state.global_step if False else result.global_step if hasattr(result, "global_step") else None,
            "train_loss": round(result.training_loss, 4), "seconds": round(dt)}
    with open(f"{out}/meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    volume.commit()
    print(json.dumps(meta, indent=2, default=str), flush=True)
    return meta


@app.function(image=gpu_image, volumes=VOLUMES, secrets=[hf_secret], timeout=60 * 30)
def push_to_hf(stage: str, repo: str) -> str:
    """Upload the merged Gemma model from the volume to a HF repo."""
    import os

    from huggingface_hub import HfApi

    volume.reload()
    src = f"{GEMMA_DIR}/{stage}/merged"
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=src, repo_id=repo, repo_type="model",
                      commit_message=f"Gemma 2 2B legal {stage.upper()} (QLoRA, merged)")
    print(f"pushed {src} -> https://huggingface.co/{repo}", flush=True)
    return repo


@app.local_entrypoint()
def run(stage: str = "sft", epochs: float = 3.0, lr: float = 2e-4, pilot: bool = False):
    finetune.remote(stage=stage, epochs=epochs, lr=lr, pilot=pilot)


@app.function(image=gpu_image, volumes=VOLUMES, secrets=[hf_secret], timeout=60 * 20)
def token_stats() -> dict:
    """Count training tokens per phase for both models (for the UI comparison)."""
    import json

    from transformers import AutoTokenizer

    volume.reload()
    slm_tok = AutoTokenizer.from_pretrained("thesreedath/slm-125m-base")
    gem_tok = AutoTokenizer.from_pretrained(BASE_IT)

    sft_rows = [json.loads(l) for l in open(SFT_CHAT, encoding="utf-8")]
    raft_rows = [json.loads(l) for l in open(RAFT_TRAIN, encoding="utf-8")]

    def sft_text(r):
        m = {x["role"]: x["content"] for x in r["messages"]}
        return f"{m.get('system', SFT_SYSTEM)}\n\n{m['user']}\n\n{m['assistant']}"

    def raft_text(r):
        return f"{RAFT_SYSTEM}\n\n{r['context']}\n\nQuestion: {r['question']}\n\n{r['answer']}"

    sft_txt = [sft_text(r) for r in sft_rows]
    raft_txt = [raft_text(r) for r in raft_rows]

    def total(tok, texts):
        return sum(len(x) for x in tok(texts, add_special_tokens=True)["input_ids"])

    out = {
        "slm_sft_per_epoch": total(slm_tok, sft_txt),
        "slm_raft_per_epoch": total(slm_tok, raft_txt),
        "gemma_sft_per_epoch": total(gem_tok, sft_txt),
        "gemma_raft_per_epoch": total(gem_tok, raft_txt),
        "n_sft": len(sft_txt), "n_raft": len(raft_txt),
    }
    print(json.dumps(out, indent=2), flush=True)
    return out


@app.local_entrypoint()
def push(stage: str = "sft", repo: str = ""):
    repo = repo or f"jonam-ai/gemma-2-2b-legal-{stage}"
    push_to_hf.remote(stage=stage, repo=repo)


@app.local_entrypoint()
def stats():
    token_stats.remote()
