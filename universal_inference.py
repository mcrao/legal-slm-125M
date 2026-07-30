"""One scale-to-zero GPU endpoint that can serve ANY of our models by id — bases (text
completion), SFT/DPO/RLAIF (chat), and RAFT (grounded) — so the site can offer live model
selectors everywhere. Models are loaded on demand and LRU-cached in GPU memory.

    modal deploy universal_inference.py

POST /generate  {model_id, prompt}                  -> completion (base models)
POST /chat      {model_id, message}                 -> instruct reply (SFT/DPO/RLAIF)
POST /raft      {model_id, context, question}       -> grounded answer (RAFT/DPO/RLAIF-raft)
All stream SSE: data: {"token": "..."}\\n\\n
"""

# NOTE: no `from __future__ import annotations` (FastAPI hint resolution).

import modal

app = modal.App("legal-slm-universal-inference")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "accelerate==1.1.1",
                 "fastapi[standard]==0.115.4")
)

# Whitelist of servable models (blocks arbitrary downloads).
ALLOWED = {
    # bases (completion)
    "jonam-ai/slm-125m-base", "thesreedath/slm-125m-base", "thesreedath/slm-500m-base",
    "google/gemma-2-2b-it",
    # sft / raft
    "jonam-ai/legal-slm-125m-sft", "jonam-ai/legal-slm-125m-raft",
    "jonam-ai/legal-slm-500m-sft", "jonam-ai/legal-slm-500m-raft",
    "jonam-ai/gemma-2-2b-legal-sft", "jonam-ai/gemma-2-2b-legal-raft",
    # dpo
    "jonam-ai/legal-slm-125m-sft-dpo", "jonam-ai/legal-slm-125m-raft-dpo",
    "jonam-ai/legal-slm-500m-sft-dpo", "jonam-ai/legal-slm-500m-raft-dpo",
    "jonam-ai/gemma-2-2b-legal-sft-dpo", "jonam-ai/gemma-2-2b-legal-raft-dpo",
    # rlaif
    "jonam-ai/legal-slm-125m-sft-rlaif", "jonam-ai/legal-slm-125m-raft-rlaif",
    "jonam-ai/legal-slm-500m-sft-rlaif", "jonam-ai/legal-slm-500m-raft-rlaif",
    "jonam-ai/gemma-2-2b-legal-sft-rlaif", "jonam-ai/gemma-2-2b-legal-raft-rlaif",
}
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
hf_cache = modal.Volume.from_name("legal-slm-hf-cache", create_if_missing=True)


@app.cls(image=image, gpu="L4", volumes={"/cache": hf_cache},
         scaledown_window=300, min_containers=0, timeout=60 * 10)
class Universal:
    @modal.enter()
    def setup(self):
        import collections
        import os

        import torch

        os.environ["HF_HOME"] = "/cache"
        self.torch = torch
        self.cache = collections.OrderedDict()   # model_id -> (model, tok, family)
        self.MAX = 3

    def _get(self, model_id):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if model_id not in ALLOWED:
            raise ValueError("model not allowed")
        if model_id in self.cache:
            self.cache.move_to_end(model_id)
            return self.cache[model_id]
        family = "gemma" if "gemma" in model_id else "slm"
        tok = AutoTokenizer.from_pretrained(model_id, cache_dir="/cache")
        if family == "slm":
            tok.chat_template = SLM_CHAT_TEMPLATE
            if tok.pad_token is None:
                tok.pad_token = "<|pad|>"
        kw = {"attn_implementation": "eager"} if family == "gemma" else {}
        model = AutoModelForCausalLM.from_pretrained(
            model_id, cache_dir="/cache", torch_dtype=torch.bfloat16, device_map={"": 0}, **kw).eval()
        while len(self.cache) >= self.MAX:
            old_id, (old_m, _, _) = self.cache.popitem(last=False)
            del old_m
            torch.cuda.empty_cache()
        self.cache[model_id] = (model, tok, family)
        hf_cache.commit()
        return self.cache[model_id]

    def _eos(self, tok, family):
        if family == "gemma":
            return [tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_turn>")]
        return [tok.convert_tokens_to_ids("<|eos|>")]

    def _stream(self, model, tok, ids, family, max_new, temperature):
        import json
        import threading

        from fastapi.responses import StreamingResponse
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        pad = tok.pad_token_id if tok.pad_token_id is not None else self._eos(tok, family)[0]
        kwargs = dict(input_ids=ids, max_new_tokens=max_new, do_sample=temperature > 0,
                      temperature=max(temperature, 0.01), top_k=50, top_p=0.9,
                      eos_token_id=self._eos(tok, family), pad_token_id=pad, streamer=streamer)

        def gen():
            t = threading.Thread(target=model.generate, kwargs=kwargs)
            t.start()
            for text in streamer:
                if text:
                    yield f"data: {json.dumps({'token': text})}\n\n"
            t.join()
            yield f"data: {json.dumps({'done': True})}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    def _chat_ids(self, tok, family, user_text):
        text = tok.apply_chat_template([{"role": "user", "content": user_text}],
                                       tokenize=False, add_generation_prompt=True)
        add_special = family == "slm" and "<|bos|>" not in text
        return tok(text, return_tensors="pt", add_special_tokens=add_special).input_ids.to("cuda")

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, Request
        from fastapi.middleware.cors import CORSMiddleware

        globals()["Request"] = Request
        api = FastAPI(title="legal-slm-universal")
        api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

        @api.get("/health")
        def health():
            return {"ok": True, "cached": list(self.cache.keys())}

        @api.post("/generate")
        async def generate(req: Request):
            b = await req.json()
            model, tok, family = self._get((b.get("model_id") or "jonam-ai/slm-125m-base").strip())
            prompt = (b.get("prompt") or "The plaintiff shall bear the burden of").strip()
            ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
            return self._stream(model, tok, ids, family,
                                max(16, min(256, int(b.get("max_new_tokens", 160)))),
                                max(0.0, min(1.5, float(b.get("temperature", 0.8)))))

        @api.post("/chat")
        async def chat(req: Request):
            b = await req.json()
            model, tok, family = self._get((b.get("model_id") or "jonam-ai/legal-slm-125m-sft").strip())
            msg = (b.get("message") or "What is a Form 10-K?").strip()
            ids = self._chat_ids(tok, family, f"{SFT_SYSTEM}\n\n{msg}")
            return self._stream(model, tok, ids, family,
                                max(16, min(256, int(b.get("max_new_tokens", 180)))),
                                max(0.0, min(1.5, float(b.get("temperature", 0.7)))))

        @api.post("/raft")
        async def raft(req: Request):
            b = await req.json()
            model, tok, family = self._get((b.get("model_id") or "jonam-ai/legal-slm-125m-raft").strip())
            context = (b.get("context") or "").strip()
            question = (b.get("question") or "What does the context say?").strip()
            if not context.lower().startswith("context:"):
                context = "Context:\n" + context
            ids = self._chat_ids(tok, family, f"{RAFT_SYSTEM}\n\n{context}\n\nQuestion: {question}")
            return self._stream(model, tok, ids, family,
                                max(16, min(256, int(b.get("max_new_tokens", 200)))),
                                max(0.0, min(1.5, float(b.get("temperature", 0.5)))))

        return api
