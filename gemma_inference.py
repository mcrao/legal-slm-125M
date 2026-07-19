"""Live inference for the Gemma 2 2B legal models (SFT + RAFT), for side-by-side
comparison with our 125M SLM on the Vercel demo.

Gemma 2B needs a GPU, so this uses a scale-to-zero L4 (idle $0, a few cents per test
session). Serves /chat (SFT model) and /raft (RAFT model) with SSE streaming.

    modal deploy gemma_inference.py
"""

# NOTE: no `from __future__ import annotations` (FastAPI hint resolution).

import modal

app = modal.App("gemma-2b-legal-inference")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "accelerate==1.1.1",
                 "fastapi[standard]==0.115.4")
)

SFT_REPO = "jonam-ai/gemma-2-2b-legal-sft"
RAFT_REPO = "jonam-ai/gemma-2-2b-legal-raft"
SFT_SYSTEM = "You are a knowledgeable legal and financial assistant. Answer accurately and concisely."
RAFT_SYSTEM = ("You are a legal and financial assistant. Use the numbered context documents "
               "to answer the question. Quote the text you rely on, then give the final answer.")

hf_cache = modal.Volume.from_name("legal-slm-hf-cache", create_if_missing=True)


@app.cls(image=image, gpu="L4", volumes={"/cache": hf_cache},
         scaledown_window=240, min_containers=0)
class Gemma:
    @modal.enter()
    def load(self):
        import os

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        os.environ["HF_HOME"] = "/cache"
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(SFT_REPO, cache_dir="/cache")
        self.sft = AutoModelForCausalLM.from_pretrained(
            SFT_REPO, cache_dir="/cache", torch_dtype=torch.bfloat16,
            device_map={"": 0}, attn_implementation="eager").eval()
        self.raft = AutoModelForCausalLM.from_pretrained(
            RAFT_REPO, cache_dir="/cache", torch_dtype=torch.bfloat16,
            device_map={"": 0}, attn_implementation="eager").eval()
        hf_cache.commit()

    def _ids(self, user_text):
        msgs = [{"role": "user", "content": user_text}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")

    def _stream(self, model, ids, max_new, temperature):
        import json
        import threading

        from fastapi.responses import StreamingResponse
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(input_ids=ids, max_new_tokens=max_new, do_sample=True,
                      temperature=temperature, top_k=50, top_p=0.9, streamer=streamer)

        def gen():
            thread = threading.Thread(target=model.generate, kwargs=kwargs)
            thread.start()
            for text in streamer:
                if text:
                    yield f"data: {json.dumps({'token': text})}\n\n"
            thread.join()
            yield f"data: {json.dumps({'done': True})}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, Request
        from fastapi.middleware.cors import CORSMiddleware

        globals()["Request"] = Request
        api = FastAPI(title="gemma-2b-legal")
        api.add_middleware(CORSMiddleware, allow_origins=["*"],
                           allow_methods=["*"], allow_headers=["*"])

        @api.get("/health")
        def health():
            return {"ok": True, "models": [SFT_REPO, RAFT_REPO]}

        @api.post("/chat")
        async def chat(req: Request):
            b = await req.json()
            msg = (b.get("message") or "What is a Form 10-K?").strip()
            ids = self._ids(f"{SFT_SYSTEM}\n\n{msg}")
            return self._stream(self.sft, ids, max(16, min(256, int(b.get("max_new_tokens", 160)))),
                                max(0.1, min(1.5, float(b.get("temperature", 0.7)))))

        @api.post("/raft")
        async def raft(req: Request):
            b = await req.json()
            context = (b.get("context") or "").strip()
            question = (b.get("question") or "What does the context say?").strip()
            if not context.lower().startswith("context:"):
                context = "Context:\n" + context
            ids = self._ids(f"{RAFT_SYSTEM}\n\n{context}\n\nQuestion: {question}")
            return self._stream(self.raft, ids, max(16, min(256, int(b.get("max_new_tokens", 180)))),
                                max(0.1, min(1.5, float(b.get("temperature", 0.5)))))

        return api
