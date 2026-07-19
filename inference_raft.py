"""Live RAFT endpoint: answer a question grounded in user-provided context.

Scale-to-zero CPU service; streams the grounded answer token-by-token (SSE).
Takes {context, question} and formats the RAFT prompt the model was trained on.

    modal deploy inference_raft.py
"""

# NOTE: no `from __future__ import annotations` (breaks FastAPI hint resolution).

import modal

app = modal.App("legal-slm-125m-raft-inference")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "fastapi[standard]==0.115.4")
)

MODEL_ID = "jonam-ai/legal-slm-125m-raft"
RAFT_SYSTEM = ("You are a legal and financial assistant. Use the numbered context "
               "documents to answer the question. Quote the text you rely on, then "
               "give the final answer. If the context does not contain the answer, "
               "say you cannot find it in the provided context instead of guessing.")

hf_cache = modal.Volume.from_name("legal-slm-hf-cache", create_if_missing=True)


@app.cls(image=image, cpu=4.0, memory=8192, volumes={"/cache": hf_cache},
         scaledown_window=300, min_containers=0)
class RAFT:
    @modal.enter()
    def load(self):
        import os

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        os.environ["HF_HOME"] = "/cache"
        torch.set_num_threads(4)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir="/cache")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, cache_dir="/cache", torch_dtype=torch.float32).eval()
        sid = self.tok.convert_tokens_to_ids
        self.bos, self.eos = sid("<|bos|>"), sid("<|eos|>")
        self.sys_t, self.user_t, self.asst_t = sid("<|system|>"), sid("<|user|>"), sid("<|assistant|>")
        self.sys_ids = self.tok(RAFT_SYSTEM, add_special_tokens=False)["input_ids"]
        hf_cache.commit()

    def _prompt_ids(self, context: str, question: str):
        context = context.strip()
        if not context.lower().startswith("context:"):
            context = "Context:\n" + context
        body = f"{context}\n\nQuestion: {question.strip()}"
        u = self.tok(body, add_special_tokens=False)["input_ids"][:900]   # leave room for the answer
        return [self.bos, self.sys_t] + self.sys_ids + [self.user_t] + u + [self.asst_t]

    @modal.asgi_app()
    def web(self):
        import json
        import threading

        from fastapi import FastAPI, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import StreamingResponse
        from transformers import TextIteratorStreamer

        globals()["Request"] = Request
        api = FastAPI(title="legal-slm-125m-raft")
        api.add_middleware(CORSMiddleware, allow_origins=["*"],
                           allow_methods=["*"], allow_headers=["*"])

        @api.get("/health")
        def health():
            return {"ok": True, "model": MODEL_ID}

        @api.post("/raft")
        async def raft(req: Request):
            body = await req.json()
            context = body.get("context") or ""
            question = (body.get("question") or "").strip() or "What does the context say?"
            max_new = max(16, min(256, int(body.get("max_new_tokens", 160))))
            temperature = max(0.1, min(1.5, float(body.get("temperature", 0.6))))

            ids = self.torch.tensor([self._prompt_ids(context, question)])
            streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
            kwargs = dict(input_ids=ids, max_new_tokens=max_new, do_sample=True,
                          temperature=temperature, top_k=50, top_p=0.9,
                          eos_token_id=self.eos, pad_token_id=self.eos, streamer=streamer)

            def event_stream():
                thread = threading.Thread(target=self.model.generate, kwargs=kwargs)
                thread.start()
                for text in streamer:
                    if text:
                        yield f"data: {json.dumps({'token': text})}\n\n"
                thread.join()
                yield f"data: {json.dumps({'done': True})}\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        return api
