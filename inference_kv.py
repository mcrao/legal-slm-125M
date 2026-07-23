"""KV-cache benchmark for the 125M SLM: same greedy generation with and without the
key/value cache, at a chosen batch size. Without the cache the model recomputes the full
attention over the whole growing sequence every step (O(n^2)); with it, each step is O(n).
Raising the batch size (more concurrent "users") amplifies the gap.

Scale-to-zero L4 GPU. Serves POST /bench -> JSON.

    modal deploy inference_kv.py
"""

# NOTE: no `from __future__ import annotations` (FastAPI hint resolution).

import modal

app = modal.App("slm-125m-kvcache")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "accelerate==1.1.1",
                 "fastapi[standard]==0.115.4")
)

MODEL_ID = "jonam-ai/slm-125m-base"
hf_cache = modal.Volume.from_name("legal-slm-hf-cache", create_if_missing=True)


@app.cls(image=image, gpu="L4", volumes={"/cache": hf_cache},
         scaledown_window=240, min_containers=0)
class KV:
    @modal.enter()
    def load(self):
        import os

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        os.environ["HF_HOME"] = "/cache"
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir="/cache")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, cache_dir="/cache", torch_dtype=torch.bfloat16,
            device_map={"": 0}).eval()
        # warm the kernels so the first timed run is fair
        ids = self.tok("The", return_tensors="pt").input_ids.to("cuda")
        self.model.generate(ids, max_new_tokens=8, do_sample=False, use_cache=True)
        hf_cache.commit()

    def _run(self, ids, max_new, use_cache):
        torch = self.torch
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        with torch.no_grad():
            out = self.model.generate(
                ids, max_new_tokens=max_new, min_new_tokens=max_new, do_sample=False,
                use_cache=use_cache, pad_token_id=self.tok.eos_token_id)
        t1.record()
        torch.cuda.synchronize()
        secs = t0.elapsed_time(t1) / 1000.0
        gen = out[:, ids.shape[1]:]
        total = gen.shape[0] * gen.shape[1]
        return secs, total, out[0, ids.shape[1]:]

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, Request
        from fastapi.middleware.cors import CORSMiddleware

        globals()["Request"] = Request
        api = FastAPI(title="slm-125m-kvcache")
        api.add_middleware(CORSMiddleware, allow_origins=["*"],
                           allow_methods=["*"], allow_headers=["*"])

        @api.get("/health")
        def health():
            return {"ok": True, "model": MODEL_ID}

        @api.post("/bench")
        async def bench(req: Request):
            b = await req.json()
            prompt = (b.get("prompt") or "The plaintiff shall bear the burden of").strip()
            max_new = max(16, min(256, int(b.get("max_new_tokens", 128))))
            batch = max(1, min(64, int(b.get("batch_size", 1))))

            ids = self.tok(prompt, return_tensors="pt").input_ids.to("cuda")
            ids = ids.repeat(batch, 1)

            with_secs, total, sample = self._run(ids, max_new, True)
            without_secs, _, _ = self._run(ids, max_new, False)
            text = self.tok.decode(sample, skip_special_tokens=True)

            return {
                "prompt": prompt, "batch_size": batch, "max_new_tokens": max_new,
                "output": text,
                "with_cache": {"seconds": round(with_secs, 3), "tokens_per_sec": round(total / with_secs, 1)},
                "without_cache": {"seconds": round(without_secs, 3), "tokens_per_sec": round(total / without_secs, 1)},
                "speedup": round(without_secs / with_secs, 2),
                "total_tokens": total,
            }

        return api
