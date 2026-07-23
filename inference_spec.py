"""Speculative decoding demo: a large TARGET model (Qwen2.5-7B-Instruct) verifies tokens
proposed by a small DRAFT model (Qwen2.5-0.5B-Instruct). Greedy speculative decoding is
*exact* — it produces the identical output to plain greedy on the target, just faster,
because the target verifies K draft tokens in one forward pass instead of generating them
one at a time.

Serves POST /spec -> JSON with, for both plain target-only greedy and speculative:
tokens/sec and the output; plus, for speculative, per-token provenance (draft vs target)
and the draft acceptance rate.

Scale-to-zero L4 GPU.  modal deploy inference_spec.py
"""

# NOTE: no `from __future__ import annotations` (FastAPI hint resolution).

import modal

app = modal.App("qwen-speculative")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "accelerate==1.1.1",
                 "fastapi[standard]==0.115.4")
)

TARGET_ID = "Qwen/Qwen2.5-7B-Instruct"
DRAFT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
K = 4  # draft tokens proposed per round
hf_cache = modal.Volume.from_name("legal-slm-hf-cache", create_if_missing=True)


@app.cls(image=image, gpu="L4", volumes={"/cache": hf_cache},
         scaledown_window=300, min_containers=0, timeout=60 * 10)
class Spec:
    @modal.enter()
    def load(self):
        import os

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        os.environ["HF_HOME"] = "/cache"
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(TARGET_ID, cache_dir="/cache")
        self.target = AutoModelForCausalLM.from_pretrained(
            TARGET_ID, cache_dir="/cache", torch_dtype=torch.bfloat16, device_map={"": 0}).eval()
        self.draft = AutoModelForCausalLM.from_pretrained(
            DRAFT_ID, cache_dir="/cache", torch_dtype=torch.bfloat16, device_map={"": 0}).eval()
        ids = self._prompt_ids("Hello")
        self._greedy(ids, 4)          # warm kernels
        self._spec(ids, 4)
        hf_cache.commit()

    def _prompt_ids(self, prompt):
        msgs = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")

    def _time(self):
        t = self.torch.cuda.Event(enable_timing=True)
        t.record()
        return t

    def _greedy(self, input_ids, max_new):
        """Plain greedy on the target with a KV cache. Returns (tokens, seconds)."""
        import torch
        from transformers import DynamicCache

        eos = self.tok.eos_token_id
        torch.cuda.synchronize(); t0 = self._time()
        cache = DynamicCache()
        with torch.no_grad():
            out = self.target(input_ids, past_key_values=cache, use_cache=True)
            logit = out.logits[:, -1, :]
            cache = out.past_key_values
            toks = []
            for _ in range(max_new):
                nid = int(logit.argmax(-1))
                toks.append(nid)
                if nid == eos:
                    break
                nxt = torch.tensor([[nid]], device=input_ids.device)
                out = self.target(nxt, past_key_values=cache, use_cache=True)
                logit = out.logits[:, -1, :]
                cache = out.past_key_values
        t1 = self._time(); torch.cuda.synchronize()
        return toks, t0.elapsed_time(t1) / 1000.0

    def _spec(self, input_ids, max_new):
        """Greedy speculative decoding, exact wrt target greedy. One target forward per
        round verifies [next_token, K draft tokens]; the bonus token rides into the next
        round's verification (no extra target pass). Returns (list[(id, source)], secs, acc)."""
        import torch
        from transformers import DynamicCache

        eos = self.tok.eos_token_id
        device = input_ids.device
        torch.cuda.synchronize(); t0 = self._time()
        tcache, dcache = DynamicCache(), DynamicCache()
        with torch.no_grad():
            to = self.target(input_ids, past_key_values=tcache, use_cache=True)
            self.draft(input_ids, past_key_values=dcache, use_cache=True)
            base = input_ids.shape[1]                      # both caches cover `base` tokens
            first = int(to.logits[:, -1, :].argmax(-1))    # target's first token (always correct)
            gen = [(first, "target")]
            next_token = first
            accepted, proposed = 0, 0
            done = first == eos

            while len(gen) < max_new and not done:
                # 1. draft proposes K tokens, starting from next_token
                q = []
                nt = torch.tensor([[next_token]], device=device)
                do = self.draft(nt, past_key_values=dcache, use_cache=True)
                dl = do.logits[:, -1, :]
                for _ in range(K):
                    nid = int(dl.argmax(-1))
                    q.append(nid)
                    do = self.draft(torch.tensor([[nid]], device=device),
                                    past_key_values=dcache, use_cache=True)
                    dl = do.logits[:, -1, :]
                proposed += K
                # 2. target verifies [next_token, q_1..q_K] in ONE forward
                verify = torch.tensor([[next_token] + q], device=device)   # [1, K+1]
                to = self.target(verify, past_key_values=tcache, use_cache=True)
                preds = to.logits.argmax(-1)[0]            # [K+1]; preds[i] = target token after verify[i]
                # 3. accept longest prefix where draft == target
                a = 0
                for i in range(K):
                    if q[i] == int(preds[i]):
                        a += 1
                    else:
                        break
                bonus = int(preds[a])
                for i in range(a):
                    gen.append((q[i], "draft"))
                    if q[i] == eos:
                        done = True
                accepted += a
                if not done:
                    gen.append((bonus, "target"))
                    if bonus == eos:
                        done = True
                # 4. crop both caches back to next_token + the accepted draft tokens
                keep = base + 1 + a
                tcache.crop(keep); dcache.crop(keep)
                base = keep
                next_token = bonus
        t1 = self._time(); torch.cuda.synchronize()
        return gen[:max_new], t0.elapsed_time(t1) / 1000.0, accepted / max(1, proposed)

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, Request
        from fastapi.middleware.cors import CORSMiddleware

        globals()["Request"] = Request
        api = FastAPI(title="qwen-speculative")
        api.add_middleware(CORSMiddleware, allow_origins=["*"],
                           allow_methods=["*"], allow_headers=["*"])

        @api.get("/health")
        def health():
            return {"ok": True, "target": TARGET_ID, "draft": DRAFT_ID}

        @api.post("/spec")
        async def spec(req: Request):
            b = await req.json()
            prompt = (b.get("prompt") or "Explain what a Form 10-K filing is, in two sentences.").strip()
            max_new = max(16, min(160, int(b.get("max_new_tokens", 96))))
            ids = self._prompt_ids(prompt)

            base_toks, base_s = self._greedy(ids, max_new)
            spec_gen, spec_s, acc = self._spec(ids, max_new)

            def decode1(tid):
                return self.tok.decode([tid], skip_special_tokens=True)

            spec_tokens = [{"text": decode1(t), "source": s} for t, s in spec_gen]
            return {
                "prompt": prompt, "max_new_tokens": max_new, "K": K,
                "target_only": {
                    "output": self.tok.decode(base_toks, skip_special_tokens=True),
                    "tokens_per_sec": round(len(base_toks) / base_s, 1),
                    "seconds": round(base_s, 3), "tokens": len(base_toks),
                },
                "speculative": {
                    "output": self.tok.decode([t for t, _ in spec_gen], skip_special_tokens=True),
                    "tokens_per_sec": round(len(spec_gen) / spec_s, 1),
                    "seconds": round(spec_s, 3), "tokens": len(spec_gen),
                    "acceptance_rate": round(acc, 3),
                    "token_provenance": spec_tokens,
                },
                "speedup": round(base_s / spec_s, 2),
            }

        return api
