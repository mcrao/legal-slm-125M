---
license: apache-2.0
base_model: jonam-ai/legal-slm-125m-sft
library_name: transformers.js
pipeline_tag: text-generation
tags:
  - onnx
  - transformers.js
  - legal
  - finance
  - edge
  - in-browser
language:
  - en
---

# legal-slm-125m-sft-onnx

An **ONNX export of [jonam-ai/legal-slm-125m-sft](https://huggingface.co/jonam-ai/legal-slm-125m-sft)**
(a 125M legal/financial assistant trained from scratch), int8-quantized so it runs
**entirely in the browser** via [transformers.js](https://github.com/huggingface/transformers.js)
and WebAssembly — no server, no API, no cost.

- **Live demo (flip the "In-browser" toggle in the Chat):** https://legal-slm-125.vercel.app

## What's here

| File | Precision | Size | Use |
|---|---|---|---|
| `onnx/model_quantized.onnx` | int8 (`dtype: "q8"`) | ~133 MB | in-browser inference |

The model downloads once to the visitor's device and is cached; inference then runs
locally on their CPU through WASM.

## Honest note on quantization

The int8 step is **not free**. Measured on the same held-out set, quantization raises
perplexity by ~38% (fp32 ≈ 7.89 → int8 ≈ 10.88). The output stays coherent, and for a
125M toy-model demo the ~4× smaller download is worth it — but this is a real accuracy
cost, not a rounding error. Measure, don't eyeball.

For best quality, use the full-precision PyTorch model
([jonam-ai/legal-slm-125m-sft](https://huggingface.co/jonam-ai/legal-slm-125m-sft)) instead.

## Usage (transformers.js)

```js
import { pipeline } from "@huggingface/transformers";

const generator = await pipeline(
  "text-generation",
  "jonam-ai/legal-slm-125m-sft-onnx",
  { dtype: "q8", device: "wasm" }
);

const out = await generator(
  "What must a plaintiff prove in a breach of contract claim?",
  { max_new_tokens: 160, temperature: 0.7, do_sample: true }
);
console.log(out[0].generated_text);
```

## Intended use & limitations

Educational, in-browser demonstration. It is a small base-derived assistant that will
confidently invent case names and figures. Not legal or financial advice. English only,
1,024-token context.
