# Llama Turkish Function-Calling Agent

A local, Turkish-speaking AI agent built on **Llama-3.1-8B-Instruct**, fine-tuned with **QLoRA** and quantized to **GGUF** for inference on consumer hardware. The agent runs a ReAct loop — reason, call a tool, observe, respond — entirely on a single 8GB GPU.

## Overview

Most function-calling fine-tunes are trained and evaluated in English. This project asks a narrower question: can a small, carefully-curated Turkish subset (~2% of the training mix) reliably steer a model's *final, user-facing* responses into Turkish — without touching the English tool-calling semantics that make up the bulk of the data — and without needing anything close to a large Turkish corpus?

The answer depends entirely on *how* that 2% is trained on, not just that it exists. That distinction shapes most of the design decisions below.

## Key design decisions

- **Hybrid language strategy.** System prompts, tool schemas, and `tool_call` JSON stay in English throughout. Only the user-facing layer — the user's own messages and the assistant's final natural-language answer — is localized. The model isn't learning a new skill in Turkish, only redirecting an existing one.
- **Two-layer loss masking.** Standard "train on responses only" recipes compute loss over every assistant turn, including English final-turn prose from the base dataset. Left unmasked, that creates a gradient that directly contradicts the Turkish-response instruction, at the same token position, for the majority of the training run. The fix: mask the English final-turn segment for every non-Turkish example, so the Turkish subset becomes the *sole* source of final-turn behavior instead of competing against it at ~50:1 odds.
- **Quantization-aware merge pipeline.** The merge/quantize pipeline produces an intermediate `q8_0` GGUF (half the disk footprint of `f16`, no measurable quality cost for K-quants) and generates an **importance matrix** from a real sample of the training distribution before the final `Q4_K_M` quantization — trading a calibration pass for a meaningfully smaller quality gap at the same file size.
- **Single training run.** Compute constraints mean there's no retry budget. Every part of the pipeline — dataset preparation, masking, the Turkish subset itself — is built to be manually auditable *before* the run, not fixed after.
- **Control lives in code, not in the prompt.** Where the model has a learned habit that a prompt cannot reliably override, the orchestrator or the tool enforces the behavior instead. Each instance below was measured, not assumed.

## Architecture

```
training/
  prepare_dataset.ipynb     # Hermes -> Llama-3.1 chat template conversion
  train.py                  # QLoRA fine-tuning with two-layer loss masking
  merge_and_quantize.py     # LoRA merge -> GGUF (q8_0 -> imatrix -> Q4_K_M)
  turkish_examples/         # Curated Turkish subset + curation methodology

inference/
  config.py                 # .env reader (stdlib only, no python-dotenv)
  prompts.py                # System prompt + variants; single source for agent AND eval
  serve.py                  # HTTP client to llama-server (model access layer)
  orchestrator.py           # ReAct loop: parsing, argument grounding, pruning, limits
  grammar/generate_gbnf.py  # GBNF generated from the live tool registry
  tools/                    # Registry + google_search, get_weather, get_system_time

eval/
  test_set.jsonl            # 74-record held-out set, 19 of them on unseen schemas
  test_set_SEMA.md          # Record schema and scoring rules
  eval_post_quant.py        # Scores the quantized model against the set
  eval_pre_quant.py         # bfloat16 baseline (not run: see Status)
```

## Hardware

Training and inference intentionally run on different hardware, and the pipeline is built around that split rather than around a single machine.

| | Hardware | Role |
|---|---|---|
| Training | 4x H100 (cloud) | QLoRA fine-tuning in bfloat16 |
| Merge & quantize | CPU (cloud VM) | LoRA merge, GGUF conversion, imatrix calibration, quantization |
| Inference | RTX 4060 Laptop, 8GB | GGUF served by `llama-server` (llama.cpp, Vulkan backend) |

Measured on the inference machine at `n_ctx=8192`, `Q4_K_M`, q8_0 KV cache, flash attention:

| | |
|---|---|
| Model weights | 4403 MiB |
| KV cache | 544 MiB (68 KB/token) |
| Compute buffer | 108 MiB |
| **Total** | **5055 / 7774 MiB** — ~2.7 GB headroom |
| Generation | ~40 tok/s |
| Prompt processing | ~175 tok/s |

Memory turned out not to be the binding constraint; prompt processing is. Every 1000 tokens of context costs ~6 seconds on every subsequent turn, which is why the orchestrator trims context and the search tool trims its own output.

> The spec called for in-process `llama-cpp-python`. That needs the CUDA toolkit, which this machine does not have, while llama.cpp's prebuilt Vulkan binaries already worked. `serve.py` is therefore an HTTP client to `llama-server`; every setting the spec required exists as a server flag, and prefix caching works over HTTP (measured: 14 tokens processed instead of 440 on the second request).

## Evaluation

`eval/test_set.jsonl` is a held-out set built by hand. It was screened for contamination against the training data — verbatim matches and a 0.70 similarity threshold — and five real leaks were found and rewritten. Tool names claimed as "unseen" were checked against all 2,985 tool names in the training data; several intuitive picks (`convert_currency`, `translate_text`, `find_restaurants`) turned out to be present and were replaced.

Scored with the grammar **off**, so the numbers reflect what the model produces unaided:

| | |
|---|---|
| Unseen tool schemas | **18/19 (94%)** |
| Seen tool schemas | 41/44 (93%) |
| JSON validity | 40/40 (100%) |
| Turkish final turn after an observation | 6/6 |

The first two lines are the point of the exercise. Equal performance on schemas the model has never seen means it learned "read the schema, build the call" rather than memorizing tool names from training.

Records where more than one behavior is defensible are **reported, not scored**. Inventing a reference answer to make a metric look complete would have made the metric worse.

## What measurement changed

Each of these was a live or evaluated failure, traced to a cause, and fixed in code rather than by rewording a prompt:

- **The masking silently deleted training data.** Hermes examples that answer directly, without a tool, consist of nothing but a final turn — so masking the final turn masked the whole example and dropped it. 803 of 851 such examples vanished, leaving a 1:71 skew toward calling a tool. The model over-triggered exactly as that ratio predicts.
- **The model fabricates arguments it does not have.** "Hava nasıl?" (no city) produced `location='Ankara'`; "Ankara'dan tren var mı?" produced `destination="user's destination"` — a literal placeholder. No prompt variant fixed it, so `orchestrator._ungrounded()` refuses to dispatch a call whose arguments cannot be traced to the conversation, and asks the user instead.
- **`num_results=1` was learned, not chosen.** The training data calls `google_search` with `num_results=1` in 34 of 55 cases. The tool now treats the parameter as a floor, not a ceiling.
- **A blocked search was reported as an empty one.** DuckDuckGo answers HTTP 202 with a CAPTCHA after ~7 requests on `/lite` and after 2 on `/html`; the tool read that as a successful search with zero hits and told the model "nothing was found". Scraping was measured to be unworkable here, so search is a provider chain — Tavily (official API) first, DuckDuckGo and Wikipedia as keyless fallbacks — with explicit block detection, and every failure message tells the model not to claim it searched.
- **Short answers are a learned length.** Turkish final turns in the training data have a median of 118 characters. A length instruction in the system prompt barely moved it; the same instruction rendered immediately before generation does, because recency is what matters.

## Status

| Component | Status |
|---|---|
| Dataset preparation (Hermes + Turkish subset) | Done |
| QLoRA training script | Done |
| LoRA merge / GGUF quantization pipeline | Done, validated end-to-end |
| Tool layer + registry | Done |
| GBNF grammar generation | Done (grammar itself is optional — see below) |
| Inference server & ReAct orchestrator | Done |
| Evaluation harness | Done |
| bfloat16 baseline comparison | **Not possible** — the merged bf16 model was deleted by the quantize pipeline before a baseline was taken |

The grammar layer is insurance rather than a requirement: the model produced 100% valid JSON with the grammar off. It earns its keep once temperature rises above zero and contexts grow.

## Setup

```bash
pip install -r requirements.txt
# torch is installed separately, matching your CUDA version:
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Inference needs no Python dependencies beyond the standard library — only a running `llama-server`.

### Search provider

Copy your [Tavily](https://app.tavily.com) key into `.env` (git-ignored):

```
TAVILY_API_KEY=tvly-...
```

Tavily is the primary provider because it is the only one here not fighting a bot filter, and because it returns page text directly — results from it skip the page-fetch step entirely. Without a key the agent still runs, falling back to DuckDuckGo and then Wikipedia; those work for a handful of queries before DuckDuckGo starts serving CAPTCHAs.

## Running the agent

Start the model server (flags must match `inference/serve.py`, or measurements no longer compare):

```bash
cd <llama.cpp build directory>
LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH ./llama-server \
  -m training/outputs/gguf/llama31-8b-tr-Q4_K_M.gguf \
  -c 8192 -np 1 -ngl 99 -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --host 127.0.0.1 --port 8080
```

Then, in a second terminal:

```bash
python inference/orchestrator.py --verbose      # interactive agent
python eval/eval_post_quant.py --out report.json  # score against the test set
```

`--verbose` logs each tool call and its outcome to stderr. `--variant` selects a system prompt variant and `--grammar` enables constrained decoding.

Training and merge/quantize scripts are run independently — see the comments at the top of each script for hardware assumptions and expected runtimes.
