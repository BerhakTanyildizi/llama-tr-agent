# Llama Turkish Function-Calling Agent

A local, Turkish-speaking AI agent built on **Llama-3.1-8B-Instruct**, fine-tuned with **QLoRA** and quantized to **GGUF** for inference on consumer hardware. The agent follows a ReAct-style loop — reason, call a tool, observe, respond — and is designed to run entirely on a single 8GB GPU.

## Overview

Most function-calling fine-tunes are trained and evaluated in English. This project asks a narrower question: can a small, carefully-curated Turkish subset (~2% of the training mix) reliably steer a model's *final, user-facing* responses into Turkish — without touching the English tool-calling semantics that make up the bulk of the data — and without needing anything close to a large Turkish corpus?

The answer depends entirely on *how* that 2% is trained on, not just that it exists. That distinction shapes most of the design decisions below.

## Key design decisions

- **Hybrid language strategy.** System prompts, tool schemas, and `tool_call` JSON stay in English throughout. Only the user-facing layer — the user's own messages and the assistant's final natural-language answer — is Turkish. The model isn't learning a new skill in Turkish, only redirecting an existing one.
- **Two-layer loss masking.** Standard "train on responses only" recipes compute loss over every assistant turn, including English final-turn prose from the base dataset. Left unmasked, that creates a gradient that directly contradicts the Turkish-response instruction, at the same token position, for the majority of the training run. The fix: mask the English final-turn segment for every non-Turkish example, so the Turkish subset becomes the *sole* source of final-turn behavior instead of competing against it at ~50:1 odds.
- **Quantization-aware merge pipeline.** The merge/quantize pipeline produces an intermediate `q8_0` GGUF (half the disk footprint of `f16`, no measurable quality cost for K-quants) and generates an **importance matrix** from a real sample of the training distribution before the final `Q4_K_M` quantization — trading a calibration pass for a meaningfully smaller quality gap at the same file size.
- **Single training run.** Compute constraints mean there's no retry budget. Every part of the pipeline — dataset preparation, masking, the Turkish subset itself — is built to be manually auditable *before* the run, not fixed after.

## Architecture

```
training/
  prepare_dataset.ipynb     # Hermes -> Llama-3.1 chat template conversion
  train.py                  # QLoRA fine-tuning with two-layer loss masking
  merge_and_quantize.py     # LoRA merge -> GGUF (q8_0 -> imatrix -> Q4_K_M)
  turkish_examples/         # Curated Turkish subset + curation methodology

inference/
  serve.py                  # llama-cpp-python model server
  orchestrator.py            # ReAct loop, context pruning, prefix caching
  grammar/                  # Dynamic GBNF generation from tool schemas
  tools/                    # google_search, get_weather, get_system_time

eval/
  eval_pre_quant.py          # bfloat16 baseline evaluation
  eval_post_quant.py         # GGUF evaluation vs. baseline
```

## Hardware

Training and inference intentionally run on different hardware, and the pipeline is built around that split rather than around a single machine.

| | Hardware | Role |
|---|---|---|
| Training | 4x H100 (cloud) | QLoRA fine-tuning in bfloat16 |
| Merge & quantize | CPU (cloud VM) | LoRA merge, GGUF conversion, imatrix calibration, quantization |
| Inference | RTX 4060, 8GB | GGUF model serving via llama-cpp-python |

The 8GB inference budget is split between model weights (`Q4_K_M` ≈ 4.9GB), KV-cache, and CUDA context — everything upstream of inference (dataset prep, training, quantization) is designed around what fits in what's left.

## Status

| Component | Status |
|---|---|
| Dataset preparation (Hermes + Turkish subset) | Done |
| QLoRA training script | Done |
| LoRA merge / GGUF quantization pipeline | Done, validated end-to-end |
| Inference server & ReAct orchestrator | Planned |
| GBNF grammar generation | Planned |
| Evaluation harness | Planned |

## Setup

```bash
pip install -r requirements.txt
# torch is installed separately, matching your CUDA version:
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Training and merge/quantize scripts are run independently — see the comments at the top of each script for hardware assumptions and expected runtimes.
