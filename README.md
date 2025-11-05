# Toyformer: a pedagogical Transformer from scratch

**Toyformer** is a minimal, teaching-oriented Transformer implementation that builds complex layers from simple **ops**.It aims to be readable, hackable, and verifiable: every block composes small functions with explicit shapes.

> Scope: correctness & teachability. No vendor kernels, no FlashAttention, no MoE. CUDA optimizations are intentionally omitted.

## Features

- **Two model families**
  - `DecoderOnlyLM` (GPT-style, causal self-attn)
  - `TransformerSeq2Seq` (Encoder–Decoder with self-attn + cross-attn)
- **Pedagogical layering**
  - `ops.py`: `matmul`, `scale`, `softmax`, `linear` (broadcast-safe, dtype-safe)
  - `layers/`: masked/self/cross MHA, FFN (GELU), Add&Norm (Pre-LN), LMHead
  - `blocks/`: causal decoder block and encoder–decoder block
- **Tokenizer pipeline** (HF `tokenizers`) with `tokenizer.json`
- **Training**
  - AMP (`torch.autocast`) and gradient accumulation
  - early stopping, best & final checkpoints
  - fixed-length batching for stable FLOPs
- **Generation**
  - temperature, top-k, repetition penalty, n-gram blocking
  - optional “adaptive-topk (pseudo top-p)” & temperature annealing
- **Reproducible diagnostics**
  - shape assertions, numerical checks (`ops.matmul` vs `torch.matmul`), memory/useful logs

## Installation

```bash
python -m venv .venv && source .venv/bin/activate    # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```
