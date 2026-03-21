# Module 0 — What Is Fine-Tuning?

## Overview

This module clarifies the vocabulary used throughout this course so you know exactly what you're doing and why.

---

## 1. Pre-training vs. Fine-tuning

| Stage | Goal | Data scale | Compute |
|---|---|---|---|
| **Pre-training** | Learn general language / world knowledge | Hundreds of GB – TB of text | Thousands of GPU-hours |
| **Fine-tuning** | Adapt to a specific task or style | Hundreds – millions of examples | Hours – days |

Most practitioners **never pre-train from scratch**. They start from a pre-trained checkpoint and fine-tune it.

---

## 2. Fine-tuning flavours

### 2a. Full Fine-tuning
Update **all** weights of the model on your data. Expensive in memory and time; rarely needed for task adaptation.

### 2b. Instruction Tuning
Fine-tune on `(instruction, input, output)` triples so the model learns to follow natural-language directions.  
Examples: Alpaca, Dolly, FLAN.

### 2c. Parameter-Efficient Fine-Tuning (PEFT)
Freeze most weights; train only a small number of new parameters. Covers:
- **LoRA** (Low-Rank Adaptation) — the most popular method; see Module 3.
- **Prefix tuning** — prepend trainable tokens to each layer.
- **Prompt tuning** — learn soft prompt embeddings only.
- **IA³** — even fewer parameters than LoRA.

### 2d. Preference Tuning / Alignment
Teach the model to produce outputs humans prefer, usually compared to a reward model:
- **RLHF** (Reinforcement Learning from Human Feedback) — complex; requires a reward model + PPO loop.
- **DPO** (Direct Preference Optimization) — simpler; trains on `(chosen, rejected)` pairs directly.
- **ORPO, SimPO** — even simpler variants gaining traction in 2024–2025.

---

## 3. What this course focuses on

```
Pre-trained model  →  [Instruction Tuning via LoRA]  →  Adapted model
                             ↑
                     affordable on Mac M4
```

We use **LoRA-based instruction tuning** because:
1. It runs on 8–16 GB of unified memory.
2. The adapter is tiny to save/share (a few MB vs. multi-GB full checkpoints).
3. The same concepts extend to QLoRA, DPO, etc. once you understand the basics.

---

## 4. Mental model: LoRA in 60 seconds

A weight matrix `W` (size `d × k`) is large. LoRA adds two tiny matrices `A` (d × r) and `B` (r × k) where `r ≪ d`.  
Only `A` and `B` are trained. The adapted weight is `W + AB`.

- `r=8` is a common default (vs. `d=768` for GPT-2 hidden dim → ~1% of parameters).
- After training, you can **merge** `AB` into `W` for zero-latency inference, or keep them separate (hot-swappable adapters).

---

## 5. Suggested reading

- [LoRA paper (Hu et al. 2021)](https://arxiv.org/abs/2106.09685) — 9 pages, very readable.
- [QLoRA paper (Dettmers et al. 2023)](https://arxiv.org/abs/2305.14314) — introduces 4-bit quantization + LoRA.
- [Hugging Face PEFT docs](https://huggingface.co/docs/peft) — practical reference.
