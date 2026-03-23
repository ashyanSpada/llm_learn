# Module 4 — Evaluation

## Goal

Compare your base model against the LoRA-tuned adapter using lightweight metrics you can run locally.

---

## 1. What we measure

We use two complementary signals:

| Metric | What it tells you | How to run |
|---|---|---|
| **Perplexity** | How surprised the model is by held-out text. Lower = better language model. | `eval.py --metric perplexity` |
| **Prompt responses** | Human-readable output quality on a fixed prompt set. | `eval.py --metric generate` |

Neither is a gold standard, but together they give a quick signal of whether training helped.

---

## 2. Running evaluation

```bash
# Evaluate base model only
python -m llm_learn.eval \
    --model_name distilgpt2 \
    --prompts_file data/eval_prompts.json

# Evaluate base model + LoRA adapter
python -m llm_learn.eval \
    --model_name distilgpt2 \
    --adapter_path runs/lora_demo \
    --prompts_file data/eval_prompts.json
```

Or via Makefile:

```bash
make eval
```

The script prints a table like:

```
Prompt                                  | Base model output              | Tuned model output
----------------------------------------|--------------------------------|----------------------------
What is a neural network?               | A neural network is a type ... | A neural network is a ...
Explain gradient descent in one line.   | Gradient descent ...           | Gradient descent minimises ...
```

---

## 3. Perplexity explained

Perplexity (PPL) = `exp(average negative log-likelihood per token)`.

- A model that perfectly predicts every token → PPL = 1.
- A random character-level model on English text → PPL ≈ 26 (alphabet size).
- `distilgpt2` on WikiText-103 → PPL ≈ 21 (pre-trained baseline).

After instruction tuning on a tiny dataset, PPL on your eval set should **decrease** if training is working.

```python
import math
import torch

def perplexity(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    return math.exp(loss.item())
```

---

## 4. Prompt-based comparison (qualitative)

`data/eval_prompts.json` contains a fixed set of short prompts. The eval script generates a response from both the base model and the tuned adapter, then prints them side by side.

This is the most intuitive check: did the tuned model get better at following instructions?

---

## 5. Deeper evaluation (when you're ready)

| Tool | What it provides |
|---|---|
| `lm-evaluation-harness` (EleutherAI) | Standardised benchmarks (MMLU, ARC, HellaSwag, …) |
| `promptfoo` | A/B prompt testing framework |
| Manual annotation | Gold standard; label 20–50 examples yourself |

For a quick sanity check, `lm-eval` with `--tasks piqa,arc_easy` takes ~10 minutes on a Mac with `distilgpt2`.

---

## 6. What "good" looks like on a tiny demo

With only ~30 training examples and 20–50 steps, you should expect:
- Small perplexity improvement on the training distribution.
- Slight improvement in instruction-following tone.
- **Not** a fully capable assistant — that requires 10k+ examples and many more steps.

The goal of this module is the **workflow**, not state-of-the-art results.
