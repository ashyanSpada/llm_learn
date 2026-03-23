# Module 2 — Dataset Preparation for Instruction Tuning

## Goal

Understand how to format and load a tiny dataset for instruction-style LLM fine-tuning.

---

## 1. The instruction-tuning format

The most common format is an `(instruction, input, output)` triple rendered into a single text prompt:

```
### Instruction:
Summarise the following paragraph in one sentence.

### Input:
Large language models are trained on vast corpora...

### Response:
LLMs are pre-trained on large text corpora and then fine-tuned for downstream tasks.
```

Some datasets omit the `input` field when it's not needed:

```
### Instruction:
What is gradient descent?

### Response:
Gradient descent is an optimisation algorithm that iteratively adjusts model parameters...
```

This format is sometimes called the **Alpaca prompt template** (from the Stanford Alpaca project).

---

## 2. JSONL file format

We store examples as [JSONL](https://jsonlines.org/) — one JSON object per line:

```jsonl
{"instruction": "What is a transformer model?", "input": "", "output": "A transformer is..."}
{"instruction": "Explain backpropagation.", "input": "", "output": "Backpropagation is..."}
```

Each line is an independent JSON object. This is easy to stream, append to, and parse.

---

## 3. The `data.py` module

`src/llm_learn/data.py` provides:

- `load_jsonl(path)` — reads a JSONL file and returns a list of dicts.
- `format_prompt(example)` — renders `instruction/input/output` into the Alpaca template.
- `build_dataset(path, tokenizer, max_length)` — tokenises examples and returns a `datasets.Dataset` ready for the Trainer.

### Example usage

```python
from llm_learn.data import load_jsonl, format_prompt

examples = load_jsonl("data/tiny_instruct.jsonl")
for ex in examples[:2]:
    print(format_prompt(ex))
    print("---")
```

---

## 4. The example dataset (`data/tiny_instruct.jsonl`)

`tiny_instruct.jsonl` contains ~30 short, factual Q&A examples covering ML concepts. It is intended purely as a functional demo — not to teach the model new facts.

---

## 5. Tips for real datasets

| Dataset | Size | Format | Where to find |
|---|---|---|---|
| `tatsu-lab/alpaca_cleaned` | ~52k | Alpaca JSON | Hugging Face Hub |
| `Open-Platypus` | ~25k | Alpaca JSON | Hugging Face Hub |
| `timdettmers/openassistant-guanaco` | ~10k | ShareGPT | Hugging Face Hub |
| Your own data | Any | Convert to JSONL | — |

To use a Hugging Face dataset instead of a local JSONL, swap `load_jsonl` for:

```python
from datasets import load_dataset
ds = load_dataset("tatsu-lab/alpaca_cleaned", split="train")
```

Then apply `format_prompt` as a `map` transformation.

---

## 6. Data quality checklist

Before training, verify:
- [ ] No empty `output` fields.
- [ ] Instructions are concise and clear.
- [ ] No personally identifiable information (PII).
- [ ] Consistent formatting (same prompt template throughout).
- [ ] Train/eval split (reserve at least 10% for evaluation).
