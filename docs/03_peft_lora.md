# Module 3 — PEFT / LoRA Fine-Tuning

## Goal

Understand LoRA conceptually and run a minimal fine-tuning script on your Mac.

---

## 1. Why PEFT?

Full fine-tuning updates every weight in the model. For a 1B-parameter model that's ~4 GB of gradients + optimizer state on top of the model itself — easily 12–16 GB just for training state. PEFT methods reduce this dramatically.

| Method | Trainable params | Memory savings | Quality |
|---|---|---|---|
| Full fine-tune | 100% | — | Highest |
| LoRA (r=8) | ~0.1–1% | ~10–30× | Near-full for many tasks |
| Prompt tuning | <0.01% | ~100× | Task-dependent |

---

## 2. LoRA mechanics

Given a weight matrix `W₀ ∈ ℝ^(d×k)`, LoRA adds:

```
W = W₀ + α/r · B·A
```

where `A ∈ ℝ^(r×k)` and `B ∈ ℝ^(d×r)` are the trainable matrices, `r` is the rank, and `α` is a scaling factor.

`W₀` is frozen; only `A` and `B` accumulate gradients. After training, you can either:
- **Keep them separate** — load `W₀` + adapter on the fly (hot-swappable).
- **Merge them** — compute `W₀ + BA` once and save a single weight (zero overhead at inference).

---

## 3. Key hyperparameters

| Parameter | Typical values | Effect |
|---|---|---|
| `lora_r` | 4, 8, 16 | Rank. Higher → more capacity but more params. Start with 8. |
| `lora_alpha` | 16, 32 | Scaling. Common to set `alpha = 2 * r`. |
| `lora_dropout` | 0.0–0.1 | Regularisation. Use 0.05 for small datasets. |
| `target_modules` | `["c_attn"]` for GPT-2 | Which weight matrices to adapt. Attention proj layers are standard. |

---

## 4. Running the training script

```bash
python -m llm_learn.train_lora \
    --model_name distilgpt2 \
    --data_path data/tiny_instruct.jsonl \
    --output_dir runs/lora_r8 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --max_steps 50 \
    --batch_size 2 \
    --grad_accum 4 \
    --lr 2e-4
```

This completes in **~2–5 minutes** on a Mac mini M4 with MPS acceleration.

The script will:
1. Load `distilgpt2` from Hugging Face Hub.
2. Wrap it with `peft.LoraConfig`.
3. Tokenise `data/tiny_instruct.jsonl` and train for `--max_steps` steps.
4. Save the LoRA adapter weights to `--output_dir`.

---

## 5. What gets saved

```
runs/lora_r8/
├── adapter_config.json     # LoRA config (r, alpha, target modules, ...)
├── adapter_model.safetensors   # the trained A and B matrices only
└── training_args.bin       # Hugging Face TrainingArguments
```

The adapter is typically **< 5 MB** for `distilgpt2` with `r=8`.

---

## 6. Merging the adapter (optional)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("distilgpt2")
model = PeftModel.from_pretrained(base, "runs/lora_r8")
merged = model.merge_and_unload()
merged.save_pretrained("runs/lora_r8_merged")
```

---

## 7. Scaling up from here

1. Increase `--lora_r` to 16 or 32 for more capacity.
2. Try larger models: `gpt2` (124M), `EleutherAI/pythia-410m`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.
3. Use a real dataset (Alpaca cleaned, ~52k examples).
4. For **QLoRA** (4-bit + LoRA), use a CUDA GPU (free Colab T4 works).

---

## 8. Troubleshooting on MPS

- **`RuntimeError: MPS backend out of memory`** → reduce `--batch_size` to 1 or add `--grad_accum 8`.
- **NaN loss** → switch to `--dtype float32`.
- **Very slow first step** → MPS compiles kernels on first use; subsequent steps are faster.
