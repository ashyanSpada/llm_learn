# llm_learn

A practical, low-compute LLM fine-tuning curriculum designed for Apple Silicon (Mac mini M4) and optionally Jetson-class devices.

## Prerequisites

- **Python** 3.10 or 3.11 (3.12+ works but some packages may lag)
- **macOS** 13+ (Ventura or newer) for Apple Silicon / MPS support; Linux also supported
- **`venv`** (built-in, recommended) or `uv` for environment management
- Git

> **Note:** You do **not** need a discrete NVIDIA GPU. Training runs on CPU or Apple's MPS (Metal Performance Shaders) backend.

## Hardware Expectations

| Device | What works | What to skip |
|---|---|---|
| Mac mini M4 (16 GB unified memory) | LoRA fine-tuning of ≤1B-param models, full fine-tune of tiny models (distilgpt2), inference | Multi-GPU, 4-bit bitsandbytes quantization (CUDA-only) |
| Jetson Nano / Orin Super | Edge inference (ONNX), tiny PyTorch inference | LoRA training of anything >125M params |
| CPU-only laptop | Short demo runs with `distilgpt2` + few steps | Anything over ~500M params or >100 steps |

> MPS acceleration on Mac M4 provides ~5-10× speedup over CPU for forward/backward passes of small models.

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/ashyanSpada/llm_learn.git
cd llm_learn

# 2. Create and activate a Python virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# 3. Install dependencies (including PyTorch with MPS support)
pip install -e ".[dev]"

# 4. Run unit tests to verify your setup
pytest tests/ -v

# 5. Run a short LoRA fine-tuning (completes in ~2 min on Mac M4)
python -m llm_learn.train_lora \
    --model_name distilgpt2 \
    --data_path data/tiny_instruct.jsonl \
    --output_dir runs/lora_demo \
    --max_steps 20 \
    --batch_size 2

# 6. Run inference with the trained adapter
python -m llm_learn.infer \
    --model_name distilgpt2 \
    --adapter_path runs/lora_demo \
    --prompt "Explain what a neural network is in simple terms."

# 7. Evaluate before/after
python -m llm_learn.eval \
    --model_name distilgpt2 \
    --adapter_path runs/lora_demo \
    --prompts_file data/eval_prompts.json
```

Or use the Makefile shortcuts:

```bash
make setup      # install deps
make test       # run tests
make train      # short LoRA training run
make infer      # run inference
make eval       # run evaluation
make lint       # ruff check + format check
```

## Learning Modules

| # | Module | Doc | Key script |
|---|---|---|---|
| 0 | What is fine-tuning? | [docs/00_overview.md](docs/00_overview.md) | — |
| 1 | macOS / Apple Silicon setup | [docs/01_environment_macos.md](docs/01_environment_macos.md) | `utils.py` |
| 2 | Dataset preparation | [docs/02_datasets.md](docs/02_datasets.md) | `data.py` |
| 3 | PEFT / LoRA fine-tuning | [docs/03_peft_lora.md](docs/03_peft_lora.md) | `train_lora.py` |
| 4 | Evaluation | [docs/04_evaluation.md](docs/04_evaluation.md) | `eval.py` |
| 5 | Deploy to Jetson (optional) | [docs/05_deploy_jetson.md](docs/05_deploy_jetson.md) | `infer.py` |

## Repository Layout

```
llm_learn/
├── README.md
├── pyproject.toml          # dependencies + tool config
├── Makefile                # convenience commands
├── data/
│   ├── tiny_instruct.jsonl # small example dataset
│   └── eval_prompts.json   # prompt suite for evaluation
├── docs/
│   ├── 00_overview.md
│   ├── 01_environment_macos.md
│   ├── 02_datasets.md
│   ├── 03_peft_lora.md
│   ├── 04_evaluation.md
│   └── 05_deploy_jetson.md
├── src/
│   └── llm_learn/
│       ├── __init__.py
│       ├── utils.py        # device selection, seed, logging
│       ├── data.py         # dataset loading + formatting
│       ├── train_lora.py   # LoRA fine-tuning script
│       ├── infer.py        # inference with base + adapter
│       └── eval.py         # lightweight evaluation
└── tests/
    ├── test_data.py
    └── test_device.py
```

## Scaling Up

Once you've mastered the demo workflow:

1. **Larger models**: Try `gpt2` (124M), `EleutherAI/pythia-160m`, or `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.
2. **Larger datasets**: Use `alpaca_data_cleaned` or `Open-Platypus` from Hugging Face Hub.
3. **QLoRA (4-bit)**: Requires CUDA; use a free Colab T4 session with the same codebase.
4. **Preference tuning (DPO)**: See Hugging Face `trl` library for a drop-in extension.

## License

MIT
