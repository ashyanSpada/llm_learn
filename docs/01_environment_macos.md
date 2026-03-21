# Module 1 — Environment Setup for macOS Apple Silicon

## Goal

Get a working Python environment that uses **Apple's MPS (Metal Performance Shaders)** backend for PyTorch on your Mac mini M4.

---

## 1. Install Homebrew (if not already installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

---

## 2. Install Python 3.11 via conda (recommended)

Using [Miniforge](https://github.com/conda-forge/miniforge) (ARM-native conda distribution):

```bash
# Download and install Miniforge for Apple Silicon
brew install miniforge

# Create a dedicated environment
conda create -n llm_learn python=3.11 -y
conda activate llm_learn
```

Alternatively, use [uv](https://github.com/astral-sh/uv):

```bash
brew install uv
uv venv --python 3.11 .venv
source .venv/bin/activate
```

---

## 3. Install PyTorch with MPS support

PyTorch ships with MPS support since version 1.12. No extra build steps are needed:

```bash
# Standard pip install — the macOS wheel includes MPS backend
pip install torch torchvision torchaudio
```

Verify MPS is available:

```python
import torch
print(torch.backends.mps.is_available())   # should print True on Mac M1/M2/M3/M4
print(torch.backends.mps.is_built())       # should print True
```

---

## 4. Install this project

```bash
git clone https://github.com/ashyanSpada/llm_learn.git
cd llm_learn
pip install -e ".[dev]"
```

---

## 5. Verify the full setup

```bash
pytest tests/ -v
```

All tests should pass. The `test_device.py` test confirms MPS detection on Mac.

---

## 6. Known limitations on Apple Silicon

| Feature | Status on MPS | Notes |
|---|---|---|
| `float16` (fp16) inference | ✅ Works | Default for `distilgpt2` |
| `bfloat16` training | ⚠️ Partial | Some ops fall back to CPU; use `float32` if you hit errors |
| `bitsandbytes` 4-bit / 8-bit | ❌ CUDA-only | Skip for now; learn QLoRA on free Colab T4 |
| FlashAttention | ❌ Not available | Use standard attention |
| Multi-GPU | ❌ N/A | Single-device only |
| `torch.compile` | ⚠️ Experimental | Works but can be slow to compile |

**Practical tips:**

- Set `--dtype float32` if you hit MPS numerical errors during training.
- If a step hangs, add `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` to your environment variables to disable MPS memory limit (use with care).
- Unified memory means CPU and GPU share RAM — keep your total model + batch size within ~60% of physical RAM for headroom.

---

## 7. Optional: Monitoring MPS usage

```bash
# CPU / memory usage
top -pid $(pgrep -f train_lora)

# Powermetrics (shows GPU utilization on Apple Silicon)
sudo powermetrics --samplers gpu_power -i 1000
```

Or use **Activity Monitor → GPU History** for a visual view.
