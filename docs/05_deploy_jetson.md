# Module 5 — Deploying to Jetson (Optional)

## Goal

Export the fine-tuned model to ONNX format and run inference on a Jetson Nano / Orin device.

---

## 1. Why ONNX for Jetson?

Jetson devices run NVIDIA Jetpack (Linux ARM64 + CUDA). The easiest path to efficient inference is:

```
PyTorch model  →  ONNX  →  TensorRT (optional)  →  Jetson inference
```

ONNX gives you:
- **Portability**: same file runs on CPU, CUDA, TensorRT.
- **Smaller binary**: no Python / PyTorch dependency at inference time.
- **Quantisation**: INT8 / FP16 conversion via TensorRT.

---

## 2. Prerequisites on your Mac (export step)

No extra installs needed if you already have the project set up. The export uses `torch.onnx.export`.

---

## 3. Export the merged model to ONNX

First, merge the LoRA adapter into the base weights:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "distilgpt2"
adapter_path = "runs/lora_demo"

tokenizer = AutoTokenizer.from_pretrained(model_name)
base = AutoModelForCausalLM.from_pretrained(model_name)
model = PeftModel.from_pretrained(base, adapter_path)
model = model.merge_and_unload()
model.eval()
```

Then export:

```python
dummy_input = tokenizer("Hello world", return_tensors="pt")["input_ids"]

torch.onnx.export(
    model,
    dummy_input,
    "runs/model.onnx",
    opset_version=14,
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}},
)
print("Exported to runs/model.onnx")
```

> **Note:** GPT-2-style models with `past_key_values` (KV-cache) require extra handling. For simplicity, this export disables KV-caching. Suitable for short sequences.

---

## 4. Transfer to Jetson

```bash
scp runs/model.onnx user@jetson-ip:/home/user/llm_demo/
```

---

## 5. Run inference on Jetson with ONNX Runtime

On the Jetson (JetPack 5.x with Python 3.8+):

```bash
pip install onnxruntime-gpu   # or onnxruntime for CPU-only
```

```python
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
sess = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])

prompt = "What is a transformer model?"
inputs = tokenizer(prompt, return_tensors="np")
logits = sess.run(["logits"], {"input_ids": inputs["input_ids"]})[0]
next_token_id = int(np.argmax(logits[0, -1]))
print("Next token:", tokenizer.decode([next_token_id]))
```

---

## 6. Optional: TensorRT acceleration

For maximum Jetson throughput, convert the ONNX model to a TensorRT engine:

```bash
# On Jetson (JetPack includes trtexec)
trtexec --onnx=model.onnx --saveEngine=model.trt --fp16
```

Then use `tensorrt` Python bindings for inference. This typically gives 2–5× speedup over ONNX Runtime on Jetson.

---

## 7. Jetson hardware considerations

| Jetson model | RAM | Recommended model size | Notes |
|---|---|---|---|
| Nano (4 GB) | 4 GB shared | ≤125M params | distilgpt2 fits comfortably |
| Orin Nano (8 GB) | 8 GB | ≤500M params | gpt2-medium possible |
| Orin NX (16 GB) | 16 GB | ≤1B params | TinyLlama / Phi-2 possible |

For the Nano with 4 GB, `distilgpt2` (82M params, ~330 MB) runs fine for inference at a few tokens/sec.

---

## 8. Summary

1. Merge LoRA adapter into base model on your Mac.
2. Export to ONNX with `torch.onnx.export`.
3. Copy to Jetson and run with ONNX Runtime.
4. Optionally convert to TensorRT for extra speed.

This path keeps the **training workflow entirely on your Mac** and uses the **Jetson purely for inference** — which matches its hardware strengths perfectly.
