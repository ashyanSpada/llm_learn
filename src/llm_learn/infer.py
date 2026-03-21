"""Run inference with a base model, optionally loading a LoRA adapter.

Usage::

    python -m llm_learn.infer --model_name distilgpt2 --prompt "What is LoRA?"
    python -m llm_learn.infer --model_name distilgpt2 --adapter_path runs/lora_demo \\
        --prompt "What is LoRA?"
"""

from __future__ import annotations

import argparse

from llm_learn.utils import get_device, get_logger, set_seed

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run inference with a causal LM and optional LoRA adapter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_name", default="distilgpt2", help="Base model ID or path.")
    p.add_argument("--adapter_path", default=None, help="Path to LoRA adapter (optional).")
    p.add_argument("--prompt", default="What is a neural network?", help="Input prompt.")
    p.add_argument("--max_new_tokens", type=int, default=100, help="Max new tokens to generate.")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    p.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling p.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--no_mps", action="store_true", help="Disable MPS on Apple Silicon.")
    return p.parse_args()


def run_inference(
    model_name: str,
    prompt: str,
    adapter_path: str | None = None,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 42,
    prefer_mps: bool = True,
) -> str:
    """Generate text from a causal LM.

    Args:
        model_name: Hugging Face model identifier.
        prompt: Text prompt to continue.
        adapter_path: Path to a saved LoRA adapter directory, or ``None`` for
            base model only.
        max_new_tokens: Number of tokens to generate.
        temperature: Sampling temperature (higher = more random).
        top_p: Nucleus sampling probability mass.
        seed: Random seed.
        prefer_mps: Use MPS on Apple Silicon when available.

    Returns:
        Generated text string (excluding the prompt).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_seed(seed)
    device = get_device(prefer_mps=prefer_mps)
    logger.info("Device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path if adapter_path else model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

    if adapter_path:
        from peft import PeftModel

        logger.info("Loading LoRA adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)

    model = model.to(device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    result = run_inference(
        model_name=args.model_name,
        prompt=args.prompt,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        prefer_mps=not args.no_mps,
    )
    print("\n=== Prompt ===")
    print(args.prompt)
    print("\n=== Response ===")
    print(result)


if __name__ == "__main__":
    main()
