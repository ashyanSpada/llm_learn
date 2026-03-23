"""Lightweight evaluation: perplexity and side-by-side prompt comparison.

Usage::

    python -m llm_learn.eval --model_name distilgpt2 \\
        --prompts_file data/eval_prompts.json

    python -m llm_learn.eval --model_name distilgpt2 \\
        --adapter_path runs/lora_demo \\
        --prompts_file data/eval_prompts.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from llm_learn.utils import get_device, get_logger, set_seed

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a causal LM (optionally with a LoRA adapter).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_name", default="distilgpt2", help="Base model ID or path.")
    p.add_argument("--adapter_path", default=None, help="Path to LoRA adapter (optional).")
    p.add_argument(
        "--prompts_file",
        default="data/eval_prompts.json",
        help="JSON file with list of prompt strings.",
    )
    p.add_argument("--max_new_tokens", type=int, default=80, help="Tokens to generate per prompt.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--no_mps", action="store_true", help="Disable MPS on Apple Silicon.")
    return p.parse_args()


def compute_perplexity(model: object, tokenizer: object, texts: list[str], device: object) -> float:
    """Compute average perplexity over a list of texts.

    Args:
        model: A Hugging Face causal LM.
        tokenizer: Matching tokenizer.
        texts: List of text strings.
        device: torch.device to run on.

    Returns:
        Average perplexity (float).
    """
    import torch

    total_loss = 0.0
    count = 0
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            loss = model(**inputs, labels=inputs["input_ids"]).loss
        total_loss += loss.item()
        count += 1
    avg_loss = total_loss / max(count, 1)
    return math.exp(avg_loss)


def generate_responses(
    model: object,
    tokenizer: object,
    prompts: list[str],
    device: object,
    max_new_tokens: int = 80,
    seed: int = 42,
) -> list[str]:
    """Generate one response per prompt.

    Args:
        model: A Hugging Face causal LM.
        tokenizer: Matching tokenizer.
        prompts: List of prompt strings.
        device: torch.device.
        max_new_tokens: Tokens to generate per prompt.
        seed: Random seed.

    Returns:
        List of generated response strings.
    """
    import torch

    set_seed(seed)
    responses = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy for reproducibility
                pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        responses.append(tokenizer.decode(new_ids, skip_special_tokens=True))
    return responses


def _load_model(model_name: str, adapter_path: str | None, device: object) -> tuple:
    """Load a model (+ optional adapter) onto *device*."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter_path if adapter_path else model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(prefer_mps=not args.no_mps)
    logger.info("Device: %s", device)

    prompts_path = Path(args.prompts_file)
    if not prompts_path.exists():
        logger.error("Prompts file not found: %s", prompts_path)
        raise SystemExit(1)

    with prompts_path.open(encoding="utf-8") as fh:
        prompts = json.load(fh)

    if not isinstance(prompts, list) or not prompts:
        logger.error("Prompts file must contain a non-empty JSON array of strings.")
        raise SystemExit(1)

    logger.info("Loaded %d evaluation prompts.", len(prompts))

    # Load base model
    logger.info("Loading base model: %s", args.model_name)
    base_model, tokenizer = _load_model(args.model_name, None, device)
    base_ppl = compute_perplexity(base_model, tokenizer, prompts, device)
    base_responses = generate_responses(
        base_model, tokenizer, prompts, device, max_new_tokens=args.max_new_tokens
    )
    logger.info("Base model perplexity: %.2f", base_ppl)

    tuned_ppl = None
    tuned_responses = None
    if args.adapter_path:
        logger.info("Loading tuned model (adapter: %s)", args.adapter_path)
        tuned_model, _ = _load_model(args.model_name, args.adapter_path, device)
        tuned_ppl = compute_perplexity(tuned_model, tokenizer, prompts, device)
        tuned_responses = generate_responses(
            tuned_model, tokenizer, prompts, device, max_new_tokens=args.max_new_tokens
        )
        logger.info("Tuned model perplexity: %.2f", tuned_ppl)

    # Print results
    print("\n" + "=" * 80)
    print(f"  Base model perplexity : {base_ppl:.2f}")
    if tuned_ppl is not None:
        delta = base_ppl - tuned_ppl
        print(f"  Tuned model perplexity: {tuned_ppl:.2f}  (Δ {delta:+.2f})")
    print("=" * 80)

    for idx, prompt in enumerate(prompts):
        print(f"\n[Prompt {idx + 1}] {prompt}")
        print(f"  Base : {base_responses[idx][:200]}")
        if tuned_responses:
            print(f"  Tuned: {tuned_responses[idx][:200]}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
