"""Minimal LoRA fine-tuning script.

Run from the repo root::

    python -m llm_learn.train_lora --help

Defaults are intentionally tiny so the demo completes quickly on a Mac mini M4
or any CPU-only machine.
"""

from __future__ import annotations

import argparse

from llm_learn.utils import get_device, get_logger, set_seed

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoRA fine-tuning of a causal language model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model_name", default="distilgpt2", help="Hugging Face model ID or local path."
    )
    p.add_argument("--data_path", default="data/tiny_instruct.jsonl", help="Path to JSONL dataset.")
    p.add_argument("--output_dir", default="runs/lora_demo", help="Directory to save adapter.")
    p.add_argument("--lora_r", type=int, default=8, help="LoRA rank.")
    p.add_argument("--lora_alpha", type=int, default=16, help="LoRA scaling factor.")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    p.add_argument(
        "--target_modules",
        nargs="+",
        default=None,
        help=(
            "Weight matrix names to apply LoRA to. Defaults to auto-detect attention projections."
        ),
    )
    p.add_argument("--max_steps", type=int, default=20, help="Maximum training steps.")
    p.add_argument("--batch_size", type=int, default=2, help="Per-device training batch size.")
    p.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps.")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    p.add_argument("--max_length", type=int, default=256, help="Max token length per example.")
    p.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Model dtype. Use float32 for MPS stability.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--no_mps", action="store_true", help="Disable MPS even on Apple Silicon.")
    return p.parse_args()


def _get_target_modules(model_name: str) -> list[str]:
    """Return sensible default target modules based on model architecture."""
    name = model_name.lower()
    if "gpt2" in name or "distilgpt" in name:
        return ["c_attn"]
    if "gpt_neox" in name or "pythia" in name:
        return ["query_key_value"]
    if "llama" in name or "mistral" in name or "phi" in name:
        return ["q_proj", "v_proj"]
    # Generic fallback — let peft auto-detect
    return ["q_proj", "v_proj"]


def train(args: argparse.Namespace) -> None:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    from llm_learn.data import build_dataset

    set_seed(args.seed)
    device = get_device(prefer_mps=not args.no_mps)
    logger.info("Using device: %s", device)

    # dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    # Load tokenizer
    logger.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    logger.info("Loading base model: %s", args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch_dtype)

    # Wrap with LoRA
    target_modules = args.target_modules or _get_target_modules(args.model_name)
    logger.info(
        "Applying LoRA — rank=%d, alpha=%d, modules=%s",
        args.lora_r,
        args.lora_alpha,
        target_modules,
    )
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    train_ds, eval_ds = build_dataset(args.data_path, tokenizer, max_length=args.max_length)

    # Training arguments
    # Hugging Face Trainer does not natively support MPS via device_map, so we
    # set no_cuda=True and move the model manually when on MPS.
    use_mps = device.type == "mps"
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=5,
        save_steps=args.max_steps,
        save_total_limit=1,
        evaluation_strategy="no",
        fp16=False,  # avoid fp16 issues on MPS/CPU
        bf16=False,
        dataloader_pin_memory=False,
        report_to="none",
        use_mps_device=use_mps,
        no_cuda=not torch.cuda.is_available(),
        seed=args.seed,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if len(eval_ds) > 0 else None,
        data_collator=data_collator,
    )

    logger.info("Starting training for %d steps…", args.max_steps)
    trainer.train()

    logger.info("Saving adapter to %s", args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Done.")


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
