"""Dataset loading and prompt-formatting utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_learn.utils import get_logger

logger = get_logger(__name__)

ALPACA_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
)

ALPACA_TEMPLATE_NO_INPUT = "### Instruction:\n{instruction}\n\n### Response:\n{output}"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file and return a list of dicts.

    Args:
        path: Path to the ``.jsonl`` file.

    Returns:
        List of parsed JSON objects (one per line).

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If a line cannot be parsed as JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    examples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno} of {path}: {exc}") from exc

    logger.info("Loaded %d examples from %s", len(examples), path)
    return examples


def format_prompt(example: dict[str, Any]) -> str:
    """Render a data example into the Alpaca instruction-tuning template.

    The template differs depending on whether an ``input`` field is present
    and non-empty.

    Args:
        example: Dict with keys ``instruction``, ``input`` (optional), and
            ``output``.

    Returns:
        Formatted prompt string.

    Raises:
        KeyError: If required keys are missing.
    """
    instruction = example["instruction"]
    output = example.get("output", "")
    inp = example.get("input", "").strip()

    if inp:
        return ALPACA_TEMPLATE.format(instruction=instruction, input=inp, output=output)
    return ALPACA_TEMPLATE_NO_INPUT.format(instruction=instruction, output=output)


def build_dataset(
    path: str | Path,
    tokenizer: Any,
    max_length: int = 256,
    split_ratio: float = 0.9,
) -> tuple[Any, Any]:
    """Load a JSONL file, format prompts, tokenise, and return train/eval splits.

    Args:
        path: Path to ``.jsonl`` dataset file.
        tokenizer: A Hugging Face tokenizer with ``__call__`` and
            ``pad_token`` attributes.
        max_length: Maximum token length per example (longer examples are
            truncated).
        split_ratio: Fraction of examples used for training (remainder for eval).

    Returns:
        Tuple of ``(train_dataset, eval_dataset)`` as
        :class:`datasets.Dataset` instances.
    """
    from datasets import Dataset

    examples = load_jsonl(path)
    texts = [format_prompt(ex) for ex in examples]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list]:
        encoded = tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    ds = Dataset.from_dict({"text": texts})
    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    ds.set_format("torch")

    split_idx = max(1, int(len(ds) * split_ratio))
    train_ds = ds.select(range(split_idx))
    eval_ds = ds.select(range(split_idx, len(ds)))

    logger.info("Dataset split: %d train / %d eval", len(train_ds), len(eval_ds))
    return train_ds, eval_ds
