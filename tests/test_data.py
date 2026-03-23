"""Tests for dataset loading and prompt formatting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_learn.data import format_prompt, load_jsonl

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_jsonl(tmp_path: Path) -> Path:
    """Write a temporary JSONL file with a few examples and return its path."""
    examples = [
        {"instruction": "What is LoRA?", "input": "", "output": "LoRA is a PEFT method."},
        {
            "instruction": "Summarise this.",
            "input": "LoRA reduces trainable parameters.",
            "output": "LoRA is efficient.",
        },
        {"instruction": "Define backprop.", "input": "", "output": "Backprop computes gradients."},
    ]
    p = tmp_path / "test.jsonl"
    with p.open("w") as fh:
        for ex in examples:
            fh.write(json.dumps(ex) + "\n")
    return p


@pytest.fixture()
def real_data_path() -> Path:
    """Return the path to the bundled tiny dataset."""
    return Path(__file__).parent.parent / "data" / "tiny_instruct.jsonl"


# ---------------------------------------------------------------------------
# load_jsonl tests
# ---------------------------------------------------------------------------


def test_load_jsonl_basic(tiny_jsonl: Path) -> None:
    examples = load_jsonl(tiny_jsonl)
    assert len(examples) == 3
    assert examples[0]["instruction"] == "What is LoRA?"


def test_load_jsonl_returns_list_of_dicts(tiny_jsonl: Path) -> None:
    examples = load_jsonl(tiny_jsonl)
    for ex in examples:
        assert isinstance(ex, dict)


def test_load_jsonl_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_jsonl(tmp_path / "nonexistent.jsonl")


def test_load_jsonl_invalid_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"ok": true}\nnot valid json\n')
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_jsonl(bad)


def test_load_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "blanks.jsonl"
    p.write_text('\n{"instruction": "hi", "output": "hey"}\n\n')
    examples = load_jsonl(p)
    assert len(examples) == 1


def test_load_real_dataset(real_data_path: Path) -> None:
    """The bundled dataset should load without errors and have >=10 examples."""
    examples = load_jsonl(real_data_path)
    assert len(examples) >= 10
    for ex in examples:
        assert "instruction" in ex
        assert "output" in ex


# ---------------------------------------------------------------------------
# format_prompt tests
# ---------------------------------------------------------------------------


def test_format_prompt_no_input() -> None:
    ex = {"instruction": "What is AI?", "input": "", "output": "AI is..."}
    result = format_prompt(ex)
    assert "### Instruction:" in result
    assert "What is AI?" in result
    assert "### Response:" in result
    assert "AI is..." in result
    assert "### Input:" not in result


def test_format_prompt_with_input() -> None:
    ex = {"instruction": "Summarise.", "input": "Long text here.", "output": "Short."}
    result = format_prompt(ex)
    assert "### Input:" in result
    assert "Long text here." in result


def test_format_prompt_missing_instruction() -> None:
    with pytest.raises(KeyError):
        format_prompt({"input": "", "output": "x"})


def test_format_prompt_output_included() -> None:
    ex = {"instruction": "Define X.", "output": "X is Y."}
    result = format_prompt(ex)
    assert "X is Y." in result


def test_format_prompt_whitespace_input_treated_as_empty() -> None:
    """An input that is only whitespace should use the no-input template."""
    ex = {"instruction": "hi", "input": "   ", "output": "hello"}
    result = format_prompt(ex)
    assert "### Input:" not in result
