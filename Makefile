.PHONY: help setup lint format test train infer eval clean

PYTHON ?= python
DATA   ?= data/tiny_instruct.jsonl
MODEL  ?= distilgpt2
OUTDIR ?= runs/lora_demo

help:   ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Install project + dev dependencies
	pip install -e ".[dev]"

lint:   ## Run ruff linter (check only)
	ruff check src/ tests/

format: ## Run ruff formatter + isort
	ruff format src/ tests/
	ruff check --fix src/ tests/

test:   ## Run unit tests with pytest
	pytest tests/ -v

train:  ## Run a short LoRA fine-tuning demo (20 steps)
	$(PYTHON) -m llm_learn.train_lora \
	  --model_name $(MODEL) \
	  --data_path  $(DATA) \
	  --output_dir $(OUTDIR) \
	  --max_steps  20 \
	  --batch_size 2 \
	  --grad_accum 4

infer:  ## Run inference with the trained adapter
	$(PYTHON) -m llm_learn.infer \
	  --model_name   $(MODEL) \
	  --adapter_path $(OUTDIR) \
	  --prompt       "What is a neural network?"

eval:   ## Evaluate base model vs. tuned adapter
	$(PYTHON) -m llm_learn.eval \
	  --model_name   $(MODEL) \
	  --adapter_path $(OUTDIR) \
	  --prompts_file data/eval_prompts.json

clean:  ## Remove training artifacts
	rm -rf runs/ __pycache__ .pytest_cache .ruff_cache
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -name '*.pyc' -delete
