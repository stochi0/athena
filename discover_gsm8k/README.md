# Discover GSM8K

Rubric-discovery environment for GSM8K.

Learn a scoring function `rubric_fn(input_text, response) -> float` from `(input, response, score)` examples. Evaluation is on held-out test examples (agreement rate; Spearman vs GT).

## Overview

- **Environment ID**: `discover_gsm8k`
- **Type**: multi-turn + tools (RLM)
- **Goal**: infer and implement `rubric_fn` from train examples, evaluated on held-out test examples
- **Key tool**: `get_rubric_run_result(fn_code, examples)` to run candidate rubric on examples

## Quickstart

### Run (from repo root)

Set `PRIME_API_KEY` for sandbox-backed runs, then:

```bash
prime eval run discover-gsm8k -m gpt-4.1-mini -a '{"dataset_path": "data.jsonl"}'
```

### Run (from `discover_gsm8k/`)

```bash
cd discover_gsm8k
uv sync
uv run prime eval run discover-gsm8k -a '{"dataset_path": "data.jsonl"}'
```

## Config

`load_environment(config)` accepts a dict (or `Config`) with:

- **`dataset_path`** (`str`, required): path to JSONL
- **`rlm_model`** (`str`, default `"gpt-4.1-mini"`): sub-LLM for RLM
- **`max_turns`** (`int`, default `10`): max RLM iterations
- **`max_examples`** (`int | None`, default `None`): cap dataset size
- **`timeout_s`** (`int`, default `30`): code execution timeout
- **`margin`** (`float`, default `0.3`): agreement threshold \(|pred - expected| \le margin\)
- **`parallelism`** (`int`, default `5`): max parallel sub-LLM calls

## Data format

JSONL rows contain:

- **`train_examples`**: list of `{ input, response, score }`
- **`test_examples`**: list of `{ input, response, score }`
- **`task_hint`** (optional): string

## Development

- **Package manager**: `uv`
- **Lint**: `ruff`
