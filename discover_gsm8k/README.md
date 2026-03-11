# Discover GSM8K

GSM8K rubric-discovery environment. Learn a scoring function `rubric_fn(input_text, response) -> float` from `(input, response, score)` examples; reward is test-set agreement rate, metric is Spearman(predicted, GT).

### Overview

- **Environment ID**: `discover_gsm8k`
- **Type**: multi-turn + tools (RLM)
- **Goal**: infer and implement `rubric_fn` from train examples, evaluated on held-out test examples
- **Key tool**: `get_rubric_run_result(fn_code, examples)` to run candidate rubric on examples

### Quickstart

Set `PRIME_API_KEY` for sandbox-backed runs, then:

```bash
prime eval run discover-gsm8k -m gpt-4.1-mini -a '{"dataset_path": "data.jsonl"}'
```

From the project directory:

```bash
cd discover_gsm8k
uv sync
uv run prime eval run discover-gsm8k -a '{"dataset_path": "data.jsonl"}'
```

### Config

`load_environment(config)` accepts a dict or `Config` with:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `dataset_path` | `str` | `""` | Path to JSONL (required) |
| `rlm_model` | `str` | `"gpt-4.1-mini"` | Sub-LLM for RLM |
| `max_turns` | `int` | `10` | Max RLM iterations |
| `max_examples` | `int \| None` | `None` | Cap on dataset size |
| `timeout_s` | `int` | `30` | Code execution timeout |
| `margin` | `float` | `0.3` | Agreement margin \|pred − expected\| ≤ margin |
| `parallelism` | `int` | `5` | Max parallel sub-LLM calls |

### Data format

JSONL rows: `train_examples`, `test_examples`, optional `task_hint`. Each example: `input`, `response`, `score`.

### Development

- **Package manager**: `uv`
- **Lint**: `ruff` (dev dependency)
