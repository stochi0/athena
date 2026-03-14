# Discover GSM8K

Rubric-discovery environment for GSM8K.

Learn a scoring function `rubric_fn(prompt, completion) -> float` from `(prompt, completion, score)` examples. Evaluation is on held-out test examples (agreement rate; Spearman vs GT).

## Overview

- **Environment ID**: `discover_gsm8k`
- **Type**: multi-turn + tools (RLM)
- **Goal**: infer and implement `rubric_fn` from train examples, evaluated on held-out test examples
- **Key tool**: `get_rubric_run_result(fn_code, examples)` to run candidate rubric on examples

## Quickstart

### Run (from repo root)

Set `PRIME_API_KEY` for sandbox-backed runs, then:

```bash
prime eval run discover-gsm8k -m gpt-4.1-mini -a '{"dataset_path": "data/data.jsonl"}'
```

### Run (from `discover_gsm8k/`)

```bash
cd discover_gsm8k
uv sync
uv run prime eval run discover-gsm8k -a '{"dataset_path": "data/data.jsonl"}'
```

### Hosted RL (2-example smoke test)

To run a minimal RL job on 2 examples via [Lab Hosted Training](https://docs.primeintellect.ai):

1. Push the environment to the Hub (once):
   ```bash
   prime env push --path . -v PRIVATE
   ```
2. In `config/rl_test_2examples.toml`, set `[[env]].id` to your Hub env (e.g. `YOUR_USERNAME/discover_gsm8k`).
3. Start the run:
   ```bash
   prime rl run config/rl_test_2examples.toml
   ```

The config uses `max_examples = 2`, `max_steps = 2`, and `batch_size = 2` for a quick smoke test.

## Config

`load_environment(config)` accepts a dict (or `Config`) with:

- `**dataset_path**` (`str`, required): path to JSONL (e.g. `data/data.jsonl`)
- `**max_train_per_task**` (`int | None`, default `2`): max train `(input, response, score)` examples per task in `contexts/<i>/task.json`; use `None` to use all
- `**max_test_per_task**` (`int | None`, default `5`): max test examples per task (in state for reward); use `None` to use all
- `**rlm_model**` (`str`, default `"gpt-4.1-mini"`): sub-LLM for RLM
- `**max_turns**` (`int`, default `100`): max RLM iterations
- `**max_examples**` (`int | None`, default `None`): cap number of tasks (rows)
- `**timeout_s**` (`int`, default `30`): code execution timeout
- `**margin**` (`float`, default `0.3`): agreement threshold |pred - expected| \le margin
- `**parallelism**` (`int`, default `5`): max parallel sub-LLM calls

Example: limit contexts to 3 train and 2 test per task:

```bash
prime eval run discover-gsm8k -a '{"dataset_path": "data/data.jsonl", "max_train_per_task": 3, "max_test_per_task": 2}'
```

## Data format

JSONL rows contain:

- `**train_examples**`: list of `{ prompt, completion, score }`
- `**test_examples**`: list of `{ prompt, completion, score }`
- `**task_hint**` (optional): string

This matches the verifiers naming convention:

- `**prompt**`: input text (what the model is asked to score / respond to)
- `**completion**`: model output text being evaluated

## Dataset generation

Dataset rows can be generated from verifiers-based source environments using `scripts/generate_dataset.py` and a YAML config.

### Requirements on source environments

Assuming the environment is implemented on top of `verifiers` and can be loaded with `verifiers.load_environment(env_id)`:

- **Dataset access**
  - The env must implement either `get_dataset(n, seed)` or `get_eval_dataset(n, seed)`.
  - Each dataset row must expose at least:
    - `prompt`: either
      - a string, or
      - a list of messages like `[{ "role": "user", "content": "..."}, ...]` with at least one non-empty user message.
    - Optionally `answer`, `task`, and `info` (used when building the scoring `State`).
- **Rubric / scoring**
  - The env must define a rubric such that `env.rubric.score_group([state])`:
    - runs without error, and
    - **sets** `state["reward"]` **to a numeric score [0, 1] for the** `(prompt, completion, answer, info, task)` tuple.

### YAML config (single or multiple source envs)

Example (single env):

```bash
uv run scripts/generate_dataset.py --config config/envs_gsm8k.yaml
uv run scripts/generate_dataset.py --config config/envs_ifeval.yaml
```

Example config (multiple envs in one file):

```yaml
out: data/mixed.jsonl

envs:
  - source_env: primeintellect/gsm8k
    n: 50
    train_per_task: 2
    test_per_task: 2
  - source_env: arcee-ai/ifeval
    n: 50
    train_per_task: 2
    test_per_task: 2
```

For each `env` entry:

- `**source_env**` (required): environment id string passed to `verifiers.load_environment`.
- `**n**` (optional, default `50`): number of source examples to sample.
- `**train_per_task**`, `**test_per_task**` (optional): caps on the number of train / test `(prompt, completion, score)` examples per JSONL row.
- `**responses_per_example**`, `**train_ratio**`, `**temperatures**`, `**seed**`, `**task_hint**` (optional): advanced knobs; see `scripts/generate_dataset.py` for details and validation rules.

## Development

- **Package manager**: `uv`
- **Lint**: `ruff`

