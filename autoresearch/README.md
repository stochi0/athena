# Autoresearch

Autonomous LLM training research as a Verifiers environment, trainable with prime-rl.

The agent gets a sandbox with the [autoresearch](https://github.com/karpathy/autoresearch) repo. It may only edit `train.py` and run 5-minute training experiments via the root tool. Goal: **minimize validation bits-per-byte (val_bpb)**. Reward: `1 / (1 + best_val_bpb)` (lower val_bpb ⇒ higher reward).

## Overview

- **Environment ID**: `autoresearch`
- **Type**: RLMEnv — Bash REPL + root tools (e.g. `run_training_tool`) + optional sub-LLM
- **Goal**: Achieve the lowest validation bits-per-byte on the fixed eval set by modifying `train.py` and running experiments.
- **Key tools**:
  - **call_bash_repl**: Edit files (e.g. `train.py`) and run shell commands in the sandbox.
  - **run_training_tool()**: Root tool that runs one 5-minute experiment (`uv run train.py` in the repo), parses `val_bpb` from output, updates best/run counts in state, and returns val_bpb and log summary.
  - **llm_batch()**: Optional sub-LLM for subtasks (e.g. summarising logs).
- **Completion**: When done, the agent must export `RLM_READY=1` and set `RLM_CONTENT` to a short summary of the best val_bpb.
- **Metrics**: `num_runs` (training runs executed), `best_val_bpb` (best val_bpb achieved), plus RLM monitor metrics.

## Quickstart

### Run (from repo root)

Requires Prime Sandboxes (and optionally GPU). Set `PRIME_API_KEY` and configure endpoints in `configs/endpoints.toml` if needed:

```bash
prime eval run autoresearch -n 2 -m openai/gpt-4.1-mini
```

With custom config (e.g. more turns, fewer examples):

```bash
prime eval run autoresearch -n 2 -m openai/gpt-4.1-mini -a '{"max_turns": 20, "num_examples": 3}'
```

### Run (from `autoresearch/`)

```bash
cd autoresearch
uv sync
uv run prime eval run autoresearch -n 2 -m openai/gpt-4.1-mini
```

### Programmatic

```python
from autoresearch.autoresearch import load_environment

env = load_environment()
# or with options:
env = load_environment({"max_turns": 20, "num_examples": 3})
```

### Hosted RL

To use the environment with [Lab Hosted Training](https://docs.primeintellect.ai):

1. Push the environment to the Hub (once):
   ```bash
   prime env push --path ./autoresearch -v PRIVATE
   ```
2. Use your Hub env ID (e.g. `YOUR_USERNAME/autoresearch`) in your RL config and run:
   ```bash
   prime rl run <your-config.toml> -e WANDB_API_KEY -e OPENAI_API_KEY
   ```

## Config

`load_environment(config, *, dataset_builder=None, **kwargs)` accepts:

- **config**: A `Config` instance, a dict (e.g. from `prime eval run -a '...'`), or `None`. Dict keys can be overridden by **kwargs**.
- **dataset_builder**: Optional callable `(num_examples: int) -> Dataset`. If provided, it is called with `num_examples` to build the dataset; each row must have `"question"`, `"task"`, and optionally `"info"`. If `None`, a default synthetic dataset is used (same question repeated `num_examples` times).

Config fields (when passing a dict or `Config`):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| **max_turns** | int | 10 | Max agent turns (each turn can include tool calls). |
| **docker_image** | str | `"ghcr.io/astral-sh/uv:python3.12-bookworm"` | Sandbox Docker image (needs git, uv; use CUDA image if `gpu_count > 0`). |
| **working_dir** | str | `"/workspace/autoresearch"` | Path inside the sandbox to the autoresearch repo. |
| **timeout_per_command_seconds** | int | 30 | Timeout for Bash REPL commands (e.g. edits). |
| **timeout_per_training_run_seconds** | int | 600 | Timeout for each `run_training_tool()` run (`uv run train.py`). |
| **gpu_count** | int | 1 | GPUs for the sandbox (1 recommended for training). |
| **memory_gb** | int | 16 | RAM for the sandbox. |
| **disk_size_gb** | int | 20 | Disk for the sandbox. |
| **sandbox_timeout_minutes** | int | 60 | Sandbox lifetime timeout (minutes). |
| **sandbox_cpu_cores** | int | 1 | CPU cores for the sandbox. |
| **start_command** | str \| None | None | If set, used as the sandbox start command; else default: clone repo, `uv sync`, `uv run prepare.py --num-shards 2`, then `exec tail -f /dev/null`. |
| **num_examples** | int | 5 | Number of dataset rows (for default synthetic dataset). |
| **repo_url** | str | `"https://github.com/karpathy/autoresearch.git"` | Git URL for the autoresearch repo. |
| **context_dir_name** | str | `"contexts"` | Directory name for per-example context (e.g. `contexts/0/`, `contexts/1/`). |

Validation: `num_examples >= 1`, `timeout_per_command_seconds >= 1`, `timeout_per_training_run_seconds >= 1`.

## Dataset rows and `info` (context_dir / context)

RLMEnv passes each dataset row’s **info** into `state["info"]` and uses it when building the REPL filesystem:

- **info["context_dir"]**: Path to a directory on the host. RLMEnv copies this directory into the rollout’s REPL fs before the worker starts (e.g. per-example configs or data).
- **info["context"]**: Optional JSON-serializable data; RLMEnv writes it to a file in the REPL fs (legacy builtin context).

Default rows use `info: {}`. To supply context, use **dataset_builder** and return rows with `"info": {"context_dir": "/path/to/dir"}` or `"info": {"context": {...}}`. Prefer placing context under **contexts/** (e.g. `contexts/0/`, `contexts/1/`) and set `info["context_dir"]` to the resolved path.

## Sandbox setup

By default, `load_environment()` builds a **start_command** that: clones the autoresearch repo into **working_dir**, runs `uv sync` and `uv run prepare.py --num-shards 2`, then keeps the container alive with `exec tail -f /dev/null`. For production you may want:

- A **custom Docker image** with the repo and data pre-baked (set `docker_image` and `start_command`).
- **GPU**: `gpu_count=1` and a CUDA image so `uv run train.py` can use the GPU.

## Reward and metrics

- **Reward**: `1 / (1 + best_val_bpb)`. Lower val_bpb ⇒ higher reward. If no successful run with a parsed val_bpb, reward is 0.
- **num_runs**: Number of training runs executed in the rollout (from `run_training_tool`).
- **best_val_bpb**: Best val_bpb achieved (lower is better; -1 if none).

## Requirements

- Prime Sandboxes (and optionally GPU quota) for running training in the sandbox.
- No extra API keys beyond what Verifiers/prime use for inference.

## Development

- **Package manager**: `uv`
- **Lint**: `ruff` (if configured)
