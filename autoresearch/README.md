# Autoresearch

Autonomous LLM training research as a Verifiers environment, trainable with prime-rl.

The agent gets a sandbox with the [autoresearch](https://github.com/karpathy/autoresearch) repo. It may edit `train.py` and run 5-minute training experiments. Goal: **minimize validation bits-per-byte (val_bpb)**. Reward: `1 / (1 + best_val_bpb)` (lower val_bpb ⇒ higher reward).

## Overview

- **Environment ID**: `autoresearch`
- **Type**: SandboxEnv (stateful tools: `bash` + `run_training`)
- **Goal**: achieve the lowest validation bits-per-byte on the fixed eval set by modifying `train.py` and running experiments
- **Key tools**: `bash` (edit files), `run_training` (run `uv run train.py`, parse val_bpb, store in state)
- **Metrics**: `num_runs`, `best_val_bpb`

## Quickstart

### Run (from repo root)

Requires Prime Sandboxes (and optionally GPU). Set `PRIME_API_KEY` and configure endpoints in `configs/endpoints.toml` if needed:

```bash
prime eval run autoresearch -n 2 -m gpt-4.1-mini
```

### Run (from `autoresearch/`)

```bash
cd autoresearch
uv sync
uv run prime eval run autoresearch -n 2 -m gpt-4.1-mini
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

`load_environment(config)` accepts a dict (or keyword args) with:

- **max_turns** (`int`, default `10`): max agent turns (each turn can include tool calls)
- **docker_image** (`str`, default `"ghcr.io/astral-sh/uv:python3.12-bookworm"`): image for the sandbox (must have git, uv; use a CUDA image if `gpu_count > 0`)
- **working_dir** (`str`, default `"/workspace/autoresearch"`): path inside the sandbox to the autoresearch repo
- **timeout_per_command_seconds** (`int`, default `30`): timeout for bash commands (e.g. edits)
- **timeout_per_training_run_seconds** (`int`, default `600`): timeout for `run_training` (uv run train.py)
- **gpu_count** (`int`, default `1`): GPUs for the sandbox (1 recommended for training)
- **memory_gb** (`int`, default `16`): RAM for the sandbox
- **disk_size_gb** (`int`, default `20`): disk for the sandbox
- **start_command** (`str | None`, default `None`): if set, used as sandbox start command; else clones repo, runs `uv sync` and `prepare.py --num-shards 2`, then tails
- **num_examples** (`int`, default `5`): dataset size (synthetic prompts)
- **repo_url** (`str`, default `"https://github.com/karpathy/autoresearch.git"`): git URL to clone for the autoresearch repo

Example: more turns and 3 eval examples:

```bash
prime eval run autoresearch -a '{"max_turns": 20, "num_examples": 3}' -m gpt-4.1-mini
```

## Sandbox setup

By default, `load_environment()` uses a `start_command` that clones the autoresearch repo into the sandbox and runs `uv sync` and `prepare.py --num-shards 2`. For production you may want:

- A **custom Docker image** that already contains the repo and data (set `docker_image` and `start_command` in `load_environment`).
- **GPU**: `gpu_count=1` (and a CUDA image) so `uv run train.py` can use the GPU.

## Required

- Prime Sandboxes (and optionally GPU quota) for running training in the sandbox.
- No extra API keys beyond what Verifiers/prime use for inference.

## Development

- **Package manager**: `uv`
- **Lint**: `ruff` (if configured)
