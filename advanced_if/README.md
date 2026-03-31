# Advanced IF (`advanced_if`)

Verifiers **single-turn** environment over [facebook/AdvancedIF](https://huggingface.co/datasets/facebook/AdvancedIF): the model reads a full conversation trajectory and emits a JSON rubric list; an LLM judge scores alignment with the gold rubrics.

**Environment ID (Prime / verifiers / Hub):** `advanced_if` — matches the `env_id` in `advanced_if.py`. The installable package name in `pyproject.toml` is `advanced-if` (hyphens); import module is `advanced_if`.

## What this environment does

1. Builds rollouts from the Hub row: `conversation_history` is rendered as role-tagged text; gold rubrics come from `prompt_metadata.rubrics` (stored as the rollout `answer`).
2. The policy completes in one assistant turn with JSON of the form `{"rubrics": ["…", …]}`.
3. `AdvancedIFJudgeRubric` (`vf.JudgeRubric`) calls a judge model; reward is the mean of three booleans the judge must return as JSON: `coverage`, `faithful`, `non_redundant` (no synonyms or loose coercion).
4. Optional `attach_dataset_stats` attaches split-level histograms to the rubric as `dataset_stats` for custom metrics.

## Repository layout

| Path | Purpose |
|------|---------|
| `advanced_if.py` | Entrypoint: `AdvancedIFEnv`, `load_environment()` for Prime / verifiers |
| `core/config.py` | `EnvironmentConfig` |
| `core/dataset.py` | Hub → rollout rows, `analyze_dataset`, `build_dataset` |
| `core/rubrics.py` | `AdvancedIFJudgeRubric`, judge client wiring |
| `core/prompts.py` | System / user / judge prompt strings |
| `configs/debug.toml` | Local smoke eval preset |
| `configs/endpoints.toml` | Optional Prime Inference endpoint registry |

## Setup

Requires **Python 3.11+**. Use **`uv`** from this directory.

```bash
cd advanced_if
uv sync
uv run python -c "import advanced_if; print(advanced_if.load_environment)"
```

**Dev:** `uv sync --group dev` (Ruff).

## Running evaluations

Run commands from **`advanced_if/`** (the directory that contains this package’s `pyproject.toml`).

### Local

```bash
uv run prime eval run configs/debug.toml
```

Set the API key for the judge client (default `judge_client_config.api_key_var` is `PRIME_API_KEY`). Point TOML at `configs/endpoints.toml` if you use a shared registry (`endpoints_path` in the eval config).

Add your own `configs/eval.toml` by copying a sibling env’s eval TOML shape: `[[eval]]`, `env_id = "advanced_if"`, `env_dir_path = "."`, and `[eval.env_args]` for `EnvironmentConfig` fields.

## Environment config

`load_environment(config, **kwargs)` builds an **`EnvironmentConfig`** (`core/config.py`). Dict kwargs are merged the same way as a single mapping.

| Field | Role |
|------|------|
| `dataset_name` / `dataset_split` | Hub dataset (default `facebook/AdvancedIF` / `train`) |
| `max_examples` / `seed` | Subset and shuffle for `build_dataset` |
| `judge_model` / `judge_sampling_args` | Judge LLM id and sampling kwargs |
| `judge_client_config` | `verifiers.types.ClientConfig` (defaults: Prime Inference URL + `PRIME_API_KEY`) |
| `max_turns` | Fixed at `1` in practice (`SingleTurnEnv`) |
| `attach_dataset_stats` | If true, rubric receives `dataset_stats` from `analyze_dataset` |

The judge client’s `api_key_var` must be present in the process environment (`verifiers` `ensure_keys`); there is no file-based key fallback.

## Dataset snapshot (`train`)

| | |
|--:|--:|
| Rows | 1645 |
| `carried_context_multi_turn_eval_v5` | 736 |
| `system_steerability_v2` | 507 |
| `complex_if_single_turn_v5` | 402 |

Rubrics per example range roughly 1–20; full histograms are produced by `analyze_dataset` when `attach_dataset_stats` is enabled.
