# LHAW RLM (`lhaw_rlm`)

Verifiers environment for the **LHAW** clarification loop on the [ScaleAI/lhaw](https://huggingface.co/datasets/ScaleAI/lhaw) dataset. Models may call `ask_user` against underspecified prompts; scoring depends on the chosen reward mode (see below).

## Reward modes

| Mode | Behavior |
|------|----------|
| `reconstruction_judge` | Clarification-focused task; an LLM judge scores how well the clarified task was reconstructed. |
| `native_reward` | Same `ask_user` interaction; reward comes from benchmark-native results in example metadata or linked result files. |

Set `reward_mode` under `[eval.env_args]` in your TOML. For `native_reward`, examples should carry `native_result` or `native_result_path`; optional fields like `native_trials`, `native_baseline_trials`, and `native_summary` support offline metrics (`pass@3`, `Ask%`, `Avg/Traj`, `Gain/Q`, etc.).

## Repository layout

| Path | Purpose |
|------|---------|
| `lhaw_rlm.py` | Entrypoint: `load_environment()` for Prime / verifiers |
| `core/` | Config, dataset transform, `LHAWRLMEnv`, judge rubric, native reward helpers |
| `configs/eval/` | `*.toml` presets (standard smoke, scale, model tiers, dataset slices, dimensions, reward modes) |
| `configs/eval/standard.toml` | Moderate local smoke config |
| `configs/endpoints.toml` | Optional shared endpoint registry |
| `scripts/launch_hosted_evals.sh` | Runs high-signal presets on Prime **hosted** workers |
| `docs/hf_dataset_schema.md` | Hugging Face dataset column reference |
| `tests/` | Pytest suite |

## Setup

Requires **Python 3.11+**. Dependencies are in `pyproject.toml` (`verifiers`, `datasets`).

```bash
cd lhaw
uv sync
uv run python -c "import lhaw_rlm; print(lhaw_rlm.load_environment)"
```

**Dev dependencies:** `uv sync --group dev` (Ruff, pytest).

## Running evaluations

Run commands from the **`lhaw/`** directory so `lhaw_rlm` and `core` resolve on `sys.path`.

### Local

```bash
uv run prime eval run configs/eval/standard.toml
```

Set the API key expected by the TOML (commonly `PRIME_API_KEY`). Add or reference `configs/endpoints.toml` if you use a shared endpoint registry.

### Hosted (Prime Evals)

Hosted jobs run on Prime infrastructure and show up in **Prime Evals**. You need a **published** environment (`prime env push`) and a Hub slug the launcher can target.

The script rewrites each preset to set `env_id` from `PRIME_EVAL_ENV_ID` and strips `env_dir_path` so workers load the package from the Hub. It does **not** use `--env-path` (that would pin to a local checkout and break hosted). Secrets are expected from the Environments Hub flow, not from ad-hoc `.env` or `--custom-secrets` in this script.

```bash
cd lhaw
export PRIME_API_KEY=...
export PRIME_EVAL_ENV_ID=your-org/lhaw_rlm   # optional; see below
prime config set-api-key "$PRIME_API_KEY"
# prime env push   # optional: publish before evals
./scripts/launch_hosted_evals.sh
```

**`PRIME_EVAL_ENV_ID`:** use the env var if set; else read `owner/name` from `.prime/.env-metadata.json` after a local `prime env push`; else default `stochi0/lhaw_rlm`.

**Presets (in run order):** `slice_outcome_critical`, `slice_benign`, `slice_divergent`, `slice_swe_bench`, `slice_mcp_atlas`, `slice_agent_company`, `dim_goal`, `dim_constraint`, `dim_input`, `dim_context`, `dim_goal_and_constraint`.

**Tuning:** `LHAW_HOSTED_POLL_INTERVAL` (default `30`), `LHAW_HOSTED_TIMEOUT_MINUTES` (default `180`). Each run uses `--allow-sandbox-access`, `--allow-instances-access`, and `--eval-name lhaw-<config-stem>`.

For CI, run the same script from `lhaw/` with `PRIME_API_KEY` and `PRIME_EVAL_ENV_ID` and the `prime` CLI available.

## Logging and debug UI

With the default Rich TUI, worker logs are tailed from `<run_dir>/eval.log` (verifiers writes this when `debug = false` in the eval TOML). For tqdm plus plain console logging, add `--debug`, e.g. `prime eval run configs/eval/standard.toml --debug`.

## Development

```bash
cd lhaw
uv sync --group dev
uv run pytest
uv run ruff check .
```

## Packaging and Environments Hub

```bash
cd lhaw
uv build
prime env push
```

PyPI package name: **`lhaw_rlm`**. After install, `import lhaw_rlm`. The wheel includes `lhaw_rlm.py` and `core/` (Hatch `only-include`).
