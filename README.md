# LHAW RLM (`lhaw_rlm`)

Verifiers environment for the **LHAW** clarification loop with two explicit reward modes:

- `reconstruction_judge`: models see underspecified prompts from the [ScaleAI/lhaw](https://huggingface.co/datasets/ScaleAI/lhaw) dataset, may call `ask_user`, and are scored by an LLM judge on the clarified task.
- `native_reward`: models still interact through `ask_user`, but reward is read from benchmark-native downstream results supplied in the example metadata or linked result files.

## Layout

| Path | Role |
|------|------|
| `lhaw_rlm.py` | Entrypoint: `load_environment()` for Prime / verifiers |
| `core/` | Config, dataset transform, `LHAWRLMEnv`, judge rubric |
| `configs/eval.toml` | Example `prime eval run` smoke config |
| `configs/eval/` | Additional `eval*.toml` presets (scale, model tiers, dataset/ambiguity/dimension slices, reward modes) |
| `scripts/run_lhaw_high_signal_hosted.sh` | Run the research “high-signal” TOMLs on Prime **hosted** workers (`--hosted`) |
| `docs/hf_dataset_schema.md` | HF dataset column reference |

## Setup

Python 3.11+. Dependencies are declared in `pyproject.toml` (`verifiers`, `datasets`).

```bash
uv sync
uv run python -c "import lhaw_rlm; print(lhaw_rlm.load_environment)"
```

Dev tools: `uv sync --group dev` (Ruff, pytest).

## Evaluation

From the repo root (so `lhaw_rlm` and `core` resolve on `sys.path`):

```bash
uv run prime eval run configs/eval.toml
```

Set the API key named in `configs/eval.toml` (e.g. `PRIME_API_KEY`). Optionally add `configs/endpoints.toml` for shared endpoint registry.

### Hosted high-signal evals (Prime Evals)

Hosted runs execute on Prime infrastructure and appear in **Prime Evals** automatically (no separate `prime eval push` for that flow). You need a **published** environment (`prime env push`) and your Hub slug (e.g. `primeintellect/lhaw_rlm`).

**Tunnel auth:** `prime config set-api-key` only configures this machine. Remote eval workers still need `PRIME_API_KEY` in their environment for Prime tunnel authentication; otherwise you may see `TunnelAuthError`. The script below exports `PRIME_API_KEY` and passes it (and optional `HF_TOKEN`) via `prime eval run --custom-secrets`. If you run `prime eval run ... --hosted` yourself, pass the same JSON, e.g. `--custom-secrets "$(python3 -c 'import json,os; print(json.dumps({"PRIME_API_KEY": os.environ["PRIME_API_KEY"]}))')"` with `PRIME_API_KEY` set in the shell.

```bash
export PRIME_API_KEY=... PRIME_EVAL_ENV_ID=your-org/lhaw_rlm
prime config set-api-key "$PRIME_API_KEY"
# Optional: publish latest env before evals
# LHAW_PUSH_ENV=1 ./scripts/run_lhaw_high_signal_hosted.sh
./scripts/run_lhaw_high_signal_hosted.sh
```

The script rewrites each TOML to use `PRIME_EVAL_ENV_ID` and drops `env_dir_path` (hosted loads the Hub package). Subsets: `LHAW_EVAL_SUBSET=all|ambiguity|domains|dimensions`. GitHub: **Actions → LHAW hosted evals** (secrets `PRIME_API_KEY`, `PRIME_EVAL_ENV_ID`).

### Reward Modes

Set `reward_mode` under `[eval.env_args]`:

- `reward_mode = "reconstruction_judge"` keeps the original clarification-only task contract and judge-based reward.
- `reward_mode = "native_reward"` switches prompts toward solving the task and expects `native_result` or `native_result_path` in each example's metadata. Optional aggregate fields like `native_trials`, `native_baseline_trials`, and `native_summary` enable paper-style offline metrics such as `pass@3`, `Ask%`, `Avg/Traj`, and `Gain/Q`.

**Logs:** With the default Rich TUI, worker logs are tailed from `<run_dir>/eval.log`, which verifiers only writes when `debug = false` in the eval TOML. For plain tqdm + console logging (no TUI), run `prime eval run configs/eval.toml --debug`.

## Packaging and Prime Hub

Build:

```bash
uv build
```

Push to the Environments Hub (from this directory):

```bash
prime env push
```

The PyPI distribution name is `lhaw_rlm`; after install, use `import lhaw_rlm`. The wheel ships `lhaw_rlm.py` and `core/` via Hatch `only-include`.

## CI

Manual workflow **LHAW hosted evals** (`.github/workflows/lhaw-hosted-evals.yml`) runs the high-signal hosted bundle; configure repository secrets `PRIME_API_KEY` and `PRIME_EVAL_ENV_ID`.
