# LHAW RLM (`lhaw_rlm`)

Verifiers environment for the **LHAW** clarification loop: models see underspecified prompts from the [ScaleAI/lhaw](https://huggingface.co/datasets/ScaleAI/lhaw) dataset, may call `ask_user` against a simulated user, and are scored with an LLM judge on the clarified task.

## Layout

| Path | Role |
|------|------|
| `lhaw_rlm.py` | Entrypoint: `load_environment()` for Prime / verifiers |
| `core/` | Config, dataset transform, `LHAWRLMEnv`, judge rubric |
| `configs/eval.toml` | Example `prime eval run` smoke config |
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

GitHub Actions runs Ruff, pytest, `uv build`, and a clean-venv import check (see `.github/workflows/ci.yml`).
