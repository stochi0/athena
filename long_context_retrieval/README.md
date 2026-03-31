# Long-context retrieval (`long_context_retrieval`)

Verifiers **RLM** environment for question answering over one or more research-paper PDFs in a local workspace. The root model uses a Python REPL with SQL (registry), vectors, graphs, scoped filesystem tools, artifact registration, and `llm_batch()` for delegation.

**Environment ID:** `long_context_retrieval`

## What this environment does

1. Resolves each example’s `workspace_dir` (or builds from `pdf_dir` / `pdf_paths`) and stages `context_dir` for the RLM sandbox.
2. Initializes a SQLite registry (documents, artifacts, provenance, namespaces) under `.workspace_state/`.
3. Exposes shared tools to root and sub-LLMs; workspace-scoped reads are confined to the paper tree (not the host `tasks/` JSONL bundle).
4. Scores rollouts with the default rubric in `core/rewards.py` (expects a structured final answer; see **Answer contract**).

## Repository layout

| Path | Purpose |
|------|---------|
| `long_context_retrieval.py` | Entrypoint: `load_environment()` for Prime / verifiers |
| `core/` | `Config`, dataset build, `LongContextRetrievalEnv`, tools, workspace init, rubric |
| `scripts/build_dataset.py` | Fetch arXiv PDFs, write `workspace/` + `tasks/` bundle |
| `configs/eval/eval.toml` | Local eval preset (model, `[eval.env_args]`) |
| `configs/rl/long-context-retrieval.toml` | Hosted RL smoke (`[env.args]`) |

## Setup

Requires **Python 3.11+**. Use **`uv`** from this directory so `long_context_retrieval` and `core` resolve correctly.

```bash
cd long_context_retrieval
uv sync
uv run python -c "import long_context_retrieval; print(long_context_retrieval.load_environment)"
```

**Dev:** `uv sync --group dev` (Ruff, pytest).

## Running evaluations

Run commands from **`long_context_retrieval/`** (the directory that contains this package’s `pyproject.toml`).

### Local

Build data once (default output `./contexts/`):

```bash
uv run python scripts/build_dataset.py --query "cat:cs.IR" --max-papers 10
uv run prime eval run configs/eval/eval.toml --skip-upload
```

Set the API key expected by the TOML (commonly `PRIME_API_KEY` and `OPENAI_API_KEY` for Prime Inference). `configs/eval/eval.toml` sets `model` and `api_base_url`.

### CLI (`env_id`)

```bash
prime eval run long_context_retrieval -a '{"dataset_path": "contexts/tasks/dataset.jsonl", "max_examples": 2}'
```

### Hosted RL (smoke)

1. Push the environment: `prime env push --path .`
2. Set `[[env]].id` in `configs/rl/long-context-retrieval.toml` to your Hub slug.
3. Run: `prime rl run configs/rl/long-context-retrieval.toml -e WANDB_API_KEY -e OPENAI_API_KEY`

## On-disk layout (`build_dataset.py`)

Default `--output-dir` is `./contexts`.

| Path | Role |
|------|------|
| `contexts/workspace/` | PDFs, `papers.json`, `.workspace_state/` — filesystem tools use this tree |
| `contexts/tasks/` | `dataset.jsonl`, `hf/` (Hugging Face `save_to_disk`), `manifest.json` — harness only |

## Environment config

`load_environment(config, **kwargs)` merges dict kwargs into **`Config`** (`core/config.py`). Fields match the **`lhaw`** pattern for RLM + sandbox (e.g. `sub_model`, `max_turns`, `repl_language`, `sub_llm_max_turns`, `max_sub_llm_parallelism`, `max_output_length`, `pip_install_packages`, `code_execution_timeout`, `max_startup_wait_seconds`, `abort_on_code_timeout`, sandbox CPU/RAM/disk/GPU/timeout/image). Prompt verbosity is fixed in `core/config.py` constants.

**Passthrough:** any key **not** on `Config` is forwarded to **`RLMEnv`** (e.g. `sandbox_labels`, `sub_max_completion_tokens`).

**Aliases:** `rlm_model` in JSON is accepted as `sub_model`.

**TOML:** same shape as sibling **`lhaw/configs/eval/*.toml`** in the Athena repo — `[eval.env_args]` here, `[env.args]` in `configs/rl/long-context-retrieval.toml`.

## Python API

```python
from long_context_retrieval import load_environment

env = load_environment({"dataset_path": "contexts/tasks/dataset.jsonl"})
env = load_environment({"workspace_dir": "/abs/path/to/workspace"})
```

Advanced (custom rows): `create_environment(cfg=..., dataset=...)` in `core/environment.py` with a `datasets.Dataset` (`prompt`, `answer`, `info.workspace_dir`).

## Tools and answer contract

REPL tools: scoped SQL, vector/graph CRUD, filesystem IO (`workspace` vs `scratch`), artifact helpers — see `core/tools.py`.

Final model answer should be JSON:

```json
{
  "answer": "short answer text",
  "citations": [
    {
      "document_id": "doc-id",
      "path": "pdfs/paper.pdf",
      "page": 1,
      "excerpt": "supporting text"
    }
  ]
}
```

## Development

```bash
cd long_context_retrieval
uv sync --group dev
uv run pytest -q
uv run ruff check .
```

## Packaging

```bash
cd long_context_retrieval
uv build
prime env push --path .
```

PyPI distribution name: **`long-context-retrieval`**. After install, `import long_context_retrieval` (module `long_context_retrieval.py` + `core/` in the wheel per Hatch `only-include`).
