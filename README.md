# Athena

Evals and environment packages (Prime / verifiers–style RLMs).

## Environments

| Directory | Description |
|-----------|-------------|
| [`lhaw/`](lhaw/README.md) | **LHAW RLM** — underspecified prompts, `ask_user` clarification, reconstruction judge / native reward (`lhaw_rlm`) |
| [`loca_bench_rlm/`](loca_bench_rlm/README.md) | LOCA Bench RLM |
| [`long_context_retrieval/`](long_context_retrieval/README.md) | Long-context retrieval |
| [`discover_gsm8k/`](discover_gsm8k/README.md) | Discover GSM8K |
| [`autoresearch/`](autoresearch/README.md) | Autoresearch |
| [`advanced_if/`](advanced_if/README.md) | **Advanced IF** — `facebook/AdvancedIF` rubric induction via **`RLMEnv`** (file-backed trajectories + partial judge tool, `advanced_if`) |

Each subdirectory is a self-contained package with its own `pyproject.toml`, `uv.lock`, and docs. Work inside that folder for install, eval, and packaging.
