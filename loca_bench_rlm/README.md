# LOCA Bench RLM

Standalone `verifiers`-style RLM environment for LOCA-bench task configs.

This directory mirrors the cleaner packaging style used by `athena/discover_gsm8k`:

- `loca_bench_rlm.py`: public `load_environment()` entrypoint
- `core/`: config, dataset loading, prompting, and evaluation helpers
- `task_configs/`: bundled smoke-test config assets
- `eval.toml`: ready-to-run local smoke eval

## What This Environment Does

For each rollout, the environment:

1. Loads a LOCA-bench task config JSON.
2. Dynamically instantiates the referenced LOCA task from the sibling codebase.
3. Copies only the agent-visible task artifacts into the RLM sandbox:
   - `agent_workspace`
   - `files`
   - `local_db`
4. Lets the model solve the task with `RLMEnv`.
5. Downloads the sandbox filesystem and reuses the original LOCA task evaluator via `env.step()`.

## Important Note

This environment is standalone in layout and packaging, but it still relies on a
LOCA-bench source tree for the actual `gem` task implementations.

Lookup order for the LOCA codebase is:

1. `LOCA_BENCH_RLM_LOCA_ROOT` environment variable
2. `loca_bench_rlm/vendor/loca_bench/`

If you need to override that, pass:

```json
{"loca_root": "/absolute/path/to/LOCA-bench"}
```

This is cleaner than duplicating LOCA task code into the environment package.
This repository now vendors the upstream LOCA-bench repo as a git submodule at
`vendor/loca_bench`, so the environment does not rely on the parent repo layout.

## Vendored checkout option

This directory already contains a vendored LOCA-bench checkout via git submodule:

```text
loca_bench_rlm/vendor/loca_bench/
```

If the submodule is missing, initialize it with:

```bash
git submodule update --init --recursive
```

## Quickstart

Run from this directory:

```bash
cd loca_bench_rlm
prime eval run eval.toml
```

If your repo-level `.env` exports an inference-only `PRIME_API_KEY`, tunnel-backed
RLM runs will fail. In that case either unset it or replace it with a key that has
tunnel/sandbox permissions:

```bash
unset PRIME_API_KEY
prime eval run eval.toml
```

## Config

`load_environment(config)` accepts a `dict` or keyword args with:

- `config_path`: task config JSON path
- `loca_root`: override path to the sibling LOCA-bench root
- `task_names`: optional task-name filter
- `max_examples`: optional cap on config entries
- `shuffle`, `seed`
- `visible_paths`
- RLM settings like `max_turns`, `repl_language`, `sub_model`
- sandbox settings like `sandbox_memory_gb`, `sandbox_timeout_minutes`

Example:

```bash
prime eval run loca_bench_rlm \
  -a '{"config_path":"task_configs/debug.json","max_examples":1,"max_turns":8}'
```

## Bundled Smoke Config

The included `task_configs/debug.json` contains a single `CanvasListTest` task for
quick validation. For larger runs, point `config_path` at another LOCA-bench task
config file from the sibling repository.
