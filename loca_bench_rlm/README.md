# LOCA Bench RLM

Standalone `verifiers`-style RLM environment for LOCA-bench task configs.

This directory mirrors the cleaner packaging style used by `athena/discover_gsm8k`:

- `loca_bench_rlm.py`: public `load_environment()` entrypoint
- `core/`: config, dataset loading, prompting, and evaluation helpers
- `task_configs/`: bundled smoke-test config assets
- `eval.toml`: ready-to-run local smoke eval
- `eval_*.toml`: ready-to-run eval presets for LOCA final sets (8k..256k)

## What This Environment Does

For each rollout, the environment:

1. Loads a LOCA-bench task config JSON.
2. Resolves a local LOCA-bench checkout:
   - uses explicit `loca_root` if provided
   - else uses `LOCA_BENCH_RLM_LOCA_ROOT` if set
   - else clones and caches LOCA-bench from GitHub on first use
3. Copies only the agent-visible task artifacts into the RLM sandbox:
   - `agent_workspace`
   - `files`
   - `local_db`
4. Lets the model solve the task with `RLMEnv`.
5. Downloads the sandbox filesystem and reuses the original LOCA task evaluator via `env.step()`.

## External LOCA Source

This environment is standalone in layout and packaging, but it still relies on a
LOCA-bench source tree for the actual `gem` task implementations.

Lookup order for the LOCA codebase is:

1. `LOCA_BENCH_RLM_LOCA_ROOT` environment variable
2. Cached checkout under `~/.cache/loca-bench/`

## Where `config_path` Comes From

`config_path` is resolved in this order:

1. Relative to this package (`loca_bench_rlm/`)
2. Relative to resolved LOCA root (`loca_root` or managed cache clone)
3. Relative to current working directory

That means both of these are valid:

- Local bundled smoke config: `task_configs/debug.json`
- LOCA task-set configs: `task-configs/final_8k_set_config.json` (and 16k/32k/64k/96k/128k/256k)

If `loca_root` is not set, LOCA configs come from the managed cache checkout.
If `loca_root` is set (or `LOCA_BENCH_RLM_LOCA_ROOT` is exported), LOCA configs come from that local checkout.

If you need to override that, pass:

```json
{"loca_root": "/absolute/path/to/LOCA-bench"}
```

By default the environment treats LOCA-bench as an external source and fetches it
from [hkust-nlp/LOCA-bench](https://github.com/hkust-nlp/LOCA-bench.git) on first use.
The checkout is cached locally and reused across runs, so this repository does not
need to vendor giant task assets.

Default cache settings:

- repo URL: `https://github.com/hkust-nlp/LOCA-bench.git`
- ref: `main`
- cache dir: `~/.cache/loca-bench`
- sparse checkout: enabled for `gem`, `loca`, `mcp_convert`, and `task-configs`

You can override those with config args or environment variables:

```bash
export LOCA_BENCH_RLM_LOCA_REF=main
export LOCA_BENCH_RLM_LOCA_REPO_URL=https://github.com/hkust-nlp/LOCA-bench.git
export LOCA_BENCH_RLM_LOCA_CACHE_DIR=~/.cache/loca-bench
```

## Quickstart

Run from this directory:

```bash
cd loca_bench_rlm
prime eval run eval.toml
```

The first run may take longer because it prepares the cached LOCA-bench checkout.
Later runs reuse the cached clone.

To run a specific LOCA set, use one of:

```bash
prime eval run eval_8k.toml
prime eval run eval_16k.toml
prime eval run eval_32k.toml
prime eval run eval_64k.toml
prime eval run eval_96k.toml
prime eval run eval_128k.toml
prime eval run eval_256k.toml
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
- `loca_root`: override path to a pre-existing LOCA-bench checkout
- `loca_repo_url`: Git remote used when bootstrapping the cache
- `loca_ref`: branch, tag, or other ref to fetch into the cache
- `loca_cache_dir`: local cache directory for managed checkouts
- `loca_sparse_checkout`: if true, only clone the runtime-relevant LOCA paths
- `task_names`: optional task-name filter
- `max_examples`: optional cap on config entries
- `shuffle`, `seed`
- `visible_paths`
- RLM settings like `max_turns`, `repl_language`, `sub_model`
- sandbox settings like `sandbox_memory_gb`, `sandbox_timeout_minutes`

Example:

```bash
prime eval run loca_bench_rlm \
  -a '{"config_path":"task-configs/final_8k_set_config.json","loca_ref":"main","max_examples":1,"max_turns":8}'
```

## Bundled Smoke Config

The included `task_configs/debug.json` contains a single `CanvasListTest` task for
quick validation. For larger runs, point `config_path` at another LOCA-bench task
config file such as `task-configs/final_8k_set_config.json`; the managed LOCA
checkout will provide that file automatically.
