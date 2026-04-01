# Advanced IF (`advanced_if`)

Verifiers **`RLMEnv`** over [facebook/AdvancedIF](https://huggingface.co/datasets/facebook/AdvancedIF): each conversation is materialized as **small files** under `trajectory/` (see `manifest.json`). The rollout prompt is **`RLM_TASK_PROMPT`** only (single user message, like **`discover_gsm8k`**); **verifiers’ built-in RLM scaffold** supplies `answer`, `call_python_repl`, and `llm_batch` docs (default **`root_prompt_verbosity = "medium"`** — override via `[eval.env_args]`). The model may call **`submit_candidate_rubrics(...)`** for limited judge feedback, then finalize with **`answer["content"]` / `answer["ready"]`**.

**Environment ID:** `advanced_if` (module `advanced_if`). Package name: `advanced-if`.

## Behaviour

1. **Dataset build** writes `contexts/advanced_if_rlm/<idx>_<benchmark>/trajectory/{manifest.json, NNNN_role.txt, ...}` (override parent with `context_parent_dir` in `EnvironmentConfig`).
2. **Rollout `info`** is only `{ "context_dir": "<abs path>" }` (copied into the sandbox for RLM). **Gold rubrics** stay in rollout **`answer`** (JSON list string) for judges only.
3. **Final rubric** for scoring comes from `state["final_answer"]` (RLM `answer["content"]` when ready); **`AdvancedIFJudgeRubric`** loads the conversation for the judge from `context_dir` on disk, not from chat.
4. **`feedback_mode`:** `score_only` | `one_violation` — controls what `submit_candidate_rubrics` returns (per-criterion judge is internal).

## Layout

| Path | Role |
|------|------|
| `advanced_if.py` | `AdvancedIFRLMEnv`, `load_environment()` |
| `core/config.py` | `EnvironmentConfig` |
| `core/dataset.py` | Hub → rollouts + workspace materialization |
| `core/trajectory_files.py` | Write / read chunked trajectory files |
| `core/prompts.py` | `RLM_TASK_PROMPT` + judge prompts |
| `core/partial_feedback.py` | Per-criterion judge + limited channel |
| `core/rubrics.py` | `AdvancedIFJudgeRubric` (trajectory from disk, candidate from `final_answer`) |
| `configs/debug.toml` | Small RLM smoke eval |
| `configs/eval_score_only.toml` / `configs/eval_one_violation.toml` | Larger evals (`feedback_mode`) |
| `configs/endpoints.toml` | Prime Inference registry |

## Setup

Python **3.11+**, **`uv`**, from **`advanced_if/`**:

```bash
uv sync
uv run python -c "import advanced_if; print(advanced_if.load_environment)"
```

## Running evals

```bash
cd advanced_if
export PRIME_API_KEY=...
uv run prime eval run configs/debug.toml
```

Extra `[eval.env_args]` keys are forwarded to **`RLMEnv`** (e.g. `sub_model`, `repl_language`, `root_prompt_verbosity` (`light` \| `medium` \| `heavy`), `sandbox_*`, `code_execution_timeout`). See **`lhaw`** / **`long_context_retrieval`** TOMLs for sandbox shapes.

## `EnvironmentConfig`

| Field | Role |
|------|------|
| `dataset_name` / `dataset_split` | Hub dataset |
| `max_examples` / `seed` | Subset + shuffle |
| `judge_model` / `judge_sampling_args` / `judge_client_config` | Judges (defaults: Prime Inference + `PRIME_API_KEY`) |
| `max_turns` | RLM root turn budget |
| `feedback_mode` | `score_only` or `one_violation` |
| `context_parent_dir` | Optional; default `contexts/advanced_if_rlm/` under this package |
| `attach_dataset_stats` | Attach split stats to rubric |

## Dataset snapshot (`train`)

| | |
|--:|--:|
| Rows | 1645 |
| `carried_context_multi_turn_eval_v5` | 736 |
| `system_steerability_v2` | 507 |
| `complex_if_single_turn_v5` | 402 |

Rubrics per example are typically 1–20 items.
