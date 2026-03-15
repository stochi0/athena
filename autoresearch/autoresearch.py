"""
Autoresearch Verifiers environment: autonomous LLM training research with prime-rl.

Uses RLMEnv (sandbox-backed Bash REPL) and a run_training_tool() root tool. Sandbox
start command clones the repo and runs uv sync + prepare. Reward = 1 / (1 + best_val_bpb).

Usage:
    env = load_environment()
    env = load_environment({"max_turns": 20, "num_examples": 3})
    prime eval run autoresearch -n 2 -m openai/gpt-4.1-mini -a '{"max_turns": 20, "num_examples": 3}'
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from datasets import Dataset

import verifiers as vf
from verifiers.envs.experimental.rlm_env import RLMEnv
from verifiers.types import State

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    max_turns: int = 10
    docker_image: str = "ghcr.io/astral-sh/uv:python3.12-bookworm"
    working_dir: str = "/workspace/autoresearch"
    timeout_per_command_seconds: int = 30
    timeout_per_training_run_seconds: int = 600
    gpu_count: int = 1
    memory_gb: int = 16
    disk_size_gb: int = 20
    start_command: str | None = None
    num_examples: int = 5
    repo_url: str = "https://github.com/karpathy/autoresearch.git"
    context_dir_name: str = "contexts"
    sandbox_timeout_minutes: int = 60
    sandbox_cpu_cores: int = 1

    @classmethod
    def from_input(cls, cfg: Config | dict | None) -> Config:
        if cfg is None:
            return cls()
        if isinstance(cfg, cls):
            return cfg
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VAL_BPB_PATTERN = re.compile(r"val_bpb:\s+([\d.]+)")
AUTORESEARCH_SYSTEM_PROMPT = """You are an autonomous research agent. Your goal is to achieve the lowest validation bits-per-byte (val_bpb) on the fixed eval set.

- You may only modify train.py in the autoresearch repo.
- Use the Bash REPL (call_bash_repl) to edit files: cd to the repo, edit train.py, then run experiments.
- Use the run_training_tool() root tool to run one 5-minute training experiment. It runs `uv run train.py` and returns val_bpb; lower is better.
- Run one or more experiments and try to improve val_bpb. You can use llm_batch() for sub-tasks (e.g. summarising logs) if needed.
- When done, export RLM_READY=1 and set RLM_CONTENT to a short summary of your best val_bpb."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Ref used by run_training_tool to get the RLMEnv instance (set after env is created).
_env_ref: list[RLMEnv | None] = [None]
_current_cfg: Config | None = None


def _reward_from_val_bpb(val_bpb: float | None) -> float:
    """Convert val_bpb to reward in [0, 1]. Lower val_bpb = higher reward."""
    if val_bpb is None or val_bpb <= 0:
        return 0.0
    return 1.0 / (1.0 + val_bpb)


# ---------------------------------------------------------------------------
# Reward + metrics (receive state and config via rubric.add_class_object)
# ---------------------------------------------------------------------------

_CACHE_BEST_BPB = "best_val_bpb"
_CACHE_NUM_RUNS = "num_runs"


async def autoresearch_reward(state: State, cfg: Config) -> float:
    """Reward = 1/(1+best_val_bpb). Lower val_bpb yields higher reward."""
    best = state.get(_CACHE_BEST_BPB) if isinstance(state, dict) else None
    if best is None or best == float("inf"):
        return 0.0
    return _reward_from_val_bpb(best)


async def num_runs_metric(state: State, cfg: Config) -> float:
    """Number of training runs executed in this rollout."""
    if not isinstance(state, dict):
        return 0.0
    return float(state.get(_CACHE_NUM_RUNS, 0))


async def best_val_bpb_metric(state: State, cfg: Config) -> float:
    """Best val_bpb achieved (for logging; lower is better)."""
    if not isinstance(state, dict):
        return -1.0
    best = state.get(_CACHE_BEST_BPB)
    if best is None or best == float("inf"):
        return -1.0
    return best


# ---------------------------------------------------------------------------
# Root tool exposed to the model (module-level so RLMEnv can use it without subclassing)
# ---------------------------------------------------------------------------


async def run_training(cfg: Config) -> str:
    """Run one 5-minute training experiment in the sandbox. Returns val_bpb and summary."""
    env = _env_ref[0]
    if env is None:
        return "Error: environment not set."
    ctx = env._root_tool_context_var.get()
    if not ctx:
        return "Error: run_training_tool must be called from the REPL."
    state = ctx.get("state")
    if not state:
        return "Error: state not available."
    sandbox_id = state.get("sandbox_id")
    if not sandbox_id:
        return "Error: sandbox not ready yet."
    working_dir = cfg.working_dir
    command = f"cd {working_dir} && uv run train.py 2>&1"
    try:
        result = await env._executor._execute_sandbox_command(
            sandbox_id,
            command,
            timeout=cfg.timeout_per_training_run_seconds,
        )
    except Exception as e:
        return f"Training run failed: {e}. (Check OOM, syntax, or timeout.)"
    stdout = (getattr(result, "stdout", None) or "").strip()
    stderr = (getattr(result, "stderr", None) or "").strip()
    combined = stdout + ("\n" + stderr if stderr else "")
    match = VAL_BPB_PATTERN.search(combined)
    if match:
        val_bpb = float(match.group(1))
        state["last_val_bpb"] = val_bpb
        state[_CACHE_BEST_BPB] = min(
            state.get(_CACHE_BEST_BPB, float("inf")), val_bpb
        )
        state[_CACHE_NUM_RUNS] = state.get(_CACHE_NUM_RUNS, 0) + 1
        return (
            f"Run completed.\nval_bpb: {val_bpb:.6f}\n"
            f"Best so far: {state[_CACHE_BEST_BPB]:.6f}\n---\n{combined}"
        )
    return (
        "Run finished but val_bpb not found in output (run may have crashed or timed out).\n"
        f"---\n{combined}"
    )


async def run_training_tool() -> str:
    """Run one 5-minute training experiment in the sandbox. Returns val_bpb and summary."""
    cfg = _current_cfg
    if cfg is None:
        return "Error: config not set (load_environment not called)."
    return await run_training(cfg)


# ---------------------------------------------------------------------------
# Dataset (default prompt)
# ---------------------------------------------------------------------------

DEFAULT_QUESTION = (
    "You are an autonomous research agent. Your goal is to achieve the lowest "
    "validation bits-per-byte (val_bpb) on the fixed eval set. You may only "
    "modify train.py. Use the bash tool to edit files and the run_training_tool "
    "to run each 5-minute experiment. After each run you get val_bpb; lower is better. "
    "Run one or more experiments and try to improve val_bpb."
)


def _build_default_dataset(num_examples: int) -> Dataset:
    """Build default dataset: num_examples identical rows with question, task, info."""
    return Dataset.from_list(
        [
            {
                "question": DEFAULT_QUESTION,
                "task": "autoresearch",
                "info": {},
            }
            for _ in range(num_examples)
        ]
    )


# ---------------------------------------------------------------------------
# load_environment
# ---------------------------------------------------------------------------


def load_environment(
    config: Config | dict | None = None,
    *,
    dataset_builder: Callable[[int], Dataset] | None = None,
    **kwargs: Any,
) -> vf.Environment:
    """Load the autoresearch Verifiers environment (RLMEnv) for eval and prime-rl.

    Sandbox clones the autoresearch repo and runs uv sync + prepare.py. Override
    docker_image and start_command to use a prebuilt image with repo and data.

    RLMEnv uses state["info"] from each dataset row:
    - info["context_dir"]: path to a directory copied into the REPL filesystem.
    - info["context"]: optional JSON-serializable data written to a file in the REPL fs.

    Args:
        config: Config instance or dict (e.g. from prime eval -a). Merged with kwargs when dict.
        dataset_builder: Optional callable(n) returning a Dataset with "question", "task",
            and optionally "info" per row. If None, uses default synthetic prompts.
        **kwargs: Override config keys when config is a dict (e.g. max_turns=20, num_examples=3).
    """
    if isinstance(config, Config):
        cfg = config
    else:
        merged = dict(config) if isinstance(config, dict) else {}
        merged.update(kwargs)
        cfg = Config.from_input(merged if merged else None)

    if cfg.num_examples < 1:
        raise ValueError("num_examples must be >= 1")
    if cfg.timeout_per_command_seconds < 1:
        raise ValueError("timeout_per_command_seconds must be >= 1")
    if cfg.timeout_per_training_run_seconds < 1:
        raise ValueError("timeout_per_training_run_seconds must be >= 1")

    rubric = vf.Rubric(funcs=[autoresearch_reward], weights=[1.0])
    rubric.add_metric(num_runs_metric)
    rubric.add_metric(best_val_bpb_metric)
    rubric.add_class_object("cfg", cfg)

    if dataset_builder is not None:
        dataset = dataset_builder(cfg.num_examples)
    else:
        dataset = _build_default_dataset(cfg.num_examples)

    # Sandbox start command: clone repo, uv sync, prepare, then keep container alive.
    if cfg.start_command is not None:
        sandbox_start_command = cfg.start_command
    else:
        sandbox_start_command = (
            f"set -e; git clone {cfg.repo_url} {cfg.working_dir}; "
            f"cd {cfg.working_dir} && uv sync && uv run prepare.py --num-shards 2; "
            "exec tail -f /dev/null"
        )

    global _current_cfg
    _current_cfg = cfg
    env = RLMEnv(
        dataset=dataset,
        rubric=rubric,
        env_id="autoresearch",
        max_iterations=cfg.max_turns,
        system_prompt=AUTORESEARCH_SYSTEM_PROMPT,
        repl_language="bash",
        root_tools=[run_training_tool],
        sandbox_docker_image=cfg.docker_image,
        sandbox_start_command=sandbox_start_command,
        sandbox_gpu_count=cfg.gpu_count,
        sandbox_memory_gb=cfg.memory_gb,
        sandbox_disk_size_gb=cfg.disk_size_gb,
        sandbox_timeout_minutes=cfg.sandbox_timeout_minutes,
        sandbox_cpu_cores=cfg.sandbox_cpu_cores,
        code_execution_timeout=cfg.timeout_per_command_seconds,
        execution_backend="sandbox",
    )
    _env_ref[0] = env
    return env
