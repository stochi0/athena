"""
Autoresearch Verifiers environment: autonomous LLM training research with prime-rl.

The agent operates in a sandbox with the autoresearch repo. It may modify train.py,
run 5-minute training experiments (uv run train.py), and is rewarded for achieving
lower validation bits-per-byte (val_bpb). Uses vf.SandboxEnv + a run_training tool
so the rubric can score from state. Compatible with prime eval run and prime-rl.
"""

import re
import time
from typing import Any

from datasets import Dataset

import verifiers as vf

# Default working directory inside the sandbox (repo root)
AUTORESEARCH_WORKDIR = "/workspace/autoresearch"

# Regex to parse val_bpb from train.py output (after "---")
VAL_BPB_PATTERN = re.compile(r"val_bpb:\s+([\d.]+)")


class AutoresearchEnv(vf.SandboxEnv):
    """
    Sandbox environment for autonomous autoresearch experiments.

    Extends SandboxEnv with:
    - working_dir set to the autoresearch repo in the sandbox
    - run_training tool: runs uv run train.py, parses val_bpb, stores in state
    - Rubric scores from state["best_val_bpb"] (higher reward = lower val_bpb)
    """

    def __init__(
        self,
        working_dir: str = AUTORESEARCH_WORKDIR,
        timeout_per_training_run_seconds: int = 600,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.working_dir = working_dir
        self.timeout_per_training_run_seconds = timeout_per_training_run_seconds
        self.add_tool(
            self.run_training,
            args_to_skip=["sandbox_id", "sandbox_state", "working_dir"],
        )

    async def setup_state(self, state: vf.State, **kwargs: Any) -> vf.State:
        state = await super().setup_state(state, **kwargs)
        state["working_dir"] = self.working_dir
        state["best_val_bpb"] = float("inf")
        state["last_val_bpb"] = None
        state["num_runs"] = 0
        return state

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> dict[str, Any]:
        updated = super().update_tool_args(
            tool_name, tool_args, messages, state, **kwargs
        )
        if tool_name == "run_training":
            updated["sandbox_id"] = state["sandbox_id"]
            updated["sandbox_state"] = state["sandbox_state"]
            updated["working_dir"] = state.get("working_dir", self.working_dir)
        return updated

    async def run_training(
        self,
        sandbox_id: str,
        sandbox_state: dict,
        working_dir: str | None = None,
    ) -> str:
        """Run one 5-minute training experiment and return val_bpb and summary.

        Executes `uv run train.py` in the sandbox. Parses val_bpb from stdout,
        updates rollout state for the rubric. Call this after editing train.py.
        """
        if not sandbox_state.get("ready"):
            await self._wait_for_sandbox_ready(
                sandbox_state, sandbox_id  # type: ignore[arg-type]
            )
        wd = working_dir or self.working_dir
        command = f"cd {wd} && uv run train.py 2>&1"
        s = time.time()
        try:
            results = await self.sandbox_client.execute_command(
                sandbox_id,
                command,
                working_dir=None,
                timeout=self.timeout_per_training_run_seconds,
            )
        except Exception as e:
            return f"Training run failed: {e}. (Check OOM, syntax, or timeout.)"
        elapsed = time.time() - s
        stdout = (results.stdout or "").strip()
        stderr = (results.stderr or "").strip()
        combined = stdout + ("\n" + stderr if stderr else "")

        match = VAL_BPB_PATTERN.search(combined)
        if match:
            val_bpb = float(match.group(1))
            sandbox_state["_last_val_bpb"] = val_bpb
            sandbox_state["_best_val_bpb"] = min(
                sandbox_state.get("_best_val_bpb", float("inf")), val_bpb
            )
            sandbox_state["_num_runs"] = sandbox_state.get("_num_runs", 0) + 1
            return (
                f"Run completed in {elapsed:.0f}s.\n"
                f"val_bpb: {val_bpb:.6f}\n"
                f"Best so far: {sandbox_state['_best_val_bpb']:.6f}\n"
                f"---\n{combined}"
            )
        return (
            f"Run finished but val_bpb not found in output (run may have crashed or timed out).\n"
            f"---\n{combined}"
        )

    async def post_rollout(self, state: vf.State) -> None:
        """Copy sandbox training results into state for the rubric."""
        sandbox_state = state.get("sandbox_state") or {}
        state["best_val_bpb"] = sandbox_state.get("_best_val_bpb", float("inf"))
        state["last_val_bpb"] = sandbox_state.get("_last_val_bpb")
        state["num_runs"] = sandbox_state.get("_num_runs", 0)


def _reward_from_val_bpb(val_bpb: float | None) -> float:
    """Convert val_bpb to reward in [0, 1]. Lower val_bpb = higher reward."""
    if val_bpb is None or val_bpb <= 0:
        return 0.0
    return 1.0 / (1.0 + val_bpb)


async def autoresearch_reward(state: vf.State) -> float:
    """Reward = 1/(1+best_val_bpb). Lower val_bpb yields higher reward."""
    best = state.get("best_val_bpb")
    if best is None or best == float("inf"):
        return 0.0
    return _reward_from_val_bpb(best)


async def num_runs_metric(state: vf.State) -> float:
    """Number of training runs executed in this rollout."""
    return float(state.get("num_runs", 0))


async def best_val_bpb_metric(state: vf.State) -> float:
    """Best val_bpb achieved (for logging; lower is better)."""
    best = state.get("best_val_bpb")
    if best is None or best == float("inf"):
        return -1.0
    return best


def load_environment(
    max_turns: int = 10,
    docker_image: str = "ghcr.io/astral-sh/uv:python3.12-bookworm",
    working_dir: str = AUTORESEARCH_WORKDIR,
    timeout_per_command_seconds: int = 30,
    timeout_per_training_run_seconds: int = 600,
    gpu_count: int = 1,
    memory_gb: int = 16,
    disk_size_gb: int = 20,
    start_command: str | None = None,
    num_examples: int = 5,
    repo_url: str = "https://github.com/karpathy/autoresearch.git",
) -> vf.Environment:
    """Load the autoresearch Verifiers environment for eval and prime-rl.

    The sandbox is configured to clone the autoresearch repo and run `uv sync`
    and optionally `uv run prepare.py` so training can run. You can override
    docker_image and start_command to use a prebuilt image that already
    contains the repo and data.

    Args:
        max_turns: Max agent turns (each turn can include tool calls).
        docker_image: Image for the sandbox (must have git, uv; GPU image if gpu_count > 0).
        working_dir: Path inside the sandbox to the autoresearch repo.
        timeout_per_command_seconds: Timeout for bash commands (e.g. edits).
        timeout_per_training_run_seconds: Timeout for run_training (uv run train.py).
        gpu_count: GPUs for the sandbox (1 recommended for training).
        memory_gb: RAM for the sandbox.
        disk_size_gb: Disk for the sandbox.
        start_command: If set, used as sandbox start command; else clones repo and tails.
        num_examples: Dataset size (synthetic prompts).
        repo_url: Git URL to clone for autoresearch repo.
    """
    if start_command is None:
        start_command = (
            f"set -e; git clone {repo_url} {working_dir}; "
            f"cd {working_dir} && uv sync && uv run prepare.py --num-shards 2; "
            "tail -f /dev/null"
        )

    rubric = vf.Rubric(funcs=[autoresearch_reward], weights=[1.0])
    rubric.add_metric(num_runs_metric)
    rubric.add_metric(best_val_bpb_metric)

    dataset = Dataset.from_list(
        [
            {
                "question": (
                    "You are an autonomous research agent. Your goal is to achieve the lowest "
                    "validation bits-per-byte (val_bpb) on the fixed eval set. You may only "
                    "modify train.py. Use the bash tool to edit files and the run_training tool "
                    "to run each 5-minute experiment. After each run you get val_bpb; lower is better. "
                    "Run one or more experiments and try to improve val_bpb."
                ),
                "task": "autoresearch",
            }
            for _ in range(num_examples)
        ]
    )

    env = AutoresearchEnv(
        dataset=dataset,
        rubric=rubric,
        max_turns=max_turns,
        docker_image=docker_image,
        start_command=start_command,
        working_dir=working_dir,
        timeout_per_command_seconds=timeout_per_command_seconds,
        timeout_per_training_run_seconds=timeout_per_training_run_seconds,
        gpu_count=gpu_count,
        memory_gb=memory_gb,
        disk_size_gb=disk_size_gb,
    )
    return env
