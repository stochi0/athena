"""
Autoresearch Verifiers environment: autonomous LLM training research with prime-rl.

Uses RLMEnv (Verifiers recursive language model environment): sandbox-backed Bash REPL,
run_training() root tool, llm_batch() for sub-LLM calls. Overrides get_sandbox_request
and on_sandbox_ready for repo setup. Reward = 1 / (1 + best_val_bpb).
"""

import re
from typing import Any, Callable

from datasets import Dataset

import verifiers as vf
from verifiers.envs.experimental.rlm_env import RLMEnv
from verifiers.envs.sandbox_env import CreateSandboxRequest
from verifiers.types import State
from verifiers.utils.tool_utils import convert_func_to_oai_tool

AUTORESEARCH_WORKDIR = "/workspace/autoresearch"
VAL_BPB_PATTERN = re.compile(r"val_bpb:\s+([\d.]+)")


def _tool_display_name(tool: Callable) -> str:
    """Match RLMEnv's tool display name for root_tool_names."""
    return getattr(tool, "__name__", getattr(tool, "__class__", type(tool)).__name__)


AUTORESEARCH_SYSTEM_PROMPT = """You are an autonomous research agent. Your goal is to achieve the lowest validation bits-per-byte (val_bpb) on the fixed eval set.

- You may only modify train.py in the autoresearch repo.
- Use the Bash REPL (call_bash_repl) to edit files: cd to the repo, edit train.py, then run experiments.
- Use the run_training() root tool to run one 5-minute training experiment. It runs `uv run train.py` and returns val_bpb; lower is better.
- Run one or more experiments and try to improve val_bpb. You can use llm_batch() for sub-tasks (e.g. summarising logs) if needed.
- When done, export RLM_READY=1 and set RLM_CONTENT to a short summary of your best val_bpb."""


class AutoresearchEnv(RLMEnv):
    """
    RLM (Recursive Language Model) environment for autoresearch.

    - REPL runs in a sandbox; get_sandbox_request / on_sandbox_ready clone the repo and run uv sync + prepare.
    - Root tool run_training() runs uv run train.py in the same sandbox and updates state for the rubric.
    - Reward = 1/(1+best_val_bpb). Metrics: num_runs, best_val_bpb.
    """

    def __init__(
        self,
        working_dir: str = AUTORESEARCH_WORKDIR,
        timeout_per_training_run_seconds: int = 600,
        repo_url: str = "https://github.com/karpathy/autoresearch.git",
        start_command_override: str | None = None,
        sandbox_docker_image: str = "ghcr.io/astral-sh/uv:python3.12-bookworm",
        sandbox_gpu_count: int = 1,
        sandbox_memory_gb: int = 16,
        sandbox_disk_size_gb: int = 20,
        sandbox_timeout_minutes: int = 60,
        sandbox_cpu_cores: int = 1,
        **kwargs: Any,
    ):
        self._autoresearch_working_dir = working_dir
        self._autoresearch_timeout = timeout_per_training_run_seconds
        self._autoresearch_repo_url = repo_url
        self._autoresearch_start_command_override = start_command_override
        super().__init__(
            repl_language="bash",
            root_tools=[],
            sandbox_docker_image=sandbox_docker_image,
            sandbox_start_command="tail -f /dev/null",
            sandbox_gpu_count=sandbox_gpu_count,
            sandbox_memory_gb=sandbox_memory_gb,
            sandbox_disk_size_gb=sandbox_disk_size_gb,
            sandbox_timeout_minutes=sandbox_timeout_minutes,
            sandbox_cpu_cores=sandbox_cpu_cores,
            **kwargs,
        )
        # Register run_training as a root tool (after super().__init__ so self exists)
        self.root_tools.insert(0, self.run_training)
        self.root_tool_map["run_training"] = self.run_training
        oai_tool = convert_func_to_oai_tool(self.run_training)
        self.root_tool_doc_funcs.insert(0, self.run_training)
        self.root_oai_tools.insert(0, oai_tool)
        self.root_tool_names = [_tool_display_name(t) for t in self.root_tools]

    def get_sandbox_request(self, state: State) -> CreateSandboxRequest:
        """Override to use training image/resources; repo setup runs in on_sandbox_ready."""
        env_vars = dict(self.sandbox_environment_vars or {})
        return CreateSandboxRequest(
            name=f"rlm-{state.get('rollout_id', 'unknown')}",
            docker_image=self.sandbox_docker_image,
            start_command=self.sandbox_start_command,
            cpu_cores=self.sandbox_cpu_cores,
            memory_gb=self.sandbox_memory_gb,
            disk_size_gb=self.sandbox_disk_size_gb,
            gpu_count=self.sandbox_gpu_count,
            timeout_minutes=self.sandbox_timeout_minutes,
            environment_vars=env_vars,
            team_id=self.sandbox_team_id,
            advanced_configs=self.sandbox_advanced_configs,
            labels=self.sandbox_labels or [],
        )

    async def on_sandbox_ready(self, state: State, sandbox_id: str) -> None:
        """Clone autoresearch repo and run uv sync + prepare.py in the sandbox."""
        cmd = self._autoresearch_start_command_override
        if cmd is None:
            wd = self._autoresearch_working_dir
            cmd = (
                f"set -e; git clone {self._autoresearch_repo_url} {wd}; "
                f"cd {wd} && uv sync && uv run prepare.py --num-shards 2"
            )
        # Run in sandbox; allow long timeout for clone + uv sync + prepare
        await self._executor._execute_sandbox_command(
            sandbox_id,
            cmd,
            timeout=max(300, self.max_startup_wait_seconds),
        )

    async def setup_state(self, state: State, **kwargs: Any) -> State:
        state = await super().setup_state(state, **kwargs)
        state["working_dir"] = self._autoresearch_working_dir
        state["best_val_bpb"] = float("inf")
        state["last_val_bpb"] = None
        state["num_runs"] = 0
        return state

    async def run_training(self) -> str:
        """Run one 5-minute training experiment in the sandbox. Call from the REPL (no args). Returns val_bpb and summary."""
        ctx = self._root_tool_context_var.get()
        if not ctx:
            return "Error: run_training must be called from the REPL."
        state = ctx.get("state")
        if not state:
            return "Error: state not available."
        sandbox_id = state.get("sandbox_id")
        if not sandbox_id:
            return "Error: sandbox not ready yet."
        working_dir = state.get("working_dir", self._autoresearch_working_dir)
        command = f"cd {working_dir} && uv run train.py 2>&1"
        try:
            result = await self._executor._execute_sandbox_command(
                sandbox_id,
                command,
                timeout=self._autoresearch_timeout,
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
            state["best_val_bpb"] = min(
                state.get("best_val_bpb", float("inf")), val_bpb
            )
            state["num_runs"] = state.get("num_runs", 0) + 1
            return (
                f"Run completed.\nval_bpb: {val_bpb:.6f}\n"
                f"Best so far: {state['best_val_bpb']:.6f}\n---\n{combined}"
            )
        return (
            "Run finished but val_bpb not found in output (run may have crashed or timed out).\n"
            f"---\n{combined}"
        )


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
    dataset_builder: Callable[[int], Dataset] | None = None,
) -> vf.Environment:
    """Load the autoresearch Verifiers environment (RLMEnv) for eval and prime-rl.

    Sandbox clones the autoresearch repo and runs uv sync + prepare.py. Override
    docker_image and start_command to use a prebuilt image with repo and data.

    RLMEnv uses state["info"] from each dataset row:
    - info["context_dir"]: path to a directory copied into the REPL filesystem
      (so the agent can read files from it; useful for per-example configs or data).
    - info["context"]: optional JSON-serializable data written to a file in the REPL fs.
    RLMEnv copies context_dir / writes context before the worker starts; the cloned
    repo is then set up in on_sandbox_ready.

    Args:
        max_turns: Max agent turns (each turn can include tool calls).
        docker_image: Sandbox image (must have git, uv; use CUDA image if gpu_count > 0).
        working_dir: Path in sandbox to the autoresearch repo.
        timeout_per_command_seconds: Timeout for bash/REPL commands.
        timeout_per_training_run_seconds: Timeout for run_training (uv run train.py).
        gpu_count: GPUs for the sandbox (1 recommended).
        memory_gb: RAM for the sandbox.
        disk_size_gb: Disk for the sandbox.
        start_command: If set, run this in sandbox instead of clone/sync/prepare.
        num_examples: Dataset size (synthetic prompts); ignored if dataset_builder is set.
        repo_url: Git URL to clone for the autoresearch repo.
        dataset_builder: Optional callable(n) returning a Dataset with "question", "task",
            and optionally "info" (dict with context_dir and/or context) per row.
    """
    rubric = vf.Rubric(funcs=[autoresearch_reward], weights=[1.0])
    rubric.add_metric(num_runs_metric)
    rubric.add_metric(best_val_bpb_metric)

    if dataset_builder is not None:
        dataset = dataset_builder(num_examples)
    else:
        question = (
            "You are an autonomous research agent. Your goal is to achieve the lowest "
            "validation bits-per-byte (val_bpb) on the fixed eval set. You may only "
            "modify train.py. Use the bash tool to edit files and the run_training tool "
            "to run each 5-minute experiment. After each run you get val_bpb; lower is better. "
            "Run one or more experiments and try to improve val_bpb."
        )
        dataset = Dataset.from_list(
            [
                {
                    "question": question,
                    "task": "autoresearch",
                    "info": {},  # RLMEnv: info["context_dir"] / info["context"] for REPL fs
                }
                for _ in range(num_examples)
            ]
        )

    return AutoresearchEnv(
        dataset=dataset,
        rubric=rubric,
        env_id="autoresearch",
        max_iterations=max_turns,
        system_prompt=AUTORESEARCH_SYSTEM_PROMPT,
        working_dir=working_dir,
        timeout_per_training_run_seconds=timeout_per_training_run_seconds,
        repo_url=repo_url,
        start_command_override=start_command,
        sandbox_docker_image=docker_image,
        sandbox_gpu_count=gpu_count,
        sandbox_memory_gb=memory_gb,
        sandbox_disk_size_gb=disk_size_gb,
        sandbox_timeout_minutes=60,
        sandbox_cpu_cores=1,
        code_execution_timeout=timeout_per_command_seconds,
        execution_backend="sandbox",
    )
