"""
LHAW Clarification RLM Environment.

Implements an RLM environment for the `ScaleAI/lhaw` dataset, where the model
is evaluated on whether it asks the right clarifying questions for an
underspecified task rather than trying to solve the task directly.

The dataset provides:
- an underspecified prompt shown to the model
- removed segments describing what information is missing
- expected clarifying questions that would resolve the ambiguity

This environment follows the same general structure as the other RLM
environments in `research-environments/environments`: it exposes a
`load_environment(...)` entrypoint, loads from Hugging Face, transforms the
dataset into `prompt / answer / info`, and uses a deterministic rubric.
"""

from __future__ import annotations

import json
import random
from typing import Any, Literal

import verifiers as vf
from datasets import load_dataset
from verifiers.envs.experimental.rlm_env import RLMEnv
from verifiers.utils.data_utils import extract_boxed_answer

HF_DATASET_NAME = "ScaleAI/lhaw"

SOURCE_DATASETS = frozenset({"MCP-Atlas", "TheAgentCompany", "SWE-Bench Pro"})
AMBIGUITY_CLASSES = frozenset({"outcome-critical", "divergent", "benign"})
INFORMATION_DIMENSIONS = frozenset({"goal", "constraint", "input", "context"})

_ENV_TIPS = """
<env_tips>
This is an underspecification-detection task, not a task-solving task.
Ask the fewest clarifying questions needed to unblock the work.
Prefer questions that resolve missing identifiers, constraints, inputs, or missing background context.
Return one clarifying question per line.
</env_tips>"""


def _as_list(value: str | list[str] | None) -> list[str]:
    """Convert a scalar or list-like option into a list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _extract_questions(answer: str) -> list[str]:
    """Parse a model response into a list of question lines."""
    raw = extract_boxed_answer(answer) or answer or ""
    raw = raw.strip()
    if not raw:
        return []
    questions = [line.strip() for line in raw.splitlines() if line.strip()]
    return questions if questions else [raw]


def _expected_question_groups(state: vf.State) -> list[dict[str, Any]]:
    """Return expected question groups from state answer/info."""
    answer = state.get("answer", "")
    try:
        parsed = json.loads(answer) if answer else []
    except json.JSONDecodeError:
        parsed = []

    if isinstance(parsed, list):
        return [group for group in parsed if isinstance(group, dict)]
    return []


def _question_efficiency_score(
    predicted_questions: list[str], groups: list[dict[str, Any]], ambiguity_class: str
) -> float:
    """Encourage asking few questions, following the paper's cost framing."""
    if not predicted_questions:
        return 0.0 if groups else 1.0

    target_count = max(1, len(groups))
    extra_questions = max(0, len(predicted_questions) - target_count)

    if ambiguity_class == "benign":
        return 1.0 / (1.0 + extra_questions + len(predicted_questions) / max(1, target_count))
    return 1.0 / (1.0 + extra_questions)


def _ask_decision_reward(
    predicted_questions: list[str], groups: list[dict[str, Any]], ambiguity_class: str
) -> float:
    """Score whether the agent chose to ask at all.

    This follows the paper's cost-sensitive framing:
    - outcome-critical: asking is necessary
    - benign: asking is usually unnecessary and should be penalized
    - divergent: sits between the two extremes
    """
    asked = bool(predicted_questions)

    if ambiguity_class == "outcome-critical":
        return 1.0 if asked else 0.0
    if ambiguity_class == "benign":
        return 1.0 if not asked else 0.0
    if ambiguity_class == "divergent":
        return 1.0 if asked else 0.5
    return 1.0 if asked else 0.5


def _paper_aligned_reward(
    predicted_questions: list[str], groups: list[dict[str, Any]], ambiguity_class: str
) -> float:
    """Primary reward aligned with the LHAW paper's clarification framing.

    The paper emphasizes that clarification should be:
    1. necessary when missing information is outcome-critical,
    2. selective because user interruption is costly, and
    3. minimized on benign tasks where agents can infer the missing detail.
    
    Since this standalone environment evaluates question-asking rather than full
    downstream task recovery, we use a class-conditional proxy:
    - outcome-critical: strongly reward asking relevant questions
    - divergent: reward helpful clarification, but allow some no-ask credit
    - benign: primarily reward not asking at all
    """
    decision = _ask_decision_reward(predicted_questions, groups, ambiguity_class)
    efficiency = _question_efficiency_score(predicted_questions, groups, ambiguity_class)

    if ambiguity_class == "outcome-critical":
        return 0.7 * decision + 0.3 * efficiency

    if ambiguity_class == "divergent":
        return 0.6 * decision + 0.4 * efficiency

    if ambiguity_class == "benign":
        if not predicted_questions:
            return 1.0
        return 0.2 * decision + 0.8 * efficiency

    return 0.5 * decision + 0.5 * efficiency


def load_environment(
    # Dataset options
    split: str = "test",
    source_dataset: Literal["all", "MCP-Atlas", "TheAgentCompany", "SWE-Bench Pro"] = "all",
    ambiguity_class: Literal["all", "outcome-critical", "divergent", "benign"] = "all",
    information_dimension: Literal["all", "goal", "constraint", "input", "context"]
    | list[Literal["goal", "constraint", "input", "context"]] = "all",
    max_examples: int | None = None,
    shuffle: bool = False,
    seed: int | None = None,
    include_env_tips: bool = False,
    # RLM options
    max_turns: int = 12,
    sub_llm_max_turns: int = 5,
    sub_model: str | None = None,
    max_sub_llm_parallelism: int = 5,
    max_output_length: int = 8192,
    code_execution_timeout: int = 120,
    abort_on_code_timeout: bool = False,
    max_startup_wait_seconds: int = 120,
    pip_install_packages: str = "",
    repl_language: Literal["bash", "python"] = "bash",
    # Sandbox resource options
    sandbox_docker_image: str = "python:3.11-slim",
    sandbox_cpu_cores: int = 1,
    sandbox_memory_gb: int = 2,
    sandbox_disk_size_gb: int = 5,
    sandbox_gpu_count: int = 0,
    sandbox_timeout_minutes: int = 60,
    **kwargs,
) -> vf.Environment:
    """
    Load the LHAW clarification RLM environment.

    Args:
        split: Dataset split to use. The public LHAW release currently exposes
            `test`.
        source_dataset: Filter by source benchmark (`MCP-Atlas`,
            `TheAgentCompany`, `SWE-Bench Pro`) or keep all rows.
        ambiguity_class: Filter by ambiguity class or keep all rows.
        information_dimension: Keep examples whose removed information contains
            all requested dimensions. Accepts `"all"`, a single dimension, or a
            list of dimensions.
        max_examples: Optional post-filter cap on dataset size.
        shuffle: Whether to shuffle the dataset.
        seed: Random seed for shuffling.
        include_env_tips: If True, append environment-specific guidance to the
            prompt.
        max_turns: Maximum REPL iterations.
        sub_llm_max_turns: Max tool-calling turns for each sub-LLM call.
        sub_model: Model for sub-LLM calls (defaults to same as root model).
        max_sub_llm_parallelism: Max concurrent sub-LLM calls.
        max_output_length: Maximum code execution output length.
        code_execution_timeout: Timeout in seconds for code execution.
        abort_on_code_timeout: If True, abort rollout on code timeout; if
            False, return the error to the model.
        max_startup_wait_seconds: Max seconds to wait for sandbox worker
            startup.
        pip_install_packages: Packages to install in sandbox.
        repl_language: The RLM execution language (`bash` or `python`).
        sandbox_docker_image: Docker image for sandbox.
        sandbox_cpu_cores: CPU cores for sandbox.
        sandbox_memory_gb: Memory in GB for sandbox.
        sandbox_disk_size_gb: Disk size in GB for sandbox.
        sandbox_gpu_count: Number of GPUs for sandbox.
        sandbox_timeout_minutes: Overall sandbox lifetime in minutes.
        **kwargs: Additional arguments passed to `RLMEnv`.

    Returns:
        Configured `RLMEnv` instance.
    """
    if source_dataset != "all" and source_dataset not in SOURCE_DATASETS:
        raise ValueError(
            f"source_dataset={source_dataset!r} is invalid. Must be 'all' or one of {sorted(SOURCE_DATASETS)}."
        )
    if ambiguity_class != "all" and ambiguity_class not in AMBIGUITY_CLASSES:
        raise ValueError(
            f"ambiguity_class={ambiguity_class!r} is invalid. Must be 'all' or one of {sorted(AMBIGUITY_CLASSES)}."
        )

    requested_dimensions = _as_list(None if information_dimension == "all" else information_dimension)
    invalid_dimensions = sorted(set(requested_dimensions) - INFORMATION_DIMENSIONS)
    if invalid_dimensions:
        raise ValueError(
            f"information_dimension contains invalid values {invalid_dimensions}. "
            f"Valid values: {sorted(INFORMATION_DIMENSIONS)}."
        )

    raw_dataset = load_dataset(HF_DATASET_NAME, split=split)

    if source_dataset != "all":
        raw_dataset = raw_dataset.filter(lambda example: example["dataset"] == source_dataset)
    if ambiguity_class != "all":
        raw_dataset = raw_dataset.filter(lambda example: example["ambiguity_class"] == ambiguity_class)
    if requested_dimensions:
        requested_dimension_set = set(requested_dimensions)
        raw_dataset = raw_dataset.filter(
            lambda example: requested_dimension_set.issubset(set(example["information_dimension"]))
        )

    def transform_example(example: dict[str, Any], idx: int) -> dict[str, Any]:
        prompt_content = (
            "Below is an underspecified task.\n"
            "Do not solve it yet.\n"
            "Ask the minimum set of clarifying questions needed before you can reliably proceed.\n"
            "Return only the clarifying question(s), one per line.\n\n"
            f"<underspecified_task>\n{example['underspecified_prompt']}\n</underspecified_task>"
        )
        if include_env_tips:
            prompt_content = prompt_content + _ENV_TIPS

        expected_questions = example.get("expected_questions", []) or []
        removed_segments = example.get("removed_segments", []) or []

        return {
            "example_id": example.get("variant_id", idx),
            "prompt": [{"role": "user", "content": prompt_content}],
            "task": "lhaw-clarification",
            "answer": json.dumps(expected_questions),
            "info": {
                "variant_id": example.get("variant_id", ""),
                "original_task": example.get("original_task", ""),
                "source_dataset": example.get("dataset", ""),
                "ambiguity_class": example.get("ambiguity_class", ""),
                "information_dimension": example.get("information_dimension", []),
                "removed_segments": removed_segments,
                "expected_questions": expected_questions,
                "expected_question_groups": len(expected_questions),
                "original_prompt": example.get("original_prompt", ""),
                "underspecified_prompt": example.get("underspecified_prompt", ""),
            },
        }

    dataset = raw_dataset.map(
        transform_example,
        with_indices=True,
        remove_columns=raw_dataset.column_names,
        writer_batch_size=100,
    )

    if shuffle:
        seed = seed if seed is not None else random.randint(1000, 100_000_000)
        dataset = dataset.shuffle(seed=seed)

    if max_examples is not None:
        if max_examples < 0:
            raise ValueError(f"max_examples must be >= 0; got {max_examples}.")
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    def clarification_quality_reward(state: vf.State, **_kwargs) -> float:
        """Primary reward aligned with the LHAW paper's cost-sensitive setup."""
        predicted_questions = _extract_questions(state.get("final_answer", ""))
        expected_groups = _expected_question_groups(state)
        ambiguity = str(state.get("info", {}).get("ambiguity_class", ""))
        return _paper_aligned_reward(predicted_questions, expected_groups, ambiguity)

    def ask_decision_reward(state: vf.State, **_kwargs) -> float:
        """Auxiliary metric: was the ask/no-ask decision appropriate for this class?"""
        predicted_questions = _extract_questions(state.get("final_answer", ""))
        expected_groups = _expected_question_groups(state)
        ambiguity = str(state.get("info", {}).get("ambiguity_class", ""))
        return _ask_decision_reward(predicted_questions, expected_groups, ambiguity)

    def question_count_reward(state: vf.State, **_kwargs) -> float:
        """Auxiliary metric: reward asking around the expected number of questions."""
        predicted_questions = _extract_questions(state.get("final_answer", ""))
        expected_groups = _expected_question_groups(state)
        ambiguity = str(state.get("info", {}).get("ambiguity_class", ""))
        return _question_efficiency_score(predicted_questions, expected_groups, ambiguity)

    rubric = vf.Rubric(
        funcs=[
            clarification_quality_reward,
            ask_decision_reward,
            question_count_reward,
        ],
        weights=[1.0, 0.0, 0.0],
    )

    sandbox_labels = kwargs.pop("sandbox_labels", ["lhaw-rlm"])
    if not (isinstance(sandbox_labels, list) and all(isinstance(label, str) for label in sandbox_labels)):
        raise ValueError(f"sandbox_labels must be of type list[str]; you provided {sandbox_labels}")
    sandbox_labels = list(set(sandbox_labels))

    return RLMEnv(
        repl_language=repl_language,
        max_turns=max_turns,
        sub_llm_max_turns=sub_llm_max_turns,
        sub_model=sub_model,
        max_sub_llm_parallelism=max_sub_llm_parallelism,
        max_output_length=max_output_length,
        code_execution_timeout=code_execution_timeout,
        abort_on_code_timeout=abort_on_code_timeout,
        max_startup_wait_seconds=max_startup_wait_seconds,
        pip_install_packages=pip_install_packages,
        sandbox_docker_image=sandbox_docker_image,
        sandbox_cpu_cores=sandbox_cpu_cores,
        sandbox_memory_gb=sandbox_memory_gb,
        sandbox_disk_size_gb=sandbox_disk_size_gb,
        sandbox_gpu_count=sandbox_gpu_count,
        sandbox_timeout_minutes=sandbox_timeout_minutes,
        dataset=dataset,
        rubric=rubric,
        sandbox_labels=sandbox_labels,
        **kwargs,
    )
