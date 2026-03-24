"""
LHAW Interactive RLM Environment.

This environment implements the closest paper-aligned interaction loop that is
practical with the released `ScaleAI/lhaw` dataset and the `verifiers` RLM
runtime:

- the model sees an underspecified prompt
- it can ask a simulated user for clarification via an `ask_user(...)` tool
- it must produce a fully specified, clarified task as its final answer
- an LLM judge compares that clarified task against the original prompt

Unlike the full paper setup, this environment does not execute the underlying
benchmark tasks (e.g. TAC / SWE-Bench / MCP-Atlas native harnesses) and
therefore cannot reproduce benchmark-native pass@3 or checkpoint metrics inside
one standalone `verifiers` environment. Instead, it faithfully models the
clarification interaction itself and scores the reconstructed task spec with an
LLM judge.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Literal

import httpx
import verifiers as vf
from datasets import load_dataset
from openai import AsyncOpenAI
from verifiers.envs.experimental.rlm_env import RLMEnv
from verifiers.utils.data_utils import extract_boxed_answer

HF_DATASET_NAME = "ScaleAI/lhaw"

SOURCE_DATASETS = frozenset({"MCP-Atlas", "TheAgentCompany", "SWE-Bench Pro"})
AMBIGUITY_CLASSES = frozenset({"outcome-critical", "divergent", "benign"})
INFORMATION_DIMENSIONS = frozenset({"goal", "constraint", "input", "context"})

_ENV_TIPS = """
<env_tips>
Use the Python REPL to reason about the task and call `ask_user(...)` when you
need missing information. Once you have enough information, write a fully
specified version of the task into `answer["content"]` and then set
`answer["ready"] = True`.
</env_tips>"""


def _as_list(value: str | list[str] | None) -> list[str]:
    """Convert a scalar or list-like option into a list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _format_removed_segments(segments: list[dict[str, Any]]) -> str:
    """Render removed segments for prompts and judging."""
    if not segments:
        return "None"

    lines = []
    for segment in segments:
        dimension = str(segment.get("dimension", ""))
        subdimension = str(segment.get("subdimension", ""))
        value = str(segment.get("value", ""))
        segment_id = str(segment.get("id", ""))
        lines.append(
            f"- id={segment_id or 'unknown'} dimension={dimension or 'unknown'} "
            f"subdimension={subdimension or 'unknown'} value={value or '(empty)'}"
        )
    return "\n".join(lines)


def _message_role(message: Any) -> str:
    """Get message role from pydantic or dict-style message objects."""
    if isinstance(message, dict):
        return str(message.get("role", ""))
    return str(getattr(message, "role", ""))


def _message_content(message: Any) -> str:
    """Get string content from pydantic or dict-style message objects."""
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", ""))


def _extract_ask_user_interactions(completion: Any) -> list[dict[str, str]]:
    """Extract ask_user question/response pairs from the final conversation."""
    if not isinstance(completion, list):
        return []

    interactions: list[dict[str, str]] = []
    pending_by_id: dict[str, int] = {}

    for message in completion:
        role = _message_role(message)

        if role == "assistant":
            tool_calls = getattr(message, "tool_calls", None)
            if isinstance(message, dict):
                tool_calls = message.get("tool_calls", tool_calls)
            if not isinstance(tool_calls, list):
                continue

            for tool_call in tool_calls:
                tool_name = getattr(tool_call, "name", "")
                tool_call_id = getattr(tool_call, "id", "")
                arguments = getattr(tool_call, "arguments", "{}")

                if isinstance(tool_call, dict):
                    tool_name = str(tool_call.get("name", tool_name))
                    tool_call_id = str(tool_call.get("id", tool_call_id))
                    arguments = tool_call.get("arguments", arguments)

                if tool_name != "ask_user":
                    continue

                question = ""
                context = ""
                try:
                    parsed_args = json.loads(arguments) if isinstance(arguments, str) else {}
                    if isinstance(parsed_args, dict):
                        question = str(parsed_args.get("question", "")).strip()
                        context = str(parsed_args.get("context", "")).strip()
                except json.JSONDecodeError:
                    question = str(arguments).strip()

                interactions.append(
                    {
                        "question": question,
                        "context": context,
                        "response": "",
                    }
                )
                if tool_call_id:
                    pending_by_id[tool_call_id] = len(interactions) - 1

        elif role == "tool":
            tool_call_id = getattr(message, "tool_call_id", "")
            if isinstance(message, dict):
                tool_call_id = str(message.get("tool_call_id", tool_call_id))
            if tool_call_id and tool_call_id in pending_by_id:
                interactions[pending_by_id[tool_call_id]]["response"] = _message_content(message).strip()

    return interactions


def _format_ask_user_transcript(interactions: list[dict[str, str]]) -> str:
    """Render ask_user interactions for judging."""
    if not interactions:
        return "No ask_user calls."

    lines = []
    for index, interaction in enumerate(interactions, start=1):
        lines.append(f"Question {index}: {interaction.get('question', '')}")
        if interaction.get("context"):
            lines.append(f"Context {index}: {interaction['context']}")
        lines.append(f"User response {index}: {interaction.get('response', '')}")
    return "\n".join(lines)


def _ambiguity_class_judge_guidance(ambiguity_class: str) -> str:
    """Return class-specific judge instructions grounded in the paper taxonomy."""
    if ambiguity_class == "outcome-critical":
        return (
            "This is an outcome-critical variant.\n"
            "- Missing information should be treated as necessary for reliable success.\n"
            "- The final clarified task should explicitly restore the critical missing details from the original task.\n"
            "- If the final clarified task still leaves a blocker unresolved, answer 'no'."
        )
    if ambiguity_class == "divergent":
        return (
            "This is a divergent variant.\n"
            "- The underspecification can lead to variable but plausible outcomes.\n"
            "- Answer 'yes' if the final clarified task is a faithful, executable reconstruction aligned with the original task,\n"
            "  even if the clarification transcript shows one reasonable path among several plausible ones.\n"
            "- Answer 'no' if the final clarified task drifts to a materially different task or leaves key ambiguity unresolved."
        )
    if ambiguity_class == "benign":
        return (
            "This is a benign variant.\n"
            "- The missing information is expected to be inferable from context or low-risk for successful completion.\n"
            "- Do not require unnecessary explicit clarification if the final clarified task is still faithful to the original task.\n"
            "- Answer 'no' only if the final clarified task introduces contradictions, loses essential intent, or is not executable."
        )
    return (
        "Use the original task as the reference and decide whether the final clarified task is faithful, executable, "
        "and does not materially distort the intended task."
    )


def _build_reconstruction_judge_prompt(
    *,
    original_prompt: str,
    underspecified_prompt: str,
    ambiguity_class: str,
    removed_segments: list[dict[str, Any]],
    interactions: list[dict[str, str]],
    final_answer: str,
) -> str:
    """Build the judge prompt for final clarified task reconstruction."""
    removed_segments_text = _format_removed_segments(removed_segments)
    interaction_text = _format_ask_user_transcript(interactions)
    class_guidance = _ambiguity_class_judge_guidance(ambiguity_class)

    return f"""You are evaluating an agent on the LHAW underspecification benchmark.

The agent originally saw this UNDERSPECIFIED task:
```text
{underspecified_prompt}
```

The full ORIGINAL task was:
```text
{original_prompt}
```

Ambiguity class:
{ambiguity_class}

Removed segments:
{removed_segments_text}

Clarification transcript:
{interaction_text}

The agent's FINAL clarified task:
```text
{final_answer}
```

Class-specific guidance:
{class_guidance}

General evaluation criteria:
- preserve the original task intent and scope
- restore missing information when required by the ambiguity class
- avoid inventing contradictory requirements not supported by the original task
- produce a clarified task that is usable and executable as an instruction

Answer "yes" only if the final clarified task should count as a faithful reconstruction of the original task under the guidance above.
Otherwise answer "no".

Respond with only "yes" or "no"."""


class LHAWJudgeRubric(vf.Rubric):
    """Judge-based rubric for the interactive LHAW clarification task."""

    def __init__(
        self,
        judge_client: AsyncOpenAI,
        judge_model: str,
        judge_sampling_args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.judge_client = judge_client
        self.judge_model = judge_model
        self.judge_sampling_args = judge_sampling_args or {}
        self.add_reward_func(self.reconstruction_reward, weight=1.0)
        self.add_metric(self.ask_user_count_metric)
        self.add_metric(self.asked_any_metric)
        self.add_metric(self.final_answer_present_metric)

    async def _judge_yes_no(self, prompt: str, state: vf.State) -> float:
        cached = state.get("judge_response")
        if isinstance(cached, dict) and prompt in cached:
            content = str(cached[prompt])
            return 1.0 if "yes" in content.lower() else 0.0

        judge_args = {k: v for k, v in self.judge_sampling_args.items() if v is not None}
        response = await self.judge_client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            **judge_args,
        )
        content = str(response.choices[0].message.content or "")
        if not isinstance(cached, dict):
            cached = {}
        cached[prompt] = content
        state["judge_response"] = cached
        return 1.0 if "yes" in content.lower() else 0.0

    async def reconstruction_reward(self, state: vf.State, **_kwargs: Any) -> float:
        """Judge whether the final clarified task faithfully recovers the original task."""
        final_answer = extract_boxed_answer(str(state.get("final_answer", ""))).strip()
        if not final_answer:
            return 0.0

        info = state.get("info", {})
        if not isinstance(info, dict):
            info = {}

        original_prompt = str(info.get("original_prompt", ""))
        underspecified_prompt = str(info.get("underspecified_prompt", ""))
        ambiguity_class = str(info.get("ambiguity_class", ""))
        removed_segments_raw = info.get("removed_segments", [])
        removed_segments = removed_segments_raw if isinstance(removed_segments_raw, list) else []
        interactions = _extract_ask_user_interactions(state.get("completion", []))

        judge_prompt = _build_reconstruction_judge_prompt(
            original_prompt=original_prompt,
            underspecified_prompt=underspecified_prompt,
            ambiguity_class=ambiguity_class,
            removed_segments=removed_segments,
            interactions=interactions,
            final_answer=final_answer,
        )

        return await self._judge_yes_no(judge_prompt, state)

    async def ask_user_count_metric(self, state: vf.State, **_kwargs: Any) -> float:
        root_tool_calls = state.get("root_tool_calls", {})
        if not isinstance(root_tool_calls, dict):
            return 0.0
        return float(root_tool_calls.get("ask_user", 0) or 0.0)

    async def asked_any_metric(self, state: vf.State, **_kwargs: Any) -> float:
        return 1.0 if await self.ask_user_count_metric(state) > 0 else 0.0

    async def final_answer_present_metric(self, state: vf.State, **_kwargs: Any) -> float:
        final_answer = extract_boxed_answer(str(state.get("final_answer", ""))).strip()
        return 1.0 if final_answer else 0.0


class LHAWInteractiveRLMEnv(RLMEnv):
    """Interactive LHAW environment with an ask_user root tool."""

    def __init__(
        self,
        dataset: Any,
        rubric: vf.Rubric,
        user_simulator_client: AsyncOpenAI,
        user_simulator_model: str,
        **kwargs: Any,
    ) -> None:
        self.user_simulator_client = user_simulator_client
        self.user_simulator_model = user_simulator_model
        super().__init__(
            dataset=dataset,
            rubric=rubric,
            root_tools=[self.ask_user],
            **kwargs,
        )

    def _get_current_state_for_root_tool(self) -> dict[str, Any]:
        context = self._root_tool_context_var.get()
        if not isinstance(context, dict):
            raise RuntimeError("ask_user is only available inside the RLM REPL.")
        state = context.get("state")
        if not isinstance(state, dict):
            raise RuntimeError("Current rollout state is unavailable.")
        return state

    async def ask_user(self, question: str, context: str = "") -> str:
        """Ask a simulated user for missing task information.

        Args:
            question: Clarifying question for the user.
            context: Optional brief context about the current state of work.

        Returns:
            A concise response derived from the original task and removed segments.
        """
        state = self._get_current_state_for_root_tool()
        info = state.get("info", {})
        if not isinstance(info, dict):
            info = {}

        primary_task = str(info.get("original_prompt", ""))
        underspecified_prompt = str(info.get("underspecified_prompt", ""))
        removed_segments = info.get("removed_segments", [])
        removed_values = []
        if isinstance(removed_segments, list):
            for segment in removed_segments:
                if isinstance(segment, dict) and segment.get("value"):
                    removed_values.append(str(segment["value"]))

        system_prompt = f"""You are simulating a user who intended a fully specified task but only gave a partial prompt.

The COMPLETE original task was:
```text
{primary_task}
```

The UNDERSPECIFIED prompt the agent actually saw was:
```text
{underspecified_prompt}
```

Removed values:
{", ".join(removed_values) if removed_values else "None"}

Your job:
- answer only the agent's clarification question
- provide the exact missing information from the original task when possible
- be concise and natural
- do not reveal hidden metadata or mention that you are a simulator
"""

        user_prompt = f"Question: {question}"
        if context.strip():
            user_prompt += f"\n\nContext: {context.strip()}"

        response = await self.user_simulator_client.chat.completions.create(
            model=self.user_simulator_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return str(response.choices[0].message.content or "").strip()


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
    # Judge / simulator options
    judge_model: str = "gpt-5-mini",
    user_simulator_model: str = "openai/gpt-5.2",
    llm_api_key_var: str = "OPENAI_API_KEY",
    llm_base_url: str | None = None,
    # RLM options
    max_turns: int = 20,
    sub_llm_max_turns: int = 5,
    sub_model: str | None = None,
    max_sub_llm_parallelism: int = 5,
    max_output_length: int = 8192,
    code_execution_timeout: int = 120,
    abort_on_code_timeout: bool = False,
    max_startup_wait_seconds: int = 120,
    pip_install_packages: str = "",
    repl_language: Literal["bash", "python"] = "python",
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
    Load the interactive LHAW RLM environment.

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
        include_env_tips: If True, append environment-specific guidance to the prompt.
        judge_model: LLM used to judge the final clarified task against the
            original prompt.
        user_simulator_model: LLM used to answer `ask_user(...)` requests.
        llm_api_key_var: Environment variable containing the API key for judge
            and simulator calls.
        llm_base_url: Optional OpenAI-compatible base URL for judge and
            simulator calls.
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
            "Your job is to recover a fully specified, executable version of the task.\n"
            "If information is missing, use the `ask_user(question, context='')` tool.\n"
            "Once you have enough information, write the fully specified task into `answer[\"content\"]`.\n"
            "Do not execute the task itself; produce the clarified task specification only.\n\n"
            f"<underspecified_task>\n{example['underspecified_prompt']}\n</underspecified_task>"
        )
        if include_env_tips:
            prompt_content = prompt_content + _ENV_TIPS

        removed_segments = example.get("removed_segments", []) or []

        return {
            "example_id": example.get("variant_id", idx),
            "prompt": [{"role": "user", "content": prompt_content}],
            "task": "lhaw-interactive-clarification",
            "answer": example.get("original_prompt", ""),
            "info": {
                "variant_id": example.get("variant_id", ""),
                "original_task": example.get("original_task", ""),
                "source_dataset": example.get("dataset", ""),
                "ambiguity_class": example.get("ambiguity_class", ""),
                "information_dimension": example.get("information_dimension", []),
                "removed_segments": removed_segments,
                "expected_questions": example.get("expected_questions", []) or [],
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

    httpx_timeout = httpx.Timeout(1200)
    httpx_limits = httpx.Limits(max_connections=256, max_keepalive_connections=256)
    httpx_client = httpx.AsyncClient(limits=httpx_limits, timeout=httpx_timeout)
    api_key = os.getenv(llm_api_key_var) if llm_api_key_var else None
    llm_client = AsyncOpenAI(
        base_url=llm_base_url,
        api_key=api_key if api_key else "EMPTY",
        http_client=httpx_client,
    )
    rubric = LHAWJudgeRubric(
        judge_client=llm_client,
        judge_model=judge_model,
    )

    sandbox_labels = kwargs.pop("sandbox_labels", ["lhaw-rlm"])
    if not (isinstance(sandbox_labels, list) and all(isinstance(label, str) for label in sandbox_labels)):
        raise ValueError(f"sandbox_labels must be of type list[str]; you provided {sandbox_labels}")
    sandbox_labels = list(set(sandbox_labels))

    return LHAWInteractiveRLMEnv(
        dataset=dataset,
        rubric=rubric,
        user_simulator_client=llm_client,
        user_simulator_model=user_simulator_model,
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
        sandbox_labels=sandbox_labels,
        **kwargs,
    )
