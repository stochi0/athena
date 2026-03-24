"""
LHAW RLM Environment.

This environment supports two reward modes over the released `ScaleAI/lhaw`
dataset and the `verifiers` RLM runtime:

- reconstruction_judge:
  - the model sees an underspecified prompt
  - it can ask a simulated user for clarification via an `ask_user(...)` tool
  - it must produce a fully specified, clarified task as its final answer
  - an LLM judge compares that clarified task against the original prompt

- native_reward:
  - the model still sees an underspecified prompt and can use `ask_user(...)`
  - reward comes from benchmark-native results supplied in the example metadata
  - the environment reports paper-style native metrics when those signals are present

Unlike the full paper setup, this environment does not execute the underlying
benchmark tasks (e.g. TAC / SWE-Bench / MCP-Atlas native harnesses) and
therefore cannot reproduce benchmark-native pass@3 or checkpoint metrics inside
one standalone `verifiers` environment from the released Hugging Face schema
alone. The native-reward mode therefore expects native downstream outputs or
trajectory-linked metadata to be provided alongside each example.
"""

from __future__ import annotations

import verifiers as vf
from core.config import (
    EnvironmentConfig,
)
from core.dataset import load_rollout_dataset
from core.env import LHAWRLMEnv
from core.judging import LHAWJudgeRubric
from core.native_reward import NativeRewardRubric
from verifiers.utils.client_utils import resolve_client_config, setup_openai_client


def _normalize_sandbox_labels(raw_labels: object) -> list[str]:
    if not (isinstance(raw_labels, list) and all(isinstance(label, str) for label in raw_labels)):
        raise ValueError(f"sandbox_labels must be of type list[str]; you provided {raw_labels}")
    return sorted(set(raw_labels))


def load_environment(
    config: EnvironmentConfig | dict[str, object] | None = None,
    **kwargs: object,
) -> vf.Environment:
    """Load the LHAW RLM environment."""

    config_field_names = set(EnvironmentConfig.__dataclass_fields__)
    resolved_config = EnvironmentConfig.from_input(config, **kwargs)
    env_kwargs = {
        key: value for key, value in kwargs.items() if key not in config_field_names
    }
    dataset = load_rollout_dataset(resolved_config)

    rubric: vf.Rubric
    if resolved_config.reward_mode == "native_reward":
        rubric = NativeRewardRubric()
    else:
        judge_client_config = resolved_config.client_config or vf.ClientConfig()
        if resolved_config.judge_client_config is not None:
            judge_client_config = vf.ClientConfig.model_validate(
                {
                    **judge_client_config.model_dump(mode="python"),
                    **resolved_config.judge_client_config.model_dump(mode="python"),
                }
            )
        judge_client = setup_openai_client(resolve_client_config(judge_client_config))

        rubric = LHAWJudgeRubric(
            judge_client=judge_client,
            judge_model=resolved_config.judge_model,
        )

    user_simulator_client_config = resolved_config.client_config or vf.ClientConfig()
    if resolved_config.user_simulator_client_config is not None:
        user_simulator_client_config = vf.ClientConfig.model_validate(
            {
                **user_simulator_client_config.model_dump(mode="python"),
                **resolved_config.user_simulator_client_config.model_dump(mode="python"),
            }
        )
    user_simulator_client = setup_openai_client(
        resolve_client_config(user_simulator_client_config)
    )

    sandbox_labels = _normalize_sandbox_labels(
        env_kwargs.pop("sandbox_labels", ["lhaw-rlm"])
    )

    return LHAWRLMEnv(
        dataset=dataset,
        rubric=rubric,
        user_simulator_client=user_simulator_client,
        user_simulator_model=resolved_config.user_simulator_model,
        reward_mode=resolved_config.reward_mode,
        repl_language=resolved_config.repl_language,
        max_turns=resolved_config.max_turns,
        sub_llm_max_turns=resolved_config.sub_llm_max_turns,
        sub_model=resolved_config.sub_model,
        max_sub_llm_parallelism=resolved_config.max_sub_llm_parallelism,
        max_output_length=resolved_config.max_output_length,
        code_execution_timeout=resolved_config.code_execution_timeout,
        abort_on_code_timeout=resolved_config.abort_on_code_timeout,
        max_startup_wait_seconds=resolved_config.max_startup_wait_seconds,
        pip_install_packages=resolved_config.pip_install_packages,
        sandbox_docker_image=resolved_config.sandbox_docker_image,
        sandbox_cpu_cores=resolved_config.sandbox_cpu_cores,
        sandbox_memory_gb=resolved_config.sandbox_memory_gb,
        sandbox_disk_size_gb=resolved_config.sandbox_disk_size_gb,
        sandbox_gpu_count=resolved_config.sandbox_gpu_count,
        sandbox_timeout_minutes=resolved_config.sandbox_timeout_minutes,
        sandbox_labels=sandbox_labels,
        **env_kwargs,
    )
