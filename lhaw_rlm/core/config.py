from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import verifiers as vf


SourceDataset = Literal["all", "MCP-Atlas", "TheAgentCompany", "SWE-Bench Pro"]
AmbiguityClass = Literal["all", "outcome-critical", "divergent", "benign"]
InformationDimension = Literal["all", "goal", "constraint", "input", "context"]
SingleInformationDimension = Literal["goal", "constraint", "input", "context"]
ReplLanguage = Literal["bash", "python"]


def as_list(value: str | list[str] | None) -> list[str]:
    """Convert a scalar or list-like option into a list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


@dataclass(frozen=True)
class EnvironmentConfig:
    split: str = "test"
    source_dataset: SourceDataset = "all"
    ambiguity_class: AmbiguityClass = "all"
    information_dimension: InformationDimension | list[SingleInformationDimension] = "all"
    max_examples: int | None = None
    shuffle: bool = False
    seed: int | None = None
    include_env_tips: bool = False
    judge_model: str = "z-ai/glm-4.7"
    user_simulator_model: str = "openai/gpt-4.1-mini"
    client_config: vf.ClientConfig | None = None
    judge_client_config: vf.ClientConfig | None = None
    user_simulator_client_config: vf.ClientConfig | None = None
    max_turns: int = 20
    sub_llm_max_turns: int = 5
    sub_model: str | None = None
    max_sub_llm_parallelism: int = 5
    max_output_length: int = 8192
    code_execution_timeout: int = 120
    abort_on_code_timeout: bool = False
    max_startup_wait_seconds: int = 120
    pip_install_packages: str = ""
    repl_language: ReplLanguage = "python"
    sandbox_docker_image: str = "python:3.11-slim"
    sandbox_cpu_cores: int = 1
    sandbox_memory_gb: int = 2
    sandbox_disk_size_gb: int = 5
    sandbox_gpu_count: int = 0
    sandbox_timeout_minutes: int = 60

    @classmethod
    def from_input(
        cls,
        config: "EnvironmentConfig | dict[str, object] | None" = None,
        **overrides: object,
    ) -> "EnvironmentConfig":
        if isinstance(config, cls):
            data = config.__dict__.copy()
        elif isinstance(config, dict):
            data = dict(config)
        elif config is None:
            data = {}
        else:
            raise TypeError(f"Unsupported config type: {type(config)!r}")

        data.update(overrides)
        valid_fields = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in data.items() if key in valid_fields})

    @property
    def requested_dimensions(self) -> list[str]:
        if self.information_dimension == "all":
            return []
        return as_list(self.information_dimension)
