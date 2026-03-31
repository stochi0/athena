from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EnvironmentConfig:
    """Configuration for the AdvancedIF rubric-generation environment."""

    dataset_name: str = "facebook/AdvancedIF"
    dataset_split: str = "train"
    max_examples: int | None = None
    seed: int = 0
    judge_model: str = "gpt-4.1-mini"
    judge_sampling_args: dict[str, Any] | None = None
    judge_api_key_var: str = "PRIME_API_KEY"
    judge_base_url: str = "https://api.pinference.ai/api/v1"
    max_turns: int = 1
    include_dataset_analysis_in_state: bool = True

    @classmethod
    def from_input(
        cls, cfg: EnvironmentConfig | dict[str, Any] | None
    ) -> EnvironmentConfig:
        if cfg is None:
            return cls()
        if isinstance(cfg, cls):
            return cfg
        allowed = {k: v for k, v in cfg.items() if k in cls.__dataclass_fields__}
        return cls(**allowed)
