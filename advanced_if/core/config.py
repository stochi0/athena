from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from verifiers.types import ClientConfig


@dataclass(frozen=True)
class EnvironmentConfig:
    """Configuration for AdvancedIF over ``RLMEnv`` (filesystem trajectory + optional partial judge tool)."""

    dataset_name: str = "facebook/AdvancedIF"
    dataset_split: str = "train"
    max_examples: int | None = None
    seed: int = 0
    judge_model: str = "gpt-4.1-mini"
    judge_sampling_args: dict[str, Any] | None = None
    judge_client_config: ClientConfig = field(default_factory=ClientConfig)
    # RLM root max turns (separate from Prime eval worker settings in TOML).
    max_turns: int = 64
    attach_dataset_stats: bool = True
    feedback_mode: str = "score_only"
    # If None, materialized trajectories go under ``<package>/contexts/advanced_if_rlm/``.
    context_parent_dir: str | None = None

    @classmethod
    def from_input(
        cls, cfg: EnvironmentConfig | dict[str, Any] | None, **kwargs: Any
    ) -> EnvironmentConfig:
        if isinstance(cfg, cls):
            return cfg
        raw: dict[str, Any] = dict(cfg) if isinstance(cfg, dict) else {}
        raw.update(kwargs)

        jcc_raw = raw.pop("judge_client_config", None)
        merged_cc = ClientConfig().model_dump(mode="python")
        if isinstance(jcc_raw, ClientConfig):
            merged_cc.update(jcc_raw.model_dump(mode="python"))
        elif isinstance(jcc_raw, dict):
            merged_cc.update(jcc_raw)

        judge_client_config = ClientConfig.model_validate(merged_cc)
        allowed = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        allowed["judge_client_config"] = judge_client_config
        return cls(**allowed)
