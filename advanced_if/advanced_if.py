"""Verifiers environment for rubric generation on facebook/AdvancedIF.

Public surface: ``load_environment`` and ``AdvancedIFEnv`` (``vf.MultiTurnEnv``).
Implementation details live under ``core/`` (same pattern as ``loca_bench_rlm``).
"""

from __future__ import annotations

from typing import Any

import verifiers as vf
from verifiers.types import State

from core.config import EnvironmentConfig
from core.dataset import build_dataset
from core.evaluation import build_rubric

class AdvancedIFEnv(vf.MultiTurnEnv):
    """Single rollout turn: model emits rubrics JSON; ``JudgeRubric`` scores vs gold."""

    def __init__(self, config: EnvironmentConfig):
        self.config = config
        dataset = build_dataset(config)
        parser = vf.Parser()
        rubric = build_rubric(config, parser)
        super().__init__(
            dataset=dataset,
            rubric=rubric,
            parser=parser,
            max_turns=config.max_turns,
            env_id="advanced_if",
        )

    async def env_response(
        self, messages: vf.Messages, state: State, **kwargs: Any
    ) -> vf.Messages:
        state["final_env_response"] = []
        return []

    @vf.stop
    async def done_after_first_turn(self, state: State, **kwargs: Any) -> bool:
        return len(state["trajectory"]) >= 1


def load_environment(
    config: EnvironmentConfig | dict[str, Any] | None = None,
    **kwargs: Any,
) -> vf.Environment:
    if isinstance(config, EnvironmentConfig):
        cfg = config
    else:
        merged = dict(config) if isinstance(config, dict) else {}
        merged.update(kwargs)
        cfg = EnvironmentConfig.from_input(merged if merged else None)
    return AdvancedIFEnv(cfg)
