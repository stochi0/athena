"""Verifiers environment for rubric generation on facebook/AdvancedIF.

Public surface: ``load_environment`` and ``AdvancedIFEnv`` (``vf.SingleTurnEnv``).
Implementation details live under ``core/``.
"""

from __future__ import annotations

from typing import Any

import verifiers as vf

from core.config import EnvironmentConfig
from core.dataset import build_dataset
from core.rubrics import build_rubric


class AdvancedIFEnv(vf.SingleTurnEnv):
    """One model turn: emit rubrics JSON; ``AdvancedIFJudgeRubric`` scores via LLM judge."""

    def __init__(self, config: EnvironmentConfig):
        self.config = config
        dataset = build_dataset(config)
        parser = vf.Parser()
        rubric = build_rubric(config, parser)
        super().__init__(
            dataset=dataset,
            rubric=rubric,
            parser=parser,
            env_id="advanced_if",
        )


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
