from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from core.config import EnvironmentConfig


@dataclass(frozen=True)
class RefineEnvironmentConfig:
    """AdvancedIF rubric refinement with limited judge tool feedback."""

    env: EnvironmentConfig
    feedback_mode: Literal["score_only", "one_violation"] = "score_only"
    max_turns: int = 8

    @classmethod
    def from_input(
        cls, cfg: RefineEnvironmentConfig | dict[str, Any] | None, **kwargs: Any
    ) -> RefineEnvironmentConfig:
        if isinstance(cfg, cls):
            return cfg
        raw: dict[str, Any] = dict(cfg) if isinstance(cfg, dict) else {}
        raw.update(kwargs)
        feedback_mode = raw.pop("feedback_mode", "score_only")
        max_turns = int(raw.pop("max_turns", 8))
        if feedback_mode not in ("score_only", "one_violation"):
            raise ValueError(
                f"feedback_mode must be 'score_only' or 'one_violation', got {feedback_mode!r}"
            )
        if max_turns < 2:
            raise ValueError("max_turns must be at least 2 for refinement (tool loop + final answer)")
        base = EnvironmentConfig.from_input(raw if raw else None)
        return cls(env=base, feedback_mode=feedback_mode, max_turns=max_turns)
