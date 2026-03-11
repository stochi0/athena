"""RL eval types: sample (input/response/score), result (reward + per-sample)."""

from __future__ import annotations

from typing_extensions import TypedDict


class EvalSample(TypedDict):
    """One (input, response, score) for rubric eval."""
    input: str
    response: str
    score: float


class EvalResult(TypedDict):
    """Result of running rubric on samples: scalar reward + per-sample scores/correct."""
    reward: float
    scores: list[float]
    correct: list[bool]
    error: str | None
