"""RL eval types: sample (prompt/completion/score), result (reward + per-sample). Aligned with verifiers: prompt = input text, completion = model output text."""

from __future__ import annotations

from typing_extensions import TypedDict


class EvalSample(TypedDict):
    """One (prompt, completion, score) for rubric eval. Matches verifiers State naming (string form)."""
    prompt: str
    completion: str
    score: float


class EvalResult(TypedDict):
    """Result of running rubric on samples: scalar reward + per-sample scores/correct."""
    reward: float
    scores: list[float]
    correct: list[bool]
    error: str | None
