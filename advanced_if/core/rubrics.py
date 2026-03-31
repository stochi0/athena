from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import AsyncOpenAI
import verifiers as vf
from verifiers.types import Messages, State

from core.config import EnvironmentConfig
from core.dataset import analyze_dataset
from core.prompts import JUDGE_PROMPT


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object from model text (handles fenced blocks)."""
    cleaned = text.strip()
    if "```" in cleaned:
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.MULTILINE)
    start = cleaned.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _normalize_key(key: str) -> str:
    return key.strip().lower().replace(" ", "_").replace("-", "_")


def _normalize_judge_obj(obj: dict[str, Any]) -> dict[str, Any]:
    return {_normalize_key(k): v for k, v in obj.items() if isinstance(k, str)}


def _coerce_score(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        x = float(value)
        if 0.0 <= x <= 1.0:
            return x
        return 1.0 if x >= 0.5 else 0.0
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "yes", "1"):
            return 1.0
        if s in ("false", "no", "0"):
            return 0.0
    return None


# Judge may paraphrase keys; map synonym groups to logical criteria.
_CRITERIA_ALIASES: tuple[tuple[str, ...], ...] = (
    ("coverage", "covers", "covers_essentials", "essential_coverage"),
    ("faithful", "faithfulness", "accuracy", "accurate", "factuality"),
    (
        "non_redundant",
        "nonredundant",
        "not_redundant",
        "concise",
        "low_redundancy",
        "minimal_redundancy",
    ),
)


def mean_judge_scores(norm: dict[str, Any]) -> float | None:
    """Return mean in [0, 1] if every criterion is present and coercible, else None."""
    scores: list[float] = []
    for aliases in _CRITERIA_ALIASES:
        found: float | None = None
        for name in aliases:
            if name in norm:
                found = _coerce_score(norm[name])
                break
        if found is None:
            return None
        scores.append(found)
    return sum(scores) / len(scores)


def parsed_rubric_count(parser: vf.Parser, completion: Messages) -> float:
    text = parser.parse_answer(completion) or ""
    obj = extract_json_object(text)
    if not obj:
        return 0.0
    rubrics = obj.get("rubrics")
    if not isinstance(rubrics, list):
        return 0.0
    return float(len([r for r in rubrics if isinstance(r, str) and r.strip()]))


class AdvancedIFJudgeRubric(vf.JudgeRubric):
    """LLM judge plus alignment reward and rubric-count metric (verifiers-native)."""

    def __init__(self, cfg: EnvironmentConfig, parser: vf.Parser | None = None):
        judge_api_key = os.getenv(cfg.judge_api_key_var, "")
        if not judge_api_key.strip():
            raise RuntimeError(
                f"Missing judge API key in env var '{cfg.judge_api_key_var}'. "
                "Set it before loading the environment."
            )

        super().__init__(
            parser=parser,
            judge_client=AsyncOpenAI(
                api_key=judge_api_key,
                base_url=cfg.judge_base_url,
            ),
            judge_model=cfg.judge_model,
            judge_prompt=JUDGE_PROMPT,
            judge_sampling_args=cfg.judge_sampling_args or {"temperature": 0.0},
        )
        self.add_reward_func(self.rubric_alignment_reward, weight=1.0)
        self.add_metric(self.rubric_count_metric)
        if cfg.include_dataset_analysis_in_state:
            self.add_class_object(
                "dataset_profile",
                analyze_dataset(cfg.dataset_name, cfg.dataset_split),
            )

    async def rubric_alignment_reward(
        self,
        judge: Any,
        prompt: Messages,
        completion: Messages,
        answer: str,
        state: State,
        **kwargs: Any,
    ) -> float:
        raw = await judge(prompt, completion, answer, state)
        obj = extract_json_object(raw)
        if not obj:
            return 0.0
        norm = _normalize_judge_obj(obj)
        score = mean_judge_scores(norm)
        return score if score is not None else 0.0

    async def rubric_count_metric(
        self, parser: vf.Parser, completion: Messages, **kwargs: Any
    ) -> float:
        return parsed_rubric_count(parser, completion)


def build_rubric(cfg: EnvironmentConfig, parser: vf.Parser) -> vf.Rubric:
    return AdvancedIFJudgeRubric(cfg, parser=parser)
