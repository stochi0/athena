from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

import verifiers as vf
from verifiers.types import Messages, State
from verifiers.utils.client_utils import resolve_client_config, setup_openai_client
from verifiers.utils.config_utils import ensure_keys

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


_JUDGE_KEYS = ("coverage", "faithful", "non_redundant")


def mean_judge_scores(obj: dict[str, Any]) -> float | None:
    """Mean in [0, 1] if all three boolean criteria are present, else None."""
    scores: list[float] = []
    for key in _JUDGE_KEYS:
        if key not in obj:
            return None
        value = obj[key]
        if not isinstance(value, bool):
            return None
        scores.append(1.0 if value else 0.0)
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


def _require_judge_api_key(config: vf.ClientConfig) -> None:
    ensure_keys([resolve_client_config(config).api_key_var])


class AdvancedIFJudgeRubric(vf.JudgeRubric):
    """LLM judge alignment reward plus rubric-count metric."""

    def __init__(self, cfg: EnvironmentConfig, parser: vf.Parser | None = None):
        _require_judge_api_key(cfg.judge_client_config)
        judge_client = setup_openai_client(cfg.judge_client_config)

        super().__init__(
            parser=parser,
            judge_client=judge_client,
            judge_model=cfg.judge_model,
            judge_prompt=JUDGE_PROMPT,
            judge_sampling_args=cfg.judge_sampling_args,
        )
        self.add_reward_func(self.rubric_alignment_reward, weight=1.0)
        self.add_metric(self.rubric_count_metric)
        if cfg.attach_dataset_stats:
            self.add_class_object(
                "dataset_stats",
                analyze_dataset(cfg.dataset_name, cfg.dataset_split),
            )

    async def rubric_alignment_reward(
        self,
        judge: Callable[..., Awaitable[str]],
        prompt: Messages,
        completion: Messages,
        answer: str,
        state: State,
        **_kwargs: Any,
    ) -> float:
        raw = await judge(prompt, completion, answer, state)
        obj = extract_json_object(raw)
        if not obj:
            return 0.0
        score = mean_judge_scores(obj)
        return score if score is not None else 0.0

    async def rubric_count_metric(
        self, parser: vf.Parser, completion: Messages, **_kwargs: Any
    ) -> float:
        return parsed_rubric_count(parser, completion)


def build_rubric(cfg: EnvironmentConfig, parser: vf.Parser) -> vf.Rubric:
    return AdvancedIFJudgeRubric(cfg, parser=parser)
