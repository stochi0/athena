from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import AsyncOpenAI
import verifiers as vf
from verifiers.types import State

from core.config import EnvironmentConfig
from core.dataset import analyze_dataset
from core.prompts import JUDGE_PROMPT


def extract_json_object(text: str) -> dict[str, Any] | None:
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


def build_rubric(cfg: EnvironmentConfig, parser: vf.Parser) -> vf.Rubric:
    judge_api_key = os.getenv(cfg.judge_api_key_var, "")
    if not judge_api_key.strip():
        raise RuntimeError(
            f"Missing judge API key in env var '{cfg.judge_api_key_var}'. "
            "Set it before loading the environment."
        )

    rubric: vf.Rubric = vf.JudgeRubric(
        parser=parser,
        judge_client=AsyncOpenAI(
            api_key=judge_api_key,
            base_url=cfg.judge_base_url,
        ),
        judge_model=cfg.judge_model,
        judge_prompt=JUDGE_PROMPT,
        judge_sampling_args=cfg.judge_sampling_args or {"temperature": 0.0},
    )

    async def rubric_alignment_reward(
        judge: Any,
        prompt: vf.Messages,
        completion: vf.Messages,
        answer: str,
        state: State,
        **kwargs: Any,
    ) -> float:
        raw = await judge(prompt, completion, answer, state)
        obj = extract_json_object(raw)
        if not obj:
            return 0.0
        checks: list[float] = []
        for key in ("coverage", "faithful", "non_redundant"):
            val = obj.get(key)
            if isinstance(val, bool):
                checks.append(1.0 if val else 0.0)
            elif isinstance(val, (int, float)):
                checks.append(1.0 if float(val) >= 0.5 else 0.0)
        if len(checks) != 3:
            return 0.0
        return sum(checks) / 3.0

    async def rubric_count_metric(state: State, **kwargs: Any) -> float:
        completion = state.get("completion")
        if not isinstance(completion, list) or not completion:
            return 0.0
        content = str(completion[-1].get("content", ""))
        obj = extract_json_object(content)
        if not obj:
            return 0.0
        rubrics = obj.get("rubrics")
        if not isinstance(rubrics, list):
            return 0.0
        return float(len([r for r in rubrics if isinstance(r, str) and r.strip()]))

    rubric.add_reward_func(rubric_alignment_reward, weight=1.0)
    rubric.add_metric(rubric_count_metric)

    if cfg.include_dataset_analysis_in_state:
        dataset_profile = analyze_dataset(cfg.dataset_name, cfg.dataset_split)
        rubric.add_class_object("dataset_profile", dataset_profile)

    return rubric
