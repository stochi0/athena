from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

import verifiers as vf
from openai import APIError, APITimeoutError, RateLimitError
from verifiers.types import Messages, State
from verifiers.utils.async_utils import maybe_await
from verifiers.utils.client_utils import resolve_client_config, setup_openai_client
from verifiers.utils.config_utils import ensure_keys

from core.config import EnvironmentConfig
from core.dataset import analyze_dataset
from core.prompts import JUDGE_PROMPT
from core.trajectory_files import read_trajectory_from_context_dir


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


def trajectory_text_for_judge(state: State | None, prompt: Messages) -> str:
    if state:
        info = state.get("info") or {}
        if isinstance(info, dict):
            cd = info.get("context_dir")
            if cd:
                text = read_trajectory_from_context_dir(str(cd))
                if text.strip():
                    return text
    if isinstance(prompt, list) and prompt:
        last = prompt[-1]
        if isinstance(last, dict) and "content" in last:
            return str(last.get("content", ""))
        content = getattr(last, "content", None)
        return str(content) if content is not None else ""
    return ""


def candidate_rubric_text(parser: vf.Parser, completion: Messages, state: State | None) -> str:
    if state and isinstance(state.get("final_answer"), str) and state["final_answer"].strip():
        return state["final_answer"].strip()
    return (parser.parse_answer(completion) or "").strip()


def parsed_rubric_count(
    parser: vf.Parser, completion: Messages, state: State | None = None
) -> float:
    text = candidate_rubric_text(parser, completion, state)
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
    """LLM judge on gold ``answer``; trajectory from on-disk context (RLM) or prompt fallback."""

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
        self.add_reward_func(self.judge_reward, weight=1.0)
        self.add_metric(self.rubric_count_metric)
        if cfg.attach_dataset_stats:
            self.add_class_object(
                "dataset_stats",
                analyze_dataset(cfg.dataset_name, cfg.dataset_split),
            )

    async def judge(
        self,
        prompt: Messages,
        completion: Messages,
        answer: str,
        state: State | None = None,
    ) -> str:
        trajectory = trajectory_text_for_judge(state, prompt)
        response = candidate_rubric_text(self.parser, completion, state)
        judge_prompt = self.judge_prompt.format(
            trajectory=trajectory,
            answer=answer,
            response=response,
        )
        cached = state.get("judge_response") if state else None
        if isinstance(cached, dict) and judge_prompt in cached:
            return cached[judge_prompt]

        judge_args = dict(self.judge_sampling_args or {})
        if "max_tokens" in judge_args:
            if judge_args["max_tokens"] is None:
                judge_args.pop("max_tokens")
            else:
                judge_args["max_completion_tokens"] = judge_args.pop("max_tokens")
        if (
            "max_completion_tokens" in judge_args
            and judge_args["max_completion_tokens"] is None
        ):
            judge_args.pop("max_completion_tokens")
        judge_args = {k: v for k, v in judge_args.items() if v is not None}

        try:
            judge_response = await maybe_await(
                self.judge_client.chat.completions.create,
                model=self.judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                **judge_args,
            )
            judge_response = str(judge_response.choices[0].message.content)
        except RateLimitError as e:
            raise RuntimeError(
                f"Judge model rate limit exceeded. Model: {self.judge_model}. {e}"
            ) from e
        except APITimeoutError as e:
            raise RuntimeError(
                f"Judge model timeout. Model: {self.judge_model}. {e}"
            ) from e
        except APIError as e:
            raise RuntimeError(
                f"Judge model API error. Model: {self.judge_model}. {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Unexpected judge error. Model: {self.judge_model}. {e}"
            ) from e

        if state:
            if not isinstance(cached, dict):
                cached = {}
            cached[judge_prompt] = judge_response
            state["judge_response"] = cached
        return judge_response

    async def judge_reward(
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
        self,
        parser: vf.Parser,
        completion: Messages,
        state: State | None = None,
        **_kwargs: Any,
    ) -> float:
        return parsed_rubric_count(parser, completion, state)


def build_rubric(cfg: EnvironmentConfig, parser: vf.Parser) -> vf.Rubric:
    return AdvancedIFJudgeRubric(cfg, parser=parser)
