from __future__ import annotations

import json
import re
from typing import Any

from openai import APIError, APITimeoutError, AsyncOpenAI, RateLimitError
from verifiers.utils.async_utils import maybe_await

from core.refine_prompts import PER_CRITERION_JUDGE_PROMPT


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


def _numbered_gold(rubrics: list[str]) -> str:
    lines = [f"{i}. {r}" for i, r in enumerate(rubrics)]
    return "\n".join(lines)


async def run_per_criterion_judge(
    client: AsyncOpenAI,
    judge_model: str,
    judge_sampling_args: dict[str, Any] | None,
    trajectory: str,
    gold_rubrics: list[str],
    candidate_rubrics: str,
) -> dict[str, Any] | None:
    """Call judge; return parsed JSON with satisfied list + optional aggregate booleans."""
    n = len(gold_rubrics)
    if n == 0:
        return {"satisfied": []}

    user = PER_CRITERION_JUDGE_PROMPT.format(
        trajectory=trajectory,
        numbered_gold=_numbered_gold(gold_rubrics),
        candidate=candidate_rubrics.strip(),
        n=n,
    )
    args = dict(judge_sampling_args or {})
    if "max_tokens" in args:
        if args["max_tokens"] is None:
            args.pop("max_tokens")
        else:
            args["max_completion_tokens"] = args.pop("max_tokens")
    if "max_completion_tokens" in args and args["max_completion_tokens"] is None:
        args.pop("max_completion_tokens")
    args = {k: v for k, v in args.items() if v is not None}

    try:
        resp = await maybe_await(
            client.chat.completions.create,
            model=judge_model,
            messages=[{"role": "user", "content": user}],
            **args,
        )
        raw = str(resp.choices[0].message.content)
    except RateLimitError as e:
        raise RuntimeError(f"Judge rate limit: {e}") from e
    except APITimeoutError as e:
        raise RuntimeError(f"Judge timeout: {e}") from e
    except APIError as e:
        raise RuntimeError(f"Judge API error: {e}") from e

    obj = extract_json_object(raw)
    if not obj:
        return None
    sat = obj.get("satisfied")
    if not isinstance(sat, list) or len(sat) != n:
        return None
    if not all(isinstance(x, bool) for x in sat):
        return None
    return obj


def format_limited_feedback(
    obj: dict[str, Any],
    gold_rubrics: list[str],
    mode: str,
) -> str:
    """Turn full judge object into the narrow channel the agent sees."""
    if mode not in ("score_only", "one_violation"):
        mode = "score_only"
    sat: list[bool] = obj.get("satisfied", [])
    if not sat:
        return "Judge: no per-criterion breakdown available; try another candidate."

    score = sum(1 for x in sat if x) / len(sat)
    if mode == "score_only":
        return (
            f"Judge feedback (score only): fraction of gold criteria covered = {score:.3f} "
            f"({sum(1 for x in sat if x)}/{len(sat)})."
        )

    for i, ok in enumerate(sat):
        if not ok and i < len(gold_rubrics):
            return (
                "Judge feedback (single violated criterion): "
                f"gold item #{i} is not adequately covered — \"{gold_rubrics[i]}\""
            )
    return (
        "Judge feedback (single violated criterion): no failing item reported; "
        f"all {len(sat)} gold checks passed in this review."
    )
