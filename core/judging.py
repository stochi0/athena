from __future__ import annotations

from typing import Any

import verifiers as vf
from verifiers.utils.async_utils import maybe_await
from verifiers.utils.data_utils import extract_boxed_answer

from .transcript import (
    extract_ask_user_interactions,
    format_ask_user_transcript,
    normalize_info,
    normalize_removed_segments,
)
from .types import AskUserInteraction


def format_removed_segments(segments: list[vf.Info]) -> str:
    """Render removed segments for prompts and judging."""
    if not segments:
        return "None"

    lines = []
    for segment in segments:
        dimension = str(segment.get("dimension", ""))
        subdimension = str(segment.get("subdimension", ""))
        value = str(segment.get("value", ""))
        segment_id = str(segment.get("id", ""))
        lines.append(
            f"- id={segment_id or 'unknown'} dimension={dimension or 'unknown'} "
            f"subdimension={subdimension or 'unknown'} value={value or '(empty)'}"
        )
    return "\n".join(lines)


def normalize_judge_sampling_args(
    sampling_args: vf.SamplingArgs | None,
) -> vf.SamplingArgs:
    judge_args = dict(sampling_args or {})
    if "max_tokens" in judge_args:
        max_tokens = judge_args.pop("max_tokens")
        if max_tokens is not None:
            judge_args["max_completion_tokens"] = max_tokens
    if judge_args.get("max_completion_tokens") is None:
        judge_args.pop("max_completion_tokens", None)
    return {key: value for key, value in judge_args.items() if value is not None}


def ambiguity_class_judge_guidance(ambiguity_class: str) -> str:
    """Return class-specific judge instructions grounded in the paper taxonomy."""
    if ambiguity_class == "outcome-critical":
        return (
            "This is an outcome-critical variant.\n"
            "- Missing information should be treated as necessary for reliable success.\n"
            "- The final clarified task should explicitly restore the critical missing details from the original task.\n"
            "- If the final clarified task still leaves a blocker unresolved, answer 'no'."
        )
    if ambiguity_class == "divergent":
        return (
            "This is a divergent variant.\n"
            "- The underspecification can lead to variable but plausible outcomes.\n"
            "- Answer 'yes' if the final clarified task is a faithful, executable reconstruction aligned with the original task,\n"
            "  even if the clarification transcript shows one reasonable path among several plausible ones.\n"
            "- Answer 'no' if the final clarified task drifts to a materially different task or leaves key ambiguity unresolved."
        )
    if ambiguity_class == "benign":
        return (
            "This is a benign variant.\n"
            "- The missing information is expected to be inferable from context or low-risk for successful completion.\n"
            "- Do not require unnecessary explicit clarification if the final clarified task is still faithful to the original task.\n"
            "- Answer 'no' only if the final clarified task introduces contradictions, loses essential intent, or is not executable."
        )
    return (
        "Use the original task as the reference and decide whether the final clarified task is faithful, executable, "
        "and does not materially distort the intended task."
    )


def build_reconstruction_judge_prompt(
    *,
    original_prompt: str,
    underspecified_prompt: str,
    ambiguity_class: str,
    removed_segments: list[vf.Info],
    interactions: list[AskUserInteraction],
    final_answer: str,
) -> str:
    """Build the judge prompt for final clarified task reconstruction."""
    removed_segments_text = format_removed_segments(removed_segments)
    interaction_text = format_ask_user_transcript(interactions)
    class_guidance = ambiguity_class_judge_guidance(ambiguity_class)

    return f"""You are evaluating an agent on the LHAW underspecification benchmark.
The agent operated in an RLM environment with a persistent REPL.
Intermediate drafts, tool usage, and clarification turns are allowed.
Judge the final clarified task, using the transcript only as supporting evidence.

The agent originally saw this UNDERSPECIFIED task:
```text
{underspecified_prompt}
```

The full ORIGINAL task was:
```text
{original_prompt}
```

Ambiguity class:
{ambiguity_class}

Removed segments:
{removed_segments_text}

Clarification transcript:
{interaction_text}

The agent's FINAL clarified task:
```text
{final_answer}
```

Class-specific guidance:
{class_guidance}

General evaluation criteria:
- preserve the original task intent and scope
- restore missing information when required by the ambiguity class
- avoid inventing contradictory requirements not supported by the original task
- produce a clarified task that is usable and executable as an instruction
- do not penalize the agent for asking clarifying questions if the final task is faithful
- do not require unnecessary clarification if the final task already reconstructs the original intent

Answer "yes" only if the final clarified task should count as a faithful reconstruction of the original task under the guidance above.
Otherwise answer "no".

Respond with only "yes" or "no"."""


class LHAWJudgeRubric(vf.JudgeRubric):
    """Judge-based rubric for the LHAW clarification task."""

    def __init__(
        self,
        judge_client: Any,
        judge_model: str,
        judge_sampling_args: vf.SamplingArgs | None = None,
    ) -> None:
        super().__init__(
            judge_client=judge_client,
            judge_model=judge_model,
            judge_sampling_args=judge_sampling_args,
        )
        self.add_reward_func(self.reconstruction_reward, weight=1.0)
        self.add_metric(self.final_answer_present_metric)

    async def _judge_yes_no(self, prompt: str, state: vf.State) -> float:
        cached = state.get("judge_response")
        if isinstance(cached, dict) and prompt in cached:
            content = str(cached[prompt])
            return 1.0 if "yes" in content.lower() else 0.0

        response = await maybe_await(
            self.judge_client.chat.completions.create,
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            **normalize_judge_sampling_args(self.judge_sampling_args),
        )
        content = str(response.choices[0].message.content or "")
        if not isinstance(cached, dict):
            cached = {}
        cached[prompt] = content
        state["judge_response"] = cached
        return 1.0 if "yes" in content.lower() else 0.0

    async def reconstruction_reward(self, state: vf.State, **_kwargs: object) -> float:
        """Judge whether the final clarified task faithfully recovers the original task."""
        final_answer = extract_boxed_answer(str(state.get("final_answer", ""))).strip()
        if not final_answer:
            return 0.0

        info = normalize_info(state.get("info", {}))

        original_prompt = str(info.get("original_prompt", ""))
        underspecified_prompt = str(info.get("underspecified_prompt", ""))
        ambiguity_class = str(info.get("ambiguity_class", ""))
        removed_segments = normalize_removed_segments(info.get("removed_segments", []))
        interactions = extract_ask_user_interactions(state.get("completion", []))

        judge_prompt = build_reconstruction_judge_prompt(
            original_prompt=original_prompt,
            underspecified_prompt=underspecified_prompt,
            ambiguity_class=ambiguity_class,
            removed_segments=removed_segments,
            interactions=interactions,
            final_answer=final_answer,
        )

        return await self._judge_yes_no(judge_prompt, state)

    async def final_answer_present_metric(
        self, state: vf.State, **_kwargs: object
    ) -> float:
        final_answer = extract_boxed_answer(str(state.get("final_answer", ""))).strip()
        return 1.0 if final_answer else 0.0
