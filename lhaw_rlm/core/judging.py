from __future__ import annotations

import json
from typing import Any, cast

import verifiers as vf
from verifiers.utils.async_utils import maybe_await
from verifiers.utils.data_utils import extract_boxed_answer
from verifiers.utils.message_utils import normalize_messages

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


def content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")

    parts: list[str] = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            continue
        text = getattr(part, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return " ".join(part for part in parts if part).strip()


def parse_ask_user_arguments(arguments: str) -> tuple[str, str]:
    question = ""
    context = ""
    try:
        parsed_args = json.loads(arguments)
    except json.JSONDecodeError:
        return arguments.strip(), ""

    if isinstance(parsed_args, dict):
        question = str(parsed_args.get("question", "")).strip()
        context = str(parsed_args.get("context", "")).strip()
    return question, context


def normalize_info(value: object) -> vf.Info:
    return cast(vf.Info, value) if isinstance(value, dict) else {}


def normalize_removed_segments(value: object) -> list[vf.Info]:
    return cast(list[vf.Info], value) if isinstance(value, list) else []


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


def extract_ask_user_interactions(
    completion: vf.Messages | None,
) -> list[AskUserInteraction]:
    """Extract ask_user question/response pairs from the final conversation."""
    if not completion:
        return []

    interactions: list[AskUserInteraction] = []
    pending_by_id: dict[str, int] = {}

    for message in normalize_messages(completion, field_name="completion"):
        if message.role == "assistant":
            for tool_call in message.tool_calls or []:
                if tool_call.name != "ask_user":
                    continue

                question, context = parse_ask_user_arguments(tool_call.arguments)

                interactions.append(
                    {
                        "question": question,
                        "context": context,
                        "response": "",
                    }
                )
                pending_by_id[tool_call.id] = len(interactions) - 1

        elif message.role == "tool" and message.tool_call_id in pending_by_id:
            interactions[pending_by_id[message.tool_call_id]]["response"] = (
                content_to_text(message.content).strip()
            )

    return interactions


def format_ask_user_transcript(interactions: list[AskUserInteraction]) -> str:
    """Render ask_user interactions for judging."""
    if not interactions:
        return "No ask_user calls."

    lines = []
    for index, interaction in enumerate(interactions, start=1):
        lines.append(f"Question {index}: {interaction.get('question', '')}")
        if interaction.get("context"):
            lines.append(f"Context {index}: {interaction['context']}")
        lines.append(f"User response {index}: {interaction.get('response', '')}")
    return "\n".join(lines)


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

Answer "yes" only if the final clarified task should count as a faithful reconstruction of the original task under the guidance above.
Otherwise answer "no".

Respond with only "yes" or "no"."""


class LHAWJudgeRubric(vf.JudgeRubric):
    """Judge-based rubric for the interactive LHAW clarification task."""

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
