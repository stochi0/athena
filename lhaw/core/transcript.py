from __future__ import annotations

import json
from typing import cast

import verifiers as vf
from verifiers.utils.message_utils import normalize_messages

from .types import AskUserInteraction


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
            interactions[pending_by_id[message.tool_call_id]]["response"] = content_to_text(
                message.content
            ).strip()

    return interactions


def format_ask_user_transcript(interactions: list[AskUserInteraction]) -> str:
    """Render ask_user interactions for prompts and debugging."""
    if not interactions:
        return "No ask_user calls."

    lines = []
    for index, interaction in enumerate(interactions, start=1):
        lines.append(f"Question {index}: {interaction.get('question', '')}")
        if interaction.get("context"):
            lines.append(f"Context {index}: {interaction['context']}")
        lines.append(f"User response {index}: {interaction.get('response', '')}")
    return "\n".join(lines)
