from __future__ import annotations

import json
from collections import Counter

import verifiers as vf
from verifiers.envs.experimental.rlm_env import RLMMonitorRubric

from .utils import normalize_items, parse_final_answer_payload


async def correctness(state: vf.State) -> float:
    expected = {item.lower() for item in normalize_items(state.get("answer"))}
    payload = parse_final_answer_payload(state.get("final_answer"))
    predicted = {item.lower() for item in normalize_items(payload.get("answer"))}
    if not expected and not predicted:
        return 1.0
    if not expected or not predicted:
        return 0.0
    if expected == predicted:
        return 1.0
    return len(expected & predicted) / len(expected)


async def citation_support(state: vf.State) -> float:
    payload = parse_final_answer_payload(state.get("final_answer"))
    citations = payload.get("citations", [])
    if not isinstance(citations, list) or not citations:
        return 0.0

    observations = json.dumps(
        state.get("root_tool_observations", []), sort_keys=True
    ).lower()
    supported = 0
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        excerpt = str(citation.get("excerpt", "")).strip().lower()
        document_id = str(citation.get("document_id", "")).strip().lower()
        if excerpt and excerpt in observations:
            supported += 1
            continue
        if document_id and document_id in observations:
            supported += 1
    return supported / len(citations)


async def grounded_tool_use(state: vf.State) -> float:
    tool_calls = state.get("root_tool_calls", {})
    invocations = state.get("root_tool_invocations", [])

    if not isinstance(tool_calls, dict):
        tool_calls = {}
    if not isinstance(invocations, list):
        invocations = []

    positive_counts = [
        int(count)
        for count in tool_calls.values()
        if isinstance(count, (int, float)) and count > 0
    ]
    total_calls = sum(positive_counts)
    unique_tools = len(positive_counts)

    if total_calls == 0 and not invocations:
        return 0.0
    if unique_tools >= 2 or total_calls >= 3 or len(invocations) >= 3:
        return 1.0
    return 0.6


async def retrieval_efficiency(state: vf.State) -> float:
    invocations = state.get("root_tool_invocations", [])
    if not isinstance(invocations, list) or not invocations:
        return 1.0
    counts = Counter(invocations)
    redundant = sum(count - 1 for count in counts.values() if count > 1)
    return max(0.0, 1.0 - (redundant / len(invocations)))


def build_default_rubric(root_tool_names: list[str] | None = None) -> vf.Rubric:
    rubric = RLMMonitorRubric(root_tool_names=root_tool_names)
    rubric.add_reward_func(correctness, weight=1.0)
    rubric.add_reward_func(citation_support, weight=0.35)
    rubric.add_reward_func(grounded_tool_use, weight=0.2)
    rubric.add_reward_func(retrieval_efficiency, weight=0.15)
    return rubric
