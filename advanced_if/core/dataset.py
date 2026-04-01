from __future__ import annotations

import json
from typing import Any

import verifiers as vf
from datasets import Dataset, load_dataset
from verifiers.types import Info, RolloutInput

from core.config import EnvironmentConfig
from core.prompts import SYSTEM_PROMPT, USER_TEMPLATE


def parse_conversation_history(raw: str) -> list[dict[str, Any]]:
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("conversation_history must be a JSON list")
    return data


def parse_rubrics_from_metadata(raw: str) -> list[str]:
    meta = json.loads(raw)
    rubrics = meta.get("rubrics", [])
    if isinstance(rubrics, str):
        rubrics = json.loads(rubrics)
    if not isinstance(rubrics, list) or not all(isinstance(x, str) for x in rubrics):
        raise ValueError("prompt_metadata.rubrics must be a list[str]")
    return rubrics


def render_trajectory(messages: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for m in messages:
        role = str(m.get("role", ""))
        content = m.get("content", "")
        if isinstance(content, list):
            content = str(content)
        blocks.append(f"[{role}]\n{content}")
    return "\n\n".join(blocks)


def parse_gold_rubrics_answer(answer: str) -> list[str]:
    """Parse rollout ``answer`` (JSON list of strings). Used for judges/tools; not shown to the policy."""
    try:
        data = json.loads(answer)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        return []
    return data


def build_rollout_row(ex: dict[str, Any], idx: int) -> RolloutInput:
    history = parse_conversation_history(ex["conversation_history"])
    rubrics = parse_rubrics_from_metadata(ex["prompt_metadata"])
    trajectory = render_trajectory(history)
    benchmark = str(ex.get("benchmark_name", "unknown"))

    return RolloutInput(
        prompt=[
            vf.SystemMessage(content=SYSTEM_PROMPT).model_dump(mode="python"),
            vf.UserMessage(
                content=USER_TEMPLATE.format(trajectory=trajectory)
            ).model_dump(mode="python"),
        ],
        answer=json.dumps(rubrics, ensure_ascii=False),
        task=f"advanced_if::{benchmark}",
        example_id=idx,
        # Only non-sensitive fields: gold rubrics live in ``answer`` for scoring / judge.
        info={"trajectory": trajectory},
    )


def analyze_dataset(dataset_name: str, dataset_split: str) -> Info:
    ds = load_dataset(dataset_name, split=dataset_split)
    n = len(ds)

    benchmark_counts: dict[str, int] = {}
    role_pattern_counts: dict[str, int] = {}
    rubric_count_histogram: dict[str, int] = {}
    assistant_turn_histogram: dict[str, int] = {}
    user_turn_histogram: dict[str, int] = {}
    message_count_histogram: dict[str, int] = {}

    for ex in ds:
        benchmark = str(ex.get("benchmark_name", "unknown"))
        benchmark_counts[benchmark] = benchmark_counts.get(benchmark, 0) + 1

        history = parse_conversation_history(ex["conversation_history"])
        roles = [str(m.get("role", "")) for m in history]
        role_key = "->".join(roles)
        role_pattern_counts[role_key] = role_pattern_counts.get(role_key, 0) + 1

        num_messages = len(history)
        message_count_histogram[str(num_messages)] = (
            message_count_histogram.get(str(num_messages), 0) + 1
        )

        num_assistant = sum(1 for r in roles if r == "assistant")
        assistant_turn_histogram[str(num_assistant)] = (
            assistant_turn_histogram.get(str(num_assistant), 0) + 1
        )

        num_user = sum(1 for r in roles if r == "user")
        user_turn_histogram[str(num_user)] = (
            user_turn_histogram.get(str(num_user), 0) + 1
        )

        rubrics = parse_rubrics_from_metadata(ex["prompt_metadata"])
        rubric_count_histogram[str(len(rubrics))] = (
            rubric_count_histogram.get(str(len(rubrics)), 0) + 1
        )

    top_benchmarks = sorted(benchmark_counts.items(), key=lambda x: x[1], reverse=True)[
        :20
    ]
    top_role_patterns = sorted(
        role_pattern_counts.items(), key=lambda x: x[1], reverse=True
    )[:20]

    return {
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
        "num_rows": n,
        "top_benchmarks": top_benchmarks,
        "top_role_patterns": top_role_patterns,
        "message_count_histogram": message_count_histogram,
        "assistant_turn_histogram": assistant_turn_histogram,
        "user_turn_histogram": user_turn_histogram,
        "rubric_count_histogram": rubric_count_histogram,
    }


def build_dataset(cfg: EnvironmentConfig) -> Dataset:
    ds = load_dataset(cfg.dataset_name, split=cfg.dataset_split)
    if cfg.max_examples is not None and cfg.max_examples > 0:
        ds = ds.shuffle(seed=cfg.seed).select(range(min(cfg.max_examples, len(ds))))

    rollout_rows = [build_rollout_row(ex, idx) for idx, ex in enumerate(ds)]
    return Dataset.from_list(rollout_rows)
