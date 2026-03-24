from __future__ import annotations

import random
from typing import cast

import verifiers as vf
from datasets import Dataset, load_dataset

from .config import EnvironmentConfig
from .constants import (
    AMBIGUITY_CLASSES,
    ENV_TIPS,
    HF_DATASET_NAME,
    INFORMATION_DIMENSIONS,
    SOURCE_DATASETS,
    TASK_PROMPT_PREFIX,
)


def validate_config(config: EnvironmentConfig) -> None:
    if config.source_dataset != "all" and config.source_dataset not in SOURCE_DATASETS:
        raise ValueError(
            f"source_dataset={config.source_dataset!r} is invalid. "
            f"Must be 'all' or one of {sorted(SOURCE_DATASETS)}."
        )
    if config.ambiguity_class != "all" and config.ambiguity_class not in AMBIGUITY_CLASSES:
        raise ValueError(
            f"ambiguity_class={config.ambiguity_class!r} is invalid. "
            f"Must be 'all' or one of {sorted(AMBIGUITY_CLASSES)}."
        )

    invalid_dimensions = sorted(set(config.requested_dimensions) - INFORMATION_DIMENSIONS)
    if invalid_dimensions:
        raise ValueError(
            f"information_dimension contains invalid values {invalid_dimensions}. "
            f"Valid values: {sorted(INFORMATION_DIMENSIONS)}."
        )

    if config.max_examples is not None and config.max_examples < 0:
        raise ValueError(f"max_examples must be >= 0; got {config.max_examples}.")


def build_prompt_content(underspecified_prompt: str, include_env_tips: bool) -> str:
    prompt_content = (
        TASK_PROMPT_PREFIX
        + f"<underspecified_task>\n{underspecified_prompt}\n</underspecified_task>"
    )
    if include_env_tips:
        prompt_content += ENV_TIPS
    return prompt_content


def transform_example(
    example: vf.Info,
    idx: int,
    *,
    include_env_tips: bool,
) -> vf.RolloutInput:
    removed_segments = example.get("removed_segments", []) or []
    underspecified_prompt = str(example.get("underspecified_prompt", ""))

    return {
        "example_id": example.get("variant_id", idx),
        "prompt": [
            {
                "role": "user",
                "content": build_prompt_content(underspecified_prompt, include_env_tips),
            }
        ],
        "task": "lhaw-interactive-clarification",
        "answer": example.get("original_prompt", ""),
        "info": {
            "variant_id": example.get("variant_id", ""),
            "original_task": example.get("original_task", ""),
            "source_dataset": example.get("dataset", ""),
            "ambiguity_class": example.get("ambiguity_class", ""),
            "information_dimension": example.get("information_dimension", []),
            "removed_segments": removed_segments,
            "expected_questions": example.get("expected_questions", []) or [],
            "original_prompt": example.get("original_prompt", ""),
            "underspecified_prompt": underspecified_prompt,
        },
    }


def load_rollout_dataset(config: EnvironmentConfig) -> Dataset:
    validate_config(config)

    raw_dataset = cast(Dataset, load_dataset(HF_DATASET_NAME, split=config.split))

    if config.source_dataset != "all":
        raw_dataset = raw_dataset.filter(
            lambda example: example["dataset"] == config.source_dataset
        )
    if config.ambiguity_class != "all":
        raw_dataset = raw_dataset.filter(
            lambda example: example["ambiguity_class"] == config.ambiguity_class
        )
    if config.requested_dimensions:
        requested_dimension_set = set(config.requested_dimensions)
        raw_dataset = raw_dataset.filter(
            lambda example: requested_dimension_set.issubset(
                set(example["information_dimension"])
            )
        )

    dataset = raw_dataset.map(
        lambda example, idx: transform_example(
            cast(vf.Info, example),
            idx,
            include_env_tips=config.include_env_tips,
        ),
        with_indices=True,
        remove_columns=raw_dataset.column_names,
        writer_batch_size=100,
    )

    if config.shuffle:
        seed = config.seed if config.seed is not None else random.randint(1000, 100_000_000)
        dataset = dataset.shuffle(seed=seed)

    if config.max_examples is not None:
        dataset = dataset.select(range(min(config.max_examples, len(dataset))))

    return dataset
