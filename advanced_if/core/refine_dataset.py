from __future__ import annotations

import json

import verifiers as vf
from datasets import Dataset
from verifiers.types import RolloutInput

from core.config import EnvironmentConfig
from core.dataset import (
    parse_conversation_history,
    parse_rubrics_from_metadata,
    render_trajectory,
)
from core.refine_prompts import REFINE_SYSTEM_PROMPT, REFINE_USER_TEMPLATE


def build_refine_rollout_row(ex: dict, idx: int) -> RolloutInput:
    history = parse_conversation_history(ex["conversation_history"])
    rubrics = parse_rubrics_from_metadata(ex["prompt_metadata"])
    trajectory = render_trajectory(history)
    benchmark = str(ex.get("benchmark_name", "unknown"))

    return RolloutInput(
        prompt=[
            vf.SystemMessage(content=REFINE_SYSTEM_PROMPT).model_dump(mode="python"),
            vf.UserMessage(
                content=REFINE_USER_TEMPLATE.format(trajectory=trajectory)
            ).model_dump(mode="python"),
        ],
        answer=json.dumps(rubrics, ensure_ascii=False),
        task=f"advanced_if_refine::{benchmark}",
        example_id=idx,
        info={"trajectory": trajectory},
    )


def build_refine_dataset(cfg: EnvironmentConfig) -> Dataset:
    from datasets import load_dataset

    ds = load_dataset(cfg.dataset_name, split=cfg.dataset_split)
    if cfg.max_examples is not None and cfg.max_examples > 0:
        ds = ds.shuffle(seed=cfg.seed).select(range(min(cfg.max_examples, len(ds))))

    rows = [build_refine_rollout_row(ex, idx) for idx, ex in enumerate(ds)]
    return Dataset.from_list(rows)
