from __future__ import annotations

import asyncio

import pytest
from core.dataset import build_prompt_content, transform_example
from core.native_reward import NativeRewardRubric, pass_at_k
from core.state import PRIVATE_METADATA_KEY


def test_build_prompt_content_changes_with_reward_mode() -> None:
    reconstruction_prompt = build_prompt_content(
        "Do the task.", include_env_tips=True, reward_mode="reconstruction_judge"
    )
    paper_prompt = build_prompt_content(
        "Do the task.", include_env_tips=True, reward_mode="native_reward"
    )

    assert "reconstruct a fully specified" in reconstruction_prompt
    assert "Do not execute or solve the task itself" in reconstruction_prompt
    assert "solve the task itself" in paper_prompt
    assert "Prefer autonomous progress" in paper_prompt


def test_pass_at_k_matches_expected_zero_and_one_edges() -> None:
    assert pass_at_k(3, 0, 3) == 0.0
    assert pass_at_k(3, 3, 3) == 1.0


def test_native_reward_uses_native_result_payload() -> None:
    rubric = NativeRewardRubric()
    state = {
        PRIVATE_METADATA_KEY: {
            "native_result": {
                "success": False,
                "score": 2,
                "total": 4,
            }
        },
        "completion": [],
        "final_answer": "",
    }

    reward = asyncio.run(rubric.native_reward(state))
    native_score = asyncio.run(rubric.native_score_metric(state))
    native_success = asyncio.run(rubric.native_success_metric(state))

    assert reward == 0.5
    assert native_score == 0.5
    assert native_success == 0.0


def test_native_reward_metrics_compute_from_trial_lists() -> None:
    rubric = NativeRewardRubric()
    state = {
        PRIVATE_METADATA_KEY: {
            "native_result": {
                "success": True,
                "score": 1,
                "total": 1,
            },
            "native_trials": [
                {"success": True, "score": 1, "total": 1, "questions_asked": 1},
                {"success": False, "score": 0, "total": 1, "questions_asked": 0},
                {"success": True, "score": 1, "total": 1, "questions_asked": 2},
            ],
            "native_baseline_trials": [
                {"success": False, "score": 0, "total": 1, "questions_asked": 0},
                {"success": False, "score": 0, "total": 1, "questions_asked": 0},
                {"success": False, "score": 0, "total": 1, "questions_asked": 0},
            ],
        },
        "completion": [],
        "final_answer": "",
    }

    pass_at_3_metric = asyncio.run(rubric.pass_at_3_metric(state))
    checkpoint_percent_metric = asyncio.run(rubric.checkpoint_percent_metric(state))
    ask_percent_metric = asyncio.run(rubric.ask_percent_metric(state))
    avg_questions_metric = asyncio.run(rubric.avg_questions_per_trajectory_metric(state))
    gain_per_question_metric = asyncio.run(rubric.gain_per_question_metric(state))

    assert pass_at_3_metric == 1.0
    assert checkpoint_percent_metric == pytest.approx(2 / 3)
    assert ask_percent_metric == pytest.approx(2 / 3)
    assert avg_questions_metric == 1.5
    assert gain_per_question_metric > 0.0


def test_transform_example_keeps_oracle_fields_out_of_public_info() -> None:
    rollout_input = transform_example(
        {
            "variant_id": "variant-1",
            "dataset": "source-bench",
            "ambiguity_class": "divergent",
            "information_dimension": ["context"],
            "original_task": "task-name",
            "underspecified_prompt": "Do the thing.",
            "original_prompt": "Do the specific thing with these exact inputs.",
            "removed_segments": [{"id": "seg-1", "value": "exact inputs"}],
            "expected_questions": [{"segment_id": "seg-1", "question": "Which inputs?"}],
            "terminal_states": "done",
            "native_result": {"success": True, "score": 1, "total": 1},
        },
        0,
        include_env_tips=False,
        reward_mode="reconstruction_judge",
    )

    public_info = rollout_input["info"]
    assert isinstance(public_info, dict)
    assert set(public_info) == {
        "variant_id",
        "source_dataset",
        "ambiguity_class",
        "information_dimension",
    }
    assert "original_prompt" not in public_info
    assert "removed_segments" not in public_info
    assert "expected_questions" not in public_info
    assert "native_result" not in public_info

    private_metadata = rollout_input[PRIVATE_METADATA_KEY]
    assert isinstance(private_metadata, dict)
    assert private_metadata["original_prompt"] == "Do the specific thing with these exact inputs."
    assert private_metadata["removed_segments"] == [{"id": "seg-1", "value": "exact inputs"}]
