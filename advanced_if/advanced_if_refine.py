"""AdvancedIF multi-turn refinement: tool submits rubrics; judge returns limited feedback only.

Uses ``facebook/AdvancedIF``, Prime Inference via ``PRIME_API_KEY`` (``judge_client_config`` defaults).
"""

from __future__ import annotations

import json
from typing import Any

import verifiers as vf
from core.dataset import parse_gold_rubrics_answer
from core.partial_feedback import format_limited_feedback, run_per_criterion_judge
from core.refine_config import RefineEnvironmentConfig
from core.refine_dataset import build_refine_dataset
from core.rubrics import _require_judge_api_key, build_rubric
from verifiers.utils.client_utils import setup_openai_client


def _make_submit_tool(
    judge_client: Any,
    judge_model: str,
    judge_sampling_args: dict[str, Any] | None,
):
    async def submit_candidate_rubrics(
        candidate_rubrics: str,
        _gold_json: str = "",
        _trajectory: str = "",
        _feedback_mode: str = "score_only",
    ) -> str:
        """Submit your current rubric JSON for limited judge feedback."""
        try:
            gold = json.loads(_gold_json)
        except json.JSONDecodeError:
            return "Tool error: internal gold rubric payload invalid."
        if not isinstance(gold, list) or not all(isinstance(x, str) for x in gold):
            return "Tool error: gold rubrics missing."

        try:
            full = await run_per_criterion_judge(
                judge_client,
                judge_model,
                judge_sampling_args,
                _trajectory,
                gold,
                candidate_rubrics,
            )
        except RuntimeError as e:
            return f"Judge error: {e}"

        if not full:
            return (
                "Judge returned an unparseable or invalid response. "
                "Ensure your candidate is clear JSON or text describing rubrics."
            )
        return format_limited_feedback(full, gold, _feedback_mode)

    return submit_candidate_rubrics


class AdvancedIFRefineEnv(vf.StatefulToolEnv):
    """Tool loop for rubric induction with partial judge feedback; final turn is plain JSON."""

    def __init__(self, config: RefineEnvironmentConfig):
        self.refine_config = config
        base = config.env
        _require_judge_api_key(base.judge_client_config)
        judge_client = setup_openai_client(base.judge_client_config)

        dataset = build_refine_dataset(base)
        parser = vf.Parser()
        rubric = build_rubric(base, parser)

        super().__init__(
            tools=[],
            dataset=dataset,
            rubric=rubric,
            parser=parser,
            env_id="advanced_if_refine",
            max_turns=config.max_turns,
        )

        submit = _make_submit_tool(
            judge_client,
            base.judge_model,
            base.judge_sampling_args,
        )
        self.add_tool(
            submit,
            args_to_skip=["_gold_json", "_trajectory", "_feedback_mode"],
        )

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> dict:
        if tool_name == "submit_candidate_rubrics":
            info = state.get("info") or {}
            gold = parse_gold_rubrics_answer(str(state.get("answer", "")))
            tool_args["_gold_json"] = json.dumps(gold, ensure_ascii=False)
            tool_args["_trajectory"] = str(info.get("trajectory", ""))
            tool_args["_feedback_mode"] = self.refine_config.feedback_mode
        return tool_args


def load_environment(
    config: RefineEnvironmentConfig | dict[str, Any] | None = None,
    **kwargs: Any,
) -> vf.Environment:
    if isinstance(config, RefineEnvironmentConfig):
        cfg = config
    else:
        merged = dict(config) if isinstance(config, dict) else {}
        merged.update(kwargs)
        cfg = RefineEnvironmentConfig.from_input(merged if merged else None)
    return AdvancedIFRefineEnv(cfg)
