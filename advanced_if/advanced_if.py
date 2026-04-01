"""AdvancedIF rubric induction via ``RLMEnv``: trajectory on disk; optional partial judge tool."""

from __future__ import annotations

from typing import Any, cast

import verifiers as vf
from core.config import EnvironmentConfig
from core.dataset import build_dataset, parse_gold_rubrics_answer
from core.partial_feedback import format_limited_feedback, run_per_criterion_judge
from core.rubrics import _require_judge_api_key, build_rubric
from core.trajectory_files import read_trajectory_from_context_dir
from verifiers.envs.experimental.rlm_env import RLMEnv
from verifiers.utils.client_utils import setup_openai_client


class AdvancedIFRLMEnv(RLMEnv):
    """Filesystem-backed conversation + ``submit_candidate_rubrics`` for limited judge feedback."""

    def __init__(self, config: EnvironmentConfig, **rlm_kwargs: Any):
        self.adv_config = config
        _require_judge_api_key(config.judge_client_config)
        self._partial_judge_client = setup_openai_client(config.judge_client_config)

        dataset = build_dataset(config)
        parser = vf.Parser()
        rubric = build_rubric(config, parser)

        repl_language = rlm_kwargs.pop("repl_language", "python")
        root_prompt_verbosity = rlm_kwargs.pop("root_prompt_verbosity", "medium")
        super().__init__(
            dataset=dataset,
            rubric=rubric,
            parser=parser,
            env_id="advanced_if",
            max_turns=config.max_turns,
            root_tools=[self.submit_candidate_rubrics],
            repl_language=repl_language,
            root_prompt_verbosity=root_prompt_verbosity,
            **rlm_kwargs,
        )

    def _state_for_root_tool(self) -> vf.State:
        ctx = self._root_tool_context_var.get()
        if not isinstance(ctx, dict):
            raise RuntimeError("submit_candidate_rubrics is only available inside the RLM REPL.")
        state = ctx.get("state")
        if not isinstance(state, dict):
            raise RuntimeError("Rollout state is unavailable.")
        return cast(vf.State, state)

    async def submit_candidate_rubrics(self, candidate_rubrics: str) -> str:
        """Submit draft rubric JSON for limited feedback (gold stays server-side in ``answer``)."""
        state = self._state_for_root_tool()
        cfg = self.adv_config
        gold = parse_gold_rubrics_answer(str(state.get("answer", "")))
        info = state.get("info") or {}
        traj = ""
        if isinstance(info, dict) and info.get("context_dir"):
            traj = read_trajectory_from_context_dir(str(info["context_dir"]))

        try:
            full = await run_per_criterion_judge(
                self._partial_judge_client,
                cfg.judge_model,
                cfg.judge_sampling_args,
                traj,
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
        return format_limited_feedback(full, gold, cfg.feedback_mode)


def load_environment(
    config: EnvironmentConfig | dict[str, Any] | None = None,
    **kwargs: Any,
) -> vf.Environment:
    """Load AdvancedIF RLM environment. Extra kwargs are passed to ``RLMEnv`` (sandbox, sub_model, …)."""
    if isinstance(config, EnvironmentConfig):
        cfg = config
        rlm_kw = dict(kwargs)
    else:
        merged = dict(config) if isinstance(config, dict) else {}
        merged.update(kwargs)
        fields = set(EnvironmentConfig.__dataclass_fields__)
        cfg = EnvironmentConfig.from_input(
            {k: v for k, v in merged.items() if k in fields} if merged else None
        )
        rlm_kw = {k: v for k, v in merged.items() if k not in fields}

    if cfg.feedback_mode not in ("score_only", "one_violation"):
        raise ValueError(
            f"feedback_mode must be 'score_only' or 'one_violation', got {cfg.feedback_mode!r}"
        )

    return AdvancedIFRLMEnv(cfg, **rlm_kw)
