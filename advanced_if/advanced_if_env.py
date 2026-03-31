"""
Standalone verifiers-native environment for rubric discovery on facebook/AdvancedIF.

Task: given a full conversation trajectory, generate a rubric list that captures
the user instruction constraints. Reward is computed by an LLM judge that compares
the generated rubric against the dataset gold rubric list.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from datasets import Dataset, load_dataset
import verifiers as vf
from verifiers.clients import Client, resolve_client
from verifiers.errors import ModelError
from verifiers.types import ClientConfig, Info, Messages, Response, State

ENV_ID = "advanced_if_rubric_discovery"
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a rubric induction assistant. Read the full trajectory and infer a concise, "
    "testable checklist of criteria for evaluating the assistant's response quality."
)

USER_TEMPLATE = """Infer evaluation rubrics from this full conversation trajectory.

Return JSON only with:
{{"rubrics": ["criterion 1", "criterion 2", ...]}}

Guidelines:
- Each criterion must be atomic and testable.
- Avoid duplicates and vague wording.
- Preserve task-specific constraints and formatting requirements.

Trajectory:
{trajectory}
"""

JUDGE_PROMPT = """Compare a candidate rubric list to the gold rubric list for the same conversation.

Conversation trajectory:
{question}

Gold rubrics:
{answer}

Candidate rubrics:
{response}

Return JSON only with keys:
- "coverage" (boolean): candidate covers the essential gold constraints.
- "faithful" (boolean): candidate does not introduce major incorrect constraints.
- "non_redundant" (boolean): candidate list is not overly repetitive.
"""


def _parse_conversation_history(raw: str) -> list[dict[str, Any]]:
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("conversation_history must be a JSON list")
    return data


def _parse_rubrics(raw: str) -> list[str]:
    meta = json.loads(raw)
    rubrics = meta.get("rubrics", [])
    if isinstance(rubrics, str):
        rubrics = json.loads(rubrics)
    if not isinstance(rubrics, list) or not all(isinstance(x, str) for x in rubrics):
        raise ValueError("prompt_metadata.rubrics must be a list[str]")
    return rubrics


def _render_trajectory(messages: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for m in messages:
        role = str(m.get("role", ""))
        content = m.get("content", "")
        if isinstance(content, list):
            content = str(content)
        blocks.append(f"[{role}]\n{content}")
    return "\n\n".join(blocks)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if "```" in cleaned:
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.MULTILINE)
    start = cleaned.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _judge_completion_text(response: Response) -> str:
    content = response.message.content
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return " ".join(chunks).strip()
    return str(content)


class AdvancedIFJudgeRubric(vf.Rubric):
    """Judge rubric using verifiers native client resolution."""

    def __init__(
        self,
        *,
        judge_client: Client | ClientConfig,
        judge_model: str,
        parser: vf.Parser | None = None,
        judge_sampling_args: dict[str, Any] | None = None,
        judge_prompt: str = JUDGE_PROMPT,
    ):
        super().__init__(parser=parser or vf.Parser())
        self._judge_vf_client = (
            resolve_client(judge_client)
            if isinstance(judge_client, ClientConfig)
            else judge_client
        )
        self.judge_model = judge_model
        self.judge_sampling_args = judge_sampling_args or {}
        self.judge_prompt = judge_prompt
        self.class_objects = {
            "parser": self.parser,
            "judge": self.judge,
            "judge_client": self._judge_vf_client,
            "judge_model": self.judge_model,
            "judge_sampling_args": self.judge_sampling_args,
            "judge_prompt": self.judge_prompt,
        }

    async def judge(
        self,
        prompt: Messages,
        completion: Messages,
        answer: str,
        state: State | None = None,
    ) -> str:
        question = str(prompt[-1].get("content", "")) if prompt else ""
        response = self.parser.parse_answer(completion) or ""
        judge_prompt = self.judge_prompt.format(
            question=question,
            answer=answer,
            response=response,
        )
        cached = state.get("judge_response") if state else None
        if isinstance(cached, dict) and judge_prompt in cached:
            return cached[judge_prompt]

        judge_args = {k: v for k, v in (self.judge_sampling_args or {}).items() if v is not None}
        try:
            judge_resp = await self._judge_vf_client.get_response(
                prompt=[{"role": "user", "content": judge_prompt}],
                model=self.judge_model,
                sampling_args=judge_args,
            )
            judge_response = _judge_completion_text(judge_resp)
        except ModelError as exc:
            self.logger.warning(
                "Judge model error for '%s': %s",
                self.judge_model,
                str(exc),
            )
            raise RuntimeError(
                f"Judge model error. Model: {self.judge_model}, Error: {str(exc)}"
            ) from exc
        except Exception as exc:
            self.logger.warning(
                "Unexpected judge error for '%s': %s",
                self.judge_model,
                str(exc),
            )
            raise RuntimeError(
                f"Unexpected judge error. Model: {self.judge_model}, Error: {str(exc)}"
            ) from exc

        if state:
            if not isinstance(cached, dict):
                cached = {}
            cached[judge_prompt] = judge_response
            state["judge_response"] = cached
        return judge_response


def _build_rollout_row(ex: dict[str, Any], idx: int) -> dict[str, Any]:
    history = _parse_conversation_history(ex["conversation_history"])
    rubrics = _parse_rubrics(ex["prompt_metadata"])
    trajectory = _render_trajectory(history)
    benchmark = str(ex.get("benchmark_name", "unknown"))

    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(trajectory=trajectory)},
        ],
        "answer": json.dumps(rubrics, ensure_ascii=False),
        "task": f"advanced_if::{benchmark}",
        "example_id": idx,
        "info": {
            "benchmark_name": benchmark,
            "num_messages": len(history),
            "num_rubrics": len(rubrics),
            "trajectory": trajectory,
            "gold_rubrics": rubrics,
        },
    }


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

        history = _parse_conversation_history(ex["conversation_history"])
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
        user_turn_histogram[str(num_user)] = user_turn_histogram.get(str(num_user), 0) + 1

        rubrics = _parse_rubrics(ex["prompt_metadata"])
        rubric_count_histogram[str(len(rubrics))] = (
            rubric_count_histogram.get(str(len(rubrics)), 0) + 1
        )

    top_benchmarks = sorted(
        benchmark_counts.items(), key=lambda x: x[1], reverse=True
    )[:20]
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


@dataclass(frozen=True)
class Config:
    dataset_name: str = "facebook/AdvancedIF"
    dataset_split: str = "train"
    max_examples: int | None = None
    seed: int = 0
    # Judge config is verifiers-native and resolves through verifiers clients.
    judge_model: str = "gpt-4.1-mini"
    judge_sampling_args: dict[str, Any] | None = None
    use_llm_judge: bool = True
    judge_api_key_var: str = "PRIME_API_KEY"
    judge_base_url: str = "https://api.pinference.ai/api/v1"
    max_turns: int = 1
    include_dataset_analysis_in_state: bool = True

    @classmethod
    def from_input(cls, cfg: Config | dict[str, Any] | None) -> Config:
        if cfg is None:
            return cls()
        if isinstance(cfg, cls):
            return cfg
        allowed = {k: v for k, v in cfg.items() if k in cls.__dataclass_fields__}
        return cls(**allowed)


def load_environment(
    config: Config | dict[str, Any] | None = None,
    **kwargs: Any,
) -> vf.Environment:
    if isinstance(config, Config):
        cfg = config
    else:
        merged = dict(config) if isinstance(config, dict) else {}
        merged.update(kwargs)
        cfg = Config.from_input(merged if merged else None)

    ds = load_dataset(cfg.dataset_name, split=cfg.dataset_split)
    if cfg.max_examples is not None and cfg.max_examples > 0:
        ds = ds.shuffle(seed=cfg.seed).select(range(min(cfg.max_examples, len(ds))))

    rollout_rows = [
        _build_rollout_row(ex, idx)
        for idx, ex in enumerate(ds)
    ]
    rollout_dataset = Dataset.from_list(rollout_rows)

    parser = vf.Parser()
    resolved_judge_client: Client | ClientConfig = ClientConfig(
        client_type="openai_chat_completions",
        api_key_var=cfg.judge_api_key_var,
        api_base_url=cfg.judge_base_url,
    )
    rubric: vf.Rubric = AdvancedIFJudgeRubric(
        judge_client=resolved_judge_client,
        judge_model=cfg.judge_model,
        parser=parser,
        judge_prompt=JUDGE_PROMPT,
        judge_sampling_args=cfg.judge_sampling_args or {"temperature": 0.0},
    )

    async def rubric_alignment_reward(
        judge: Any,
        prompt: vf.Messages,
        completion: vf.Messages,
        answer: str,
        state: State,
        **kwargs: Any,
    ) -> float:
        raw = await judge(prompt, completion, answer, state)
        obj = _extract_json_object(raw)
        if not obj:
            return 0.0
        checks = []
        for key in ("coverage", "faithful", "non_redundant"):
            val = obj.get(key)
            if isinstance(val, bool):
                checks.append(1.0 if val else 0.0)
            elif isinstance(val, (int, float)):
                checks.append(1.0 if float(val) >= 0.5 else 0.0)
        if len(checks) != 3:
            return 0.0
        return sum(checks) / 3.0

    async def rubric_count_metric(state: State, **kwargs: Any) -> float:
        completion = state.get("completion")
        if not isinstance(completion, list) or not completion:
            return 0.0
        content = str(completion[-1].get("content", ""))
        obj = _extract_json_object(content)
        if not obj:
            return 0.0
        rubrics = obj.get("rubrics")
        if not isinstance(rubrics, list):
            return 0.0
        return float(len([r for r in rubrics if isinstance(r, str) and r.strip()]))

    rubric.add_reward_func(rubric_alignment_reward, weight=1.0)
    rubric.add_metric(rubric_count_metric)

    dataset_profile = analyze_dataset(cfg.dataset_name, cfg.dataset_split)
    if cfg.include_dataset_analysis_in_state:
        rubric.add_class_object("dataset_profile", dataset_profile)

    class AdvancedIFRubricDiscoveryEnv(vf.MultiTurnEnv):
        async def env_response(
            self, messages: vf.Messages, state: State, **kwargs: Any
        ) -> vf.Messages:
            # Single-turn task: no extra environment response, then stop.
            state["final_env_response"] = []
            return []

        @vf.stop
        async def done_after_first_turn(self, state: State, **kwargs: Any) -> bool:
            return len(state["trajectory"]) >= 1

    return AdvancedIFRubricDiscoveryEnv(
        dataset=rollout_dataset,
        rubric=rubric,
        parser=parser,
        max_turns=cfg.max_turns,
        env_id=ENV_ID,
    )

