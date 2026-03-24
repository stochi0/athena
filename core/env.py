from __future__ import annotations

from typing import Any, cast

import verifiers as vf
from datasets import Dataset
from verifiers.envs.experimental.rlm_env import RLMEnv
from verifiers.utils.async_utils import maybe_await


def extract_removed_values(removed_segments: object) -> list[str]:
    if not isinstance(removed_segments, list):
        return []
    return [
        str(segment["value"])
        for segment in removed_segments
        if isinstance(segment, dict) and segment.get("value")
    ]


class LHAWRLMEnv(RLMEnv):
    """LHAW environment with an ask_user root tool."""

    def __init__(
        self,
        dataset: Dataset,
        rubric: vf.Rubric,
        user_simulator_client: Any,
        user_simulator_model: str,
        **kwargs: object,
    ) -> None:
        self.user_simulator_client = user_simulator_client
        self.user_simulator_model = user_simulator_model
        super().__init__(
            dataset=dataset,
            rubric=rubric,
            root_tools=[self.ask_user],
            **kwargs,
        )

    def _get_current_state_for_root_tool(self) -> vf.State:
        context = self._root_tool_context_var.get()
        if not isinstance(context, dict):
            raise RuntimeError("ask_user is only available inside the RLM REPL.")
        state = context.get("state")
        if not isinstance(state, dict):
            raise RuntimeError("Current rollout state is unavailable.")
        return cast(vf.State, state)

    async def ask_user(self, question: str, context: str = "") -> str:
        """Ask a simulated user for missing task information."""
        state = self._get_current_state_for_root_tool()
        info = state.get("info", {})
        if not isinstance(info, dict):
            info = {}

        primary_task = str(info.get("original_prompt", ""))
        underspecified_prompt = str(info.get("underspecified_prompt", ""))
        removed_values = extract_removed_values(info.get("removed_segments", []))

        system_prompt = f"""You are the intended user in an RLM clarification loop.
The agent is trying to turn an underspecified task into a complete, executable task specification.

The COMPLETE original task was:
```text
{primary_task}
```

The UNDERSPECIFIED prompt the agent actually saw was:
```text
{underspecified_prompt}
```

Removed values:
{", ".join(removed_values) if removed_values else "None"}

Your job:
- answer only the agent's clarification question
- provide the exact missing information from the original task when possible
- be concise and natural
- prefer direct answers over explanations unless the question asks for explanation
- do not reveal hidden metadata or mention that you are a simulator
"""

        user_prompt = f"Question: {question}"
        if context.strip():
            user_prompt += f"\n\nContext: {context.strip()}"

        response = await maybe_await(
            self.user_simulator_client.chat.completions.create,
            model=self.user_simulator_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return str(response.choices[0].message.content or "").strip()
