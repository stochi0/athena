from __future__ import annotations

import contextvars
from pathlib import Path
from typing import Any

import verifiers as vf
from datasets import Dataset
from dotenv import load_dotenv
from verifiers.envs.experimental.rlm_env import RLMEnv

from . import config
from .adapters import FileAdapter, GraphAdapter, SQLiteAdapter, VectorAdapter
from .config import Config
from .context_builder import prepare_rows
from .rewards import build_default_rubric
from .tools import WorkspaceTools
from .workspace import (
    build_workspace_state,
    ensure_workspace,
    get_paths_from_workspace_state,
)

load_dotenv()


class LongContextRetrievalEnv(WorkspaceTools, RLMEnv):
    """RLM environment for autonomous QA over long-context document workspaces."""

    def __init__(
        self,
        *,
        cfg: Config,
        dataset: Any = None,
        rubric: vf.Rubric | None = None,
        workspace_anchor: Path | None = None,
        **kwargs: Any,
    ) -> None:
        self._workspace_anchor = (
            workspace_anchor.resolve()
            if workspace_anchor is not None
            else (
                Path(cfg.path_anchor).resolve()
                if cfg.path_anchor
                else Path.cwd().resolve()
            )
        )
        self._subtool_state_var: contextvars.ContextVar[dict[str, Any] | None] = (
            contextvars.ContextVar("long_context_retrieval_env_subtool_state", default=None)
        )
        self._sqlite = SQLiteAdapter()
        self._vector = VectorAdapter()
        self._graph = GraphAdapter()
        self._files = FileAdapter()

        shared_tools = [
            self.sql_query,
            self.sql_write,
            self.vector_list_collections,
            self.vector_search,
            self.vector_upsert,
            self.vector_delete,
            self.graph_query,
            self.graph_write,
            self.fs_list,
            self.fs_read,
            self.fs_write,
            self.fs_mkdir,
            self.fs_delete,
            self.register_artifact,
            self.register_provenance,
        ]

        super().__init__(
            max_turns=cfg.max_turns,
            tools=shared_tools,
            root_tools=[],
            sub_tools=[],
            sub_llm_max_turns=cfg.sub_llm_max_turns,
            sub_model=cfg.rlm_model,
            sub_prompt_verbosity=cfg.sub_prompt_verbosity,
            root_prompt_verbosity=cfg.root_prompt_verbosity,
            repl_language=cfg.repl_language,
            pip_install_packages=cfg.pip_install_packages,
            code_execution_timeout=cfg.code_execution_timeout,
            max_output_length=cfg.max_output_length,
            dataset=dataset,
            rubric=rubric
            or build_default_rubric(
                root_tool_names=[tool.__name__ for tool in shared_tools]
            ),
            system_prompt=config.SYSTEM_PROMPT,
            env_id=cfg.env_id,
            sub_max_completion_tokens=cfg.sub_max_completion_tokens,
            root_max_completion_tokens=cfg.root_max_completion_tokens,
            **kwargs,
        )

    async def setup_state(self, state: vf.State, **kwargs: Any) -> vf.State:
        info = state.get("info") or {}
        if not isinstance(info, dict):
            info = {}

        normalized_info = ensure_workspace(info, self._workspace_anchor)
        workspace = build_workspace_state(get_paths_from_workspace_state(normalized_info))

        staged_info = dict(normalized_info)
        staged_info["context_dir"] = workspace["workspace_dir"]
        state["info"] = staged_info
        state["workspace"] = workspace

        state = await super().setup_state(state, **kwargs)

        state["info"] = normalized_info
        state["workspace"] = workspace
        return state

    async def env_response(
        self,
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> vf.Messages:
        self._subtool_state_var.set(state)
        return await super().env_response(messages, state, **kwargs)


def create_environment(
    *,
    cfg: Config,
    dataset: Any,
    rubric: vf.Rubric | None = None,
    workspace_anchor: Path | None = None,
    **kwargs: Any,
) -> vf.Environment:
    if not isinstance(dataset, Dataset):
        anchor = workspace_anchor.resolve() if workspace_anchor else Path.cwd().resolve()
        dataset = Dataset.from_list(prepare_rows(list(dataset), anchor))
    return LongContextRetrievalEnv(
        cfg=cfg,
        dataset=dataset,
        rubric=rubric,
        workspace_anchor=workspace_anchor,
        **kwargs,
    )
