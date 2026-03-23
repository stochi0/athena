from __future__ import annotations

import contextvars
import json
import logging
from pathlib import Path
from typing import Any

import verifiers as vf
from datasets import Dataset
from dotenv import load_dotenv
from verifiers.envs.experimental.rlm_env import RLMEnv

from .adapters import FileAdapter, GraphAdapter, SQLiteAdapter, VectorAdapter
from .constants import (
    DEFAULT_REPL_LANGUAGE,
    DEFAULT_ROOT_PROMPT_VERBOSITY,
    DEFAULT_SUB_PROMPT_VERBOSITY,
    SYSTEM_PROMPT,
)
from .rewards import build_default_rubric
from .types import WorkspaceConfig
from .utils import ensure_json, json_safe
from .workspace import ensure_workspace, resolve_workspace_paths

load_dotenv()

logger = logging.getLogger(__name__)


class LongContextRetrievalEnv(RLMEnv):
    """RLM environment for autonomous QA over long context documents."""

    def __init__(
        self,
        *,
        max_turns: int = 20,
        sub_llm_max_turns: int = 4,
        sub_model: str | None = None,
        sub_prompt_verbosity: str = DEFAULT_SUB_PROMPT_VERBOSITY,
        root_prompt_verbosity: str = DEFAULT_ROOT_PROMPT_VERBOSITY,
        repl_language: str = DEFAULT_REPL_LANGUAGE,
        pip_install_packages: str = "chromadb networkx pypdf",
        code_execution_timeout: int = 300,
        max_output_length: int = 8192,
        dataset: Any = None,
        rubric: vf.Rubric | None = None,
        **kwargs: Any,
    ) -> None:
        self._subtool_state_var: contextvars.ContextVar[dict[str, Any] | None] = (
            contextvars.ContextVar("long_context_retrieval_subtool_state", default=None)
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
            max_turns=max_turns,
            tools=shared_tools,
            root_tools=[],
            sub_tools=[],
            sub_llm_max_turns=sub_llm_max_turns,
            sub_model=sub_model,
            sub_prompt_verbosity=sub_prompt_verbosity,
            root_prompt_verbosity=root_prompt_verbosity,
            repl_language=repl_language,
            pip_install_packages=pip_install_packages,
            code_execution_timeout=code_execution_timeout,
            max_output_length=max_output_length,
            dataset=dataset,
            rubric=rubric
            or build_default_rubric(
                root_tool_names=[tool.__name__ for tool in shared_tools]
            ),
            system_prompt=SYSTEM_PROMPT,
            **kwargs,
        )

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs: Any
    ) -> vf.Messages:
        token = self._subtool_state_var.set(state)
        try:
            return await super().env_response(messages, state, **kwargs)
        finally:
            self._subtool_state_var.reset(token)

    def _require_active_state(self) -> dict[str, Any]:
        root_context = self._root_tool_context_var.get()
        if root_context and root_context.get("state") is not None:
            return root_context["state"]
        state = self._subtool_state_var.get()
        if state is None:
            raise RuntimeError(
                "No active rollout state is available for tool execution."
            )
        return state

    def _workspace_paths(self, state: dict[str, Any]):
        info = state.get("info") or {}
        cache_root = info.get("workspace_cache_root")
        workspace_root = info.get("workspace_dir") or info.get("context_dir")
        if not workspace_root:
            raise RuntimeError("Workspace root is not configured.")
        if not cache_root:
            raise RuntimeError("workspace_cache_root is not configured.")
        return resolve_workspace_paths(
            WorkspaceConfig(
                workspace_root=Path(str(workspace_root)),
                cache_root=Path(str(cache_root)),
            )
        )

    def _rollout_id(self, state: dict[str, Any]) -> str:
        return str(state.get("rollout_id", "default"))

    def _record_tool_use(
        self, state: dict[str, Any], tool_name: str, args: dict[str, Any], result: Any
    ) -> None:
        signature = json.dumps(
            {"tool": tool_name, "args": json_safe(args)},
            sort_keys=True,
            default=str,
        )
        state.setdefault("root_tool_invocations", []).append(signature)
        state.setdefault("root_tool_observations", []).append(
            {"tool": tool_name, "args": json_safe(args), "result": json_safe(result)}
        )

    def _registry_connection(self, state: dict[str, Any]):
        paths = self._workspace_paths(state)
        return self._sqlite.connection(
            paths=paths,
            scope="registry",
            db_name="registry",
            rollout_id=self._rollout_id(state),
        )

    def sql_query(
        self, query: str, scope: str = "registry", db_name: str = "registry"
    ) -> str:
        """Run a SELECT/WITH query against the registry or a named cache/scratch SQLite DB."""
        state = self._require_active_state()
        result = self._sqlite.query(
            paths=self._workspace_paths(state),
            rollout_id=self._rollout_id(state),
            query=query,
            scope=scope,
            db_name=db_name,
        )
        self._record_tool_use(
            state,
            "sql_query",
            {"query": query, "scope": scope, "db_name": db_name},
            result,
        )
        return json.dumps(result)

    def sql_write(
        self, stmt: str, scope: str = "scratch", db_name: str = "main"
    ) -> str:
        """Execute a SQL write statement against a named cache/scratch SQLite DB."""
        state = self._require_active_state()
        result = self._sqlite.write(
            paths=self._workspace_paths(state),
            rollout_id=self._rollout_id(state),
            stmt=stmt,
            scope=scope,
            db_name=db_name,
        )
        self._record_tool_use(
            state,
            "sql_write",
            {"stmt": stmt, "scope": scope, "db_name": db_name},
            result,
        )
        return json.dumps(result)

    def vector_list_collections(self, scope: str = "cache") -> str:
        """List vector collections in the cache or scratch namespace."""
        state = self._require_active_state()
        result = self._vector.list_collections(
            paths=self._workspace_paths(state),
            rollout_id=self._rollout_id(state),
            scope=scope,
        )
        self._record_tool_use(
            state, "vector_list_collections", {"scope": scope}, result
        )
        return json.dumps(result)

    def vector_search(
        self,
        query: str,
        collection: str,
        n: int = 5,
        scope: str = "cache",
        where_json: str = "{}",
    ) -> str:
        """Search a named vector collection in cache or scratch scope."""
        state = self._require_active_state()
        result = self._vector.search(
            paths=self._workspace_paths(state),
            rollout_id=self._rollout_id(state),
            query=query,
            collection=collection,
            n=n,
            scope=scope,
            where_json=where_json,
        )
        self._record_tool_use(
            state,
            "vector_search",
            {
                "query": query,
                "collection": collection,
                "n": n,
                "scope": scope,
                "where_json": ensure_json(where_json, {}, "where_json"),
            },
            result,
        )
        return json.dumps(result)

    def vector_upsert(
        self,
        ids: list[str],
        docs: list[str],
        meta_json: str,
        collection: str,
        scope: str = "scratch",
    ) -> str:
        """Upsert documents into a named vector collection."""
        state = self._require_active_state()
        result = self._vector.upsert(
            paths=self._workspace_paths(state),
            rollout_id=self._rollout_id(state),
            scope=scope,
            collection=collection,
            ids=ids,
            docs=docs,
            meta_json=meta_json,
        )
        self._record_tool_use(
            state,
            "vector_upsert",
            {
                "ids": ids,
                "collection": collection,
                "scope": scope,
                "meta_json": ensure_json(meta_json, [], "meta_json"),
            },
            result,
        )
        return json.dumps(result)

    def vector_delete(
        self,
        collection: str,
        ids_json: str = "[]",
        where_json: str = "{}",
        scope: str = "scratch",
    ) -> str:
        """Delete entries from a named vector collection."""
        state = self._require_active_state()
        result = self._vector.delete(
            paths=self._workspace_paths(state),
            rollout_id=self._rollout_id(state),
            scope=scope,
            collection=collection,
            ids_json=ids_json,
            where_json=where_json,
        )
        self._record_tool_use(
            state,
            "vector_delete",
            {
                "collection": collection,
                "scope": scope,
                "ids_json": ensure_json(ids_json, [], "ids_json"),
                "where_json": ensure_json(where_json, {}, "where_json"),
            },
            result,
        )
        return json.dumps(result)

    def graph_query(
        self,
        op: str,
        params_json: str = "{}",
        graph_name: str = "main",
        scope: str = "scratch",
    ) -> str:
        """Query a named graph in cache or scratch scope."""
        state = self._require_active_state()
        result = self._graph.query(
            paths=self._workspace_paths(state),
            scope=scope,
            graph_name=graph_name,
            rollout_id=self._rollout_id(state),
            op=op,
            params_json=params_json,
        )
        self._record_tool_use(
            state,
            "graph_query",
            {
                "op": op,
                "params_json": ensure_json(params_json, {}, "params_json"),
                "graph_name": graph_name,
                "scope": scope,
            },
            result,
        )
        return json.dumps(result)

    def graph_write(
        self,
        nodes_json: str = "[]",
        edges_json: str = "[]",
        graph_name: str = "main",
        scope: str = "scratch",
    ) -> str:
        """Write nodes and edges to a named graph in cache or scratch scope."""
        state = self._require_active_state()
        result = self._graph.write(
            paths=self._workspace_paths(state),
            scope=scope,
            graph_name=graph_name,
            rollout_id=self._rollout_id(state),
            nodes_json=nodes_json,
            edges_json=edges_json,
        )
        self._record_tool_use(
            state,
            "graph_write",
            {
                "nodes_json": ensure_json(nodes_json, [], "nodes_json"),
                "edges_json": ensure_json(edges_json, [], "edges_json"),
                "graph_name": graph_name,
                "scope": scope,
            },
            result,
        )
        return json.dumps(result)

    def fs_list(self, path: str = ".", scope: str = "workspace") -> str:
        """List files under a workspace, cache, or scratch filesystem scope."""
        state = self._require_active_state()
        result = self._files.list(
            paths=self._workspace_paths(state),
            scope=scope,
            rollout_id=self._rollout_id(state),
            rel_path=path,
        )
        self._record_tool_use(state, "fs_list", {"path": path, "scope": scope}, result)
        return json.dumps(result)

    def fs_read(
        self, path: str, scope: str = "workspace", encoding: str = "utf-8"
    ) -> str:
        """Read a text file from workspace, cache, or scratch scope."""
        state = self._require_active_state()
        result = self._files.read(
            paths=self._workspace_paths(state),
            scope=scope,
            rollout_id=self._rollout_id(state),
            rel_path=path,
            encoding=encoding,
        )
        self._record_tool_use(
            state,
            "fs_read",
            {"path": path, "scope": scope, "encoding": encoding},
            {"content_length": len(result)},
        )
        return result

    def fs_write(
        self,
        path: str,
        content: str,
        scope: str = "scratch",
        overwrite: bool = False,
        encoding: str = "utf-8",
    ) -> str:
        """Write a text file into cache or scratch scope."""
        state = self._require_active_state()
        result = self._files.write(
            paths=self._workspace_paths(state),
            scope=scope,
            rollout_id=self._rollout_id(state),
            rel_path=path,
            content=content,
            overwrite=overwrite,
            encoding=encoding,
        )
        self._record_tool_use(
            state,
            "fs_write",
            {"path": path, "scope": scope, "overwrite": overwrite},
            result,
        )
        return json.dumps(result)

    def fs_mkdir(self, path: str, scope: str = "scratch") -> str:
        """Create a directory inside cache or scratch scope."""
        state = self._require_active_state()
        result = self._files.mkdir(
            paths=self._workspace_paths(state),
            scope=scope,
            rollout_id=self._rollout_id(state),
            rel_path=path,
        )
        self._record_tool_use(state, "fs_mkdir", {"path": path, "scope": scope}, result)
        return json.dumps(result)

    def fs_delete(
        self, path: str, scope: str = "scratch", recursive: bool = False
    ) -> str:
        """Delete a file or directory inside cache or scratch scope."""
        state = self._require_active_state()
        result = self._files.delete(
            paths=self._workspace_paths(state),
            scope=scope,
            rollout_id=self._rollout_id(state),
            rel_path=path,
            recursive=recursive,
        )
        self._record_tool_use(
            state,
            "fs_delete",
            {"path": path, "scope": scope, "recursive": recursive},
            result,
        )
        return json.dumps(result)

    def register_artifact(
        self,
        artifact_id: str,
        kind: str,
        scope: str,
        path: str = "",
        source_document_id: str = "",
        metadata_json: str = "{}",
    ) -> str:
        """Register a derived artifact in the system registry."""
        state = self._require_active_state()
        metadata = ensure_json(metadata_json, {}, "metadata_json")
        if not isinstance(metadata, dict):
            raise ValueError("metadata_json must decode to an object.")
        conn = self._registry_connection(state)
        conn.execute(
            """
            INSERT OR REPLACE INTO artifacts (
                artifact_id, kind, scope, path, source_document_id, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_id,
                kind,
                scope,
                path or None,
                source_document_id or None,
                json.dumps(metadata, sort_keys=True),
            ),
        )
        conn.commit()
        result = {"ok": True, "artifact_id": artifact_id}
        self._record_tool_use(
            state,
            "register_artifact",
            {
                "artifact_id": artifact_id,
                "kind": kind,
                "scope": scope,
                "path": path,
                "source_document_id": source_document_id,
                "metadata_json": metadata,
            },
            result,
        )
        return json.dumps(result)

    def register_provenance(
        self,
        artifact_id: str,
        document_id: str,
        excerpt: str = "",
        page_number: int | None = None,
        section: str = "",
        metadata_json: str = "{}",
    ) -> str:
        """Register provenance linking an artifact back to a source document."""
        state = self._require_active_state()
        metadata = ensure_json(metadata_json, {}, "metadata_json")
        if not isinstance(metadata, dict):
            raise ValueError("metadata_json must decode to an object.")
        conn = self._registry_connection(state)
        conn.execute(
            """
            INSERT INTO artifact_provenance (
                artifact_id, document_id, page_number, excerpt, section, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_id,
                document_id,
                page_number,
                excerpt,
                section or None,
                json.dumps(metadata, sort_keys=True),
            ),
        )
        conn.commit()
        result = {"ok": True, "artifact_id": artifact_id, "document_id": document_id}
        self._record_tool_use(
            state,
            "register_provenance",
            {
                "artifact_id": artifact_id,
                "document_id": document_id,
                "page_number": page_number,
                "excerpt": excerpt,
                "section": section,
                "metadata_json": metadata,
            },
            result,
        )
        return json.dumps(result)


def create_environment(
    *,
    dataset: Any,
    rubric: vf.Rubric | None = None,
    **kwargs: Any,
) -> LongContextRetrievalEnv:
    if not isinstance(dataset, Dataset):
        dataset = Dataset.from_list(_normalize_rows(list(dataset), Path.cwd()))
    return LongContextRetrievalEnv(dataset=dataset, rubric=rubric, **kwargs)


def _normalize_rows(rows: list[dict[str, Any]], anchor: Path) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized_row = dict(row)
        normalized_row["info"] = ensure_workspace(
            dict(normalized_row.get("info") or {}), anchor
        )
        normalized_rows.append(normalized_row)
    return normalized_rows


def _load_rows_from_path(dataset_path: Path, anchor: Path) -> list[dict[str, Any]]:
    if dataset_path.is_file() and dataset_path.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in dataset_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return _normalize_rows(rows, anchor)
    if dataset_path.is_dir() and (dataset_path / "dataset_info.json").is_file():
        dataset = Dataset.load_from_disk(str(dataset_path))
        return _normalize_rows(dataset.to_list(), anchor)
    raise FileNotFoundError(
        f"dataset_path must be a .jsonl file or a HuggingFace dataset directory: {dataset_path}"
    )
