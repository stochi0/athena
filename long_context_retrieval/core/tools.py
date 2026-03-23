from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .utils import ensure_json, json_safe
from .workspace import get_paths_from_workspace_state


class WorkspaceTools:
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

    def _paths(self, state: dict[str, Any]):
        workspace = state.get("workspace")
        if not isinstance(workspace, dict):
            raise RuntimeError("Workspace state is not initialized.")
        return get_paths_from_workspace_state(workspace)

    def _rollout_id(self, state: dict[str, Any]) -> str:
        return str(state.get("rollout_id", "default"))

    def _record_tool_use(
        self,
        state: dict[str, Any],
        tool_name: str,
        args: dict[str, Any],
        result: Any,
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

    def _registry(self, state: dict[str, Any]):
        paths = self._paths(state)
        return self._sqlite.connection(
            paths=paths,
            scope="registry",
            db_name="registry",
            rollout_id=self._rollout_id(state),
        )

    def _run_json_tool(
        self,
        *,
        tool_name: str,
        args: dict[str, Any],
        callback: Callable[[dict[str, Any]], Any],
    ) -> str:
        state = self._require_active_state()
        result = callback(state)
        self._record_tool_use(state, tool_name, args, result)
        return json.dumps(result)

    def sql_query(
        self,
        query: str,
        scope: str = "registry",
        db_name: str = "registry",
    ) -> str:
        """Run a SELECT/WITH query against the registry or a named state/scratch SQLite DB."""
        return self._run_json_tool(
            tool_name="sql_query",
            args={"query": query, "scope": scope, "db_name": db_name},
            callback=lambda state: self._sqlite.query(
                paths=self._paths(state),
                rollout_id=self._rollout_id(state),
                query=query,
                scope=scope,
                db_name=db_name,
            ),
        )

    def sql_write(
        self,
        stmt: str,
        scope: str = "scratch",
        db_name: str = "main",
    ) -> str:
        """Execute a SQL write statement against a named state/scratch SQLite DB."""
        return self._run_json_tool(
            tool_name="sql_write",
            args={"stmt": stmt, "scope": scope, "db_name": db_name},
            callback=lambda state: self._sqlite.write(
                paths=self._paths(state),
                rollout_id=self._rollout_id(state),
                stmt=stmt,
                scope=scope,
                db_name=db_name,
            ),
        )

    def vector_list_collections(self, scope: str = "scratch") -> str:
        """List vector collections in the state or scratch namespace."""
        return self._run_json_tool(
            tool_name="vector_list_collections",
            args={"scope": scope},
            callback=lambda state: self._vector.list_collections(
                paths=self._paths(state),
                rollout_id=self._rollout_id(state),
                scope=scope,
            ),
        )

    def vector_search(
        self,
        query: str,
        collection: str,
        n: int = 5,
        scope: str = "scratch",
        where_json: str = "{}",
    ) -> str:
        """Search a named vector collection in state or scratch scope."""
        return self._run_json_tool(
            tool_name="vector_search",
            args={
                "query": query,
                "collection": collection,
                "n": n,
                "scope": scope,
                "where_json": ensure_json(where_json, {}, "where_json"),
            },
            callback=lambda state: self._vector.search(
                paths=self._paths(state),
                rollout_id=self._rollout_id(state),
                query=query,
                collection=collection,
                n=n,
                scope=scope,
                where_json=where_json,
            ),
        )

    def vector_upsert(
        self,
        ids: list[str],
        docs: list[str],
        meta_json: str,
        collection: str,
        scope: str = "scratch",
    ) -> str:
        """Upsert documents into a named vector collection."""
        return self._run_json_tool(
            tool_name="vector_upsert",
            args={
                "ids": ids,
                "collection": collection,
                "scope": scope,
                "meta_json": ensure_json(meta_json, [], "meta_json"),
            },
            callback=lambda state: self._vector.upsert(
                paths=self._paths(state),
                rollout_id=self._rollout_id(state),
                scope=scope,
                collection=collection,
                ids=ids,
                docs=docs,
                meta_json=meta_json,
            ),
        )

    def vector_delete(
        self,
        collection: str,
        ids_json: str = "[]",
        where_json: str = "{}",
        scope: str = "scratch",
    ) -> str:
        """Delete entries from a named vector collection."""
        return self._run_json_tool(
            tool_name="vector_delete",
            args={
                "collection": collection,
                "scope": scope,
                "ids_json": ensure_json(ids_json, [], "ids_json"),
                "where_json": ensure_json(where_json, {}, "where_json"),
            },
            callback=lambda state: self._vector.delete(
                paths=self._paths(state),
                rollout_id=self._rollout_id(state),
                scope=scope,
                collection=collection,
                ids_json=ids_json,
                where_json=where_json,
            ),
        )

    def graph_query(
        self,
        op: str,
        params_json: str = "{}",
        graph_name: str = "main",
        scope: str = "scratch",
    ) -> str:
        """Query a named graph in state or scratch scope."""
        return self._run_json_tool(
            tool_name="graph_query",
            args={
                "op": op,
                "params_json": ensure_json(params_json, {}, "params_json"),
                "graph_name": graph_name,
                "scope": scope,
            },
            callback=lambda state: self._graph.query(
                paths=self._paths(state),
                scope=scope,
                graph_name=graph_name,
                rollout_id=self._rollout_id(state),
                op=op,
                params_json=params_json,
            ),
        )

    def graph_write(
        self,
        nodes_json: str = "[]",
        edges_json: str = "[]",
        graph_name: str = "main",
        scope: str = "scratch",
    ) -> str:
        """Write nodes and edges to a named graph in state or scratch scope."""
        return self._run_json_tool(
            tool_name="graph_write",
            args={
                "nodes_json": ensure_json(nodes_json, [], "nodes_json"),
                "edges_json": ensure_json(edges_json, [], "edges_json"),
                "graph_name": graph_name,
                "scope": scope,
            },
            callback=lambda state: self._graph.write(
                paths=self._paths(state),
                scope=scope,
                graph_name=graph_name,
                rollout_id=self._rollout_id(state),
                nodes_json=nodes_json,
                edges_json=edges_json,
            ),
        )

    def fs_list(self, path: str = ".", scope: str = "workspace") -> str:
        """List files under a workspace, state, or scratch filesystem scope."""
        return self._run_json_tool(
            tool_name="fs_list",
            args={"path": path, "scope": scope},
            callback=lambda state: self._files.list(
                paths=self._paths(state),
                scope=scope,
                rollout_id=self._rollout_id(state),
                rel_path=path,
            ),
        )

    def fs_read(
        self,
        path: str,
        scope: str = "workspace",
        encoding: str = "utf-8",
    ) -> str:
        """Read a text file from workspace, state, or scratch scope."""
        state = self._require_active_state()
        result = self._files.read(
            paths=self._paths(state),
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
        """Write a text file into state or scratch scope."""
        return self._run_json_tool(
            tool_name="fs_write",
            args={"path": path, "scope": scope, "overwrite": overwrite},
            callback=lambda state: self._files.write(
                paths=self._paths(state),
                scope=scope,
                rollout_id=self._rollout_id(state),
                rel_path=path,
                content=content,
                overwrite=overwrite,
                encoding=encoding,
            ),
        )

    def fs_mkdir(self, path: str, scope: str = "scratch") -> str:
        """Create a directory inside state or scratch scope."""
        return self._run_json_tool(
            tool_name="fs_mkdir",
            args={"path": path, "scope": scope},
            callback=lambda state: self._files.mkdir(
                paths=self._paths(state),
                scope=scope,
                rollout_id=self._rollout_id(state),
                rel_path=path,
            ),
        )

    def fs_delete(
        self,
        path: str,
        scope: str = "scratch",
        recursive: bool = False,
    ) -> str:
        """Delete a file or directory inside state or scratch scope."""
        return self._run_json_tool(
            tool_name="fs_delete",
            args={"path": path, "scope": scope, "recursive": recursive},
            callback=lambda state: self._files.delete(
                paths=self._paths(state),
                scope=scope,
                rollout_id=self._rollout_id(state),
                rel_path=path,
                recursive=recursive,
            ),
        )

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

        conn = self._registry(state)
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

        conn = self._registry(state)
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
