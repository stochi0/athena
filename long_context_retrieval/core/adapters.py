from __future__ import annotations

import copy
import itertools
import pickle
import sqlite3
import types
from pathlib import Path
from typing import Any

import chromadb
import networkx as nx

from .types import WorkspacePaths
from .utils import ensure_json


class SQLiteAdapter:
    def __init__(self) -> None:
        self._connections: dict[tuple[str, str], sqlite3.Connection] = {}

    def _scope_root(self, *, paths: WorkspacePaths, scope: str, rollout_id: str) -> Path:
        if scope == "registry":
            return paths.state_root
        if scope == "state":
            root = paths.sql_root
        elif scope == "scratch":
            root = paths.scratch_root / rollout_id / "sql"
        else:
            raise ValueError("scope must be one of: registry, state, scratch.")
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _db_path(self, *, paths: WorkspacePaths, scope: str, db_name: str, rollout_id: str) -> Path:
        if scope == "registry":
            if db_name not in {"registry", "main"}:
                raise ValueError("The registry scope only exposes the registry database.")
            return paths.registry_db
        return self._scope_root(paths=paths, scope=scope, rollout_id=rollout_id) / f"{db_name}.db"

    def connection(
        self, *, paths: WorkspacePaths, scope: str, db_name: str, rollout_id: str
    ) -> sqlite3.Connection:
        db_path = self._db_path(
            paths=paths,
            scope=scope,
            db_name=db_name,
            rollout_id=rollout_id,
        )
        key = (scope, str(db_path))
        conn = self._connections.get(key)
        if conn is None:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            self._connections[key] = conn
        return conn

    def execute_sql(
        self,
        *,
        paths: WorkspacePaths,
        rollout_id: str,
        statement: str,
        scope: str,
        db_name: str,
    ) -> dict[str, Any]:
        """Run one SQLite statement (DDL/DML/DQL). Commits when there is no result row set.

        Python's sqlite3 allows only one statement per `execute`; chain multiple calls if needed.
        """
        conn = self.connection(paths=paths, scope=scope, db_name=db_name, rollout_id=rollout_id)
        cursor = conn.execute(statement)
        if cursor.description is not None:
            rows = [dict(row) for row in cursor.fetchall()]
            rc = cursor.rowcount
            return {
                "rows": rows,
                "rowcount": rc if rc is not None and rc >= 0 else len(rows),
            }
        conn.commit()
        rc = cursor.rowcount
        return {
            "ok": True,
            "rows": None,
            "rowcount": rc if rc is not None and rc >= 0 else 0,
        }


class VectorAdapter:
    def __init__(self) -> None:
        self._clients: dict[str, chromadb.PersistentClient] = {}

    def _scope_root(self, *, paths: WorkspacePaths, scope: str, rollout_id: str) -> Path:
        if scope == "state":
            root = paths.vector_root
        elif scope == "scratch":
            root = paths.scratch_root / rollout_id / "vector"
        else:
            raise ValueError("Vector scope must be state or scratch.")
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _client(
        self, *, paths: WorkspacePaths, scope: str, rollout_id: str
    ) -> chromadb.PersistentClient:
        root = self._scope_root(paths=paths, scope=scope, rollout_id=rollout_id)
        key = str(root)
        client = self._clients.get(key)
        if client is None:
            client = chromadb.PersistentClient(path=str(root))
            self._clients[key] = client
        return client

    def list_collections(
        self, *, paths: WorkspacePaths, rollout_id: str, scope: str
    ) -> list[dict[str, Any]]:
        client = self._client(paths=paths, scope=scope, rollout_id=rollout_id)
        return [{"name": item.name} for item in client.list_collections()]

    def search(
        self,
        *,
        paths: WorkspacePaths,
        rollout_id: str,
        query: str,
        collection: str,
        n: int,
        scope: str,
        where_json: str,
    ) -> list[dict[str, Any]]:
        client = self._client(paths=paths, scope=scope, rollout_id=rollout_id)
        where = ensure_json(where_json, {}, "where_json")
        if not isinstance(where, dict):
            raise ValueError("where_json must decode to an object.")
        target = client.get_collection(collection)
        result = target.query(
            query_texts=[query],
            n_results=max(1, n),
            where=where or None,
        )
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        hits: list[dict[str, Any]] = []
        for index, doc_id in enumerate(ids):
            hits.append(
                {
                    "id": doc_id,
                    "document": docs[index] if index < len(docs) else None,
                    "metadata": metadatas[index] if index < len(metadatas) else None,
                    "distance": distances[index] if index < len(distances) else None,
                    "collection": collection,
                    "scope": scope,
                }
            )
        return hits

    def upsert(
        self,
        *,
        paths: WorkspacePaths,
        rollout_id: str,
        scope: str,
        collection: str,
        ids: list[str],
        docs: list[str],
        meta_json: str,
    ) -> dict[str, Any]:
        metadata = ensure_json(meta_json, [], "meta_json")
        if not isinstance(metadata, list):
            raise ValueError("meta_json must decode to a list.")
        if not (len(ids) == len(docs) == len(metadata)):
            raise ValueError("ids, docs, and metadata must have the same length.")
        client = self._client(paths=paths, scope=scope, rollout_id=rollout_id)
        client.get_or_create_collection(collection).upsert(
            ids=ids,
            documents=docs,
            metadatas=metadata,
        )
        return {"ok": True, "count": len(ids), "collection": collection, "scope": scope}

    def delete(
        self,
        *,
        paths: WorkspacePaths,
        rollout_id: str,
        scope: str,
        collection: str,
        ids_json: str,
        where_json: str,
    ) -> dict[str, Any]:
        ids = ensure_json(ids_json, [], "ids_json")
        where = ensure_json(where_json, {}, "where_json")
        client = self._client(paths=paths, scope=scope, rollout_id=rollout_id)
        target = client.get_collection(collection)
        target.delete(ids=ids or None, where=where or None)
        return {"ok": True, "collection": collection, "scope": scope}

    def get(
        self,
        *,
        paths: WorkspacePaths,
        rollout_id: str,
        scope: str,
        collection: str,
        ids_json: str,
        where_json: str,
        limit: int | None,
        offset: int | None,
        include_embeddings: bool,
    ) -> dict[str, Any]:
        ids = ensure_json(ids_json, [], "ids_json")
        where = ensure_json(where_json, {}, "where_json")
        if not isinstance(ids, list):
            raise ValueError("ids_json must decode to a list.")
        if not isinstance(where, dict):
            raise ValueError("where_json must decode to an object.")
        client = self._client(paths=paths, scope=scope, rollout_id=rollout_id)
        target = client.get_collection(collection)
        kwargs: dict[str, Any] = {"include": ["metadatas", "documents"]}
        if include_embeddings:
            kwargs["include"].append("embeddings")
        if ids:
            kwargs["ids"] = ids
        if where:
            kwargs["where"] = where
        if limit is not None:
            kwargs["limit"] = int(limit)
        if offset is not None:
            kwargs["offset"] = int(offset)
        return dict(target.get(**kwargs))


# NetworkX callables exposed via graph_query(op="algo"). Extends fixed op set without arbitrary code exec.
_NX_ALGO_ALLOWLIST: frozenset[str] = frozenset(
    {
        "pagerank",
        "google_matrix",
        "hits",
        "betweenness_centrality",
        "edge_betweenness_centrality",
        "closeness_centrality",
        "degree_centrality",
        "in_degree_centrality",
        "out_degree_centrality",
        "eigenvector_centrality",
        "katz_centrality",
        "clustering",
        "triangles",
        "average_clustering",
        "square_clustering",
        "transitivity",
        "number_connected_components",
        "is_connected",
        "node_connected_component",
        "connected_components",
        "average_shortest_path_length",
        "shortest_path",
        "shortest_path_length",
        "all_pairs_shortest_path_length",
        "all_pairs_shortest_path",
        "single_source_shortest_path",
        "single_source_shortest_path_length",
        "has_path",
        "density",
        "diameter",
        "radius",
        "center",
        "periphery",
        "degree_histogram",
        "is_tree",
        "is_forest",
        "is_directed_acyclic_graph",
        "topological_sort",
        "topological_generations",
        "ancestors",
        "descendants",
    }
)


def _jsonify_nx_result(value: Any, *, depth: int = 0, gen_limit: int = 2048) -> Any:
    if depth > 24:
        return "<max-depth>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, nx.Graph):
        return serialize_graph(value)
    if isinstance(value, dict):
        return {
            str(k): _jsonify_nx_result(v, depth=depth + 1, gen_limit=gen_limit)
            for k, v in value.items()
        }
    if isinstance(value, tuple):
        return [_jsonify_nx_result(x, depth=depth + 1, gen_limit=gen_limit) for x in value]
    if isinstance(value, (list, set)):
        seq = list(value)
        if len(seq) > gen_limit:
            return [
                _jsonify_nx_result(x, depth=depth + 1, gen_limit=gen_limit)
                for x in seq[:gen_limit]
            ] + [f"<truncated: {len(seq) - gen_limit} more items>"]
        return [_jsonify_nx_result(x, depth=depth + 1, gen_limit=gen_limit) for x in seq]
    if isinstance(value, types.GeneratorType):
        return [
            _jsonify_nx_result(x, depth=depth + 1, gen_limit=gen_limit)
            for x in itertools.islice(value, gen_limit)
        ]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "__iter__") and not isinstance(
        value, (str, bytes, dict, list, tuple, set, nx.Graph)
    ):
        try:
            return [
                _jsonify_nx_result(x, depth=depth + 1, gen_limit=gen_limit)
                for x in itertools.islice(iter(value), gen_limit)
            ]
        except TypeError:
            pass
    return str(value)


class GraphAdapter:
    def _graph_path(
        self, *, paths: WorkspacePaths, scope: str, graph_name: str, rollout_id: str
    ) -> Path:
        if scope == "state":
            root = paths.graph_root
        elif scope == "scratch":
            root = paths.scratch_root / rollout_id / "graphs"
        else:
            raise ValueError("Graph scope must be state or scratch.")
        root.mkdir(parents=True, exist_ok=True)
        return root / f"{graph_name}.pkl"

    def _load_graph(
        self, *, paths: WorkspacePaths, scope: str, graph_name: str, rollout_id: str
    ) -> nx.Graph:
        graph_path = self._graph_path(
            paths=paths, scope=scope, graph_name=graph_name, rollout_id=rollout_id
        )
        if not graph_path.exists():
            return nx.Graph()
        with graph_path.open("rb") as fh:
            graph = pickle.load(fh)
        if not isinstance(graph, nx.Graph):
            raise TypeError(f"Expected a NetworkX graph in {graph_path}")
        return graph

    def _save_graph(
        self,
        *,
        paths: WorkspacePaths,
        scope: str,
        graph_name: str,
        rollout_id: str,
        graph: nx.Graph,
    ) -> None:
        graph_path = self._graph_path(
            paths=paths, scope=scope, graph_name=graph_name, rollout_id=rollout_id
        )
        with graph_path.open("wb") as fh:
            pickle.dump(graph, fh)

    def query(
        self,
        *,
        paths: WorkspacePaths,
        scope: str,
        graph_name: str,
        rollout_id: str,
        op: str,
        params_json: str,
    ) -> Any:
        params = ensure_json(params_json, {}, "params_json")
        if not isinstance(params, dict):
            raise ValueError("params_json must decode to an object.")
        graph = self._load_graph(
            paths=paths, scope=scope, graph_name=graph_name, rollout_id=rollout_id
        )
        edge_types = params.get("edge_types") or params.get("edge_type")
        if edge_types:
            graph = filter_graph_by_edge_types(graph, edge_types)

        if op == "neighbors":
            node = params["node"]
            if node not in graph:
                return []
            return [{"id": neighbor, **graph.nodes[neighbor]} for neighbor in graph.neighbors(node)]
        if op == "shortest_path":
            return nx.shortest_path(graph, source=params["source"], target=params["target"])
        if op == "subgraph":
            return serialize_graph(graph.subgraph(params.get("nodes", [])).copy())
        if op == "bfs":
            tree = nx.bfs_tree(graph, params["source"], depth_limit=int(params.get("depth", 1)))
            return serialize_graph(graph.subgraph(tree.nodes()).copy())
        if op == "dump":
            return serialize_graph(graph)
        if op == "algo":
            name = str(params.get("name") or params.get("algorithm", "")).strip()
            if name not in _NX_ALGO_ALLOWLIST:
                sample = ", ".join(sorted(_NX_ALGO_ALLOWLIST)[:20])
                raise ValueError(
                    f"graph_query(op='algo') unknown algorithm {name!r}. "
                    f"Use a name from networkx (examples: {sample}, ...)."
                )
            fn = getattr(nx, name)
            if not callable(fn):
                raise ValueError(f"networkx.{name} is not callable")
            extra_args = list(params.get("args", []))
            extra_kw = {str(k): v for k, v in dict(params.get("kwargs", {})).items()}
            try:
                raw = fn(graph, *extra_args, **extra_kw)
            except Exception as exc:
                raise ValueError(f"networkx.{name}({extra_args!r}, {extra_kw!r}) failed: {exc}") from exc
            return _jsonify_nx_result(raw)
        raise ValueError(
            "graph_query op must be one of: neighbors, shortest_path, subgraph, bfs, dump, algo."
        )

    def write(
        self,
        *,
        paths: WorkspacePaths,
        scope: str,
        graph_name: str,
        rollout_id: str,
        nodes_json: str,
        edges_json: str,
    ) -> dict[str, Any]:
        nodes = ensure_json(nodes_json, [], "nodes_json")
        edges = ensure_json(edges_json, [], "edges_json")
        if not isinstance(nodes, list) or not isinstance(edges, list):
            raise ValueError("nodes_json and edges_json must decode to lists.")
        graph = self._load_graph(
            paths=paths, scope=scope, graph_name=graph_name, rollout_id=rollout_id
        )
        for node in nodes:
            node_id = node.get("id")
            if node_id is None:
                raise ValueError("Each node must include an 'id' field.")
            attrs = {key: value for key, value in node.items() if key != "id"}
            graph.add_node(node_id, **attrs)
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            if source is None or target is None:
                raise ValueError("Each edge must include 'source' and 'target'.")
            attrs = {key: value for key, value in edge.items() if key not in {"source", "target"}}
            graph.add_edge(source, target, **attrs)
        self._save_graph(
            paths=paths,
            scope=scope,
            graph_name=graph_name,
            rollout_id=rollout_id,
            graph=graph,
        )
        return {"ok": True, "nodes_added": len(nodes), "edges_added": len(edges)}


class FileAdapter:
    def scope_root(self, *, paths: WorkspacePaths, scope: str, rollout_id: str) -> Path:
        if scope == "workspace":
            return paths.workspace_root
        if scope == "state":
            return paths.state_root
        if scope == "scratch":
            root = paths.scratch_root / rollout_id / "files"
            root.mkdir(parents=True, exist_ok=True)
            return root
        raise ValueError("scope must be one of: workspace, state, scratch.")

    def resolve_path(
        self, *, paths: WorkspacePaths, scope: str, rollout_id: str, rel_path: str
    ) -> Path:
        root = self.scope_root(paths=paths, scope=scope, rollout_id=rollout_id).resolve()
        target = (root / rel_path).resolve()
        if root not in target.parents and target != root:
            raise ValueError("Requested path escapes the allowed scope root.")
        return target

    def list(
        self, *, paths: WorkspacePaths, scope: str, rollout_id: str, rel_path: str
    ) -> list[str]:
        root = self.resolve_path(paths=paths, scope=scope, rollout_id=rollout_id, rel_path=rel_path)
        if root.is_file():
            return [root.name]
        if not root.exists():
            return []
        return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())

    def read(
        self,
        *,
        paths: WorkspacePaths,
        scope: str,
        rollout_id: str,
        rel_path: str,
        encoding: str,
    ) -> str:
        path = self.resolve_path(paths=paths, scope=scope, rollout_id=rollout_id, rel_path=rel_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {rel_path}")
        return path.read_text(encoding=encoding, errors="replace")

    def write(
        self,
        *,
        paths: WorkspacePaths,
        scope: str,
        rollout_id: str,
        rel_path: str,
        content: str,
        overwrite: bool,
        encoding: str,
    ) -> dict[str, Any]:
        path = self.resolve_path(paths=paths, scope=scope, rollout_id=rollout_id, rel_path=rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {rel_path}")
        path.write_text(content, encoding=encoding)
        return {"ok": True, "path": str(path)}

    def mkdir(
        self, *, paths: WorkspacePaths, scope: str, rollout_id: str, rel_path: str
    ) -> dict[str, Any]:
        path = self.resolve_path(paths=paths, scope=scope, rollout_id=rollout_id, rel_path=rel_path)
        path.mkdir(parents=True, exist_ok=True)
        return {"ok": True, "path": str(path)}

    def delete(
        self,
        *,
        paths: WorkspacePaths,
        scope: str,
        rollout_id: str,
        rel_path: str,
        recursive: bool,
    ) -> dict[str, Any]:
        path = self.resolve_path(paths=paths, scope=scope, rollout_id=rollout_id, rel_path=rel_path)
        if not path.exists():
            return {"ok": True, "deleted": False}
        if path.is_dir():
            if not recursive:
                raise ValueError("Use recursive=True to delete directories.")
            for child in sorted(path.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            path.rmdir()
        else:
            path.unlink()
        return {"ok": True, "deleted": True}


def serialize_graph(graph: nx.Graph) -> dict[str, Any]:
    nodes = [{"id": node_id, **attrs} for node_id, attrs in graph.nodes(data=True)]
    edges = []
    for source, target, attrs in graph.edges(data=True):
        row = {"source": source, "target": target}
        row.update(attrs)
        edges.append(row)
    return {"nodes": nodes, "edges": edges}


def filter_graph_by_edge_types(graph: nx.Graph, edge_types: str | list[str]) -> nx.Graph:
    if isinstance(edge_types, str):
        wanted = {edge_types}
    else:
        wanted = set(edge_types)
    filtered = graph.__class__()
    filtered.add_nodes_from(graph.nodes(data=True))
    for source, target, attrs in graph.edges(data=True):
        if attrs.get("type") in wanted:
            filtered.add_edge(source, target, **copy.deepcopy(attrs))
    return filtered
