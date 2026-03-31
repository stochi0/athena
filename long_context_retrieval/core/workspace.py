from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from . import config
from .types import WorkspaceConfig, WorkspacePaths, WorkspaceState
from .utils import stable_document_id


def get_paths(workspace_config: WorkspaceConfig) -> WorkspacePaths:
    state_root = workspace_config.state_root.resolve()
    workspace_root = workspace_config.workspace_root.resolve()
    return WorkspacePaths(
        workspace_root=workspace_root,
        state_root=state_root,
        registry_db=state_root / config.REGISTRY_DB,
        vector_root=state_root / config.VECTOR_DIRNAME,
        graph_root=state_root / config.GRAPH_DIRNAME,
        sql_root=state_root / config.SQL_DIRNAME,
        artifacts_root=state_root / config.ARTIFACTS_DIRNAME,
        scratch_root=state_root / config.SCRATCH_DIRNAME,
    )


def build_workspace_state(paths: WorkspacePaths) -> WorkspaceState:
    document_count = sum(1 for _ in paths.workspace_root.rglob("*.pdf"))
    return {
        "workspace_dir": str(paths.workspace_root),
        "workspace_state_root": str(paths.state_root),
        "registry_db": str(paths.registry_db),
        "vector_root": str(paths.vector_root),
        "graph_root": str(paths.graph_root),
        "sql_root": str(paths.sql_root),
        "artifacts_root": str(paths.artifacts_root),
        "scratch_root": str(paths.scratch_root),
        "document_count": document_count,
    }


def get_paths_from_workspace_state(workspace: WorkspaceState | dict[str, Any]) -> WorkspacePaths:
    workspace_dir = workspace.get("workspace_dir")
    state_root = workspace.get("workspace_state_root")
    if not workspace_dir:
        raise RuntimeError("Workspace directory is not configured.")
    if not state_root:
        raise RuntimeError("Workspace state root is not configured.")
    return get_paths(
        WorkspaceConfig(
            workspace_root=Path(str(workspace_dir)),
            state_root=Path(str(state_root)),
        )
    )


def _default_state_root(workspace_root: Path) -> Path:
    return workspace_root / config.WORKSPACE_STATE_DIRNAME


def build_workspace(
    *,
    pdf_paths: list[str],
    workspace_root: str | None = None,
    state_root: str | None = None,
) -> WorkspaceConfig:
    src_paths = [Path(path).expanduser().resolve() for path in pdf_paths]
    if not src_paths:
        raise ValueError("pdf_paths must contain at least one PDF.")
    if workspace_root:
        root = Path(workspace_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
    else:
        root = src_paths[0].parent
    pdf_dir = root / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    for src in src_paths:
        if not src.is_file():
            raise FileNotFoundError(f"PDF not found: {src}")
        dest = pdf_dir / src.name
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
    resolved_state_root = (
        Path(state_root).expanduser().resolve() if state_root else _default_state_root(root)
    )
    return WorkspaceConfig(workspace_root=root, state_root=resolved_state_root)


def ensure_workspace(info: dict[str, Any], anchor: Path) -> dict[str, Any]:
    normalized = dict(info)
    workspace_dir = normalized.get("workspace_dir")
    pdf_dir = normalized.get("pdf_dir")
    pdf_paths = normalized.get("pdf_paths")

    if workspace_dir:
        workspace_root = Path(str(workspace_dir)).expanduser()
        if not workspace_root.is_absolute():
            workspace_root = (anchor / workspace_root).resolve()
        else:
            workspace_root = workspace_root.resolve()
        if not workspace_root.exists():
            raise FileNotFoundError(f"Workspace directory not found: {workspace_root}")
        state_root_value = normalized.get("workspace_state_root")
        state_root = (
            Path(str(state_root_value)).expanduser().resolve()
            if state_root_value
            else _default_state_root(workspace_root)
        )
        config = WorkspaceConfig(workspace_root=workspace_root, state_root=state_root)
    elif pdf_dir:
        pdf_root = Path(str(pdf_dir)).expanduser()
        if not pdf_root.is_absolute():
            pdf_root = (anchor / pdf_root).resolve()
        else:
            pdf_root = pdf_root.resolve()
        paths = sorted(str(path) for path in pdf_root.rglob("*.pdf"))
        config = build_workspace(
            pdf_paths=paths,
            workspace_root=str(normalized.get("workspace_root") or pdf_root),
            state_root=normalized.get("workspace_state_root"),
        )
    elif pdf_paths:
        resolved_pdf_paths: list[str] = []
        for raw in pdf_paths:
            path = Path(str(raw)).expanduser()
            if not path.is_absolute():
                path = (anchor / path).resolve()
            else:
                path = path.resolve()
            resolved_pdf_paths.append(str(path))
        config = build_workspace(
            pdf_paths=resolved_pdf_paths,
            workspace_root=normalized.get("workspace_root"),
            state_root=normalized.get("workspace_state_root"),
        )
    else:
        raise ValueError(
            "Workspace info must include one of: workspace_dir, pdf_dir, or pdf_paths."
        )

    paths = get_paths(config)
    init_workspace(paths)
    normalized["workspace_dir"] = str(paths.workspace_root)
    normalized["workspace_state_root"] = str(paths.state_root)
    return normalized


def init_workspace(paths: WorkspacePaths) -> None:
    paths.state_root.mkdir(parents=True, exist_ok=True)
    paths.vector_root.mkdir(parents=True, exist_ok=True)
    paths.graph_root.mkdir(parents=True, exist_ok=True)
    paths.sql_root.mkdir(parents=True, exist_ok=True)
    paths.artifacts_root.mkdir(parents=True, exist_ok=True)
    paths.scratch_root.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(paths.registry_db)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            rel_path TEXT NOT NULL,
            abs_path TEXT NOT NULL,
            file_name TEXT NOT NULL,
            title TEXT,
            author TEXT,
            page_count INTEGER,
            file_size_bytes INTEGER NOT NULL,
            source_type TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            scope TEXT NOT NULL,
            path TEXT,
            source_document_id TEXT,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS artifact_provenance (
            provenance_id INTEGER PRIMARY KEY AUTOINCREMENT,
            artifact_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            page_number INTEGER,
            excerpt TEXT,
            section TEXT,
            metadata_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS namespaces (
            namespace TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            scope TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    for pdf_path in sorted(paths.workspace_root.rglob("*.pdf")):
        register_document(conn, paths, pdf_path)

    document_paths = sorted(
        str(path.relative_to(paths.workspace_root)) for path in paths.workspace_root.rglob("*.pdf")
    )
    workspace_state_root = paths.workspace_root / config.WORKSPACE_STATE_DIRNAME
    workspace_state_root.mkdir(parents=True, exist_ok=True)
    manifest_path = workspace_state_root / "workspace_manifest.json"
    manifest = {
        "workspace_root": str(paths.workspace_root),
        "workspace_state_root": str(workspace_state_root),
        "registry_db": str(paths.registry_db),
        "documents": document_paths,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    overview_path = workspace_state_root / "workspace_overview.txt"
    overview_lines = [
        "Workspace quickref (read first)",
        "",
        "Boot: manifest + SQLite (`sql_query`: any statement on registry or scratch/state DBs) → vectors/graphs as needed → `llm_batch` → REPL aggregate → verify citations.",
        f"Roots: workspace={paths.workspace_root} state={workspace_state_root} registry={paths.registry_db} scratch={paths.scratch_root}",
        "Tables: documents, artifacts, artifact_provenance, namespaces. Task Q/A live in chat only.",
        'Final JSON in answer["content"]: {"answer":"...","citations":[{"document_id","path","page","excerpt"}]}',
        "",
        "PDFs:",
    ]
    if document_paths:
        overview_lines.extend(f"- {path}" for path in document_paths)
    else:
        overview_lines.append("- (no PDFs discovered)")
    overview_path.write_text("\n".join(overview_lines) + "\n", encoding="utf-8")
    conn.commit()
    conn.close()


def register_document(conn: sqlite3.Connection, paths: WorkspacePaths, pdf_path: Path) -> None:
    metadata: dict[str, Any] = {}
    reader = PdfReader(str(pdf_path))
    info = reader.metadata or {}
    page_count = len(reader.pages)
    title = getattr(info, "title", None) or info.get("/Title")
    author = getattr(info, "author", None) or info.get("/Author")
    metadata["pdf_metadata"] = {
        "title": title,
        "author": author,
        "page_count": page_count,
    }

    document_id = stable_document_id(pdf_path)
    conn.execute(
        """
        INSERT OR REPLACE INTO documents (
            document_id,
            rel_path,
            abs_path,
            file_name,
            title,
            author,
            page_count,
            file_size_bytes,
            source_type,
            metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            document_id,
            str(pdf_path.relative_to(paths.workspace_root)),
            str(pdf_path.resolve()),
            pdf_path.name,
            title,
            author,
            page_count,
            pdf_path.stat().st_size,
            "pdf",
            json.dumps(metadata, sort_keys=True),
        ),
    )
