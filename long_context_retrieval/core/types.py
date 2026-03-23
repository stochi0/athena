from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict


@dataclass(frozen=True)
class WorkspaceConfig:
    workspace_root: Path
    state_root: Path


@dataclass(frozen=True)
class WorkspacePaths:
    workspace_root: Path
    state_root: Path
    registry_db: Path
    vector_root: Path
    graph_root: Path
    sql_root: Path
    artifacts_root: Path
    scratch_root: Path


class WorkspaceState(TypedDict):
    workspace_dir: str
    workspace_state_root: str
    registry_db: str
    vector_root: str
    graph_root: str
    sql_root: str
    artifacts_root: str
    scratch_root: str
    document_count: int
