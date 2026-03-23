from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
