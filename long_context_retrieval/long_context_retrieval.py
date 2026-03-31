"""
Long-context retrieval environment.

Usage:
    env = load_environment({"workspace_dir": "/abs/path/to/workspace"})
    env = load_environment({"dataset_path": "contexts/dataset.jsonl"})
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import verifiers as vf
from core.config import Config
from core.context_builder import build_dataset
from core.environment import create_environment


def load_environment(
    config: Config | dict[str, Any] | None = None,
    **kwargs: Any,
) -> vf.Environment:
    """Load env. Accepts config dict or kwargs for `prime eval run -a`."""
    if isinstance(config, Config):
        cfg = config
    else:
        merged = dict(config) if isinstance(config, dict) else {}
        merged.update(kwargs)
        cfg = Config.from_input(merged if merged else None)

    _pkg_dir = Path(__file__).resolve().parent
    _candidate_root = _pkg_dir.parent
    env_root = (
        _candidate_root
        if (_candidate_root / "pyproject.toml").is_file()
        else _pkg_dir
    )
    dataset = build_dataset(cfg, env_root)
    return create_environment(cfg=cfg, dataset=dataset, workspace_anchor=env_root)

