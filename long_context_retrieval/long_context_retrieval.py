"""
Long-context retrieval environment.

Usage:
    env = load_environment({"workspace_dir": "/abs/path/to/workspace"})
    env = load_environment({"dataset_path": "contexts/tasks/dataset.jsonl"})
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
    """Load env. Accepts config dict or kwargs for `prime eval run -a`.

    Keys not in `Config` are forwarded to `RLMEnv` (same pattern as `lhaw_rlm.load_environment`).
    """
    if isinstance(config, Config):
        cfg = config
        rlm_extras = dict(kwargs)
    else:
        merged = dict(config) if isinstance(config, dict) else {}
        merged.update(kwargs)
        fields = Config.__dataclass_fields__
        cfg = Config.from_input(
            {k: v for k, v in merged.items() if k in fields} if merged else None
        )
        rlm_extras = {k: v for k, v in merged.items() if k not in fields}

    _pkg_dir = Path(__file__).resolve().parent
    _candidate_root = _pkg_dir.parent
    env_root = _candidate_root if (_candidate_root / "pyproject.toml").is_file() else _pkg_dir
    dataset = build_dataset(cfg, env_root)
    return create_environment(cfg=cfg, dataset=dataset, workspace_anchor=env_root, **rlm_extras)
