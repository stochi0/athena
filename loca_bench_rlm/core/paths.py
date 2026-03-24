from __future__ import annotations

import os
import importlib
import sys
from pathlib import Path
from typing import Any, Iterable


def get_env_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_vendor_loca_root() -> Path:
    return get_env_root() / "vendor" / "loca_bench"


def get_loca_root(loca_root: str | Path | None = None) -> Path:
    if loca_root is None:
        env_override = os.getenv("LOCA_BENCH_RLM_LOCA_ROOT")
        if env_override:
            return Path(env_override).resolve()
        vendored = get_vendor_loca_root()
        if vendored.exists():
            return vendored.resolve()
        raise FileNotFoundError(
            "Could not resolve a LOCA-bench checkout. Expected one of: "
            "LOCA_BENCH_RLM_LOCA_ROOT, "
            f"{vendored}, "
            "or an explicit `loca_root` argument."
        )
    return Path(loca_root).resolve()


def ensure_loca_import_path(loca_root: Path) -> None:
    loca_root_str = str(loca_root)
    if loca_root_str not in sys.path:
        sys.path.insert(0, loca_root_str)


def resolve_path(path: str | Path, search_roots: Iterable[Path]) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    return (next(iter(search_roots)) / candidate).resolve()


def dynamic_import_class(class_path: str) -> type[Any]:
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def resolve_placeholders(value: Any, *, task_dir: Path) -> Any:
    replacements = {
        "{task_workspace}": str(task_dir),
        "{agent_workspace}": str(task_dir / "agent_workspace"),
    }
    if isinstance(value, str):
        for placeholder, replacement in replacements.items():
            value = value.replace(placeholder, replacement)
        return value
    if isinstance(value, list):
        return [resolve_placeholders(item, task_dir=task_dir) for item in value]
    if isinstance(value, dict):
        return {
            key: resolve_placeholders(item, task_dir=task_dir)
            for key, item in value.items()
        }
    return value
