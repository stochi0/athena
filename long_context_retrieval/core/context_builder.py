from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from datasets import Dataset

from .settings import (
    CACHE_DIRNAME,
    Config,
    ENV_TIPS,
    SYSTEM_PROMPT,
    USER_PROMPT,
    WORKSPACE_CONTEXT_NOTE,
)
from .types import WorkspaceConfig
from .workspace import ensure_workspace, get_paths, init_workspace

__all__ = [
    "SYSTEM_PROMPT",
    "prepare_rows",
    "read_rows",
    "build_rows",
    "build_dataset",
]


def prepare_rows(rows: list[dict[str, Any]], anchor: Path) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["prompt"] = _prepare_prompt_messages(item.get("prompt"))
        item["info"] = ensure_workspace(dict(item.get("info") or {}), anchor)
        prepared.append(item)
    return prepared


def _prepare_prompt_messages(raw_prompt: Any) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if isinstance(raw_prompt, list):
        for message in raw_prompt:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if role is None or content is None:
                continue
            messages.append({"role": str(role), "content": str(content)})

    if not messages:
        messages = [{"role": "user", "content": USER_PROMPT}]

    existing_user_content = "\n\n".join(
        message["content"] for message in messages if message["role"] == "user"
    )
    if WORKSPACE_CONTEXT_NOTE not in existing_user_content:
        messages.append({"role": "user", "content": WORKSPACE_CONTEXT_NOTE})
    if ENV_TIPS not in existing_user_content:
        messages.append({"role": "user", "content": ENV_TIPS})
    return messages


def read_rows(dataset_path: Path, anchor: Path) -> list[dict[str, Any]]:
    if dataset_path.is_file() and dataset_path.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in dataset_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return prepare_rows(rows, anchor)
    if dataset_path.is_dir() and (dataset_path / "dataset_info.json").is_file():
        dataset = Dataset.load_from_disk(str(dataset_path))
        return prepare_rows(dataset.to_list(), anchor)
    raise FileNotFoundError(
        f"dataset_path must be a .jsonl file or a HuggingFace dataset directory: {dataset_path}"
    )


def _resolve_path(path: str, anchor: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (anchor / resolved).resolve()
    else:
        resolved = resolved.resolve()
    return resolved


def build_rows(cfg: Config, anchor: Path) -> list[dict[str, Any]]:
    output_root = Path(cfg.dataset_output_dir)
    if not output_root.is_absolute():
        output_root = (anchor / output_root).resolve()
    staged = output_root / "dataset_hf"

    has_workspace = any(
        [cfg.workspace_dir, cfg.context_dir, cfg.pdf_dir, cfg.pdf_paths]
    )
    if cfg.dataset_path:
        path = _resolve_path(cfg.dataset_path, anchor)
        return read_rows(path, anchor)

    if (
        not has_workspace
        and staged.is_dir()
        and (staged / "dataset_info.json").is_file()
    ):
        dataset = Dataset.load_from_disk(str(staged))
        return prepare_rows(dataset.to_list(), anchor)

    workspace_hint = cfg.workspace_dir or cfg.context_dir
    if not any([workspace_hint, cfg.pdf_dir, cfg.pdf_paths]):
        workspace_root = Path(tempfile.mkdtemp(prefix="long_context_retrieval_ws_"))
        init_workspace(
            get_paths(
                WorkspaceConfig(
                    workspace_root=workspace_root,
                    cache_root=workspace_root.parent
                    / CACHE_DIRNAME
                    / workspace_root.name,
                )
            )
        )
        workspace_hint = str(workspace_root)

    info = ensure_workspace(
        {
            "workspace_dir": workspace_hint,
            "pdf_dir": cfg.pdf_dir,
            "pdf_paths": cfg.pdf_paths,
            "workspace_cache_root": cfg.workspace_cache_root,
        },
        anchor,
    )
    rows = [
        {
            "prompt": _prepare_prompt_messages(
                [{"role": "user", "content": USER_PROMPT}]
            ),
            "answer": json.dumps([""]),
            "info": info,
        }
    ]
    if cfg.max_examples is not None:
        return rows[: cfg.max_examples]
    return rows


def build_dataset(cfg: Config, env_root: Path) -> Dataset:
    """Build the environment dataset from config, resolving relative paths from the env root."""
    anchor = Path(cfg.path_anchor).resolve() if cfg.path_anchor else env_root
    rows = build_rows(cfg, anchor)
    if cfg.max_examples is not None:
        rows = rows[: cfg.max_examples]
    return Dataset.from_list(rows)
