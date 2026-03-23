from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from datasets import Dataset
from core.environment import (
    LongContextRetrievalEnv,
    _load_rows_from_path,
    _normalize_rows,
    create_environment,
)
from core.workspace import (
    ensure_workspace,
    initialize_workspace,
    resolve_workspace_paths,
)
from core.constants import DEFAULT_CACHE_DIRNAME, DEFAULT_DATASET_OUTPUT_DIR
from core.types import WorkspaceConfig

logger = logging.getLogger(__name__)


def load_environment(
    *,
    dataset_path: str | None = None,
    dataset_output_dir: str | None = None,
    path_anchor: str | None = None,
    context_dir: str | None = None,
    workspace_dir: str | None = None,
    pdf_dir: str | None = None,
    pdf_paths: list[str] | None = None,
    workspace_cache_root: str | None = None,
    **kwargs: Any,
) -> LongContextRetrievalEnv:
    anchor = Path(path_anchor).resolve() if path_anchor else Path.cwd()
    output_root = Path(dataset_output_dir or DEFAULT_DATASET_OUTPUT_DIR)
    if not output_root.is_absolute():
        output_root = (anchor / output_root).resolve()
    hf_dataset = output_root / "dataset_hf"

    has_explicit_workspace = any([workspace_dir, context_dir, pdf_dir, pdf_paths])

    if dataset_path:
        dataset_path_obj = Path(dataset_path).expanduser()
        if not dataset_path_obj.is_absolute():
            dataset_path_obj = (anchor / dataset_path_obj).resolve()
        else:
            dataset_path_obj = dataset_path_obj.resolve()
        rows = _load_rows_from_path(dataset_path_obj, anchor)
    elif (
        not has_explicit_workspace
        and hf_dataset.is_dir()
        and (hf_dataset / "dataset_info.json").is_file()
    ):
        dataset = Dataset.load_from_disk(str(hf_dataset))
        rows = _normalize_rows(dataset.to_list(), anchor)
        logger.info("Loaded %d examples from %s", len(rows), hf_dataset)
    else:
        workspace_hint = workspace_dir or context_dir
        if not any([workspace_hint, pdf_dir, pdf_paths]):
            tmp_workspace_root = Path(
                tempfile.mkdtemp(prefix="long_context_retrieval_ws_")
            )
            initialize_workspace(
                resolve_workspace_paths(
                    WorkspaceConfig(
                        workspace_root=tmp_workspace_root,
                        cache_root=tmp_workspace_root.parent
                        / DEFAULT_CACHE_DIRNAME
                        / tmp_workspace_root.name,
                    )
                )
            )
            workspace_hint = str(tmp_workspace_root)
        info = ensure_workspace(
            {
                "workspace_dir": workspace_hint,
                "pdf_dir": pdf_dir,
                "pdf_paths": pdf_paths,
                "workspace_cache_root": workspace_cache_root,
            },
            anchor,
        )
        rows = [
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": "Answer the question using the research-paper workspace and provide citations.",
                    }
                ],
                "answer": json.dumps([""]),
                "info": info,
            }
        ]

    return create_environment(dataset=Dataset.from_list(rows), **kwargs)
