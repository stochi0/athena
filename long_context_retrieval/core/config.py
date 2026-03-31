from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from typing import Any

CONTEXTS_DIR = "contexts"
# Under the contexts root (or any `--output-dir`): task rows + HF mirror + build manifest.
TASK_BUNDLE_SUBDIR = "tasks"
TASK_BUNDLE_HF_DIRNAME = "hf"

WORKSPACE_STATE_DIRNAME = ".workspace_state"
REGISTRY_DB = "registry.db"
VECTOR_DIRNAME = "vector"
GRAPH_DIRNAME = "graphs"
SQL_DIRNAME = "sql"
ARTIFACTS_DIRNAME = "artifacts"
SCRATCH_DIRNAME = "scratch"

ROOT_PROMPT_VERBOSITY = "light"
SUB_PROMPT_VERBOSITY = "light"
REPL_LANGUAGE = "python"
PIP_INSTALL_PACKAGES = "chromadb networkx pypdf"
CODE_EXECUTION_TIMEOUT = 180
MAX_OUTPUT_LENGTH = 8192
MAX_TURNS = 200
SUB_LLM_MAX_TURNS = 60
ENV_ID = "long_context_retrieval"
USER_PROMPT = "Answer from the PDF workspace; ground claims and cite."

# Single user appendix (one message) to avoid duplicate instructions vs system prompt.
WORKSPACE_CONTEXT_NOTE = (
    "FS = workspace PDFs/metadata only (no host bundles). "
    "Boot: `fs_read` overview + manifest; `sql_query` any SQLite (one stmt/call) on registry or scratch/state; "
    "`vector_get` / search / upsert / delete; `graph_query` or op `algo` + NetworkX name. "
    "Parallel `llm_batch()`; REPL aggregates."
).strip()

SYSTEM_PROMPT = dedent(
    """
    RLM over a PDF workspace: registry SQLite, vectors, graphs, scoped fs, Python REPL. User msgs =
    task spec; evidence = workspace + tools only (no outside knowledge, no host JSONL outside workspace).

    Workflow: overview + manifest → SQL (any statement), vectors, graphs as needed → parallel `llm_batch` (triage/extract/verify) → REPL merges.

    Submit via REPL: `answer["content"]` = JSON string with keys `answer` (string) and `citations` (array of
    {document_id, path, page, excerpt}). Then `answer["ready"] = True`. Every material claim needs citations
    (PDF preferred; register provenance for derived artifacts).

    Token budget: shortest clear `llm_batch` prompts; sub-LLMs answer tersely unless returning quotes.
    """
).strip()


@dataclass(frozen=True)
class Config:
    """Env + dataset options (mirrors `lhaw.core.config.EnvironmentConfig` scope for RLM/sandbox)."""

    dataset_path: str | None = None
    dataset_output_dir: str = CONTEXTS_DIR
    max_examples: int | None = None
    path_anchor: str | None = None
    workspace_dir: str | None = None
    pdf_dir: str | None = None
    pdf_paths: list[str] | None = None
    workspace_state_root: str | None = None
    env_id: str = ENV_ID

    sub_model: str | None = None
    max_turns: int = MAX_TURNS
    repl_language: str = REPL_LANGUAGE
    sub_llm_max_turns: int = SUB_LLM_MAX_TURNS
    pip_install_packages: str = PIP_INSTALL_PACKAGES
    code_execution_timeout: int = CODE_EXECUTION_TIMEOUT
    max_output_length: int = MAX_OUTPUT_LENGTH
    max_sub_llm_parallelism: int = 5
    max_startup_wait_seconds: int = 120
    abort_on_code_timeout: bool = False

    sandbox_docker_image: str = "python:3.11-slim"
    sandbox_cpu_cores: int = 1
    sandbox_memory_gb: int = 2
    sandbox_disk_size_gb: int = 5
    sandbox_gpu_count: int = 0
    sandbox_timeout_minutes: int = 60

    @classmethod
    def from_input(cls, cfg: "Config | dict[str, Any] | None") -> "Config":
        if cfg is None:
            return cls()
        if isinstance(cfg, cls):
            return cfg
        merged: dict[str, Any] = dict(cfg)
        if merged.get("rlm_model") is not None and merged.get("sub_model") is None:
            merged["sub_model"] = merged.pop("rlm_model")
        else:
            merged.pop("rlm_model", None)
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in merged.items() if k in fields})

    def to_env_args(self) -> dict[str, Any]:
        """Return JSON-serializable args for `vf.load_environment(env_id, env_args)`."""
        return {
            "dataset_path": self.dataset_path,
            "dataset_output_dir": self.dataset_output_dir,
            "max_examples": self.max_examples,
            "sub_model": self.sub_model,
            "max_turns": int(self.max_turns),
            "repl_language": self.repl_language,
            "sub_llm_max_turns": int(self.sub_llm_max_turns),
            "pip_install_packages": self.pip_install_packages,
            "code_execution_timeout": int(self.code_execution_timeout),
            "max_output_length": int(self.max_output_length),
            "max_sub_llm_parallelism": int(self.max_sub_llm_parallelism),
            "max_startup_wait_seconds": int(self.max_startup_wait_seconds),
            "abort_on_code_timeout": self.abort_on_code_timeout,
            "sandbox_docker_image": self.sandbox_docker_image,
            "sandbox_cpu_cores": int(self.sandbox_cpu_cores),
            "sandbox_memory_gb": int(self.sandbox_memory_gb),
            "sandbox_disk_size_gb": int(self.sandbox_disk_size_gb),
            "sandbox_gpu_count": int(self.sandbox_gpu_count),
            "sandbox_timeout_minutes": int(self.sandbox_timeout_minutes),
            "path_anchor": self.path_anchor,
            "workspace_dir": self.workspace_dir,
            "pdf_dir": self.pdf_dir,
            "pdf_paths": self.pdf_paths,
            "workspace_state_root": self.workspace_state_root,
            "env_id": self.env_id,
        }
